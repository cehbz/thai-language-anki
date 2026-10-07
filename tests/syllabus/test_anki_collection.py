"""anki_collection (spec 4 r16 section 4): the deck's own collection over
the `anki` library. The import runs against a real collection in a temp
directory (no network); the sync runs against a fake of the library's
Collection, answering in its own protobuf types, since a real sync needs
AnkiWeb."""
import sqlite3

import pytest
from anki.errors import NetworkError, SyncError, SyncErrorKind
from anki.import_export_pb2 import ImportAnkiPackageUpdateCondition
from anki.sync import SyncOutput
from anki.sync_pb2 import MediaSyncStatusResponse, SyncAuth

from thai_syllabus import anki_collection as ac
from thai_syllabus.anki_import import read_collection_file
from thai_syllabus.compile import compile_syllabus

from .test_compile import Fixture, _fully_seeded

NO_PROVENANCE = dict(current_rubric={}, prior=(), provenance_source=lambda sha: None)
KEY = ac.SyncKey(hkey="k3y", endpoint=None)


@pytest.fixture
def package(tmp_path):
    fx = Fixture(tmp_path)
    compile_syllabus(_fully_seeded(fx), fx.db, fx.media, fx.out_path, **NO_PROVENANCE)
    return fx.out_path


def _dues(path):
    conn = sqlite3.connect(path)
    try:
        return {(guid, ord_): due for guid, ord_, due in conn.execute(
            "select n.guid, c.ord, c.due from cards c join notes n on n.id = c.nid")}
    finally:
        conn.close()


def _package_dues(package, tmp_path):
    import zipfile
    out = tmp_path / "package.anki2"
    out.write_bytes(zipfile.ZipFile(package).read("collection.anki2"))
    return _dues(out)


# --- the import, into a real collection ----------------------------------------

def test_the_import_options_are_ankiconnects_importer_with_notetypes_merged():
    # AnkiConnect's importPackage ran anki.importing.AnkiPackageImporter:
    # a note updated when the package's is newer (oldMod < note[MOD]), a
    # notetype's styling when the package's is newer, new cards with their
    # scheduling and revlog, deck presets left alone for the Default one.
    options = ac.IMPORT_OPTIONS
    if_newer = ImportAnkiPackageUpdateCondition.IMPORT_ANKI_PACKAGE_UPDATE_CONDITION_IF_NEWER
    assert (options.merge_notetypes, options.update_notes, options.update_notetypes,
            options.with_scheduling, options.with_deck_configs) == (
        True, if_newer, if_newer, True, False)


def test_an_import_into_a_new_collection_lands_every_card_at_the_packages_due(package,
                                                                              tmp_path):
    path = tmp_path / "anki" / "collection.anki2"
    collection = ac.open_collection(path, KEY)
    try:
        assert collection.card_count() == 0
        counts = collection.import_package(package)
        assert collection.card_count() == len(_package_dues(package, tmp_path))
    finally:
        collection.close()
    assert counts.new == 5 and (counts.updated, counts.conflicting) == (0, 0)
    assert _dues(path) == _package_dues(package, tmp_path)
    assert len(read_collection_file(path).col.cards) == len(_dues(path))


def test_a_reimport_leaves_an_existing_cards_due_alone(package, tmp_path):
    path = tmp_path / "anki" / "collection.anki2"
    collection = ac.open_collection(path, KEY)
    collection.import_package(package)
    collection.close()
    conn = sqlite3.connect(path)
    with conn:
        card = conn.execute("select id from cards order by id limit 1").fetchone()[0]
        conn.execute("update cards set due = 999999 where id = ?", (card,))
    conn.close()
    collection = ac.open_collection(path, KEY)
    counts = collection.import_package(package)
    collection.close()
    conn = sqlite3.connect(path)
    assert conn.execute("select due from cards where id = ?", (card,)).fetchone()[0] == 999999
    conn.close()
    assert (counts.new, counts.duplicate) == (0, 5)


def test_the_import_counts_read_as_a_log_phrase():
    assert str(ac.ImportCounts(new=3, updated=2, duplicate=7, conflicting=1)) == (
        "3 new, 2 updated, 7 unchanged, 1 conflicting")


# --- the sync, against a fake of anki's Collection ---------------------------

class FakeAnki:
    """anki.collection.Collection's sync surface: `required` is what
    sync_collection answers; media_sync_status is active for `media_polls`
    polls; `raises` maps a method name to the error it raises."""

    def __init__(self, required=SyncOutput.NO_CHANGES, new_endpoint=None, media_polls=2,
                 cards=0, raises=None):
        self.required, self.new_endpoint = required, new_endpoint
        self.media_polls, self.cards = media_polls, cards
        self.raises = raises or {}
        self.calls = []

    def _maybe_raise(self, name):
        if name in self.raises:
            raise self.raises[name]

    def sync_collection(self, auth, sync_media):
        self.calls.append(("sync_collection", auth.hkey, auth.endpoint, sync_media))
        self._maybe_raise("sync_collection")
        out = SyncOutput(required=self.required)
        if self.new_endpoint:
            out.new_endpoint = self.new_endpoint
        return out

    def close_for_full_sync(self):
        self.calls.append(("close_for_full_sync",))

    def full_upload_or_download(self, *, auth, server_usn, upload):
        self.calls.append(("full_upload_or_download", auth.hkey, auth.endpoint, upload))
        self._maybe_raise("full_upload_or_download")

    def reopen(self, after_full_sync=False):
        self.calls.append(("reopen", after_full_sync))

    def sync_media(self, auth):
        self.calls.append(("sync_media", auth.hkey, auth.endpoint))

    def media_sync_status(self):
        self.calls.append(("media_sync_status",))
        if self.media_polls:
            self.media_polls -= 1
            return MediaSyncStatusResponse(active=True)
        self._maybe_raise("media_sync_status")
        return MediaSyncStatusResponse(active=False)

    def card_count(self):
        return self.cards

    def close(self):
        self.calls.append(("close",))


def collection(fake, key=KEY, sleeps=None):
    return ac.DeckCollection(fake, key, sleep=(sleeps.append if sleeps is not None
                                               else lambda s: None))


@pytest.mark.parametrize("required,demand", [
    (SyncOutput.NO_CHANGES, ac.SyncDemand.NONE),
    (SyncOutput.FULL_DOWNLOAD, ac.SyncDemand.DOWNLOAD),
    (SyncOutput.FULL_UPLOAD, ac.SyncDemand.UPLOAD),
    (SyncOutput.FULL_SYNC, ac.SyncDemand.CONFLICT)])
def test_a_sync_says_what_full_sync_the_server_demands(required, demand):
    fake = FakeAnki(required=required)
    assert collection(fake).sync() is demand
    assert fake.calls == [("sync_collection", "k3y", "", False)]


def test_the_sync_key_names_the_endpoint_when_the_deck_does():
    fake = FakeAnki()
    collection(fake, ac.SyncKey("k3y", "https://sync.example.org/")).sync()
    assert fake.calls == [("sync_collection", "k3y", "https://sync.example.org/", False)]


def test_a_full_download_closes_downloads_and_reopens_the_collection():
    fake = FakeAnki(required=SyncOutput.FULL_DOWNLOAD)
    deck = collection(fake)
    deck.sync()
    deck.full_download()
    assert fake.calls[1:] == [("close_for_full_sync",),
                              ("full_upload_or_download", "k3y", "", False),
                              ("reopen", True)]


def test_a_redirect_moves_the_rest_of_the_session_to_the_new_endpoint():
    fake = FakeAnki(required=SyncOutput.FULL_DOWNLOAD,
                    new_endpoint="https://sync9.ankiweb.net/")
    deck = collection(fake)
    deck.sync()
    deck.full_download()
    deck.sync_media()
    assert ("full_upload_or_download", "k3y", "https://sync9.ankiweb.net/", False) in fake.calls
    assert ("sync_media", "k3y", "https://sync9.ankiweb.net/") in fake.calls


def test_a_failed_full_download_still_reopens_and_says_why():
    fake = FakeAnki(raises={"full_upload_or_download": NetworkError(
        "connection reset", None, None, None)})
    deck = collection(fake)
    with pytest.raises(ac.CollectionError, match="connection reset"):
        deck.full_download()
    assert fake.calls[-1] == ("reopen", True)


def test_a_media_sync_runs_until_the_backend_says_it_is_done():
    sleeps = []
    fake = FakeAnki(media_polls=3)
    collection(fake, sleeps=sleeps).sync_media()
    assert fake.calls == [("sync_media", "k3y", ""), *[("media_sync_status",)] * 4]
    assert sleeps == [ac.MEDIA_POLL_SECONDS] * 3


@pytest.mark.parametrize("method", ["sync_collection", "media_sync_status"])
def test_a_sync_error_is_a_collection_error_with_ankis_message(method):
    error = SyncError("AnkiWeb ID or password was incorrect", None, None, None,
                      SyncErrorKind.AUTH)
    deck = collection(FakeAnki(media_polls=0, raises={method: error}))
    with pytest.raises(ac.CollectionError, match="password was incorrect"):
        deck.sync() if method == "sync_collection" else deck.sync_media()


def test_close_closes_the_library_collection():
    fake = FakeAnki()
    collection(fake).close()
    assert fake.calls == [("close",)]


# --- the login --------------------------------------------------------------

class LoginAnki:
    def __init__(self, path, raises=None):
        self.path, self.raises = path, raises
        self.logins, self.closed = [], False

    def sync_login(self, username, password, endpoint):
        self.logins.append((username, password, endpoint))
        if self.raises:
            raise self.raises
        return SyncAuth(hkey="h-" + username, endpoint=endpoint)

    def close(self):
        self.closed = True


def test_a_login_returns_the_sync_key_from_a_scratch_collection(tmp_path):
    opened = []

    def factory(path):
        opened.append(LoginAnki(path))
        return opened[-1]

    key = ac.login("me@example.org", "pw", endpoint=None, collection_factory=factory)
    assert key == ac.SyncKey("h-me@example.org", None)
    (anki,) = opened
    assert anki.logins == [("me@example.org", "pw", None)] and anki.closed
    assert "collection.anki2" not in anki.path


def test_a_refused_login_is_a_collection_error(tmp_path):
    error = SyncError("AnkiWeb ID or password was incorrect", None, None, None,
                      SyncErrorKind.AUTH)
    with pytest.raises(ac.CollectionError, match="incorrect"):
        ac.login("me", "pw", endpoint=None,
                 collection_factory=lambda path: LoginAnki(path, raises=error))
