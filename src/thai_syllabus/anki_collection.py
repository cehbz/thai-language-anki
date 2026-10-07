"""The deck's own Anki collection (architecture r6 section 6, spec 4 r16
section 4): one adapter over the `anki` library's Collection, opened for
a pacer tick and closed after it, synced with AnkiWeb by the deck's own
process.

A collection sync (`sync`) does the normal sync itself and answers the
full sync the server demands instead, if any; the caller decides whether
to take one (`full_download`). The media sync runs in Anki's backend in
the background; `sync_media` starts it and waits until the backend says
it is done. A redirect the server answers a sync with holds for the rest
of the session, as Anki's own client keeps the newest endpoint.

The import is AnkiConnect's `importPackage` (anki.importing's
AnkiPackageImporter, which it ran) on Anki's current importer, with
notetypes merged: a note updates when the package's is newer, a notetype
when the package's is newer, a new card arrives with the package's due
and an existing (guid, ord) is left as it is, and deck presets are not
imported.

Every failure the library raises (anki.errors.AnkiException) leaves as a
CollectionError carrying Anki's message.
"""
from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from anki.collection import Collection, ImportAnkiPackageOptions, ImportAnkiPackageRequest
from anki.errors import AnkiException
from anki.import_export_pb2 import ImportAnkiPackageUpdateCondition
from anki.sync import SyncOutput
from anki.sync_pb2 import SyncAuth

MEDIA_POLL_SECONDS = 0.25

_IF_NEWER = ImportAnkiPackageUpdateCondition.IMPORT_ANKI_PACKAGE_UPDATE_CONDITION_IF_NEWER
IMPORT_OPTIONS = ImportAnkiPackageOptions(
    merge_notetypes=True, update_notes=_IF_NEWER, update_notetypes=_IF_NEWER,
    with_scheduling=True, with_deck_configs=False)


class CollectionError(Exception):
    """The collection, its sync or its import failed; the message is Anki's."""


class SyncDemand(Enum):
    """What a collection sync left undone: nothing, or the full sync the
    server demands."""
    NONE = "none"
    DOWNLOAD = "download"   # the local collection has no cards
    UPLOAD = "upload"       # the server's collection has no cards
    CONFLICT = "conflict"   # either way, the other side's changes lost


_DEMANDS = {SyncOutput.NO_CHANGES: SyncDemand.NONE,
            SyncOutput.FULL_DOWNLOAD: SyncDemand.DOWNLOAD,
            SyncOutput.FULL_UPLOAD: SyncDemand.UPLOAD,
            SyncOutput.FULL_SYNC: SyncDemand.CONFLICT}


@dataclass(frozen=True)
class SyncKey:
    """The AnkiWeb sync key (Anki's hkey) and the sync server, None for
    AnkiWeb."""
    hkey: str
    endpoint: str | None = None


@dataclass(frozen=True)
class ImportCounts:
    """An import's notes by outcome, from Anki's import log."""
    new: int = 0
    updated: int = 0
    duplicate: int = 0
    conflicting: int = 0

    def __str__(self) -> str:
        return (f"{self.new} new, {self.updated} updated, {self.duplicate} unchanged, "
                f"{self.conflicting} conflicting")


class DeckCollection:
    """One open session on the deck's collection: `col` is an
    anki.collection.Collection."""

    def __init__(self, col: Any, key: SyncKey, *,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._col = col
        self._auth = SyncAuth(hkey=key.hkey, endpoint=key.endpoint)
        self._sleep = sleep

    def sync(self) -> SyncDemand:
        try:
            out = self._col.sync_collection(self._auth, False)
        except AnkiException as exc:
            raise CollectionError(str(exc)) from None
        if out.new_endpoint:
            self._auth = SyncAuth(hkey=self._auth.hkey, endpoint=out.new_endpoint)
        return _DEMANDS[out.required]

    def full_download(self) -> None:
        """Replaces the collection with the server's; the session stays open."""
        self._col.close_for_full_sync()
        try:
            self._col.full_upload_or_download(auth=self._auth, server_usn=None, upload=False)
        except AnkiException as exc:
            raise CollectionError(str(exc)) from None
        finally:
            self._col.reopen(after_full_sync=True)

    def sync_media(self) -> None:
        try:
            self._col.sync_media(self._auth)
            while self._col.media_sync_status().active:
                self._sleep(MEDIA_POLL_SECONDS)
        except AnkiException as exc:
            raise CollectionError(str(exc)) from None

    def card_count(self) -> int:
        return self._col.card_count()

    def import_package(self, apkg: Path) -> ImportCounts:
        request = ImportAnkiPackageRequest(package_path=str(apkg), options=IMPORT_OPTIONS)
        try:
            log = self._col.import_anki_package(request).log
        except AnkiException as exc:
            raise CollectionError(str(exc)) from None
        return ImportCounts(new=len(log.new), updated=len(log.updated),
                            duplicate=len(log.duplicate), conflicting=len(log.conflicting))

    def close(self) -> None:
        self._col.close()


def open_collection(path: Path, key: SyncKey) -> DeckCollection:
    """Opens the collection at `path`, creating an empty one (and its
    directory) where there is none."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return DeckCollection(Collection(str(path)), key)
    except AnkiException as exc:
        raise CollectionError(str(exc)) from None


def login(username: str, password: str, *, endpoint: str | None,
          collection_factory: Callable[[str], Any] = Collection) -> SyncKey:
    """AnkiWeb's sync key for the account, logged in from a scratch
    collection so the deck's own is not opened."""
    with tempfile.TemporaryDirectory(prefix="anki-login-") as scratch:
        col = collection_factory(str(Path(scratch) / "login.anki2"))
        try:
            auth = col.sync_login(username=username, password=password, endpoint=endpoint)
        except AnkiException as exc:
            raise CollectionError(str(exc)) from None
        finally:
            col.close()
    return SyncKey(auth.hkey, endpoint)
