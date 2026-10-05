"""anki_import over AnkiConnect (spec 3 section 8 `pacer`, spec 4 section 4):
the harvest while Anki is open reads the same cards, notes, models and
revlog through AnkiConnect's actions and lands the same study and learner
rows the collection-file path lands, so a harvest by either path is
idempotent against the other.

`FileAnkiConnect` answers AnkiConnect's read actions from a compiled
collection file in the add-on's own response shapes (AnkiConnect
__init__.py: findCards, findNotes, getDecks, cardReviews, cardsInfo,
notesInfo, findModelsByName, getDeckConfig), reading the file's tables
directly rather than through anki_import's loader."""
import json
import sqlite3
from pathlib import Path

import pytest

from thai_syllabus.anki_import import (FAMILY_QUERY, import_anki_connect, import_collection,
                                       read_anki_connect)
from thai_syllabus.ankiconnect import AnkiConnect, AnkiFailed
from thai_syllabus.compile import compile_syllabus
from thai_syllabus.rulebook import sentence_note_id
from thai_syllabus.store import SyllabusDb

from .test_anki_import import (_LISTENING, _RICE_CLOZE, _extract_collection, _find_sentence_card,
                               _find_word_card, _open_rw, _review_note_field_index)
from .test_compile import Fixture, _fully_seeded

NO_PROVENANCE = dict(current_rubric={}, prior=(), provenance_source=lambda sha: None)


class FileAnkiConnect:
    """AnkiConnect's read actions over a legacy (col.models JSON)
    collection file. Only the queries the reader sends are understood."""

    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.calls: list[tuple[str, dict]] = []
        models, decks, dconf = self.conn.execute("select models, decks, dconf from col").fetchone()
        self.models = json.loads(models)
        self.decks = json.loads(decks)
        self.dconf = json.loads(dconf)

    def call(self, action, **params):
        self.calls.append((action, params))
        return getattr(self, action)(**params)

    def _family_cards(self, extra=""):
        return [cid for (cid,) in self.conn.execute(
            "select c.id from cards c join notes n on n.id = c.nid "
            f"where (' ' || n.tags || ' ') like '% family::%' {extra} order by c.id")]

    def findCards(self, query):
        if query == FAMILY_QUERY:
            return self._family_cards()
        if query == FAMILY_QUERY + " -flag:0":
            return self._family_cards("and c.flags != 0")
        raise AssertionError(f"unexpected findCards query {query!r}")

    def findNotes(self, query):
        assert query == FAMILY_QUERY + " ReviewNote:_*", query
        out = []
        for nid, mid, flds, tags in self.conn.execute("select id, mid, flds, tags from notes"):
            names = [f["name"] for f in self.models[str(mid)]["flds"]]
            values = dict(zip(names, flds.split("\x1f")))
            if " family::" in f" {tags} " and values.get("ReviewNote", ""):
                out.append(nid)
        return out

    def _deck_name(self, did):
        return self.decks[str(did)]["name"]

    def getDecks(self, cards):
        out: dict[str, list[int]] = {}
        for cid in cards:
            (did,) = self.conn.execute("select did from cards where id=?", (cid,)).fetchone()
            out.setdefault(self._deck_name(did), []).append(cid)
        return out

    def cardReviews(self, deck, startID):
        did = next(int(k) for k, d in self.decks.items() if d["name"] == deck)
        return [list(row) for row in self.conn.execute(
            "select id, cid, usn, ease, ivl, lastIvl, factor, time, type from revlog "
            "where id>? and cid in (select id from cards where did=?)", (startID, did))]

    def _fields(self, model, flds):
        return {f["name"]: {"value": value, "order": f["ord"]}
                for f, value in zip(model["flds"], flds.split("\x1f"))}

    def cardsInfo(self, cards):
        out = []
        for cid in cards:
            row = self.conn.execute(
                "select c.id, c.nid, c.ord, c.did, c.flags, n.mid, n.flds from cards c "
                "join notes n on n.id = c.nid where c.id=?", (cid,)).fetchone()
            if row is None:
                out.append({})
                continue
            cid, nid, ord_, did, flags, mid, flds = row
            model = self.models[str(mid)]
            out.append({"cardId": cid, "fields": self._fields(model, flds), "fieldOrder": ord_,
                        "question": "<q>", "answer": "<a>", "modelName": model["name"],
                        "ord": ord_, "deckName": self._deck_name(did), "css": "",
                        "factor": 0, "interval": 0, "note": nid, "type": 0, "queue": 0,
                        "due": 1, "reps": 0, "lapses": 0, "left": 0, "mod": 0,
                        "nextReviews": [], "flags": flags})
        return out

    def notesInfo(self, notes):
        out = []
        for nid in notes:
            mid, flds, tags = self.conn.execute(
                "select mid, flds, tags from notes where id=?", (nid,)).fetchone()
            model = self.models[str(mid)]
            cards = [cid for (cid,) in self.conn.execute(
                "select id from cards where nid=? order by ord", (nid,))]
            out.append({"noteId": nid, "profile": "User 1", "tags": tags.split(),
                        "fields": self._fields(model, flds), "modelName": model["name"],
                        "mod": 0, "cards": cards})
        return out

    def findModelsByName(self, modelNames):
        by_name = {m["name"]: m for m in self.models.values()}
        return [by_name[name] for name in modelNames]

    def getDeckConfig(self, deck):
        found = [d for d in self.decks.values() if d["name"] == deck]
        if not found:
            return False
        if found[0].get("dyn"):
            return found[0]
        return {**self.dconf.get(str(found[0]["conf"]), self.dconf["1"]), "dyn": False}


def _copy_db(src: SyllabusDb, path: Path) -> SyllabusDb:
    dst = sqlite3.connect(str(path))
    src._con.backup(dst)
    dst.close()
    return SyllabusDb(path)


def _rows(db: SyllabusDb):
    study = db._con.execute(
        "select family, anchor, card_kind, compile_id, ts, grade, time_ms, member_index, "
        "speaker_id from study order by family, anchor, card_kind, ts").fetchall()
    cache = sorted(db._con.execute(
        "select port, backend, key, subject, question, answer from cache").fetchall())
    return study, cache


@pytest.fixture
def studied(tmp_path):
    """A compiled deck whose collection carries what a study session leaves:
    reviews on a word's Listening card, a sentence's Listening and Cloze
    cards and an unknown card, flags on a word's Production card and the
    Cloze card, and ReviewNote text on the word."""
    fx = Fixture(tmp_path)
    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path, **NO_PROVENANCE)
    path = _extract_collection(fx.out_path, tmp_path / "extracted")
    conn = _open_rw(path)
    listening, rice_note = _find_word_card(conn, "ข้าว", "Listening")
    production, _ = _find_word_card(conn, "ข้าว", "Production")
    cloze, _ = _find_sentence_card(conn, text_sha, _RICE_CLOZE)
    sentence_listening, _ = _find_sentence_card(conn, text_sha, _LISTENING)
    for ts, cid, ease in ((1_700_000_000_000, listening, 3), (1_700_000_000_500, cloze, 1),
                          (1_700_000_001_000, sentence_listening, 4),
                          (1_700_000_002_000, 999_999_999, 2)):
        conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                     (ts, cid, -1, ease, 1, 0, 2500, 4200, 1))
    conn.execute("update cards set flags=1 where id in (?, ?)", (production, cloze))
    idx = _review_note_field_index(conn)
    (flds,) = conn.execute("select flds from notes where id=?", (rice_note,)).fetchone()
    parts = flds.split("\x1f")
    parts[idx] = "the tone sounds falling"
    conn.execute("update notes set flds=? where id=?", ("\x1f".join(parts), rice_note))
    conn.commit()
    conn.close()
    return fx, path, text_sha


def test_a_harvest_through_anki_connect_lands_the_rows_the_file_lands(studied, tmp_path):
    fx, path, text_sha = studied
    by_file = _copy_db(fx.db, tmp_path / "by-file.db")
    by_connect = _copy_db(fx.db, tmp_path / "by-connect.db")
    file_report = import_collection(path, by_file, **NO_PROVENANCE)
    connect_report = import_anki_connect(FileAnkiConnect(path), by_connect, **NO_PROVENANCE)
    assert (file_report.revlog_imported, file_report.flags_imported,
            file_report.notes_harvested) == (3, 2, 1)
    assert (connect_report.revlog_imported, connect_report.flags_imported,
            connect_report.notes_harvested) == (3, 2, 1)
    assert _rows(by_connect) == _rows(by_file)
    assert connect_report.warnings == file_report.warnings


def test_a_harvest_by_either_path_is_idempotent_against_the_other(studied, tmp_path):
    fx, path, _text_sha = studied
    db = _copy_db(fx.db, tmp_path / "both.db")
    import_collection(path, db, **NO_PROVENANCE)
    before = _rows(db)
    again = import_anki_connect(FileAnkiConnect(path), db, **NO_PROVENANCE)
    assert (again.revlog_imported, again.flags_imported, again.notes_harvested) == (0, 0, 0)
    assert _rows(db) == before


def test_the_reviews_are_asked_for_since_the_newest_harvested_one(studied, tmp_path):
    fx, path, _text_sha = studied
    db = _copy_db(fx.db, tmp_path / "since.db")
    import_collection(path, db, **NO_PROVENANCE)
    connect = FileAnkiConnect(path)
    import_anki_connect(connect, db, **NO_PROVENANCE)
    starts = {params["startID"] for action, params in connect.calls if action == "cardReviews"}
    assert starts == {1_700_000_001_000}
    asked = [params["cards"] for action, params in connect.calls if action == "cardsInfo"]
    assert len(asked) == 1 and len(asked[0]) == 2        # only the two flagged cards


def test_a_harvest_through_anki_connect_skips_manual_and_rescheduled_entries(studied,
                                                                            tmp_path):
    """cardReviews' ninth column is the revlog type: MANUAL (4) and
    RESCHEDULED (5) land no study row, as on the file path."""
    fx, path, _text_sha = studied
    conn = _open_rw(path)
    listening, _ = _find_word_card(conn, "ข้าว", "Listening")
    for ts, kind in ((1_700_000_003_000, 4), (1_700_000_004_000, 5)):
        conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                     (ts, listening, -1, 0, 1, 0, 2500, 0, kind))
    conn.commit()
    conn.close()
    db = _copy_db(fx.db, tmp_path / "connect.db")

    report = import_anki_connect(FileAnkiConnect(path), db, **NO_PROVENANCE)

    assert report.revlog_imported == 3
    assert not [r for r in db.study_rows() if r.ts in (1_700_000_003_000, 1_700_000_004_000)]


def test_an_anki_connect_error_propagates(studied, tmp_path):
    fx, _path, _text_sha = studied

    def post(url, body, timeout):
        return json.dumps({"result": None, "error": "collection is not available"}).encode()

    with pytest.raises(AnkiFailed, match="collection is not available"):
        import_anki_connect(AnkiConnect(post=post), fx.db, **NO_PROVENANCE)


# --- the live collection's shapes ---------------------------------------------
# Answers as AnkiConnect 26.8.1 (add-on 2055492159) gives them for the live
# thai-ff deck (read from its collection.anki2, Anki closed): the word
# notetype's fields and templates, a revlog row's nine columns (a first
# "Again" on a new card: usn -1, ivl -60, factor 0, type 0), deck thai-ff.

LIVE_WORD_FIELDS = ["Thai", "Meaning", "Picture", "Audio", "Ipa", "Classifier", "FrontGloss",
                    "TestSpelling", "ProductiveTarget", "ReviewNote", "CompileId",
                    "OtherSenses", "FormSide"]
LIVE_WORD_TEMPLATES = ["Listening", "Production", "Reading", "Spelling"]


class LiveShapedAnkiConnect:
    def __init__(self):
        self.calls = []

    def call(self, action, **params):
        self.calls.append((action, params))
        fields = {name: {"value": "", "order": i} for i, name in enumerate(LIVE_WORD_FIELDS)}
        fields["Thai"]["value"] = "ข้าว"
        fields["CompileId"]["value"] = "3f2a:1791037874000"
        return {
            "findCards": lambda: [1791037874301, 1791037874302] if "flag" not in params["query"]
            else [],
            "findNotes": lambda: [],
            "getDecks": lambda: {"thai-ff": [1791037874301, 1791037874302]},
            "cardReviews": lambda: [[1791140000000, 1791037874302, -1, 1, -60, 0, 0, 60000, 0]],
            "cardsInfo": lambda: [{"cardId": 1791037874302, "fields": fields, "fieldOrder": 0,
                                   "modelName": "word", "ord": 0, "deckName": "thai-ff",
                                   "note": 1791037874300, "flags": 0, "type": 1, "queue": 1}],
            "notesInfo": lambda: [{"noteId": 1791037874300, "profile": "User 1",
                                   "tags": ["family::word", "word::rice"], "fields": fields,
                                   "modelName": "word", "mod": 1791037874,
                                   "cards": [1791037874301, 1791037874302]}],
            "findModelsByName": lambda: [{
                "id": 2562845518, "name": "word", "type": 0,
                "flds": [{"name": n, "ord": i} for i, n in enumerate(LIVE_WORD_FIELDS)],
                "tmpls": [{"name": n, "ord": i, "qfmt": "", "afmt": ""}
                          for i, n in enumerate(LIVE_WORD_TEMPLATES)]}],
            "getDeckConfig": lambda: {"id": 1, "name": "Default", "dyn": False,
                                      "new": {"bury": True}, "rev": {"bury": True},
                                      "buryInterdayLearning": True},
        }[action]()


def test_the_live_shapes_map_to_a_study_row(tmp_path):
    db = SyllabusDb(tmp_path / "syllabus.db")
    report = import_anki_connect(LiveShapedAnkiConnect(), db, **NO_PROVENANCE)
    assert (report.revlog_imported, report.warnings) == (1, ())
    (record,) = db.records("word", "rice", "listening")
    assert (record.ts, record.grade, record.time_ms, record.compile_id) == (
        1791140000000, 1, 60000, "3f2a:1791037874000")


def test_a_bury_setting_off_in_the_decks_preset_warns(tmp_path):
    connect = LiveShapedAnkiConnect()
    original = connect.call

    def call(action, **params):
        answer = original(action, **params)
        if action == "getDeckConfig":
            answer = {**answer, "rev": {"bury": False}}
        return answer

    connect.call = call
    snapshot = read_anki_connect(connect)
    assert snapshot.warnings == (
        "deck 'thai-ff' uses preset 'Default' with bury review siblings off, so Anki can show "
        "a note's sibling cards on the same day; turn them on in that preset "
        "(spec 4 section 2)",)
