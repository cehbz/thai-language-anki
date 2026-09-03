"""anki_import.py (spec 4 section 4): the return path -- revlog import,
flag import, ReviewNote harvest, one command / one report.

Builds a real .apkg through compile.compile_syllabus (the same fixture
tests/syllabus/test_compile.py uses), extracts it (a real Anki install
would do the same on import), then injects synthetic revlog rows, card
flags, and ReviewNote text directly into the extracted collection.anki2
-- exactly what a learner's real Anki session would produce -- before
running the importer against that path, read-only.
"""
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from thai_syllabus.anki_import import import_collection
from thai_syllabus.cachekeys import sha
from thai_syllabus.compile import compile_syllabus

from .test_compile import Fixture, _fully_seeded


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


def _extract_collection(apkg_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(apkg_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir / "collection.anki2"


def _open_rw(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def _models_notes_cards(conn):
    (models_json,) = conn.execute("select models from col").fetchone()
    models = json.loads(models_json)
    notes = conn.execute("select id, mid, flds, tags from notes").fetchall()
    cards = conn.execute("select id, nid, ord from cards").fetchall()
    return models, notes, cards


def _field_index(model, name):
    return next(i for i, f in enumerate(model["flds"]) if f["name"] == name)


def _find_word_card(conn, thai_field_value: str, template_name: str):
    """(card_id, note_id) for the FIRST card of the given template on the
    word note whose Thai field matches."""
    models, notes, cards = _models_notes_cards(conn)
    word_model = next(m for m in models.values() if m["name"] == "word")
    tmpl_ord = next(i for i, t in enumerate(word_model["tmpls"])
                    if t["name"] == template_name)
    thai_idx = _field_index(word_model, "Thai")
    target_nid = None
    for nid, mid, flds, tags in notes:
        if str(mid) != word_model["id"]:
            continue
        if flds.split("\x1f")[thai_idx] == thai_field_value:
            target_nid = nid
            break
    assert target_nid is not None
    for cid, nid, ord_ in cards:
        if nid == target_nid and ord_ == tmpl_ord:
            return cid, target_nid
    raise AssertionError(f"no {template_name} card found for {thai_field_value!r}")


def _review_note_field_index(conn, model_name="word"):
    (models_json,) = conn.execute("select models from col").fetchone()
    models = json.loads(models_json)
    model = next(m for m in models.values() if m["name"] == model_name)
    return _field_index(model, "ReviewNote")


@pytest.fixture
def compiled(fx):
    """-> (fx, compiled, collection_path): a real compiled deck, extracted
    to a plain collection.anki2 path ready for synthetic edits + import.
    """
    syllabus = _fully_seeded(fx)
    compile_result = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "extracted")
    return fx, compile_result, collection_path


# --- revlog import -----------------------------------------------------

def test_revlog_import_appends_a_study_row_with_the_revlogs_own_ts(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Listening")
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, card_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db)
    assert report.revlog_imported == 1

    records = fx.db.records("rice::listening")
    assert len(records) == 1
    assert records[0].ts == 1_700_000_000_000
    assert records[0].grade == 3
    assert records[0].time_ms == 4200
    assert records[0].compile_id == compile_result.compile_id


def test_revlog_import_is_idempotent_by_card_key_and_ts(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Listening")
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, card_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    r1 = import_collection(collection_path, fx.db)
    r2 = import_collection(collection_path, fx.db)
    assert r1.revlog_imported == 1
    assert r2.revlog_imported == 0
    assert r2.revlog_skipped >= 1
    assert len(fx.db.records("rice::listening")) == 1


def test_revlog_import_skips_an_unrecognized_card_with_a_reason(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    # A card id with no matching row at all.
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_001, 999999999, 0, 2, 500, 500, 2500, 3000, 1))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db)
    assert report.revlog_skipped >= 1
    assert any(k == "revlog" and "999999999" in ident for k, ident, reason in report.skips)


# --- flag import ---------------------------------------------------------

def test_flag_import_writes_a_learner_assessment_row(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Production")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db)
    assert report.flags_imported == 1

    rows = fx.db.assessments_of("rice")
    flag_rows = [r for r in rows if r.backend == "learner" and r.question.get("role") == "picture-for-word"]
    assert len(flag_rows) == 1
    assert flag_rows[0].answer["value"] == "unacceptable-none"


def test_flag_on_a_tone_correctness_role_queues_reverification_not_override(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Listening")  # recording-for-word
    conn.execute("update cards set flags=2 where id=?", (card_id,))
    conn.commit()
    conn.close()

    from thai_syllabus.derivations import current_best
    before = current_best(fx.db, "rice", "recording")

    report = import_collection(collection_path, fx.db)
    assert report.flags_imported == 1

    after = current_best(fx.db, "rice", "recording")
    # A tone-correctness flag must NOT override current_best (the learner
    # is unqualified there, spec 3's AUTHORITY_ORDER) -- it queues
    # re-verification instead.
    assert after.artifact_sha == before.artifact_sha
    rows = fx.db.assessments_of("rice")
    reverify_rows = [r for r in rows if r.backend == "learner"
                     and r.question.get("kind") == "reverify-request"]
    assert len(reverify_rows) == 1
    assert reverify_rows[0].question["role"] == "recording-for-word"


def test_flag_import_is_idempotent_per_flags_state(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Production")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db)
    report2 = import_collection(collection_path, fx.db)
    assert report2.flags_imported == 0
    assert report2.flags_skipped >= 1


# --- ReviewNote harvest ----------------------------------------------------

def test_review_note_harvest_appends_a_learner_row_keyed_by_note_and_text_sha(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    (note_id, flds) = conn.execute(
        "select id, flds from notes limit 1").fetchone()
    fields = flds.split("\x1f")
    fields[idx] = "the picture looks off"
    conn.execute("update notes set flds=? where id=?",
                ("\x1f".join(fields), note_id))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db)
    assert report.notes_harvested == 1

    subject = str(note_id)
    rows = fx.db.assessments_of(subject)
    harvest_rows = [r for r in rows if r.backend == "learner-note"]
    assert len(harvest_rows) == 1
    assert harvest_rows[0].answer["text"] == "the picture looks off"
    expected_key = f"learner-note:{note_id}:{sha('the picture looks off')}"
    assert harvest_rows[0].key == expected_key


def test_review_note_reharvest_of_the_same_text_is_a_no_op(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    (note_id, flds) = conn.execute("select id, flds from notes limit 1").fetchone()
    fields = flds.split("\x1f")
    fields[idx] = "same text"
    conn.execute("update notes set flds=? where id=?", ("\x1f".join(fields), note_id))
    conn.commit()
    conn.close()

    r1 = import_collection(collection_path, fx.db)
    r2 = import_collection(collection_path, fx.db)
    assert r1.notes_harvested == 1
    assert r2.notes_harvested == 0
    assert r2.notes_skipped >= 1
    assert len(fx.db.assessments_of(str(note_id))) == 1


def test_review_note_edited_text_is_a_new_row(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    (note_id, flds) = conn.execute("select id, flds from notes limit 1").fetchone()
    fields = flds.split("\x1f")
    fields[idx] = "first version"
    conn.execute("update notes set flds=? where id=?", ("\x1f".join(fields), note_id))
    conn.commit()
    conn.close()
    import_collection(collection_path, fx.db)

    conn = _open_rw(collection_path)
    (note_id2, flds2) = conn.execute(
        "select id, flds from notes where id=?", (note_id,)).fetchone()
    fields2 = flds2.split("\x1f")
    fields2[idx] = "edited version"
    conn.execute("update notes set flds=? where id=?", ("\x1f".join(fields2), note_id))
    conn.commit()
    conn.close()

    report2 = import_collection(collection_path, fx.db)
    assert report2.notes_harvested == 1
    rows = fx.db.assessments_of(str(note_id))
    harvest_rows = [r for r in rows if r.backend == "learner-note"]
    assert len(harvest_rows) == 2
    assert {r.answer["text"] for r in harvest_rows} == {"first version", "edited version"}


def test_review_note_cleared_field_appends_nothing_and_retracts_nothing(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    (note_id, flds) = conn.execute("select id, flds from notes limit 1").fetchone()
    fields = flds.split("\x1f")
    fields[idx] = "a note"
    conn.execute("update notes set flds=? where id=?", ("\x1f".join(fields), note_id))
    conn.commit()
    conn.close()
    import_collection(collection_path, fx.db)

    conn = _open_rw(collection_path)
    (note_id2, flds2) = conn.execute(
        "select id, flds from notes where id=?", (note_id,)).fetchone()
    fields2 = flds2.split("\x1f")
    fields2[idx] = ""
    conn.execute("update notes set flds=? where id=?", ("\x1f".join(fields2), note_id))
    conn.commit()
    conn.close()

    report2 = import_collection(collection_path, fx.db)
    assert report2.notes_harvested == 0
    rows = fx.db.assessments_of(str(note_id))
    harvest_rows = [r for r in rows if r.backend == "learner-note"]
    assert len(harvest_rows) == 1  # the earlier row is untouched, still there
    assert harvest_rows[0].answer["text"] == "a note"


# --- read-only ---------------------------------------------------------

def test_import_does_not_modify_the_collection_file(compiled):
    fx, compile_result, collection_path = compiled
    before = collection_path.read_bytes()
    import_collection(collection_path, fx.db)
    after = collection_path.read_bytes()
    assert before == after


def test_import_takes_the_collection_path_as_a_parameter_not_hardcoded():
    import inspect
    sig = inspect.signature(import_collection)
    assert "collection_path" in sig.parameters
