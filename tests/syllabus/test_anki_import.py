"""anki_import.py (spec 4 section 4): the return path -- revlog import,
flag import, ReviewNote harvest, one command / one report.

Builds a real .apkg through compile.compile_syllabus (the same fixture
tests/syllabus/test_compile.py uses), extracts it (a real Anki install
would do the same on import), then injects synthetic revlog rows, card
flags, and ReviewNote text directly into the extracted collection.anki2
-- exactly what a learner's real Anki session would produce -- before
running the importer against that path, read-only.
"""
import dataclasses
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from thai_syllabus.anki_import import card_identities, import_collection
from thai_syllabus.cachekeys import sha
from thai_syllabus.compile import compile_syllabus
from thai_syllabus.wiring import _DbMediaIndex

from .test_compile import Fixture, _fully_seeded, _pair_only_syllabus


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
    compile_result = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
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

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.revlog_imported == 1

    records = fx.db.records("word", "rice", "listening")
    assert len(records) == 1
    assert records[0].ts == 1_700_000_000_000
    assert records[0].grade == 3
    assert records[0].time_ms == 4200
    assert records[0].compile_id == compile_result.compile_id


def test_revlog_import_is_idempotent_by_family_anchor_kind_and_ts(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Listening")
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, card_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    r1 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    r2 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert r1.revlog_imported == 1
    assert r2.revlog_imported == 0
    assert r2.revlog_skipped >= 1
    assert len(fx.db.records("word", "rice", "listening")) == 1


def test_revlog_import_reports_a_duplicate_as_skipped_already_present(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Listening")
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, card_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    r2 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert any(k == "revlog" and reason == "skipped: already present"
              for k, ident, reason in r2.skips)


def test_revlog_import_skips_an_unrecognized_card_with_a_reason(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    # A card id with no matching row at all.
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_001, 999999999, 0, 2, 500, 500, 2500, 3000, 1))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.revlog_skipped >= 1
    assert any(k == "revlog" and "999999999" in ident for k, ident, reason in report.skips)


# --- flag import ---------------------------------------------------------

def test_flag_on_a_production_card_with_a_current_picture_is_a_rating(compiled):
    # Production's front IS the word's picture (like Listening's front is
    # its recording): a flag there rates that specific picture, it is not
    # a card-level flag.
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Production")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    from thai_syllabus.derivations import current_best
    picture = current_best(fx.db, "rice", "picture", current_rubric={}, prior=(),
                           provenance_source=lambda s: None)
    assert picture.artifact_sha is not None

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.flags_imported == 1

    rows = fx.db.assessments_of("rice")
    rating_rows = [r for r in rows if r.backend == "learner" and r.question.get("kind") == "rating"
                  and r.question.get("role") == "picture-for-word"]
    assert len(rating_rows) == 1
    assert rating_rows[0].question["artifact_sha"] == picture.artifact_sha
    assert rating_rows[0].answer["value"] == "unacceptable-none"


def test_flag_on_a_production_card_with_no_current_picture_is_a_card_flag(compiled):
    # No artifact to rate: the flag falls back to a card-level flag
    # (spec 4 section 4).
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Production")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    from thai_syllabus.cachekeys import LearnerKey
    from thai_syllabus.derivations import current_best
    picture = current_best(fx.db, "rice", "picture", current_rubric={}, prior=(),
                           provenance_source=lambda s: None)
    fx.db.append(port="assess", backend="learner",
                key=LearnerKey(artifact_sha=picture.artifact_sha, role="picture-for-word"),
                subject="rice", question={"role": "picture-for-word",
                                          "artifact_sha": picture.artifact_sha, "kind": "rating"},
                answer={"value": "unacceptable-none"})
    assert current_best(fx.db, "rice", "picture", current_rubric={}, prior=(),
                        provenance_source=lambda s: None).artifact_sha is None

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.flags_imported == 1

    rows = fx.db.assessments_of("rice")
    flag_rows = [r for r in rows if r.backend == "learner" and r.question.get("kind") == "card-flag"
                and r.question.get("card_kind") == "production"]
    assert len(flag_rows) == 1


def test_card_flag_on_a_reading_card_is_a_card_flag_row(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Reading")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    rows = fx.db.assessments_of("rice")
    assert any(a.question.get("kind") == "card-flag" and a.question["family"] == "word"
              and a.question["card_kind"] == "reading" for a in rows)
    assert not any(a.question.get("kind") == "flag-import-marker" for a in rows)


def test_sentence_listening_flag_lands_on_the_sentence_subject(fx):
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "sentence_flag_role_extracted")

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_sentence_card(conn, text_sha, "Listening")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    rows = [a for a in fx.db.assessments_of(text_sha) if a.backend == "learner"]
    assert rows and rows[-1].question["role"] == "recording-for-sentence"


def test_flag_on_a_sentence_cloze_card_with_a_scene_picture_rates_that_picture(fx):
    # Cloze's front carries the sentence's scene picture, the way
    # Production's front carries the word's picture: a flag there rates
    # that picture.
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    fx.seed_picture(text_sha, "a man eating rice")
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "cloze_flag_extracted")

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_sentence_card(conn, text_sha, "Cloze")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    from thai_syllabus.derivations import current_best
    scene = current_best(fx.db, text_sha, "picture", current_rubric={}, prior=(),
                         provenance_source=lambda s: None)
    assert scene.artifact_sha is not None

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.flags_imported == 1

    rating_rows = [r for r in fx.db.assessments_of(text_sha)
                  if r.backend == "learner" and r.question.get("kind") == "rating"
                  and r.question.get("role") == "scene-for-sentence"]
    assert len(rating_rows) == 1
    assert rating_rows[0].question["artifact_sha"] == scene.artifact_sha
    assert rating_rows[0].answer["value"] == "unacceptable-none"


def test_flag_on_a_sentence_cloze_card_with_no_scene_picture_is_a_card_flag(fx):
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "cloze_noscene_extracted")

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_sentence_card(conn, text_sha, "Cloze")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    rows = fx.db.assessments_of(text_sha)
    assert any(r.question.get("kind") == "card-flag" and r.question.get("card_kind") == "cloze"
              for r in rows)
    assert not any(r.question.get("role") == "scene-for-sentence" for r in rows)


def test_flag_import_is_idempotent(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Reading")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    n = len(fx.db.assessments_of("rice"))
    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert len(fx.db.assessments_of("rice")) == n


def test_flag_on_a_tone_correctness_role_queues_reverification_not_override(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Listening")  # recording-for-word
    conn.execute("update cards set flags=2 where id=?", (card_id,))
    conn.commit()
    conn.close()

    from thai_syllabus.derivations import current_best
    before = current_best(fx.db, "rice", "recording", current_rubric={}, prior=(),
                          provenance_source=lambda s: None)

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.flags_imported == 1

    after = current_best(fx.db, "rice", "recording", current_rubric={}, prior=(),
                         provenance_source=lambda s: None)
    # A tone-correctness flag must NOT override current_best (the learner
    # is unqualified there, spec 3's AUTHORITY_ORDER) -- it queues
    # re-verification instead.
    assert after.artifact_sha == before.artifact_sha
    rows = fx.db.assessments_of("rice")
    reverify_rows = [r for r in rows if r.backend == "learner"
                     and r.question.get("kind") == "reverify"]
    assert len(reverify_rows) == 1
    assert reverify_rows[0].question["role"] == "recording-for-word"


def test_flag_import_is_idempotent_per_flags_state(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, note_id = _find_word_card(conn, "ข้าว", "Production")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    report2 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report2.flags_imported == 0
    assert report2.flags_skipped >= 1


def _find_pair_card(conn, pair_id: str):
    """(card_id, note_id) for the first Recognition card of any member
    note of `pair_id` (its MemberKey starts with "<pair_id>:")."""
    models, notes, cards = _models_notes_cards(conn)
    pair_model = next(m for m in models.values() if m["name"] == "minimal_pair")
    member_key_idx = _field_index(pair_model, "MemberKey")
    target_nid = None
    for nid, mid, flds, tags in notes:
        if str(mid) != pair_model["id"]:
            continue
        if flds.split("\x1f")[member_key_idx].startswith(pair_id + ":"):
            target_nid = nid
            break
    assert target_nid is not None
    for cid, nid, ord_ in cards:
        if nid == target_nid:
            return cid, target_nid
    raise AssertionError(f"no Recognition card found for pair {pair_id!r}")


def _find_sentence_card(conn, sentence_sha: str, template_name: str):
    """(card_id, note_id) for the given template on the one sentence note
    tagged sentence::SENTENCE_SHA -- the note's own guid-bearing anchor,
    one note per adopted Sentence (spec 4 r5)."""
    models, notes, cards = _models_notes_cards(conn)
    sentence_model = next(m for m in models.values() if m["name"] == "sentence")
    tmpl_ord = next(i for i, t in enumerate(sentence_model["tmpls"])
                    if t["name"] == template_name)
    sentence_tag = f"sentence::{sentence_sha}"
    target_nid = None
    for nid, mid, flds, tags in notes:
        if str(mid) != sentence_model["id"]:
            continue
        if sentence_tag in [t for t in tags.split(" ") if t]:
            target_nid = nid
            break
    assert target_nid is not None
    for cid, nid, ord_ in cards:
        if nid == target_nid and ord_ == tmpl_ord:
            return cid, target_nid
    raise AssertionError(f"no {template_name} card found for sentence {sentence_sha!r}")


def test_flag_on_a_pair_recognition_card_lands_under_the_pair_id(fx):
    # Both member notes' cards anchor on a per-member MemberKey (task
    # C2) -- a flag is about the PAIR (its rendition, current_best, and
    # compile's own audio resolution are all keyed on the pair id), so
    # the assessment row must land under the pair id, not a member's key.
    syllabus, pair = _pair_only_syllabus()
    fx.seed_rendition(pair, {"near": "near", "far": "far"}, speaker="s1")
    syllabus = dataclasses.replace(syllabus, media=_DbMediaIndex(db=fx.db, pairs=(pair,)))
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "pair_flag_extracted")

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_pair_card(conn, "p1")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.flags_imported == 1

    rows = fx.db.assessments_of("p1")
    flag_rows = [r for r in rows if r.backend == "learner"
                and r.question.get("kind") == "card-flag"]
    assert len(flag_rows) == 1


def test_flag_on_a_sentence_listening_card_lands_under_the_text_sha(fx):
    # A sentence's recording rows live under its text_sha (compile.py's
    # own resolver.sound(text_sha, "recording")) -- a flag on the
    # sentence's Listening card must land under that same subject.
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "sentence_flag_extracted")

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_sentence_card(conn, text_sha, "Listening")
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.flags_imported == 1

    rows = fx.db.assessments_of(text_sha)
    flag_rows = [r for r in rows if r.backend == "learner"]
    assert len(flag_rows) >= 1


# --- reasks over a real import ---------------------------------------------

def test_a_lapsed_sentence_card_imported_into_a_real_db_yields_one_sentence_reask(fx):
    # derivations.reasks keys a sentence's StudyRecords by the sentence's
    # own text_sha -- the entity subject import_collection writes revlog
    # rows under (never a per-Target anchor); this walks a real
    # SyllabusDb built by a real import, not only a fake StudyReader.
    from thai_syllabus.cachekeys import LearnerKey
    from thai_syllabus.derivations import reasks
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "sentence_reask_extracted")

    fx.db.append(port="assess", backend="learner",
                key=LearnerKey(artifact_sha="a" * 64, role="recording-for-sentence"),
                subject=text_sha,
                question={"role": "recording-for-sentence", "artifact_sha": "a" * 64,
                         "rubric": None, "kind": "rating"},
                answer={"value": "good"})

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_sentence_card(conn, text_sha, "Listening")
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, card_id, 0, 1, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    found = reasks(fx.db, fx.db, syllabus)
    assert [(r.subject, r.kind, r.subject_kind, r.rating) for r in found] == \
           [(text_sha, "recording", "sentence", "good")]


# --- ReviewNote harvest ----------------------------------------------------

def _set_review_note(conn, note_id: int, idx: int, text: str) -> None:
    (flds,) = conn.execute("select flds from notes where id=?", (note_id,)).fetchone()
    fields = flds.split("\x1f")
    fields[idx] = text
    conn.execute("update notes set flds=? where id=?", ("\x1f".join(fields), note_id))


def test_review_note_row_is_keyed_by_anchor(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    _, note_id = _find_word_card(conn, "ข้าว", "Listening")
    _set_review_note(conn, note_id, idx, "the picture looks off")
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.notes_harvested == 1
    assert any(a.key.startswith("learner-note:rice:") for a in fx.db.assessments_of("rice"))


def test_review_note_harvest_appends_a_learner_row_keyed_by_anchor_and_text_sha(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    _, note_id = _find_word_card(conn, "ข้าว", "Listening")
    _set_review_note(conn, note_id, idx, "the picture looks off")
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.notes_harvested == 1

    rows = fx.db.assessments_of("rice")
    harvest_rows = [r for r in rows if r.backend == "learner-note"]
    assert len(harvest_rows) == 1
    assert harvest_rows[0].answer["text"] == "the picture looks off"
    expected_key = f"learner-note:rice:{sha('the picture looks off')}"
    assert harvest_rows[0].key == expected_key


def test_review_note_reharvest_of_the_same_text_is_a_no_op(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    _, note_id = _find_word_card(conn, "ข้าว", "Listening")
    _set_review_note(conn, note_id, idx, "same text")
    conn.commit()
    conn.close()

    r1 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    r2 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert r1.notes_harvested == 1
    assert r2.notes_harvested == 0
    assert r2.notes_skipped >= 1
    assert len([r for r in fx.db.assessments_of("rice") if r.backend == "learner-note"]) == 1


def test_review_note_edited_text_is_a_new_row(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    _, note_id = _find_word_card(conn, "ข้าว", "Listening")
    _set_review_note(conn, note_id, idx, "first version")
    conn.commit()
    conn.close()
    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    conn = _open_rw(collection_path)
    _set_review_note(conn, note_id, idx, "edited version")
    conn.commit()
    conn.close()

    report2 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report2.notes_harvested == 1
    rows = fx.db.assessments_of("rice")
    harvest_rows = [r for r in rows if r.backend == "learner-note"]
    assert len(harvest_rows) == 2
    assert {r.answer["text"] for r in harvest_rows} == {"first version", "edited version"}


def test_review_note_cleared_field_appends_nothing_and_retracts_nothing(compiled):
    fx, compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    idx = _review_note_field_index(conn)
    _, note_id = _find_word_card(conn, "ข้าว", "Listening")
    _set_review_note(conn, note_id, idx, "a note")
    conn.commit()
    conn.close()
    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    conn = _open_rw(collection_path)
    _set_review_note(conn, note_id, idx, "")
    conn.commit()
    conn.close()

    report2 = import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report2.notes_harvested == 0
    rows = fx.db.assessments_of("rice")
    harvest_rows = [r for r in rows if r.backend == "learner-note"]
    assert len(harvest_rows) == 1  # the earlier row is untouched, still there
    assert harvest_rows[0].answer["text"] == "a note"


# --- card identity ---------------------------------------------------------

def test_pair_member_cards_have_distinct_anchors(fx):
    # Both member notes of one pair used to anchor on the pair id alone,
    # so their Recognition cards collapsed onto one anchor -- the anchor
    # is now each member's own MemberKey (pair id, speaker, index).
    syllabus, pair = _pair_only_syllabus()
    fx.seed_rendition(pair, {"near": "near", "far": "far"}, speaker="s1")
    syllabus = dataclasses.replace(syllabus, media=_DbMediaIndex(db=fx.db, pairs=(pair,)))
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "pair_extracted")

    identities = card_identities(collection_path)
    anchors = {i.anchor for i in identities if i.family == "minimal_pair"}
    assert anchors == {"p1:s1:0", "p1:s1:1"}


def test_sentence_card_anchor_is_the_text_sha(fx):
    # One note per adopted Sentence (spec 4 r5): its anchor is the note's
    # own sentence::TEXT_SHA tag alone, even though several target:: tags
    # (one per filled target) sit alongside it -- never composed with a
    # target id, unlike a pair member's MemberKey anchor above.
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "sentence_anchor_extracted")

    identities = card_identities(collection_path)
    sentence_identities = [i for i in identities if i.family == "sentence"]
    assert sentence_identities
    assert {i.anchor for i in sentence_identities} == {text_sha}
    assert set(sentence_identities[0].target_ids) == {
        "pom/receptive", "gin/receptive", "rice/receptive", "rice/productive"}


# --- read-only ---------------------------------------------------------

def test_import_does_not_modify_the_collection_file(compiled):
    fx, compile_result, collection_path = compiled
    before = collection_path.read_bytes()
    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    after = collection_path.read_bytes()
    assert before == after


def test_import_takes_the_collection_path_as_a_parameter_not_hardcoded():
    import inspect
    sig = inspect.signature(import_collection)
    assert "collection_path" in sig.parameters
