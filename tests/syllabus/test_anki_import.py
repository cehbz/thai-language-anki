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

from thai_syllabus.anki_import import _connect_readonly, card_identities, import_collection
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


def test_revlog_import_skips_manual_and_rescheduled_entries(compiled):
    """Spec 2 section 2: a revlog entry of kind MANUAL (4: Forget, Reset)
    or RESCHEDULED (5: Set Due Date), Anki 26.8's RevlogEntry.ReviewKind,
    is no review and lands no study row; a FILTERED one (3) is a review."""
    fx, _compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    card_id, _note_id = _find_word_card(conn, "ข้าว", "Listening")
    for ts, ease, kind in ((1_700_000_000_000, 0, 4), (1_700_000_000_100, 0, 5),
                           (1_700_000_000_200, 3, 3)):
        conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                    (ts, card_id, 0, ease, 1, 1, 2500, 0, kind))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)

    assert [r.ts for r in fx.db.records("word", "rice", "listening")] == [1_700_000_000_200]
    assert report.revlog_imported == 1 and report.revlog_skipped == 2
    assert sum(1 for k, _i, reason in report.skips
               if k == "revlog" and reason.startswith("not a review")) == 2


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
    card_id, _note_id = _find_sentence_card(conn, text_sha, _LISTENING)
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
    card_id, _note_id = _find_sentence_card(conn, text_sha, _RICE_CLOZE)
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


def test_flag_on_a_sentence_cloze_card_with_no_current_scene_picture_is_a_card_flag(fx):
    # The Cloze card compiled with its scene picture (spec 4 r10); the
    # learner rejected that picture before the import, so the flag has no
    # artifact to rate and falls back to a card-level flag.
    from thai_syllabus.cachekeys import LearnerKey
    from thai_syllabus.derivations import current_best, role_of
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    fx.seed_picture(text_sha, "a man eating rice")
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "cloze_noscene_extracted")

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_sentence_card(conn, text_sha, _RICE_CLOZE)
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    scene = current_best(fx.db, text_sha, "picture", current_rubric={}, prior=(),
                         provenance_source=lambda s: None)
    role = role_of(fx.db, text_sha, "picture")
    fx.db.append(port="assess", backend="learner",
                key=LearnerKey(artifact_sha=scene.artifact_sha, role=role),
                subject=text_sha, question={"role": role, "artifact_sha": scene.artifact_sha,
                                            "kind": "rating"},
                answer={"value": "unacceptable-none"})
    assert current_best(fx.db, text_sha, "picture", current_rubric={}, prior=(),
                        provenance_source=lambda s: None).artifact_sha is None

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    rows = fx.db.assessments_of(text_sha)
    assert any(r.question.get("kind") == "card-flag" and r.question.get("card_kind") == "cloze"
              for r in rows)
    # the one learner rating is the rejection above; the flag added none
    assert len([r for r in rows if r.backend == "learner"
                and r.question.get("kind") == "rating"]) == 1


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


def test_a_review_and_a_flag_on_a_spelling_groups_listening_card_map_to_its_first_word(fx):
    """spec 4 r12: แก้ว (glass, drinking / the material) has one Listening
    card, on the first Word's note; its review and its flag land on that
    Word, whose recording it plays."""
    from .test_compile import glass_group

    syllabus, _media = glass_group(fx)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path, force=True,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "extracted")
    conn = _open_rw(collection_path)
    models, notes, cards = _models_notes_cards(conn)
    word_model = next(m for m in models.values() if m["name"] == "word")
    listening_ord = next(i for i, t in enumerate(word_model["tmpls"]) if t["name"] == "Listening")
    thai_idx = _field_index(word_model, "Thai")
    glass_nids = {nid for nid, mid, flds, _tags in notes
                  if str(mid) == word_model["id"] and flds.split("\x1f")[thai_idx] == "แก้ว"}
    listening = [cid for cid, nid, ord_ in cards if nid in glass_nids and ord_ == listening_ord]
    assert len(listening) == 1
    assert compiled.report.findings == ()
    (card_id,) = listening
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                 (1_700_000_000_000, card_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.execute("update cards set flags=2 where id=?", (card_id,))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.revlog_imported == 1 and report.flags_imported == 1
    assert len(fx.db.records("word", "glass-drinking", "listening")) == 1
    assert fx.db.records("word", "glass-material", "listening") == []
    assert [r.question["role"] for r in fx.db.assessments_of("glass-drinking")
            if r.backend == "learner" and r.question.get("kind") == "reverify"] == [
        "recording-for-word"]
    assert not [r for r in fx.db.assessments_of("glass-material") if r.backend == "learner"]


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


# The card ords of _fully_seeded's "ผมกินข้าว" (I eat rice) sentence note:
# its Listening card, and the Cloze card of rice/productive in rice's slot,
# the sentence's third distinct word (spec 4 r11).
_LISTENING, _RICE_CLOZE = 0, 3


def _find_sentence_card(conn, sentence_sha: str, ord_: int):
    """(card_id, note_id) for card `ord_` of the sentence note tagged
    sentence::SENTENCE_SHA."""
    models, notes, cards = _models_notes_cards(conn)
    sentence_model = next(m for m in models.values() if m["name"] == "sentence")
    target_nid = next(nid for nid, mid, flds, tags in notes
                      if str(mid) == sentence_model["id"]
                      and f"sentence::{sentence_sha}" in tags.split(" "))
    for cid, nid, card_ord in cards:
        if nid == target_nid and card_ord == ord_:
            return cid, target_nid
    raise AssertionError(f"no card {ord_} found for sentence {sentence_sha!r}")


def test_flag_on_a_pair_recognition_card_lands_under_the_pair_id(fx):
    # Both member notes' cards anchor on a per-member MemberKey (task
    # C2) -- a flag is about the PAIR (its rendition, current_best, and
    # compile's own audio resolution are all keyed on the pair id), so
    # the assessment row must land under the pair id, not a member's key.
    syllabus, pair = _pair_only_syllabus()
    fx.seed_rendition(pair, {"near": "near", "far": "far"}, speaker="s1")
    for member in pair.members:
        fx.seed_picture(member, member)
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
    card_id, _note_id = _find_sentence_card(conn, text_sha, _LISTENING)
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
    card_id, _note_id = _find_sentence_card(conn, text_sha, _LISTENING)
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, card_id, 0, 1, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    found = reasks(fx.db, fx.db, syllabus)
    assert [(r.subject, r.kind, r.subject_kind, r.rating) for r in found] == \
           [(text_sha, "recording", "sentence", "good")]


def test_a_lapsed_cloze_card_imported_into_a_real_db_reasks_the_scene_picture(fx):
    # A Cloze card's study rows land under SENTENCE_SHA:TARGET_ID (spec 4
    # r9); the scene-picture reask reads every Cloze anchor of the
    # sentence, so a lapse on one re-asks the sentence's picture rating.
    from thai_syllabus.cachekeys import LearnerKey
    from thai_syllabus.derivations import reasks
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "cloze_reask_extracted")

    fx.db.append(port="assess", backend="learner",
                key=LearnerKey(artifact_sha="a" * 64, role="scene-for-sentence"),
                subject=text_sha,
                question={"role": "scene-for-sentence", "artifact_sha": "a" * 64,
                         "rubric": None, "kind": "rating"},
                answer={"value": "good"})

    conn = _open_rw(collection_path)
    card_id, _note_id = _find_sentence_card(conn, text_sha, _RICE_CLOZE)
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, card_id, 0, 1, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)

    found = reasks(fx.db, fx.db, syllabus)
    assert [(r.subject, r.kind, r.subject_kind, r.rating) for r in found] == \
           [(text_sha, "picture", "sentence", "good")]


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
    for member in pair.members:
        fx.seed_picture(member, member)
    syllabus = dataclasses.replace(syllabus, media=_DbMediaIndex(db=fx.db, pairs=(pair,)))
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "pair_extracted")

    identities = card_identities(collection_path)
    anchors = {i.anchor for i in identities if i.family == "minimal_pair"}
    assert anchors == {"p1:s1:0", "p1:s1:1"}


def test_sentence_listening_card_anchor_is_the_text_sha(fx):
    # The sentence note's anchor is its own sentence::TEXT_SHA tag alone,
    # even though several target:: tags (one per filled target) sit
    # alongside it.
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "sentence_anchor_extracted")

    identities = card_identities(collection_path)
    listening = [i for i in identities if i.family == "sentence" and i.kind_slug == "listening"]
    assert len(listening) == 1
    assert listening[0].anchor == text_sha
    assert listening[0].sentence_sha == text_sha
    assert set(listening[0].target_ids) == {
        "pom/receptive", "gin/receptive", "rice/receptive", "rice/productive"}


def test_sentence_cloze_card_anchor_is_the_sentence_and_its_target(fx):
    # Spec 4 r9: a Cloze card's anchor is SENTENCE_SHA:TARGET_ID, composed
    # from its sentence:: and target:: tags the way a pair member's
    # MemberKey composes its three; its entity subject stays the text_sha.
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "cloze_anchor_extracted")

    cloze = [i for i in card_identities(collection_path)
             if i.family == "sentence" and i.kind_slug == "cloze"]
    assert [(i.anchor, i.sentence_sha, i.target_ids) for i in cloze] == \
        [(f"{text_sha}:rice/productive", text_sha, ("rice/productive",))]


def test_a_cloze_cards_revlog_lands_under_its_sentence_and_target(fx):
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "cloze_revlog_extracted")

    conn = _open_rw(collection_path)
    cloze_id, _ = _find_sentence_card(conn, text_sha, _RICE_CLOZE)
    listening_id, _ = _find_sentence_card(conn, text_sha, _LISTENING)
    for ts, card_id in ((1_700_000_000_000, cloze_id), (1_700_000_000_001, listening_id)):
        conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                     (ts, card_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.revlog_imported == 2
    assert [r.ts for r in fx.db.records("sentence", f"{text_sha}:rice/productive", "cloze")] == \
        [1_700_000_000_000]
    assert fx.db.records("sentence", text_sha, "cloze") == []
    assert [r.ts for r in fx.db.records("sentence", text_sha, "listening")] == \
        [1_700_000_000_001]


def test_a_flag_on_a_cloze_card_is_keyed_by_its_sentence_and_target(fx):
    from thai_syllabus.cachekeys import FlagKey
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "cloze_flagkey_extracted")

    conn = _open_rw(collection_path)
    card_id, _ = _find_sentence_card(conn, text_sha, _RICE_CLOZE)
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    anchor = f"{text_sha}:rice/productive"
    (row,) = [r for r in fx.db.assessments_of(text_sha) if r.backend == "learner"]
    assert (row.question["anchor"], row.question["card_kind"]) == (anchor, "cloze")
    assert fx.db.latest("assess", "learner", FlagKey(family="sentence", anchor=anchor,
                                                     card_kind="cloze", flags=1)) is not None


def test_an_audio_cloze_review_and_flag_map_to_its_slots_anchor_under_their_own_kind(fx):
    """Spec 4 r13 section 4: an AudioCloze card maps back as its slot's
    Cloze card does, card kind audio_cloze; its flag is card-level,
    though the sentence has a scene picture a Cloze flag would rate."""
    from thai_syllabus.cachekeys import FlagKey
    from thai_syllabus.compile import CLOZE_SLOTS
    from thai_syllabus.rulebook import sentence_note_id
    from .test_compile import _seed_gap

    syllabus = _fully_seeded(fx)
    _seed_gap(fx, syllabus)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "audio_cloze_extracted")

    conn = _open_rw(collection_path)
    card_id, _ = _find_sentence_card(conn, text_sha, CLOZE_SLOTS + _RICE_CLOZE)
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                 (1_700_000_000_000, card_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.execute("update cards set flags=1 where id=?", (card_id,))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    anchor = f"{text_sha}:rice/productive"
    assert (report.revlog_imported, report.flags_imported) == (1, 1)
    assert [r.ts for r in fx.db.records("sentence", anchor, "audio_cloze")] == [1_700_000_000_000]
    assert fx.db.records("sentence", anchor, "cloze") == []
    (row,) = [r for r in fx.db.assessments_of(text_sha) if r.backend == "learner"]
    assert (row.question["role"], row.question["anchor"], row.question["card_kind"]) == (
        "card-flag", anchor, "audio_cloze")
    assert fx.db.latest("assess", "learner", FlagKey(family="sentence", anchor=anchor,
                                                     card_kind="audio_cloze", flags=1))
    assert (anchor, "audio_cloze", ("rice/productive",)) in {
        (i.anchor, i.kind_slug, i.target_ids) for i in card_identities(collection_path)}


def test_a_sentence_notes_review_note_harvests_one_row_under_its_text_sha(fx):
    """A sentence's Listening and Cloze cards are siblings of one note
    (spec 4 r11): its ReviewNote is one learner-note row keyed by the
    note's own anchor, the text_sha."""
    from thai_syllabus.rulebook import sentence_note_id

    syllabus = _fully_seeded(fx)
    text_sha = sentence_note_id(syllabus.sentences[0])
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                    current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "sentence_note_extracted")

    conn = _open_rw(collection_path)
    _, listening_nid = _find_sentence_card(conn, text_sha, _LISTENING)
    _, cloze_nid = _find_sentence_card(conn, text_sha, _RICE_CLOZE)
    assert cloze_nid == listening_nid
    _set_review_note(conn, listening_nid, _review_note_field_index(conn, "sentence"),
                     "the audio is clipped")
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.notes_harvested == 1
    rows = [r for r in fx.db.assessments_of(text_sha) if r.backend == "learner-note"]
    assert [r.key for r in rows] == [f"learner-note:{text_sha}:{sha('the audio is clipped')}"]


def _compiled_two_productive_words(fx):
    """"กินข้าว" (eat rice) filling eat/productive (slot 1, card ord 1)
    and rice/productive (slot 2, card ord 2), compiled and extracted:
    (text_sha, collection path).
    """
    from thai_syllabus.rulebook import sentence_note_id
    from .test_compile import _two_productive_words

    syllabus, kin_khaao = _two_productive_words(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    return sentence_note_id(kin_khaao), _extract_collection(fx.out_path, fx.tmp_path / "two_extracted")


def test_import_maps_a_study_row_and_a_flag_on_a_cloze_card_ord_to_its_sentence_and_target(fx):
    # Spec 4 r11: a Cloze card is (sentence note, card ord); the ord is
    # its slot, and the slot names its Target.
    from thai_syllabus.cachekeys import FlagKey

    text_sha, collection_path = _compiled_two_productive_words(fx)
    conn = _open_rw(collection_path)
    eat_id, _ = _find_sentence_card(conn, text_sha, 1)
    rice_id, _ = _find_sentence_card(conn, text_sha, 2)
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                 (1_700_000_000_000, rice_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.execute("update cards set flags=1 where id=?", (eat_id,))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert (report.revlog_imported, report.flags_imported) == (1, 1)
    assert [r.ts for r in fx.db.records("sentence", f"{text_sha}:rice/productive", "cloze")] == \
        [1_700_000_000_000]
    assert fx.db.latest("assess", "learner", FlagKey(
        family="sentence", anchor=f"{text_sha}:eat/productive", card_kind="cloze",
        flags=1)) is not None
    identities = {(i.anchor, i.kind_slug, i.target_ids) for i in card_identities(collection_path)
                  if i.family == "sentence"}
    assert identities == {
        (text_sha, "listening", ("eat/productive", "eat/receptive", "rice/productive",
                                 "rice/receptive")),
        (f"{text_sha}:eat/productive", "cloze", ("eat/productive",)),
        (f"{text_sha}:rice/productive", "cloze", ("rice/productive",))}


def test_a_cloze_card_on_a_slot_whose_word_has_no_productive_target_is_skipped(fx):
    # A slot whose word carries no productive Target names none: a card
    # there (only a collection edited by hand can hold one) is skipped,
    # counted.
    text_sha, collection_path = _compiled_two_productive_words(fx)
    conn = _open_rw(collection_path)
    rice_id, rice_nid = _find_sentence_card(conn, text_sha, 2)
    models, _notes, _cards = _models_notes_cards(conn)
    model = next(m for m in models.values() if m["name"] == "sentence")
    _set_review_note(conn, rice_nid, _field_index(model, "ClozeTarget2"), "")
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                 (1_700_000_000_000, rice_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert (report.revlog_imported, report.revlog_skipped) == (0, 1)
    assert fx.db.records("sentence", text_sha, "cloze") == []


def _reviewed_after_push_out(fx):
    """"กินข้าว"'s eat Cloze card (ord 1) compiled at cap 2, then the
    note updated in place to a cap-1 compile's fields, which empty slot
    1 (as Anki updates a note on re-import, keeping the card and its
    schedule), and a review on that card: (capped syllabus, the
    sentence, collection path).
    """
    from .test_compile import _capped_eat

    both, kin_khaao = _capped_eat(fx, cap=2)
    compile_syllabus(both, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    collection_path = _extract_collection(fx.out_path, fx.tmp_path / "pushed_out_extracted")

    (fx.tmp_path / "capped").mkdir()
    fx_capped = Fixture(fx.tmp_path / "capped")
    capped, _ = _capped_eat(fx_capped, cap=1)
    compile_syllabus(capped, fx_capped.db, fx_capped.media, fx_capped.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    capped_conn = _open_rw(_extract_collection(fx_capped.out_path, fx.tmp_path / "capped_extracted"))
    ((capped_flds,),) = capped_conn.execute(
        "select flds from notes where tags like ?",
        (f"% sentence::{kin_khaao.text_sha} %",)).fetchall()
    capped_conn.close()

    conn = _open_rw(collection_path)
    eat_id, nid = _find_sentence_card(conn, kin_khaao.text_sha, 1)
    conn.execute("update notes set flds=? where id=?", (capped_flds, nid))
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                 (1_700_000_000_000, eat_id, 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()
    return capped, kin_khaao, collection_path


def test_a_review_on_a_cloze_card_whose_slot_emptied_maps_to_its_pair(fx):
    # Spec 4 r11: an unfilled slot still names its word's productive
    # Target, so a review Anki recorded on the card kept there lands
    # under its (sentence, Target).
    _capped, kin_khaao, collection_path = _reviewed_after_push_out(fx)
    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.revlog_imported == 1
    assert [r.ts for r in fx.db.records(
        "sentence", f"{kin_khaao.text_sha}:eat/productive", "cloze")] == [1_700_000_000_000]


def test_a_review_on_a_pushed_out_pair_makes_it_fill_again_under_the_cap(fx):
    # The imported review makes the pair studied (spec 1 r26), so under
    # the same cap it fills "กินข้าว" again and its card returns at ord 1.
    from thai_syllabus.wiring import studied_cloze_pairs

    capped, kin_khaao, collection_path = _reviewed_after_push_out(fx)
    assert "eat/productive" not in {t.id for t in capped.productive_fills(kin_khaao)}
    import_collection(collection_path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    studied = studied_cloze_pairs(fx.db, capped.sentences, capped.targets)
    assert studied == frozenset({(kin_khaao.text_sha, "eat/productive")})
    refilled = dataclasses.replace(capped, studied_cloze_pairs=studied)
    assert [t.id for t in refilled.productive_fills(kin_khaao)] == \
        ["eat/productive", "rice/productive"]
    compile_syllabus(refilled, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    conn = _open_rw(_extract_collection(fx.out_path, fx.tmp_path / "refilled_extracted"))
    assert _find_sentence_card(conn, kin_khaao.text_sha, 1)
    conn.close()


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


# --- schema 18 (task 1 brief) ------------------------------------------
#
# Anki's current collection shape (verified against a live collection.anki2,
# ver=18): notetypes live in their own notetypes/fields/templates tables,
# not col.models JSON -- col.models is '' there. Built synthetically here,
# never by opening or copying the user's real collection.

_SCHEMA18_DDL = """
create table col (
    id integer primary key, crt integer, mod integer, scm integer,
    ver integer, dty integer, usn integer, ls integer, conf text,
    models text, decks text, dconf text, tags text
);
create table notetypes (id integer primary key, name text);
create table fields (ntid integer, ord integer, name text);
create table templates (ntid integer, ord integer, name text);
create table decks (id integer primary key, name text, mtime_secs integer,
                    usn integer, common blob, kind blob);
create table deck_config (id integer primary key, name text, mtime_secs integer,
                          usn integer, config blob);
create table notes (
    id integer primary key, guid text, mid integer, mod integer,
    usn integer, tags text, flds text, sfld text, csum integer,
    flags integer, data text
);
create table cards (
    id integer primary key, nid integer, did integer, ord integer,
    mod integer, usn integer, type integer, queue integer,
    due integer, ivl integer, factor integer, reps integer,
    lapses integer, left integer, odue integer, odid integer,
    flags integer, data text
);
create table revlog (
    id integer primary key, cid integer, usn integer, ease integer,
    ivl integer, lastIvl integer, factor integer, time integer,
    type integer
);
"""


_SCHEMA18_FIELD_VALUES = {"Thai": "ข้าว", "CompileId": "compile-rice-1"}


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | 0x80 if n else byte)
        if not n:
            return bytes(out)


def _proto(fields: dict[int, int | bytes]) -> bytes:
    """A protobuf message of varint (int) and length-delimited (bytes)
    fields, the encoding of Anki's decks.kind and deck_config.config."""
    out = b""
    for number, value in fields.items():
        if isinstance(value, bytes):
            out += _varint(number << 3 | 2) + _varint(len(value)) + value
        else:
            out += _varint(number << 3) + _varint(value)
    return out


# DeckConfig.Config's bury_new, bury_reviews, bury_interday_learning
# (Anki 26.09 deck_config.proto); a false bool is absent, as Anki writes it.
_BURY_FIELDS = (27, 28, 29)


def _build_schema18_collection(path: Path, *, review_note_text: str = "",
                               fields: tuple[str, ...] = ("Thai", "CompileId", "ReviewNote"),
                               ver: int | None = 18,
                               bury: tuple[bool, bool, bool] = (False, False, False),
                               deck_preset: int = 9) -> dict:
    """A synthetic Anki schema-18 collection.anki2 (task 1 brief): one
    "word" notetype (by default Thai/CompileId/ReviewNote fields, a
    Listening template) read from notetypes/fields/templates, col.models=''
    -- the shape _load_collection must branch on. `ver` defaults to 18
    (the verified live collection's own value); pass None to leave it
    unset, the other signal _load_models branches on (round 2 review).
    One word note tagged family::word/word::rice, one Listening card on
    it; `fields` lets a caller omit CompileId to exercise the "model has
    no such field" -> "" fallback (round 2 review). The card's deck
    "thai-ff" uses preset "Thai" whose new/review/interday-learning
    sibling burying is `bury` (Anki's defaults: all off); `deck_preset`
    names another preset id, one with no deck_config row when not 1 or 9.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA18_DDL)
    if ver is None:
        conn.execute("insert into col (id, models) values (1, '')")
    else:
        conn.execute("insert into col (id, ver, models) values (1, ?, '')", (ver,))

    ntid = 1
    conn.execute("insert into notetypes (id, name) values (?, 'word')", (ntid,))
    for ord_, name in enumerate(fields):
        conn.execute("insert into fields (ntid, ord, name) values (?, ?, ?)",
                    (ntid, ord_, name))
    for ord_, name in enumerate(["Listening"]):
        conn.execute("insert into templates (ntid, ord, name) values (?, ?, ?)",
                    (ntid, ord_, name))
    conn.execute("insert into decks (id, name, kind) values (1, 'Default', ?)",
                 (_proto({1: _proto({1: 1})}),))
    conn.execute("insert into decks (id, name, kind) values (7, 'thai-ff', ?)",
                 (_proto({1: _proto({1: deck_preset})}),))
    conn.execute("insert into deck_config (id, name, config) values (1, 'Default', ?)",
                 (_proto({}),))
    conn.execute("insert into deck_config (id, name, config) values (9, 'Thai', ?)",
                 (_proto({number: 1 for number, on in zip(_BURY_FIELDS, bury) if on}),))

    nid = 1
    values = {**_SCHEMA18_FIELD_VALUES, "ReviewNote": review_note_text}
    flds = "\x1f".join(values.get(name, "") for name in fields)
    conn.execute("insert into notes (id, mid, flds, tags) values (?, ?, ?, ?)",
                (nid, ntid, flds, " family::word word::rice "))
    card_id = 1
    conn.execute("insert into cards (id, nid, did, ord, flags, odid) values (?, ?, 7, 0, 0, 0)",
                (card_id, nid))
    conn.commit()
    conn.close()
    return {"note_id": nid, "card_id": card_id}


@pytest.fixture
def schema18(tmp_path):
    path = tmp_path / "collection.anki2"
    ids = _build_schema18_collection(path, review_note_text="the tones sound off")
    return path, ids


def test_schema18_card_identity_reads_the_notetypes_fields_templates_tables(schema18):
    path, ids = schema18
    identities = card_identities(path)
    assert len(identities) == 1
    identity = identities[0]
    assert identity.family == "word"
    assert identity.anchor == "rice"
    assert identity.kind_slug == "listening"
    assert identity.compile_id == "compile-rice-1"


def test_schema18_revlog_import_appends_a_study_row(fx, schema18):
    path, ids = schema18
    conn = sqlite3.connect(str(path))
    conn.execute("insert into revlog values (?,?,?,?,?,?,?,?,?)",
                (1_700_000_000_000, ids["card_id"], 0, 3, 1000, 1000, 2500, 4200, 1))
    conn.commit()
    conn.close()

    report = import_collection(path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.revlog_imported == 1

    records = fx.db.records("word", "rice", "listening")
    assert len(records) == 1
    assert records[0].ts == 1_700_000_000_000
    assert records[0].grade == 3
    assert records[0].time_ms == 4200
    assert records[0].compile_id == "compile-rice-1"


def test_schema18_review_note_harvest_appends_a_learner_row(fx, schema18):
    path, ids = schema18
    report = import_collection(path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.notes_harvested == 1

    rows = [r for r in fx.db.assessments_of("rice") if r.backend == "learner-note"]
    assert len(rows) == 1
    assert rows[0].answer["text"] == "the tones sound off"


# --- sibling burying in the deck's preset (spec 4 section 2) ----------------

def test_import_warns_when_the_decks_preset_buries_no_siblings(fx, tmp_path):
    path = tmp_path / "collection.anki2"
    _build_schema18_collection(path)
    report = import_collection(path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.warnings == (
        "deck 'thai-ff' uses preset 'Thai' with bury new siblings, bury review siblings, "
        "bury interday learning siblings off, so Anki can show a note's sibling cards on "
        "the same day; turn them on in that preset (spec 4 section 2)",)
    assert report.revlog_skipped == 0


def test_import_names_only_the_bury_settings_that_are_off(fx, tmp_path):
    path = tmp_path / "collection.anki2"
    _build_schema18_collection(path, bury=(True, False, True))
    report = import_collection(path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert len(report.warnings) == 1
    assert "with bury review siblings off," in report.warnings[0]


def test_import_warns_nothing_when_the_decks_preset_buries_every_sibling(fx, tmp_path):
    path = tmp_path / "collection.anki2"
    _build_schema18_collection(path, bury=(True, True, True))
    report = import_collection(path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.warnings == ()


def test_import_reads_a_legacy_collections_deck_preset(compiled):
    # The .apkg's own legacy collection: its options group (genanki's)
    # buries new and review siblings and has no interday-learning setting.
    fx, _compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    (dconf_json,) = conn.execute("select dconf from col").fetchone()
    dconf = json.loads(dconf_json)
    dconf["1"]["rev"]["bury"] = False
    conn.execute("update col set dconf=?", (json.dumps(dconf),))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert len(report.warnings) == 1
    assert report.warnings[0].startswith(f"deck 'deck' uses preset {dconf['1']['name']!r} with "
                              "bury review siblings, bury interday learning siblings off,")


def test_import_reads_the_default_preset_for_a_deck_whose_preset_is_missing(fx, tmp_path):
    # Anki falls back to the Default preset (id 1) when a deck's config id
    # has no deck_config row.
    path = tmp_path / "collection.anki2"
    _build_schema18_collection(path, bury=(True, True, True), deck_preset=42)
    report = import_collection(path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert len(report.warnings) == 1
    assert report.warnings[0].startswith("deck 'thai-ff' uses preset 'Default' with bury new "
                                         "siblings, bury review siblings, bury interday "
                                         "learning siblings off,")


def test_import_reads_a_legacy_collections_default_preset_for_a_missing_preset(compiled):
    fx, _compile_result, collection_path = compiled
    conn = _open_rw(collection_path)
    decks_json, dconf_json = conn.execute("select decks, dconf from col").fetchone()
    decks, dconf = json.loads(decks_json), json.loads(dconf_json)
    for deck in decks.values():
        deck["conf"] = 42
    dconf["1"]["rev"]["bury"] = False
    conn.execute("update col set decks=?, dconf=?", (json.dumps(decks), json.dumps(dconf)))
    conn.commit()
    conn.close()

    report = import_collection(collection_path, fx.db,
                               current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert len(report.warnings) == 1
    assert report.warnings[0].startswith(f"deck 'deck' uses preset {dconf['1']['name']!r} with "
                              "bury review siblings, bury interday learning siblings off,")


# --- unicase collation (round-1 review) ---------------------------------

def test_connect_readonly_registers_unicase_so_a_collated_query_does_not_raise(tmp_path):
    """Demonstrates the need for _connect_readonly's `unicase` collation
    registration: Anki's own collection.anki2 declares some notes/cards
    indexes `collate unicase`, and a query whose WHERE/ORDER BY touches
    such a column raises OperationalError under Python's sqlite3 the
    moment the collation isn't registered on that connection -- true even
    read-only, even though nothing here writes anything or opens the
    user's real collection.
    """
    path = tmp_path / "unicase.db"
    # Build the fixture through a connection that DOES register a
    # `unicase` collation (sqlite3 requires it be resolvable to create an
    # index that declares it), mirroring Anki's own Rust backend, which
    # registers it before ever touching the file.
    setup = sqlite3.connect(str(path))
    setup.create_collation("unicase", lambda a, b: (a > b) - (a < b))
    setup.execute("create table t (name text collate unicase)")
    setup.execute("create index t_name on t(name)")
    setup.execute("insert into t values ('x')")
    setup.commit()
    setup.close()

    # Reopened read-only WITHOUT registering the collation -- what plain
    # sqlite3.connect(f"file:{path}?mode=ro", uri=True) gives -- a query
    # comparing the collated column fails.
    bare = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="unicase"):
        bare.execute("select name from t where name = 'x'").fetchall()
    bare.close()

    # The same query through _connect_readonly succeeds: it registers the
    # (no-op) stand-in collation first.
    conn = _connect_readonly(path)
    assert conn.execute("select name from t where name = 'x'").fetchall() == [("x",)]
    conn.close()


# --- _load_models ver-is-None branch (round-2 review) -------------------

def test_schema18_with_ver_unset_still_reads_the_notetypes_tables(fx, tmp_path):
    # _load_models treats a null col.ver the same as ver>=18 (belt and
    # suspenders alongside the empty-col.models signal) -- this is the
    # one branch of that condition the existing schema18 fixture (which
    # always sets ver=18) never exercised.
    path = tmp_path / "collection.anki2"
    ids = _build_schema18_collection(path, review_note_text="", ver=None)

    identities = card_identities(path)
    assert len(identities) == 1
    assert identities[0].family == "word"
    assert identities[0].anchor == "rice"
    assert identities[0].kind_slug == "listening"


# --- ReviewNote harvest records CompileId (spec 4 r8 section 4, round-2
# review) -------------------------------------------------------------

def test_review_note_harvest_records_the_notes_compile_id(compiled):
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
    assert harvest_rows[0].question["compile_id"] == compile_result.compile_id


def test_review_note_harvest_records_empty_compile_id_when_the_model_lacks_the_field(fx, tmp_path):
    path = tmp_path / "collection.anki2"
    _build_schema18_collection(path, review_note_text="no compile id field here",
                               fields=("Thai", "ReviewNote"))

    report = import_collection(path, fx.db,
                      current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert report.notes_harvested == 1

    rows = [r for r in fx.db.assessments_of("rice") if r.backend == "learner-note"]
    assert len(rows) == 1
    assert rows[0].question["compile_id"] == ""
