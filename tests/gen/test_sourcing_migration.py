"""Folding the old image stores into records.

Two of the four hold content the record cannot regenerate. These pin what
must survive the fold, and what must not be invented by it.
"""
import yaml

from thai_deck_gen.media.migrate_sourcing import migrate
from thai_deck_gen.media.sourcing import SourcingLog


def _old_stores(root, review_items, candidate_rows):
    work = root / "work"
    (work / "candidates" / "pw-0").mkdir(parents=True, exist_ok=True)
    (work / "image_review.yaml").write_text(
        yaml.safe_dump({"items": review_items}, allow_unicode=True),
        encoding="utf-8")
    (work / "candidates" / "pw-0" / "candidates.yaml").write_text(
        yaml.safe_dump({"corpora": ["openverse"], "candidates": candidate_rows},
                       allow_unicode=True), encoding="utf-8")


def test_the_failed_query_history_survives_migration(tmp_path):
    """This is the only store with content the record cannot regenerate:
    without it every previously exhausted word is searched again from
    scratch, and the rephrase step starts blind."""
    _old_stores(tmp_path,
                review_items=[{"note_id": "pw-0", "term": "word-a",
                               "tried": ["openverse"],
                               "queries": ["red color", "crimson"],
                               "rubric": "abc123"}],
                candidate_rows=[])

    assert migrate(tmp_path, {"pw-0": ("picture_word", "word-a")},
                   "2026-09-01") == 1

    record = SourcingLog.load(tmp_path).get("picture_word", "word-a")
    assert record.queries_tried == ["red color", "crimson"]
    assert {a.rubric for a in record.attempts} == {"abc123"}


def test_candidate_verdicts_survive_migration(tmp_path):
    """2,178 candidates were judged to produce these. Re-judging them would
    cost cash to learn what is already on disk."""
    _old_stores(tmp_path,
                review_items=[{"note_id": "pw-0", "term": "word-a",
                               "tried": ["openverse"], "queries": ["red color"],
                               "rubric": "abc123"}],
                candidate_rows=[
                    {"file": "0.jpg", "url": "http://o/0.jpg",
                     "source": "openverse", "license": "cc0", "passed": False,
                     "failed_rules": ["judge/image-irrelevant"],
                     "accepted": False}])

    migrate(tmp_path, {"pw-0": ("picture_word", "word-a")}, "2026-09-01")

    candidates = SourcingLog.load(tmp_path).get(
        "picture_word", "word-a").attempts[0].candidates
    assert len(candidates) == 1
    assert candidates[0].url == "http://o/0.jpg"
    assert candidates[0].failed_rules == ("judge/image-irrelevant",)


def test_an_accepted_candidate_migrates_as_a_judge_decision(tmp_path):
    """A word with an accepted picture is settled, and must not be
    re-searched the first time the new code runs."""
    _old_stores(tmp_path,
                review_items=[],
                candidate_rows=[
                    {"file": "0.jpg", "url": "http://o/0.jpg",
                     "source": "openverse", "license": "cc0", "passed": True,
                     "failed_rules": [], "accepted": True}])

    migrate(tmp_path, {"pw-0": ("picture_word", "word-a")}, "2026-09-01")

    decision = SourcingLog.load(tmp_path).get("picture_word", "word-a").decision
    assert decision is not None and decision.kind == "judge-accepted"


def test_a_note_the_caller_cannot_name_is_skipped(tmp_path):
    """The old stores key on note id and outlive the notes. A stale entry is
    not a subject and must not invent one."""
    _old_stores(tmp_path,
                review_items=[{"note_id": "pw-99", "term": "gone",
                               "tried": [], "queries": ["x"], "rubric": "r"}],
                candidate_rows=[])

    assert migrate(tmp_path, {}, "2026-09-01") == 0
    assert SourcingLog.load(tmp_path).records() == []


def test_migration_twice_does_not_double_the_history(tmp_path):
    """Migration is a one-shot over an append-only log, so it has to be safe
    to re-run: the first run may have been killed."""
    _old_stores(tmp_path,
                review_items=[{"note_id": "pw-0", "term": "word-a",
                               "tried": [], "queries": ["red color"],
                               "rubric": "abc123"}],
                candidate_rows=[])
    subjects = {"pw-0": ("picture_word", "word-a")}

    migrate(tmp_path, subjects, "2026-09-01")
    migrate(tmp_path, subjects, "2026-09-01")

    assert SourcingLog.load(tmp_path).get(
        "picture_word", "word-a").queries_tried == ["red color"]


def test_a_subject_with_nothing_recorded_about_it_is_not_written(tmp_path):
    """Most words have neither a review entry nor a candidate pool. Writing an
    empty record for each would put 654 subjects in the log saying nothing."""
    _old_stores(tmp_path, review_items=[], candidate_rows=[])

    assert migrate(tmp_path, {"pw-1": ("picture_word", "word-b")},
                   "2026-09-01") == 0
    assert SourcingLog.load(tmp_path).records() == []
