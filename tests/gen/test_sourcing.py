"""What a sourcing record must hold, stated without reference to images.

The record is a decision log over subjects: what was tried to picture one,
and how it ended. Nothing here needs a network, a judge, or a deck.
"""
from thai_deck_gen.media.sourcing import (Attempt, Candidate, Decision,
                                          Record, SourcingLog, next_mechanism)


def _attempt(query="red color", passed=False, dated="2026-09-01"):
    return Attempt(
        query=query, query_source="gloss", corpora=("openverse",),
        rubric="abc123",
        candidates=(Candidate(url="http://o/1.jpg", source="openverse",
                              license="cc0", file="0.jpg", passed=passed,
                              failed_rules=() if passed
                              else ("judge/image-irrelevant",)),),
        dated=dated)


def test_an_appended_attempt_survives_a_process_that_never_saves(tmp_path):
    """A run killed inside the image filler is the normal case here, not the
    exception: it already cost 445 files their provenance once."""
    log = SourcingLog.load(tmp_path)
    log.record_attempt("picture_word", "word-a", _attempt())

    reloaded = SourcingLog.load(tmp_path)
    assert len(reloaded.get("picture_word", "word-a").attempts) == 1
    assert reloaded.get("picture_word", "word-a").attempts[0].query == "red color"


def test_attempts_accumulate_in_order(tmp_path):
    """The history is the point. An attempt that overwrote the last one would
    lose exactly the evidence a rephrase needs."""
    log = SourcingLog.load(tmp_path)
    log.record_attempt("picture_word", "word-a", _attempt(query="first"))
    log.record_attempt("picture_word", "word-a", _attempt(query="second"))

    queries = [a.query for a in SourcingLog.load(tmp_path)
               .get("picture_word", "word-a").attempts]
    assert queries == ["first", "second"]


def test_a_subject_with_no_events_reads_as_an_empty_record(tmp_path):
    """Absence must be a record, not a KeyError: every caller asks about
    subjects that have never been attempted."""
    record = SourcingLog.load(tmp_path).get("picture_word", "unseen")
    assert record.attempts == [] and record.decision is None


def test_the_same_subject_in_two_families_is_two_records(tmp_path):
    """A spelling pattern and a word can be the same string."""
    log = SourcingLog.load(tmp_path)
    log.record_attempt("picture_word", "same", _attempt(query="w"))
    log.record_attempt("spelling_sound", "same", _attempt(query="p"))

    reloaded = SourcingLog.load(tmp_path)
    assert [a.query for a in reloaded.get("picture_word", "same").attempts] == ["w"]
    assert [a.query for a in reloaded.get("spelling_sound", "same").attempts] == ["p"]


def test_a_torn_line_does_not_lose_the_rest_of_the_log(tmp_path):
    """A killed write leaves a partial line. Losing the whole history to it
    would defeat the reason for appending."""
    log = SourcingLog.load(tmp_path)
    log.record_attempt("picture_word", "word-a", _attempt())
    path = tmp_path / "work" / "image_sourcing.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"family": "pic',
                    encoding="utf-8")

    assert len(SourcingLog.load(tmp_path).get("picture_word", "word-a").attempts) == 1


def test_the_last_decision_for_a_subject_wins(tmp_path):
    """A decision can be revised; the log is append-only, so the fold decides
    which one stands."""
    log = SourcingLog.load(tmp_path)
    log.record_decision("picture_word", "word-a",
                        Decision(kind="judge-accepted", file="images/a.jpg",
                                 reason=None, dated="2026-09-01"))
    log.record_decision("picture_word", "word-a",
                        Decision(kind="human-supplied", file="images/b.jpg",
                                 reason="hand picked", dated="2026-09-02"))

    decision = SourcingLog.load(tmp_path).get("picture_word", "word-a").decision
    assert decision.kind == "human-supplied"
    assert decision.file == "images/b.jpg"


def test_save_compacts_without_changing_what_loads(tmp_path):
    """Appending grows the log; save() is the compaction, and it must be a
    no-op semantically or a compaction could silently rewrite history."""
    log = SourcingLog.load(tmp_path)
    log.record_attempt("picture_word", "word-a", _attempt(query="first"))
    log.record_attempt("picture_word", "word-a", _attempt(query="second"))
    log.record_decision("picture_word", "word-a",
                        Decision(kind="judge-accepted", file="images/a.jpg",
                                 reason=None, dated="2026-09-01"))
    before = SourcingLog.load(tmp_path).get("picture_word", "word-a")

    SourcingLog.load(tmp_path).save(tmp_path)
    after = SourcingLog.load(tmp_path).get("picture_word", "word-a")

    assert [a.query for a in after.attempts] == [a.query for a in before.attempts]
    assert after.decision.kind == before.decision.kind


def test_an_unknown_decision_kind_is_refused(tmp_path):
    """The four kinds are the vocabulary the derived state reads. A fifth
    written by a typo would settle a subject in a way nothing understands."""
    import pytest
    log = SourcingLog.load(tmp_path)
    with pytest.raises(ValueError):
        log.record_decision("picture_word", "word-a",
                            Decision(kind="probably-fine", file=None,
                                     reason=None, dated="2026-09-01"))


# --- which mechanism a subject is owed, derived from its record ---

RUBRIC = "abc123"


def _tried(query, source="gloss", rubric=RUBRIC):
    return Attempt(query=query, query_source=source, corpora=("openverse",),
                   rubric=rubric, candidates=(), dated="2026-09-01")


def test_a_subject_never_attempted_is_searched():
    assert next_mechanism(Record("picture_word", "a"), ["red color"],
                          RUBRIC) == "search"


def test_a_query_that_has_not_been_tried_earns_a_search():
    """A new phrase is new information. This is what makes editing the word
    list take effect without a flag to reset."""
    record = Record("picture_word", "a", attempts=[_tried("red color")])
    assert next_mechanism(record, ["red color", "crimson"], RUBRIC) == "search"


def test_a_changed_rubric_reopens_a_subject_already_searched():
    """Relaxing what disqualifies an image makes yesterday's rejections worth
    reconsidering; so does adding a library nobody had searched."""
    record = Record("picture_word", "a", attempts=[_tried("red color")])
    assert next_mechanism(record, ["red color"], "different") == "search"


def test_exhausted_queries_move_the_subject_to_rephrase():
    record = Record("picture_word", "a", attempts=[_tried("red color")])
    assert next_mechanism(record, ["red color"], RUBRIC) == "rephrase"


def test_a_failed_rephrase_moves_the_subject_to_consult():
    """Rephrase gets one turn. A second is the same model with the same
    evidence, and paying twice for that is the loop this replaces."""
    record = Record("picture_word", "a", attempts=[
        _tried("red color"), _tried("crimson swatch", source="judge")])
    assert next_mechanism(record, ["red color", "crimson swatch"],
                          RUBRIC) == "consult"


def test_a_consulted_subject_waits_for_the_human():
    record = Record("picture_word", "a", attempts=[
        _tried("red color"), _tried("crimson swatch", source="judge"),
        _tried("scarlet paint", source="human")])
    assert next_mechanism(record, ["red color", "crimson swatch",
                                   "scarlet paint"], RUBRIC) == "waiting"


def test_any_decision_settles_the_subject():
    for kind in ("judge-accepted", "human-accepted", "human-supplied",
                 "human-unpicturable"):
        record = Record("picture_word", "a", attempts=[_tried("red color")],
                        decision=Decision(kind=kind, file=None, reason=None,
                                          dated="2026-09-01"))
        assert next_mechanism(record, ["red color", "new"], RUBRIC) == "settled", kind


def test_a_human_decision_outranks_an_untried_query():
    """An automated rule may warn about a human decision. It may never
    reverse one, and a new query is not grounds to reopen it."""
    record = Record("picture_word", "a",
                    decision=Decision(kind="human-unpicturable", file=None,
                                      reason="no photograph serves this",
                                      dated="2026-09-01"))
    assert next_mechanism(record, ["anything at all"], RUBRIC) == "settled"
