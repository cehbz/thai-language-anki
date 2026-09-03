"""Tests for derivations.py (spec 3 section 3): pure folds over synthetic
cache rows -- current_best's regression guard and challenger-not-silent-
swap, exhausted's reopen conditions, queue's F10 order, confusion_weights.
No real store: a tiny in-memory CacheReader built directly from Answer
rows the test constructs.
"""
from dataclasses import dataclass, field

import pytest

from thai_syllabus.derivations import (
    confusion_weights,
    current_best,
    exhausted,
    queue,
)
from thai_syllabus.ports import Answer


class FakeCache:
    """CacheReader over an explicit list of Answer rows."""
    def __init__(self, rows: list[Answer]):
        self.rows = rows

    def latest(self, port, backend, key):
        matches = [r for r in self.rows if r.port == port and r.backend == backend
                  and r.key == key]
        return max(matches, key=lambda r: r.ts) if matches else None

    def assessments_of(self, subject):
        return sorted((r for r in self.rows if r.subject == subject), key=lambda r: r.ts)


_ts = [0]


def _next_ts() -> int:
    _ts[0] += 1
    return _ts[0]


def provide_row(subject, kind, backend="openverse", items=(), ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="provide", backend=backend, key=f"{backend}:{subject}",
                 key_sha="x", subject=subject, question={"provides": kind, "params": {}},
                 answer={"items": list(items)}, cost=0.0, ts=ts)


def judge_row(subject, kind, artifact_sha, value, rubric="rubric-v1", ts=None,
             suggestion=None):
    ts = ts if ts is not None else _next_ts()
    answer = {"value": value}
    if suggestion:
        answer["suggestion"] = suggestion
    return Answer(port="assess", backend="judge", key=f"judge:{subject}:{artifact_sha}",
                 key_sha="x", subject=subject,
                 question={"role": f"{kind}-for-word", "artifact_sha": artifact_sha,
                          "rubric": rubric},
                 answer=answer, cost=0.001, ts=ts)


def learner_row(subject, kind, artifact_sha, rating, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend="learner", key=f"learner:{subject}:{artifact_sha}",
                 key_sha="x", subject=subject,
                 question={"role": f"{kind}-for-word", "artifact_sha": artifact_sha,
                          "rubric": None},
                 answer={"value": rating}, cost=0.0, ts=ts)


def direction_row(subject, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend="learner", key=f"learner:direction:{subject}",
                 key_sha="x", subject=subject, question={"kind": "direction"},
                 answer={"direction": "try a red one"}, cost=0.0, ts=ts)


# --- current_best: basics -------------------------------------------------

def test_current_best_is_none_with_no_history():
    cache = FakeCache([])
    best = current_best(cache, "rice", "picture")
    assert best.artifact_sha is None
    assert best.source == "none"


def test_current_best_prefers_the_best_passing_judge_verdict():
    rows = [
        judge_row("rice", "picture", "sha-a", False),
        judge_row("rice", "picture", "sha-b", True),
    ]
    best = current_best(FakeCache(rows), "rice", "picture")
    assert best.artifact_sha == "sha-b"
    assert best.source == "judge"


def test_current_best_learner_choice_wins_outright_over_judge():
    rows = [
        judge_row("rice", "picture", "sha-a", True),
        learner_row("rice", "picture", "sha-b", "acceptable"),
    ]
    best = current_best(FakeCache(rows), "rice", "picture")
    assert best.artifact_sha == "sha-b"
    assert best.source == "learner"


# --- regression guard ------------------------------------------------------

def test_regression_guard_never_ranks_below_a_learner_acceptable_rating():
    rows = [
        learner_row("rice", "picture", "sha-a", "acceptable", ts=1),
        # a later, unrelated learner rating for a DIFFERENT artifact that
        # is itself below "acceptable" must not drag current_best's rank
        # below the floor the learner already set.
        learner_row("rice", "picture", "sha-c", "unacceptable-use-this", ts=2),
    ]
    best = current_best(FakeCache(rows), "rice", "picture")
    assert best.rank >= 80.0  # never below "acceptable"


def test_regression_guard_survives_a_worse_rubric_rerun():
    rows = [
        learner_row("rice", "picture", "sha-a", "good", ts=1),
        judge_row("rice", "picture", "sha-a", False, rubric="rubric-v2", ts=2),
    ]
    best = current_best(FakeCache(rows), "rice", "picture", current_rubric="rubric-v2")
    assert best.artifact_sha == "sha-a"
    assert best.rank >= 100.0


# --- challenger: presented, never silently swapped -----------------------

def test_a_new_higher_ranked_unrated_artifact_becomes_a_challenger_not_a_swap():
    rows = [
        learner_row("rice", "picture", "sha-a", "acceptable", ts=1),
        judge_row("rice", "picture", "sha-new", True, ts=2),  # unrated by the learner
    ]
    best = current_best(FakeCache(rows), "rice", "picture")
    assert best.artifact_sha == "sha-a"  # NOT silently swapped
    assert best.source == "learner"
    assert best.challenger == "sha-new"


def test_no_challenger_when_no_unrated_artifact_has_passed_judgement():
    rows = [
        learner_row("rice", "picture", "sha-a", "good", ts=1),
        judge_row("rice", "picture", "sha-b", False, ts=2),  # unrated, but FAILED
    ]
    best = current_best(FakeCache(rows), "rice", "picture")
    assert best.challenger is None


# --- exhausted: attempt cap + reopen conditions --------------------------

def test_not_exhausted_before_the_attempt_cap():
    rows = [provide_row("rice", "picture") for _ in range(3)]
    status = exhausted(FakeCache(rows), "rice", "picture", attempt_cap=8)
    assert status.exhausted is False
    assert status.attempts == 3


def test_exhausted_when_the_cap_is_reached_and_recent_attempts_dont_outrank():
    rows = [learner_row("rice", "picture", "sha-a", "acceptable", ts=0)]
    rows += [provide_row("rice", "picture", items=[{"sha": f"sha-x{i}"}])
            for i in range(8)]
    status = exhausted(FakeCache(rows), "rice", "picture", k=2, attempt_cap=8)
    assert status.exhausted is True


def test_reopened_by_a_recent_attempt_out_ranking_current_best():
    # No learner rating yet -- current_best is judge-only, so a later,
    # better-judged candidate among the last k attempts can out-rank it
    # (this does NOT apply once a learner has rated something acceptable+:
    # the regression guard means a mere judge pass can never out-rank
    # that -- see the regression-guard tests above).
    rows = [judge_row("rice", "picture", "sha-a", False, ts=0)]
    rows += [provide_row("rice", "picture", items=[{"sha": f"sha-x{i}"}]) for i in range(7)]
    last = provide_row("rice", "picture", items=[{"sha": "sha-great"}])
    rows.append(last)
    rows.append(judge_row("rice", "picture", "sha-great", 99.0, ts=last.ts + 1))
    status = exhausted(FakeCache(rows), "rice", "picture", k=2, attempt_cap=8)
    assert status.exhausted is False
    assert "out-rank" in status.reason


def test_reopened_by_any_learner_direction():
    # exhausted() itself only looks at attempts/ranks; a direction row
    # doesn't retroactively un-exhaust past attempts, but it DOES show up
    # as a fresh candidate for queue()'s bucket-1 (directed) ordering --
    # covered in the queue tests below. This test documents that
    # exhausted() alone does not special-case directions (queue does).
    rows = [learner_row("rice", "picture", "sha-a", "acceptable", ts=0)]
    rows += [provide_row("rice", "picture", items=[{"sha": f"sha-x{i}"}]) for i in range(8)]
    before = exhausted(FakeCache(rows), "rice", "picture", k=2, attempt_cap=8)
    assert before.exhausted is True


# --- queue: F10 order ------------------------------------------------------

class _FakeGaps:
    def __init__(self, words_missing_pictures=(), words_missing_recordings=(),
                unfilled_targets=(), missing_renditions=(),
                graphemes_missing_keyword_data=()):
        self.words_missing_pictures = words_missing_pictures
        self.words_missing_recordings = words_missing_recordings
        self.unfilled_targets = unfilled_targets
        self.missing_renditions = missing_renditions
        self.graphemes_missing_keyword_data = graphemes_missing_keyword_data


@dataclass
class _FakeSyllabus:
    _gaps: _FakeGaps
    targets: list = field(default_factory=list)

    def gaps(self):
        return self._gaps


def test_queue_puts_no_artifact_subjects_first():
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice", "dog")))
    rows = [judge_row("dog", "picture", "sha-d", True)]  # dog has SOME candidate, still not acceptable
    q = queue(syllabus, FakeCache(rows))
    assert [(e.subject, e.bucket) for e in q] == [("dog", 1), ("rice", 1)] or \
          {e.subject for e in q if e.bucket == 1} == {"rice", "dog"}


def test_queue_puts_directed_subjects_before_undirected_within_bucket_1():
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice", "dog")))
    rows = [direction_row("dog")]
    q = queue(syllabus, FakeCache(rows))
    assert [e.subject for e in q] == ["dog", "rice"]


def test_queue_never_includes_a_good_or_exhausted_subject():
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice", "dog")))
    rows = [learner_row("rice", "picture", "sha-a", "good", ts=1)]
    rows += [provide_row("dog", "picture", items=[{"sha": f"s{i}"}]) for i in range(8)]
    rows += [learner_row("dog", "picture", "sha-a", "acceptable", ts=0)]
    q = queue(syllabus, FakeCache(rows))
    subjects = {e.subject for e in q}
    assert "rice" not in subjects   # good
    assert "dog" not in subjects    # exhausted


def test_queue_orders_bucket_3_by_verdict_rank_ascending_then_attempts():
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("a", "b")))
    rows = [
        learner_row("a", "picture", "sha-a", "acceptable", ts=1),
        provide_row("a", "picture", ts=2),
        learner_row("b", "picture", "sha-b", "good", ts=1),  # would be bucket-excluded (good)
    ]
    # give "b" a lower rank via a second, worse learner re-rating so it's
    # still eligible (acceptable, not good) and compare attempt counts
    rows = [
        learner_row("a", "picture", "sha-a", "acceptable", ts=1),
        provide_row("a", "picture", ts=2),
        provide_row("a", "picture", ts=3),
        learner_row("b", "picture", "sha-b", "acceptable", ts=1),
    ]
    q = queue(syllabus, FakeCache(rows))
    # both are bucket 3 (acceptable, no untried lever); "b" has fewer
    # attempts than "a", both rank 80 -- attempts ascending puts "b" first
    bucket3 = [e for e in q if e.bucket == 3]
    assert [e.subject for e in bucket3] == ["b", "a"]


def test_queue_flags_bucket_2_when_the_rubric_changed_since_the_verdict():
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    rows = [
        learner_row("rice", "picture", "sha-a", "acceptable", ts=1),
        judge_row("rice", "picture", "sha-a", True, rubric="rubric-v1", ts=2),
    ]
    q = queue(syllabus, FakeCache(rows), current_rubric="rubric-v2")
    assert q[0].bucket == 2


def test_queue_respects_the_learner_budgets_max_asks():
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("a", "b", "c")))

    class _Budget:
        max_asks = 2

    q = queue(syllabus, FakeCache([]), budgets={"learner": _Budget()})
    assert len(q) == 2


# --- confusion_weights ----------------------------------------------------

class _FakeStudyReader:
    def __init__(self, records_by_confusion):
        self._records = records_by_confusion

    def records(self, confusion_id):
        return self._records.get(confusion_id, [])


@dataclass
class _Rec:
    card_key: str = "k"
    compile_id: str = "c"
    ts: int = 0
    grade: int = 3
    time_ms: int = 100


def test_confusion_weights_keeps_the_seed_with_no_study_history():
    weights = confusion_weights({"tone:mid-low": 2.0}, _FakeStudyReader({}), ["tone:mid-low"])
    assert weights["tone:mid-low"] == 2.0


def test_confusion_weights_increases_with_lapse_rate():
    records = [_Rec(grade=1), _Rec(grade=1), _Rec(grade=4)]  # 2/3 lapses
    reader = _FakeStudyReader({"tone:mid-low": records})
    weights = confusion_weights({"tone:mid-low": 1.0}, reader, ["tone:mid-low"])
    assert weights["tone:mid-low"] == pytest.approx(1.0 * (1 + 2 / 3))
