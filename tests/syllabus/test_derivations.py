"""Tests for derivations.py (spec 3 section 6): pure folds over synthetic
cache rows -- current_best's regression guard, pending, next_source,
exhausted, improved, directed, queue's F10 buckets, challengers, reasks,
confusion_weights. No real store for most cases: a tiny in-memory
CacheReader built directly from Answer rows the test constructs; a handful
of current_best cases exercise the real SyllabusDb for genuine
assessments_of ordering.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from thai_syllabus.derivations import (
    CurrentBest,
    adoptable_drafts,
    challengers,
    confusion_weights,
    current_best,
    directed,
    exhausted,
    improved,
    next_source,
    pending,
    queue,
    reasks,
)
from thai_syllabus.assessor import AssessQuestion, Assessor, JudgeBackend
from thai_syllabus.cachekeys import BatchMarkerKey
from thai_syllabus.entities import MinimalPair, SoundConfusion, text_sha
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.ports import Answer, StudyRecord
from thai_syllabus.store import SyllabusDb
from thai_syllabus.syllabus import Syllabus

from .builders import sentence, syl, target, word
from .fakes import FakeTokenizer


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


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


@pytest.fixture
def cache():
    return FakeCache([])


def _no_provenance(artifact_sha: str) -> str | None:
    """provenance_source for every call that doesn't test the provenance-
    prior tie-break itself: no candidate gets a bonus.
    """
    return None


_ts = [0]


def _next_ts() -> int:
    _ts[0] += 1
    return _ts[0]


def provide_row(subject, kind, backend="openverse", items=(), ts=None):
    """`kind` is the need kind (e.g. "picture"), never "picture-bytes" --
    a bytes-fetch row is distinguished by `backend` (imgfetch/audiofetch),
    not by a suffixed kind.
    """
    ts = ts if ts is not None else _next_ts()
    return Answer(port="provide", backend=backend, key=f"{backend}:{subject}:{ts}",
                 key_sha="x", subject=subject, question={"kind": kind, "params": {}},
                 answer={"items": list(items)}, cost=0.0, ts=ts)


def judge_row(subject, kind, artifact_sha, value, rubric="rubric-v1", ts=None,
             suggestion=None):
    ts = ts if ts is not None else _next_ts()
    answer = {"value": value}
    if suggestion:
        answer["suggestion"] = suggestion
    return Answer(port="assess", backend="judge", key=f"judge:{subject}:{artifact_sha}:{ts}",
                 key_sha="x", subject=subject,
                 question={"role": f"{kind}-for-word", "artifact_sha": artifact_sha,
                          "rubric": rubric, "kind": kind},
                 answer=answer, cost=0.001, ts=ts)


def learner_row(subject, kind, artifact_sha, rating, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend="learner", key=f"learner:{subject}:{artifact_sha}:{ts}",
                 key_sha="x", subject=subject,
                 question={"role": f"{kind}-for-word", "artifact_sha": artifact_sha,
                          "rubric": None, "kind": "rating"},
                 answer={"value": rating}, cost=0.0, ts=ts)


def direction_row(subject, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend="learner", key=f"learner:direction:{subject}:{ts}",
                 key_sha="x", subject=subject, question={"kind": "direction"},
                 answer={"direction": "try a red one"}, cost=0.0, ts=ts)


def card_flag_row(subject, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend="learner", key=f"learner:card-flag:{subject}:{ts}",
                 key_sha="x", subject=subject, question={"kind": "card-flag"},
                 answer={"note": "front is blank"}, cost=0.0, ts=ts)


def reverify_row(subject, role, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend="learner", key=f"learner:reverify:{subject}:{ts}",
                 key_sha="x", subject=subject,
                 question={"kind": "reverify", "role": role},
                 answer={"flagged": True}, cost=0.0, ts=ts)


def mechanical_row(subject, role, artifact_sha, value, ts=None, kind="recording",
                   backend="mechanical"):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend=backend, key=f"mech:{subject}:{artifact_sha}:{ts}",
                 key_sha="x", subject=subject,
                 question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                          "kind": kind},
                 answer={"value": value}, cost=0.0, ts=ts)


# --- fixture helpers matching the task brief's representative scenarios ----

def seed_ask(cache, subject, kind, *, source, ts):
    cache.rows.append(provide_row(subject, kind, backend=source, ts=ts))


def seed_artifact(cache, subject, artifact_sha, *, ts, judge_pass, rubric="rubric-v1"):
    """One provide row producing `artifact_sha` (a bytes-fetch row, not a
    Source ask of its own) and, when `judge_pass`, a passing judge verdict
    on it at the same ts.
    """
    cache.rows.append(provide_row(subject, "picture", backend="imgfetch",
                                  items=[{"sha": artifact_sha}], ts=ts))
    if judge_pass:
        cache.rows.append(judge_row(subject, "picture", artifact_sha, True, rubric=rubric, ts=ts))


def seed_judge_pass(cache, subject, artifact_sha, *, rubric):
    seed_artifact(cache, subject, artifact_sha, ts=_next_ts(), judge_pass=True, rubric=rubric)


def seed_rating(cache, subject, artifact_sha, rating):
    cache.rows.append(learner_row(subject, "picture", artifact_sha, rating, ts=_next_ts()))


def seed_card_flag(cache, subject):
    cache.rows.append(card_flag_row(subject, ts=_next_ts()))


R = "rubric-v1"


def sources_for(kind):
    return ("openverse", "wikimedia", "pexels")


def no_sources(kind):
    return ()


# --- current_best: basics -------------------------------------------------

def test_current_best_is_none_with_no_history(cache):
    best = current_best(cache, "rice", "picture", current_rubric={}, prior=(),
                        provenance_source=_no_provenance)
    assert best.artifact_sha is None
    assert best.source is None


def test_current_best_prefers_the_best_passing_judge_verdict(cache):
    cache.rows += [
        judge_row("rice", "picture", "sha-a", False),
        judge_row("rice", "picture", "sha-b", True),
    ]
    best = current_best(cache, "rice", "picture", current_rubric={}, prior=(),
                        provenance_source=_no_provenance)
    assert best.artifact_sha == "sha-b"
    assert best.source == "judge"


def test_current_best_carries_the_speaker_a_provide_item_names(cache):
    cache.rows.append(provide_row("pair-1", "rendition", backend="forvo",
                                  items=[{"member": "near", "sha": "a" * 64,
                                         "speaker": {"id": "forvo:somchai", "kind": "native",
                                                    "sex": "male"}}]))
    cache.rows.append(mechanical_row("pair-1", "rendition-for-pair", "a" * 64, True,
                                     kind="rendition", backend="rendition"))
    best = current_best(cache, "pair-1", "rendition", current_rubric={}, prior=(),
                        provenance_source=_no_provenance)
    assert best.speaker.id == "forvo:somchai"
    assert best.speaker.sex == "male"


def test_current_best_learner_choice_wins_outright_over_judge(cache):
    cache.rows += [
        judge_row("rice", "picture", "sha-a", True),
        learner_row("rice", "picture", "sha-b", "acceptable"),
    ]
    best = current_best(cache, "rice", "picture", current_rubric={}, prior=(),
                        provenance_source=_no_provenance)
    assert best.artifact_sha == "sha-b"
    assert best.source == "learner"


# --- regression guard ------------------------------------------------------

def test_regression_guard_never_ranks_below_a_learner_acceptable_rating(cache):
    cache.rows += [
        learner_row("rice", "picture", "sha-a", "acceptable", ts=1),
        # a later, unrelated learner rating for a DIFFERENT artifact that
        # is itself below "acceptable" must not drag current_best's rank
        # below the floor the learner already set.
        learner_row("rice", "picture", "sha-c", "unacceptable-use-this", ts=2),
    ]
    best = current_best(cache, "rice", "picture", current_rubric={}, prior=(),
                        provenance_source=_no_provenance)
    assert best.rank >= 80.0  # never below "acceptable"


def test_regression_guard_survives_a_worse_rubric_rerun(cache):
    cache.rows += [
        learner_row("rice", "picture", "sha-a", "good", ts=1),
        judge_row("rice", "picture", "sha-a", False, rubric="rubric-v2", ts=2),
    ]
    best = current_best(cache, "rice", "picture",
                        current_rubric={"picture-for-word": "rubric-v2"}, prior=(),
                        provenance_source=_no_provenance)
    assert best.artifact_sha == "sha-a"
    assert best.rank >= 100.0


# --- improved --------------------------------------------------------------

def test_improved_is_false_when_only_the_rank_changed():
    assert improved(CurrentBest("a" * 64, "judge", 50.0), CurrentBest("a" * 64, "judge", 60.0)) is False


def test_improved_is_true_on_a_new_artifact():
    assert improved(CurrentBest(None, None, -1.0), CurrentBest("a" * 64, "judge", 50.0)) is True


def test_improved_is_false_when_the_new_pick_is_also_none():
    assert improved(CurrentBest(None, None, -1.0), CurrentBest(None, None, -1.0)) is False


# --- pending -----------------------------------------------------------

def _batch_marker_submitted(db, batch_id, subjects, roles):
    db.append(port="assess", backend="judge", key=BatchMarkerKey(batch_id), subject="batch",
              question={"kind": "batch", "batch_id": batch_id, "subjects": list(subjects),
                       "roles": list(roles)},
              answer={"status": "submitted"})


def _batch_marker_resolved(db, batch_id, status="resolved"):
    db.append(port="assess", backend="judge", key=BatchMarkerKey(batch_id), subject="batch",
              question={"kind": "batch", "batch_id": batch_id}, answer={"status": status})


def test_pending_true_while_submitted_false_once_resolved(db):
    _batch_marker_submitted(db, "b1", ["w"], ["picture-for-word"])
    assert pending(db, "w", "picture") is True
    _batch_marker_resolved(db, "b1")
    assert pending(db, "w", "picture") is False


def test_pending_via_assessor_submit_and_resolve(db):
    """The same contract, exercised through Assessor.submit()/resolve()
    rather than a hand-built marker row.
    """
    class BT:
        def __init__(self):
            self.status_value = "in_progress"

        def submit(self, requests):
            return "b1"

        def status(self, batch_id):
            return self.status_value

        def results(self, batch_id):
            return {}

    bt = BT()
    jb = JudgeBackend(model="m", transport="batch", batch_transport=bt)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="a", rubric="r",
                       kind="picture")
    bid = a.submit(a.ask_many("judge", [q]).collected)

    assert pending(db, "w", "picture") is True

    bt.status_value = "ended"
    a.resolve(bid)

    assert pending(db, "w", "picture") is False


# --- next_source -----------------------------------------------------------

def test_next_source_skips_a_source_asked_since_current_best_last_changed(cache):
    seed_ask(cache, "rice", "picture", source="wikimedia", ts=1)  # before any artifact existed
    seed_artifact(cache, "rice", "a" * 64, ts=2, judge_pass=True)  # current-best changes here
    seed_ask(cache, "rice", "picture", source="wikimedia", ts=3)  # asked again, since the change
    assert next_source(cache, "rice", "picture", ("openverse", "wikimedia", "pexels")) == "openverse"


def test_next_source_with_no_artifact_counts_every_ask(cache):
    seed_ask(cache, "rice", "picture", source="openverse", ts=1)
    assert next_source(cache, "rice", "picture", ("openverse", "wikimedia", "pexels")) == "wikimedia"


def test_next_source_is_none_once_every_source_asked_since_the_change(cache):
    seed_artifact(cache, "rice", "a" * 64, ts=1, judge_pass=True)
    seed_ask(cache, "rice", "picture", source="openverse", ts=2)
    seed_ask(cache, "rice", "picture", source="wikimedia", ts=3)
    seed_ask(cache, "rice", "picture", source="pexels", ts=4)
    assert next_source(cache, "rice", "picture", ("openverse", "wikimedia", "pexels")) is None


# --- exhausted ---------------------------------------------------------

def test_not_exhausted_while_a_source_remains_untried(cache):
    seed_ask(cache, "rice", "picture", source="openverse", ts=1)
    status = exhausted(cache, "rice", "picture", sources=("openverse", "wikimedia"), attempt_cap=8)
    assert status.exhausted is False


def test_exhausted_once_every_source_is_asked_since_the_change(cache):
    seed_artifact(cache, "rice", "a" * 64, ts=1, judge_pass=False)
    seed_ask(cache, "rice", "picture", source="openverse", ts=2)
    seed_ask(cache, "rice", "picture", source="wikimedia", ts=3)
    status = exhausted(cache, "rice", "picture", sources=("openverse", "wikimedia"), attempt_cap=8)
    assert status.exhausted is True
    assert status.attempts == 2


def test_exhausted_once_the_attempt_cap_is_reached_even_with_sources_left(cache):
    seed_ask(cache, "rice", "picture", source="openverse", ts=1)
    status = exhausted(cache, "rice", "picture", sources=("openverse", "wikimedia", "pexels"),
                       attempt_cap=1)
    assert status.exhausted is True


def test_reopened_by_a_new_source_in_the_roster(cache):
    seed_artifact(cache, "rice", "a" * 64, ts=1, judge_pass=False)
    seed_ask(cache, "rice", "picture", source="openverse", ts=2)
    status_before = exhausted(cache, "rice", "picture", sources=("openverse",), attempt_cap=8)
    status_after = exhausted(cache, "rice", "picture", sources=("openverse", "wikimedia"),
                             attempt_cap=8)
    assert status_before.exhausted is True
    assert status_after.exhausted is False


# --- directed ------------------------------------------------------------

def test_directed_true_on_a_direction_row(cache):
    cache.rows.append(direction_row("rice"))
    assert directed(cache, "rice") is True


def test_directed_true_on_a_card_flag_row(cache):
    cache.rows.append(card_flag_row("rice"))
    assert directed(cache, "rice") is True


def test_directed_true_on_an_unconsumed_reverify_row(cache):
    cache.rows.append(reverify_row("rice", "recording-for-word", ts=1))
    assert directed(cache, "rice") is True


def test_directed_false_once_a_reverify_row_is_answered_by_a_newer_mechanical_verdict(cache):
    cache.rows.append(reverify_row("rice", "recording-for-word", ts=1))
    cache.rows.append(mechanical_row("rice", "recording-for-word", "sha-a", True, ts=2))
    assert directed(cache, "rice") is False


def test_directed_false_with_no_flags_at_all(cache):
    cache.rows.append(judge_row("rice", "picture", "sha-a", True))
    assert directed(cache, "rice") is False


# --- queue: F10 buckets ------------------------------------------------------

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
    words: list = field(default_factory=list)
    pairs: list = field(default_factory=list)

    def gaps(self):
        return self._gaps


def _one_word_syllabus(subject="rice"):
    return _FakeSyllabus(_FakeGaps(words_missing_pictures=(subject,)))


def _queue(syllabus, cache, **kwargs):
    kwargs.setdefault("current_rubric", {"picture-for-word": R})
    kwargs.setdefault("prior", ())
    kwargs.setdefault("sources_for", sources_for)
    kwargs.setdefault("attempt_cap", 8)
    kwargs.setdefault("provenance_source", _no_provenance)
    return queue(syllabus, cache, **kwargs)


def test_bucket_1_when_no_artifact_exists(cache):
    syllabus = _one_word_syllabus()
    entries = _queue(syllabus, cache)
    assert [e.bucket for e in entries] == [1]


def test_bucket_1_when_the_learner_rejected_the_current_artifact(cache):
    syllabus = _one_word_syllabus()
    seed_judge_pass(cache, "rice", "a" * 64, rubric=R)
    seed_rating(cache, "rice", "a" * 64, "unacceptable-use-this")
    entry = next(e for e in _queue(syllabus, cache) if e.subject == "rice")
    assert entry.bucket == 1


def test_bucket_1_no_artifact_excluded_once_exhausted_and_undirected(cache):
    syllabus = _one_word_syllabus()
    seed_ask(cache, "rice", "picture", source="openverse", ts=1)
    seed_ask(cache, "rice", "picture", source="wikimedia", ts=2)
    seed_ask(cache, "rice", "picture", source="pexels", ts=3)
    entries = _queue(syllabus, cache)
    assert entries == []


def test_bucket_1_exhausted_but_directed_stays_queued(cache):
    syllabus = _one_word_syllabus()
    seed_ask(cache, "rice", "picture", source="openverse", ts=1)
    seed_ask(cache, "rice", "picture", source="wikimedia", ts=2)
    seed_ask(cache, "rice", "picture", source="pexels", ts=3)
    seed_card_flag(cache, "rice")
    entries = _queue(syllabus, cache)
    assert len(entries) == 1
    assert entries[0].bucket == 1
    assert entries[0].directed is True


def test_bucket_2_when_the_rubric_changed_since_the_verdict(cache):
    # The artifact is anchored by a LEARNER rating (unaffected by rubric
    # staleness -- learner choice wins outright), so it stays current-best
    # while its own judge verdict, under the old rubric, is stale under
    # the new one -- exactly "a rubric change left the current artifact
    # without a verdict under current_rubric".
    syllabus = _one_word_syllabus()
    seed_rating(cache, "rice", "a" * 64, "acceptable")
    cache.rows.append(judge_row("rice", "picture", "a" * 64, True, rubric="rubric-v1",
                                ts=_next_ts()))
    entry = next(e for e in _queue(syllabus, cache,
                                   current_rubric={"picture-for-word": "rubric-v2"},
                                   sources_for=no_sources)
                if e.subject == "rice")
    assert entry.bucket == 2


def test_bucket_2_when_a_judge_suggestion_is_unasked(cache):
    syllabus = _one_word_syllabus()
    cache.rows.append(provide_row("rice", "picture", backend="imgfetch",
                                  items=[{"sha": "a" * 64}], ts=1))
    cache.rows.append(judge_row("rice", "picture", "a" * 64, True, ts=2, suggestion="a redder one"))
    entry = next(e for e in _queue(syllabus, cache, sources_for=no_sources) if e.subject == "rice")
    assert entry.bucket == 2


def test_judge_passed_unrated_picture_queues_in_bucket_3(cache):
    syllabus = _one_word_syllabus()
    seed_judge_pass(cache, "rice", "a" * 64, rubric=R)
    entry = next(e for e in _queue(syllabus, cache, sources_for=no_sources) if e.subject == "rice")
    assert entry.bucket == 3


def test_bucket_3_orders_by_rank_ascending_then_attempts(cache):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("a", "b")))
    seed_artifact(cache, "a", "sha-a", ts=1, judge_pass=True)
    cache.rows.append(judge_row("a", "picture", "sha-a", 90.0, ts=2))
    seed_artifact(cache, "b", "sha-b", ts=3, judge_pass=True)
    entries = _queue(syllabus, cache, sources_for=no_sources)
    bucket3 = [e for e in entries if e.bucket == 3]
    assert [e.subject for e in bucket3] == ["b", "a"]  # b (rank 50) ranks below a (rank 90)


def test_good_subjects_are_excluded_from_the_queue(cache):
    syllabus = _one_word_syllabus()
    seed_judge_pass(cache, "rice", "a" * 64, rubric=R)
    seed_rating(cache, "rice", "a" * 64, "good")
    assert _queue(syllabus, cache) == []


def test_pending_subjects_are_excluded_from_the_queue(db):
    syllabus = _one_word_syllabus()
    _batch_marker_submitted(db, "b1", ["rice"], ["picture-for-word"])
    assert _queue(syllabus, db) == []


def test_directed_subjects_sort_first_within_their_bucket(cache):
    # "dog" < "rice" alphabetically, so flagging "rice" (not "dog") proves
    # the ordering comes from `directed`, not from the subject tie-break.
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice", "dog")))
    seed_card_flag(cache, "rice")
    entries = _queue(syllabus, cache)
    assert [e.subject for e in entries] == ["rice", "dog"]


def test_card_flag_directs_the_subject(cache):
    syllabus = _one_word_syllabus()
    seed_judge_pass(cache, "rice", "a" * 64, rubric=R)
    seed_card_flag(cache, "rice")
    entry = next(e for e in _queue(syllabus, cache, sources_for=no_sources) if e.subject == "rice")
    assert entry.directed is True


# --- challengers -------------------------------------------------------

def test_challenger_found_for_a_learner_accepted_picture(cache):
    syllabus = _one_word_syllabus()
    seed_rating(cache, "rice", "a" * 64, "acceptable")
    cache.rows.append(provide_row("rice", "picture", backend="imgfetch",
                                  items=[{"sha": "b" * 64}], ts=_next_ts()))
    cache.rows.append(judge_row("rice", "picture", "b" * 64, True, rubric=R, ts=_next_ts()))
    found = challengers(cache, syllabus, current_rubric={"picture-for-word": R}, prior=(),
                        provenance_source=_no_provenance)
    assert found == [("rice", "a" * 64, "b" * 64)]


def test_no_challenger_when_no_machine_candidate_outranks_the_accepted_pick(cache):
    syllabus = _one_word_syllabus()
    seed_rating(cache, "rice", "a" * 64, "acceptable")
    # the accepted artifact carries its own passing machine verdict too
    cache.rows.append(judge_row("rice", "picture", "a" * 64, True, rubric=R, ts=_next_ts()))
    cache.rows.append(provide_row("rice", "picture", backend="imgfetch",
                                  items=[{"sha": "b" * 64}], ts=_next_ts()))
    cache.rows.append(judge_row("rice", "picture", "b" * 64, False, rubric=R, ts=_next_ts()))
    found = challengers(cache, syllabus, current_rubric={"picture-for-word": R}, prior=(),
                        provenance_source=_no_provenance)
    assert found == []  # "b" fails judge fit -- no passing unrated candidate exists


# --- reasks ----------------------------------------------------------------

def test_reasks_flags_a_good_rated_card_with_enough_lapses(cache):
    syllabus = SimpleNamespace(words=[SimpleNamespace(id="rice")], pairs=[])
    seed_rating(cache, "rice", "a" * 64, "good")

    class _Study:
        def records(self, card_key):
            return [StudyRecord(card_key=card_key, compile_id="c1", ts=i, grade=1, time_ms=100)
                   for i in range(3)] if card_key == "rice::picture" else []

    found = reasks(cache, _Study(), syllabus, lapse_threshold=2,
                   card_keys_for=lambda s: [f"{s}::picture"])
    assert found == [("rice", "rice::picture")]


def test_reasks_yields_nothing_below_the_lapse_threshold(cache):
    syllabus = SimpleNamespace(words=[SimpleNamespace(id="rice")], pairs=[])
    seed_rating(cache, "rice", "a" * 64, "good")

    class _Study:
        def records(self, card_key):
            return [StudyRecord(card_key=card_key, compile_id="c1", ts=1, grade=4, time_ms=100)]

    found = reasks(cache, _Study(), syllabus, lapse_threshold=2,
                   card_keys_for=lambda s: [f"{s}::picture"])
    assert found == []


# --- confusion_weights ----------------------------------------------------

class _FakeStudyReader:
    """No real store: study_by_confusion (Syllabus) is what folds these
    rows by confusion; this fake only serves study_rows/records.
    """
    def __init__(self, rows):
        self._rows = list(rows)

    def records(self, card_key):
        return [r for r in self._rows if r.card_key == card_key]

    def study_rows(self):
        return list(self._rows)


@dataclass
class _Rec:
    card_key: str = "k"
    compile_id: str = "c"
    ts: int = 0
    grade: int = 3
    time_ms: int = 100


def _confusion_syllabus(confusion_id: str, pair_id: str) -> Syllabus:
    confusion = SoundConfusion(id=ConfusionId(confusion_id), dimension="tone",
                               sounds=("mid", "low"))
    mid_w = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    low_w = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    pair = MinimalPair.create(id=PairId(pair_id), confusion=confusion, members=(mid_w, low_w))
    return Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())


def test_confusion_weights_keeps_the_seed_with_no_study_history():
    syllabus = _confusion_syllabus("tone:mid-low", "p1")
    weights = confusion_weights({"tone:mid-low": 2.0}, syllabus, _FakeStudyReader([]))
    assert weights["tone:mid-low"] == 2.0


def test_confusion_weights_increases_with_lapse_rate():
    syllabus = _confusion_syllabus("tone:mid-low", "p1")
    records = [_Rec(card_key="p1::recognition", grade=1),
              _Rec(card_key="p1::recognition", grade=1),
              _Rec(card_key="p1::recognition", grade=4)]  # 2/3 lapses
    reader = _FakeStudyReader(records)
    weights = confusion_weights({"tone:mid-low": 1.0}, syllabus, reader)
    assert weights["tone:mid-low"] == pytest.approx(1.0 * (1 + 2 / 3))


# --- authority-driven current_best, preference, provenance prior -----------
#
# These exercise the real SyllabusDb (the `db` fixture) rather than
# FakeCache: they need genuine `assessments_of` ordering/newest-wins.

def _provide(db, subject, kind, backend, shas, ts=None):
    db.append(port="provide", backend=backend, key=f"{backend}:{subject}:{len(shas)}",
              subject=subject, question={"kind": kind, "params": {}},
              answer={"items": [{"sha": s} for s in shas]}, ts=ts)


_KIND_BY_ROLE = {"picture-for-word": "picture", "recording-for-word": "recording"}


def _verdict(db, subject, backend, role, sha, value, rubric="r"):
    db.append(port="assess", backend=backend, key=f"{backend}:{rubric}:{sha}:{role}",
              subject=subject, question={"role": role, "artifact_sha": sha, "rubric": rubric,
                                        "kind": _KIND_BY_ROLE[role]},
              answer={"value": value})


def test_mechanical_pass_ranks_a_recording(db):
    _provide(db, "w", "recording", "forvo", ["s1"])
    _verdict(db, "w", "mechanical", "recording-for-word", "s1", True, rubric=None)
    best = current_best(db, "w", "recording", current_rubric={}, prior=(),
                        provenance_source=_no_provenance)
    assert best.artifact_sha == "s1" and best.rank == 50.0 and best.source == "mechanical"


def test_mechanical_pass_ranks_a_recording_under_a_rubric_mapping_for_its_own_role(db):
    # Mechanical rows are written with rubric=None (attempts.py's
    # convention -- they check ground truth, not a judge prompt). A
    # current_rubric mapping that names recording-for-word (e.g. because
    # the judge rubric changed) must not stale a mechanical verdict on
    # that account.
    _provide(db, "w", "recording", "forvo", ["s1"])
    _verdict(db, "w", "mechanical", "recording-for-word", "s1", True, rubric=None)
    best = current_best(db, "w", "recording",
                        current_rubric={"recording-for-word": "some judge rubric"}, prior=(),
                        provenance_source=_no_provenance)
    assert best.artifact_sha == "s1" and best.rank == 50.0 and best.source == "mechanical"


def test_mechanical_never_ranks_a_picture(db):
    _provide(db, "w", "picture", "openverse", ["s1"])
    _verdict(db, "w", "mechanical", "picture-for-word", "s1", True, rubric=None)
    best = current_best(db, "w", "picture", current_rubric={}, prior=(),
                        provenance_source=_no_provenance)
    assert best.artifact_sha is None


def test_preference_orders_passing_pictures(db):
    _provide(db, "w", "picture", "openverse", ["a", "b", "c"])
    for s in "abc":
        _verdict(db, "w", "judge", "picture-for-word", s, True, rubric="fit")
    db.append(port="assess", backend="judge", key="judge:x:abc:picture-preference", subject="w",
              question={"role": "picture-preference", "artifact_sha": None, "rubric": "pref",
                        "kind": "picture", "params": {"candidates": ["a", "b", "c"]}},
              answer={"value": ["b", "c", "a"]})
    best = current_best(db, "w", "picture",
                        current_rubric={"picture-for-word": "fit", "picture-preference": "pref"},
                        prior=(), provenance_source=_no_provenance)
    assert best.artifact_sha == "b" and 50.0 < best.rank <= 70.0


def test_provenance_prior_breaks_ties_below_one_rank_point(db):
    """Models the real two-step write (spec 3 section 3): forvo's own
    Source ask carries no sha (a lookup only); the sha arrives on a
    SEPARATE audiofetch bytes row, and the real Source name ("forvo")
    lives only in the media table (db.add_media), never on that bytes
    row's own backend. attempts.provenance_source_for(db) is exactly the
    callable attempts.py/run.py/reviewserver.py wire in production --
    a provenance_source that read the CACHE row's backend instead would
    see "audiofetch", not "forvo", and this test would then pick "t" (tts,
    a one-step write whose own backend already says "tts") over "f".
    """
    from datetime import date

    from thai_syllabus.attempts import provenance_source_for

    db.append(port="provide", backend="forvo", key="forvo:w", subject="w",
             question={"kind": "recording", "params": {"word": "w"}}, answer={"items": []})
    db.append(port="provide", backend="audiofetch", key="https://forvo.example/f.mp3",
             subject="w",
             question={"kind": "recording", "params": {"url": "https://forvo.example/f.mp3"}},
             answer={"items": [{"sha": "f", "ext": "mp3"}]})
    db.add_media(sha="f", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.example/f.mp3", licence="cc-by",
                acquired=date(2026, 1, 1))

    db.append(port="provide", backend="tts", key="tts:w", subject="w",
             question={"kind": "recording", "params": {"text": "w"}},
             answer={"items": [{"sha": "t", "ext": "mp3", "voice": "v1"}]})
    db.add_media(sha="t", kind="recording", ext="mp3", source="tts", origin="v1",
                licence="google-tts", acquired=date(2026, 1, 1))

    for s in ("t", "f"):
        _verdict(db, "w", "mechanical", "recording-for-word", s, True, rubric=None)
    best = current_best(db, "w", "recording", current_rubric={},
                        prior=("commission", "forvo", "tts"),
                        provenance_source=provenance_source_for(db))
    assert best.artifact_sha == "f" and 50.0 < best.rank < 51.0


def test_role_scoped_rubric_mapping_marks_only_that_role_stale(db):
    _provide(db, "w", "picture", "openverse", ["a"])
    _verdict(db, "w", "judge", "picture-for-word", "a", True, rubric="old")
    assert current_best(db, "w", "picture", current_rubric={"picture-for-word": "new"},
                        prior=(), provenance_source=_no_provenance).artifact_sha is None
    assert current_best(db, "w", "picture", current_rubric={"sentence-for-target": "x"},
                        prior=(), provenance_source=_no_provenance).artifact_sha == "a"


# --- adoptable_drafts: what the run adopts a cover of -----------------------

_DRAFT_JSON = ('{"sentences": [{"text": "\u0e01\u0e34\u0e19", "gloss": "eat", '
               '"targets": ["eat/receptive"]}]}')          # กิน: eat
_DRAFT_SHA = text_sha("กิน")                # กิน: eat


def _draft_syllabus(sentences=()):
    return Syllabus(words=(word("eat", "กิน", "eat"),),   # กิน: eat
                    targets=(target("eat/receptive", "eat"),),
                    sentences=tuple(sentences), tokenizer=FakeTokenizer())


def _fills_row(value=True, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend="fills", key=f"fills:{ts}", key_sha="x",
                 subject=_DRAFT_SHA,
                 question={"role": "sentence-for-target", "artifact_sha": None, "rubric": None,
                          "kind": "sentence", "subject_kind": "sentence",
                          "params": {"target": "eat/receptive"}},
                 answer={"value": value}, cost=0.0, ts=ts)


def _sentence_verdict(backend, value, rubric=None, ts=None):
    ts = ts if ts is not None else _next_ts()
    return Answer(port="assess", backend=backend, key=f"{backend}:{ts}", key_sha="x",
                 subject=_DRAFT_SHA,
                 question={"role": "sentence-for-target", "artifact_sha": None,
                          "rubric": rubric, "kind": "sentence", "subject_kind": "sentence",
                          "params": {}},
                 answer={"value": value}, cost=0.0, ts=ts)


def _drafted(cache):
    cache.rows.append(provide_row("sentence-drafts", "sentence", backend="llm-sentence",
                                  items=[]))
    cache.rows[-1].answer["items"] = [_DRAFT_JSON]
    return cache


def test_adoptable_drafts_offers_a_draft_that_fills_and_the_judge_passed(cache):
    _drafted(cache)
    cache.rows += [_fills_row(), _sentence_verdict("judge", True, rubric="R")]
    adoptable = adoptable_drafts(cache, _draft_syllabus(), current_rubric={"sentence-for-target": "R"})
    assert [(s.text, tuple(t.id for t in ts)) for s, ts in adoptable] == [
        ("กิน", ("eat/receptive",))]        # กิน: eat
    assert adoptable[0][0].gloss == "eat"


def test_an_adopted_draft_carries_the_drafting_model_and_the_runs_clock(cache):
    from datetime import date as _date
    _drafted(cache)
    cache.rows += [_fills_row(), _sentence_verdict("judge", True, rubric="R")]
    adopted, _targets = adoptable_drafts(cache, _draft_syllabus(),
                                         current_rubric={"sentence-for-target": "R"},
                                         model="claude-x", today=lambda: _date(2026, 9, 5))[0]
    assert adopted.provenance.origin == "claude-x"
    assert adopted.provenance.acquired == _date(2026, 9, 5)


def test_a_target_confirmed_by_two_fills_rows_is_offered_once(cache):
    """A draft re-verified on a later run has a second fills row for the
    same target; the cover must not see it twice."""
    _drafted(cache)
    cache.rows += [_fills_row(), _fills_row(), _sentence_verdict("judge", True, rubric="R")]
    _sentence, targets = adoptable_drafts(cache, _draft_syllabus(),
                                          current_rubric={"sentence-for-target": "R"})[0]
    assert [t.id for t in targets] == ["eat/receptive"]


def test_a_draft_the_judge_failed_is_not_adoptable(cache):
    _drafted(cache)
    cache.rows += [_fills_row(), _sentence_verdict("judge", False, rubric="R")]
    assert adoptable_drafts(cache, _draft_syllabus(),
                            current_rubric={"sentence-for-target": "R"}) == []


def test_a_learner_rating_outranks_the_judge_on_a_draft(cache):
    """Authority order for sentence-for-target is learner > judge: a draft
    the learner called acceptable is adoptable however the judge voted."""
    _drafted(cache)
    cache.rows += [_fills_row(), _sentence_verdict("judge", False, rubric="R"),
                   _sentence_verdict("learner", "acceptable")]
    assert len(adoptable_drafts(cache, _draft_syllabus(),
                                current_rubric={"sentence-for-target": "R"})) == 1


def test_a_learner_rejection_outranks_a_judge_pass_on_a_draft(cache):
    _drafted(cache)
    cache.rows += [_fills_row(), _sentence_verdict("judge", True, rubric="R"),
                   _sentence_verdict("learner", "unacceptable-none")]
    assert adoptable_drafts(cache, _draft_syllabus(),
                            current_rubric={"sentence-for-target": "R"}) == []


def test_a_draft_that_fills_nothing_is_not_adoptable(cache):
    _drafted(cache)
    cache.rows += [_fills_row(value=False), _sentence_verdict("judge", True, rubric="R")]
    assert adoptable_drafts(cache, _draft_syllabus(),
                            current_rubric={"sentence-for-target": "R"}) == []


def test_a_draft_already_adopted_is_not_offered_again(cache):
    _drafted(cache)
    cache.rows += [_fills_row(), _sentence_verdict("judge", True, rubric="R")]
    adopted = sentence("กิน", gloss="eat")   # กิน: eat
    assert adoptable_drafts(cache, _draft_syllabus([adopted]),
                            current_rubric={"sentence-for-target": "R"}) == []


def test_a_stale_judge_verdict_does_not_make_a_draft_adoptable(cache):
    _drafted(cache)
    cache.rows += [_fills_row(), _sentence_verdict("judge", True, rubric="old-R")]
    assert adoptable_drafts(cache, _draft_syllabus(),
                            current_rubric={"sentence-for-target": "R"}) == []
