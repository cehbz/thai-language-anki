"""Tests for run.py (spec 3 section 7): one sentence attempt over the open
Targets, adoption of what the judge passed, then one Source per queued need
-- against Sourcing fakes (no real Provider/Assessor/network).
"""
import pytest

from thai_syllabus.assessor import JudgeUnreachable
from thai_syllabus.attempts import AttemptResult, Sourcing, Spend
from thai_syllabus.run import (
    FORVO_DEFAULT_DAILY_BUDGET,
    LEARNER_DEFAULT_SESSION_BUDGET,
    Budget,
    run,
)
from thai_syllabus import run as run_mod
from thai_syllabus.store import SyllabusDb

from .builders import sentence


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _Gaps:
    def __init__(self, pictures=(), recordings=(), sentences=(), graphemes=()):
        self.words_missing_pictures, self.words_missing_recordings = pictures, recordings
        self.unfilled_targets, self.missing_renditions = sentences, ()
        self.graphemes_missing_keyword_data = graphemes


class _Syl:
    def __init__(self, gaps):
        self._gaps, self.targets, self.sentences = gaps, [], ()

    def gaps(self):
        return self._gaps

    def cover(self, drafts):
        return []

    def with_sentences(self, new):
        # the real Syllabus.with_sentences returns a new immutable value;
        # this fake's gaps() ignores sentences entirely.
        return self


def _ctx(db, syl):
    return Sourcing(syllabus=syl, provider=None, assessor=None, db=db, media_store=None,
                    rubrics={}, provenance_prior=())


def _patch(monkeypatch, results, sentence_result=AttemptResult(attempted=False), drafts=()):
    """Replaces attempt/sentence_attempt/adoptable_drafts. A `results` entry
    that is an exception class is raised instead of returned."""
    calls = []

    def fake_attempt(ctx, need, source):
        calls.append((need, source))
        result = results.get((need.subject, source),
                             AttemptResult(True, spend={source: Spend(1, 0.0)}))
        if isinstance(result, type) and issubclass(result, Exception):
            raise result("no judge")
        return result

    def fake_sentence_attempt(ctx, max_targets=40):
        if isinstance(sentence_result, type) and issubclass(sentence_result, Exception):
            raise sentence_result("no judge")
        return sentence_result

    monkeypatch.setattr(run_mod, "attempt", fake_attempt)
    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "adoptable_drafts",
                        lambda cache, syllabus, **kwargs: list(drafts))
    return calls


# --- one source per need per run (spec 3 section 7) ------------------------

def test_run_asks_one_source_per_need_and_leaves_escalation_to_the_next_run(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("w",)))), {})
    assert [s for _need, s in calls] == ["openverse"]
    assert report.attempted == 1 and report.pending == 0


def test_run_counts_a_need_whose_questions_went_unanswered_as_pending(db, monkeypatch):
    _patch(monkeypatch, {("w", "openverse"): AttemptResult(True, questions=["q"])})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("w",)))), {})
    assert report.pending == 1


def test_run_skips_a_need_whose_source_budget_is_spent(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(recordings=("a", "b")))), {"forvo": Budget(max_asks=1)})
    assert [(n.subject, s) for n, s in calls] == [("a", "forvo")]


def test_run_never_attempts_sentence_needs_per_subject(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert calls == []


def test_run_counts_a_kind_with_no_registered_sources_as_neither_attempted_nor_available(
        db, monkeypatch):
    # SOURCES has no entry for "grapheme-keyword", so derivations.exhausted
    # excludes such a subject from the queue outright.
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(graphemes=("g1",)))), {})
    assert calls == []
    assert report.attempted == 0 and report.available == 0


# --- adoption: the cover over what the judge passed ------------------------

class _AdoptingSyl:
    """Stands in for the real Syllabus's adoption effect: cover() takes the
    verified drafts whose Targets are still open, and with_sentences()
    closes those Targets."""
    def __init__(self, unfilled_targets):
        self.targets, self.sentences = [], ()
        self._unfilled = tuple(unfilled_targets)
        self._covered: tuple[str, ...] = ()

    def gaps(self):
        return _Gaps(sentences=self._unfilled)

    def cover(self, drafts):
        chosen = [(s, ts) for s, ts in drafts if any(t in self._unfilled for t in ts)]
        self._covered = tuple(t for _s, ts in chosen for t in ts)
        return chosen

    def with_sentences(self, new):
        return _AdoptingSyl(t for t in self._unfilled if t not in self._covered)


def test_run_adopts_a_cover_of_the_adoptable_drafts_before_computing_the_queue(db, monkeypatch):
    captured = []

    def fake_queue(syllabus, cache, **kwargs):
        captured.append(syllabus)
        from thai_syllabus.derivations import QueueEntry
        return [QueueEntry(subject=t, kind="picture", bucket=1)
                for t in syllabus.gaps().unfilled_targets]

    monkeypatch.setattr(run_mod, "queue", fake_queue)
    calls = _patch(monkeypatch, {}, drafts=[(sentence("x"), ("t1",))])

    report = run(_ctx(db, _AdoptingSyl(("t1", "t2"))), {})

    assert report.sentences_adopted == 1
    assert [s.text for s in db.all_sentences()] == ["x"]
    assert captured[0].gaps().unfilled_targets == ("t2",)   # t1 already covered
    assert [n.subject for n, _s in calls] == ["t2"]


# --- documented defaults (spec 3 section 7) -------------------------------

def test_forvo_default_daily_budget_is_450_asks():
    assert FORVO_DEFAULT_DAILY_BUDGET.max_asks == 450


def test_learner_default_session_budget_is_20_questions():
    assert LEARNER_DEFAULT_SESSION_BUDGET.max_asks == 20


# --- RunReport persistence: /stats history needs a source ----------------

def test_a_run_that_does_almost_nothing_still_appends_a_row(db, monkeypatch):
    _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps())), {})
    rows = [r for r in db.assessments_of("run") if r.port == "run"]
    assert len(rows) == 1


def test_two_runs_each_get_their_own_keyed_row(db, monkeypatch):
    _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps())), {})
    run(_ctx(db, _Syl(_Gaps())), {})
    rows = [r for r in db.assessments_of("run") if r.port == "run"]
    assert len(rows) == 2
    assert rows[0].ts != rows[1].ts


# --- Spend ------------------------------------------------------------

def test_spend_add_increments_by_given_asks():
    s = Spend()
    s.add(3, 0.6)
    assert s.asks == 3 and s.cost == pytest.approx(0.6)


def test_a_budget_says_when_a_spend_has_reached_it():
    assert Budget(max_asks=5).exceeded_by(Spend(asks=5))
    assert not Budget(max_asks=5).exceeded_by(Spend(asks=4))
    assert Budget(max_cost=1.0).exceeded_by(Spend(cost=1.0))
    assert not Budget(max_cost=1.0).exceeded_by(Spend(cost=0.5))


# --- the report counts what went wrong ------------------------------------

def test_run_stops_at_the_first_unreachable_judge(db, monkeypatch):
    """An unreachable judge is a dead wire, not a per-need failure: every
    remaining need would fail the same way. The run stops and says so
    instead of grinding through the whole queue in silence."""
    calls = _patch(monkeypatch, {("a", "openverse"): JudgeUnreachable})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert [n.subject for n, _s in calls] == ["a"]   # no second need
    assert report.unreachable is True
    assert report.available == 2                     # both needs still to do
    assert db.latest("run", "runreport", "runreport").answer["unreachable"] is True


def test_an_unreachable_sentence_attempt_stops_the_run_too(db, monkeypatch):
    calls = _patch(monkeypatch, {}, sentence_result=JudgeUnreachable)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert calls == []                               # no need attempted at all
    assert report.unreachable is True


def test_run_sums_excluded_candidates_across_attempts(db, monkeypatch):
    _patch(monkeypatch, {
        ("a", "openverse"): AttemptResult(True, excluded={"k1": "gone"}),
        ("b", "openverse"): AttemptResult(True, excluded={"k2": "gone", "k3": "gone"})},
        sentence_result=AttemptResult(False, excluded={"k4": "gone"}))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert report.excluded == 4
    assert db.latest("run", "runreport", "runreport").answer["excluded"] == 4


def test_a_run_with_nothing_wrong_reports_zero_excluded_and_reachable(db, monkeypatch):
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert report.excluded == 0 and report.unreachable is False


def test_available_excludes_the_sentence_entries_the_loop_skips(db, monkeypatch):
    """Sentence needs are handled once per run by sentence_attempt, so the
    per-need loop skips them -- counting them as "available" would report
    work still to do that this run had already done."""
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1", "t2"), graphemes=("g1",)))), {})
    assert calls == []
    assert report.available == 0
