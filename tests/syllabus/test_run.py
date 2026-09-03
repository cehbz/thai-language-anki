"""Tests for run.py (spec 3 sections 4/7): a pending-aware loop over
derivations.queue()'s entries, escalating attempts.SOURCES per kind via
attempts.attempt(), plus one per-run attempts.sentence_attempt() pass --
against Sourcing fakes (no real Provider/Assessor/network).
"""
import pytest

from thai_syllabus.attempts import Outcome, SentenceOutcome, Sourcing
from thai_syllabus.run import (
    FORVO_DEFAULT_DAILY_BUDGET,
    LEARNER_DEFAULT_SESSION_BUDGET,
    Budget,
    Spend,
    run,
)
from thai_syllabus import run as run_mod
from thai_syllabus.store import SyllabusDb


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
        self._gaps, self.targets = gaps, []

    def gaps(self):
        return self._gaps

    def with_sentences(self, new):
        # a real Syllabus.with_sentences returns a new immutable value;
        # this fake's gaps() ignores sentences entirely, so returning self
        # is behaviourally equivalent for these tests.
        return self


def _ctx(db, syl):
    return Sourcing(syllabus=syl, provider=None, assessor=None, db=db, media_store=None,
                    rubrics={}, provenance_prior=())


def _patch(monkeypatch, outcomes, sentence=SentenceOutcome(0, (), False, {})):
    calls = []

    def fake_attempt(ctx, need, source):
        calls.append((need, source))
        return outcomes.get((need.subject, source), Outcome(True, False, False, {source: (1, 0.0)}))
    monkeypatch.setattr(run_mod, "attempt", fake_attempt)
    monkeypatch.setattr(run_mod, "sentence_attempt", lambda ctx, max_targets=40: sentence)
    return calls


# --- escalation over SOURCES ----------------------------------------------

def test_run_escalates_sources_until_improved(db, monkeypatch):
    calls = _patch(monkeypatch, {("w", "wikimedia"): Outcome(True, False, True, {"wikimedia": (1, 0.0)})})
    r = run(_ctx(db, _Syl(_Gaps(pictures=("w",)))), {})
    assert [s for _, s in calls] == ["openverse", "wikimedia"]
    assert r.attempted == 1 and r.improved == 1 and r.pending == 0


def test_run_stops_a_pending_need_without_escalating(db, monkeypatch):
    calls = _patch(monkeypatch, {("w", "openverse"): Outcome(True, True, False, {})})
    r = run(_ctx(db, _Syl(_Gaps(pictures=("w",)))), {})
    assert [s for _, s in calls] == ["openverse"] and r.pending == 1


def test_run_skips_a_source_whose_budget_is_spent(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(recordings=("a", "b")))), {"forvo": Budget(max_asks=1)})
    forvo_calls = [n for n, s in calls if s == "forvo"]
    assert len(forvo_calls) == 1 and len([1 for _, s in calls if s == "tts"]) == 2


def test_run_counts_adopted_sentences_and_persists_pending(db, monkeypatch):
    from .builders import sentence
    _patch(monkeypatch, {}, sentence=SentenceOutcome(2, (sentence("x"),), True, {"llm-sentence": (1, 0.0)}))
    r = run(_ctx(db, _Syl(_Gaps())), {})
    assert r.sentences_adopted == 1 and r.pending == 1
    row = db.latest("run", "runreport", "runreport")
    assert row.answer["pending"] == 1 and row.answer["sentences_adopted"] == 1


def test_run_never_attempts_sentence_needs_per_subject(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert calls == []


def test_run_counts_a_kind_with_no_registered_sources_as_available_not_attempted(db, monkeypatch):
    # SOURCES has no entry for "grapheme-keyword" -- sources_for() returns
    # () and the need is never attempted, but it was still queued.
    calls = _patch(monkeypatch, {})
    r = run(_ctx(db, _Syl(_Gaps(graphemes=("g1",)))), {})
    assert calls == []
    assert r.attempted == 0 and r.available == 1


# --- apply sentence adoptions before computing the queue -------------------

class _AdoptingSyl:
    """Stands in for `Syllabus.with_sentences`'s real effect (shrinking
    `gaps().unfilled_targets`) without any real fills()/Syllabus machinery
    -- `with_sentences` here just drops whatever target ids its adopted
    items name (via a bare `.target_id`, not a real Sentence)."""
    def __init__(self, unfilled_targets):
        self.targets = []
        self._unfilled = tuple(unfilled_targets)

    def gaps(self):
        return _Gaps(sentences=self._unfilled)

    def with_sentences(self, new):
        filled = {getattr(s, "target_id", s) for s in new}
        return _AdoptingSyl(t for t in self._unfilled if t not in filled)


class _AdoptedMarker:
    def __init__(self, target_id):
        self.target_id = target_id


def test_run_applies_sentence_adoptions_before_computing_the_queue(db, monkeypatch):
    captured = []

    def fake_queue(syllabus, cache, **kwargs):
        captured.append(syllabus)
        # a queue entry per still-open target, standing in for whatever
        # non-sentence need derivations.queue would actually produce --
        # what matters here is whether "t1" (already adopted) is still
        # among them.
        from thai_syllabus.derivations import QueueEntry
        return [QueueEntry(subject=t, kind="picture", bucket=1)
               for t in syllabus.gaps().unfilled_targets]

    monkeypatch.setattr(run_mod, "queue", fake_queue)
    calls = _patch(monkeypatch, {("t2", "openverse"): Outcome(True, False, True, {"openverse": (1, 0.0)})},
                  sentence=SentenceOutcome(1, (_AdoptedMarker("t1"),), False, {}))

    run(_ctx(db, _AdoptingSyl(("t1", "t2"))), {})

    assert captured[0].gaps().unfilled_targets == ("t2",)  # t1 already dropped
    assert [n.subject for n, _ in calls] == ["t2"]  # never attempted for t1


# --- documented defaults (spec 3 section 4) -------------------------------

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

    def new_ctx():
        return _ctx(db, _Syl(_Gaps()))

    run(new_ctx(), {})
    run(new_ctx(), {})
    rows = [r for r in db.assessments_of("run") if r.port == "run"]
    assert len(rows) == 2
    assert rows[0].ts != rows[1].ts


# --- Spend ------------------------------------------------------------

def test_spend_add_increments_by_given_asks():
    s = Spend()
    s.add(3, 0.6)
    assert s.asks == 3 and s.cost == pytest.approx(0.6)


def test_spend_exceeds_checks_asks_and_cost():
    assert Spend(asks=5).exceeds(Budget(max_asks=5))
    assert not Spend(asks=4).exceeds(Budget(max_asks=5))
    assert Spend(cost=1.0).exceeds(Budget(max_cost=1.0))
    assert not Spend(cost=0.5).exceeds(Budget(max_cost=1.0))
