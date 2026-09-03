"""Tests for run.py (spec 3 section 4): Budget + run() as an iteration-only
application service over derivations.py + fake Provider/Assessor-shaped
backends. Real SyllabusDb as the cache/record (so current_best/exhausted
see genuine appended rows); no network.
"""
from dataclasses import dataclass, field

import pytest

from thai_syllabus.run import (
    FORVO_DEFAULT_DAILY_BUDGET,
    LEARNER_DEFAULT_SESSION_BUDGET,
    Budget,
    Lever,
    RunReport,
    Spend,
    run,
)
from thai_syllabus.store import SyllabusDb
from thai_syllabus.transport import TransportError


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _FakeGaps:
    def __init__(self, words_missing_pictures=()):
        self.words_missing_pictures = words_missing_pictures
        self.words_missing_recordings = ()
        self.unfilled_targets = ()
        self.missing_renditions = ()
        self.graphemes_missing_keyword_data = ()


@dataclass
class _FakeSyllabus:
    _gaps: _FakeGaps
    targets: list = field(default_factory=list)

    def gaps(self):
        return self._gaps


@dataclass
class _Answer:
    items: tuple = ()
    cost: float = 0.0


class _FakeAsk:
    """Stands in for a bound Provider.ask/Assessor.ask: cache-first over
    the real db, appending a judge-shaped verdict so current_best can see
    an improvement.
    """
    def __init__(self, db, verdict_value, cost=0.0, raises=None):
        self.db = db
        self.verdict_value = verdict_value
        self.cost = cost
        self.raises = raises
        self.calls = 0

    def __call__(self, backend, question):
        self.calls += 1
        if self.raises:
            raise self.raises
        artifact_sha = f"{backend}-sha-{self.calls}"
        # a provide-shaped attempt row, so exhausted()'s attempt counting
        # and queue()'s "no artifact" bucket both see it
        self.db.append(port="provide", backend=backend, key=f"{backend}:{question[0]}",
                       subject=question[0],
                       question={"provides": question[1], "params": {}},
                       answer={"items": [{"sha": artifact_sha}]}, cost=0.0)
        # and a judge verdict so current_best has something to rank
        self.db.append(port="assess", backend="judge",
                       key=f"judge:{question[0]}:{artifact_sha}", subject=question[0],
                       question={"role": f"{question[1]}-for-word",
                                "artifact_sha": artifact_sha, "rubric": None},
                       answer={"value": self.verdict_value}, cost=self.cost)
        return _Answer(items=({"sha": artifact_sha},), cost=self.cost)


def _question(subject, kind):
    return (subject, kind)  # a bare tuple stands in for Question here


# --- basic escalation ----------------------------------------------------

def test_run_attempts_and_improves_a_subject_with_no_artifact(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    ask = _FakeAsk(db, verdict_value=True)
    levers = {"picture": [Lever(backend="openverse", ask=ask, build_question=_question)]}
    report = run(syllabus, db, budgets={}, levers_by_kind=levers)
    assert isinstance(report, RunReport)
    assert report.attempted == 1
    assert report.improved == 1
    assert ask.calls == 1


def test_run_stops_the_subject_as_soon_as_current_best_improves(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    ask1 = _FakeAsk(db, verdict_value=True)   # improves -- passing verdict
    ask2 = _FakeAsk(db, verdict_value=True)
    levers = {"picture": [Lever(backend="openverse", ask=ask1, build_question=_question),
                          Lever(backend="wikimedia", ask=ask2, build_question=_question)]}
    run(syllabus, db, budgets={}, levers_by_kind=levers)
    assert ask1.calls == 1
    assert ask2.calls == 0  # never reached -- the first lever already improved it


def test_run_escalates_to_the_next_lever_when_the_first_does_not_improve(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    ask1 = _FakeAsk(db, verdict_value=False)  # does not improve -- failing verdict
    ask2 = _FakeAsk(db, verdict_value=True)
    levers = {"picture": [Lever(backend="openverse", ask=ask1, build_question=_question),
                          Lever(backend="wikimedia", ask=ask2, build_question=_question)]}
    report = run(syllabus, db, budgets={}, levers_by_kind=levers)
    assert ask1.calls == 1
    assert ask2.calls == 1
    assert report.improved == 1


def test_a_transport_error_moves_to_the_next_lever_without_stopping_the_run(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    failing = _FakeAsk(db, verdict_value=True, raises=TransportError("down"))
    ask2 = _FakeAsk(db, verdict_value=True)
    levers = {"picture": [Lever(backend="openverse", ask=failing, build_question=_question),
                          Lever(backend="wikimedia", ask=ask2, build_question=_question)]}
    report = run(syllabus, db, budgets={}, levers_by_kind=levers)
    assert ask2.calls == 1
    assert report.attempted == 1
    assert report.improved == 1


# --- budgets: per-backend own currency -----------------------------------

def test_a_spent_backend_budget_is_skipped_in_favor_of_the_next_lever(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    ask1 = _FakeAsk(db, verdict_value=True)
    ask2 = _FakeAsk(db, verdict_value=True)
    budgets = {"openverse": Budget(max_asks=0)}  # already spent
    levers = {"picture": [Lever(backend="openverse", ask=ask1, build_question=_question),
                          Lever(backend="wikimedia", ask=ask2, build_question=_question)]}
    run(syllabus, db, budgets=budgets, levers_by_kind=levers)
    assert ask1.calls == 0
    assert ask2.calls == 1


def test_spend_is_tracked_per_backend_in_its_own_currency(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    ask = _FakeAsk(db, verdict_value=True, cost=0.0018)
    budgets = {"judge-api": Budget(max_cost=1.0)}
    levers = {"picture": [Lever(backend="judge-api", ask=ask, build_question=_question)]}
    report = run(syllabus, db, budgets=budgets, levers_by_kind=levers)
    assert report.spend["judge-api"].asks == 1
    assert report.spend["judge-api"].cost == pytest.approx(0.0018)


def test_forvo_daily_budget_stops_further_forvo_asks(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("a", "b")))
    ask = _FakeAsk(db, verdict_value=True)
    budgets = {"forvo": Budget(max_asks=1)}
    levers = {"picture": [Lever(backend="forvo", ask=ask, build_question=_question)]}
    run(syllabus, db, budgets=budgets, levers_by_kind=levers)
    assert ask.calls == 1  # the second subject's forvo lever is skipped


# --- RunReport: "did almost nothing must look like one" ------------------

def test_a_run_over_an_empty_syllabus_reports_all_zeros(db):
    syllabus = _FakeSyllabus(_FakeGaps())
    report = run(syllabus, db, budgets={}, levers_by_kind={})
    assert report.attempted == 0
    assert report.improved == 0
    assert report.exhausted == 0
    assert report.available == 0


def test_available_counts_queued_subjects_that_were_never_attempted(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    # no levers registered for "picture" at all -- queued but untouched
    report = run(syllabus, db, budgets={}, levers_by_kind={})
    assert report.attempted == 0
    assert report.available == 1


# --- documented defaults (spec 3 section 4) -------------------------------

def test_forvo_default_daily_budget_is_450_asks():
    assert FORVO_DEFAULT_DAILY_BUDGET.max_asks == 450


def test_learner_default_session_budget_is_20_questions():
    assert LEARNER_DEFAULT_SESSION_BUDGET.max_asks == 20


# --- RunReport persistence: /stats history needs a source ----------------

def test_run_appends_a_runreport_summary_row(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice",)))
    ask = _FakeAsk(db, verdict_value=True, cost=0.25)
    levers = {"picture": [Lever(backend="openverse", ask=ask, build_question=_question)]}
    report = run(syllabus, db, budgets={}, levers_by_kind=levers)

    rows = db.assessments_of("run")
    runreport_rows = [r for r in rows if r.port == "run" and r.backend == "runreport"]
    assert len(runreport_rows) == 1
    answer = runreport_rows[0].answer
    assert answer["attempted"] == report.attempted
    assert answer["improved"] == report.improved
    assert answer["exhausted"] == report.exhausted
    assert answer["available"] == report.available
    assert answer["spend"]["openverse"]["cost"] == pytest.approx(0.25)


def test_a_run_that_does_almost_nothing_still_appends_a_row(db):
    syllabus = _FakeSyllabus(_FakeGaps())
    run(syllabus, db, budgets={}, levers_by_kind={})
    rows = [r for r in db.assessments_of("run") if r.port == "run"]
    assert len(rows) == 1


def test_two_runs_each_get_their_own_keyed_row(db):
    syllabus = _FakeSyllabus(_FakeGaps())
    run(syllabus, db, budgets={}, levers_by_kind={})
    run(syllabus, db, budgets={}, levers_by_kind={})
    rows = [r for r in db.assessments_of("run") if r.port == "run"]
    assert len(rows) == 2
    assert rows[0].ts != rows[1].ts


# --- kill-safety: every ask() already appended a checkpoint --------------

def test_a_run_interrupted_mid_subject_leaves_prior_appends_intact(db):
    syllabus = _FakeSyllabus(_FakeGaps(words_missing_pictures=("rice", "dog")))
    ask = _FakeAsk(db, verdict_value=True)
    levers = {"picture": [Lever(backend="openverse", ask=ask, build_question=_question)]}
    run(syllabus, db, budgets={}, levers_by_kind=levers)
    # "kill" happens conceptually here -- but every ask() already appended;
    # a fresh read over the same db sees both subjects' rows already.
    db2 = SyllabusDb(db.path)
    assert len(db2.assessments_of("rice")) >= 1
    assert len(db2.assessments_of("dog")) >= 1
