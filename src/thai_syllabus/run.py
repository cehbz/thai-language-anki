"""The batch run (spec 3 section 4): Budget + run(syllabus, budgets), an
application service -- iteration only, every policy is a derivation it
calls (derivations.py). Kill-safe: every ask() already appends as a
checkpoint (spec 2), so the loop itself holds no state that would be lost
between iterations; a run stopped at any point leaves the cache exactly
as far along as it got.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .derivations import current_best, exhausted, queue
from .ports import CacheReader, RecordWriter
from .transport import TransportError

__all__ = ["Budget", "Spend", "Lever", "RunReport", "run"]


@dataclass(frozen=True)
class Budget:
    """Per-backend spend cap for one run, in that backend's own currency
    (spec 3 section 4): asks (forvo: 450/day default), dollars (judge
    api/batch), subscription quota (judge cli), learner attention (session
    default 20 questions, ~25 min -- pull-based: this caps what queue()
    hands the learner, never what is owed). Either field may be set; a
    backend with neither is unbounded.
    """
    max_asks: int | None = None
    max_cost: float | None = None


FORVO_DEFAULT_DAILY_BUDGET = Budget(max_asks=450)
LEARNER_DEFAULT_SESSION_BUDGET = Budget(max_asks=20)


@dataclass
class Spend:
    asks: int = 0
    cost: float = 0.0

    def record(self, cost: float) -> None:
        self.asks += 1
        self.cost += cost

    def exceeds(self, budget: Budget) -> bool:
        if budget.max_asks is not None and self.asks >= budget.max_asks:
            return True
        if budget.max_cost is not None and self.cost >= budget.max_cost:
            return True
        return False


@dataclass(frozen=True)
class Lever:
    """One escalation step for a `kind`'s need: an ask against a Provider
    or Assessor backend. `ask` is a bound `Provider.ask` / `Assessor.ask`
    (or any callable(backend, question) -> object with a `.cost`
    attribute); `build_question` turns (subject, kind) into that ask's
    Question/AssessQuestion. Levers for one kind are supplied cheapest-
    first by the caller -- run() does not re-order them (spec 3 section 4:
    "escalate backends cheapest-first").
    """
    backend: str
    ask: Callable[[str, Any], Any]
    build_question: Callable[[str, str], Any]


@dataclass
class RunReport:
    """"a run that did almost nothing must look like one" (spec 3 section
    4): attempted/improved/exhausted/available are subject counts (not ask
    counts), so a run that touched nothing shows zeros everywhere except
    `available`.
    """
    attempted: int = 0
    improved: int = 0
    exhausted: int = 0
    available: int = 0
    spend: dict[str, Spend] = field(default_factory=dict)


def run(syllabus, cache: CacheReader, budgets: Mapping[str, Budget],
       levers_by_kind: Mapping[str, Sequence[Lever]], *,
       current_rubric: str | None = None) -> RunReport:
    """The application service (spec 3 section 4): for every subject
    queue() orders, escalate its kind's levers cheapest-first; stop the
    subject when current_best improves or its levers/budgets run out.
    Iteration only -- current_best/exhausted/queue are the derivations
    that decide everything; this function calls them and nothing else.
    """
    report = RunReport()
    spend: dict[str, Spend] = {name: Spend() for name in budgets}
    report.spend = spend

    entries = queue(syllabus, cache, budgets=budgets, current_rubric=current_rubric)
    for entry in entries:
        before = current_best(cache, entry.subject, entry.kind,
                              current_rubric=current_rubric)
        rank = before.rank
        attempted_this_subject = False
        improved_this_subject = False

        for lever in levers_by_kind.get(entry.kind, ()):
            budget = budgets.get(lever.backend)
            s = spend.setdefault(lever.backend, Spend())
            if budget is not None and s.exceeds(budget):
                continue  # this backend's budget is spent -- try the next lever

            question = lever.build_question(entry.subject, entry.kind)
            try:
                answer = lever.ask(lever.backend, question)
            except TransportError:
                continue  # miss NOT cached (spec 3 section 7) -- try the next lever

            attempted_this_subject = True
            s.record(getattr(answer, "cost", 0.0))

            after = current_best(cache, entry.subject, entry.kind,
                                 current_rubric=current_rubric)
            if after.rank > rank:
                improved_this_subject = True
                break  # stop the subject when current_best improves

        if attempted_this_subject:
            report.attempted += 1
        if improved_this_subject:
            report.improved += 1
        status = exhausted(cache, entry.subject, entry.kind, current_rubric=current_rubric)
        if status.exhausted:
            report.exhausted += 1

    report.available = len(entries) - report.attempted
    _persist_report(cache, report)
    return report


def _persist_report(cache: CacheReader, report: RunReport) -> None:
    """Appends one summary row per run() call (port="run",
    backend="runreport") so a run's own outcome has a durable source --
    RunReport itself is only ever an in-memory return value, and nothing
    else in spec 3 writes one to the cache (reviewserver.py's /stats reads
    cache rows only). `key` is a constant, readable label ("runreport");
    the `cache` table's primary key is (key_sha, ts) (store.py), so every
    call still lands its own row -- "keyed on timestamp", not deduplicated
    by key. Silently a no-op when `cache` is read-only (doesn't also
    satisfy RecordWriter): run() itself only ever needs cache read access,
    so this is best-effort persistence, not a hard requirement of the
    application service's contract.
    """
    if not isinstance(cache, RecordWriter):
        return
    cache.append(
        port="run", backend="runreport", key="runreport", subject="run",
        question={},
        answer={"attempted": report.attempted, "improved": report.improved,
                "exhausted": report.exhausted, "available": report.available,
                "spend": {name: {"asks": s.asks, "cost": s.cost}
                         for name, s in report.spend.items()}},
        cost=sum(s.cost for s in report.spend.values()))
