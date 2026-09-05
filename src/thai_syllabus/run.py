"""The batch run (spec 3 section 4/7): a pending-aware loop over
derivations.queue()'s entries -- one attempts.sentence_attempt() pass per
run, then, for every other queued need, attempts.attempt() escalating
attempts.SOURCES cheapest-first until improved, pending, or exhausted.
Iteration only -- every policy (queue/current_best/exhausted, and what an
attempt IS for a kind) lives in derivations.py/attempts.py; this module
calls them and holds no state that would be lost between iterations, since
every ask() already appended a checkpoint before this loop ever sees it
(spec 2) -- a run stopped at any point leaves the cache exactly as far
along as it got.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .attempts import Need, Sourcing, attempt, provenance_source_for, sentence_attempt, sources_for
from .derivations import DEFAULT_ATTEMPT_CAP, exhausted, queue
from .ports import CacheReader, RecordWriter

__all__ = ["Budget", "Spend", "RunReport", "run"]


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

    def add(self, asks: int, cost: float) -> None:
        """`asks` asks (already counted by the caller -- see
        attempts._add's cache-hit exclusion), their total cost."""
        self.asks += asks
        self.cost += cost

    def exceeds(self, budget: Budget) -> bool:
        if budget.max_asks is not None and self.asks >= budget.max_asks:
            return True
        if budget.max_cost is not None and self.cost >= budget.max_cost:
            return True
        return False


@dataclass
class RunReport:
    """"a run that did almost nothing must look like one" (spec 3 section
    4): attempted/improved/exhausted/available are subject counts (not ask
    counts), so a run that touched nothing shows zeros everywhere except
    `available`, which counts the queued needs this run did NOT attempt --
    sentence needs excluded, since the per-run sentence attempt covers
    those and they are never work left over. `pending` counts one per queued need whose source
    escalation stopped on a pending (judge-batch-out) verdict, plus one
    more if the per-run sentence attempt itself came back pending.

    `excluded` and `unreachable` are the run's "what went wrong" fields
    (failing undetectably is a bug): `excluded` sums the questions the
    judge could not prepare across every attempt -- candidates dropped, not
    candidates rejected -- and `unreachable` says a judge TransportError
    (or an answer-nothing result) ended an attempt, which stops the run.
    """
    attempted: int = 0
    improved: int = 0
    exhausted: int = 0
    available: int = 0
    pending: int = 0
    sentences_adopted: int = 0
    excluded: int = 0
    unreachable: bool = False
    spend: dict[str, Spend] = field(default_factory=dict)


def _record_spend(spend: dict[str, Spend], outcome_spend: Mapping[str, tuple[int, float]]) -> None:
    """Merges an Outcome/SentenceOutcome's `{backend: (asks, cost)}` into
    the run's running per-backend Spend totals."""
    for backend, (asks, cost) in outcome_spend.items():
        spend.setdefault(backend, Spend()).add(asks, cost)


def run(ctx: Sourcing, budgets: Mapping[str, Budget], *,
       sentence_targets_per_run: int = 40) -> RunReport:
    """The application service (spec 3 section 4/7): one sentence_attempt()
    pass first (its adoptions applied to `ctx.syllabus` before the queue is
    computed, since sentence_attempt itself never mutates it), then for
    every other queued need, escalate attempts.SOURCES cheapest-first,
    stopping the need when an attempt improves, goes pending (a judge batch
    is out -- do not escalate past it), or its sources/budgets run out.

    The whole RUN stops at the first unreachable attempt (a judge that
    cannot be reached at all): no further need is attempted, the still-open
    needs stay counted in `available`, and the persisted report carries
    unreachable=True so the failure is visible afterwards, not silent.
    """
    report = RunReport()
    spend: dict[str, Spend] = {name: Spend() for name in budgets}

    so = sentence_attempt(ctx, max_targets=sentence_targets_per_run)
    _record_spend(spend, so.spend)
    report.sentences_adopted = len(so.adopted)
    report.excluded += so.excluded
    report.unreachable = report.unreachable or so.unreachable
    if so.pending:
        report.pending += 1
    ctx.syllabus = ctx.syllabus.with_sentences(so.adopted)

    entries = queue(ctx.syllabus, ctx.db, current_rubric=ctx.rubrics, prior=ctx.provenance_prior,
                    sources_for=sources_for, attempt_cap=DEFAULT_ATTEMPT_CAP,
                    provenance_source=provenance_source_for(ctx.db))
    # Sentence needs are covered once per run by sentence_attempt above, so
    # this loop never sees them and they are not work left over: counting
    # them in `available` reported as still-to-do exactly what the run had
    # just done.
    entries = [e for e in entries if e.kind != "sentence"]
    for entry in entries:
        # An unreachable judge (here or in the sentence attempt above) is a
        # dead wire, not a per-need failure: every remaining need would fail
        # the same way, so the run stops and the report says so rather than
        # grinding the whole queue against it.
        if report.unreachable:
            break
        need = Need(entry.subject, entry.kind)
        attempted = improved = unreachable = False
        for source in sources_for(need.kind):
            budget = budgets.get(source)
            s = spend.setdefault(source, Spend())
            if budget is not None and s.exceeds(budget):
                continue
            out = attempt(ctx, need, source)
            _record_spend(spend, out.spend)
            report.excluded += out.excluded
            attempted = attempted or out.attempted
            if out.unreachable:
                unreachable = True
                break
            if out.pending:
                report.pending += 1
                break
            if out.improved:
                improved = True
                break
        report.attempted += int(attempted)
        report.improved += int(improved)
        if unreachable:
            report.unreachable = True
            break
        if exhausted(ctx.db, entry.subject, entry.kind, sources=sources_for(entry.kind),
                    attempt_cap=DEFAULT_ATTEMPT_CAP).exhausted:
            report.exhausted += 1

    report.available = len(entries) - report.attempted
    report.spend = spend
    _persist_report(ctx.db, report)
    return report


def _persist_report(cache: CacheReader, report: RunReport) -> None:
    """Appends one summary row per run() call (port="run",
    backend="runreport") so a run's own outcome has a durable source --
    RunReport itself is only ever an in-memory return value, and nothing
    else writes one to the cache. `key` is a constant, readable label
    ("runreport"); the `cache` table's primary key is (key_sha, ts)
    (store.py), so every call still lands its own row -- "keyed on
    timestamp", not deduplicated by key. Silently a no-op when `cache` is
    read-only (doesn't also satisfy RecordWriter): run() itself only ever
    needs cache read access, so this is best-effort persistence, not a
    hard requirement of the application service's contract.
    """
    if not isinstance(cache, RecordWriter):
        return
    cache.append(
        port="run", backend="runreport", key="runreport", subject="run",
        question={},
        answer={"attempted": report.attempted, "improved": report.improved,
                "exhausted": report.exhausted, "available": report.available,
                "pending": report.pending, "sentences_adopted": report.sentences_adopted,
                "excluded": report.excluded, "unreachable": report.unreachable,
                "spend": {name: {"asks": s.asks, "cost": s.cost}
                         for name, s in report.spend.items()}},
        cost=sum(s.cost for s in report.spend.values()))
