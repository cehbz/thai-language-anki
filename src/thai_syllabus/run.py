"""The batch run (spec 3 section 7): resolve what the last run left, one
sentence attempt over the open Targets, then one Source per queued need.

Iteration only -- every policy (queue/current_best/exhausted/next_source,
and what an attempt IS for a kind) lives in derivations.py/attempts.py.
Escalation to the next source happens on the NEXT run, for every transport
alike, so a run is cheap and repeatable; nothing here holds state that
would be lost part-way, since every ask() appended its own checkpoint
before this loop ever saw it (spec 2).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .assessor import JudgeUnreachable
from .attempts import (
    AttemptResult,
    Need,
    Sourcing,
    Spend,
    attempt,
    current_best_of,
    provenance_source_for,
    sentence_attempt,
    sources_for,
)
from .derivations import (
    DEFAULT_ATTEMPT_CAP,
    adoptable_drafts,
    exhausted,
    improved,
    next_source,
    queue,
)
from .entities import Sentence
from .ports import CacheReader, RecordWriter

__all__ = ["Budget", "Spend", "RunReport", "run"]


@dataclass(frozen=True)
class Budget:
    """Per-backend spend cap for one run, in that backend's own currency
    (spec 3 section 7): asks (forvo: 450/day default), dollars (judge
    api/batch), subscription quota (judge cli), learner attention (session
    default 20 questions -- pull-based: this caps what queue() hands the
    learner, never what is owed). Either field may be set; a backend with
    neither is unbounded.
    """
    max_asks: int | None = None
    max_cost: float | None = None

    def exceeded_by(self, spend: Spend) -> bool:
        """Whether `spend` has reached this cap -- budget policy, so it
        lives with the Budget and not with the Spend it measures."""
        if self.max_asks is not None and spend.asks >= self.max_asks:
            return True
        if self.max_cost is not None and spend.cost >= self.max_cost:
            return True
        return False


FORVO_DEFAULT_DAILY_BUDGET = Budget(max_asks=450)
LEARNER_DEFAULT_SESSION_BUDGET = Budget(max_asks=20)


@dataclass
class RunReport:
    """"a run that did almost nothing must look like one" (F10):
    attempted/improved/exhausted/available are subject counts, not ask
    counts. `available` counts the queued needs this run did NOT attempt
    (sentence needs excluded -- the per-run sentence attempt covers those).
    `pending` counts the needs whose judge questions this run collected
    without an answer.

    `excluded` and `unreachable` are the "what went wrong" fields: the
    questions the judge could not prepare (candidates dropped, not
    candidates rejected), and a judge that could not be reached at all,
    which stops the run.
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


def _record_spend(spend: dict[str, Spend], attempt_spend: Mapping[str, Spend]) -> None:
    for backend, incurred in attempt_spend.items():
        spend.setdefault(backend, Spend()).add(incurred.asks, incurred.cost)


def _adopt_sentences(ctx: Sourcing) -> tuple[Sentence, ...]:
    """The cover over every verified draft on record (Syllabus.cover),
    written to the sentences table and applied to `ctx.syllabus`.
    """
    chosen = ctx.syllabus.cover(adoptable_drafts(
        ctx.db, ctx.syllabus, current_rubric=ctx.rubrics, model=ctx.judge_model,
        today=ctx.today))
    for sentence, _targets in chosen:
        ctx.db.add_sentence(text_sha=sentence.text_sha, text=sentence.text,
                            gloss=sentence.gloss, voice=sentence.voice,
                            source=sentence.provenance.source,
                            origin=sentence.provenance.origin,
                            licence=sentence.provenance.licence,
                            acquired=sentence.provenance.acquired)
    adopted = tuple(sentence for sentence, _targets in chosen)
    ctx.syllabus = ctx.syllabus.with_sentences(adopted)
    return adopted


def run(ctx: Sourcing, budgets: Mapping[str, Budget], *,
        sentence_targets_per_run: int = 40) -> RunReport:
    """One sentence attempt over the open Targets, adoption of whatever the
    judge has passed, then one Source per queued need (the cheapest not yet
    tried since current-best last changed).

    The whole run stops at the first unreachable judge: no further need is
    attempted, the still-open needs stay counted in `available`, and the
    persisted report says so, rather than grinding the queue against a dead
    wire.
    """
    report = RunReport()
    spend: dict[str, Spend] = {name: Spend() for name in budgets}

    report.sentences_adopted = len(_adopt_sentences(ctx))
    try:
        drafting = sentence_attempt(ctx, max_targets=sentence_targets_per_run)
    except JudgeUnreachable:
        report.unreachable = True
        drafting = AttemptResult(attempted=False)
    _record_spend(spend, drafting.spend)
    report.excluded += len(drafting.excluded)
    report.pending += int(bool(drafting.questions))
    if not report.unreachable:
        report.sentences_adopted += len(_adopt_sentences(ctx))

    entries = [e for e in queue(ctx.syllabus, ctx.db, current_rubric=ctx.rubrics,
                                prior=ctx.provenance_prior, sources_for=sources_for,
                                attempt_cap=DEFAULT_ATTEMPT_CAP,
                                provenance_source=provenance_source_for(ctx.db))
               if e.kind != "sentence"]
    for entry in entries:
        if report.unreachable:
            break
        need = Need(entry.subject, entry.kind, entry.subject_kind)
        source = next_source(ctx.db, need.subject, need.kind, sources_for(need.kind))
        if source is None:
            continue
        budget = budgets.get(source)
        if budget is not None and budget.exceeded_by(spend.setdefault(source, Spend())):
            continue
        before = current_best_of(ctx, need.subject, need.kind)
        try:
            result = attempt(ctx, need, source)
        except JudgeUnreachable:
            report.unreachable = True
            break
        _record_spend(spend, result.spend)
        report.excluded += len(result.excluded)
        report.attempted += int(result.attempted)
        if result.questions:
            report.pending += 1
        elif improved(before, current_best_of(ctx, need.subject, need.kind)):
            report.improved += 1
        if exhausted(ctx.db, need.subject, need.kind, sources=sources_for(need.kind),
                     attempt_cap=DEFAULT_ATTEMPT_CAP).exhausted:
            report.exhausted += 1

    report.available = len(entries) - report.attempted
    report.spend = spend
    _persist_report(ctx.db, report)
    return report


def _persist_report(cache: CacheReader, report: RunReport) -> None:
    """One summary row per run() call (port="run", backend="runreport") so
    a run's own outcome has a durable source -- RunReport itself is only
    ever an in-memory return value. `key` is a constant label; the `cache`
    table's primary key is (key_sha, ts), so every call lands its own row.
    A no-op when `cache` is read-only: run() itself only needs read access,
    so this is best-effort persistence, not part of its contract.
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
