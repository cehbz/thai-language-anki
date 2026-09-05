"""The batch run (spec 3 section 7): the previous run's judge batch
resolved and what it passed adopted, one sentence attempt over the open
Targets, one Source per queued need, and every question collected on the
way submitted as one batch.

Iteration only -- every policy (queue/current_best/exhausted/next_source,
what a picture still owes a preference question, and what an attempt IS
for a kind) lives in derivations.py/attempts.py. Escalation to the next
source happens on the NEXT run, for every transport alike, so a run is
cheap and repeatable; nothing here holds state that would be lost
part-way, since every ask() appended its own checkpoint before this loop
ever saw it (spec 2).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time

from .assessor import JudgeUnreachable, PreparedQuestion
from .attempts import (
    AttemptResult,
    Need,
    Sourcing,
    Spend,
    attempt,
    current_best_of,
    preference_attempt,
    provenance_source_for,
    sentence_attempt,
)
from .derivations import (
    QueuedNeeds,
    QueueEntry,
    adoptable_drafts,
    improved,
    next_source,
    queued,
)
from .entities import Sentence
from .ports import RecordWriter
from .record import asks_since, spend_since
from .transport import TransportError

__all__ = ["Budget", "Spend", "RunReport", "run"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Budget:
    """One backend's spend cap for a day, in that backend's own currency
    (spec 3 section 7): asks (forvo: 450/day), dollars (judge api/batch),
    subscription quota (judge cli). A run measures a cap against what the
    record says that backend already spent since midnight plus what the
    run itself has spent. Either field may be set; a backend with neither
    is unbounded. The learner cap is the review screen's, over the
    questions it hands the learner, not a Source budget this loop spends.
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


@dataclass(frozen=True)
class RunReport:
    """"a run that did almost nothing must look like one" (F10): every
    count is a subject count, not an ask count, and `available` is every
    need Syllabus.gaps() listed -- so `available` always equals `attempted`
    + `exhausted` + `pending` + `unserved` + `budgeted` + `deferred`.

    `attempted`: needs whose Source was asked (the one that met a dead
    judge included -- its ask was made and appended), plus, when the
    sentence attempt ran this run, every open Target need it was handed
    -- at most `max_targets` (the per-run cap) of them, whatever the
    attempt's own AttemptResult.targets_handed says (no Source is asked
    per Target; the attempt itself is what serves them -- `drafted`
    separately counts the drafts it produced, whether or not they covered
    a Target). `improved`: needs whose current-best artifact changed.
    `exhausted`: needs with no source left, whether the queue dropped
    them or the loop found none. `pending`: subjects with a question in
    an unresolved batch, this run's own submission included.
    `sentences_adopted`: drafts this run covered open Targets with.
    `unserved`: needs whose kind has no Source and no per-run pass either
    (derivations.QueuedNeeds.unserved). `budgeted`: needs skipped this run
    because their Source's day budget was already spent -- a per-need skip
    in the loop, or, when the llm-sentence budget gated the sentence
    attempt out entirely, every open Target need within the per-run cap
    it would have been handed. `deferred`: available needs this run never
    even considered -- the open Targets beyond the per-run cap (handed or
    not, the excess was never looked at), plus, when the run ended before
    looking at any need at all (a batch still out from the previous run,
    or the judge unreachable while resolving one), `available` minus
    `pending`; zero in every other case.

    The "what went wrong" fields: `excluded` counts questions the judge
    could not prepare (candidates dropped, not candidates rejected),
    `source_failures` counts each Source that failed on the wire (skipped
    for the rest of the run, never fatal), and `unreachable` says the
    judge could not be reached at all, which stops the run.
    """
    attempted: int = 0
    improved: int = 0
    exhausted: int = 0
    available: int = 0
    pending: int = 0
    sentences_adopted: int = 0
    drafted: int = 0
    excluded: int = 0
    unreachable: bool = False
    batch_id: str | None = None
    source_failures: dict[str, int] = field(default_factory=dict)
    spend: dict[str, Spend] = field(default_factory=dict)
    unserved: int = 0
    budgeted: int = 0
    deferred: int = 0


@dataclass
class _Tally:
    """One run's running counts, the report's own fields before they are
    frozen into it."""
    attempted: int = 0
    improved: int = 0
    exhausted: int = 0
    excluded: int = 0
    sentences_adopted: int = 0
    drafted: int = 0
    budgeted: int = 0
    deferred: int = 0
    unreachable: bool = False
    questions: list[PreparedQuestion] = field(default_factory=list)
    source_failures: dict[str, int] = field(default_factory=dict)
    spend: dict[str, Spend] = field(default_factory=dict)

    def collect(self, result: AttemptResult) -> None:
        """What an attempt produced, whatever the need was: its questions,
        its exclusions, and its spend per backend."""
        self.questions += result.questions
        self.excluded += len(result.excluded)
        self.drafted += result.drafted
        for backend, incurred in result.spend.items():
            self.spend.setdefault(backend, Spend()).add(incurred.asks, incurred.cost)


def _adopt_sentences(ctx: Sourcing) -> int:
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
    adopted: tuple[Sentence, ...] = tuple(sentence for sentence, _targets in chosen)
    ctx.syllabus = ctx.syllabus.with_sentences(adopted)
    return len(adopted)


def _resolve_previous_batch(ctx: Sourcing, tally: _Tally,
                            outstanding: tuple[str, frozenset[str]]
                            ) -> tuple[str, frozenset[str]] | None:
    """Resolves the batch the last run left and puts the preference
    questions its verdicts opened into this run's own. Returns the batch
    still out afterwards -- one that has not ended releases nothing -- and
    None once it has released.
    """
    batch_id, subjects = outstanding
    ctx.assessor.resolve(batch_id)
    still_out = ctx.assessor.unresolved_batch()
    if still_out is not None:
        return still_out[0], frozenset(still_out[1])
    tally.collect(preference_attempt(ctx, sorted(subjects)))
    return None


def day_start_ns(today: date) -> int:
    """Midnight local on `today`, in the nanoseconds a cache row's ts is
    stamped in -- the window a per-day budget is summed over."""
    return int(datetime.combine(today, time.min).timestamp() * 1_000_000_000)


def _spent_today(ctx: Sourcing, budgets: Mapping[str, Budget]) -> dict[str, Spend]:
    """What the record says each budgeted backend has already spent since
    midnight, read once: this run's own asks are counted from the tally,
    not read back as they land.
    """
    since = day_start_ns(ctx.today())
    return {name: Spend(asks=asks_since(ctx.db, name, since),
                        cost=spend_since(ctx.db, name, since))
            for name in budgets}


def _spent_on(source: str, carried: Mapping[str, Spend], tally: _Tally) -> Spend:
    """`source`'s day so far: what the record already held plus what this
    run has spent on it."""
    already, mine = carried.get(source, Spend()), tally.spend.setdefault(source, Spend())
    return Spend(asks=already.asks + mine.asks, cost=already.cost + mine.cost)


def _needs(ctx: Sourcing) -> QueuedNeeds:
    return queued(ctx.syllabus, ctx.db, current_rubric=ctx.rubrics,
                  prior=ctx.provenance_prior, sources_for=ctx.sources_for,
                  attempt_cap=ctx.attempt_cap,
                  provenance_source=provenance_source_for(ctx.db))


def _open_target_count(ctx: Sourcing) -> int:
    """How many Targets are still unfilled right now -- `queued()` excludes
    "sentence" kind needs from `entries`/`exhausted`/`unserved` entirely
    (the run's own sentence attempt serves them, not a Source), so this is
    the only place their count reaches `attempted`/`budgeted`/`deferred`.
    """
    return len(ctx.syllabus.gaps().unfilled_targets)


def _try_each_need(ctx: Sourcing, entries: Sequence[QueueEntry], budgets: Mapping[str, Budget],
                   carried: Mapping[str, Spend], tally: _Tally) -> None:
    """One Source per need -- the cheapest not yet tried since current-best
    last changed. A Source that fails on the wire is skipped for the rest
    of the run; an unreachable judge stops it there and then, leaving the
    still-open needs counted in `available`, rather than grinding the
    queue against a dead wire.
    """
    dead_sources: set[str] = set()
    for entry in entries:
        if entry.kind == "sentence":
            continue  # the per-run sentence attempt covers every open Target
        need = Need(entry.subject, entry.kind, entry.subject_kind)
        sources = ctx.sources_for(need.kind)
        if not sources:
            continue  # no Source serves this kind: it is nobody's to attempt
        source = next_source(ctx.db, need.subject, need.kind, sources)
        if source is None:
            tally.exhausted += 1
            continue
        if source in dead_sources:
            continue
        budget = budgets.get(source)
        if budget is not None and budget.exceeded_by(_spent_on(source, carried, tally)):
            tally.budgeted += 1
            continue
        before = current_best_of(ctx, need.subject, need.kind)
        try:
            result = attempt(ctx, need, source)
        except JudgeUnreachable:
            tally.unreachable = True
            tally.attempted += 1
            return
        except TransportError as e:
            dead_sources.add(source)
            tally.source_failures[source] = tally.source_failures.get(source, 0) + 1
            _log.warning("source %s failed for %s/%s: %s", source, need.subject, need.kind, e)
            continue
        tally.collect(result)
        tally.attempted += int(result.attempted)
        if improved(before, current_best_of(ctx, need.subject, need.kind)):
            tally.improved += 1


def run(ctx: Sourcing, budgets: Mapping[str, Budget], *,
        sentence_targets_per_run: int = 40) -> RunReport:
    """One pass: resolve, adopt, draft sentences once, try each queued
    need at its next source, submit everything collected as one batch.

    Two things end the pass early. A judge that cannot be reached -- at
    the resolve, inside an attempt, or at the submit -- stops it there,
    reported and persisted (the cli exits non-zero). A batch still
    unanswered keeps it from attempting or submitting anything: at most
    one batch is ever outstanding.
    """
    tally = _Tally(spend={name: Spend() for name in budgets})
    # Read before any ask this run makes lands on the record -- the
    # sentence attempt's own llm-sentence row, once appended, would
    # otherwise count twice against its day budget: once read back here,
    # once already in `tally.spend`.
    carried = _spent_today(ctx, budgets)
    previous = ctx.assessor.unresolved_batch()
    still_out = previous
    if previous is not None:
        try:
            still_out = _resolve_previous_batch(ctx, tally, previous)
        except JudgeUnreachable:
            tally.unreachable = True
            needs = _needs(ctx)
            pending = frozenset(previous[1])
            return _finish(ctx, tally, needs, batch_id=previous[0], pending=pending,
                           extra_deferred=needs.available - len(pending))
    tally.sentences_adopted += _adopt_sentences(ctx)
    if still_out is not None:
        # The run ended here, before it ever looked at a need: every
        # available need this run never considered (not even pending, in
        # a still-earlier batch) is deferred, not lost.
        needs = _needs(ctx)
        pending = still_out[1]
        return _finish(ctx, tally, needs, batch_id=still_out[0], pending=pending,
                       extra_deferred=needs.available - len(pending))

    # Measured before the attempt runs (or is gated out): sentence_attempt
    # hands over at most `sentence_targets_per_run` of these (the per-run
    # cap), so anything beyond it was never even considered this run.
    open_before = _open_target_count(ctx)
    sentence_budget = budgets.get("llm-sentence")
    if sentence_budget is None or not sentence_budget.exceeded_by(
            _spent_on("llm-sentence", carried, tally)):
        try:
            result = sentence_attempt(ctx, max_targets=sentence_targets_per_run)
            tally.collect(result)
        except JudgeUnreachable:
            tally.unreachable = True
            handed = min(open_before, sentence_targets_per_run)
            tally.attempted += handed
            tally.deferred += open_before - handed
            return _finish(ctx, tally, _needs(ctx), batch_id=None, pending=frozenset())
        # An inline transport answers inside that attempt: what it
        # verified there is adoptable in this same run.
        tally.sentences_adopted += _adopt_sentences(ctx)
        # The sentence attempt served every open Target within the
        # per-run cap it was handed -- no Source is asked per Target, so
        # this is how they reach `attempted` (a Target the drafts covered
        # has already left gaps() by now and needs no separate
        # accounting); the excess beyond the cap was never considered
        # this run at all, so it is deferred instead.
        excess = open_before - result.targets_handed
        tally.attempted += _open_target_count(ctx) - excess
        tally.deferred += excess
    else:
        # The llm-sentence budget kept the attempt from running at all:
        # every open Target need within the per-run cap it would have
        # been handed is budget-constrained, same as a per-need Source
        # skip below; the excess beyond the cap is deferred, same as when
        # the attempt runs.
        handed = min(open_before, sentence_targets_per_run)
        tally.budgeted += handed
        tally.deferred += open_before - handed

    needs = _needs(ctx)
    _try_each_need(ctx, needs.entries, budgets, carried, tally)
    if tally.unreachable:
        return _finish(ctx, tally, needs, batch_id=None, pending=frozenset())

    try:
        batch_id = ctx.assessor.submit(tally.questions)
    except JudgeUnreachable:
        tally.unreachable = True
        return _finish(ctx, tally, needs, batch_id=None, pending=frozenset())
    pending = (frozenset(q.question.subject for q in tally.questions)
               if batch_id is not None else frozenset())
    return _finish(ctx, tally, needs, batch_id=batch_id, pending=pending)


def _finish(ctx: Sourcing, tally: _Tally, needs: QueuedNeeds, *, batch_id: str | None,
            pending: frozenset[str], extra_deferred: int = 0) -> RunReport:
    """The run's own outcome, as one durable row and one return value.
    `deferred` is `tally.deferred` (the sentence attempt's per-run-cap
    excess, accumulated as the run went) plus `extra_deferred`, which is
    nonzero only when the run ended before it ever looked at a need at
    all (a batch still out, or the judge unreachable while resolving
    one): every available need it never considered, pending or not.
    """
    report = RunReport(
        attempted=tally.attempted, improved=tally.improved,
        exhausted=needs.exhausted + tally.exhausted, available=needs.available,
        pending=len(pending), sentences_adopted=tally.sentences_adopted,
        drafted=tally.drafted, excluded=tally.excluded, unreachable=tally.unreachable,
        batch_id=batch_id, source_failures=tally.source_failures, spend=tally.spend,
        unserved=needs.unserved, budgeted=tally.budgeted,
        deferred=tally.deferred + extra_deferred)
    _persist_report(ctx.db, report)
    return report


def _persist_report(record: RecordWriter, report: RunReport) -> None:
    """One summary row per run() call (port="run", backend="runreport") so
    a run's own outcome has a durable source -- RunReport itself is only
    ever an in-memory return value. `key` is a constant label; the `cache`
    table's primary key is (key_sha, ts), so every call lands its own row.
    """
    record.append(
        port="run", backend="runreport", key="runreport", subject="run",
        question={"kind": "runreport"},
        answer={"attempted": report.attempted, "improved": report.improved,
                "exhausted": report.exhausted, "available": report.available,
                "pending": report.pending, "sentences_adopted": report.sentences_adopted,
                "drafted": report.drafted,
                "excluded": report.excluded, "unreachable": report.unreachable,
                "batch_id": report.batch_id, "source_failures": dict(report.source_failures),
                "spend": {name: {"asks": s.asks, "cost": s.cost}
                          for name, s in report.spend.items()},
                "unserved": report.unserved, "budgeted": report.budgeted,
                "deferred": report.deferred},
        cost=sum(s.cost for s in report.spend.values()))
