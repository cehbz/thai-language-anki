"""The batch run (spec 3 section 7): the previous run's judge batch
resolved and what it passed adopted, one sentence attempt over the open
Targets, one Source per queued need, and every question collected on the
way submitted as one batch.

Iteration only: every policy (queue/current_best/exhausted/next_source,
what a picture still owes a preference question, what an attempt is for a
kind) lives in derivations.py and attempts.py. Escalation to the next
source happens on the next run, for every transport alike; every ask()
appended its own checkpoint before this loop saw it (spec 2).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from time import time_ns

from .assessor import JudgeUnreachable, PreparedQuestion
from .cachekeys import RunReportKey
from .attempts import (
    AttemptResult,
    Need,
    Sourcing,
    Spend,
    assess_first,
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
    available_need_keys,
    improved,
    next_source,
    open_words,
    queued,
)
from .entities import Sentence
from .ports import RecordWriter
from .record import asks_since, fetches_since, spend_since
from .transport import QuotaExhausted, TransportError

__all__ = ["Budget", "Spend", "RunReport", "run"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Budget:
    """One backend's spend cap for a day, in that backend's own currency
    (spec 3 section 7), measured against what the record says it spent
    since `day_starts` plus what this run has spent. `max_asks` and
    `max_cost` may each be set independently; a backend with neither is
    unbounded. `day_starts` is `"HH:MM"` plus `Z` or `+HH:MM`/`-HH:MM`
    (day_start_ns's format); None sums from local midnight.
    """
    max_asks: int | None = None
    max_cost: float | None = None
    day_starts: str | None = None

    def exceeded_by(self, spend: Spend) -> bool:
        """Whether `spend` has reached this cap."""
        if self.max_asks is not None and spend.asks >= self.max_asks:
            return True
        if self.max_cost is not None and spend.cost >= self.max_cost:
            return True
        return False


FORVO_DEFAULT_DAILY_BUDGET = Budget(max_asks=450, day_starts="22:00Z")
LEARNER_DEFAULT_SESSION_BUDGET = Budget(max_asks=20)


@dataclass(frozen=True)
class RunReport:
    """One run's account, in needs -- one (subject, kind) each, never
    asks. `available` is every need Syllabus.gaps() listed, and equals
    `attempted` + `exhausted` + `pending` + `unserved` + `budgeted` +
    `deferred`: every need lands in exactly one bucket. `preferences`
    sits outside that identity (its need has already left `available`).
    """
    # needs whose Source was asked, plus the words the sentence attempt
    # was handed a still-open Target for this run
    attempted: int = 0
    improved: int = 0          # needs whose current-best artifact changed
    exhausted: int = 0         # needs with no source left
    available: int = 0         # every need Syllabus.gaps() listed
    # needs with a question in an unresolved batch, this run's submission
    # included, narrowed to needs `available` still counts
    pending: int = 0
    sentences_adopted: int = 0  # drafts this run covered open Targets with
    drafted: int = 0            # drafts the sentence attempt produced
    excluded: int = 0           # questions the judge could not prepare
    # one {subject, artifact_sha, reason} per exclusion, for a screen to
    # read back per subject (record.excluded_candidates)
    excluded_items: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    unreachable: bool = False   # the judge could not be reached, which stops the run
    batch_id: str | None = None
    # per Source that failed on the wire (skipped for the rest of the run)
    source_failures: dict[str, int] = field(default_factory=dict)
    spend: dict[str, Spend] = field(default_factory=dict)
    unserved: int = 0          # needs whose kind has no Source and no per-run pass
    budgeted: int = 0          # needs whose Source's day budget was already spent
    deferred: int = 0          # available needs this run never considered
    # questions ranking a word's passing pictures once their fits resolved
    preferences: int = 0


@dataclass
class _Tally:
    """One run's running counts, the report's own fields before they are
    frozen into it."""
    attempted: int = 0
    improved: int = 0
    exhausted: int = 0
    excluded: int = 0
    excluded_items: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    sentences_adopted: int = 0
    drafted: int = 0
    budgeted: int = 0
    deferred: int = 0
    unreachable: bool = False
    questions: list[PreparedQuestion] = field(default_factory=list)
    source_failures: dict[str, int] = field(default_factory=dict)
    spend: dict[str, Spend] = field(default_factory=dict)
    preferences: int = 0
    # Needs whose own attempt this run collected judge questions for --
    # not yet counted as `attempted`: whether they land in `pending` or
    # fall back to `attempted` is known once submit() settles the batch's
    # fate. Internal only, never a RunReport field.
    pending_candidates: int = 0

    def collect(self, result: AttemptResult) -> None:
        """What an attempt produced, whatever the need was: its
        questions, its exclusions (each naming its own subject and
        artifact_sha), its drafts, and its spend per backend.

        A candidate can be excluded twice across the same need's own
        calls this run -- assess-first's own exclusion (attempts.py
        assess_first, spec 3 section 5) collected on fall-through, then
        the same candidate excluded again by the source attempt that
        follows (e.g. _judge_pictures re-asks the fit question for every
        candidate on record, not only the freshly fetched ones). One
        candidate must count once in RunReport.excluded (spec 3 section
        7), so an exclusion naming an artifact_sha is deduplicated here
        on (subject, artifact_sha) across every `collect` call this
        tally sees. An exclusion naming no artifact_sha (artifact_sha is
        None -- e.g. one `fills` question per Target under one sentence
        draft) names no candidate to collide on, so each such exclusion
        keeps its own item even when it shares a subject with another.
        """
        self.questions += result.questions
        seen = {(item["subject"], item["artifact_sha"]) for item in self.excluded_items
                if item["artifact_sha"] is not None}
        for item in result.excluded.values():
            if item.artifact_sha is not None:
                key = (item.subject, item.artifact_sha)
                if key in seen:
                    continue
                seen.add(key)
            self.excluded_items += (
                {"subject": item.subject, "artifact_sha": item.artifact_sha,
                 "reason": item.reason},)
        self.excluded = len(self.excluded_items)
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
                            clauses=sentence.clauses,
                            gloss=sentence.gloss, voice=sentence.voice,
                            source=sentence.provenance.source,
                            origin=sentence.provenance.origin,
                            licence=sentence.provenance.licence,
                            acquired=sentence.provenance.acquired)
    adopted: tuple[Sentence, ...] = tuple(sentence for sentence, _targets in chosen)
    ctx.syllabus = ctx.syllabus.with_sentences(adopted)
    return len(adopted)


def _resolve_previous_batch(ctx: Sourcing, tally: _Tally,
                            outstanding: tuple[str, frozenset[tuple[str, str]]]
                            ) -> tuple[str, frozenset[tuple[str, str]]] | None:
    """Resolves the batch the last run left and puts the preference
    questions its verdicts opened into this run's own. Returns the batch
    still out afterwards -- one that has not ended releases nothing -- and
    None once it has released.
    """
    batch_id, batch_needs = outstanding
    ctx.assessor.resolve(batch_id)
    still_out = ctx.assessor.unresolved_batch()
    if still_out is not None:
        return still_out[0], frozenset(still_out[1])
    preference = preference_attempt(ctx, sorted({subject for subject, _kind in batch_needs}))
    tally.collect(preference)
    # Whether each of these ranks a need that has since left `available`
    # (-> `preferences`) or one a learner rejection with no acceptable
    # floor kept open even as a new candidate joined its passing set (->
    # `pending` instead) is decided once, in run(), against the same
    # `available_need_keys` snapshot `pending` itself uses -- not here,
    # before `_adopt_sentences` has even run this same resolve.
    return None


_DAY_STARTS_RE = re.compile(
    r"^(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d)(?P<zone>Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$")


def parse_day_starts(day_starts: str) -> tuple[time, timezone]:
    """The wall time and zone a `day_starts` string names: `HH:MM`
    followed by `Z` or a `+HH:MM`/`-HH:MM` offset. Raises ValueError,
    naming `day_starts`, on anything else.
    """
    match = _DAY_STARTS_RE.match(day_starts)
    if match is None:
        raise ValueError(
            f"day_starts: {day_starts!r} does not parse as HH:MM plus Z or +/-HH:MM")
    hour, minute, zone = int(match["hour"]), int(match["minute"]), match["zone"]
    if zone == "Z":
        offset = timedelta(0)
    else:
        sign = 1 if zone[0] == "+" else -1
        zone_hour, zone_minute = zone[1:].split(":")
        offset = sign * timedelta(hours=int(zone_hour), minutes=int(zone_minute))
    return time(hour, minute), timezone(offset)


def day_start_ns(now: datetime, day_starts: str | None) -> int:
    """The most recent instant at `day_starts`'s wall time, in
    `day_starts`'s own zone, at or before `now` -- local midnight in
    `now`'s own zone when `day_starts` is None. Nanoseconds, the unit a
    cache row's ts is stamped in -- the window a per-day budget is
    summed over.
    """
    wall, zone = (time.min, now.tzinfo) if day_starts is None else parse_day_starts(day_starts)
    at_zone = now.astimezone(zone)
    candidate = at_zone.replace(hour=wall.hour, minute=wall.minute, second=0, microsecond=0)
    if candidate > at_zone:
        candidate -= timedelta(days=1)
    return int(candidate.timestamp() * 1_000_000_000)


def _spent_today(ctx: Sourcing, budgets: Mapping[str, Budget]) -> dict[str, Spend]:
    """What the record says each budgeted backend spent since its own
    budget's `day_starts`, read once; this run's own asks are counted
    from the tally. A source's day counts its Source asks plus the bytes
    fetches attributed to it (Forvo counts downloads as requests).
    """
    now = datetime.now().astimezone()
    spent: dict[str, Spend] = {}
    for name, budget in budgets.items():
        since = day_start_ns(now, budget.day_starts)
        spent[name] = Spend(asks=asks_since(ctx.db, name, since) + fetches_since(ctx.db, name, since),
                            cost=spend_since(ctx.db, name, since))
    return spent


def _spent_on(source: str, carried: Mapping[str, Spend], tally: _Tally) -> Spend:
    """`source`'s day so far: what the record already held plus what this
    run has spent on it."""
    already, mine = carried.get(source, Spend()), tally.spend.setdefault(source, Spend())
    return Spend(asks=already.asks + mine.asks, cost=already.cost + mine.cost)


def _needs(ctx: Sourcing, collected_this_run: frozenset[tuple[str, str]] = frozenset(),
          *, now_ns: int) -> QueuedNeeds:
    """One queue build (spec 3 r19 section 6a/9): `now_ns` is the run's own
    single clock read (run()'s outermost caller), never re-read here or in
    derivations.py, so every need this pass considers ages against the
    same instant.
    """
    return queued(ctx.syllabus, ctx.db, current_rubric=ctx.rubrics,
                  prior=ctx.provenance_prior, sources_for=ctx.sources_for,
                  attempt_cap=ctx.attempt_cap, transient_cap=ctx.transient_cap,
                  provenance_source=provenance_source_for(ctx.db),
                  collected_this_run=collected_this_run,
                  nothing_ttl=ctx.nothing_ttl, now_ns=now_ns)


def _pending_needs(ctx: Sourcing, batch_needs: frozenset[tuple[str, str]]) -> int:
    """How many of the outstanding batch's (subject, kind) questions name
    a need `available` still counts -- `pending` on a path that ends
    before any attempt, in needs, the measure every other path uses.
    """
    return len(frozenset(batch_needs) & available_need_keys(ctx.syllabus))


def _unconsidered(needs: QueuedNeeds, pending: int) -> int:
    """The available needs left over once the outstanding batch's pending
    ones and the queue's own exhausted and unserved counts are taken out:
    what a run that ended before any attempt defers. _finish adds the
    latter two buckets itself. Refuses counts that between them claim
    more needs than `available` holds -- a negative `deferred` is a
    defect in the three buckets, not a number to clamp.
    """
    claimed = pending + needs.exhausted + needs.unserved
    if claimed > needs.available:
        raise ValueError(
            "a run ending before any attempt claimed more needs than are available: "
            f"available={needs.available} pending={pending} "
            f"exhausted={needs.exhausted} unserved={needs.unserved}")
    return needs.available - claimed


def _try_each_need(ctx: Sourcing, entries: Sequence[QueueEntry], budgets: Mapping[str, Budget],
                   carried: Mapping[str, Spend], tally: _Tally, *, now_ns: int) -> int:
    """Assess-first, then one Source per need: the fit questions a
    candidate on record is owed (spec 3 section 5), else the cheapest
    source not yet tried since current-best last changed. A Source that
    fails on the wire is skipped for the rest of the run and every need
    waiting on it is deferred; an unreachable judge stops the loop. A
    Source that states its own quota is spent (spec 3 section 6a's Quota
    state) is skipped for the rest of the run too, and every need on it,
    this one included, counts budgeted, not deferred -- the same bucket a
    spent day budget uses, and it is never a source_failures entry.
    Returns how many entries it never reached (zero unless a dead judge
    stopped it), for run() to defer. `now_ns` is run()'s own single clock
    read (spec 3 r19 section 6a/9), passed straight to next_source rather
    than re-read here.
    """
    dead_sources: set[str] = set()
    budgeted_sources: set[str] = set()
    for index, entry in enumerate(entries):
        need = Need(entry.subject, entry.kind, entry.subject_kind)
        before = current_best_of(ctx, need.subject, need.kind)
        try:
            result = assess_first(ctx, need)
        except JudgeUnreachable:
            tally.unreachable = True
            tally.attempted += 1
            return len(entries) - index - 1
        if result is None or not result.attempted:
            if result is not None:
                # assess-first's own exclusions (spec 3 section 5): every
                # awaiting candidate was excluded, so its `excluded` must
                # still reach the report (section 7) even though the
                # attempt falls through to a source below.
                tally.collect(result)
            sources = ctx.sources_for(need.kind)
            source = next_source(ctx.db, need.subject, need.kind, sources,
                                transient_cap=ctx.transient_cap,
                                nothing_ttl=ctx.nothing_ttl, now_ns=now_ns)
            if source is None:
                tally.exhausted += 1
                continue
            if source in dead_sources:
                # No row was written for this need: the next run asks the
                # same source again.
                tally.deferred += 1
                continue
            if source in budgeted_sources:
                # Same reasoning as a spent day budget below: the source
                # said its own allowance is gone, so this need is
                # budgeted, not deferred, and asks nothing.
                tally.budgeted += 1
                continue
            budget = budgets.get(source)
            if budget is not None and budget.exceeded_by(_spent_on(source, carried, tally)):
                tally.budgeted += 1
                continue
            try:
                result = attempt(ctx, need, source)
            except JudgeUnreachable:
                tally.unreachable = True
                tally.attempted += 1
                return len(entries) - index - 1
            except QuotaExhausted:
                budgeted_sources.add(source)
                tally.budgeted += 1
                _log.warning("source %s: quota exhausted; budgeted for the rest of the run",
                            source)
                continue
            except TransportError as e:
                dead_sources.add(source)
                tally.deferred += 1
                tally.source_failures[source] = tally.source_failures.get(source, 0) + 1
                _log.warning("source %s failed for %s/%s: %s", source, need.subject, need.kind, e)
                continue
        tally.collect(result)
        if result.questions:
            # Its verdict is now this run's own submission to make --
            # `pending` (once submit() below settles the batch's fate)
            # accounts for it; counting it here too would double it
            # against `available`.
            tally.pending_candidates += 1
        else:
            tally.attempted += int(result.attempted)
        if improved(before, current_best_of(ctx, need.subject, need.kind)):
            tally.improved += 1
    return 0


def run(ctx: Sourcing, budgets: Mapping[str, Budget], *,
        sentence_targets_per_run: int = 40) -> RunReport:
    """One pass: resolve, adopt, draft sentences once, try each queued
    need at its next source, submit everything collected as one batch.
    An unreachable judge -- at the resolve, in an attempt, or at the
    submit -- ends the pass there, reported and persisted; a batch still
    unanswered ends it before any attempt, so at most one batch is out.
    A drafter transport failure counts under source_failures["llm-sentence"]
    and defers every word with an open Target; the loop runs.
    """
    # One clock read for the whole run (spec 3 r19 section 6a/9): every
    # queue build and the attempt loop's own next_source calls age a
    # `nothing` row against this same instant. The attempts read it back
    # through `ctx.now_ns`, which this run overrides for its duration
    # (restored in `finally`) instead of letting them re-read the clock.
    now_ns = time_ns()
    original_now_ns = ctx.now_ns
    ctx.now_ns = lambda: now_ns
    try:
        return _run_pass(ctx, budgets, now_ns, sentence_targets_per_run=sentence_targets_per_run)
    finally:
        ctx.now_ns = original_now_ns


def _run_pass(ctx: Sourcing, budgets: Mapping[str, Budget], now_ns: int, *,
             sentence_targets_per_run: int) -> RunReport:
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
            needs = _needs(ctx, now_ns=now_ns)
            pending = _pending_needs(ctx, previous[1])
            return _finish(ctx, tally, needs, batch_id=previous[0], pending=pending,
                           extra_deferred=_unconsidered(needs, pending))
    tally.sentences_adopted += _adopt_sentences(ctx)
    if still_out is not None:
        # The run ended here, before it ever looked at a need: every
        # available need this run never considered (not even pending, in
        # a still-earlier batch) is deferred, not lost.
        needs = _needs(ctx, now_ns=now_ns)
        pending = _pending_needs(ctx, still_out[1])
        return _finish(ctx, tally, needs, batch_id=still_out[0], pending=pending,
                       extra_deferred=_unconsidered(needs, pending))

    # (subject, kind) needs a resolve-time preference question already
    # named -- the only questions collected this early. Whichever bucket
    # each lands in below (pending or preferences), the loop must not
    # also attempt that same need: run.RunReport's one-bucket-per-need
    # rule. Keyed by (subject, kind), the key derivations.pending() reads
    # the outstanding batch's own questions under: a word's other
    # still-open needs (its recording, say) are never held back by its
    # picture's question, whichever of the two named it.
    collected_at_resolve = frozenset(
        (q.question.subject, q.question.kind) for q in tally.questions)

    # An open Target's need is its word's, one however many Targets that
    # word has (derivations.available_needs), and that is the unit every
    # bucket below counts it in -- `sentence_targets_per_run` caps the
    # Targets the attempt is handed, never the needs.
    open_words_before = open_words(ctx.syllabus)
    sentence_budget = budgets.get("llm-sentence")
    if sentence_budget is None or not sentence_budget.exceeded_by(
            _spent_on("llm-sentence", carried, tally)):
        try:
            result = sentence_attempt(ctx, max_targets=sentence_targets_per_run)
            tally.collect(result)
        except JudgeUnreachable:
            tally.unreachable = True
            # The drafts were asked for; the judge died at the check.
            tally.attempted += len(open_words_before)
            needs = _needs(ctx, collected_at_resolve, now_ns=now_ns)
            _fold_unsubmitted(tally, collected_at_resolve, available_need_keys(ctx.syllabus))
            # The judge died before the loop ran at all: every queued need
            # behind it was never looked at this run.
            tally.deferred += len(needs.entries)
            return _finish(ctx, tally, needs, batch_id=None, pending=0)
        except TransportError as e:
            tally.source_failures["llm-sentence"] = (
                tally.source_failures.get("llm-sentence", 0) + 1)
            tally.deferred += len(open_words_before)
            _log.warning("drafter llm-sentence failed: %s", e)
            result = None
        if result is not None:
            # An inline transport answers inside that attempt: what it
            # verified there is adoptable in this same run.
            tally.sentences_adopted += _adopt_sentences(ctx)
            # No Source is asked per open Target -- the sentence attempt is
            # what serves them. A word whose Targets it was handed is
            # `attempted`; one it withheld at the no-fit cap (spec 3 r19
            # section 5, AttemptResult.subjects_exhausted) is `exhausted`,
            # once, and never also deferred; one it never reached (the
            # per-run Target cap) is `deferred`; one the adopted drafts
            # closed has already left gaps() and takes no bucket at all.
            open_words_after = open_words(ctx.syllabus)
            served = result.subjects_handed | result.subjects_exhausted
            tally.attempted += len(result.subjects_handed & open_words_after)
            tally.exhausted += len(result.subjects_exhausted & open_words_after)
            tally.deferred += len((open_words_before - served) & open_words_after)
    else:
        # The llm-sentence budget kept the attempt from running at all:
        # every word with an open Target is budget-constrained, same as a
        # per-need Source skip below. The per-Target cap bounds one
        # attempt's own hand-over and applies only when the attempt runs.
        tally.budgeted += len(open_words_before)

    needs = _needs(ctx, collected_at_resolve, now_ns=now_ns)
    # One snapshot of the need keys `available` counts, read here beside
    # the queue itself: `available`, `pending` and `preferences` are all
    # measured against the same list of needs, whatever the attempts
    # below then close.
    avail = available_need_keys(ctx.syllabus)
    unreached = _try_each_need(ctx, needs.entries, budgets, carried, tally, now_ns=now_ns)
    if tally.unreachable:
        # The dead judge stopped the loop where it stood: every queued
        # need past that point was never looked at this run.
        tally.deferred += unreached
        _fold_unsubmitted(tally, collected_at_resolve, avail)
        return _finish(ctx, tally, needs, batch_id=None, pending=0)

    try:
        batch_id = ctx.assessor.submit(tally.questions)
    except JudgeUnreachable:
        tally.unreachable = True
        _fold_unsubmitted(tally, collected_at_resolve, avail)
        return _finish(ctx, tally, needs, batch_id=None, pending=0)
    if batch_id is None:
        # Nothing collected (submit() is a no-op on an empty batch): the
        # same fold, over counts that are both zero unless a question was
        # collected -- and a collected question is always submitted.
        _fold_unsubmitted(tally, collected_at_resolve, avail)
        pending = 0
    else:
        # One bucket per need, decided against the `avail` snapshot taken
        # above beside the queue -- the same list `available` counts --
        # and keyed the way that list is keyed: a resolve-time question on
        # a need `available` still counts (a learner rejection with no
        # acceptable floor, kept open even as a new candidate joined its
        # passing set) is pending; one whose need has left `available` is
        # a preference, outside the identity and never counted into
        # `pending` as well. A word's satisfied picture and its open
        # recording are two need keys, and land in the two different
        # buckets. Each bucket counts needs, one per (subject, kind)
        # however many questions that need carries.
        pending = len({(q.question.subject, q.question.kind)
                       for q in tally.questions} & avail)
        tally.preferences = sum(1 for need in collected_at_resolve if need not in avail)
    return _finish(ctx, tally, needs, batch_id=batch_id, pending=pending)


def _fold_unsubmitted(tally: _Tally, collected_at_resolve: frozenset[tuple[str, str]],
                      avail: frozenset[tuple[str, str]]) -> None:
    """Where a question this run collected but never sent lands. A need
    whose own attempt raised one was asked: `attempted`. A resolve-time
    question is measured against `avail`, the same snapshot the submitted
    path uses -- a need `available` still counts was never attempted at
    all (`deferred`, its question collected again next run), one whose
    need has left `available` is a preference, outside the identity.
    """
    tally.attempted += tally.pending_candidates
    tally.pending_candidates = 0
    tally.deferred += sum(1 for need in collected_at_resolve if need in avail)
    tally.preferences += sum(1 for need in collected_at_resolve if need not in avail)


def _finish(ctx: Sourcing, tally: _Tally, needs: QueuedNeeds, *, batch_id: str | None,
            pending: int, extra_deferred: int = 0) -> RunReport:
    """The run's outcome, as one durable row and one return value.
    `deferred` is `tally.deferred` plus `extra_deferred`, the latter
    nonzero only when the run ended before looking at any need.
    """
    report = RunReport(
        attempted=tally.attempted, improved=tally.improved,
        exhausted=needs.exhausted + tally.exhausted, available=needs.available,
        pending=pending, sentences_adopted=tally.sentences_adopted,
        drafted=tally.drafted, excluded=tally.excluded, excluded_items=tally.excluded_items,
        unreachable=tally.unreachable,
        batch_id=batch_id, source_failures=tally.source_failures, spend=tally.spend,
        unserved=needs.unserved, budgeted=tally.budgeted,
        deferred=tally.deferred + extra_deferred, preferences=tally.preferences)
    _persist_report(ctx.db, report)
    return report


def _persist_report(record: RecordWriter, report: RunReport) -> None:
    """One summary row per run() call (port="run", backend="runreport") so
    a run's own outcome has a durable source -- RunReport itself is only
    ever an in-memory return value. `key` is a constant label; the `cache`
    table's primary key is (key_sha, ts), so every call lands its own row.
    """
    record.append(
        port="run", backend="runreport", key=RunReportKey(), subject="run",
        question={"kind": "runreport"},
        answer={"attempted": report.attempted, "improved": report.improved,
                "exhausted": report.exhausted, "available": report.available,
                "pending": report.pending, "sentences_adopted": report.sentences_adopted,
                "drafted": report.drafted,
                "excluded": report.excluded,
                "excluded_items": [dict(item) for item in report.excluded_items],
                "unreachable": report.unreachable,
                "batch_id": report.batch_id, "source_failures": dict(report.source_failures),
                "spend": {name: {"asks": s.asks, "cost": s.cost}
                          for name, s in report.spend.items()},
                "unserved": report.unserved, "budgeted": report.budgeted,
                "deferred": report.deferred, "preferences": report.preferences},
        cost=sum(s.cost for s in report.spend.values()))
