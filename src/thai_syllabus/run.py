"""The batch run (spec 3 section 7): the previous run's judge batch
resolved and what it passed adopted, one comment pass reading every
unread learner comment (spec 3 r30 section 5), one sentence attempt over
the open Targets, one phrase attempt drafting a search phrase for every
open picture need lacking one (spec 3 r24 section 5), one Source per
queued need, and every question collected on the way submitted as one
batch.

Iteration only: every policy (queue/current_best/exhausted/next_source,
what a picture still owes a preference question, what an attempt is for a
kind) lives in derivations.py and attempts.py. Escalation to the next
source happens on the next run, for every transport alike; every ask()
appended its own checkpoint before this loop saw it (spec 2).
"""
from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from time import time_ns
from typing import NamedTuple

from .assessor import AssessQuestion, JudgeUnreachable, PreparedQuestion
from .authority import role_for
from .cachekeys import RunReportKey
from .attempts import (
    AttemptResult,
    Need,
    Sourcing,
    Spend,
    adjudication_attempt,
    assess_first,
    attempt,
    comment_attempt,
    current_best_of,
    draft_refusal,
    phrase_attempt,
    picture_query_for,
    preference_attempt,
    provenance_source_for,
    retire_sentence,
    sentence_attempt,
)
from .derivations import (
    QueuedNeeds,
    QueueEntry,
    adjudications,
    adoptable_drafts,
    available_need_keys,
    directed,
    improved,
    next_source,
    open_words,
    queued,
    role_of,
)
from .entities import Pronunciation, Sentence, Word
from .ids import WordId
from .phonology import corroborates, default_engines
from .ports import RecordWriter
from .record import (
    asks_since,
    draft_sentence,
    fetches_since,
    ratings_for_role,
    retired_texts,
    rows_for,
    sentence_drafts,
    spend_since,
)
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
    # words whose adjudicated pronunciation this run wrote to words.yaml
    # (spec 3 r28 section 5) -- an event, not a need: a Word is curated
    # data, so it sits outside the identity above
    adjudicated: int = 0
    # verdicts this run checked that no engine corroborated, so the words
    # stay disputed (spec 3 r29) -- an event count beside `adjudicated`,
    # also outside the identity
    stayed_disputed: int = 0
    drafted: int = 0            # drafts the sentence attempt produced
    # adopted Sentences the run deleted, whatever caused it (spec 3 r30
    # section 7): a recording need exhausted with no passing candidate
    # (F13, spec 3 section 5) -- a learner row that outlives the rule (F9)
    # keeps the sentence instead -- or a learner comment's retire_sentence
    retired: int = 0
    # the comment pass (spec 3 r30 section 5): comments read this run,
    # actions taken, requests the deck could not act on -- events, not
    # needs, outside the identity above
    comments_read: int = 0
    comment_actions: int = 0
    comment_unactionable: int = 0
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
    # available needs this run never considered; also a queued need whose
    # own sentence this same pass already retired (F13) -- the queue was
    # built before the retirement, so its remaining entries (e.g. the
    # retired sentence's scene picture) are skipped rather than attempted
    deferred: int = 0
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
    adjudicated: int = 0
    stayed_disputed: int = 0
    drafted: int = 0
    retired: int = 0
    comments_read: int = 0
    comment_actions: int = 0
    comment_unactionable: int = 0
    # subjects (text_shas) of sentences this pass retired -- a later
    # queue entry naming one of these is skipped, never attempted
    # (run._try_each_need)
    retired_subjects: set[str] = field(default_factory=set)
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
        # The comment pass's own counts (spec 3 r30 section 5): every
        # other attempt leaves these at 0. `retired` is a count, not a
        # bucket -- _retire_exhausted_sentence adds F13's own the same way.
        self.retired += result.retired
        self.comments_read += result.comments_read
        self.comment_actions += result.comment_actions
        self.comment_unactionable += result.comment_unactionable
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


def _recover_orphaned_drafts(ctx: Sourcing) -> AttemptResult:
    """D2 (spec 3 r30 section 5): every sentence draft on record
    (record.sentence_drafts) that is neither adopted nor retired, asked
    again -- cache-first through `ctx.assessor.ask_many`, whose JudgeKey
    carries the rubric sha, so a draft already holding a fresh
    sentence-for-target verdict under the current rubric collects
    nothing and only an unjudged one does.

    A batch that never came back orphans whatever drafts it carried, a
    drafting ask's and a comment's replacement alike, and neither pass
    re-raises the question on its own: the drafting ask is cached by
    prompt (a re-ask is a hit, and the drafter may well answer with
    other texts), and a comment is read once. Without this the draft
    would sit on record for ever, neither adopted nor refused.

    The question is the one `sentence_attempt` (and
    `attempts._draft_replacement`) raises for a draft: role
    sentence-for-target, the current rubric, `{text, gloss, word}` with
    the sentence's own last used word. A draft is asked about only when
    it could still be adopted: `attempts.draft_refusal`, the one
    acceptance test both of those passes apply (the Sentence invariant,
    the clause cap, at least one still-open Target filled, the
    per-sentence Target cap), decides, so the judge is never asked about
    a draft the run would refuse anyway -- a curated change since it was
    drafted, or its Targets filled in the meantime. A refusal is a
    routine, permanent fact about the draft, logged at debug. Drafts are
    not needs: the questions ride this run's batch like the sentence
    attempt's, and no bucket counts them.
    """
    adopted = {s.text_sha for s in ctx.syllabus.sentences}
    retired = retired_texts(ctx.db)
    role = role_for("sentence")
    questions: list[AssessQuestion] = []
    for draft in sentence_drafts(ctx.db):
        if draft.text_sha in adopted or draft.text_sha in retired or not draft.gloss:
            continue
        sentence = draft_sentence(draft, ctx.today)
        refusal = draft_refusal(ctx, sentence)
        if refusal is not None:
            _log.debug("orphaned draft not asked about: %s: %s", refusal, draft.text)
            continue
        try:
            last_word = ctx.syllabus.word(ctx.syllabus.last_used_word(sentence)).thai
        except (KeyError, ValueError) as e:
            _log.debug("orphaned draft not asked about: %s: %s", e, draft.text)
            continue
        questions.append(AssessQuestion(
            subject=draft.text_sha, role=role, artifact_sha=None, rubric=ctx.rubrics[role],
            params={"text": draft.text, "gloss": draft.gloss, "word": last_word},
            kind="sentence", subject_kind="sentence"))
    if not questions:
        return AttemptResult(attempted=False)
    result = ctx.assessor.ask_many("judge", questions)
    spend: dict[str, Spend] = {}
    for verdict in result.resolved.values():
        spend.setdefault("judge", Spend()).add(0 if verdict.hit else 1,
                                               float(verdict.cost or 0.0))
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


class _Adjudicated(NamedTuple):
    """What one adjudication pass came to: `written` words whose
    pronunciation reached words.yaml, `stayed_disputed` verdicts no
    engine corroborated (spec 3 r29). Both are RunReport fields.
    """
    written: int = 0
    stayed_disputed: int = 0


def _materialize_adjudications(ctx: Sourcing) -> _Adjudicated:
    """Spec 3 r28 section 5: every adjudicated pronunciation the engines
    corroborate becomes the Word's, corroboration `adjudicated`, written
    to curated words.yaml under this writing command; the rest stay
    disputed, logged, and counted as `stayed_disputed` (r29). Engines
    load lazily, only when there is a verdict to check, so a run with
    nothing to materialize never pulls in pythainlp/torch.

    words.yaml is rewritten whole, from the same (Word, category) rows
    the loader produced, in their own order -- only the adjudicated
    Words' `pron` differs, so no row is added, dropped or moved (spec 2
    section 6's Guard counts the rows).
    """
    found = adjudications(ctx.db, ctx.syllabus, current_rubric=ctx.rubrics)
    if not found or ctx.curated_dir is None:
        return _Adjudicated()
    engines = ctx.engines or default_engines()
    updated: dict[WordId, Word] = {}
    not_corroborated = 0
    for word_id, syllables in found.items():
        w = ctx.syllabus.word(word_id)
        if corroborates(syllables, w.thai, engines):
            updated[word_id] = dataclasses.replace(
                w, pron=Pronunciation(syllables=syllables, corroboration="adjudicated"))
        else:
            not_corroborated += 1
            _log.info("adjudication of %s (%s) not corroborated by an engine; stays disputed",
                      word_id, w.thai)
    if not_corroborated:
        _log.warning(
            "adjudication: %d of %d verdicts not corroborated by an engine; "
            "the words stay disputed", not_corroborated, len(found))
    if not updated:
        return _Adjudicated(stayed_disputed=not_corroborated)
    # Deferred: curated.py reads parse_day_starts from this module, so a
    # top-level import here would be a cycle.
    from .curated import save_words

    rows = [(updated.get(w.id, w), ctx.syllabus.category_of(w.id)) for w in ctx.syllabus.words]
    save_words(ctx.curated_dir / "words.yaml", rows)
    ctx.syllabus = ctx.syllabus.with_words(tuple(w for w, _c in rows))
    return _Adjudicated(written=len(updated), stayed_disputed=not_corroborated)


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
    # Picture keys only: preference_attempt ranks a word's passing
    # PICTURES (derivations.pictures_awaiting_preference), so only a
    # subject whose picture the batch just judged can have opened one.
    # Availability is not the filter -- the whole point of the
    # `preferences` bucket is a picture need that has since left
    # `available` -- the artifact kind is. Without this, every word the
    # adjudication ask put in the batch for its pronunciation (spec 3
    # r28: not a need at all) would be folded over here and could raise
    # a preference question nothing asked for.
    preference = preference_attempt(ctx, sorted(
        {subject for subject, kind in batch_needs if kind == "picture"}))
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


def _learner_outlives_the_rule(ctx: Sourcing, need: Need) -> bool:
    """F9: a learner row on this need outlives a rule change (spec 3
    section 5's retirement carve-out), so the sentence is kept: a rating
    row under the need's own role naming "unacceptable-use-this"
    (record.ratings_for_role -- a nomination, never a veto, spec 3
    section 4 r8), a provide row the learner supplied directly
    (reviewserver.append_supply's local-path shape, backend "learner"),
    or the sentence being directed (derivations.directed -- a learner
    direction row, an unconsumed reverify row, or a card-flag row): a
    directed exhausted sentence is kept, not retired, so the feedback
    screen can still show it. A URL supply's own provide row carries
    whatever backend fetched it, not "learner", so its nominating rating
    row is what this reads there; the provide-row check catches a
    local-path supply's row too, on its own, before append_supply's
    rating row is even considered.
    """
    rows = rows_for(ctx.db, need.subject, need.kind)
    role = role_of(ctx.db, need.subject, need.kind, rows)
    if any(r.answer.get("value") == "unacceptable-use-this"
           for r in ratings_for_role(ctx.db.assessments_of(need.subject), role)):
        return True
    if any(r.port == "provide" and r.backend == "learner" for r in rows):
        return True
    return directed(ctx.db, need.subject)


def _retire_exhausted_sentence(ctx: Sourcing, need: Need, tally: _Tally) -> None:
    """F13 (spec 3 section 5, docs/principles.md): `need` is a sentence's
    recording need already known exhausted (no source left,
    derivations.next_source). Retires the sentence unless a candidate
    still holds a passing verdict (derivations.current_best is not None)
    or a learner row outlives the rule (_learner_outlives_the_rule, F9):
    deletes the sentences row (store.SyllabusDb.delete_sentence -- its
    drafts stay in the record), reports the removal to the writing
    command's Guard when run() was threaded one (safety.writing_command),
    logs it, and counts it in RunReport.retired -- an event outside the
    run's needs-partition identity (spec 3 section 7): the need itself
    still landed in whichever bucket found it exhausted, and the sentence's
    own needs leave `available` only on the next run, once it is gone.

    The retirement itself -- the retirement row, the delete, the Guard
    report and the `ctx.syllabus` replacement -- is
    attempts.retire_sentence, the one mechanism the comment pass (r30)
    retires through too; F13's own contribution is the two guards above,
    the reason it passes ("recording exhausted", with no replacement
    hint and no comment it was derived from), and the tally:
    `need.subject` is added to `tally.retired_subjects` so
    `_try_each_need` can skip that sentence's other still-queued needs
    (its scene picture) rather than attempt them against gone data, and
    `tally.retired` counts every deleted adopted Sentence, whatever
    caused it.
    """
    if current_best_of(ctx, need.subject, need.kind).artifact_sha is not None:
        return
    if _learner_outlives_the_rule(ctx, need):
        return
    retire_sentence(ctx, need.subject, reason="recording exhausted")
    tally.retired_subjects.add(need.subject)
    tally.retired += 1


def _maybe_retire_exhausted_sentence(ctx: Sourcing, need: Need, tally: _Tally) -> None:
    """Whether `need` (just tried, or already known exhausted before any
    attempt this run) is a sentence's recording need with no source left
    -- the trigger _retire_exhausted_sentence's own F13 check applies to.
    Every other need is untouched.
    """
    if need.kind != "recording" or need.subject_kind != "sentence":
        return
    sources = ctx.sources_for(need.kind)
    if next_source(ctx.db, need.subject, need.kind, sources, transient_cap=ctx.transient_cap,
                   nothing_ttl=ctx.nothing_ttl, now_ns=ctx.now_ns()) is not None:
        return
    _retire_exhausted_sentence(ctx, need, tally)


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
    than re-read here. A remaining entry naming a sentence this same pass
    already retired (F13, tally.retired_subjects) is skipped, counted
    `deferred` -- the queue was built before the retirement, so a still-
    queued sibling need (e.g. the retired sentence's scene picture) is
    never attempted against the now-deleted sentence.
    """
    dead_sources: set[str] = set()
    budgeted_sources: set[str] = set()
    for index, entry in enumerate(entries):
        need = Need(entry.subject, entry.kind, entry.subject_kind)
        if need.subject_kind == "sentence" and need.subject in tally.retired_subjects:
            tally.deferred += 1
            continue
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
            if need.kind == "picture" and picture_query_for(ctx, need) is None:
                # Spec 3 r25 section 5: no direction, suggestion or
                # drafted phrase on record -- nothing to search. The
                # need waits for the phrase ask (this run's failed, or
                # its answer omitted this subject) and counts deferred.
                tally.deferred += 1
                continue
            sources = ctx.sources_for(need.kind)
            source = next_source(ctx.db, need.subject, need.kind, sources,
                                transient_cap=ctx.transient_cap,
                                nothing_ttl=ctx.nothing_ttl, now_ns=now_ns)
            if source is None:
                tally.exhausted += 1
                if need.kind == "recording" and need.subject_kind == "sentence":
                    _retire_exhausted_sentence(ctx, need, tally)
                continue
            if source in dead_sources:
                # Spec 3 r26 section 7: a source dead for the run counts
                # as tried for this pass only -- the need takes its next
                # live source now, and nothing about the dead one reaches
                # the record. No live source left: deferred (no row was
                # written for this need; the next run starts over).
                live = [s for s in sources if s not in dead_sources]
                source = next_source(ctx.db, need.subject, need.kind, live,
                                    transient_cap=ctx.transient_cap,
                                    nothing_ttl=ctx.nothing_ttl, now_ns=now_ns)
                if source is None:
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
        _maybe_retire_exhausted_sentence(ctx, need, tally)
    return 0


def run(ctx: Sourcing, budgets: Mapping[str, Budget], *,
        sentence_targets_per_run: int = 40) -> RunReport:
    """One pass: resolve, adopt, re-ask the judge about any draft a lost
    batch orphaned (spec 3 r30 section 5), read every unread learner
    comment once, draft sentences once, draft an image phrase once for
    every open picture need lacking one (spec 3 r24 section 5), try each
    queued need at its next source, submit everything collected as one
    batch. An unreachable judge -- at the resolve, in a pass, in an
    attempt, or at the submit -- ends the pass there, reported and
    persisted; a batch still unanswered ends it before any attempt, so at
    most one batch is out. A drafter transport failure counts under
    source_failures["llm-sentence"] and defers every word with an open
    Target; one under source_failures["llm-phrase"] leaves every picture
    need still lacking a phrase waiting (deferred) for this pass; the
    comment reader's (or its parse ask's) counts under
    source_failures["llm-comment"] and leaves the comments unread until
    the next run. Either way the loop runs.
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
    materialized = _materialize_adjudications(ctx)
    tally.adjudicated += materialized.written
    tally.stayed_disputed += materialized.stayed_disputed
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

    # D2 (spec 3 r30 section 5), before either drafting pass: a draft on
    # record that no verdict ever came back for -- a lost batch orphans a
    # drafter's and a comment's replacement alike -- has its own judge
    # question collected again, cache-first, so a draft already judged
    # under the current rubric collects nothing. Drafts are not needs: no
    # bucket, `pending` unaffected. A dead judge here ends the run before
    # any Target was handed, the same as the comment pass below.
    try:
        tally.collect(_recover_orphaned_drafts(ctx))
    except JudgeUnreachable:
        return _judge_died_before_the_loop(ctx, tally, collected_at_resolve, now_ns=now_ns)

    # One reading ask per run (spec 3 r30 section 5) over every unread
    # learner comment, before the sentence attempt: a retirement it makes
    # reopens Targets the drafter is handed this same run, so `available`
    # (read beside the queue below) and the buckets agree. Its
    # replacement drafts' judge questions ride this run's batch like the
    # sentence attempt's (drafts are not needs: no bucket). A reader or
    # parser transport failure counts under source_failures and otherwise
    # never breaks the run; a dead judge at a replacement's check ends it,
    # every open Target and queued need deferred.
    #
    # That check comes after the pass has already written its retirement,
    # direction, rating and reading rows, so the attempt reports the dead
    # judge (AttemptResult.judge_unreachable) instead of raising: the
    # counts are collected into the tally FIRST -- the report must say
    # what was deleted and read -- and only then does the run take the
    # judge-death path. A judge death before anything was written still
    # raises and lands on the same path with the counts at zero.
    try:
        comments = comment_attempt(ctx)
    except JudgeUnreachable:
        return _judge_died_before_the_loop(ctx, tally, collected_at_resolve, now_ns=now_ns)
    except TransportError as e:
        tally.source_failures["llm-comment"] = tally.source_failures.get("llm-comment", 0) + 1
        _log.warning("comment reader llm-comment failed: %s", e)
    else:
        tally.collect(comments)
        if comments.judge_unreachable:
            return _judge_died_before_the_loop(ctx, tally, collected_at_resolve, now_ns=now_ns)

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

    # One drafting ask per run (spec 3 r24 section 5), after sentence
    # drafting and before the need loop: every open picture need -- word
    # or scene, this pass's newly adopted sentences included -- lacking a
    # drafted phrase gets one, so picture_query_for finds it below. No
    # need bucket of its own: a picture need this drafts nothing for
    # waits in the loop (r25: no query on record, deferred). A drafter
    # transport failure counts under source_failures["llm-phrase"] and
    # otherwise never breaks the run.
    try:
        tally.collect(phrase_attempt(ctx))
    except TransportError as e:
        tally.source_failures["llm-phrase"] = tally.source_failures.get("llm-phrase", 0) + 1
        _log.warning("drafter llm-phrase failed: %s", e)

    # One adjudication ask per run (spec 3 r28 section 5): every word
    # still lacking a corroborated pronunciation. No need bucket: words
    # are not needs; the questions ride this run's batch and
    # _materialize_adjudications reads the verdicts back next run.
    try:
        tally.collect(adjudication_attempt(ctx))
    except JudgeUnreachable:
        tally.unreachable = True
        needs = _needs(ctx, collected_at_resolve, now_ns=now_ns)
        _fold_unsubmitted(tally, collected_at_resolve, available_need_keys(ctx.syllabus))
        tally.deferred += len(needs.entries)
        return _finish(ctx, tally, needs, batch_id=None, pending=0)

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
        batch_id = ctx.assessor.submit(_one_per_key(tally.questions))
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


def _judge_died_before_the_loop(ctx: Sourcing, tally: _Tally,
                                collected_at_resolve: frozenset[tuple[str, str]], *,
                                now_ns: int) -> RunReport:
    """A pass that runs before the sentence attempt met an unreachable
    judge (spec 3 r30 section 5: the comment pass's replacement check,
    the orphaned-draft recovery). Nothing was handed to the drafter and
    the need loop never ran, so every word with an open Target and every
    queued need is deferred, the questions collected at resolve fold
    where `_fold_unsubmitted` puts them, and the identity holds.
    """
    tally.unreachable = True
    needs = _needs(ctx, collected_at_resolve, now_ns=now_ns)
    _fold_unsubmitted(tally, collected_at_resolve, available_need_keys(ctx.syllabus))
    tally.deferred += len(open_words(ctx.syllabus)) + len(needs.entries)
    return _finish(ctx, tally, needs, batch_id=None, pending=0)


def _one_per_key(questions: Sequence[PreparedQuestion]) -> list[PreparedQuestion]:
    """The run's collected questions, one per cache key. `submit` refuses
    two prepared questions sharing a key, and two passes of one run can
    legitimately raise the same one: the D2 recovery collects an orphaned
    draft's judge question, and the drafting ask -- served from its own
    prompt cache, so it answers exactly what it answered before -- hands
    that very draft to `ask_many` again in the same run (spec 3 r30
    section 5). `Assessor.ask_many` already collects a key once within
    one call; this does the same across the run's calls, keeping the
    first, and says once per run how many it coalesced.
    """
    seen: set[str] = set()
    out: list[PreparedQuestion] = []
    for q in questions:
        encoded = q.key.encode()
        if encoded in seen:
            continue
        seen.add(encoded)
        out.append(q)
    if len(out) < len(questions):
        _log.info("two passes collected the same judge question %d time(s) this run; "
                  "each is submitted once", len(questions) - len(out))
    return out


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
        adjudicated=tally.adjudicated, stayed_disputed=tally.stayed_disputed,
        drafted=tally.drafted, retired=tally.retired,
        comments_read=tally.comments_read, comment_actions=tally.comment_actions,
        comment_unactionable=tally.comment_unactionable,
        excluded=tally.excluded, excluded_items=tally.excluded_items,
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
                "adjudicated": report.adjudicated,
                "stayed_disputed": report.stayed_disputed,
                "drafted": report.drafted, "retired": report.retired,
                "comments_read": report.comments_read,
                "comment_actions": report.comment_actions,
                "comment_unactionable": report.comment_unactionable,
                "excluded": report.excluded,
                "excluded_items": [dict(item) for item in report.excluded_items],
                "unreachable": report.unreachable,
                "batch_id": report.batch_id, "source_failures": dict(report.source_failures),
                "spend": {name: {"asks": s.asks, "cost": s.cost}
                          for name, s in report.spend.items()},
                "unserved": report.unserved, "budgeted": report.budgeted,
                "deferred": report.deferred, "preferences": report.preferences},
        cost=sum(s.cost for s in report.spend.values()))
