"""What an attempt is for each need (spec 3 section 5): one Source asked
under the need's own subject, whatever it returns ingested, the speaker
it came from recorded, and the judge questions the run will ask
collected.

A Need is an artifact kind ("picture", "recording", "rendition") and the
kind of thing its subject is; a sentence's scene picture and reading are
those same kinds under a sentence subject, which is what puts them in
their own Assess roles (authority.role_for).

An attempt appends and nothing else: current-best, improved, pending,
exhausted and what there is to adopt are derivations.py's folds over the
rows these asks append, and run.py drives the loop.

Under an inline judge transport every collected question resolves inside
the call and `questions` comes back empty; under a batch transport the
misses come back in `questions` for the run to submit, and a question
that could not be prepared is in `excluded`. An unreachable judge raises
JudgeUnreachable out of ask_many and stops the run.
"""
from __future__ import annotations

import functools
import json
import time
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

from . import ipa, record
from .assessor import (UNTRUSTED, AssessQuestion, Assessor, Excluded, JudgeUnreachable,
                       PreparedQuestion, deck_field)
from .authority import role_for
from .cachekeys import (AttemptOutcomeKey, CommentReadingKey, DirectionKey, PhraseKey, ProvideKey,
                        RenditionAskKey, RetirementKey, rendition_identity, sha)
from .compile import card_meaning
from .derivations import (
    DEFAULT_ATTEMPT_CAP,
    DEFAULT_SENTENCE_NOTHING_CAP,
    DEFAULT_TRANSIENT_CAP,
    CurrentBest,
    aged_out,
    available_needs,
    current_best,
    passing_pictures,
    pictures_awaiting_preference,
    refused_drafts,
    sentence_exhausted,
    unjudged_candidates,
)
from .entities import Clauses, Target, Word, clauses_to_json, element_word, is_corroborated
from .ids import PairId, WordId
from .learner import ACTION_RATINGS, CommentRef, append_direction, append_rating
from .media import Speaker
from .phonology import Engines
from .provider import Provider, ProviderAnswer, Question, forvo_limit_body
from .record import (COMMENT_PROMPT_VERSION, COMMENT_SUBJECT, DRAFT_SUBJECT, PARSE_SUBJECT,
                     PHRASE_SUBJECT)
from .safety import Guard
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus
from .transport import FetchRefused, QuotaExhausted, SynthesisRefused, TransportError
from .tts import FEMALE_VOICES, MALE_VOICES, pick_voice

__all__ = ["Need", "Sourcing", "Spend", "AttemptResult", "SOURCES", "SubjectKind",
           "VoiceConstraint",
           "sources_for", "provenance_source_for", "current_best_of",
           "attempt", "assess_first", "sentence_attempt", "preference_attempt",
           "phrase_attempt", "picture_query_for", "adjudication_attempt", "retire_sentence",
           "comment_attempt", "COMMENTS_PER_ASK", "draft_refusal",
           "DEFAULT_SENTENCE_MAX_CLAUSES", "DEFAULT_SENTENCE_INTRODUCIBLE_PER_ASK",
           "DEFAULT_SENTENCE_TARGETS_PER_SENTENCE"]

_log = logging.getLogger(__name__)

# The drafting prompt's own clause cap default (spec 3 r23 section 5/8): a
# sentence over this many clauses outruns the 5 s recording duration cap,
# so the cure is at drafting -- Sourcing.sentence_max_clauses, wired from
# providers.yaml's own sentence_max_clauses (wiring.build_sourcing).
DEFAULT_SENTENCE_MAX_CLAUSES = 2

# sentence_attempt's own cap default (spec 3 r24 section 5/8) on how many
# sentence-introduced, unmet Targets one drafting ask is handed -- a batch
# dominated by introducible targets (e.g. 36 of 40) starved the drafter of
# room for the receptive backlog and yielded almost nothing adopted. Wired
# from providers.yaml's own sentence_introducible_per_ask (wiring.build_sourcing).
DEFAULT_SENTENCE_INTRODUCIBLE_PER_ASK = 5
# spec 3 r27 section 5/8: the most open Targets one drafted sentence may
# fill -- more is a word list in disguise, the thing sentences exist to
# avoid; met words beyond that are filler and free.
DEFAULT_SENTENCE_TARGETS_PER_SENTENCE = 3

# The comment pass's own cap (spec 3 r30 section 5) on how many unread
# comments one reading ask is handed, oldest first: one prompt carries a
# full card's worth of facts per comment, and a backlog of hundreds would
# be one unreadable ask. A comment held back keeps no reading row, so the
# next run hands it. Not config: the prompt's shape decides it, the way
# the sentence attempt's own caps above do.
COMMENTS_PER_ASK = 40

# Cheapest source first, per ARTIFACT kind (spec 3 section 5). A sentence's
# own recording and scene picture are the same artifact kinds a word's are;
# only the subject differs.
SOURCES: dict[str, tuple[str, ...]] = {
    # spec 3 r26 section 5: the keyed corpus first, the challenge-prone
    # anonymous-tier corpus second
    "picture": ("pexels", "openverse", "wikimedia"),
    "recording": ("forvo", "tts"),
    "rendition": ("forvo", "tts"),
}

# Forvo's own sex codes, and the Speaker vocabulary they map onto.
_FORVO_SEX = {"m": "male", "f": "female"}


def sources_for(kind: str) -> tuple[str, ...]:
    return SOURCES.get(kind, ())


SubjectKind = Literal["word", "pair", "grapheme", "sentence"]

# A recording's or rendition's voice constraint (spec 1 section 1 (r10);
# spec 3 section 5): "male"/"female" admit that sex's own pool alone,
# "any" admits both.
VoiceConstraint = Literal["male", "female", "any"]

# The three outcome values an attempt-outcome row's answer["outcome"] carries.
Outcome = Literal["candidates", "nothing", "transient-failure"]


@dataclass(frozen=True)
class Need:
    """(subject, artifact kind) plus what the subject is: a word's picture
    and a sentence's scene picture are both kind "picture", and the
    subject kind is what decides the role and the attempt.
    """
    subject: str
    kind: str                          # picture | recording | rendition | grapheme-keyword
    subject_kind: SubjectKind = "word"

    @property
    def role(self) -> str:
        return role_for(self.kind, self.subject_kind)


@dataclass
class Spend:
    """One backend's asks and cost within a run, in that backend's own
    currency (spec 3 section 7). A cache hit adds no ask."""
    asks: int = 0
    cost: float = 0.0

    def add(self, asks: int, cost: float) -> None:
        self.asks += asks
        self.cost += cost


@dataclass
class Sourcing:
    """Everything an attempt needs; built by wiring.build_sourcing."""
    syllabus: Syllabus
    provider: Provider
    assessor: Assessor
    db: SyllabusDb                     # RecordWriter + CacheReader + add_media/add_sentence
    media_store: MediaStore
    rubrics: Mapping[str, str]         # role -> rubric text (rulebook.rubrics_for)
    provenance_prior: Sequence[str] = ()
    image_candidates: int = 5
    today: Callable[[], date] = date.today
    # Voice pools per sex; a recording draws from the pool its voice
    # constraint allows (E2, E7).
    voices: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: {
        "male": tuple(MALE_VOICES), "female": tuple(FEMALE_VOICES)})
    judge_model: str = "llm"
    # The Sources each artifact kind may be asked for, cheapest first, and
    # the attempt count exhausted() stops at.
    sources_for: Callable[[str], Sequence[str]] = field(default=sources_for)
    attempt_cap: int = DEFAULT_ATTEMPT_CAP
    transient_cap: int = DEFAULT_TRANSIENT_CAP
    # Per-source ageing (spec 3 r19 section 6a/9): days after which a
    # `nothing` outcome stops counting as tried, so next_source offers
    # a growing corpus (Forvo) again. A source absent here never ages.
    nothing_ttl: Mapping[str, int] = field(default_factory=dict)
    # The clock aged_out() reads. run() overrides this for the duration of
    # a pass so every attempt reads the same instant the pass's queue was
    # built against, restoring the original callable when the pass ends;
    # the default reads the wall clock afresh for a caller outside a run.
    now_ns: Callable[[], int] = field(default=time.time_ns)
    # The `nothing` outcomes a word's sentence need may carry since its
    # last handed draft before the drafter stops being handed its Targets
    # (spec 3 r19 section 5, derivations.sentence_exhausted).
    sentence_nothing_cap: int = DEFAULT_SENTENCE_NOTHING_CAP
    # The drafting prompt's own clause cap (spec 3 r23 section 5/8): a
    # draft over this many clauses is refused like an invariant failure
    # (sentence_attempt's acceptance loop), never adopted.
    sentence_max_clauses: int = DEFAULT_SENTENCE_MAX_CLAUSES
    # sentence_attempt's own cap (spec 3 r24 section 5/8) on how many
    # sentence-introduced, unmet Targets one drafting ask is handed; the
    # rest of the handed batch is the next non-introduced open Targets.
    sentence_introducible_per_ask: int = DEFAULT_SENTENCE_INTRODUCIBLE_PER_ASK
    sentence_targets_per_sentence: int = DEFAULT_SENTENCE_TARGETS_PER_SENTENCE
    # The writing command's own account of deliberate removals (spec 2
    # section 6, safety.writing_command): threaded onto ctx the same way
    # cli._cmd_run sets it, so run()'s own retirement of an exhausted
    # sentence (F13, spec 3 section 5) can report it. None outside a
    # writing command (e.g. a caller that never wraps run() in one), in
    # which case a retirement is still counted and logged, just not
    # guarded.
    guard: Guard | None = None
    # The deck's curated/ directory (spec 2 section 1), so the run can
    # write back the one curated file it owns a change to: words.yaml,
    # the adjudicated pronunciations (spec 3 r28 section 5,
    # run._materialize_adjudications). None outside a wired deck, in
    # which case an adjudication is derived and nothing is written.
    curated_dir: Path | None = None
    # The pronunciation engines the adjudication check runs against
    # (phonology.Engines). None means the real ones, resolved lazily on
    # the first verdict there is to check -- pythainlp/torch never load
    # for a run with nothing to materialize, and a test injects fakes.
    engines: Engines | None = None


@dataclass(frozen=True)
class AttemptResult:
    """What one attempt did."""
    # a Source ask was made, hit or miss, or assess-first asked the judge
    attempted: bool
    # the judge questions collected for the run's batch, empty inline
    questions: list[PreparedQuestion] = field(default_factory=list)
    # assessor.ManyResult's own dict: question key -> Excluded
    excluded: dict[str, Excluded] = field(default_factory=dict)
    spend: dict[str, Spend] = field(default_factory=dict)
    drafted: int = 0                   # drafts filling an open Target (sentence attempt only)
    # open Targets the sentence attempt handed the drafter, min(open, max_targets)
    targets_handed: int = 0
    # the words those Targets belong to -- one (word, "sentence") need
    # each, which is how the run accounts for them
    subjects_handed: frozenset[str] = frozenset()
    # the words whose open Targets the sentence attempt withheld because
    # their sentence need is exhausted (spec 3 r19 section 5): neither
    # attempted nor deferred -- the run counts them `exhausted`
    subjects_exhausted: frozenset[str] = frozenset()
    # the comment pass (spec 3 r30 section 5): comments read this run,
    # actions taken, requests the deck could not act on (refused actions
    # included); `retired` counts the sentences it deleted -- events,
    # all outside the run's needs identity
    comments_read: int = 0
    comment_actions: int = 0
    comment_unactionable: int = 0
    retired: int = 0
    # the judge could not be reached AFTER this attempt had already
    # written rows (the comment pass's own check of its replacement
    # drafts): the counts above are real and must reach the report, so
    # the attempt returns instead of raising and the run collects the
    # result first, then takes its judge-death path (spec 3 r30 section
    # 5). A death before anything was written still raises.
    judge_unreachable: bool = False


# --- the outcome row (spec 3 section 6; spec 2 section 2) -------------------

@dataclass
class _Fetches:
    """One source's own asks and fetches within one attempt, tracked to
    decide the outcome row spec 3 section 6 defines: `candidates` (at
    least one artifact stored and reached the check), `nothing` (this
    source answered and nothing usable came of it, for a non-transport
    reason), or `transient-failure` (the source's own ask, or every fetch
    it needed, failed on the wire). A partial fetch success with at least
    one candidate is always `candidates`, whatever else failed alongside
    it.
    """
    candidates: list[str] = field(default_factory=list)
    attempts: int = 0
    transient_failures: int = 0
    served_refusals: int = 0
    # Every url handed to imgfetch this attempt, ingested or refused
    # (picture attempts only; record.tried_urls unions this list's
    # outcome-row copies across attempts).
    tried: list[str] = field(default_factory=list)
    # Set by failed(served=True), read by _download_forvo to decide
    # whether a miss re-asks the lookup; cleared by stored() and missed().
    last_refusal_served: bool = False

    def stored(self, sha: str) -> None:
        self.attempts += 1
        self.candidates.append(sha)
        self.last_refusal_served = False

    def missed(self) -> None:
        """A fetch attempted and answered, producing no candidate for a
        non-transport reason."""
        self.attempts += 1
        self.last_refusal_served = False

    def failed(self, *, served: bool = False) -> None:
        """A fetch, or the source's own ask, failed on the wire. `served`
        marks a served refusal (spec 3 section 6a): a typed reason other
        than wire."""
        self.attempts += 1
        self.transient_failures += 1
        self.last_refusal_served = served
        if served:
            self.served_refusals += 1

    @property
    def outcome(self) -> Outcome:
        if self.candidates:
            return "candidates"
        if self.attempts and self.transient_failures == self.attempts:
            return "transient-failure"
        return "nothing"


def _append_outcome(ctx: Sourcing, need: Need, source: str, outcome: Outcome,
                    candidates: Sequence[str], *, tried: Sequence[str] = ()) -> None:
    """One outcome row per (need, source) an attempt asks, after the ask
    and its fetches (spec 3 section 6): the row every derivation over
    next_source/exhausted folds over. `outcome` is "candidates" when at
    least one artifact from it was stored. `tried` is every url a picture
    attempt handed to imgfetch this attempt, ingested or refused; empty
    for a recording or rendition attempt's row.
    """
    ctx.db.append(port="attempt", backend=source,
                  key=AttemptOutcomeKey(subject=need.subject, kind=need.kind, source=source),
                  subject=need.subject,
                  question={"kind": need.kind, "subject_kind": need.subject_kind,
                            "source": source},
                  answer={"outcome": outcome, "candidates": list(candidates),
                          "tried": list(tried)})


# --- reading the record -----------------------------------------------------

def _word_of(ctx: Sourcing, subject: str) -> Word:
    """The Word a need's subject names, refusing by name when there is
    none."""
    word = ctx.syllabus.find_word(WordId(subject))
    if word is None:
        raise ValueError(f"need {subject!r} names no word in the syllabus")
    return word


def provenance_source_for(db: SyllabusDb) -> Callable[[str], str | None]:
    """current_best's provenance_source over `db`: the `media` table's
    own `source` for a sha."""
    def get(sha: str) -> str | None:
        prov = db.media_provenance(sha)
        return prov.get("source") if prov else None
    return get


def current_best_of(ctx: Sourcing, subject: str, kind: str) -> CurrentBest:
    return current_best(ctx.db, subject, kind, current_rubric=ctx.rubrics,
                        prior=ctx.provenance_prior,
                        provenance_source=provenance_source_for(ctx.db))


def _candidate_shas(ctx: Sourcing, need: Need) -> list[str]:
    return record.candidate_shas(record.rows_for(ctx.db, need.subject, need.kind))


# --- spend ------------------------------------------------------------------

def _count(spend: dict[str, Spend], backend: str, answer) -> None:
    spend.setdefault(backend, Spend()).add(0 if answer.hit else 1, float(answer.cost or 0.0))


def _count_verdicts(spend: dict[str, Spend], backend: str, result) -> None:
    for verdict in result.resolved.values():
        _count(spend, backend, verdict)


# --- pictures (Word) and scene pictures (Sentence) --------------------------

def picture_query_for(ctx: Sourcing, need: Need) -> str | None:
    """The query on record, in precedence (spec 3 section 5): the latest
    learner direction; a judge suggestion newer than the last Source
    ask; the drafted phrase (record.latest_phrase, appended by
    `phrase_attempt`). None when none is on record (r25): the need
    waits -- the gloss is the drafter's input, never a search.
    """
    return record.latest_phrase(ctx.db.assessments_of(need.subject)) or None


def _picture_params(ctx: Sourcing, need: Need, query: str | None) -> dict[str, Any]:
    """What the judge's fit prompt reads back: the thing the picture is for,
    its gloss, and the phrase it was searched for (None for a candidate
    already on record, assess-first)."""
    if need.subject_kind == "sentence":
        sentence = ctx.syllabus.sentence(need.subject)
        thing, gloss = sentence.text, sentence.gloss
    else:
        word = _word_of(ctx, need.subject)
        thing, gloss = word.thai, word.meaning
    return {"word": thing, "meaning": gloss, "gloss_shown": gloss, "phrase": query}


def _picture_attempt(ctx: Sourcing, need: Need, source: str) -> AttemptResult:
    """One attempt (spec 3 section 5): search, imgfetch each hit, judge.
    A served refusal of every hit of a *cached* answer re-asks the search
    once within the attempt and ingests what is new; a search asked live
    in this attempt is not re-asked, its hits having just been served
    (spec 3 section 6a's re-ask rule)."""
    spend: dict[str, Spend] = {}
    query = picture_query_for(ctx, need)
    if query is None:
        raise ValueError(f"picture need {need.subject!r} has no query on record: "
                         "no direction, suggestion or drafted phrase (spec 3 section 5)")
    fetches = _Fetches()
    already = record.tried_urls(ctx.db, need.subject, need.kind, source)
    question = Question(subject=need.subject, provides="picture",
                        params={"query": query}, kind=need.kind, subject_kind=need.subject_kind)
    # An aged-out `nothing` re-offers the source (spec 3 r19 section 6a):
    # ask it afresh, never the cached empty answer.
    fresh = aged_out(ctx.db, need.subject, need.kind, source,
                     nothing_ttl=ctx.nothing_ttl, now_ns=ctx.now_ns())
    ask = ctx.provider.reask if fresh else ctx.provider.ask
    try:
        hits = ask(source, question)
    except QuotaExhausted:
        raise
    except TransportError:
        fetches.failed()
        _append_outcome(ctx, need, source, fetches.outcome, fetches.candidates,
                        tried=fetches.tried)
        raise
    _count(spend, source, hits)
    hit_items = [i for i in hits.items if isinstance(i, Mapping) and i.get("url")]
    fresh_hits = [i for i in hit_items if i["url"] not in already]
    tried_items = fresh_hits[:ctx.image_candidates]
    for item in tried_items:
        _ingest_picture(ctx, need, item, source, spend, fetches)
    if not fetches.candidates and fetches.served_refusals and hits.hit:
        try:
            hits = ctx.provider.reask(source, question)
        except QuotaExhausted:
            raise
        except TransportError:
            fetches.failed()
            _append_outcome(ctx, need, source, fetches.outcome, fetches.candidates,
                            tried=fetches.tried)
            raise
        _count(spend, source, hits)
        excluded = already | {i["url"] for i in tried_items}
        fresh_items = [i for i in hits.items
                       if isinstance(i, Mapping) and i.get("url") and i["url"] not in excluded]
        for item in fresh_items[:ctx.image_candidates]:
            _ingest_picture(ctx, need, item, source, spend, fetches)
    _append_outcome(ctx, need, source, fetches.outcome, fetches.candidates, tried=fetches.tried)
    return _judge_pictures(ctx, need, query, spend)


def _ingest_picture(ctx: Sourcing, need: Need, item: Mapping, source: str,
                    spend: dict[str, Spend], fetches: _Fetches) -> None:
    """One search hit's bytes through imgfetch, with a media row naming
    where it came from. A url the fetcher refuses is logged and counted on
    `fetches`: served or wire. `fetches.tried` records the url on the two
    outcomes spec 3 section 5 calls fetched: ingested, or a served
    FetchRefused (every reason but wire). A wire FetchRefused, and a bare
    TransportError, are both spec 3 section 6a's transient failures -- a
    retry may succeed, so neither marks the url tried; it stays open to a
    later attempt.
    """
    url = item["url"]
    try:
        got = ctx.provider.ask("imgfetch", Question(
            subject=need.subject, provides="picture-bytes", params={"url": url},
            kind=need.kind, subject_kind=need.subject_kind))
    except FetchRefused as e:
        _log.warning("imgfetch refused %s for %s/%s: %s", url, need.subject, need.kind, e)
        if e.served:
            fetches.tried.append(url)
        fetches.failed(served=e.served)
        return
    except TransportError as e:
        _log.warning("imgfetch failed on %s for %s/%s: %s", url, need.subject, need.kind, e)
        fetches.failed()
        return
    _count(spend, "imgfetch", got)
    stored = None
    for fetched in got.items:
        sha = fetched.get("sha")
        if not sha:
            continue
        ctx.db.add_media(sha=sha, kind="picture", ext=str(fetched.get("ext", "jpg")),
                         source=str(item.get("source", source)),
                         origin=str(item.get("origin") or url),
                         licence=str(item.get("licence") or "unknown"), acquired=ctx.today())
        stored = sha
    if stored:
        fetches.tried.append(url)
        fetches.stored(stored)
    else:
        fetches.missed()


def _judge_pictures(ctx: Sourcing, need: Need, query: str | None,
                    spend: dict[str, Spend]) -> AttemptResult:
    """One fit question per candidate on record, cache-first; and, under
    an inline transport with more than one passing picture, one
    preference question over the passing set (a word's pictures only).
    Under a batch transport that preference question is the run's, once
    the fits are in."""
    role = need.role
    params = _picture_params(ctx, need, query)
    shas = _candidate_shas(ctx, need)
    questions = [AssessQuestion(subject=need.subject, role=role, artifact_sha=sha,
                                rubric=ctx.rubrics[role], params=params, kind=need.kind,
                                subject_kind=need.subject_kind)
                 for sha in shas]
    result = ctx.assessor.ask_many("judge", questions)
    _count_verdicts(spend, "judge", result)
    collected = list(result.collected)
    excluded = dict(result.excluded)
    if ctx.assessor.inline and need.subject_kind == "word":
        passing = passing_pictures(ctx.db, need.subject, current_rubric=ctx.rubrics)
        if len(passing) > 1:
            preference = ctx.assessor.ask_many("judge", [_preference_question(
                ctx, need, passing, params["word"], params["meaning"])])
            _count_verdicts(spend, "judge", preference)
            collected += preference.collected
            excluded.update(preference.excluded)
    return AttemptResult(attempted=True, questions=collected, excluded=excluded, spend=spend)


def _preference_question(ctx: Sourcing, need: Need, candidates: Sequence[str],
                         thing: str, gloss: str) -> AssessQuestion:
    """One ordering question over a need's passing pictures. The
    candidate set is its identity (cachekeys.preference_identity), so a
    set that grows is a new question."""
    return AssessQuestion(subject=need.subject, role="picture-preference",
                          rubric=ctx.rubrics["picture-preference"],
                          params={"candidates": list(candidates), "word": thing,
                                  "meaning": gloss},
                          kind=need.kind, subject_kind=need.subject_kind)


def preference_attempt(ctx: Sourcing, subjects: Sequence[str]) -> AttemptResult:
    """The ordering question for each of `subjects` that is a word whose
    passing pictures have none yet
    (derivations.pictures_awaiting_preference).
    """
    spend: dict[str, Spend] = {}
    questions: list[AssessQuestion] = []
    for subject in sorted(subjects):
        word = ctx.syllabus.find_word(WordId(subject))
        if word is None:
            continue
        candidates = pictures_awaiting_preference(ctx.db, subject, current_rubric=ctx.rubrics)
        if candidates:
            questions.append(_preference_question(
                ctx, Need(subject, "picture", "word"), candidates, word.thai, word.meaning))
    if not questions:
        return AttemptResult(attempted=False)
    result = ctx.assessor.ask_many("judge", questions)
    _count_verdicts(spend, "judge", result)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


def _phrase_item(ctx: Sourcing, subject: str, subject_kind: str) -> tuple[str, str]:
    """The Thai text and English gloss `_phrase_prompt` lists for one open
    picture need's subject (spec 3 section 5): a sentence's own text and
    gloss, or a word's Thai form and meaning.
    """
    if subject_kind == "sentence":
        sentence = ctx.syllabus.sentence(subject)
        return sentence.text, sentence.gloss
    word = _word_of(ctx, subject)
    return word.thai, word.meaning


def _phrase_prompt(ctx: Sourcing, needs: Sequence[tuple[str, str]]) -> str:
    """The phrase-drafting prompt (spec 3 section 5): one line per open
    picture need lacking a drafted phrase -- its subject (a word id or a
    sentence text_sha), Thai text and English gloss delimited as deck
    data (assessor.UNTRUSTED/deck_field, as the sentence-drafting
    prompt's own refused block delimits deck text). Asks for a short,
    concrete, proper-noun-free image-search phrase per item.
    """
    lines = []
    for subject, subject_kind in needs:
        text, gloss = _phrase_item(ctx, subject, subject_kind)
        lines.append(f"- subject: {subject}  text: {deck_field(text)}  "
                     f"gloss: {deck_field(gloss)}")
    return (
        "For each item, write a short English image-search phrase (at most six words, "
        "concrete, no proper nouns) that would find an illustrative photo.\n"
        f"{UNTRUSTED}\n"
        "Items:\n" + "\n".join(lines) + "\n"
        'Output JSON only: {"phrases": [{"subject": "...", "phrase": "..."}]}')


def phrase_attempt(ctx: Sourcing) -> AttemptResult:
    """One drafting ask per run (spec 3 section 5) over every open picture
    need -- word or scene -- with no drafted phrase on record
    (record.drafted_phrase): a short English image-search phrase for
    each, so `picture_query_for` finds one and the need is searched
    (r25: without one it waits). Skipped -- no ask made, `attempted=False` -- once every open
    picture need already has one.

    The batch prompt is asked once on the drafter transport (`llm-phrase`,
    cachekeys.LlmPromptKey keyed by the prompt itself, appended under
    record.PHRASE_SUBJECT). Its answer -- `{"phrases": [{"subject": "...",
    "phrase": "..."}]}` (record.parse_phrases) -- then appends one
    provide row per subject the answer actually names (backend llm,
    provides "phrase", key cachekeys.PhraseKey), so `record.drafted_phrase`
    finds it on this and every later run; an item naming a subject that
    was not asked for is ignored, and an item the answer omits is simply
    asked for again next run. A subject that already carries a learner
    direction is never handed to the drafter: `record.latest_phrase`
    always prefers the direction over a drafted phrase, so drafting one
    would be dead weight.
    """
    spend: dict[str, Spend] = {}
    needs = [(subject, subject_kind) for subject, kind, subject_kind
            in available_needs(ctx.syllabus) if kind == "picture"]
    lacking: dict[str, str] = {}
    for subject, subject_kind in needs:
        rows = ctx.db.assessments_of(subject)
        if record.directions(rows):
            continue   # the direction always wins (record.latest_phrase)
        if record.drafted_phrase(rows) is None:
            lacking[subject] = subject_kind
    if not lacking:
        return AttemptResult(attempted=False)
    prompt = _phrase_prompt(ctx, sorted(lacking.items()))
    # subject_kind "batch": the ask's own subject (PHRASE_SUBJECT) is a
    # word and a scene need mixed together, neither one thing -- Question
    # types it as plain str, and nothing folds over this row's own
    # subject_kind (the per-subject phrase rows below carry the real one).
    question = Question(subject=PHRASE_SUBJECT, provides="phrase", kind="picture",
                        subject_kind="batch", params={"prompt": prompt})
    try:
        answer = ctx.provider.ask("llm-phrase", question)
    except TransportError as e:
        # spec 3 section 2: an answer phrasing none of these items is not
        # positively recognized (wiring.build_provider's own llm-phrase
        # recognizer, record.parse_phrases) -- LlmBackend.fetch already
        # raised and cached nothing, so the next run re-asks the same
        # lacking set instead of this becoming a permanent empty cache
        # hit (fix round 2 finding 2). The caller (run._run_pass) counts
        # this under source_failures["llm-phrase"] the same as any other
        # drafter transport failure.
        _log.warning("phrase drafter failed for %d asked item(s): %s", len(lacking), e)
        raise
    _count(spend, "llm-phrase", answer)
    drafted = record.parse_phrases(str(answer.items[0])) if answer.items else {}
    for subject, phrase in drafted.items():
        subject_kind = lacking.get(subject)
        if subject_kind is None:
            continue
        ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject=subject),
                      subject=subject,
                      question={"provides": "phrase", "kind": "picture",
                                "subject_kind": subject_kind},
                      answer={"phrase": phrase})
    return AttemptResult(attempted=True, spend=spend)


# --- retirement (Sentence): the one mechanism both callers use --------------

def retire_sentence(ctx: Sourcing, text_sha: str, *, reason: str,
                    replacement_hint: str | None = None,
                    derived_from: CommentRef | None = None) -> None:
    """Retire one adopted Sentence (spec 3 section 5; F13 and the comment
    pass, r30): one retirement row first (port "attempt", backend "run",
    cachekeys.RetirementKey -- an append is a checkpoint, and the row is
    what record.retirements reads once the sentences row is gone: the
    text is never re-adopted and the drafter is told not to propose it,
    with the reason and the replacement hint beside it), then the
    sentences row deleted and reported to the writing command's Guard,
    and `ctx.syllabus` replaced without it so the rest of this pass and
    every later cycle over the same ctx read the deck as it now is (its
    Targets reopen). `text` goes on the row when ctx.syllabus still
    holds the sentence. The caller decides whether to retire and keeps
    its own counts.
    """
    rows = record.rows_for(ctx.db, text_sha, "recording")
    text = next((s.text for s in ctx.syllabus.sentences if s.text_sha == text_sha), None)
    question: dict[str, Any] = {"kind": "retirement", "subject_kind": "sentence",
                                "reason": reason, "candidates": len(record.candidate_shas(rows))}
    if text is not None:
        question["text"] = text
    if replacement_hint:
        question["replacement_hint"] = replacement_hint
    if derived_from is not None:
        question["comment_sha"] = derived_from.comment_sha
        question["prompt_version"] = derived_from.prompt_version
    ctx.db.append(port="attempt", backend="run", key=RetirementKey(text_sha), subject=text_sha,
                  question=question, answer={"retired": True})
    ctx.db.delete_sentence(text_sha)
    ctx.syllabus = replace(
        ctx.syllabus, sentences=tuple(s for s in ctx.syllabus.sentences if s.text_sha != text_sha))
    if ctx.guard is not None:
        ctx.guard.removed("sentences", [text_sha])
    _log.info("retired sentence %s (%s): %s (%d candidates)", text_sha, text or "?", reason,
              question["candidates"])


# --- comments (any subject): the reading pass ------------------------------

_FAMILY_OF_SUBJECT_KIND = {"word": "word", "sentence": "sentence", "pair": "minimal_pair",
                          "grapheme": "grapheme"}

# What a handed comment the answer named no reading for is recorded as
# (design decision 7): the prompt is keyed by its own text, so the very
# same handed set would read the very same cached answer for ever. One
# reading row with no action and this one unactionable line counts it,
# shows it (record.reading_view) and closes it.
_NO_READING = "the model gave no reading"

# What a comment whose subject the syllabus cannot account for is
# recorded as: the deck can say nothing about a card whose subject is
# gone (a retired sentence), and a comment whose own recorded
# subject_kind disagrees with what the syllabus holds is a row this pass
# refuses to read rather than build facts from the wrong family.
_SUBJECT_GONE = "subject not in the syllabus"
_SUBJECT_KIND_MISMATCH = "subject kind mismatch"


def _subject_kind_in(syllabus: Syllabus, subject: str) -> str | None:
    """What `subject` is in this syllabus, by identity alone: a word id,
    a sentence text_sha, a pair id or a grapheme symbol; None when gone
    (a retired sentence's comment is not handed)."""
    if syllabus.find_word(WordId(subject)) is not None:
        return "word"
    if any(s.text_sha == subject for s in syllabus.sentences):
        return "sentence"
    if any(p.id == subject for p in syllabus.pairs):
        return "pair"
    if any(g.symbol == subject for g in syllabus.graphemes):
        return "grapheme"
    return None


def _handable(ctx: Sourcing, comment: record.Comment) -> tuple[str | None, str | None]:
    """(subject_kind, refusal) for one comment: the kind the SYLLABUS
    resolves its subject to (`_subject_kind_in`), never the kind the row
    recorded -- a row is data, and building a word's facts for a subject
    the syllabus holds as a grapheme would raise out of the whole pass.
    The recorded kind is checked against it and a disagreement refuses
    the comment rather than guessing which is right; so does a subject
    the syllabus no longer holds. A refusal is a reading row saying so
    (never an exception, and never an ask).
    """
    resolved = _subject_kind_in(ctx.syllabus, comment.subject)
    if resolved is None:
        return None, _SUBJECT_GONE
    if comment.subject_kind is not None and comment.subject_kind != resolved:
        _log.warning("comment %s on %s: recorded subject_kind %r, syllabus says %r",
                     comment.comment_sha, comment.subject, comment.subject_kind, resolved)
        return None, _SUBJECT_KIND_MISMATCH
    return resolved, None


def _card_type(comment: record.Comment, subject_kind: str) -> tuple[str, str]:
    """The card type label and its one-line meaning the reader is handed
    (compile.CARD_MEANINGS); a session comment names its question."""
    if comment.card_kind == "question":
        return (f"question ({comment.question_kind}) about the subject's {comment.artifact_kind}",
                "A session question about that artifact; the subject's compiled cards were shown "
                "above it.")
    family = _FAMILY_OF_SUBJECT_KIND[subject_kind]
    return (f"{family} / {comment.card_kind}",
            card_meaning(family, comment.card_kind) or "(no meaning on record for this card type)")


def _subject_facts(ctx: Sourcing, subject: str, subject_kind: str) -> list[str]:
    """The facts about the commented card's subject the reader is given,
    every deck field delimited as untrusted data."""
    syllabus = ctx.syllabus
    if subject_kind == "word":
        w = _word_of(ctx, subject)
        return [f"word id: {w.id}", f"thai: {deck_field(w.thai)}",
                f"meaning: {deck_field(w.meaning)}", f"pronunciation: {ipa.render(w.pron)}",
                f"category: {syllabus.category_of(w.id) or '(none)'}"]
    if subject_kind == "sentence":
        s = syllabus.sentence(subject)
        def gloss_of(e) -> str:
            return f"{element_word(e)} ({deck_field(syllabus.word(element_word(e)).meaning)})"

        clauses = " ".join("[" + ", ".join(gloss_of(e) for e in clause) + "]"
                           for clause in s.clauses)
        fills = ", ".join(str(t.id) for t in syllabus.fill_set(s)) or "(none)"
        return [f"text: {deck_field(s.text)}", f"gloss: {deck_field(s.gloss)}",
                f"clauses (word id with its gloss): {clauses}", f"fills targets: {fills}"]
    if subject_kind == "pair":
        pair = syllabus.pair(PairId(subject))
        members = "; ".join(
            f"{m}: {deck_field(syllabus.word(m).thai)} ({deck_field(syllabus.word(m).meaning)})"
            for m in pair.members)
        return [f"minimal pair {pair.id} on confusion {pair.confusion}", f"members: {members}"]
    if subject_kind == "grapheme":
        g = next((g for g in syllabus.graphemes if g.symbol == subject), None)
        if g is not None:
            return [f"grapheme {deck_field(g.symbol)} ({g.kind}), sound {g.sound}, "
                    f"keyword word {g.keyword}"]
    # `_handable` resolves the kind off the syllabus itself, so this is
    # unreachable from the pass: an explicit refusal by name, never a
    # bare StopIteration or KeyError out of a fact lookup.
    raise ValueError(f"comment subject {subject!r} is no {subject_kind!r} this syllabus holds")


def _shown_facts(ctx: Sourcing, comment: record.Comment) -> list[str]:
    """The artifacts the commented card actually showed (spec 5 r5's own
    `shown`), each with what there is to say about it."""
    out: list[str] = []
    picture = comment.shown.get("picture")
    if picture:
        # the query the last picture search carried, else the one on
        # record for the next (record.latest_phrase: direction >
        # suggestion > phrase) -- read over the subject's WHOLE row set,
        # since a learner direction is a row of its own kind and
        # `rows_for(..., "picture")` cannot see it (picture_query_for
        # reads assessments_of for the same reason)
        rows = record.without_vetoed_readings(ctx.db.assessments_of(comment.subject))
        pictures = [r for r in rows if r.question.get("kind") == "picture"]
        query = record.latest_query(pictures) or record.latest_phrase(rows)
        out.append(f"picture {picture} (search query: {deck_field(query) if query else '(none)'})")
    for sha_ in comment.shown.get("recordings") or []:
        prov = ctx.db.media_provenance(sha_) or {}
        out.append(f"recording {sha_} (source: {prov.get('source') or 'unknown'})")
    return out or ["(no artifact shown)"]


_COMMENT_VOCABULARY = (
    "Actions you may take, each a JSON object with \"action\" and the parameters shown:\n"
    "- direction(kind: picture|recording, text): the English phrase the subject's next search of "
    "that artifact kind uses (a picture direction is the next image-search query).\n"
    "- retire_sentence(reason, replacement_hint): delete the sentence from the deck (sentence "
    "subjects only); its Targets reopen; the hint guides the next draft.\n"
    "- replacement_sentence(thai, gloss): a replacement sentence in Thai using only the deck's "
    "vocabulary, with an English gloss stating exactly what it says; it is parsed and judged "
    "like any draft.\n"
    "- rate(kind: picture|recording, value: 1..4): the learner's rating of the shown artifact "
    "(1 unacceptable, 2 unacceptable but use this one, 3 acceptable, 4 good).\n"
    "- gloss_on(word): show the English gloss on that word's picture front; the word must be "
    "the comment's own subject.\n"
    "- none(remark): the comment asks for nothing the deck does.\n"
    "Anything else the comment asks for goes under \"unactionable\" as text.")


def _comment_prompt(ctx: Sourcing, handed: Sequence[tuple[record.Comment, str]]) -> str:
    """The reading prompt (spec 3 r30 section 5): per comment, its sha,
    text, card type with meaning, subject facts and shown artifacts,
    every deck field delimited as untrusted data; the action vocabulary
    once; the answer shape."""
    blocks = []
    for comment, subject_kind in handed:
        label, meaning = _card_type(comment, subject_kind)
        lines = [f"- comment {comment.comment_sha}: {deck_field(comment.text)}",
                 f"  card: {label} -- {meaning}",
                 f"  subject ({subject_kind}):"]
        lines += [f"    {fact}" for fact in _subject_facts(ctx, comment.subject, subject_kind)]
        lines += ["  shown:"] + [f"    {fact}" for fact in _shown_facts(ctx, comment)]
        blocks.append("\n".join(lines))
    return (
        "A learner wrote a comment on a flashcard of a Thai deck. For each comment, say in one "
        "line what the learner means with respect to that card, then list the actions the deck "
        "should take from the vocabulary below, in order.\n"
        f"{UNTRUSTED}\n"
        f"{_COMMENT_VOCABULARY}\n"
        "Comments:\n" + "\n".join(blocks) + "\n"
        'Output JSON only: {"readings": [{"comment": "<sha as given>", "reading": "<one line>", '
        '"actions": [{"action": "...", ...}], "unactionable": ["..."]}]}')


def _artifact_for(comment: record.Comment, kind: str) -> tuple[str | None, str | None]:
    """The shown artifact a rate action is about: (sha, refusal)."""
    if kind == "picture":
        sha_ = comment.shown.get("picture")
        return (sha_, None) if sha_ else (None, "the card showed no picture")
    recordings = list(comment.shown.get("recordings") or [])
    if len(recordings) == 1:
        return recordings[0], None
    return None, f"the card showed {len(recordings)} recordings"


def draft_refusal(ctx: Sourcing, sentence, open_targets: Sequence[Target] | None = None
                  ) -> str | None:
    """Why a drafted `sentence` could not be adopted, or None when it
    could: the Sentence invariant (Syllabus.check_sentence), the clause
    cap, at least one still-open Target filled, and the per-sentence
    Target cap (spec 3 section 5, r27). The one acceptance test every
    pass that can raise a draft's judge question applies --
    `sentence_attempt`, the comment pass's `replacement_sentence`, and
    the run's D2 recovery over the drafts on record -- so no pass asks
    the judge about a draft another would refuse, and the reason reads
    the same wherever it is reported (a reading row's `refused`, a log
    line).

    `open_targets` is the caller's own snapshot of the Targets still open
    (`sentence_attempt` reads it once for the whole run); None reads
    gaps() here.
    """
    try:
        ctx.syllabus.check_sentence(sentence)
    except ValueError as e:
        return str(e)
    if len(sentence.clauses) > ctx.sentence_max_clauses:
        # spec 3 section 5: more clauses than the cap refuses the draft
        # like the Sentence invariant above -- local and mechanical, the
        # provide row keeping it.
        return f"{len(sentence.clauses)} clauses (cap {ctx.sentence_max_clauses})"
    if open_targets is None:
        open_ids = set(ctx.syllabus.gaps().unfilled_targets)
        open_targets = [t for t in ctx.syllabus.targets if t.id in open_ids]
    fills = ctx.syllabus.fill_set(sentence)
    filled = [t for t in open_targets if t in fills]
    if not filled:
        return "fills no open Target"
    if len(filled) > ctx.sentence_targets_per_sentence:
        # spec 3 r27 section 5: more open Targets than the cap is a word
        # list in disguise -- refused like the clause cap; met words
        # beyond the filled ones are filler and free.
        return f"{len(filled)} targets (cap {ctx.sentence_targets_per_sentence})"
    return None


def _refused(action: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {**action, "outcome": "refused", "reason": reason}


def _done(action: Mapping[str, Any]) -> dict[str, Any]:
    return {**action, "outcome": "done"}


def _draft_replacement(ctx: Sourcing, action: Mapping[str, Any],
                       parses: Mapping[str, Clauses], ref: CommentRef,
                       questions: list[AssessQuestion]) -> dict[str, Any]:
    """replacement_sentence: the parsed clauses become a draft accepted
    the way sentence_attempt accepts one (invariant, clause cap, fills an
    open Target, target cap), appended as the same provide row under
    DRAFT_SUBJECT that a drafting ask leaves -- record.sentence_drafts
    reads it back -- with its sentence-for-target question collected for
    the run's batch; adoption is the next run's, once the verdict lands.
    """
    text = action["thai"].strip()
    clauses = parses.get(text)
    if clauses is None:
        return _refused(action, "no parse returned for this text")
    draft = record.SentenceDraft(clauses=clauses, text=text, gloss=action["gloss"].strip())
    if draft.text_sha in {s.text_sha for s in ctx.syllabus.sentences}:
        return _refused(action, "already adopted")
    sentence = record.draft_sentence(draft, ctx.today)
    refusal = draft_refusal(ctx, sentence)
    if refusal is not None:
        return _refused(action, refusal)
    ctx.db.append(port="provide", backend="llm",
                  key=ProvideKey(source="llm-comment", kind="sentence", query=draft.text_sha),
                  subject=DRAFT_SUBJECT,
                  question={"provides": "sentence", "kind": "sentence", "subject_kind": "sentence",
                            "comment_sha": ref.comment_sha, "prompt_version": ref.prompt_version},
                  answer={"items": [json.dumps({"sentences": [
                      {"clauses": clauses_to_json(clauses), "text": text, "gloss": draft.gloss}]},
                      ensure_ascii=False)]})
    role = role_for("sentence")
    last_word = ctx.syllabus.word(ctx.syllabus.last_used_word(sentence)).thai
    questions.append(AssessQuestion(
        subject=draft.text_sha, role=role, artifact_sha=None, rubric=ctx.rubrics[role],
        params={"text": text, "gloss": draft.gloss, "word": last_word},
        kind="sentence", subject_kind="sentence"))
    return _done(action)


# The subject kinds that have a picture and a recording need of their own
# (derivations.available_needs): a `direction` or `rate` names one of
# those two artifact kinds (record._COMMENT_ARTIFACT_KINDS) and nothing
# else, and only a word and a sentence have such a need. Under any other
# subject `authority.role_for` falls back to the word role, so the row
# would be written under a role no fold over that subject ever reads --
# dead, while the screen reports an action taken. A pair's rendition
# would be the one other judged artifact, but the comment vocabulary
# cannot name it, so it needs no exception here.
_ARTIFACT_SUBJECT_KINDS = frozenset({"word", "sentence"})


def _act(ctx: Sourcing, comment: record.Comment, subject_kind: str, action: Mapping[str, Any],
         parses: Mapping[str, Clauses], ref: CommentRef,
         questions: list[AssessQuestion]) -> dict[str, Any]:
    """One action executed as its existing typed row, marked with the
    comment; the record of what happened goes on the reading row."""
    name = action["action"]
    if name == "none":
        # design decision 5: an act with no side effect -- a remark --
        # and still an action taken, never an unactionable request.
        return _done(action)
    if name in ("direction", "rate") and subject_kind not in _ARTIFACT_SUBJECT_KINDS:
        return _refused(action, f"the subject has no {action['kind']} need")
    if name == "direction":
        append_direction(ctx.db, subject=comment.subject,
                         role=role_for(action["kind"], subject_kind), text=action["text"].strip(),
                         subject_kind=subject_kind, derived_from=ref)
        return _done(action)
    if name == "rate":
        sha_, refusal = _artifact_for(comment, action["kind"])
        if refusal:
            return _refused(action, refusal)
        rating = ACTION_RATINGS[action["value"]]
        if rating == "unacceptable-none":
            current = current_best_of(ctx, comment.subject, action["kind"]).artifact_sha
            if current != sha_:
                return _refused(action, f"the card no longer shows that {action['kind']}")
        append_rating(ctx.db, subject=comment.subject, role=role_for(action["kind"], subject_kind),
                      rating=rating, artifact_sha=sha_, subject_kind=subject_kind, derived_from=ref)
        return _done(action)
    if name == "gloss_on":
        # the parser already drops a gloss_on naming a word other than
        # the comment's own subject (design decision 3,
        # record.parse_comment_readings); these guard what is left.
        if subject_kind != "word":
            return _refused(action, "the subject is not a word")
        if ctx.syllabus.find_word(WordId(action["word"])) is None:
            return _refused(action, f"no word {action['word']!r}")
        ctx.db.append(port="assess", backend="learner",
                      key=DirectionKey(subject=action["word"], role="gloss-on",
                                       text_sha=sha("gloss on")),
                      subject=action["word"],
                      question={"kind": "gloss-on", "role": "picture-for-word",
                                "subject_kind": "word", "comment_sha": ref.comment_sha,
                                "prompt_version": ref.prompt_version},
                      answer={"direction": "gloss on"})
        return _done(action)
    if name == "retire_sentence":
        if subject_kind != "sentence":
            return _refused(action, "the subject is not a sentence")
        if not any(s.text_sha == comment.subject for s in ctx.syllabus.sentences):
            return _refused(action, "not an adopted sentence")
        retire_sentence(ctx, comment.subject, reason=action["reason"].strip(),
                        replacement_hint=(action.get("replacement_hint") or "").strip() or None,
                        derived_from=ref)
        return _done(action)
    if name == "replacement_sentence":
        return _draft_replacement(ctx, action, parses, ref, questions)
    return _refused(action, "outside the vocabulary")


def _parse_replacements(ctx: Sourcing, readings: Mapping[str, record.CommentReading],
                        spend: dict[str, Spend]) -> dict[str, Clauses]:
    """One parse ask (record.parse_prompt, the migration's own) over every
    replacement text the readings name, before anything is executed, so a
    parse failure leaves nothing half done. Empty when no reading names
    one -- no ask is made.
    """
    texts = sorted({a["thai"].strip() for r in readings.values() for a in r.actions
                    if a["action"] == "replacement_sentence"})
    if not texts:
        return {}
    vocabulary = sorted(ctx.syllabus.words, key=lambda w: w.id)
    parsed = ctx.provider.ask("llm-parse", Question(
        subject=PARSE_SUBJECT, provides="parse", kind="sentence", subject_kind="sentence",
        params={"prompt": record.parse_prompt(texts, vocabulary)}))
    _count(spend, "llm-parse", parsed)
    parses: dict[str, Clauses] = {}
    for item in parsed.items:
        parses.update(record.parses_in(str(item)))
    return parses


def comment_attempt(ctx: Sourcing) -> AttemptResult:
    """One reading ask per run (spec 3 r30 section 5) over the oldest
    COMMENTS_PER_ASK comments with no reading row under
    COMMENT_PROMPT_VERSION whose subject the syllabus still holds: the
    prompt hands each comment with its card, its subject's facts and what
    it showed; the answer (record.parse_comment_readings, handed the
    comment sha -> subject map so nothing is executed against a comment
    nobody asked about, read over every answer item, the first reading of
    a sha winning) is executed comment by comment, each action as its
    existing typed row marked with the comment (learner.CommentRef), then
    one reading row per comment. A replacement's Thai goes through one
    parse ask first (record.parse_prompt, the migration's own), before
    anything is executed, so a parse failure leaves nothing half done.

    A comment the answer names but was not handed is dropped and logged;
    a handed one it says nothing about is recorded as read with no action
    and one unactionable line (`_NO_READING`), never left to be re-asked
    as the same cached prompt for ever. A comment whose subject the
    syllabus cannot account for (`_handable`) is never handed and never
    raises: it gets its own reading row saying so, closing it. A comment
    over the cap is held back whole -- no row, no reading -- so the next
    run hands it. `llm-comment` and `llm-parse` transport failures
    propagate (run counts them under source_failures) -- both come before
    any row is written, so nothing is half done.

    A judge that cannot be reached at the replacement drafts' check comes
    after every row was written, so it is reported rather than raised:
    the result carries its counts, no questions, and
    `judge_unreachable`, and the run collects it before ending the pass
    (spec 3 r30 section 5). The drafts keep their provide rows; the run's
    own D2 recovery raises their questions again next run.
    """
    spend: dict[str, Spend] = {}
    handable: list[tuple[record.Comment, str]] = []
    closed: list[tuple[record.Comment, str]] = []
    for comment in record.comments(ctx.db):
        rows = ctx.db.assessments_of(comment.subject)
        if record.reading_of(rows, comment.comment_sha, COMMENT_PROMPT_VERSION) is not None:
            continue
        subject_kind, refusal = _handable(ctx, comment)
        if refusal is not None:
            closed.append((comment, refusal))
        else:
            handable.append((comment, str(subject_kind)))
    handed, held_back = handable[:COMMENTS_PER_ASK], handable[COMMENTS_PER_ASK:]
    if held_back:
        _log.info("comment pass: %d comment(s) handed (cap %d), %d held back for the next run",
                  len(handed), COMMENTS_PER_ASK, len(held_back))
    if not handed and not closed:
        return AttemptResult(attempted=False)

    readings: dict[str, record.CommentReading] = {}
    if handed:
        question = Question(subject=COMMENT_SUBJECT, provides="comment-reading", kind="comment",
                            subject_kind="batch", params={"prompt": _comment_prompt(ctx, handed)})
        answer = ctx.provider.ask("llm-comment", question)
        _count(spend, "llm-comment", answer)
        subjects = {c.comment_sha: c.subject for c, _ in handed}
        for item in answer.items:
            for comment_sha, reading in record.parse_comment_readings(str(item), subjects).items():
                readings.setdefault(comment_sha, reading)
    parses = _parse_replacements(ctx, readings, spend)

    questions: list[AssessQuestion] = []
    read = actions_done = unactionable = retired = 0
    for comment, refusal in closed:
        # no ask was made about it and no action is possible: the row is
        # what counts it, shows it (record.reading_view) and closes it.
        ctx.db.append(port="assess", backend="llm",
                      key=CommentReadingKey(comment.comment_sha, COMMENT_PROMPT_VERSION),
                      subject=comment.subject,
                      question={"kind": "comment-reading", "comment_sha": comment.comment_sha,
                                "prompt_version": COMMENT_PROMPT_VERSION,
                                "subject_kind": comment.subject_kind or "",
                                "anchor": comment.anchor, "card_kind": comment.card_kind},
                      answer={"reading": "", "actions": [], "unactionable": [refusal]})
        read += 1
        unactionable += 1
    for comment, subject_kind in handed:
        ref = CommentRef(comment.comment_sha, COMMENT_PROMPT_VERSION)
        reading = readings.get(comment.comment_sha)
        if reading is None:
            _log.warning("comment %s: the reader named no reading for it", comment.comment_sha)
            reading = record.CommentReading(reading="", actions=(), unactionable=(_NO_READING,))
        before = len(ctx.syllabus.sentences)
        records = [_act(ctx, comment, subject_kind, a, parses, ref, questions)
                   for a in reading.actions]
        retired += before - len(ctx.syllabus.sentences)
        actions_done += sum(1 for r in records if r["outcome"] == "done")
        unactionable += (sum(1 for r in records if r["outcome"] == "refused")
                         + len(reading.unactionable))
        read += 1
        ctx.db.append(port="assess", backend="llm",
                      key=CommentReadingKey(comment.comment_sha, COMMENT_PROMPT_VERSION),
                      subject=comment.subject,
                      question={"kind": "comment-reading", "comment_sha": comment.comment_sha,
                                "prompt_version": COMMENT_PROMPT_VERSION,
                                "subject_kind": subject_kind, "anchor": comment.anchor,
                                "card_kind": comment.card_kind},
                      answer={"reading": reading.reading, "actions": records,
                              "unactionable": list(reading.unactionable)})
    result = None
    unreachable = False
    if questions:
        try:
            result = ctx.assessor.ask_many("judge", questions)
        except JudgeUnreachable:
            # Every retirement, direction, rating and reading row above is
            # already on the record -- an append is a checkpoint -- so the
            # counts must reach the report even though the run is about to
            # end. Reported, not raised: the run collects this result and
            # then takes its judge-death path (`judge_unreachable`). The
            # replacement drafts keep their provide rows and the D2
            # recovery re-raises their questions next run.
            _log.warning("comment pass: the judge could not be reached to check %d replacement "
                         "draft(s); %d comment(s) were read and their rows stand", len(questions),
                         read)
            unreachable = True
        else:
            _count_verdicts(spend, "judge", result)
    return AttemptResult(attempted=True,
                         questions=list(result.collected) if result else [],
                         excluded=dict(result.excluded) if result else {}, spend=spend,
                         comments_read=read, comment_actions=actions_done,
                         comment_unactionable=unactionable, retired=retired,
                         judge_unreachable=unreachable)


# --- adjudication (Word): the pronunciation ask ------------------------------

def adjudication_attempt(ctx: Sourcing) -> AttemptResult:
    """One judge question per Word whose pronunciation is not corroborated
    (spec 3 r28 section 5): the pronunciation-for-word role, text only,
    cache-first through the assessor -- a word answered under the current
    rubric collects nothing new. The verdicts land with the run's batch;
    run._materialize_adjudications reads them back next run.

    A Word is not a need (it is curated data, not a gap gaps() lists), so
    this pass owns no RunReport bucket: its questions ride the run's one
    batch and `pending` never counts them.
    """
    words = [w for w in ctx.syllabus.words if not is_corroborated(w.pron.corroboration)]
    if not words:
        return AttemptResult(attempted=False)
    role = role_for("pronunciation")
    questions = [AssessQuestion(subject=str(w.id), role=role, artifact_sha=None,
                                rubric=ctx.rubrics[role],
                                params={"thai": w.thai, "meaning": w.meaning},
                                kind="pronunciation", subject_kind="word") for w in words]
    spend: dict[str, Spend] = {}
    result = ctx.assessor.ask_many("judge", questions)
    _count_verdicts(spend, "judge", result)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


# --- recordings (Word) and sentence recordings ------------------------------

def _voice_constraint(ctx: Sourcing, need: Need) -> VoiceConstraint:
    """The recording's voice constraint follows the sentence's speaker
    marking (spec 1 section 1 (r10)): "female" when the marking is
    `{"female"}`, "male" when it is `{"male"}`, else "male" where the
    recording plays on a productive back (E2, spec 3 section 5) -- the
    aggregate decides what serves a productive Target -- and "any"
    otherwise. A marking holding both sexes cannot reach here:
    Syllabus.check_sentence refuses such a sentence. `serves` (whether the
    recording plays on a productive back) is read only once the marking
    itself does not decide -- the marking checks first.
    """
    if need.subject_kind == "sentence":
        sentence = ctx.syllabus.sentence(need.subject)
        marking = ctx.syllabus.marking(sentence)

        def serves() -> bool:
            return ctx.syllabus.sentence_serves_productive(sentence)
    else:
        word = _word_of(ctx, need.subject)
        marking = frozenset({word.speaker}) - {None}

        def serves() -> bool:
            return ctx.syllabus.serves_productive(word.id)
    if marking == frozenset({"female"}):
        return "female"
    if marking == frozenset({"male"}):
        return "male"
    return "male" if serves() else "any"


def _pool(ctx: Sourcing, constraint: VoiceConstraint) -> list[str]:
    """The voices a constraint admits: one sex's pool under "male" or
    "female", both under "any" (spec 3 section 5)."""
    voices = (list(ctx.voices.get("male", ())) + list(ctx.voices.get("female", ()))
              if constraint == "any"
              else list(ctx.voices.get(constraint, ())))
    if not voices:
        raise ValueError(f"no {constraint!r} voice pool is configured for a tts recording")
    return voices


def _tts_speaker(ctx: Sourcing, voice: str) -> Speaker:
    """TTS supplies sex and timbre only (spec 3 section 5): the pool names
    the sex, age and accent stay unknown."""
    if voice in ctx.voices.get("male", ()):
        sex = "male"
    elif voice in ctx.voices.get("female", ()):
        sex = "female"
    else:
        sex = "unknown"
    return Speaker(id=f"tts:{voice}", kind="synthetic", sex=sex)


def _forvo_speaker(item: Mapping) -> Speaker:
    """The item's own sex and country (spec 2); what Forvo left out stays
    "unknown"."""
    return Speaker(id=f"forvo:{item['username']}", kind="native",
                   sex=_FORVO_SEX.get(str(item.get("sex") or "").lower(), "unknown"),
                   region=str(item.get("country") or "unknown"))


def _forvo_lookup(ctx: Sourcing, subject: str, thai: str, spend: dict[str, Spend],
                  *, subject_kind: SubjectKind = "word",
                  constraint: VoiceConstraint = "any", fresh: bool = False) -> list[Mapping]:
    """One lookup, cached forever, appended under `subject`. Under a
    "male" or "female" constraint only speakers Forvo states are that
    sex are admitted (spec 1 section 1 (r10); E2: a productive back
    plays in the learner's register); "any" admits every item. `fresh`
    re-asks over the cached answer (spec 3 section 6a's re-ask rule)."""
    ask = ctx.provider.reask if fresh else ctx.provider.ask
    answer = ask("forvo", Question(subject=subject, provides="recording",
                                   params={"word": thai}, kind="recording",
                                   subject_kind=subject_kind))
    _count(spend, "forvo", answer)
    items = [i for i in answer.items
             if isinstance(i, Mapping) and i.get("pathmp3") and i.get("username")]
    if constraint == "any":
        return items
    return [i for i in items if _forvo_speaker(i).sex == constraint]


def _store(ctx: Sourcing, got: ProviderAnswer, *, source: str, origin: str, licence: str,
           speaker: Speaker) -> str | None:
    """The first fetched item's sha, its speaker recorded before the media
    row that references it."""
    for fetched in got.items:
        sha = fetched.get("sha")
        if not sha:
            continue
        ctx.db.add_speaker(speaker)
        ctx.db.add_media(sha=sha, kind="recording", ext=str(fetched.get("ext", "mp3")),
                         source=source, origin=origin, licence=licence,
                         acquired=ctx.today(), speaker_id=speaker.id)
        return sha
    return None


def _download_forvo(ctx: Sourcing, subject: str, item: Mapping, spend: dict[str, Spend],
                    fetches: _Fetches, *, subject_kind: SubjectKind = "word",
                    relookup: Callable[[], Sequence[Mapping]] | None = None
                    ) -> tuple[str, Mapping] | None:
    """One item's mp3 through audiofetch, and the item its sha actually
    came from (the retried item on a re-ask, `item` otherwise). A served
    refusal of its url re-asks the lookup through `relookup` once and
    retries the item found under the same Forvo id (spec 3 section 6a);
    an item with no id is not retried. A second refusal, a wire failure,
    or a failed re-lookup, counts as transient. A re-lookup that hits
    Forvo's own quota (`QuotaExhausted`) is not a transient failure: it
    propagates unchanged to the attempt function's own catch, the same
    contract a quota hit on the first lookup has."""
    got = _fetch_forvo_item(ctx, subject, item, fetches, subject_kind=subject_kind)
    if got is None and fetches.last_refusal_served and relookup is not None:
        key = item.get("id")
        try:
            candidates = relookup()
        except QuotaExhausted:
            raise
        except TransportError as e:
            _log.warning("forvo re-lookup failed for %s: %s", subject, e)
            fetches.failed()
            return None
        fresh = next((i for i in candidates if key is not None and i.get("id") == key), None)
        if fresh is not None:
            got = _fetch_forvo_item(ctx, subject, fresh, fetches, subject_kind=subject_kind)
            item = fresh
    if got is None:
        return None
    _count(spend, "audiofetch", got)
    _count(spend, "forvo", got)   # Forvo counts the download as a request (spec 3 section 4)
    sha = _store(ctx, got, source="forvo", origin=item["pathmp3"], licence="forvo",
                 speaker=_forvo_speaker(item))
    if sha:
        fetches.stored(sha)
        return sha, item
    fetches.missed()
    return None


def _fetch_forvo_item(ctx: Sourcing, subject: str, item: Mapping, fetches: _Fetches, *,
                      subject_kind: SubjectKind) -> ProviderAnswer | None:
    """One item's mp3 through audiofetch. A content-type refusal whose
    body is Forvo's own daily-limit statement is Quota, not a served
    refusal (spec 3 section 6a): audiofetch refuses Forvo's JSON error
    body the same way it refuses any unexpected content-type, so this is
    where that body is told apart from every other served refusal and
    raised typed instead of counted and logged."""
    url = item["pathmp3"]
    try:
        return ctx.provider.ask("audiofetch", Question(
            subject=subject, provides="recording-bytes",
            params={"url": url, "speaker": item["username"], "speaker_kind": "native",
                    "source": "forvo"}, kind="recording", subject_kind=subject_kind))
    except FetchRefused as e:
        if e.reason == "content-type" and _forvo_limit_refusal(e):
            raise QuotaExhausted("forvo") from e
        _log.warning("audiofetch refused %s for %s: %s", url, subject, e)
        fetches.failed(served=e.served)
        return None
    except TransportError as e:
        _log.warning("audiofetch failed on %s for %s: %s", url, subject, e)
        fetches.failed()
        return None


def _forvo_limit_refusal(e: FetchRefused) -> bool:
    """Whether a content-type FetchRefused's body is Forvo's own
    daily-limit statement (spec 3 section 6a): the body parsed as json
    and handed to the same predicate the lookup's 400 body uses
    (provider.forvo_limit_body). A body that is not json, or json of any
    other shape, is not Forvo's quota -- it stays a plain served
    refusal."""
    try:
        parsed = json.loads(e.body)
    except json.JSONDecodeError:
        return False
    return forvo_limit_body(parsed)


def _relookup_once(ctx: Sourcing, subject: str, thai: str, spend: dict[str, Spend], *,
                   subject_kind: SubjectKind, constraint: VoiceConstraint,
                   memo: dict[str, Sequence[Mapping]]) -> Sequence[Mapping]:
    """One fresh lookup per subject within an attempt, memoized in `memo`."""
    if subject not in memo:
        memo[subject] = _forvo_lookup(ctx, subject, thai, spend, subject_kind=subject_kind,
                                      constraint=constraint, fresh=True)
    return memo[subject]


def _synthesize(ctx: Sourcing, subject: str, text: str, voice: str, spend: dict[str, Spend],
                *, subject_kind: SubjectKind = "word",
                fetches: _Fetches | None = None) -> str | None:
    try:
        got = ctx.provider.ask("tts", Question(subject=subject, provides="recording",
                                               params={"text": text, "voice": voice},
                                               kind="recording", subject_kind=subject_kind))
    except SynthesisRefused as e:
        _log.warning("tts refused %s for %s: %s", voice, subject, e)
        if fetches is not None:
            fetches.missed()
        return None
    _count(spend, "tts", got)
    sha = _store(ctx, got, source="tts", origin=voice, licence="google-tts",
                 speaker=_tts_speaker(ctx, voice))
    if fetches is not None:
        if sha:
            fetches.stored(sha)
        else:
            fetches.missed()
    return sha


def _check(ctx: Sourcing, questions: Sequence[AssessQuestion], spend: dict[str, Spend]):
    """The mechanical duration/format check: ground truth for what it
    checks, and the authority that ranks a recording."""
    result = ctx.assessor.ask_many("mechanical", questions)
    _count_verdicts(spend, "mechanical", result)
    return result


def _recording_attempt(ctx: Sourcing, need: Need, source: str) -> AttemptResult:
    spend: dict[str, Spend] = {}
    text = (ctx.syllabus.sentence(need.subject).text if need.subject_kind == "sentence"
            else _word_of(ctx, need.subject).thai)
    constraint = _voice_constraint(ctx, need)
    fetches = _Fetches()
    if source == "forvo":
        # An aged-out `nothing` re-offers forvo (spec 3 r19 section 6a):
        # ask it afresh, never the cached empty answer.
        fresh = aged_out(ctx.db, need.subject, need.kind, "forvo",
                         nothing_ttl=ctx.nothing_ttl, now_ns=ctx.now_ns())
        try:
            items = _forvo_lookup(ctx, need.subject, text, spend,
                                  subject_kind=need.subject_kind, constraint=constraint,
                                  fresh=fresh)
            memo: dict[str, Sequence[Mapping]] = {}
            relookup = functools.partial(_relookup_once, ctx, need.subject, text, spend,
                                         subject_kind=need.subject_kind, constraint=constraint,
                                         memo=memo)
            for item in items:
                _download_forvo(ctx, need.subject, item, spend, fetches,
                                subject_kind=need.subject_kind, relookup=relookup)
        except QuotaExhausted:
            raise
        except TransportError:
            fetches.failed()
            _append_outcome(ctx, need, source, fetches.outcome, fetches.candidates)
            raise
    elif source == "tts":
        voice = pick_voice(need.subject, _pool(ctx, constraint))
        try:
            _synthesize(ctx, need.subject, text, voice, spend,
                        subject_kind=need.subject_kind, fetches=fetches)
        except QuotaExhausted:
            raise
        except TransportError:
            fetches.failed()
            _append_outcome(ctx, need, source, fetches.outcome, fetches.candidates)
            raise
    else:
        raise ValueError(f"no recording source named {source!r}")
    _append_outcome(ctx, need, source, fetches.outcome, fetches.candidates)
    result = _check(ctx, [AssessQuestion(subject=need.subject, role=need.role,
                                         artifact_sha=sha, kind=need.kind,
                                         subject_kind=need.subject_kind)
                          for sha in _candidate_shas(ctx, need)], spend)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


# --- renditions (MinimalPair) -----------------------------------------------

def _rendition_attempt(ctx: Sourcing, need: Need, source: str) -> AttemptResult:
    """One recording per member by one speaker (spec 3 section 2's
    compound question), appended under the pair; Forvo's per-member
    lookups stay cached under the members. A Source that cannot guarantee
    one speaker answers empty. When members is non-empty, the outcome
    row's candidates carry the rendition identity
    (cachekeys.rendition_identity) alongside the member recording shas,
    so the row anchors escalation on the rendition current-best
    (spec 3 section 6a)."""
    spend: dict[str, Spend] = {}
    pair = ctx.syllabus.pair(PairId(need.subject))
    words = {member: _word_of(ctx, member) for member in pair.members}
    constraint = ctx.syllabus.pair_voice_constraint(pair.id)
    fetches = _Fetches()

    try:
        if source == "forvo":
            members = _forvo_rendition(ctx, pair, words, constraint, spend, fetches)
        elif source == "tts":
            members = _tts_rendition(ctx, pair, words, constraint, spend, fetches)
        else:
            raise ValueError(f"no rendition source named {source!r}")
    except QuotaExhausted:
        raise
    except TransportError:
        fetches.failed()
        _append_outcome(ctx, need, source, fetches.outcome, fetches.candidates)
        raise

    ctx.db.append(port="provide", backend=source,
                  key=RenditionAskKey(source=source, pair_id=pair.id), subject=pair.id,
                  question={"provides": "rendition", "kind": "rendition",
                            "subject_kind": "pair",
                            "params": {"members": list(pair.members)}},
                  answer={"items": [{"member": member, "sha": sha,
                                     "speaker": asdict(speaker)}
                                    for member, (sha, speaker) in members.items()]})
    shas = {member: sha for member, (sha, _speaker) in members.items()}
    outcome_candidates = ([*fetches.candidates, rendition_identity(shas)] if members
                          else fetches.candidates)
    _append_outcome(ctx, need, source, fetches.outcome, outcome_candidates)
    if not members:
        return AttemptResult(attempted=True, spend=spend)
    result = ctx.assessor.ask_many("rendition", [AssessQuestion(
        subject=pair.id, role=need.role, artifact_sha=rendition_identity(shas),
        kind="rendition", subject_kind="pair",
        params={"members": shas, "member_checks": _check_members(ctx, members, spend)})])
    _count_verdicts(spend, "rendition", result)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


def _check_members(ctx: Sourcing, members: Mapping[str, tuple[str, Speaker]],
                   spend: dict[str, Spend]) -> dict[str, bool]:
    """Each member's own recording, checked under the member's own
    subject, and handed to the rendition check. A question that never
    resolved is left out; RenditionBackend refuses to judge a member set
    with an unchecked member (PreparationError), which excludes the
    question for the run."""
    questions = {member: AssessQuestion(subject=member, role=role_for("recording"),
                                        artifact_sha=sha, kind="recording", subject_kind="word")
                 for member, (sha, _speaker) in members.items()}
    result = _check(ctx, list(questions.values()), spend)
    return {member: bool(v.value)
            for member, q in questions.items()
            if (v := result.resolved.get(ctx.assessor.key_of("mechanical", q))) is not None}


def _forvo_rendition(ctx: Sourcing, pair, words, constraint: VoiceConstraint,
                     spend: dict[str, Spend],
                     fetches: _Fetches) -> dict[str, tuple[str, Speaker]]:
    """The intersection of the members' lookups by username: the first
    speaker who said every member. Every member's lookup runs before any
    download is attempted.
    """
    fresh = aged_out(ctx.db, pair.id, "rendition", "forvo",
                     nothing_ttl=ctx.nothing_ttl, now_ns=ctx.now_ns())
    by_member = {m: _forvo_lookup(ctx, m, words[m].thai, spend, constraint=constraint,
                                  fresh=fresh)
                 for m in pair.members}
    memo: dict[str, Sequence[Mapping]] = {}
    shared = set.intersection(*[{i["username"] for i in items} for items in by_member.values()])
    for username in sorted(shared):
        members: dict[str, tuple[str, Speaker]] = {}
        for member in pair.members:
            item = next(i for i in by_member[member] if i["username"] == username)
            relookup = functools.partial(_relookup_once, ctx, member, words[member].thai, spend,
                                         subject_kind="word", constraint=constraint, memo=memo)
            stored = _download_forvo(ctx, member, item, spend, fetches, relookup=relookup)
            if stored is not None:
                sha, stored_item = stored
                members[member] = (sha, _forvo_speaker(stored_item))
        if len(members) == len(pair.members):
            return members
    return {}


def _tts_rendition(ctx: Sourcing, pair, words, constraint: VoiceConstraint,
                   spend: dict[str, Spend],
                   fetches: _Fetches) -> dict[str, tuple[str, Speaker]]:
    """One voice across the members. A member's synthesis that fails on
    the wire raises out of the loop; a member's synthesis the service
    refuses ends the loop with no member set for the rest, without
    asking them. An earlier member's own success stays recorded on
    `fetches` either way.
    """
    voice = pick_voice(pair.id, _pool(ctx, constraint))
    speaker = _tts_speaker(ctx, voice)
    members: dict[str, tuple[str, Speaker]] = {}
    for member in pair.members:
        sha = _synthesize(ctx, member, words[member].thai, voice, spend, fetches=fetches)
        if sha is None:
            break
        members[member] = (sha, speaker)
    return members if len(members) == len(pair.members) else {}


# --- the sentence attempt (per run, over the open Targets) ------------------

def _entry_vocabulary(syllabus: Syllabus, targets: Sequence[Target]) -> list[Word]:
    """The vocabulary a sentence prompt may draw on, entirely: the
    picture-introduced words met in entry order (Syllabus.order) up to
    the furthest handed target -- a met sentence-introduced word
    (`Syllabus.met_sentence_introduced_targets`) interleaved at its own
    entry within that same bounded walk when its entry falls at or
    before the furthest handed target, appended after the walk, still in
    entry order, when its entry falls beyond it. An unmet
    sentence-introduced word stays out throughout.
    """
    met_targets = syllabus.met_sentence_introduced_targets()
    wanted = {t.id for t in targets}
    by_id = {t.id: t for t in syllabus.targets}
    vocabulary: list[Word] = []
    seen: set[WordId] = set()
    remaining = set(wanted)
    for entry in syllabus.order():
        if entry.kind != "word_target":
            continue
        entry_target = by_id[entry.id]
        word_id = entry_target.word
        include = entry_target.introduction == "picture_card" or entry.id in met_targets
        if include and word_id not in seen:
            seen.add(word_id)
            vocabulary.append(syllabus.word(word_id))
        remaining.discard(entry.id)
        if not remaining:
            break
    for entry in syllabus.order():
        if entry.kind != "word_target":
            continue
        entry_target = by_id[entry.id]
        if (entry_target.introduction == "sentence" and entry.id in met_targets
                and entry_target.word not in seen):
            seen.add(entry_target.word)
            vocabulary.append(syllabus.word(entry_target.word))
    return vocabulary


def _has_digit_suffix(word_id: str) -> bool:
    base, _, suffix = word_id.rpartition("-")
    return bool(base) and suffix.isdigit()


def _example_clause_ids(vocabulary: Sequence[Word]) -> tuple[str, str]:
    """The ids the worked example draws from the prompt's own vocabulary
    where possible (spec 3 r24 section 5): the repeated word takes the
    first vocabulary id, else the literal `little`; the suffixed id
    takes the first vocabulary id carrying a -<digit> suffix, else the
    literal `delicious-2`.
    """
    repeated = vocabulary[0].id if vocabulary else "little"
    suffixed = next((w.id for w in vocabulary if _has_digit_suffix(w.id)), "delicious-2")
    return repeated, suffixed


def _sentence_prompt(syllabus: Syllabus, targets: Sequence[Target],
                     refused: Sequence[tuple[str, str]] = (),
                     *, sentence_max_clauses: int,
                     sentence_targets_per_sentence: int = DEFAULT_SENTENCE_TARGETS_PER_SENTENCE) -> str:
    """The drafting prompt (spec 3 section 5): the met vocabulary once as
    id/thai/meaning lines, a Targets line per picture-introduced handed
    target and per handed sentence-introduced target some adopted
    sentence already fills, an Introducible line per handed
    sentence-introduced target no adopted sentence fills, the profile
    register, the existing sentence openings to avoid, and the clause
    rendering rule (spec 1 section 1). Asks for at most `sentence_max_clauses`
    clauses per sentence (spec 3 r23 section 5/8: a longer sentence outruns
    the 5 s recording cap). When `refused` (derivations.refused_drafts) is
    non-empty, a block lists those texts as sentences not to propose
    again, each with the verdict's evidence delimited the way the
    assessor prompts delimit deck fields (assessor.deck_field, over the
    untrusted-data notice given once before the block), before the
    output-format sentence (spec 3 r19 section 5), which requires
    vocabulary ids exactly as listed, suffix included, and carries one
    worked example item showing a suffixed id and a repeated word,
    built from the prompt's own vocabulary where possible
    (`_example_clause_ids`, spec 3 r24 section 5).
    """
    vocabulary = _entry_vocabulary(syllabus, targets)
    met_targets = syllabus.met_sentence_introduced_targets()
    target_lines = []
    introducible_lines = []
    for target in targets:
        word = syllabus.word(target.word)
        line = f"- target {target.id}: {record.vocabulary_line(word)}"
        if target.introduction == "sentence" and target.id not in met_targets:
            introducible_lines.append(line)
        else:
            target_lines.append(line)
    openings = sorted({syllabus.word(s.words[0]).thai for s in syllabus.sentences if s.words})
    sections = ("Vocabulary, in the order met:\n"
               + "\n".join("- " + record.vocabulary_line(w) for w in vocabulary) + "\n")
    if target_lines:
        sections += "Targets:\n" + "\n".join(target_lines) + "\n"
    if introducible_lines:
        sections += ("Introducible (at most one per sentence):\n"
                    + "\n".join(introducible_lines) + "\n")
    refused_lines = (f"- {text} — {deck_field(evidence)}" if evidence else f"- {text}"
                     for text, evidence in refused)
    refused_block = (f"Do not propose these sentences; each failed review:\n{UNTRUSTED}\n"
                     + "\n".join(refused_lines) + "\n"
                     if refused else "")
    repeated_id, suffixed_id = _example_clause_ids(vocabulary)
    example = json.dumps({"clauses": [["i-male-speaker", "eat", [repeated_id, "ๆ"], suffixed_id]],
                          "text": "...", "gloss": "..."}, ensure_ascii=False)
    return ("Draft flashcard sentences in colloquial Central Thai for a learner whose register is "
            f"{syllabus.profile.register}.\n"
            "Each JSON item is one sentence. Write as many natural sentences as it takes to "
            f"cover the targets below; a sentence fills at most {sentence_targets_per_sentence} "
            "of the targets (two or three is right) and may use any other listed vocabulary "
            "besides. A sentence may "
            "introduce at most one word from the Introducible list and must otherwise use only "
            "the vocabulary below.\n"
            f"Each sentence has at most {sentence_max_clauses} clauses.\n"
            "Give each sentence an English gloss that states exactly what it says.\n"
            + (f"Avoid starting with any of: {', '.join(openings)}.\n" if openings else "")
            + sections
            + refused_block
            + "Write each sentence as clauses of vocabulary ids in order; a clause renders as "
            "its words' Thai concatenated, clauses are separated by one space; write a repeated "
            'word as [id, "ๆ"]; standard spelling (ครับ, never คับ); numbers as number words; '
            "no punctuation or digits. Use vocabulary ids exactly as listed, suffix included; "
            f"for example: {example}. "
            'Output JSON only: {"sentences": [{"clauses": [["id", ...], ...], "text": "...", '
            '"gloss": "..."}]}')


def sentence_attempt(ctx: Sourcing, *, max_targets: int = 40) -> AttemptResult:
    """One drafting ask per run over the open Targets (spec 3 section 5),
    at most `max_targets` of them (AttemptResult.targets_handed says how
    many, and subjects_handed which words they belong to). The handed
    targets are the next open Targets in order, of which at most
    `ctx.sentence_introducible_per_ask` (spec 3 r24 section 5/8) are
    introducible -- sentence-introduced (`introduction == "sentence"`)
    and not yet met (`Syllabus.met_sentence_introduced_targets`); an
    introducible Target beyond that cap is skipped rather than handed,
    and does not count against `max_targets`, so the remainder of the
    handed batch is the next non-introduced open Targets in order. This
    keeps a run dominated by introducible targets (e.g. 36 of 40) from
    starving the handed batch of the receptive backlog the drafter can
    actually place several of per sentence. Each merged
    draft becomes a Sentence (record.draft_sentence); acceptance is
    `draft_refusal` -- the Sentence invariant (Syllabus.check_sentence),
    the clause cap `ctx.sentence_max_clauses` (spec 3 section 5: "more
    clauses than the cap refuses the draft"; the drafting prompt itself
    already asks for at most that many), at least one still-open Target
    filled (Syllabus.fill_set), and the per-sentence Target cap (r27) --
    the same test the comment pass's replacement and the run's D2
    recovery apply. A refused draft is logged ("draft refused: %s: %s",
    the reason and the text) and skipped, nothing else. The judge
    question carries the text, gloss, and the sentence's own last used
    word (Syllabus.last_used_word). Adoption is the run's, after the
    verdicts land. The drafting prompt also names the texts the judge
    has already failed (derivations.refused_drafts, spec 3 r19 section
    5) so the drafter does not propose them again.

    A word whose sentence need is at the no-fit cap
    (derivations.sentence_exhausted under ctx.sentence_nothing_cap) is
    withheld: its Targets are not handed over, and
    `subjects_exhausted` names it so the run counts it `exhausted`
    rather than attempted or deferred. A no-fit answer -- spec 3 r19
    section 5's `{"sentences": [], "reason": "..."}`, read by
    record.parse_no_fit -- appends one `nothing` outcome row per handed
    Target's word, under that WORD as subject with the Target ids in the
    row's question, and asks the judge nothing. A no-fit served from the
    provider cache is re-asked once first (section 6a): the cap counts
    the drafter's refusals, never the runs that read the same cached one.
    """
    spend: dict[str, Spend] = {}
    syllabus = ctx.syllabus
    unfilled = syllabus.gaps().unfilled_targets
    all_open_ids = set(unfilled)
    word_of = {t.id: t.word for t in syllabus.targets}
    withheld = frozenset(
        w for w in {word_of[t] for t in unfilled if t in word_of}
        if sentence_exhausted(ctx.db, w, cap=ctx.sentence_nothing_cap).exhausted)
    handable = [t for t in unfilled if word_of.get(t) not in withheld]
    target_of = {t.id: t for t in syllabus.targets}
    met_targets = syllabus.met_sentence_introduced_targets()
    selected: list[str] = []
    introducible_handed = 0
    for tid in handable:
        if len(selected) >= max_targets:
            break
        candidate = target_of.get(tid)
        introducible = (candidate is not None and candidate.introduction == "sentence"
                       and tid not in met_targets)
        if introducible:
            if introducible_handed >= ctx.sentence_introducible_per_ask:
                continue   # over the introducible cap -- skipped, not counted against max_targets
            introducible_handed += 1
        selected.append(tid)
    open_ids = set(selected)
    targets = [t for t in syllabus.targets if t.id in open_ids]
    if not targets:
        return AttemptResult(attempted=False, subjects_exhausted=withheld)
    open_targets = [t for t in syllabus.targets if t.id in all_open_ids]

    refused = refused_drafts(ctx.db, syllabus, current_rubric=ctx.rubrics)
    question = Question(
        subject=DRAFT_SUBJECT, provides="sentence", kind="sentence", subject_kind="sentence",
        params={"prompt": _sentence_prompt(
            syllabus, targets, refused, sentence_max_clauses=ctx.sentence_max_clauses,
            sentence_targets_per_sentence=ctx.sentence_targets_per_sentence)})
    answer = ctx.provider.ask("llm-sentence", question)
    _count(spend, "llm-sentence", answer)

    no_fit = _no_fit_in(answer)
    if no_fit is not None and answer.hit:
        # Spec 3 r19 section 6a's re-ask rule. A no-fit adds nothing to the
        # prompt's refused block, so the next run's prompt -- and its cache
        # key -- is the very same one; served from the cache it would cache
        # one more `nothing` row per run for a refusal the drafter never
        # made, and sentence_exhausted would count runs instead. Re-ask
        # once and let the fresh answer (drafts or a new no-fit) be the
        # run's answer; only an answer that was not a hit is recorded.
        answer = ctx.provider.reask("llm-sentence", question)
        _count(spend, "llm-sentence", answer)
        no_fit = _no_fit_in(answer)
    if no_fit is not None:
        _append_no_fit(ctx, targets, no_fit)
        return AttemptResult(attempted=True, drafted=0, targets_handed=len(targets),
                             subjects_handed=frozenset(t.word for t in targets),
                             subjects_exhausted=withheld, spend=spend)

    adopted = {s.text_sha for s in syllabus.sentences}
    questions: list[AssessQuestion] = []
    raw_drafts = [d for item in answer.items for d in record.parse_drafts(str(item))]
    for draft in record.merge_drafts(raw_drafts):
        if draft.text_sha in adopted:
            continue
        sentence = record.draft_sentence(draft, ctx.today)
        refusal = draft_refusal(ctx, sentence, open_targets)
        if refusal is not None:
            _log.warning("draft refused: %s: %s", refusal, draft.text)
            continue
        last_word = syllabus.word(syllabus.last_used_word(sentence)).thai
        questions.append(AssessQuestion(
            subject=draft.text_sha, role=role_for("sentence"), artifact_sha=None,
            rubric=ctx.rubrics[role_for("sentence")],
            params={"text": draft.text, "gloss": draft.gloss, "word": last_word},
            kind="sentence", subject_kind="sentence"))
    result = ctx.assessor.ask_many("judge", questions)
    _count_verdicts(spend, "judge", result)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend,
                         drafted=len(questions), targets_handed=len(targets),
                         subjects_handed=frozenset(t.word for t in targets),
                         subjects_exhausted=withheld)


def _no_fit_in(answer: ProviderAnswer) -> str | None:
    """The reason the first no-fit item in a drafting answer gives (spec 3
    r19 section 5, record.parse_no_fit), or None when no item is one."""
    return next((reason for item in answer.items
                if (reason := record.parse_no_fit(str(item)))), None)


def _append_no_fit(ctx: Sourcing, targets: Sequence[Target], reason: str) -> None:
    """One `nothing` outcome row per handed Target's word (spec 3 r19
    section 5): the drafter answered that nothing fits, and the record
    keeps that per word, since the sentence need's subject is the word
    everywhere -- the Target ids it covered sit in the row's question.
    `derivations.sentence_exhausted` counts these rows against the
    no-fit cap.
    """
    by_word: dict[str, list[str]] = {}
    for t in targets:
        by_word.setdefault(str(t.word), []).append(str(t.id))
    for word_id, target_ids in by_word.items():
        ctx.db.append(port="attempt", backend="llm",
                      key=AttemptOutcomeKey(subject=word_id, kind="sentence", source="llm"),
                      subject=word_id,
                      question={"kind": "sentence", "source": "llm", "subject_kind": "word",
                                "targets": target_ids},
                      answer={"outcome": "nothing", "candidates": [], "reason": reason})


_ATTEMPTS: dict[str, Callable[[Sourcing, Need, str], AttemptResult]] = {
    "picture": _picture_attempt,
    "recording": _recording_attempt,
    "rendition": _rendition_attempt,
}


def _assess_pictures(ctx: Sourcing, need: Need) -> AttemptResult:
    """The fit questions on a picture need's candidates already on record
    (spec 3 section 5 assess-first): no phrase -- the search that
    produced such a candidate is not this attempt's, so the question
    names none and the rubric's "pass if no phrase is given" applies.
    """
    return _judge_pictures(ctx, need, None, {})


def _assess_recordings(ctx: Sourcing, need: Need) -> AttemptResult:
    """Spec 3 r23 section 5: the mechanical duration/format check on the
    need's candidates with no mechanical verdict under this subject
    (derivations.unjudged_candidates), the same AssessQuestion shape
    _recording_attempt builds at the end of its own attempt. Mechanical
    is inline and free -- no batch transport -- so every question
    resolves within the call and `questions` always comes back empty.
    """
    spend: dict[str, Spend] = {}
    shas = unjudged_candidates(ctx.db, need.subject, need.kind, current_rubric=ctx.rubrics)
    questions = [AssessQuestion(subject=need.subject, role=need.role, artifact_sha=sha,
                                kind=need.kind, subject_kind=need.subject_kind)
                for sha in shas]
    result = _check(ctx, questions, spend)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


# The assessment a kind's assess-first step runs: the fit/check questions
# on every candidate on record, cache-first (_judge_pictures/_check asks
# nothing for a candidate already judged under the current rubric or
# already mechanically checked under this subject).
_ASSESS_FIRST: dict[str, Callable[[Sourcing, Need], AttemptResult]] = {
    "picture": _assess_pictures,
    "recording": _assess_recordings,
}


def assess_first(ctx: Sourcing, need: Need) -> AttemptResult | None:
    """Spec 3 section 5 assess-first: the fit questions on the need's
    candidates with no verdict under the current rubric; no source is
    asked and no outcome row is written. None when no candidate awaits a
    verdict. When every awaiting question was excluded, returns an
    unattempted result carrying those exclusions (spec 3 section 7's
    RunReport.excluded) so the caller's fall-through to a source still
    reports them, rather than dropping them on the floor -- built with
    dataclasses.replace so the assess step's own spend (and any other
    field) survives onto the unattempted result, not just `excluded`.
    Logs the excluded candidates when it falls through.
    """
    awaiting = unjudged_candidates(ctx.db, need.subject, need.kind, current_rubric=ctx.rubrics)
    if not awaiting:
        return None
    assess = _ASSESS_FIRST.get(need.kind)
    if assess is None:
        raise ValueError(f"no assess-first step is defined for artifact kind {need.kind!r} "
                         f"(subject {need.subject!r}, a {need.subject_kind})")
    result = assess(ctx, need)
    excluded_shas = {e.artifact_sha for e in result.excluded.values()}
    if all(sha in excluded_shas for sha in awaiting):
        _log.warning("assess-first for %s/%s: every awaiting candidate was excluded (%s); "
                     "asking a source", need.subject, need.kind, ", ".join(sorted(awaiting)))
        return replace(result, attempted=False)
    return result


def attempt(ctx: Sourcing, need: Need, source: str) -> AttemptResult:
    """One Source asked for one need, under the need's own subject."""
    make = _ATTEMPTS.get(need.kind)
    if make is None:
        raise ValueError(f"no attempt is defined for artifact kind {need.kind!r} "
                         f"(subject {need.subject!r}, a {need.subject_kind})")
    return make(ctx, need, source)
