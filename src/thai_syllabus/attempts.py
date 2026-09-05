"""What an attempt IS for each need (spec 3 section 5): one Source asked
under the need's own subject, whatever it returns ingested, the speaker it
came from recorded, and the judge questions the run will ask collected.

A Need is an artifact kind and the kind of thing its subject is. The
artifact kinds are the ones the media store and compile already know --
"picture", "recording", "rendition" -- and a sentence's own scene picture
and reading are those same kinds under a sentence subject; the subject
kind is what puts them in their own Assess roles (authority.role_for).

An attempt appends and nothing else. It never adopts a sentence, never
ranks a candidate and never decides what to try next: current-best,
improved, pending, exhausted and what there is to adopt are
derivations.py's folds over the rows these asks append, and the run
(run.py) drives the loop.

Under an inline judge transport every collected question resolves inside
this call, so `questions` comes back empty and the attempt converges in
one pass -- the picture preference question included. Under a batch
transport the misses come back in `questions` for the run to submit as
one batch; a question that could not be prepared is in `excluded` and its
candidate is unusable. A judge that cannot be reached at all raises
JudgeUnreachable out of ask_many and stops the run.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from . import record
from .assessor import AssessQuestion, Assessor, PreparedQuestion
from .authority import role_for
from .cachekeys import RenditionAskKey, rendition_identity
from .derivations import (
    DEFAULT_ATTEMPT_CAP,
    CurrentBest,
    current_best,
    passing_pictures,
    pictures_awaiting_preference,
)
from .entities import Target, Word
from .ids import PairId, WordId
from .media import Speaker
from .provider import Provider, ProviderAnswer, Question
from .query import QUERY_HINTS, picture_query
from .record import DRAFT_SUBJECT, SentenceDraft
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus
from .transport import TransportError
from .tts import FEMALE_VOICES, MALE_VOICES, pick_voice

__all__ = ["Need", "Sourcing", "Spend", "AttemptResult", "SOURCES", "SubjectKind",
           "sources_for", "provenance_source_for", "current_best_of",
           "attempt", "sentence_attempt", "preference_attempt"]

_log = logging.getLogger(__name__)

# Cheapest source first, per ARTIFACT kind (spec 3 section 5). A sentence's
# own recording and scene picture are the same artifact kinds a word's are;
# only the subject differs.
SOURCES: dict[str, tuple[str, ...]] = {
    "picture": ("openverse", "wikimedia", "pexels"),
    "recording": ("forvo", "tts"),
    "rendition": ("forvo", "tts"),
}

# Forvo's own sex codes, and the Speaker vocabulary they map onto.
_FORVO_SEX = {"m": "male", "f": "female"}


def sources_for(kind: str) -> tuple[str, ...]:
    return SOURCES.get(kind, ())


SubjectKind = Literal["word", "pair", "grapheme", "sentence"]


@dataclass(frozen=True)
class Need:
    """(subject, artifact kind) plus what the subject IS: a word's picture
    and a sentence's scene picture are both kind "picture" and differ only
    in their subject, which is what decides the role and the attempt.
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
    query_hints: Mapping[str, str] = field(default_factory=lambda: dict(QUERY_HINTS))
    judge_model: str = "llm"
    # The Sources each artifact kind may be asked for, cheapest first, and
    # the attempt count exhausted() stops at.
    sources_for: Callable[[str], Sequence[str]] = field(default=sources_for)
    attempt_cap: int = DEFAULT_ATTEMPT_CAP


@dataclass(frozen=True)
class AttemptResult:
    """`attempted`: a Source ask was made, hit or miss. `questions`: the
    judge questions this attempt collected for the run's batch, empty
    under an inline transport. `excluded`: encoded key -> why a question
    could not be prepared. `spend`: per backend. `drafted`: the sentence
    drafts this attempt produced that fill an open Target (0 for every
    attempt that is not the sentence attempt).
    """
    attempted: bool
    questions: list[PreparedQuestion] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    spend: dict[str, Spend] = field(default_factory=dict)
    drafted: int = 0


# --- reading the record -----------------------------------------------------

def _word_of(ctx: Sourcing, subject: str) -> Word:
    """The Word a need's subject names, refusing by name when there is
    none -- a bare KeyError on a word id says nothing about the need."""
    word = ctx.syllabus.find_word(WordId(subject))
    if word is None:
        raise ValueError(f"need {subject!r} names no word in the syllabus")
    return word


def provenance_source_for(db: SyllabusDb) -> Callable[[str], str | None]:
    """current_best's provenance_source over `db`: the media table's own
    `source` for a sha, not a cache row's backend -- a bytes-fetch row
    (imgfetch/audiofetch) carries no Source name at all."""
    def get(sha: str) -> str | None:
        prov = db.media_provenance(sha)
        return prov.get("source") if prov else None
    return get


def current_best_of(ctx: Sourcing, subject: str, kind: str) -> CurrentBest:
    return current_best(ctx.db, subject, kind, current_rubric=ctx.rubrics,
                        prior=ctx.provenance_prior,
                        provenance_source=provenance_source_for(ctx.db))


def _phrase(ctx: Sourcing, subject: str) -> str | None:
    """The image phrase a human or a judge drafted: the latest learner
    direction, else a judge suggestion newer than the last provide row."""
    rows = ctx.db.assessments_of(subject)
    directions = [r for r in rows if r.port == "assess" and r.backend == "learner"
                  and r.answer.get("direction")]
    if directions:
        return max(directions, key=lambda r: r.ts).answer["direction"]
    last_provide = max((r.ts for r in rows if r.port == "provide"), default=-1)
    suggestions = [r for r in rows if r.port == "assess" and r.backend == "judge"
                   and r.answer.get("suggestion") and r.ts > last_provide]
    if suggestions:
        return max(suggestions, key=lambda r: r.ts).answer["suggestion"]
    return None


def _candidate_shas(ctx: Sourcing, need: Need) -> list[str]:
    return record.candidate_shas(record.rows_for(ctx.db, need.subject, need.kind))


# --- spend ------------------------------------------------------------------

def _count(spend: dict[str, Spend], backend: str, answer) -> None:
    spend.setdefault(backend, Spend()).add(0 if answer.hit else 1, float(answer.cost or 0.0))


def _count_verdicts(spend: dict[str, Spend], backend: str, result) -> None:
    for verdict in result.resolved.values():
        _count(spend, backend, verdict)


# --- pictures (Word) and scene pictures (Sentence) --------------------------

def _picture_query_for(ctx: Sourcing, need: Need) -> str:
    """A drafted phrase; else, for a scene, the sentence's own English
    gloss, and for a word its gloss head term with the category qualifier.
    The corpora index English metadata, so the query is English either way.
    """
    phrase = _phrase(ctx, need.subject)
    if phrase:
        return phrase
    if need.subject_kind == "sentence":
        gloss = ctx.syllabus.sentence(need.subject).gloss.strip()
        if not gloss:
            raise ValueError(
                f"sentence {need.subject!r} has no gloss to search a scene picture for")
        return gloss
    word = _word_of(ctx, need.subject)
    return picture_query(word, ctx.syllabus.category_of(word.id), None, ctx.query_hints)


def _picture_params(ctx: Sourcing, need: Need, query: str) -> dict[str, Any]:
    """What the judge's fit prompt reads back: the thing the picture is for,
    its gloss, and the phrase it was searched for."""
    if need.subject_kind == "sentence":
        sentence = ctx.syllabus.sentence(need.subject)
        thing, gloss = sentence.text, sentence.gloss
    else:
        word = _word_of(ctx, need.subject)
        thing, gloss = word.thai, word.meaning
    return {"word": thing, "meaning": gloss, "gloss_shown": gloss, "phrase": query}


def _picture_attempt(ctx: Sourcing, need: Need, source: str) -> AttemptResult:
    spend: dict[str, Spend] = {}
    query = _picture_query_for(ctx, need)
    hits = ctx.provider.ask(source, Question(subject=need.subject, provides="picture",
                                             params={"query": query}, kind=need.kind,
                                             subject_kind=need.subject_kind))
    _count(spend, source, hits)
    hit_items = [i for i in hits.items if isinstance(i, Mapping) and i.get("url")]
    for item in hit_items[:ctx.image_candidates]:
        _ingest_picture(ctx, need, item, source, spend)
    return _judge_pictures(ctx, need, query, spend)


def _ingest_picture(ctx: Sourcing, need: Need, item: Mapping, source: str,
                    spend: dict[str, Spend]) -> None:
    """One search hit's bytes through imgfetch, with a media row naming
    where it came from. A url the fetcher refuses is named and skipped: a
    download that failed is not the Source's answer."""
    url = item["url"]
    try:
        got = ctx.provider.ask("imgfetch", Question(
            subject=need.subject, provides="picture-bytes", params={"url": url},
            kind=need.kind, subject_kind=need.subject_kind))
    except TransportError as e:
        _log.warning("imgfetch refused %s for %s/%s: %s", url, need.subject, need.kind, e)
        return
    _count(spend, "imgfetch", got)
    for fetched in got.items:
        sha = fetched.get("sha")
        if not sha:
            continue
        ctx.db.add_media(sha=sha, kind="picture", ext=str(fetched.get("ext", "jpg")),
                         source=str(item.get("source", source)),
                         origin=str(item.get("origin") or url),
                         licence=str(item.get("licence") or "unknown"), acquired=ctx.today())


def _judge_pictures(ctx: Sourcing, need: Need, query: str,
                    spend: dict[str, Spend]) -> AttemptResult:
    """One fit question per candidate on record, cache-first; and, under an
    inline transport, where more than one picture passes, one preference
    question over the passing set. Under a batch transport that preference
    question is the run's, once the fits are in -- so the ask is gated on
    the transport, not on whether this attempt happened to collect
    anything (a batch attempt of pure cache hits collects nothing).
    Preference orders a word's pictures only: derivations folds a
    preference ranking for that subject alone."""
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
    """One ordering question over a need's passing pictures, carrying the
    need's own kind and subject kind. The candidate set is its identity
    (cachekeys.preference_identity), so a set that grows is a new
    question."""
    return AssessQuestion(subject=need.subject, role="picture-preference",
                          rubric=ctx.rubrics["picture-preference"],
                          params={"candidates": list(candidates), "word": thing,
                                  "meaning": gloss},
                          kind=need.kind, subject_kind=need.subject_kind)


def preference_attempt(ctx: Sourcing, subjects: Sequence[str]) -> AttemptResult:
    """The ordering question for each of `subjects` that is a word whose
    passing pictures have none (derivations.pictures_awaiting_preference)
    -- what a batch transport leaves open until its fit verdicts land, and
    the run asks once they have.
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


# --- recordings (Word) and sentence recordings ------------------------------

def _voice_constraint(ctx: Sourcing, need: Need) -> str:
    """"male" where the recording plays on a productive back (E2), "any"
    otherwise -- the aggregate decides what serves a productive Target."""
    if need.subject_kind == "sentence":
        serves = ctx.syllabus.sentence_serves_productive(ctx.syllabus.sentence(need.subject))
    else:
        serves = ctx.syllabus.serves_productive(_word_of(ctx, need.subject).id)
    return "male" if serves else "any"


def _pool(ctx: Sourcing, constraint: str) -> list[str]:
    voices = (list(ctx.voices.get("male", ()))
              if constraint == "male"
              else list(ctx.voices.get("male", ())) + list(ctx.voices.get("female", ())))
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
    """The item's own sex and country (spec 2); anything Forvo left out
    stays "unknown" and never counts as coverage."""
    return Speaker(id=f"forvo:{item['username']}", kind="native",
                   sex=_FORVO_SEX.get(str(item.get("sex") or "").lower(), "unknown"),
                   region=str(item.get("country") or "unknown"))


def _forvo_lookup(ctx: Sourcing, subject: str, thai: str, spend: dict[str, Spend],
                  *, subject_kind: SubjectKind = "word",
                  constraint: str = "any") -> list[Mapping]:
    """One lookup, cached forever, appended under `subject` -- a pair
    member's lookup is the row its own recording need reads. Under a "male"
    constraint only speakers Forvo says are male are admitted: a recording
    that plays on a productive back has to be in the learner's register
    (E2), and an unstated sex is not a claim that it is."""
    answer = ctx.provider.ask("forvo", Question(subject=subject, provides="recording",
                                                params={"word": thai}, kind="recording",
                                                subject_kind=subject_kind))
    _count(spend, "forvo", answer)
    items = [i for i in answer.items
             if isinstance(i, Mapping) and i.get("pathmp3") and i.get("username")]
    if constraint != "male":
        return items
    return [i for i in items if _forvo_speaker(i).sex == "male"]


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


def _download_forvo(ctx: Sourcing, subject: str, item: Mapping,
                    spend: dict[str, Spend], *, subject_kind: SubjectKind = "word") -> str | None:
    url = item["pathmp3"]
    try:
        got = ctx.provider.ask("audiofetch", Question(
            subject=subject, provides="recording-bytes",
            params={"url": url, "speaker": item["username"], "speaker_kind": "native",
                    "source": "forvo"}, kind="recording", subject_kind=subject_kind))
    except TransportError as e:
        _log.warning("audiofetch refused %s for %s: %s", url, subject, e)
        return None
    _count(spend, "audiofetch", got)
    return _store(ctx, got, source="forvo", origin=url, licence="forvo",
                  speaker=_forvo_speaker(item))


def _synthesize(ctx: Sourcing, subject: str, text: str, voice: str,
                spend: dict[str, Spend], *, subject_kind: SubjectKind = "word") -> str | None:
    got = ctx.provider.ask("tts", Question(subject=subject, provides="recording",
                                           params={"text": text, "voice": voice},
                                           kind="recording", subject_kind=subject_kind))
    _count(spend, "tts", got)
    return _store(ctx, got, source="tts", origin=voice, licence="google-tts",
                  speaker=_tts_speaker(ctx, voice))


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
    if source == "forvo":
        for item in _forvo_lookup(ctx, need.subject, text, spend,
                                  subject_kind=need.subject_kind, constraint=constraint):
            _download_forvo(ctx, need.subject, item, spend, subject_kind=need.subject_kind)
    elif source == "tts":
        voice = pick_voice(need.subject, _pool(ctx, constraint))
        _synthesize(ctx, need.subject, text, voice, spend, subject_kind=need.subject_kind)
    else:
        raise ValueError(f"no recording source named {source!r}")
    result = _check(ctx, [AssessQuestion(subject=need.subject, role=need.role,
                                         artifact_sha=sha, kind=need.kind,
                                         subject_kind=need.subject_kind)
                          for sha in _candidate_shas(ctx, need)], spend)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


# --- renditions (MinimalPair) -----------------------------------------------

def _rendition_attempt(ctx: Sourcing, need: Need, source: str) -> AttemptResult:
    """One recording per member by one speaker (spec 3 section 2's compound
    question), appended under the pair -- the need's own subject -- even
    though Forvo's per-member lookups are cached under the members. A
    Source that cannot guarantee one speaker answers empty."""
    spend: dict[str, Spend] = {}
    pair = ctx.syllabus.pair(PairId(need.subject))
    words = {member: _word_of(ctx, member) for member in pair.members}
    constraint = ctx.syllabus.pair_voice_constraint(pair.id)

    if source == "forvo":
        members = _forvo_rendition(ctx, pair, words, constraint, spend)
    elif source == "tts":
        members = _tts_rendition(ctx, pair, words, constraint, spend)
    else:
        raise ValueError(f"no rendition source named {source!r}")

    ctx.db.append(port="provide", backend=source,
                  key=RenditionAskKey(source=source, pair_id=pair.id), subject=pair.id,
                  question={"provides": "rendition", "kind": "rendition",
                            "subject_kind": "pair",
                            "params": {"members": list(pair.members)}},
                  answer={"items": [{"member": member, "sha": sha,
                                     "speaker": asdict(speaker)}
                                    for member, (sha, speaker) in members.items()]})
    if not members:
        return AttemptResult(attempted=True, spend=spend)
    shas = {member: sha for member, (sha, _speaker) in members.items()}
    result = ctx.assessor.ask_many("rendition", [AssessQuestion(
        subject=pair.id, role=need.role, artifact_sha=rendition_identity(shas),
        kind="rendition", subject_kind="pair",
        params={"members": shas, "member_checks": _check_members(ctx, members, spend)})])
    _count_verdicts(spend, "rendition", result)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend)


def _check_members(ctx: Sourcing, members: Mapping[str, tuple[str, Speaker]],
                   spend: dict[str, Spend]) -> dict[str, bool]:
    """Each member's own recording, checked under the MEMBER's subject, so
    the member word's recording need reads the same verdict -- and handed
    on to the rendition check, which is the one decider on whether these
    recordings are a rendition. A question that never resolved counts as
    failing: nothing was verified."""
    questions = {member: AssessQuestion(subject=member, role=role_for("recording"),
                                        artifact_sha=sha, kind="recording", subject_kind="word")
                 for member, (sha, _speaker) in members.items()}
    result = _check(ctx, list(questions.values()), spend)
    return {member: bool(v.value)
            if (v := result.resolved.get(ctx.assessor.key_of("mechanical", q))) is not None
            else False
            for member, q in questions.items()}


def _forvo_rendition(ctx: Sourcing, pair, words, constraint: str,
                     spend: dict[str, Spend]) -> dict[str, tuple[str, Speaker]]:
    """The intersection of the members' lookups by username: the first
    speaker who said every member."""
    by_member = {m: _forvo_lookup(ctx, m, words[m].thai, spend, constraint=constraint)
                 for m in pair.members}
    shared = set.intersection(*[{i["username"] for i in items} for items in by_member.values()])
    for username in sorted(shared):
        members: dict[str, tuple[str, Speaker]] = {}
        for member in pair.members:
            item = next(i for i in by_member[member] if i["username"] == username)
            sha = _download_forvo(ctx, member, item, spend)
            if sha is not None:
                members[member] = (sha, _forvo_speaker(item))
        if len(members) == len(pair.members):
            return members
    return {}


def _tts_rendition(ctx: Sourcing, pair, words, constraint: str,
                   spend: dict[str, Spend]) -> dict[str, tuple[str, Speaker]]:
    """One voice across the members."""
    voice = pick_voice(pair.id, _pool(ctx, constraint))
    speaker = _tts_speaker(ctx, voice)
    members: dict[str, tuple[str, Speaker]] = {}
    for member in pair.members:
        sha = _synthesize(ctx, member, words[member].thai, voice, spend)
        if sha is not None:
            members[member] = (sha, speaker)
    return members if len(members) == len(pair.members) else {}


# --- the sentence attempt (per run, over the open Targets) ------------------

def _sentence_prompt(syllabus: Syllabus, targets: Sequence[Target]) -> str:
    lines = []
    for target in targets:
        word = syllabus.word(target.word)
        met = ", ".join(w.thai for w in syllabus.vocabulary_met_by(target))
        lines.append(f"- target {target.id}: word {word.thai} ({word.meaning}); may use: {met}")
    openings = sorted({syllabus.tokenizer.tokens(s.text)[0] for s in syllabus.sentences
                       if syllabus.tokenizer.tokens(s.text)})
    return ("Draft flashcard sentences in colloquial Central Thai for a learner whose register is "
            f"{syllabus.profile.register}.\n"
            "Write one short sentence per target, or one sentence covering several targets when "
            "their permitted vocabularies allow it. Use only the listed words for each target; "
            "every other word in a sentence must appear in that target's 'may use' list.\n"
            "Give each sentence an English gloss that states exactly what it says.\n"
            + (f"Avoid starting with any of: {', '.join(openings)}.\n" if openings else "")
            + "Targets:\n" + "\n".join(lines) + "\n"
            'Output JSON only: {"sentences": [{"text": "...", "gloss": "...", '
            '"targets": ["<target id>", ...]}]}')


def _fills(ctx: Sourcing, draft: SentenceDraft, targets_by_id: Mapping[str, Target],
           spend: dict[str, Spend]) -> list[Target]:
    """fills() on each Target the draft claims: the verdict that decides
    whether the draft is worth a judge question at all."""
    claimed = [targets_by_id[t] for t in draft.claimed if t in targets_by_id]
    questions = {target.id: AssessQuestion(
        subject=draft.text_sha, role=role_for("sentence"), artifact_sha=None,
        params={"target": target.id, "text": draft.text, "gloss": draft.gloss},
        kind="sentence", subject_kind="sentence") for target in claimed}
    result = ctx.assessor.ask_many("fills", list(questions.values()))
    _count_verdicts(spend, "fills", result)
    return [target for target in claimed
            if (v := result.resolved.get(ctx.assessor.key_of("fills", questions[target.id])))
            is not None and v.value is True]


def sentence_attempt(ctx: Sourcing, *, max_targets: int = 40) -> AttemptResult:
    """One drafting ask per run over the open Targets (spec 3 section 5),
    each draft verified with fills() against the Targets it claims and,
    where it fills one, put to the judge with its gloss. Adoption is the
    run's, after the verdicts land."""
    spend: dict[str, Spend] = {}
    syllabus = ctx.syllabus
    open_ids = set(syllabus.gaps().unfilled_targets[:max_targets])
    targets = [t for t in syllabus.targets if t.id in open_ids]
    if not targets:
        return AttemptResult(attempted=False)

    answer = ctx.provider.ask("llm-sentence", Question(
        subject=DRAFT_SUBJECT, provides="sentence", kind="sentence", subject_kind="sentence",
        params={"prompt": _sentence_prompt(syllabus, targets)}))
    _count(spend, "llm-sentence", answer)

    targets_by_id = {t.id: t for t in syllabus.targets}
    adopted = {s.text_sha for s in syllabus.sentences}
    questions: list[AssessQuestion] = []
    for draft in [d for item in answer.items for d in record.drafts_in(str(item))]:
        if draft.text_sha in adopted:
            continue
        filled = [t for t in _fills(ctx, draft, targets_by_id, spend) if t.id in open_ids]
        if not filled:
            continue
        questions.append(AssessQuestion(
            subject=draft.text_sha, role=role_for("sentence"), artifact_sha=None,
            rubric=ctx.rubrics[role_for("sentence")],
            params={"text": draft.text, "gloss": draft.gloss,
                    "word": syllabus.word(filled[0].word).thai},
            kind="sentence", subject_kind="sentence"))
    result = ctx.assessor.ask_many("judge", questions)
    _count_verdicts(spend, "judge", result)
    return AttemptResult(attempted=True, questions=list(result.collected),
                         excluded=dict(result.excluded), spend=spend,
                         drafted=len(questions))


_ATTEMPTS: dict[str, Callable[[Sourcing, Need, str], AttemptResult]] = {
    "picture": _picture_attempt,
    "recording": _recording_attempt,
    "rendition": _rendition_attempt,
}


def attempt(ctx: Sourcing, need: Need, source: str) -> AttemptResult:
    """One Source asked for one need, under the need's own subject."""
    make = _ATTEMPTS.get(need.kind)
    if make is None:
        raise ValueError(f"no attempt is defined for artifact kind {need.kind!r} "
                         f"(subject {need.subject!r}, a {need.subject_kind})")
    return make(ctx, need, source)
