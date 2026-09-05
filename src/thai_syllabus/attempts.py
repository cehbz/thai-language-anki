"""Attempts (spec 3 section 5): one Source tried for one need -- assess
every candidate on record (cache-first), then, if that alone hasn't
already improved current-best, fetch new candidates from the Source,
assess again over the whole candidate set, and re-derive current-best.
Every provide/assess goes through the ports, so each step appends and
the attempt is kill-safe. run.py iterates needs and sources; this module
knows what an attempt IS for each need kind.

An unavailable Source (Provider.ask raising KeyError -- no backend
configured for it -- or TransportError) is skipped: that source is
unusable this run, nothing is cached, the attempt may still succeed on
what's already on record. An unavailable OR unreachable Assessor is
different: nothing can be judged at all, so it ends the whole attempt
before any Source is tried, rather than spending on a search whose
results could never be assessed. Three shapes mean that, all handled the
same way (see `_judge_many`): Assessor.ask_many raising KeyError (no
"judge" backend registered), raising TransportError (the batch branch's
submit failed), or coming back with nothing resolved, nothing pending
AND nothing excluded for a non-empty question list (the inline branch
swallowed a per-question TransportError on every question). Each is
logged at WARNING -- a run that judged nothing must say so, not look
like a run that found nothing.

A question the judge EXCLUDED (ManyResult.excluded: it could not be
prepared -- an artifact sha resolving to no file) is not that. The judge
is reachable and answered what it could; that one candidate is unusable.
It is named at WARNING and skipped, and the attempt continues to the
Source.

Both shapes are also COUNTED, not just logged: Outcome/SentenceOutcome
carry `excluded` (how many questions the judge could not prepare, deduped
per attempt) and `unreachable` (the judge ended this attempt), so the run
report can say what went wrong instead of looking like a quiet run that
found nothing.

Outcome.spend counts one ask per backend unless the answer was served
from the cache, and the PORT says which: every ProviderAnswer/Verdict
carries its own `hit`. (Comparing an answer's `ts` against a start
timestamp cannot: a verdict this same attempt wrote a moment ago is
newer than the start, so every re-read of it counted as another ask.)
Outcome.attempted is true whenever spend records any real ask, fit or
preference included.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .assessor import AssessQuestion, Assessor
from .authority import ROLE_FOR_KIND
from .cachekeys import CacheKey, MechanicalKey
from .cachekeys import sha as _sha
from .derivations import CurrentBest, current_best
from .entities import Sentence, Target, text_sha
from .ids import WordId
from .media import Provenance, Speaker
from .provider import Provider, ProviderAnswer, Question
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus
from .transport import TransportError
from .tts import pick_voice

__all__ = ["Need", "Sourcing", "Outcome", "SentenceOutcome", "SOURCES", "sources_for",
           "candidates_of", "current_best_of", "attempt", "sentence_attempt", "select_cover"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Need:
    subject: str
    kind: str          # picture | recording | rendition | sentence | grapheme-keyword


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
    tts_voices: Sequence[str] = ("default",)
    judge_model: str = "llm"


@dataclass(frozen=True)
class Outcome:
    """`excluded` counts the questions the judge could not PREPARE during
    this attempt (deduped by cache key: one unusable candidate is one
    exclusion even though an attempt judges its candidate set twice, before
    and after the Source). `unreachable` says the judge could not be
    reached at all and ended the attempt -- the run stops on it rather than
    grinding through every remaining need against a dead wire.
    """
    attempted: bool                    # any real (non-cached) ask ran, provide or assess
    pending: bool
    improved: bool
    spend: dict[str, tuple[int, float]] = field(default_factory=dict)  # backend -> (asks, cost)
    excluded: int = 0
    unreachable: bool = False


@dataclass(frozen=True)
class SentenceOutcome:
    drafted: int
    adopted: tuple[Sentence, ...]
    pending: bool
    spend: dict[str, tuple[int, float]] = field(default_factory=dict)
    excluded: int = 0
    unreachable: bool = False


@dataclass
class _Tally:
    """One attempt's judge bookkeeping: the cache keys the judge could not
    prepare, as a SET -- an attempt asks the same fit question twice (once
    before the Source, once after), and one unusable candidate is one
    exclusion, not two.
    """
    excluded_keys: set[CacheKey] = field(default_factory=set)

    @property
    def excluded(self) -> int:
        return len(self.excluded_keys)


SOURCES: dict[str, tuple[str, ...]] = {
    "picture": ("openverse", "wikimedia", "pexels"),
    "recording": ("forvo", "tts"),
    "rendition": ("forvo", "tts"),
}


def sources_for(kind: str) -> tuple[str, ...]:
    return SOURCES.get(kind, ())


def current_best_of(ctx: Sourcing, subject: str, kind: str) -> CurrentBest:
    return current_best(ctx.db, subject, kind, current_rubric=ctx.rubrics,
                        provenance_prior=ctx.provenance_prior,
                        provenance=ctx.db.media_provenance)


def candidates_of(db: SyllabusDb, subject: str, kind: str) -> list[str]:
    """Every artifact sha on record for the need: provide-row items with a
    sha (any backend, this kind) plus a migrated machine-chosen marker.
    judge-batch-pending:* marker rows carry no sha and match neither
    branch below, so they are ignored on their own; de-duplicated as the
    list is built.
    """
    shas: list[str] = []
    for r in db.assessments_of(subject):
        if r.port == "provide" and r.question.get("provides") in (kind, f"{kind}-bytes"):
            for item in r.answer.get("items", []):
                if isinstance(item, Mapping) and item.get("sha") and item["sha"] not in shas:
                    shas.append(item["sha"])
        elif r.port == "assess" and r.backend == "machine-chosen" and kind == "picture":
            s = r.answer.get("sha")
            if s and s not in shas:
                shas.append(s)
    return shas


def _add(spend: dict[str, tuple[int, float]], backend: str, cost: float, hit: bool = False) -> None:
    asks, total = spend.get(backend, (0, 0.0))
    spend[backend] = (asks + (0 if hit else 1), total + float(cost or 0.0))


def _attempted(spend: dict[str, tuple[int, float]]) -> bool:
    return any(asks for asks, _ in spend.values())


def _word(ctx: Sourcing, subject: str):
    return ctx.syllabus.find_word(WordId(subject))


def _phrase(ctx: Sourcing, subject: str) -> str | None:
    """Latest learner direction wins; else a judge suggestion newer than the
    last provide; else None (the caller falls back to the meaning)."""
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


def _picture_params(ctx: Sourcing, subject: str) -> dict[str, Any]:
    w = _word(ctx, subject)
    return {"word": w.thai if w else subject, "meaning": w.meaning if w else "",
            "gloss_shown": w.meaning if w else "", "phrase": _phrase(ctx, subject)}


def _judge_many(ctx: Sourcing, questions: Sequence[AssessQuestion], tally: _Tally):
    """Assessor.ask_many("judge", ...) under the unreachable-judge
    contract (module docstring). The judge is UNREACHABLE when ask_many
    raises TransportError, or when resolved, pending AND excluded are all
    empty for a non-empty question list -- nothing came back and nothing
    was even attempted, so no verdict can be had; logged at WARNING and
    raised as TransportError for the caller to end the attempt on.

    An EXCLUDED question is the opposite case: the judge is reachable and
    answered what it could, but that question could not be prepared (an
    artifact sha resolving to no file). Those candidates are unusable, not
    evidence of a dead wire -- named at WARNING and skipped, and the
    attempt carries on to the Source. Ending it instead would wedge the
    subject: the stale row that names the missing artifact is still on
    record on every later run.

    An empty question list is the availability probe and answers nothing
    -- never unreachable.
    """
    try:
        res = ctx.assessor.ask_many("judge", questions)
    except TransportError as e:
        _log.warning("judge unreachable (%s); ending the attempt before any source spend", e)
        raise
    if res.excluded:
        tally.excluded_keys.update(res.excluded)
        _log.warning("the judge could not prepare %d of %d question(s) (%s); "
                     "those candidates are unusable -- continuing the attempt",
                     len(res.excluded), len(questions),
                     ", ".join(k.encode() for k in res.excluded))
    if questions and not res.resolved and not res.pending and not res.excluded:
        _log.warning("judge answered none of %d question(s), reported none pending and "
                     "excluded none; ending the attempt before any source spend",
                     len(questions))
        raise TransportError("the judge resolved no questions and reported none pending")
    return res


def _fit_pictures(ctx: Sourcing, subject: str, shas: Sequence[str],
                  spend: dict[str, tuple[int, float]], tally: _Tally) -> tuple[list[str], bool]:
    """Fits every sha (cache-first; may be empty -- ask_many still probes
    backend availability, so an unregistered judge backend's KeyError
    surfaces even with zero candidates). Returns (passing shas, pending).
    A KeyError (no "judge" backend at all) or a TransportError (the judge
    is unreachable, see _judge_many) propagates -- the caller ends the
    whole attempt over either, see attempt()'s module doc.
    """
    rubric = ctx.rubrics["picture-for-word"]
    params = _picture_params(ctx, subject)
    questions = [AssessQuestion(subject=subject, role="picture-for-word", artifact_sha=s,
                                rubric=rubric, params=params) for s in shas]
    res = _judge_many(ctx, questions, tally)
    for v in res.resolved.values():
        _add(spend, "judge", v.cost, hit=v.hit)
    if res.pending:
        return [], True
    passing = sorted(s for s, q in zip(shas, questions)
                     if (v := res.resolved.get(ctx.assessor.key_of("judge", q))) is not None
                     and v.value is True)
    return passing, False


def _prefer_pictures(ctx: Sourcing, subject: str, passing: Sequence[str],
                     spend: dict[str, tuple[int, float]], tally: _Tally) -> bool:
    """One preference question over every currently-passing candidate --
    cache-first, so an unchanged passing set (same shas, same rubric)
    costs nothing. Returns True when pending in a batch.
    """
    params = _picture_params(ctx, subject)
    pref = AssessQuestion(subject=subject, role="picture-preference", rubric=ctx.rubrics["picture-preference"],
                          params={"candidates": sorted(passing), "word": params["word"],
                                  "meaning": params["meaning"]})
    res = _judge_many(ctx, [pref], tally)
    for v in res.resolved.values():
        _add(spend, "judge", v.cost, hit=v.hit)
    return bool(res.pending)


def _assess_all_candidates(ctx: Sourcing, subject: str,
                           spend: dict[str, tuple[int, float]], tally: _Tally) -> bool:
    """Fits every candidate_of() on record, then -- if two or more pass --
    asks one preference question over the WHOLE passing set, not just
    whatever was newly fit (both cache-first, so an attempt that changed
    nothing costs nothing). Returns True when a verdict is pending in a
    batch. Raises KeyError when the judge backend is not registered at
    all, or TransportError when it is registered but unreachable (an
    unavailable/unreachable Assessor, distinct from an unavailable
    Source -- see the module docstring); the caller ends the attempt over
    either.
    """
    shas = candidates_of(ctx.db, subject, "picture")
    passing, is_pending = _fit_pictures(ctx, subject, shas, spend, tally)
    if is_pending:
        return True
    if len(passing) < 2:
        return False
    return _prefer_pictures(ctx, subject, passing, spend, tally)


def _picture_attempt(ctx: Sourcing, need: Need, source: str) -> Outcome:
    spend: dict[str, tuple[int, float]] = {}
    tally = _Tally()
    before = current_best_of(ctx, need.subject, need.kind)
    try:
        if _assess_all_candidates(ctx, need.subject, spend, tally):
            return Outcome(attempted=_attempted(spend), pending=True, improved=False, spend=spend,
                           excluded=tally.excluded)
    except KeyError:   # no "judge" backend registered at all
        return Outcome(attempted=False, pending=False, improved=False, spend=spend,
                       excluded=tally.excluded)
    except TransportError:   # registered but unreachable -- the run stops on this
        return Outcome(attempted=False, pending=False, improved=False, spend=spend,
                       excluded=tally.excluded, unreachable=True)
    if current_best_of(ctx, need.subject, need.kind).rank > before.rank:
        return Outcome(attempted=_attempted(spend), pending=False, improved=True, spend=spend,
                       excluded=tally.excluded)

    w = _word(ctx, need.subject)
    query = _phrase(ctx, need.subject) or (w.meaning if w and w.meaning else need.subject)
    try:
        hits = ctx.provider.ask(source, Question(subject=need.subject, provides="picture",
                                                 params={"query": query}))
    except (TransportError, KeyError):
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend,
                       excluded=tally.excluded)
    _add(spend, source, hits.cost, hit=hits.hit)

    for item in [i for i in hits.items if isinstance(i, Mapping) and i.get("url")][:ctx.image_candidates]:
        url = item["url"]
        try:
            got = ctx.provider.ask("imgfetch", Question(subject=need.subject, provides="picture-bytes",
                                                        params={"url": url}))
        except (TransportError, KeyError):
            continue
        _add(spend, "imgfetch", got.cost, hit=got.hit)
        for fetched in got.items:
            sha, ext = fetched.get("sha"), fetched.get("ext", "jpg")
            if not sha:
                continue
            ctx.db.add_media(sha=sha, kind="picture", ext=ext, source=str(item.get("source", source)),
                             origin=str(item.get("origin") or url), licence=str(item.get("licence") or "unknown"),
                             acquired=ctx.today())

    try:
        if _assess_all_candidates(ctx, need.subject, spend, tally):
            return Outcome(attempted=_attempted(spend), pending=True, improved=False, spend=spend,
                           excluded=tally.excluded)
    except KeyError:
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend,
                       excluded=tally.excluded)
    except TransportError:
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend,
                       excluded=tally.excluded, unreachable=True)
    after = current_best_of(ctx, need.subject, need.kind)
    return Outcome(attempted=_attempted(spend), pending=False, improved=after.rank > before.rank,
                   spend=spend, excluded=tally.excluded)


def _assess_recordings(ctx: Sourcing, subject: str, shas: Sequence[str], role: str,
                       spend: dict[str, tuple[int, float]]) -> None:
    """Mechanically assesses every sha for one subject, cache-first, via
    ask_many -- so a missing "mechanical" backend's KeyError surfaces even
    with zero shas (same pattern as _fit_pictures/_assess_all_candidates:
    ask_many probes backend availability before anything else). A
    per-question TransportError (ffprobe missing) is dropped by ask_many's
    own inline loop, not fatal to the rest.
    """
    questions = [AssessQuestion(subject=subject, role=role, artifact_sha=s) for s in shas]
    res = ctx.assessor.ask_many("mechanical", questions)
    for v in res.resolved.values():
        _add(spend, "mechanical", v.cost, hit=v.hit)


def _assess_members(ctx: Sourcing, members: Mapping[str, str],
                    spend: dict[str, tuple[int, float]]) -> dict[str, bool]:
    """Mechanically assesses each pair member's own recording sha under
    the MEMBER's subject (never the pair id) -- so the member word's own
    "recording" current_best sees the verdict too (wiring._DbMediaIndex.
    rendition_provenance reads current_best(member, "recording")).
    Returns member -> pass/fail; a member whose question never resolved
    (dropped by a TransportError) counts as failing -- nothing to rank on.
    """
    role = ROLE_FOR_KIND["recording"]
    questions = {m: AssessQuestion(subject=m, role=role, artifact_sha=s) for m, s in members.items()}
    res = ctx.assessor.ask_many("mechanical", list(questions.values()))
    for v in res.resolved.values():
        _add(spend, "mechanical", v.cost, hit=v.hit)
    return {m: bool(v.value) if (v := res.resolved.get(ctx.assessor.key_of("mechanical", q))) is not None else False
           for m, q in questions.items()}


def _forvo_items(ctx: Sourcing, subject: str, thai: str,
                 spend: dict[str, tuple[int, float]]) -> list[Mapping]:
    ans = ctx.provider.ask("forvo", Question(subject=subject, provides="recording", params={"word": thai}))
    _add(spend, "forvo", ans.cost, hit=ans.hit)
    return [i for i in ans.items if isinstance(i, Mapping) and i.get("pathmp3") and i.get("username")]


def _store_recording(ctx: Sourcing, got: ProviderAnswer, *, source: str, origin: str, licence: str,
                     speaker_id: str, speaker_kind: str) -> str | None:
    """The shared "first item with a sha -> add_media -> return its sha"
    loop shared by _download_forvo and _synthesize. Records the speaker
    with what this attempt knows (kind only -- sex/age_band/region are
    unknown here) only on the path that also writes the media row
    referencing it -- no speaker row without a referencing media row.
    """
    for fetched in got.items:
        s = fetched.get("sha")
        if s:
            ctx.db.add_speaker(Speaker(id=speaker_id, kind=speaker_kind))
            ctx.db.add_media(sha=s, kind="recording", ext=str(fetched.get("ext", "mp3")), source=source,
                             origin=origin, licence=licence, acquired=ctx.today(),
                             speaker_id=speaker_id)
            return s
    return None


def _download_forvo(ctx: Sourcing, subject: str, item: Mapping,
                    spend: dict[str, tuple[int, float]]) -> str | None:
    """A TransportError from audiofetch is a skip (this one url failed);
    a KeyError (no "audiofetch" backend registered) propagates to the
    caller's Source guard rather than being swallowed here.
    """
    url, user = item["pathmp3"], item["username"]
    try:
        got = ctx.provider.ask("audiofetch", Question(subject=subject, provides="recording-bytes",
                                                      params={"url": url, "speaker": user,
                                                              "speaker_kind": "native", "source": "forvo"}))
    except TransportError:
        return None
    _add(spend, "audiofetch", got.cost, hit=got.hit)
    return _store_recording(ctx, got, source="forvo", origin=url, licence="forvo",
                            speaker_id=f"forvo:{user}", speaker_kind="native")


def _synthesize(ctx: Sourcing, subject: str, thai: str, voice: str | None,
                spend: dict[str, tuple[int, float]]) -> str | None:
    """A TransportError from tts is a skip; a KeyError (no "tts" backend
    registered) propagates to the caller's Source guard.
    """
    params: dict[str, Any] = {"text": thai}
    if voice:
        params["voice"] = voice
    try:
        got = ctx.provider.ask("tts", Question(subject=subject, provides="recording", params=params))
    except TransportError:
        return None
    _add(spend, "tts", got.cost, hit=got.hit)
    used_voice = got.items[0].get("voice") if got.items else None
    return _store_recording(ctx, got, source="tts", origin=str(used_voice or ""), licence="google-tts",
                            speaker_id=str(used_voice or "tts"), speaker_kind="synthetic")


def _recording_attempt(ctx: Sourcing, need: Need, source: str) -> Outcome:
    spend: dict[str, tuple[int, float]] = {}
    role = ROLE_FOR_KIND["recording"]
    before = current_best_of(ctx, need.subject, "recording")
    try:
        _assess_recordings(ctx, need.subject, candidates_of(ctx.db, need.subject, "recording"), role, spend)
    except KeyError:
        return Outcome(attempted=False, pending=False, improved=False, spend=spend)
    if current_best_of(ctx, need.subject, "recording").rank > before.rank:
        return Outcome(attempted=_attempted(spend), pending=False, improved=True, spend=spend)

    w = _word(ctx, need.subject)
    thai = w.thai if w else need.subject
    new: list[str] = []
    try:
        if source == "forvo":
            for item in _forvo_items(ctx, need.subject, thai, spend):
                s = _download_forvo(ctx, need.subject, item, spend)
                if s:
                    new.append(s)
        elif source == "tts":
            # production draws from the configured (male) pool, spec 3
            # section 5 -- never the backend's own unqualified default.
            s = _synthesize(ctx, need.subject, thai,
                            pick_voice(need.subject, list(ctx.tts_voices)), spend)
            if s:
                new.append(s)
    except (TransportError, KeyError):
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend)

    try:
        _assess_recordings(ctx, need.subject, new, role, spend)
    except KeyError:
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend)
    after = current_best_of(ctx, need.subject, "recording")
    return Outcome(attempted=_attempted(spend), pending=False, improved=after.rank > before.rank, spend=spend)


def _record_rendition(ctx: Sourcing, pair_id: str, members: Mapping[str, str],
                      passed: Mapping[str, bool], context: str) -> None:
    """value is True only if every member's mechanical verdict passed;
    otherwise False, with the failing members named in evidence. This
    row's own `params["members"]` is what wiring._DbMediaIndex.
    rendition_provenance reads FIRST (the shas that actually made up
    THIS rendition attempt) -- it falls back to each member's own
    current-best recording only when no such pair-level row is
    current-best yet.
    """
    joined = ",".join(members[m] for m in sorted(members))
    ok = all(passed.values())
    evidence = context if ok else (
        f"{context}; failing: {', '.join(sorted(m for m, p in passed.items() if not p))}")
    artifact_sha = _sha(joined)
    key = MechanicalKey(check="rendition", params=f"v1:{pair_id}", artifact_sha=artifact_sha)
    ctx.db.append(port="assess", backend="mechanical",
                  key=key, subject=pair_id,
                  question={"role": ROLE_FOR_KIND["rendition"], "artifact_sha": artifact_sha,
                            "rubric": None, "params": {"members": dict(members)}},
                  answer={"value": ok, "evidence": evidence})


def _rendition_attempt(ctx: Sourcing, need: Need, source: str) -> Outcome:
    spend: dict[str, tuple[int, float]] = {}
    pair = next((p for p in ctx.syllabus.pairs if p.id == need.subject), None)
    if pair is None:
        return Outcome(attempted=False, pending=False, improved=False, spend=spend)
    try:
        ctx.assessor.ask_many("mechanical", [])  # probe: unavailable ends before any Source is tried
    except KeyError:
        return Outcome(attempted=False, pending=False, improved=False, spend=spend)

    before = current_best_of(ctx, need.subject, "rendition")
    words = {m: ctx.syllabus.word(WordId(m)) for m in pair.members}
    members: dict[str, str] = {}
    try:
        if source == "forvo":
            items = {m: _forvo_items(ctx, m, words[m].thai, spend) for m in pair.members}
            common = (set.intersection(*[{i["username"] for i in its} for its in items.values()])
                      if items else set())
            for user in sorted(common):
                members = {}
                for m in pair.members:
                    item = next(i for i in items[m] if i["username"] == user)
                    s = _download_forvo(ctx, m, item, spend)
                    if s:
                        members[m] = s
                if len(members) == len(pair.members):
                    passed = _assess_members(ctx, members, spend)
                    _record_rendition(ctx, need.subject, members, passed, f"speaker forvo:{user}")
                    break
                members = {}
        elif source == "tts":
            voice = pick_voice(need.subject, list(ctx.tts_voices))
            for m in pair.members:
                s = _synthesize(ctx, m, words[m].thai, voice, spend)
                if s:
                    members[m] = s
            if len(members) == len(pair.members):
                passed = _assess_members(ctx, members, spend)
                _record_rendition(ctx, need.subject, members, passed, f"voice {voice}")
    except (TransportError, KeyError):
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend)

    after = current_best_of(ctx, need.subject, "rendition")
    return Outcome(attempted=_attempted(spend), pending=False, improved=after.rank > before.rank, spend=spend)


_ATTEMPTS = {"picture": _picture_attempt, "recording": _recording_attempt,
            "rendition": _rendition_attempt}


def attempt(ctx: Sourcing, need: Need, source: str) -> Outcome:
    fn = _ATTEMPTS.get(need.kind)
    if fn is None:
        return Outcome(attempted=False, pending=False, improved=False, spend={})
    return fn(ctx, need, source)


# --- sentence attempt (spec 3 section 5): draft over open targets, verify
# with fills(), judge, adopt by greedy set cover -----------------------------

def _strip_fences(text: str) -> str:
    return re.sub(r"^```[a-z]*\n|\n```$", "", text.strip())


def _sentence_prompt(syl: Syllabus, targets: Sequence[Target]) -> str:
    lines = []
    for t in targets:
        w = syl.word(t.word)
        met = ", ".join(x.thai for x in syl.vocabulary_met_by(t))
        lines.append(f"- target {t.id}: word {w.thai} ({w.meaning}); may use: {met}")
    openings = sorted({syl.tokenizer.tokens(s.text)[0] for s in syl.sentences
                       if syl.tokenizer.tokens(s.text)})
    return ("Draft flashcard sentences in colloquial Central Thai for a learner whose register is "
           f"{syl.profile.register}.\n"
           "Write one short sentence per target, or one sentence covering several targets when their "
           "permitted vocabularies allow it. Use only the listed words for each target; every other "
           "word in a sentence must appear in that target's 'may use' list.\n"
           + (f"Avoid starting with any of: {', '.join(openings)}.\n" if openings else "")
           + "Targets:\n" + "\n".join(lines) + "\n"
           'Output JSON only: {"sentences": [{"text": "...", "targets": ["<target id>", ...]}]}')


def _drafts_from(text: str) -> list[dict]:
    try:
        data = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for d in (data.get("sentences") if isinstance(data, dict) else []) or []:
        if isinstance(d, dict) and d.get("text"):
            out.append({"text": str(d["text"]).strip(), "targets": list(d.get("targets") or []),
                       "gloss": str(d.get("gloss") or "")})
    return out


def _adopt(ctx: Sourcing, s: Sentence, model: str) -> None:
    ctx.db.add_sentence(text_sha=s.text_sha, text=s.text, gloss=s.gloss, voice=s.voice,
                        source="llm", origin=model, licence="generated", acquired=ctx.today())


def select_cover(passing: Sequence[tuple[Sentence, Sequence[Target]]],
                 open_targets: set[str]) -> list[Sentence]:
    """Greedy set cover: adopt the draft filling the most still-open
    targets (ties: shorter text) until no draft fills an open target."""
    uncovered = set(open_targets)
    remaining = list(passing)
    chosen: list[Sentence] = []
    while remaining:
        best = max(remaining, key=lambda sf: (len({t.id for t in sf[1]} & uncovered), -len(sf[0].text)))
        gain = {t.id for t in best[1]} & uncovered
        if not gain:
            break
        chosen.append(best[0])
        uncovered -= gain
        remaining.remove(best)
    return chosen


def _mechanical_fills(ctx: Sourcing, syl: Syllabus, targets_by_id: Mapping[str, Target],
                      s: Sentence, ts: str, draft: Mapping[str, Any], state_id: str) -> list[Target]:
    """fills() on every syllabus target for this draft, cached by text_sha
    -- a text's fills result cannot change under an unchanged Syllabus
    state, so the row is written once per (text, state_id) and every
    later run with that same state_id reads it back instead of
    re-verifying and re-appending a row for the identical question.
    """
    key = MechanicalKey(check="fills", params="v1", artifact_sha=ts)
    cached = ctx.db.latest("assess", "mechanical", key)
    if cached is not None and cached.question.get("params", {}).get("state_id") == state_id:
        return [targets_by_id[i] for i in cached.question["params"].get("fills", []) if i in targets_by_id]
    filled = [t for t in syl.targets if syl.fills(s, t)]
    ctx.db.append(port="assess", backend="mechanical", key=key, subject=ts,
                 question={"role": "sentence-for-target", "artifact_sha": ts, "rubric": None,
                          "params": {"fills": [t.id for t in filled],
                                    "claimed": [c for c in draft.get("targets", []) if c in targets_by_id],
                                    "state_id": state_id}},
                 answer={"value": bool(filled),
                        "evidence": f"fills {len(filled)} target(s)" if filled else "fills no target"})
    return filled


def _verify_and_judge(ctx: Sourcing, syl: Syllabus, drafts: Sequence[dict], model: str,
                      spend: dict[str, tuple[int, float]], tally: _Tally
                      ) -> tuple[list[Sentence], bool]:
    """fills() on every draft (recorded as a mechanical row regardless of
    outcome; see _mechanical_fills), then one judge question per draft
    that fills at least one currently-open target -- a draft that only
    fills already-covered targets is worthless to select_cover and costs
    no judge spend -- and adopts passes by greedy set cover over the
    currently-open targets. An unreachable judge raises TransportError out
    of _judge_many -- sentence_attempt ends over it, exactly as the other
    attempts do (module docstring).
    """
    targets_by_id = {t.id: t for t in syl.targets}
    known = {s.text_sha for s in ctx.db.all_sentences()}
    open_targets = set(syl.gaps().unfilled_targets)
    state_id = syl.state_id()
    candidates: list[tuple[Sentence, AssessQuestion]] = []
    filled_by_candidate: list[list[Target]] = []
    for d in drafts:
        text = d["text"]
        ts = text_sha(text)
        if ts in known:
            continue
        s = Sentence(text=text, gloss=d.get("gloss", ""), voice="learner_voice",
                    provenance=Provenance(source="llm", origin=model, licence="generated",
                                          acquired=ctx.today()))
        filled = _mechanical_fills(ctx, syl, targets_by_id, s, ts, d, state_id)
        if not filled:
            continue
        filled_open = [t for t in filled if t.id in open_targets]
        if not filled_open:
            continue
        filled_by_candidate.append(filled_open)
        # artifact_sha=None: a sentence judgment is text-only (the text
        # rides in params, read by sentence_prompt) -- not an artifact
        # attachment. JudgeBackend.cache_key/attachments both fall back to
        # `subject` (=ts) when artifact_sha is None, so the cache key is
        # unchanged; setting artifact_sha=ts here instead made attachments()
        # try to resolve ts as a media sha and raise TransportError, silently
        # dropping every sentence-for-target question from ask_many's result.
        candidates.append((s, AssessQuestion(
            subject=ts, role="sentence-for-target", artifact_sha=None,
            rubric=ctx.rubrics["sentence-for-target"],
            params={"text": text, "word": syl.word(filled_open[0].word).thai})))
    if not candidates:
        return [], False
    res = _judge_many(ctx, [q for _, q in candidates], tally)
    for v in res.resolved.values():
        _add(spend, "judge", v.cost, hit=v.hit)
    passing: list[tuple[Sentence, list[Target]]] = []
    for (s, q), filled in zip(candidates, filled_by_candidate):
        v = res.resolved.get(ctx.assessor.key_of("judge", q))
        if v is not None and v.value is True:
            passing.append((s, filled))
    adopted = select_cover(passing, open_targets)
    for s in adopted:
        _adopt(ctx, s, model)
    return adopted, bool(res.pending)


def _prior_drafts(ctx: Sourcing) -> list[dict]:
    """Every draft any earlier run's "llm-sentence" provide row produced,
    re-parsed -- how a batch verdict that lands after drafting gets
    adopted on a later run without re-drafting.
    """
    out: list[dict] = []
    for r in ctx.db.assessments_of("run"):
        if r.port == "provide" and r.backend == "llm-sentence":
            for item in r.answer.get("items", []):
                out.extend(_drafts_from(str(item)))
    return out


def sentence_attempt(ctx: Sourcing, *, max_targets: int = 40) -> SentenceOutcome:
    """Probes the judge first (no "judge" backend registered ends the
    attempt before any draft or Source ask, same convention as
    _rendition_attempt's mechanical probe); adopts any already-passing
    prior draft; then, while targets remain open, drafts new sentences,
    verifies with fills(), judges, and adopts by greedy set cover (spec 3
    section 5). `ctx.syllabus` itself is never mutated -- a local working
    Syllabus tracks this run's own adoptions for gaps()/the prompt, and
    the run applies `with_sentences(out.adopted)` to make it durable.

    A judge that is registered but unreachable ends the attempt the same
    way the unregistered one does -- before the drafting ask when the
    prior-draft pass discovers it, and adopting nothing (keeping whatever
    the drafting ask already spent) when the post-drafting pass does.
    """
    spend: dict[str, tuple[int, float]] = {}
    tally = _Tally()
    try:
        ctx.assessor.ask_many("judge", [])
    except KeyError:
        return SentenceOutcome(drafted=0, adopted=(), pending=False, spend={})
    model = ctx.judge_model
    syl = ctx.syllabus
    try:
        adopted, pending = _verify_and_judge(ctx, syl, _prior_drafts(ctx), model, spend, tally)
    except TransportError:
        return SentenceOutcome(drafted=0, adopted=(), pending=False, spend=spend,
                               excluded=tally.excluded, unreachable=True)
    if adopted:
        syl = syl.with_sentences(adopted)
    open_ids = set(syl.gaps().unfilled_targets[:max_targets])
    targets = [t for t in syl.targets if t.id in open_ids]
    if not targets:
        return SentenceOutcome(drafted=0, adopted=tuple(adopted), pending=pending, spend=spend,
                               excluded=tally.excluded)
    try:
        ans = ctx.provider.ask("llm-sentence", Question(subject="run", provides="sentence",
                                                        params={"prompt": _sentence_prompt(syl, targets)}))
    except (TransportError, KeyError):
        return SentenceOutcome(drafted=0, adopted=tuple(adopted), pending=pending, spend=spend,
                               excluded=tally.excluded)
    _add(spend, "llm-sentence", ans.cost, hit=ans.hit)
    drafts = [d for item in ans.items for d in _drafts_from(str(item))]
    try:
        more, pending2 = _verify_and_judge(ctx, syl, drafts, model, spend, tally)
    except TransportError:
        return SentenceOutcome(drafted=len(drafts), adopted=tuple(adopted), pending=pending,
                               spend=spend, excluded=tally.excluded, unreachable=True)
    adopted += more
    return SentenceOutcome(drafted=len(drafts), adopted=tuple(adopted), pending=pending or pending2,
                           spend=spend, excluded=tally.excluded)
