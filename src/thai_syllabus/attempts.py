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
what's already on record. An unavailable Assessor (Assessor.ask_many
raising KeyError -- no "judge" backend registered) is different: nothing
can be judged at all, so it ends the whole attempt before any Source is
tried, rather than spending on a search whose results could never be
assessed.

Outcome.spend counts one ask per backend unless the answer was served
from the cache: `attempt` captures a start timestamp before doing
anything, and an answer whose `ts` predates that start was already on
record (a hit, free); everything newer was actually asked this attempt.
Outcome.attempted is true whenever spend records any real ask, fit or
preference included.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .assessor import AssessQuestion, Assessor, ROLE_FOR_KIND
from .cachekeys import sha as _sha
from .derivations import CurrentBest, current_best
from .ids import WordId
from .provider import Provider, ProviderAnswer, Question
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus
from .transport import TransportError
from .tts import pick_voice

__all__ = ["Need", "Sourcing", "Outcome", "SOURCES", "sources_for", "candidates_of",
           "current_best_of", "attempt"]


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
    attempted: bool                    # any real (non-cached) ask ran, provide or assess
    pending: bool
    improved: bool
    spend: dict[str, tuple[int, float]] = field(default_factory=dict)  # backend -> (asks, cost)


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


def _fit_pictures(ctx: Sourcing, subject: str, shas: Sequence[str],
                  spend: dict[str, tuple[int, float]], start: int) -> tuple[list[str], bool]:
    """Fits every sha (cache-first; may be empty -- ask_many still probes
    backend availability, so an unregistered judge backend's KeyError
    surfaces even with zero candidates). Returns (passing shas, pending).
    A TransportError is treated as a transient miss (nothing judged, not
    pending); a KeyError (no "judge" backend at all) propagates -- the
    caller ends the whole attempt over it, see attempt()'s module doc.
    """
    rubric = ctx.rubrics["picture-for-word"]
    params = _picture_params(ctx, subject)
    questions = [AssessQuestion(subject=subject, role="picture-for-word", artifact_sha=s,
                                rubric=rubric, params=params) for s in shas]
    try:
        res = ctx.assessor.ask_many("judge", questions)
    except TransportError:
        return [], False
    for v in res.resolved.values():
        _add(spend, "judge", v.cost, hit=v.ts < start)
    if res.pending:
        return [], True
    passing = sorted(s for s, q in zip(shas, questions)
                     if (v := res.resolved.get(ctx.assessor.key_of("judge", q))) is not None
                     and v.value is True)
    return passing, False


def _prefer_pictures(ctx: Sourcing, subject: str, passing: Sequence[str],
                     spend: dict[str, tuple[int, float]], start: int) -> bool:
    """One preference question over every currently-passing candidate --
    cache-first, so an unchanged passing set (same shas, same rubric)
    costs nothing. Returns True when pending in a batch.
    """
    params = _picture_params(ctx, subject)
    pref = AssessQuestion(subject=subject, role="picture-preference", rubric=ctx.rubrics["picture-preference"],
                          params={"candidates": sorted(passing), "word": params["word"],
                                  "meaning": params["meaning"]})
    try:
        res = ctx.assessor.ask_many("judge", [pref])
    except TransportError:
        return False
    for v in res.resolved.values():
        _add(spend, "judge", v.cost, hit=v.ts < start)
    return bool(res.pending)


def _assess_all_candidates(ctx: Sourcing, subject: str, spend: dict[str, tuple[int, float]],
                           start: int) -> bool:
    """Fits every candidate_of() on record, then -- if two or more pass --
    asks one preference question over the WHOLE passing set, not just
    whatever was newly fit (both cache-first, so an attempt that changed
    nothing costs nothing). Returns True when a verdict is pending in a
    batch. Raises KeyError when the judge backend is not registered at
    all (an unavailable Assessor, distinct from an unavailable Source --
    see the module docstring); the caller ends the attempt over it.
    """
    shas = candidates_of(ctx.db, subject, "picture")
    passing, is_pending = _fit_pictures(ctx, subject, shas, spend, start)
    if is_pending:
        return True
    if len(passing) < 2:
        return False
    return _prefer_pictures(ctx, subject, passing, spend, start)


def _picture_attempt(ctx: Sourcing, need: Need, source: str, start: int) -> Outcome:
    spend: dict[str, tuple[int, float]] = {}
    before = current_best_of(ctx, need.subject, need.kind)
    try:
        if _assess_all_candidates(ctx, need.subject, spend, start):
            return Outcome(attempted=_attempted(spend), pending=True, improved=False, spend=spend)
    except KeyError:
        return Outcome(attempted=False, pending=False, improved=False, spend=spend)
    if current_best_of(ctx, need.subject, need.kind).rank > before.rank:
        return Outcome(attempted=_attempted(spend), pending=False, improved=True, spend=spend)

    w = _word(ctx, need.subject)
    query = _phrase(ctx, need.subject) or (w.meaning if w and w.meaning else need.subject)
    try:
        hits = ctx.provider.ask(source, Question(subject=need.subject, provides="picture",
                                                 params={"query": query}))
    except (TransportError, KeyError):
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend)
    _add(spend, source, hits.cost, hit=hits.ts < start)

    for item in [i for i in hits.items if isinstance(i, Mapping) and i.get("url")][:ctx.image_candidates]:
        url = item["url"]
        try:
            got = ctx.provider.ask("imgfetch", Question(subject=need.subject, provides="picture-bytes",
                                                        params={"url": url}))
        except (TransportError, KeyError):
            continue
        _add(spend, "imgfetch", got.cost, hit=got.ts < start)
        for fetched in got.items:
            sha, ext = fetched.get("sha"), fetched.get("ext", "jpg")
            if not sha:
                continue
            ctx.db.add_media(sha=sha, kind="picture", ext=ext, source=str(item.get("source", source)),
                             origin=str(item.get("origin") or url), licence=str(item.get("licence") or "unknown"),
                             acquired=ctx.today())

    try:
        if _assess_all_candidates(ctx, need.subject, spend, start):
            return Outcome(attempted=_attempted(spend), pending=True, improved=False, spend=spend)
    except KeyError:
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend)
    after = current_best_of(ctx, need.subject, need.kind)
    return Outcome(attempted=_attempted(spend), pending=False, improved=after.rank > before.rank, spend=spend)


def _assess_recordings(ctx: Sourcing, subject: str, shas: Sequence[str], role: str,
                       spend: dict[str, tuple[int, float]], start: int) -> None:
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
        _add(spend, "mechanical", v.cost, hit=v.ts < start)


def _assess_members(ctx: Sourcing, members: Mapping[str, str],
                    spend: dict[str, tuple[int, float]], start: int) -> dict[str, bool]:
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
        _add(spend, "mechanical", v.cost, hit=v.ts < start)
    return {m: bool(v.value) if (v := res.resolved.get(ctx.assessor.key_of("mechanical", q))) is not None else False
           for m, q in questions.items()}


def _forvo_items(ctx: Sourcing, subject: str, thai: str,
                 spend: dict[str, tuple[int, float]], start: int) -> list[Mapping]:
    ans = ctx.provider.ask("forvo", Question(subject=subject, provides="recording", params={"word": thai}))
    _add(spend, "forvo", ans.cost, hit=ans.ts < start)
    return [i for i in ans.items if isinstance(i, Mapping) and i.get("pathmp3") and i.get("username")]


def _store_recording(ctx: Sourcing, got: ProviderAnswer, *, source: str, origin: str, licence: str,
                     speaker_id: str, speaker_kind: str) -> str | None:
    """The shared "first item with a sha -> add_media -> return its sha"
    loop shared by _download_forvo and _synthesize.
    """
    for fetched in got.items:
        s = fetched.get("sha")
        if s:
            ctx.db.add_media(sha=s, kind="recording", ext=str(fetched.get("ext", "mp3")), source=source,
                             origin=origin, licence=licence, acquired=ctx.today(),
                             speaker_id=speaker_id, speaker_kind=speaker_kind)
            return s
    return None


def _download_forvo(ctx: Sourcing, subject: str, item: Mapping,
                    spend: dict[str, tuple[int, float]], start: int) -> str | None:
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
    _add(spend, "audiofetch", got.cost, hit=got.ts < start)
    return _store_recording(ctx, got, source="forvo", origin=url, licence="forvo",
                            speaker_id=f"forvo:{user}", speaker_kind="native")


def _synthesize(ctx: Sourcing, subject: str, thai: str, voice: str | None,
                spend: dict[str, tuple[int, float]], start: int) -> str | None:
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
    _add(spend, "tts", got.cost, hit=got.ts < start)
    used_voice = got.items[0].get("voice") if got.items else None
    return _store_recording(ctx, got, source="tts", origin=str(used_voice or ""), licence="google-tts",
                            speaker_id=str(used_voice or "tts"), speaker_kind="synthetic")


def _recording_attempt(ctx: Sourcing, need: Need, source: str, start: int) -> Outcome:
    spend: dict[str, tuple[int, float]] = {}
    role = ROLE_FOR_KIND["recording"]
    before = current_best_of(ctx, need.subject, "recording")
    try:
        _assess_recordings(ctx, need.subject, candidates_of(ctx.db, need.subject, "recording"), role, spend, start)
    except KeyError:
        return Outcome(attempted=False, pending=False, improved=False, spend=spend)
    if current_best_of(ctx, need.subject, "recording").rank > before.rank:
        return Outcome(attempted=_attempted(spend), pending=False, improved=True, spend=spend)

    w = _word(ctx, need.subject)
    thai = w.thai if w else need.subject
    new: list[str] = []
    try:
        if source == "forvo":
            for item in _forvo_items(ctx, need.subject, thai, spend, start):
                s = _download_forvo(ctx, need.subject, item, spend, start)
                if s:
                    new.append(s)
        elif source == "tts":
            s = _synthesize(ctx, need.subject, thai, None, spend, start)
            if s:
                new.append(s)
    except (TransportError, KeyError):
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend)

    try:
        _assess_recordings(ctx, need.subject, new, role, spend, start)
    except KeyError:
        return Outcome(attempted=_attempted(spend), pending=False, improved=False, spend=spend)
    after = current_best_of(ctx, need.subject, "recording")
    return Outcome(attempted=_attempted(spend), pending=False, improved=after.rank > before.rank, spend=spend)


def _record_rendition(ctx: Sourcing, pair_id: str, members: Mapping[str, str],
                      passed: Mapping[str, bool], context: str) -> None:
    """value is True only if every member's mechanical verdict passed;
    otherwise False, with the failing members named in evidence.
    """
    joined = ",".join(members[m] for m in sorted(members))
    ok = all(passed.values())
    evidence = context if ok else (
        f"{context}; failing: {', '.join(sorted(m for m, p in passed.items() if not p))}")
    ctx.db.append(port="assess", backend="mechanical",
                  key=f"mech:rendition:v1:{pair_id}:{_sha(joined)}", subject=pair_id,
                  question={"role": ROLE_FOR_KIND["rendition"], "artifact_sha": _sha(joined),
                            "rubric": None, "params": {"members": dict(members)}},
                  answer={"value": ok, "evidence": evidence})


def _rendition_attempt(ctx: Sourcing, need: Need, source: str, start: int) -> Outcome:
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
            items = {m: _forvo_items(ctx, m, words[m].thai, spend, start) for m in pair.members}
            common = (set.intersection(*[{i["username"] for i in its} for its in items.values()])
                      if items else set())
            for user in sorted(common):
                members = {}
                for m in pair.members:
                    item = next(i for i in items[m] if i["username"] == user)
                    s = _download_forvo(ctx, m, item, spend, start)
                    if s:
                        members[m] = s
                if len(members) == len(pair.members):
                    passed = _assess_members(ctx, members, spend, start)
                    _record_rendition(ctx, need.subject, members, passed, f"speaker forvo:{user}")
                    break
                members = {}
        elif source == "tts":
            voice = pick_voice(need.subject, list(ctx.tts_voices))
            for m in pair.members:
                s = _synthesize(ctx, m, words[m].thai, voice, spend, start)
                if s:
                    members[m] = s
            if len(members) == len(pair.members):
                passed = _assess_members(ctx, members, spend, start)
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
    return fn(ctx, need, source, time.time_ns())
