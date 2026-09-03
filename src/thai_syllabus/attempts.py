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

from .assessor import AssessQuestion, Assessor
from .derivations import CurrentBest, current_best
from .ids import WordId
from .provider import Provider, Question
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus
from .transport import TransportError

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


_ATTEMPTS = {"picture": _picture_attempt}   # Task 8 adds recording and rendition


def attempt(ctx: Sourcing, need: Need, source: str) -> Outcome:
    fn = _ATTEMPTS.get(need.kind)
    if fn is None:
        return Outcome(attempted=False, pending=False, improved=False, spend={})
    return fn(ctx, need, source, time.time_ns())
