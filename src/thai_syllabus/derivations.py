"""Derivations (spec 3 section 6): pure folds over the cache, never
stored. current_best, pending, next_source, exhausted, improved, directed,
queue, challengers, reasks, confusion_weights.

record.py holds the row-selecting folds (rows_for, source_asks,
candidate_shas, learner_ratings, directions, judge_verdicts, ratings_for_role,
latest_query, unresolved_batch); every row a provide/assess writer appends
names its need kind, or a learner row's own row kind, in question["kind"]
(record.py's docstring). This module reads through record.py only -- no
`provide` row's `provides` string or `assess` row's `role` string is
matched by membership or prefix here, and no cache key is ever parsed.

`current_rubric` is the role -> rubric text mapping rulebook.rubrics_for
produces; it is a required keyword argument everywhere it appears (no
None form) -- a role absent from the mapping is never stale on that
account (`stale`).

`cache`/`syllabus` are always the first parameters: every function here
is a pure fold over an injected CacheReader.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from . import record
from .authority import AUTHORITY_ORDER, role_for
from .entities import Sentence, Syllable, Target, is_corroborated
from .ids import WordId
from .media import Provenance, Speaker
from .phonology import syllables_from_verdict
from .ports import Answer, CacheReader, StudyReader, StudyRecord
from .record import LEARNER_RANK
from .syllabus import Syllabus

_log = logging.getLogger(__name__)

__all__ = [
    "CurrentBest", "current_best", "learner_ranks", "vetoed",
    "role_of", "adoptable_drafts", "refused_drafts",
    "JudgeVerdict", "judge_verdict", "deciding_verdict",
    "pending", "adjudications",
    "attempts_since_change", "tried_sources", "next_source",
    "ExhaustedStatus", "exhausted", "sentence_exhausted",
    "improved",
    "directed",
    "QueueEntry", "queue", "QueuedNeeds", "queued",
    "all_needs", "available_needs", "available_need_keys", "open_words",
    "passing_pictures", "pictures_awaiting_preference",
    "unjudged_candidates",
    "Challenger", "challengers",
    "Reask", "reasks", "DEFAULT_REASK_LAPSES",
    "confusion_weights",
    "LEARNER_RANK",
    "stale",
    "DEFAULT_ATTEMPT_CAP",
    "DEFAULT_TRANSIENT_CAP",
    "DEFAULT_SENTENCE_NOTHING_CAP",
]

# LEARNER_RANK (record.py): a numeric rank on the same scale judge
# verdicts use, so on a role where the learner ranks (AUTHORITY_ORDER
# names "learner") current_best's regression guard ("never below an
# artifact the learner rated acceptable") is a plain numeric comparison.
# Judge pass (True/1.0) ranks below learner "acceptable" so a judge run
# alone can never outrank a learner's endorsement there (spec 3 section
# 6); on a veto-only role (spec 3 section 4 r8) no rating ranks at all.
_JUDGE_PASS_RANK = 50.0
_JUDGE_FAIL_RANK = 0.0
_GOOD_RANK = LEARNER_RANK["good"]

# The attempt cap exhausted() enforces, read from this one place: run.py
# and reviewserver.py pass it explicitly until providers.yaml wires a
# configured value.
DEFAULT_ATTEMPT_CAP = 8

# The transient-outcome cap tried_sources() enforces (spec 3 section 6a):
# a source at this many transient-failure outcomes since the anchor
# counts as tried.
DEFAULT_TRANSIENT_CAP = 3

# tried_sources()'s ageing window unit (spec 3 r19 section 6a/9): a
# nothing_ttl day, in the ts's own nanosecond units.
_NANOS_PER_DAY = 86_400 * 1_000_000_000

# The no-fit cap sentence_exhausted() enforces (spec 3 r19 section 5): a
# word whose sentence need has this many `nothing` outcome rows since its
# last handed draft is not handed to the drafter again, and the feedback
# screen asks the learner for a direction instead.
DEFAULT_SENTENCE_NOTHING_CAP = 3

# The one artifact kind with no Source (attempts.SOURCES has no entry for
# it) and no per-run pass either -- unlike "sentence", which the run's own
# sentence attempt serves. queued() counts its needs as unserved rather
# than entering, exhausting, or pending them.
_UNSERVED_KIND = "grapheme-keyword"

# The kind the run's own per-run sentence attempt serves for every open
# Target, directed or not: queued() emits no entry, exhausted count, or
# unserved count for it (open_words is its own accounting).
_RUN_SENTENCE_KIND = "sentence"


def _judge_rank(value) -> float:
    if isinstance(value, bool):
        return _JUDGE_PASS_RANK if value else _JUDGE_FAIL_RANK
    if isinstance(value, (int, float)):
        return float(value)
    return _JUDGE_FAIL_RANK


def _ratings_by_artifact(rating_rows: Sequence[Answer]) -> dict[str, tuple[int, str]]:
    """artifact_sha -> (latest ts, rating), newest wins per artifact, over
    an already role-scoped list of rating rows (record.ratings_for_role).
    """
    out: dict[str, tuple[int, str]] = {}
    for r in rating_rows:
        artifact_sha = r.question.get("artifact_sha") or r.answer.get("artifact_sha")
        if not artifact_sha:
            continue
        prev = out.get(artifact_sha)
        if prev is None or r.ts > prev[0]:
            out[artifact_sha] = (r.ts, r.answer.get("value"))
    return out


def _stale(row: Answer, current_rubric: Mapping[str, str]) -> bool:
    """True when a verdict row's rubric no longer matches the mapping's
    entry for its own role. A role absent from `current_rubric` is never
    stale on that account. A rubric is a judge parameter: every other
    backend carries none (mechanical and the other ground-truth checks
    answer about the artifact, the learner answers about the role), so a
    rubric change can never make one of their rows stale.
    """
    if row.question.get("rubric") is None and row.backend != "judge":
        return False
    role = row.question.get("role")
    if role in current_rubric:
        return row.question.get("rubric") != current_rubric[role]
    return False


# Public export: a caller reading verdict rows outside this module applies
# the same role-scoped staleness (and the same mechanical-row exemption)
# this module's own folds use.
stale = _stale


def _machine_ranks(rows: Sequence[Answer], kind: str, role: str,
                   current_rubric: Mapping[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    """Machine rank per artifact (spec 3 section 6): the first backend in
    AUTHORITY_ORDER[role] (bar "learner") with a verdict row decides that
    artifact's rank -- on a role the learner ranks, current_best folds the
    learner's own rating in on top of this; on a veto-only role (spec 3
    section 4 r8) this is the whole ranking, the learner only vetoes.
    Of one backend's several verdicts on an artifact, the newest ranks
    (spec 3 section 6). Returns (ranks, deciding backend per sha);
    pictures also fold in preference-row bonuses.
    """
    order = [b for b in AUTHORITY_ORDER.get(role, ("judge",)) if b != "learner"]
    by_backend: dict[str, dict[str, float]] = {}
    for r in sorted(rows, key=lambda r: r.ts):
        if r.port != "assess" or r.backend not in order or _stale(r, current_rubric):
            continue
        if r.question.get("role") != role:
            continue
        sha_ = r.question.get("artifact_sha")
        if not sha_:
            continue
        rank = _judge_rank(r.answer.get("value"))
        ranks = by_backend.setdefault(r.backend, {})
        ranks[sha_] = rank
    out: dict[str, float] = {}
    sources: dict[str, str] = {}
    shas = sorted({s for ranks in by_backend.values() for s in ranks})
    for s in shas:
        for backend in order:               # most authoritative first
            if s in by_backend.get(backend, {}):
                out[s] = by_backend[backend][s]
                sources[s] = backend
                break
    if kind == "picture":
        _apply_preference(rows, out, current_rubric)
    return out, sources


def _apply_preference(rows: Sequence[Answer], ranks: dict[str, float],
                      current_rubric: Mapping[str, str]) -> None:
    """The newest picture-preference row under the current rubric whose
    candidates all pass adds a positional bonus (spec 3 section 6),
    20.0 * (n - 1 - i) / max(n - 1, 1) at rank position i.
    """
    passing = {s for s, r in ranks.items() if r > _JUDGE_FAIL_RANK}
    prefs = [r for r in record.judge_verdicts(rows, "picture-preference")
            if not _stale(r, current_rubric)
            and set(r.question.get("params", {}).get("candidates", [])) <= passing]
    if not prefs:
        return
    newest = max(prefs, key=lambda r: r.ts)
    ranking = [s for s in newest.answer.get("value", []) if s in passing]
    n = len(ranking)
    for i, s in enumerate(ranking):
        ranks[s] += 20.0 * (n - 1 - i) / max(n - 1, 1)


def _apply_prior(ranks: dict[str, float], prior: Sequence[str],
                 provenance_source: Callable[[str], str | None]) -> None:
    """Provenance-prior tie-break (spec 3 section 6): a passing
    artifact's rank += (len(prior) - index) / (len(prior) + 1), index
    being `provenance_source(sha)`'s position in `prior`; a source absent
    from `prior` gets no bonus. The bonus is under 1.0, so it breaks ties
    only, and `provenance_source` reads the `media` table's own `source`.
    """
    if not prior:
        return
    for s, r in list(ranks.items()):
        if r <= _JUDGE_FAIL_RANK:
            continue
        source = provenance_source(s)
        if source is None:
            continue
        try:
            idx = list(prior).index(source)
        except ValueError:
            continue
        ranks[s] = r + (len(prior) - idx) / (len(prior) + 1)


def _speaker_for(rows: Sequence[Answer], artifact_sha: str | None) -> Speaker | None:
    """The Speaker a provide row's item carries alongside `artifact_sha`
    (spec 3 section 2's compound rendition answer: `{items: [{member,
    sha, speaker}, ...]}`) -- None when no row's item names one.
    """
    if artifact_sha is None:
        return None
    for r in rows:
        if r.port != "provide":
            continue
        for item in r.answer.get("items", []):
            if not isinstance(item, Mapping) or item.get("sha") != artifact_sha:
                continue
            speaker = item.get("speaker")
            if isinstance(speaker, Mapping) and "id" in speaker and "kind" in speaker:
                return Speaker(id=speaker["id"], kind=speaker["kind"],
                              sex=speaker.get("sex", "unknown"),
                              age_band=speaker.get("age_band", "unknown"),
                              region=speaker.get("region", "unknown"))
    return None


# --- current_best -----------------------------------------------------------

@dataclass(frozen=True)
class CurrentBest:
    artifact_sha: str | None
    source: str | None   # "learner" | the deciding machine backend (e.g. "judge",
                         # "mechanical") | None when nothing is current-best
    rank: float
    speaker: Speaker | None = None


def role_of(cache: CacheReader, subject: str, kind: str,
            rows: Sequence[Answer] | None = None) -> str:
    """The Assess role this (subject, kind) is judged under: the kind's
    role for the kind of thing the subject is, as the subject's own rows
    name it (record.subject_kind_of).
    """
    rows = record.rows_for(cache, subject, kind) if rows is None else rows
    return role_for(kind, record.subject_kind_of(rows))


def learner_ranks(role: str) -> bool:
    """True when AUTHORITY_ORDER names "learner" for `role`: the learner's
    rating orders current_best's candidates there (picture-for-word,
    scene-for-sentence, sentence-for-target). False on a role where the
    learner only vetoes -- recording-for-word, recording-for-sentence,
    rendition-for-pair (spec 3 section 4 r8). A role absent from
    AUTHORITY_ORDER defaults to True, the pre-r8 behavior.
    """
    return "learner" in AUTHORITY_ORDER.get(role, ("learner",))


def _vetoed_shas(learner_ratings_by_artifact: Mapping[str, tuple[int, str]]) -> set[str]:
    """artifact shas whose latest learner rating is "unacceptable-none"
    (spec 3 section 4 r8): on a role the learner never ranks, that rating
    rejects the artifact_sha it names outright, excluding it from the
    machine candidate set until a newer learner rating on the same sha
    lifts it. "unacceptable-use-this" names the sha the learner wants
    used instead -- a candidate nomination, not a rejection, so it never
    vetoes; it still needs a machine verdict to rank, exactly like a
    supplied artifact. "acceptable"/"good" never veto either (they never
    rank on this role -- current_best's caller ignores them for ranking).
    """
    return {sha_ for sha_, (_ts, rating) in learner_ratings_by_artifact.items()
           if rating == "unacceptable-none"}


def vetoed(cache: CacheReader, subject: str, role: str, artifact_sha: str | None) -> bool:
    """True only when the latest learner rating naming `artifact_sha`
    under `role` is "unacceptable-none" (spec 3 section 4 r8) -- a
    rejection of that specific sha. "unacceptable-use-this" (a candidate
    nomination) and "acceptable"/"good" (never ranking on a veto-only
    role) are not a veto; neither is having no rating on `artifact_sha`
    at all. None for `artifact_sha` is never vetoed (no sha to reject).
    The one row fold every veto check reads (current_best's own
    _vetoed_shas folds the same rule over several shas at once).
    """
    if artifact_sha is None:
        return False
    on_sha = [r for r in record.ratings_for_role(cache.assessments_of(subject), role)
             if (r.question.get("artifact_sha") or r.answer.get("artifact_sha"))
             == artifact_sha]
    if not on_sha:
        return False
    return max(on_sha, key=lambda r: r.ts).answer.get("value") == "unacceptable-none"


def current_best(cache: CacheReader, subject: str, kind: str, *,
                 current_rubric: Mapping[str, str], prior: Sequence[str] = (),
                 provenance_source: Callable[[str], str | None]) -> CurrentBest:
    """The fold spec 3 section 6 defines, per (subject, kind)'s own role
    (AUTHORITY_ORDER[role]): where "learner" is named (picture-for-word,
    scene-for-sentence, sentence-for-target) the learner's rating wins
    outright, subject to the regression floor -- else the candidate the
    most authoritative backend that has spoken ranks highest, provenance
    prior among equals. A remaining tie goes to the lower artifact sha
    (spec 3 section 6). Where "learner" is not named (recording-for-word,
    recording-for-sentence, rendition-for-pair, spec 3 section 4 r8) the
    learner only vetoes: an "unacceptable-none" rating excludes its sha
    until a newer rating on the same sha lifts it; "unacceptable-use-this"
    names a candidate the machine still has to rank, and "acceptable"/
    "good" never rank either -- the returned source is always the
    machine backend, never "learner".
    """
    rows = record.rows_for(cache, subject, kind)
    role = role_of(cache, subject, kind, rows)
    rating_rows = record.ratings_for_role(cache.assessments_of(subject), role)
    learner_ratings = _ratings_by_artifact(rating_rows)
    machine_ranks, machine_sources = _machine_ranks(rows, kind, role, current_rubric)

    if not learner_ranks(role):
        # r8: the learner vetoes on this role and never ranks -- an
        # "unacceptable-none" rating excludes its sha from the machine
        # candidate set (and so from the provenance-prior tie-break
        # below); "unacceptable-use-this" nominates a sha the machine
        # still has to rank, and "acceptable"/"good" are recorded but
        # ignored here -- the source returned is always the machine
        # backend, never "learner".
        for sha_ in _vetoed_shas(learner_ratings):
            machine_ranks.pop(sha_, None)
        _apply_prior(machine_ranks, prior, provenance_source)
        passing = {s: r for s, r in machine_ranks.items() if r > _JUDGE_FAIL_RANK}
        if passing:
            best_sha = max(sorted(passing), key=passing.get)
            return CurrentBest(artifact_sha=best_sha, source=machine_sources.get(best_sha),
                               rank=passing[best_sha], speaker=_speaker_for(rows, best_sha))
        return CurrentBest(artifact_sha=None, source=None, rank=-1.0)

    _apply_prior(machine_ranks, prior, provenance_source)

    latest_learner_row = max(rating_rows, key=lambda r: r.ts, default=None)

    # regression floor: the best rating the learner has EVER given any
    # artifact for this (subject, kind), acceptable-or-better only.
    floor = max((LEARNER_RANK[rating] for _, rating in learner_ratings.values()
                if rating in ("good", "acceptable")), default=None)

    if latest_learner_row is not None:
        rating = latest_learner_row.answer.get("value")
        if rating == "unacceptable-none":
            if floor is not None:
                best_sha = max(learner_ratings,
                               key=lambda s: LEARNER_RANK[learner_ratings[s][1]])
                return CurrentBest(artifact_sha=best_sha, source="learner", rank=floor,
                                   speaker=_speaker_for(rows, best_sha))
            return CurrentBest(artifact_sha=None, source=None, rank=-1.0)
        artifact_sha = (latest_learner_row.question.get("artifact_sha")
                        or latest_learner_row.answer.get("artifact_sha"))
        rank = max(LEARNER_RANK[rating], floor if floor is not None else -1.0)
        return CurrentBest(artifact_sha=artifact_sha, source="learner", rank=rank,
                           speaker=_speaker_for(rows, artifact_sha))

    # Only a genuinely passing machine verdict (rank above the fail floor)
    # counts as a usable current_best -- an all-failing history must read
    # the same as "no candidate at all" (rank -1.0).
    passing = {s: r for s, r in machine_ranks.items() if r > _JUDGE_FAIL_RANK}
    if passing:
        best_sha = max(sorted(passing), key=passing.get)
        return CurrentBest(artifact_sha=best_sha, source=machine_sources.get(best_sha),
                           rank=passing[best_sha], speaker=_speaker_for(rows, best_sha))

    return CurrentBest(artifact_sha=None, source=None, rank=-1.0)


# --- the verdict a surface shows next to an artifact ---------------------

@dataclass(frozen=True)
class JudgeVerdict:
    artifact_sha: str
    passed: bool
    evidence: str | None = None
    backend: str = "judge"


def judge_verdict(cache: CacheReader, subject: str, kind: str, artifact_sha: str, *,
                  current_rubric: Mapping[str, str]) -> JudgeVerdict | None:
    """The newest judge verdict on `artifact_sha` under (subject, kind)'s
    role that is fresh under `current_rubric` -- the same freshness
    current_best ranks by. None when there is none.
    """
    rows = record.rows_for(cache, subject, kind)
    role = role_of(cache, subject, kind, rows)
    fresh = [r for r in record.judge_verdicts(rows, role)
            if r.question.get("artifact_sha") == artifact_sha and not _stale(r, current_rubric)]
    if not fresh:
        return None
    latest = max(fresh, key=lambda r: r.ts)
    return JudgeVerdict(artifact_sha=artifact_sha,
                        passed=_judge_rank(latest.answer.get("value")) > _JUDGE_FAIL_RANK,
                        evidence=latest.answer.get("evidence"))


def deciding_verdict(cache: CacheReader, subject: str, kind: str, artifact_sha: str, *,
                     current_rubric: Mapping[str, str]) -> JudgeVerdict | None:
    """The newest fresh verdict on `artifact_sha` by the backend that
    decides (subject, kind)'s role -- the first non-learner entry of
    AUTHORITY_ORDER[role] (spec 5 r7 section 1: what a rejected candidate
    shows the learner as its reason). None when that backend has none.
    """
    rows = record.rows_for(cache, subject, kind)
    role = role_of(cache, subject, kind, rows)
    order = [b for b in AUTHORITY_ORDER.get(role, ("judge",)) if b != "learner"]
    if not order:
        return None
    backend = order[0]
    mine = [r for r in rows if r.port == "assess" and r.backend == backend
            and r.question.get("role") == role
            and r.question.get("artifact_sha") == artifact_sha
            and not _stale(r, current_rubric)]
    if not mine:
        return None
    latest = max(mine, key=lambda r: r.ts)
    return JudgeVerdict(artifact_sha=artifact_sha,
                        passed=_judge_rank(latest.answer.get("value")) > _JUDGE_FAIL_RANK,
                        evidence=latest.answer.get("evidence"), backend=backend)


# --- pending -----------------------------------------------------------

def pending(cache: CacheReader, subject: str, kind: str) -> bool:
    """True while the newest submitted judge batch marker names a
    question asked for the (subject, kind) need itself
    (record.unresolved_batch) -- the key queued() reads the questions
    this run collected under. A word whose picture is in the batch still
    has its recording need queued. Resolving the batch releases the whole
    marker, so every need it named stops being pending together.
    """
    found = record.unresolved_batch(cache)
    if found is None:
        return False
    _batch_id, subjects, _roles, kinds = found
    return (subject, kind) in zip(subjects, kinds, strict=True)


# --- adjudications ------------------------------------------------------

def adjudications(cache: CacheReader, syllabus, *, current_rubric: Mapping[str, str]
                  ) -> dict[WordId, tuple[Syllable, ...]]:
    """Per word still lacking a corroborated pronunciation, the newest
    fresh pronunciation-for-word verdict's syllables (spec 3 r28); a
    word with none is absent. A verdict under a superseded rubric is not
    fresh (`stale`), so the word waits for the re-ask instead.

    What the run then does with the answer -- check it against the
    engines, write the corroborated ones to words.yaml -- is
    run._materialize_adjudications; this is the fold that says which
    verdicts are on offer.
    """
    out: dict[WordId, tuple[Syllable, ...]] = {}
    for w in syllabus.words:
        if is_corroborated(w.pron.corroboration):
            continue
        rows = [r for r in cache.assessments_of(str(w.id))
                if r.port == "assess" and r.backend == "judge"
                and r.question.get("role") == "pronunciation-for-word"
                and not _stale(r, current_rubric)]
        if not rows:
            continue
        value = max(rows, key=lambda r: r.ts).answer.get("value")
        if isinstance(value, Mapping) and value.get("syllables"):
            out[w.id] = syllables_from_verdict(value)
    return out


# --- next_source / attempts_since_change --------------------------------

def _no_provenance_source(artifact_sha: str) -> str | None:
    """provenance_source for a current_best() call with an empty `prior`,
    which never calls it.
    """
    return None


def _anchor_ts(cache: CacheReader, subject: str, kind: str, rows: Sequence[Answer]) -> int:
    """The newer of two tss (architecture section 4: any learner input
    reopens a need): the ts of the earliest attempt-outcome row (port
    `attempt`) whose candidates include current-best's artifact
    (current-best taken rubric-agnostically here -- escalation tracks
    when a candidate was produced, -1 while no artifact exists yet), and
    the newest learner rating row under the need's own role. A supply
    always carries its own implicit rating (append_supply): that rating
    row alone covers a supply's own reset, every kind of learner input
    resetting escalation the same way -- the source roster is asked
    again from the cheapest.
    """
    best = current_best(cache, subject, kind, current_rubric={}, prior=(),
                        provenance_source=_no_provenance_source)
    if best.artifact_sha is None:
        change_ts = -1
    else:
        producing = [r.ts for r in rows if r.port == "attempt"
                     and best.artifact_sha in (r.answer.get("candidates") or ())]
        change_ts = min(producing) if producing else -1

    role = role_of(cache, subject, kind, rows)
    rating_ts = max((r.ts for r in record.ratings_for_role(cache.assessments_of(subject), role)),
                    default=-1)
    return max(change_ts, rating_ts)


def attempts_since_change(cache: CacheReader, subject: str, kind: str) -> list[Answer]:
    """Attempt-outcome rows (port `attempt`) under (subject, kind) with ts
    greater than the ts of the row that produced current-best's artifact
    -- every such row counts when no artifact exists yet. Only
    `candidates` and `nothing` outcomes count as tried; a
    `transient-failure` outcome counts only at the transient cap (spec 3
    section 6a, tried_sources).
    """
    rows = record.rows_for(cache, subject, kind)
    since_ts = _anchor_ts(cache, subject, kind, rows)
    return [r for r in rows if r.port == "attempt" and r.ts > since_ts
            and r.answer.get("outcome") in ("candidates", "nothing")]


def tried_sources(cache: CacheReader, subject: str, kind: str, *,
                  transient_cap: int, nothing_ttl: Mapping[str, int] = {},
                  now_ns: int | None = None) -> frozenset[str]:
    """The sources tried since current-best last changed: those with a
    `candidates` outcome, those with a `nothing` outcome not aged out, and
    those with `transient_cap` `transient-failure` outcomes (spec 3
    section 6a). Ageing (spec 3 r19 section 6a/9): a `nothing` row from a
    source named in `nothing_ttl` (days, keyed by source) whose own `ts`
    is older than `now_ns` minus that many days no longer counts as
    tried on its own (the caller reads the clock once per pass; a ttl
    without `now_ns` is refused, the folds never read it) -- a
    growing corpus (Forvo) may have new hits by then. `candidates` rows
    and the transient-failure count are never aged; a source absent from
    `nothing_ttl` never ages either.
    """
    now_ns = _clock_for(nothing_ttl, now_ns)
    rows = record.rows_for(cache, subject, kind)
    since_ts = _anchor_ts(cache, subject, kind, rows)
    outcomes = [r for r in rows if r.port == "attempt" and r.ts > since_ts]

    def _fresh_nothing(r: Answer) -> bool:
        days = nothing_ttl.get(r.backend)
        if days is None:
            return True
        return r.ts >= now_ns - days * _NANOS_PER_DAY

    tried = {r.backend for r in outcomes if r.answer.get("outcome") == "candidates"
             or (r.answer.get("outcome") == "nothing" and _fresh_nothing(r))}
    transient: dict[str, int] = {}
    for r in outcomes:
        if r.answer.get("outcome") == "transient-failure":
            transient[r.backend] = transient.get(r.backend, 0) + 1
    tried.update(s for s, n in transient.items() if n >= transient_cap)
    return frozenset(tried)


def _clock_for(nothing_ttl: Mapping[str, int], now_ns: int | None) -> int:
    """The instant an ageing fold measures against: the caller's own
    `now_ns`, read once per pass (spec 3 r19 section 6a/9). A ttl with no
    clock is refused rather than read here; with no ttl the value is
    unused and 0 stands in."""
    if nothing_ttl and now_ns is None:
        raise ValueError("now_ns: a nothing_ttl needs the caller's clock (spec 3 section 6a)")
    return 0 if now_ns is None else now_ns


def aged_out(cache: CacheReader, subject: str, kind: str, source: str, *,
             nothing_ttl: Mapping[str, int], now_ns: int | None) -> bool:
    """Whether `source` is offered for (subject, kind) again only because
    its `nothing` answer aged out (spec 3 r19 section 6a): the newest
    outcome row from `source` since current-best last changed is a
    `nothing` older than the source's ttl. The attempt then asks the
    source afresh (Provider.reask) instead of reading its cached empty
    answer, which is what "a fresh lookup appends a new row" means. False
    when the source has no ttl, no row, a newer `candidates` or
    `transient-failure` row, or a `nothing` still inside its ttl.
    """
    days = nothing_ttl.get(source)
    if days is None:
        return False
    now_ns = _clock_for(nothing_ttl, now_ns)
    rows = record.rows_for(cache, subject, kind)
    since_ts = _anchor_ts(cache, subject, kind, rows)
    mine = [r for r in rows if r.port == "attempt" and r.ts > since_ts and r.backend == source]
    if not mine:
        return False
    newest = max(mine, key=lambda r: r.ts)
    return newest.answer.get("outcome") == "nothing" and newest.ts < now_ns - days * _NANOS_PER_DAY


def next_source(cache: CacheReader, subject: str, kind: str,
                sources: Sequence[str], *, transient_cap: int,
                nothing_ttl: Mapping[str, int] = {},
                now_ns: int | None = None) -> str | None:
    """The first of `sources` (cheapest first) not in tried_sources; None
    once every source is tried since current-best last changed. A source
    whose only `nothing` outcome has aged out of `nothing_ttl` (spec 3 r19
    section 6a/9) is offered again here.
    """
    tried = tried_sources(cache, subject, kind, transient_cap=transient_cap,
                          nothing_ttl=nothing_ttl, now_ns=now_ns)
    for source in sources:
        if source not in tried:
            return source
    return None


# --- exhausted ---------------------------------------------------------

@dataclass(frozen=True)
class ExhaustedStatus:
    exhausted: bool
    attempts: int


def _sentence_anchor_ts(cache: CacheReader, word: str) -> int:
    """The ts a word's sentence need is counted from: the newest learner
    row of ANY kind on the word, or -1 when it carries none.

    Spec 3 r19 section 6a reopens a sentence-exhausted word on "a learner
    row", and the row the feedback screen's direction question writes is
    a `direction` row -- it carries no rating value, so `_anchor_ts`'s
    own learner term (record.ratings_for_role, LEARNER_RANK-valued rating
    rows under the need's role) cannot see it. Its other term, the ts of
    the attempt that produced current-best's artifact, is always -1 here:
    a sentence need stores no artifact under the word. So `_anchor_ts`
    has nothing to add, and the newest learner row on the subject is the
    whole anchor -- architecture section 4's "any learner input reopens a
    need", read as widely as the direction question needs.
    """
    return max((r.ts for r in cache.assessments_of(word) if r.backend == "learner"),
              default=-1)


def sentence_exhausted(cache: CacheReader, word: str, *,
                       cap: int = DEFAULT_SENTENCE_NOTHING_CAP) -> ExhaustedStatus:
    """`word`'s sentence need after `cap` no-fit answers (spec 3 r19
    section 5): the `nothing` attempt-outcome rows under (word,
    "sentence") since `_sentence_anchor_ts`, one per handed draft the
    drafter answered "nothing fits" to. At the cap the word is not handed
    to the drafter again and the feedback screen asks the learner for a
    direction; a learner row on the word reopens it (section 6a), which
    is what the anchor carries.
    """
    rows = record.rows_for(cache, word, "sentence")
    since_ts = _sentence_anchor_ts(cache, word)
    nothing = [r for r in rows if r.port == "attempt" and r.ts > since_ts
              and r.answer.get("outcome") == "nothing"]
    return ExhaustedStatus(exhausted=len(nothing) >= cap, attempts=len(nothing))


def exhausted(cache: CacheReader, subject: str, kind: str, *,
              sources: Sequence[str], attempt_cap: int, transient_cap: int,
              sentence_nothing_cap: int = DEFAULT_SENTENCE_NOTHING_CAP,
              nothing_ttl: Mapping[str, int] = {},
              now_ns: int | None = None) -> ExhaustedStatus:
    """Every source in `sources` is tried since current-best last changed,
    or the attempt count since then reached `attempt_cap`; a source at
    the transient cap counts as one attempt. Reopened by a learner row, a
    new source, or -- spec 3 r19 section 6a/9, while attempts_since_change
    stays below `attempt_cap` (ageing re-offers a source; it does not
    forget the attempts already made) -- a `nothing` outcome
    ageing out of `nothing_ttl` (both passed through to tried_sources and
    next_source).

    Kind "sentence" has no Source roster of its own -- the run's own
    sentence attempt serves it -- so it is `sentence_exhausted` under
    `sentence_nothing_cap` instead (spec 3 r19 section 5).
    """
    if kind == _RUN_SENTENCE_KIND:
        return sentence_exhausted(cache, subject, cap=sentence_nothing_cap)
    since = attempts_since_change(cache, subject, kind)
    capped = tried_sources(cache, subject, kind, transient_cap=transient_cap,
                           nothing_ttl=nothing_ttl, now_ns=now_ns) - {
        r.backend for r in since}
    attempts = len(since) + len(capped)
    is_exhausted = (next_source(cache, subject, kind, sources, transient_cap=transient_cap,
                                nothing_ttl=nothing_ttl, now_ns=now_ns) is None
                    or attempts >= attempt_cap)
    return ExhaustedStatus(exhausted=is_exhausted, attempts=attempts)


# --- improved ------------------------------------------------------------

def improved(before: CurrentBest, after: CurrentBest) -> bool:
    """A changed artifact -- a re-ranking among unchanged artifacts is
    never improvement (spec 3 section 7).
    """
    return after.artifact_sha is not None and after.artifact_sha != before.artifact_sha


# --- directed ------------------------------------------------------------

def directed(cache: CacheReader, subject: str) -> bool:
    """True when `subject` carries a learner direction row, an unconsumed
    reverify row (no mechanical/listener verdict on its own role newer
    than it), or a card-flag row.
    """
    rows = cache.assessments_of(subject)
    if any(r.backend == "learner" and r.question.get("kind") == "direction" for r in rows):
        return True
    if any(r.question.get("kind") == "card-flag" for r in rows):
        return True
    for r in rows:
        if r.question.get("kind") != "reverify":
            continue
        role = r.question.get("role")
        answered = any(m.port == "assess" and m.backend in ("mechanical", "listener")
                      and m.question.get("role") == role and m.ts > r.ts
                      for m in rows)
        if not answered:
            return True
    return False


def _has_untried_lever(cache: CacheReader, subject: str, kind: str, rows: Sequence[Answer],
                       current_rubric: Mapping[str, str], sources: Sequence[str], *,
                       transient_cap: int, nothing_ttl: Mapping[str, int] = {},
                       now_ns: int | None = None) -> bool:
    """A candidate has no verdict under the current rubric, a judge
    suggestion has not been followed by a new attempt, or an unasked
    source remains (spec 3 section 6 bucket 2) -- including one a
    `nothing` outcome aged back out of `nothing_ttl` (spec 3 r19 section
    6a/9). "Followed by a new attempt" is measured against the last
    Source ask (record.last_source_ask_ts), never a provide row that is
    an answer rather than an ask -- attempts.phrase_attempt's own
    per-subject phrase row included (fix round 2 finding 1): a phrase
    drafted after a pending suggestion must not hide it from bucket 2.
    """
    judge_rows = [r for r in rows if r.port == "assess" and r.backend == "judge"]
    if unjudged_candidates(cache, subject, kind, current_rubric=current_rubric):
        return True
    provide_ts = record.last_source_ask_ts(rows)
    if any(r.answer.get("suggestion") and r.ts > provide_ts for r in judge_rows):
        return True
    return next_source(cache, subject, kind, sources, transient_cap=transient_cap,
                       nothing_ttl=nothing_ttl, now_ns=now_ns) is not None


# --- queue: F10 order ----------------------------------------------------

@dataclass(frozen=True)
class QueueEntry:
    subject: str
    kind: str
    # What `subject` is (word | pair | sentence | grapheme) -- the attempt
    # and the role both turn on it, and an id alone does not say.
    subject_kind: str = "word"
    bucket: int = 3   # 1 = no-artifact/unacceptable, 2 = untried lever, 3 = acceptable/unrated
    directed: bool = False
    rank: float = 0.0
    attempts: int = 0


def available_needs(syllabus) -> list[tuple[str, str, str]]:
    """(subject, artifact kind, subject kind) per gap. A sentence's
    recording and scene picture carry a word's artifact kinds, and are
    told apart by their subject kind.
    """
    gaps = syllabus.gaps()
    target_word = {t.id: t.word for t in syllabus.targets}
    candidates: list[tuple[str, str, str]] = []
    candidates += [(w, "picture", "word") for w in gaps.words_missing_pictures]
    candidates += [(w, "recording", "word") for w in gaps.words_missing_recordings]
    candidates += [(target_word.get(t, t), "sentence", "word") for t in gaps.unfilled_targets]
    # gaps.missing_renditions names ConfusionIds; attempts._rendition_attempt
    # looks a pair up by PairId. The need's subject is the pair's own id,
    # not the confusion it covers.
    candidates += [(p.id, "rendition", "pair") for p in syllabus.pairs
                  if p.confusion in gaps.missing_renditions]
    candidates += [(s, "recording", "sentence") for s in gaps.sentence_recordings]
    candidates += [(s, "picture", "sentence") for s in gaps.scene_pictures]
    candidates += [(g, "grapheme-keyword", "grapheme")
                  for g in gaps.graphemes_missing_keyword_data]
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def open_words(syllabus) -> frozenset[str]:
    """The words with a Target still unfilled -- the subject of every
    "sentence" need `available_needs` lists, one per word however many
    Targets that word has, which is the unit the run's own sentence
    attempt is accounted for in (run.RunReport).
    """
    return frozenset(subject for subject, kind, _subject_kind in available_needs(syllabus)
                    if kind == _RUN_SENTENCE_KIND)


def all_needs(syllabus) -> list[tuple[str, str, str]]:
    """(subject, artifact kind, subject kind) for every need the deck has,
    satisfied or not (spec 5 section 3's coverage universe): one picture
    and one recording need per targeted word (once, however many Targets
    name it), one rendition per pair, one keyword picture per grapheme,
    one recording and one scene picture per sentence.
    """
    seen_words: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for t in syllabus.targets:
        if t.word in seen_words:
            continue
        seen_words.add(t.word)
        out.append((t.word, "picture", "word"))
        out.append((t.word, "recording", "word"))
    for p in syllabus.pairs:
        out.append((p.id, "rendition", "pair"))
    for g in syllabus.graphemes:
        out.append((g.symbol, "grapheme-keyword", "grapheme"))
    for s in syllabus.sentences:
        out.append((s.text_sha, "recording", "sentence"))
        out.append((s.text_sha, "picture", "sentence"))
    return out


@dataclass(frozen=True)
class QueuedNeeds:
    """queue()'s entries and what the same pass left out: `available` is
    every need gaps() lists, `exhausted` those among them out of sources
    with nothing directing them, `unserved` those whose kind has no
    Source and no per-run pass either. `available` equals `exhausted` +
    `unserved` + `entries`, "sentence" needs aside: those are the run's
    own sentence attempt to serve and account for
    (run.RunReport.attempted/budgeted/deferred).
    """
    entries: list[QueueEntry]
    available: int
    exhausted: int
    unserved: int = 0


def available_need_keys(syllabus) -> frozenset[tuple[str, str]]:
    """Every (subject, artifact kind) `available_needs` names -- one
    member per need `available` counts (run.RunReport). A batch's own
    (subject, kind) questions intersect it need for need.
    """
    return frozenset((subject, kind) for subject, kind, _subject_kind
                     in available_needs(syllabus))


def queue(syllabus, cache: CacheReader, *, current_rubric: Mapping[str, str],
         prior: Sequence[str], sources_for: Callable[[str], Sequence[str]],
         attempt_cap: int, transient_cap: int, provenance_source: Callable[[str], str | None],
         collected_this_run: frozenset[tuple[str, str]] = frozenset(),
         nothing_ttl: Mapping[str, int] = {},
         now_ns: int | None = None) -> list[QueueEntry]:
    return queued(syllabus, cache, current_rubric=current_rubric, prior=prior,
                  sources_for=sources_for, attempt_cap=attempt_cap,
                  transient_cap=transient_cap,
                  provenance_source=provenance_source,
                  collected_this_run=collected_this_run,
                  nothing_ttl=nothing_ttl, now_ns=now_ns).entries


def queued(syllabus, cache: CacheReader, *, current_rubric: Mapping[str, str],
          prior: Sequence[str], sources_for: Callable[[str], Sequence[str]],
          attempt_cap: int, transient_cap: int, provenance_source: Callable[[str], str | None],
          collected_this_run: frozenset[tuple[str, str]] = frozenset(),
          nothing_ttl: Mapping[str, int] = {},
          now_ns: int | None = None) -> QueuedNeeds:
    """queue()'s entries plus the counts the same pass left out.
    `collected_this_run` names the (subject, kind) needs this run already
    collected a question for; each is skipped like an already-pending
    need. It is keyed by kind, so a word's other open needs stay queued.
    `nothing_ttl`/`now_ns` (spec 3 r19 section 6a/9) reach exhausted()'s
    and _has_untried_lever()'s own next_source/tried_sources folds, so a
    growing source's aged-out `nothing` reopens the need here too.
    """
    entries: list[QueueEntry] = []
    candidates = available_needs(syllabus)
    out_of_options = 0
    unserved = 0
    for subject, kind, subject_kind in candidates:
        if kind == _UNSERVED_KIND:
            # No Source serves this kind (attempts.SOURCES has no entry
            # for it) and, unlike "sentence", no per-run pass covers it
            # either: it can never become an entry, exhausted, or pending.
            unserved += 1
            continue
        if kind == _RUN_SENTENCE_KIND:
            continue
        if pending(cache, subject, kind) or (subject, kind) in collected_this_run:
            continue  # already has a question outstanding -- reported once, not queued again
        best = current_best(cache, subject, kind, current_rubric=current_rubric, prior=prior,
                            provenance_source=provenance_source)
        if best.rank >= _GOOD_RANK:
            continue  # good -- never queued

        rows = record.rows_for(cache, subject, kind)
        role = role_of(cache, subject, kind, rows)
        is_vetoed = vetoed(cache, subject, role, best.artifact_sha)
        is_directed = directed(cache, subject)
        sources = sources_for(kind)
        # attempts: the same count exhausted() reports -- a source at the
        # transient cap is one attempt (spec 3 section 6).
        # Never kind "sentence": queued() skips it above, so exhausted()'s
        # own sentence_nothing_cap never decides anything from here.
        status = exhausted(cache, subject, kind, sources=sources, attempt_cap=attempt_cap,
                          transient_cap=transient_cap, nothing_ttl=nothing_ttl, now_ns=now_ns)
        attempts = status.attempts

        if best.artifact_sha is None or is_vetoed:
            awaiting = unjudged_candidates(cache, subject, kind, current_rubric=current_rubric)
            if status.exhausted and not is_directed and not awaiting:
                out_of_options += 1
                continue  # out of machine options, nothing directs it, no
                          # candidate awaits a verdict -- excluded
            bucket = 1
        elif _has_untried_lever(cache, subject, kind, rows, current_rubric, sources,
                                transient_cap=transient_cap, nothing_ttl=nothing_ttl,
                                now_ns=now_ns):
            bucket = 2
        else:
            bucket = 3

        entries.append(QueueEntry(subject=subject, kind=kind, subject_kind=subject_kind,
                                  bucket=bucket, directed=is_directed, rank=best.rank,
                                  attempts=attempts))

    entries.sort(key=lambda e: (e.bucket, not e.directed, e.rank, e.attempts, e.subject, e.kind))
    return QueuedNeeds(entries=entries, available=len(candidates), exhausted=out_of_options,
                       unserved=unserved)


# --- assess-first: the candidates a verdict is owed to ---------------------

def unjudged_candidates(cache: CacheReader, subject: str, kind: str, *,
                        current_rubric: Mapping[str, str]) -> tuple[str, ...]:
    """`subject`'s candidates of `kind` with no verdict deciding them yet,
    in record.candidate_shas order (spec 3 section 5 assess-first).

    For a role whose deciding backend is mechanical
    (authority.AUTHORITY_ORDER[role][0] == "mechanical":
    recording-for-word, recording-for-sentence, r23) a candidate awaits
    when no mechanical verdict row under this subject names its sha --
    mechanical asks about the artifact itself, not a rubric, so a role
    absent from `current_rubric` still gets assessed here, and a params
    change is its own cache miss (re-asks by itself, nothing to track).

    Every other role is judge-decided: empty for a role absent from
    `current_rubric` (the judge ranks nothing there); else a candidate
    awaits when no judge verdict under the current rubric names its sha.
    """
    rows = record.rows_for(cache, subject, kind)
    role = role_of(cache, subject, kind, rows)
    if AUTHORITY_ORDER.get(role, ("judge",))[0] == "mechanical":
        judged = {r.question.get("artifact_sha") for r in rows
                 if r.port == "assess" and r.backend == "mechanical"}
        return tuple(s for s in record.candidate_shas(rows) if s not in judged)
    if role not in current_rubric:
        return ()
    judged = {r.question.get("artifact_sha") for r in record.judge_verdicts(rows, role)
              if not _stale(r, current_rubric)}
    return tuple(s for s in record.candidate_shas(rows) if s not in judged)


# --- the preference question a resolved batch leaves open ------------------

def passing_pictures(cache: CacheReader, subject: str, *,
                     current_rubric: Mapping[str, str]) -> tuple[str, ...]:
    """`subject`'s picture candidates whose fit verdict under the current
    rubric passed, sorted -- the set a preference question orders (spec 3
    section 6). The one place that set is decided: an attempt reads it
    back after its own fit questions resolved, and the run reads it back
    once a batch does.
    """
    rows = record.rows_for(cache, subject, "picture")
    role = role_of(cache, subject, "picture", rows)
    return tuple(sorted({r.question["artifact_sha"]
                        for r in record.judge_verdicts(rows, role)
                        if r.answer.get("value") is True and r.question.get("artifact_sha")
                        and not _stale(r, current_rubric)}))


def pictures_awaiting_preference(cache: CacheReader, subject: str, *,
                                 current_rubric: Mapping[str, str]) -> tuple[str, ...]:
    """`subject`'s pictures passing fit under the current rubric, when
    more than one passes and no preference verdict ranks exactly that
    set: the set to put to the judge. Empty otherwise.
    """
    passing = passing_pictures(cache, subject, current_rubric=current_rubric)
    if len(passing) < 2:
        return ()
    ranked = [r for r in record.judge_verdicts(cache.assessments_of(subject),
                                              "picture-preference")
             if not _stale(r, current_rubric)
             and set(r.question.get("params", {}).get("candidates", [])) == set(passing)]
    return () if ranked else passing


# --- challengers -----------------------------------------------------------

@dataclass(frozen=True)
class Challenger:
    """One challenge: the need it is about, the artifact the learner
    accepted, and the candidate now outranking it.
    """
    subject: str
    kind: str
    subject_kind: str
    current_sha: str
    challenger_sha: str


def challengers(cache: CacheReader, syllabus, *, current_rubric: Mapping[str, str],
                prior: Sequence[str],
                provenance_source: Callable[[str], str | None]) -> list[Challenger]:
    """Every need where a machine-ranked candidate under `current_rubric`
    outranks a learner-accepted artifact; presented, never auto-switched.
    """
    out: list[Challenger] = []
    for subject, kind, subject_kind in available_needs(syllabus):
        best = current_best(cache, subject, kind, current_rubric=current_rubric, prior=prior,
                            provenance_source=provenance_source)
        if best.source != "learner" or best.artifact_sha is None:
            continue
        rows = record.rows_for(cache, subject, kind)
        role = role_of(cache, subject, kind, rows)
        rating_rows = record.ratings_for_role(cache.assessments_of(subject), role)
        rated = set(_ratings_by_artifact(rating_rows))
        machine_ranks, _sources = _machine_ranks(rows, kind, role, current_rubric)
        _apply_prior(machine_ranks, prior, provenance_source)
        # Compared against the accepted artifact's OWN machine rank (0.0
        # when it has none), never against best.rank -- the learner's
        # rank floor is not a machine authority a candidate must clear.
        accepted_rank = machine_ranks.get(best.artifact_sha, _JUDGE_FAIL_RANK)
        candidates = [(sha, rank) for sha, rank in machine_ranks.items()
                     if sha not in rated and rank > accepted_rank]
        if not candidates:
            continue
        challenger_sha = max(candidates, key=lambda t: t[1])[0]
        out.append(Challenger(subject=subject, kind=kind, subject_kind=subject_kind,
                              current_sha=best.artifact_sha, challenger_sha=challenger_sha))
    return out


# --- reasks ----------------------------------------------------------------

# The re-ask lapse threshold (spec 5 section 1 kind 4) when rulebook.yaml's
# thresholds carries no "reask/lapses" entry -- one lapse is already the
# contradiction F9 re-asks over ("the evidence contradicts (shown)").
DEFAULT_REASK_LAPSES = 1

# The study card_kind whose StudyRecords exercise a given (subject_kind,
# need kind) need's artifact -- the same (family, card_kind) pairing
# anki_import.py's flag import reads a role from, read the other way.
_REASK_CARD_KIND: dict[tuple[str, str], str] = {
    ("word", "picture"): "production",
    ("word", "recording"): "listening",
    ("sentence", "recording"): "listening",
    ("sentence", "picture"): "cloze",
}


@dataclass(frozen=True)
class Reask:
    """A learner rating spec 5 section 1 kind 4 re-asks: F9's "the
    evidence contradicts" a rating of "acceptable" or better.
    `subject`/`kind`/`subject_kind` name the need as every derivation
    does (a rendition's subject is its pair id); `rating` is the
    contradicted answer, `evidence` the lapse StudyRecords, oldest first.
    """
    subject: str
    kind: str
    subject_kind: str
    rating: str
    evidence: tuple[StudyRecord, ...]


def _reask_candidate(cache: CacheReader, study: StudyReader, *, subject: str, kind: str,
                     subject_kind: str, family: str, anchors: Sequence[str], card_kind: str,
                     lapse_threshold: int) -> Reask | None:
    """None unless the need's own role has a newest rating that ranks
    "acceptable" or better (LEARNER_RANK) AND at least `lapse_threshold`
    lapse StudyRecords after it. The gate reads the newest role rating
    directly (record.ratings_for_role), never current_best: on a veto-only
    role (spec 3 section 4 r8) a rating below "acceptable"
    ("unacceptable-none" or "unacceptable-use-this") is that newest rating
    and fails the gate on its own, with no need to consult what
    current_best resolved to.
    """
    role = role_for(kind, subject_kind)
    ratings = record.ratings_for_role(cache.assessments_of(subject), role)
    if not ratings:
        return None
    latest = max(ratings, key=lambda r: r.ts)
    rating = latest.answer.get("value")
    if LEARNER_RANK.get(rating, -1.0) < LEARNER_RANK["acceptable"]:
        return None
    lapses = tuple(sorted((r for anchor in anchors
                          for r in study.records(family, anchor, card_kind) if r.grade <= 1),
                         key=lambda r: r.ts))
    if len(lapses) < lapse_threshold:
        return None
    return Reask(subject=subject, kind=kind, subject_kind=subject_kind, rating=rating,
                evidence=lapses)


def reasks(cache: CacheReader, study: StudyReader, syllabus, *,
          lapse_threshold: int = DEFAULT_REASK_LAPSES) -> list[Reask]:
    """Every need -- a word's picture or recording, a sentence's
    recording or scene picture, a pair's rendition -- rated "acceptable"
    or better whose card has since accumulated at least `lapse_threshold`
    lapses (spec 5 section 1 kind 4). Each need's StudyRecords come from
    (family, anchor, card_kind); a sentence's anchor is its own text_sha,
    the entity subject anki_import.py writes study rows under, one Reask
    per sentence.
    """
    out: list[Reask] = []
    for w in syllabus.words:
        for kind in ("picture", "recording"):
            found = _reask_candidate(cache, study, subject=w.id, kind=kind, subject_kind="word",
                                     family="word", anchors=(w.id,),
                                     card_kind=_REASK_CARD_KIND[("word", kind)],
                                     lapse_threshold=lapse_threshold)
            if found is not None:
                out.append(found)

    for s in syllabus.sentences:
        for kind in ("recording", "picture"):
            found = _reask_candidate(cache, study, subject=s.text_sha, kind=kind,
                                     subject_kind="sentence", family="sentence",
                                     anchors=(s.text_sha,),
                                     card_kind=_REASK_CARD_KIND[("sentence", kind)],
                                     lapse_threshold=lapse_threshold)
            if found is not None:
                out.append(found)

    for p in syllabus.pairs:
        found = _reask_candidate(cache, study, subject=p.id, kind="rendition",
                                 subject_kind="pair", family="minimal_pair",
                                 anchors=(p.id,),
                                 card_kind="recognition", lapse_threshold=lapse_threshold)
        if found is not None:
            out.append(found)

    return out


# --- confusion_weights ---------------------------------------------------

def confusion_weights(seed: Mapping[str, float], syllabus: Syllabus,
                      study: StudyReader) -> dict[str, float]:
    """curated seed x the aggregate's own study grouping (spec 3 section
    6), over `syllabus.confusions`. A grade <= 1 (Anki's "again") is a
    lapse; a confusion with no study history keeps its seed weight.
    """
    grouped = syllabus.study_by_confusion(study)
    weights: dict[str, float] = {}
    for confusion in syllabus.confusions:
        cid = confusion.id
        base = seed.get(cid, 1.0)
        records = grouped.get(cid, [])
        if not records:
            weights[cid] = base
            continue
        lapses = sum(1 for r in records if r.grade <= 1)
        lapse_rate = lapses / len(records)
        weights[cid] = base * (1.0 + lapse_rate)
    return weights


# --- adoptable_drafts -------------------------------------------------------

def _role_row(rows: Sequence[Answer], role: str,
             current_rubric: Mapping[str, str]) -> tuple[str, Answer] | None:
    """(deciding backend, its newest fresh row) for a text-only verdict on
    `role`: the first backend in AUTHORITY_ORDER[role] with a fresh row
    decides -- ties within that backend go to the newest row. None when
    none has spoken.
    """
    for backend in AUTHORITY_ORDER.get(role, ("judge",)):
        spoken = [r for r in rows if r.port == "assess" and r.backend == backend
                 and r.question.get("role") == role and not _stale(r, current_rubric)]
        if not spoken:
            continue
        return backend, max(spoken, key=lambda r: r.ts)
    return None


def _rank_of(backend: str, row: Answer) -> float:
    """The numeric rank `row`'s own answer carries, on the shared judge/
    learner scale -- LEARNER_RANK for a learner row, `_judge_rank`
    otherwise.
    """
    value = row.answer.get("value")
    if backend == "learner":
        return LEARNER_RANK.get(value, _JUDGE_FAIL_RANK)
    return _judge_rank(value)


def _role_rank(rows: Sequence[Answer], role: str,
               current_rubric: Mapping[str, str]) -> tuple[str, float] | None:
    """(deciding backend, rank) for a text-only verdict on `role`: the
    first backend in AUTHORITY_ORDER[role] with a fresh row decides.
    None when none has spoken.
    """
    decided = _role_row(rows, role, current_rubric)
    if decided is None:
        return None
    backend, row = decided
    return backend, _rank_of(backend, row)


def adoptable_drafts(cache: CacheReader, syllabus, *, current_rubric: Mapping[str, str],
                     model: str = "llm", today: Callable[[], date] = date.today
                     ) -> list[tuple[Sentence, tuple[Target, ...]]]:
    """Every unadopted sentence draft with a gloss, accepted by the
    Sentence invariant (Syllabus.check_sentence -- a refused draft is
    logged and skipped), that fills at least one still-open Target
    (Syllabus.fill_set, spec 1 section 3) and whose sentence-for-target
    assessment passes (authority order deciding), with those Targets.
    `model` and `today` go on the Sentence's provenance. A text F13 has
    retired (spec 3 section 5, record.retired_texts) is never re-adopted,
    even once its own Targets reopen and it is the passing draft on
    file -- its own recording proved unsourceable, and that does not
    change just because the sentences row was deleted.
    """
    adopted = {s.text_sha for s in syllabus.sentences}
    retired = record.retired_texts(cache)
    provenance = Provenance(source="llm", origin=model, licence="generated", acquired=today())
    unfilled = set(syllabus.gaps().unfilled_targets)
    out: list[tuple[Sentence, tuple[Target, ...]]] = []
    for draft in record.sentence_drafts(cache):
        if draft.text_sha in adopted or draft.text_sha in retired:
            continue
        if not draft.gloss:
            continue
        sentence = Sentence(clauses=draft.clauses, text=draft.text, gloss=draft.gloss,
                            voice="learner_voice", provenance=provenance)
        try:
            syllabus.check_sentence(sentence)
        except ValueError as e:
            _log.warning("adoptable_drafts: draft refused: %s", e)
            continue
        fills = syllabus.fill_set(sentence)
        filled = tuple(t for t in syllabus.targets if t.id in unfilled and t in fills)
        if not filled:
            continue
        rows = cache.assessments_of(draft.text_sha)
        role = role_for("sentence", record.subject_kind_of(rows))
        ranked = _role_rank(rows, role, current_rubric)
        if ranked is None or ranked[1] <= _JUDGE_FAIL_RANK:
            continue
        out.append((sentence, filled))
    return out


def refused_drafts(cache: CacheReader, syllabus, *, current_rubric: Mapping[str, str],
                   limit: int = 20) -> list[tuple[str, str]]:
    """The texts not to propose again (spec 3 section 5): every unadopted
    sentence draft (`record.sentence_drafts`) whose sentence-for-target
    verdict, under the current rubric and the same authority order
    `adoptable_drafts` reads (`_role_row`), fails (`_JUDGE_FAIL_RANK`),
    newest draft first, at most `limit`. `record.sentence_drafts` returns
    drafts oldest first and merges each text to its one draft, so the
    newest-first order here is by each text's first appearance among the
    asks, latest such ask first; a text's newest fresh row for the
    deciding backend is what `_role_row` returns, so a stale-only verdict
    (no fresh row) decides nothing and a text that later passed is not
    listed. `(text, evidence)`, evidence the deciding row's own
    `answer["evidence"]`, whitespace collapsed and cut to 200 characters
    (empty when it named none).

    A text F13 has retired (spec 3 section 5, record.retired_texts) is
    listed too, evidence "retired: recording exhausted", whatever its own
    sentence-for-target verdict was (it passed -- that is why it was
    adopted in the first place; retirement is about its recording, never
    its quality) -- combined with the failing-verdict texts above into
    the one newest-first list `limit` caps together.
    """
    adopted = {s.text_sha for s in syllabus.sentences}
    retired = record.retired_texts(cache)
    out: list[tuple[str, str]] = []
    for draft in reversed(record.sentence_drafts(cache)):
        if draft.text_sha in retired:
            out.append((draft.text, "retired: recording exhausted"))
            if len(out) >= limit:
                break
            continue
        if draft.text_sha in adopted:
            continue
        rows = cache.assessments_of(draft.text_sha)
        role = role_for("sentence", record.subject_kind_of(rows))
        decided = _role_row(rows, role, current_rubric)
        if decided is None:
            continue
        backend, row = decided
        if _rank_of(backend, row) > _JUDGE_FAIL_RANK:
            continue
        evidence = " ".join(str(row.answer.get("evidence") or "").split())[:200]
        out.append((draft.text, evidence))
        if len(out) >= limit:
            break
    return out
