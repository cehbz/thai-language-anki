"""Derivations (spec 3 section 3): pure folds over the cache, never
stored. current_best, exhausted, queue, confusion_weights.

record.py holds the row-selecting folds (rows_for, source_asks,
candidate_shas, learner_ratings, directions, judge_verdicts,
latest_query); every row a provide/assess writer appends names its need
kind, or a learner row's own row kind, in question["kind"] (record.py's
docstring). This module reads through record.py only -- no `provide`
row's `provides` string or `assess` row's `role` string is matched by
membership or prefix here.

A rating row's own question["kind"] is "rating", never a need kind, so
it is not among rows_for(subject, kind)'s result; current_best/exhausted
instead scope record.ratings_for_role(cache.assessments_of(subject),
ROLE_FOR_KIND[kind]) to one need by its question["role"] -- an explicit
equality read of typed data, not an inference.

`cache`/`syllabus` are always the first parameters (the spec's prose
`current_best(subject, kind)` elides the obvious reader dependency; this
implementation makes it explicit so every function is a pure fold over an
injected CacheReader, testable with synthetic rows and no real store).
"""
from __future__ import annotations

from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass
from typing import Union

from . import record
from .authority import AUTHORITY_ORDER, ROLE_FOR_KIND, role_for
from .ports import Answer, CacheReader, StudyReader
from .record import LEARNER_RANK
from .syllabus import Syllabus

__all__ = [
    "CurrentBest", "current_best",
    "ExhaustedStatus", "exhausted",
    "QueueEntry", "queue",
    "pending",
    "confusion_weights",
    "LEARNER_RANK",
    "stale",
]

# A rubric filter: None matches every rubric; a str applies to every role
# (old, pre-authority-table behavior); a role -> rubric mapping is
# role-scoped (a verdict for a role absent from the mapping is never
# stale on that account).
Rubric = Union[str, Mapping[str, str], None]

# LEARNER_RANK (record.py): a numeric rank on the same scale judge
# verdicts use, so current_best's regression guard ("never below an
# artifact the learner rated acceptable") is a plain numeric comparison.
# Judge pass (True/1.0) ranks below learner "acceptable" so a judge run
# alone can never outrank a learner's endorsement (spec 3 section 3).
_JUDGE_PASS_RANK = 50.0
_JUDGE_FAIL_RANK = 0.0
_ACCEPTABLE_FLOOR = LEARNER_RANK["acceptable"]
_GOOD_RANK = LEARNER_RANK["good"]


def _judge_rank(value) -> float:
    if isinstance(value, bool):
        return _JUDGE_PASS_RANK if value else _JUDGE_FAIL_RANK
    if isinstance(value, (int, float)):
        return float(value)
    return _JUDGE_FAIL_RANK


def _shas_since(rows: Sequence[Answer], since_ts: int) -> set[str]:
    """Every artifact sha any provide row in `rows` at or after `since_ts`
    carries -- the candidates the attempts in that window actually
    produced.
    """
    shas: set[str] = set()
    for r in rows:
        if r.port != "provide" or r.ts < since_ts:
            continue
        for item in r.answer.get("items", []):
            if isinstance(item, Mapping) and item.get("sha"):
                shas.add(item["sha"])
    return shas


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


def _stale(row: Answer, current_rubric: Rubric) -> bool:
    """True when a verdict row's rubric no longer matches -- a str rubric
    applies to every role (old behavior); a role -> rubric mapping only
    stales the roles it names (a role absent from the mapping is never
    stale on that account).

    Mechanical rows carry no rubric at all (`rubric: None` -- they check
    ground truth, e.g. recording duration/format, not a judge prompt) --
    a rubric change can never make one stale, so a mechanical row with no
    rubric is exempted before either comparison runs (it would otherwise
    read as "rubric changed from None" under any current_rubric).
    """
    if current_rubric is None:
        return False
    if row.question.get("rubric") is None and row.backend == "mechanical":
        return False
    role = row.question.get("role")
    if isinstance(current_rubric, str):
        return row.question.get("rubric") != current_rubric
    if role in current_rubric:
        return row.question.get("rubric") != current_rubric[role]
    return False


# Public export: reviewserver._judge_verdict_line's rubric filter has to
# apply the same str-or-mapping semantics (and the same mechanical-row
# exemption) this module's own folds use -- re-deriving it there let a
# mapping silently miscompare (== against a dict). `stale` is the public
# name for the same function `_stale` is used under internally.
stale = _stale


def _machine_ranks(rows: Sequence[Answer], kind: str,
                   current_rubric: Rubric) -> tuple[dict[str, float], dict[str, str]]:
    """Authority-driven machine rank per artifact (spec 3 section 3
    amendment): for the kind's role, walk AUTHORITY_ORDER[role] skipping
    "learner" (the learner is folded in separately by current_best); the
    first backend in that order with a verdict row for an artifact
    decides its rank. Returns (ranks, sources) -- sources names the
    deciding backend per sha, for CurrentBest.source. Pictures additionally
    fold in preference-row bonuses (_apply_preference).
    """
    role = ROLE_FOR_KIND.get(kind, kind)
    order = [b for b in AUTHORITY_ORDER.get(role, ("judge",)) if b != "learner"]
    by_backend: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.port != "assess" or r.backend not in order or _stale(r, current_rubric):
            continue
        if r.question.get("role") != role:
            continue
        sha_ = r.question.get("artifact_sha")
        if not sha_:
            continue
        rank = _judge_rank(r.answer.get("value"))
        ranks = by_backend.setdefault(r.backend, {})
        if sha_ not in ranks or rank > ranks[sha_]:
            ranks[sha_] = rank
    out: dict[str, float] = {}
    sources: dict[str, str] = {}
    shas = {s for ranks in by_backend.values() for s in ranks}
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
                      current_rubric: Rubric) -> None:
    """The newest picture-preference row under the current rubric whose
    candidates all pass adds a positional bonus (spec 3 section 3
    amendment), 20.0 * (n - 1 - i) / max(n - 1, 1) at rank position i.
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


def _apply_prior(ranks: dict[str, float], provenance_prior: Sequence[str],
                 provenance: Callable[[str], Mapping | None] | None) -> None:
    """Provenance-prior tie-break (spec 3 section 3 amendment): for
    artifacts still passing, rank += (len(prior) - index) / (len(prior) +
    1), where index is the artifact's provenance source's position in
    provenance_prior (a source absent from the prior gets no bonus). Only
    applied to already-passing ranks, and always < 1.0 so it can only
    break ties, never outrank a genuinely better machine verdict.
    """
    if not provenance_prior or provenance is None:
        return
    for s, r in list(ranks.items()):
        if r <= _JUDGE_FAIL_RANK:
            continue
        prov = provenance(s) or {}
        try:
            idx = list(provenance_prior).index(prov.get("source"))
        except ValueError:
            continue
        ranks[s] = r + (len(provenance_prior) - idx) / (len(provenance_prior) + 1)


# --- current_best -----------------------------------------------------------

@dataclass(frozen=True)
class CurrentBest:
    artifact_sha: str | None
    source: str          # "learner" | the deciding machine backend (e.g. "judge",
                         # "mechanical") | "none"
    rank: float
    challenger: str | None = None  # an unrated artifact ranking higher, presented not swapped


def current_best(cache: CacheReader, subject: str, kind: str, *,
                 current_rubric: Rubric = None,
                 provenance_prior: Sequence[str] = (),
                 provenance: Callable[[str], Mapping | None] | None = None) -> CurrentBest:
    rows = record.rows_for(cache, subject, kind)
    role = ROLE_FOR_KIND.get(kind, kind)
    rating_rows = record.ratings_for_role(cache.assessments_of(subject), role)
    learner_ratings = _ratings_by_artifact(rating_rows)
    machine_ranks, machine_sources = _machine_ranks(rows, kind, current_rubric)
    _apply_prior(machine_ranks, provenance_prior, provenance)

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
                return CurrentBest(artifact_sha=best_sha, source="learner", rank=floor)
            return CurrentBest(artifact_sha=None, source="none", rank=-1.0)
        artifact_sha = (latest_learner_row.question.get("artifact_sha")
                        or latest_learner_row.answer.get("artifact_sha"))
        rank = max(LEARNER_RANK[rating], floor if floor is not None else -1.0)
        challenger = _best_challenger(machine_ranks, learner_ratings)
        return CurrentBest(artifact_sha=artifact_sha, source="learner", rank=rank,
                           challenger=challenger)

    # Only a genuinely passing machine verdict (rank above the fail floor)
    # counts as a usable current_best -- an all-failing history must read
    # the same as "no candidate at all" (rank -1.0), not as "improved"
    # over none, or a subsequent real pass would never register as an
    # improvement (it would already be numerically <= a failed 0.0... and
    # worse, a fail alone would look like progress over nothing).
    passing = {s: r for s, r in machine_ranks.items() if r > _JUDGE_FAIL_RANK}
    if passing:
        best_sha = max(passing, key=passing.get)
        return CurrentBest(artifact_sha=best_sha, source=machine_sources.get(best_sha, "none"),
                           rank=passing[best_sha])

    return CurrentBest(artifact_sha=None, source="none", rank=-1.0)


def _best_challenger(machine_ranks: Mapping[str, float],
                     learner_ratings: Mapping[str, tuple]) -> str | None:
    """Any machine-passed artifact the learner hasn't rated is an eligible
    challenger -- presented, never silently swapped in (spec 3 section 3).
    Not gated on out-ranking the current pick: the point is to surface a
    challenger for the learner to look at, not to pre-judge it against
    their existing choice.
    """
    candidates = [(sha, rank) for sha, rank in machine_ranks.items()
                 if sha not in learner_ratings and rank > _JUDGE_FAIL_RANK]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[1])[0]


# --- exhausted ---------------------------------------------------------

@dataclass(frozen=True)
class ExhaustedStatus:
    exhausted: bool
    attempts: int
    reason: str | None = None


def exhausted(cache: CacheReader, subject: str, kind: str, *, k: int = 2,
             attempt_cap: int = 8, current_rubric: Rubric = None) -> ExhaustedStatus:
    rows = record.rows_for(cache, subject, kind)
    provide_rows = record.source_asks(rows)
    attempts = len(provide_rows)
    if attempts < attempt_cap:
        return ExhaustedStatus(exhausted=False, attempts=attempts, reason="attempt cap not reached")

    best = current_best(cache, subject, kind, current_rubric=current_rubric)
    if best.artifact_sha is None:
        return ExhaustedStatus(exhausted=False, attempts=attempts, reason="no current_best yet")

    # "the last k attempts' candidates" = every sha fetched at or after the
    # k-th-last Source ask, since the fetches an ask causes are written
    # after it and are what that attempt actually produced.
    last_k_shas = _shas_since(rows, provide_rows[-k:][0].ts if provide_rows else 0)

    # Compare the last k attempts' candidates against the best that stood
    # BEFORE them, not against current_best (which already folds them in
    # -- comparing against itself would make "the last attempt IS the
    # best" look like it never out-ranked anything).
    machine_ranks, _sources = _machine_ranks(rows, kind, current_rubric)
    role = ROLE_FOR_KIND.get(kind, kind)
    learner_ratings = _ratings_by_artifact(record.ratings_for_role(cache.assessments_of(subject), role))
    prior_machine_best = max((r for s, r in machine_ranks.items() if s not in last_k_shas),
                             default=-1.0)
    prior_learner_best = max((LEARNER_RANK[rating] for s, (_, rating) in learner_ratings.items()
                              if s not in last_k_shas and rating in ("good", "acceptable")),
                             default=-1.0)
    prior_best_rank = max(prior_machine_best, prior_learner_best)

    if any(machine_ranks.get(s, -1.0) > prior_best_rank for s in last_k_shas):
        return ExhaustedStatus(exhausted=False, attempts=attempts,
                               reason="a recent attempt out-ranks the prior current_best")
    return ExhaustedStatus(exhausted=True, attempts=attempts)


# --- queue: F10 order ----------------------------------------------------

@dataclass(frozen=True)
class QueueEntry:
    subject: str
    kind: str
    bucket: int   # 1 = no-artifact/unacceptable, 2 = untried lever, 3 = rank by verdict
    directed: bool = False
    rank: float = 0.0
    attempts: int = 0


def _gap_candidates(syllabus) -> list[tuple[str, str]]:
    gaps = syllabus.gaps()
    target_word = {t.id: t.word for t in syllabus.targets}
    candidates: list[tuple[str, str]] = []
    candidates += [(w, "picture") for w in gaps.words_missing_pictures]
    candidates += [(w, "recording") for w in gaps.words_missing_recordings]
    candidates += [(target_word.get(t, t), "sentence") for t in gaps.unfilled_targets]
    candidates += [(c, "rendition") for c in gaps.missing_renditions]
    candidates += [(g, "grapheme-keyword") for g in gaps.graphemes_missing_keyword_data]
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def pending(cache: CacheReader, subject: str, kind: str) -> bool:
    """True while a judge batch is out and hasn't resolved (spec 3
    section 3 amendment): `subject` appears in the newest unresolved
    batch marker's subjects (record.unresolved_batch) paired with
    `kind`'s own role. Resolving the batch (Assessor.resolve) always
    releases the whole marker at once, so this needs no per-key verdict
    check -- once resolved, every subject it named stops being pending,
    whether or not each of its questions actually got a verdict.
    """
    found = record.unresolved_batch(cache)
    if found is None:
        return False
    _batch_id, subjects, roles = found
    role = role_for(kind)
    return any(s == subject and r == role for s, r in zip(subjects, roles))


def _has_untried_option(rows: Sequence[Answer], current_rubric: Rubric,
                        known_sources: Container[str] | None) -> bool:
    judge_rows = [r for r in rows if r.port == "assess" and r.backend == "judge"]
    if judge_rows and current_rubric is not None:
        if any(_stale(r, current_rubric) for r in judge_rows):
            return True  # rubric changed since verdicts -- machine verdicts re-rank
    provide_ts = max((r.ts for r in rows if r.port == "provide"), default=-1)
    if any(r.answer.get("suggestion") and r.ts > provide_ts for r in judge_rows):
        return True  # a judge suggestion has not been followed by a new attempt
    if known_sources is not None:
        tried = {r.backend for r in rows if r.port == "provide"}
        if set(known_sources) - tried:
            return True  # an unsearched source remains
    return False


def queue(syllabus, cache: CacheReader, *, budgets: Mapping[str, object] | None = None,
         current_rubric: Rubric = None,
         known_sources: Mapping[str, Container[str]] | None = None) -> list[QueueEntry]:
    entries: list[QueueEntry] = []
    for subject, kind in _gap_candidates(syllabus):
        if pending(cache, subject, kind):
            continue  # a batch is still out -- don't re-queue while awaiting it
        rows = record.rows_for(cache, subject, kind)
        best = current_best(cache, subject, kind, current_rubric=current_rubric)
        status = exhausted(cache, subject, kind, current_rubric=current_rubric)
        if best.rank >= _GOOD_RANK or status.exhausted:
            continue  # never: good/exhausted -- exhausted surfaces on the feedback screen
        directed = bool(record.directions(cache.assessments_of(subject)))
        attempts = len(record.source_asks(rows))   # Source asks, not the fetches they caused
        if best.artifact_sha is None or best.source != "learner" or best.rank < _ACCEPTABLE_FLOOR:
            bucket = 1
        elif _has_untried_option(rows, current_rubric,
                                 (known_sources or {}).get(kind)):
            bucket = 2
        else:
            bucket = 3
        entries.append(QueueEntry(subject=subject, kind=kind, bucket=bucket,
                                  directed=directed, rank=best.rank, attempts=attempts))

    entries.sort(key=lambda e: (e.bucket, not e.directed, e.rank, e.attempts, e.subject, e.kind))

    if budgets is not None:
        learner_budget = budgets.get("learner") if hasattr(budgets, "get") else None
        max_asks = getattr(learner_budget, "max_asks", None) if learner_budget is not None else None
        if max_asks is not None:
            entries = entries[:max_asks]
    return entries


# --- confusion_weights ---------------------------------------------------

def confusion_weights(seed: Mapping[str, float], syllabus: Syllabus,
                      study: StudyReader) -> dict[str, float]:
    """curated seed x the aggregate's own study grouping (spec 3 section
    3). A StudyRecord grade <= 1 (Anki's "again") counts as a lapse; a
    confusion with no study history yet just keeps its seed weight. Every
    confusion iterated is one the Syllabus itself carries
    (`syllabus.confusions`), not an arbitrary caller-supplied id list.
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
