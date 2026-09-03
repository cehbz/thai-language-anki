"""Derivations (spec 3 section 3): pure folds over the cache, never
stored. current_best, exhausted, queue, confusion_weights.

Row conventions these folds rely on (not fixed elsewhere in the spec, so
fixed here -- see the implementation report's ambiguity notes):
  - A `provide` row's `question` carries {"provides": kind, "params": {...}}
    (provider.py); its `answer["items"]` is a list, each item optionally
    carrying "sha" once the artifact is content-addressed (imgfetch/tts
    write one, a bare image-search result before imgfetch does not).
  - An `assess` row's `question` carries {"role": role, "artifact_sha":
    sha_or_null, "rubric": rubric_or_null} (assessor.py); its
    `answer["value"]` is the judge's/learner's verdict -- bool or score
    for judge, a rating string for the learner backend
    ("good"/"acceptable"/"unacceptable-use-this"/"unacceptable-none").
  - `kind` (this module's parameter, matching the spec's terse
    `current_best(subject, kind)`) matches a `provide` row via
    question["provides"] == kind, and an `assess` row via
    question["role"] == kind or question["role"].startswith(kind + "-")
    (bridging the two ports' vocabularies -- "picture" matches the judge
    role "picture-for-word").
  - A learner "direction" fact (steering, no artifact yet) is any row
    whose answer carries a non-null "direction" key, or whose question
    carries {"kind": "direction"} (migrate.py's convention).

`cache`/`syllabus` are always the first parameters (the spec's prose
`current_best(subject, kind)` elides the obvious reader dependency; this
implementation makes it explicit so every function is a pure fold over an
injected CacheReader, testable with synthetic rows and no real store).
"""
from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass

from .ports import Answer, CacheReader, StudyReader

__all__ = [
    "CurrentBest", "current_best",
    "ExhaustedStatus", "exhausted",
    "QueueEntry", "queue",
    "confusion_weights",
    "LEARNER_RANK",
]

# Learner rating -> a numeric rank on the same scale judge verdicts use,
# so current_best's regression guard ("never below an artifact the
# learner rated acceptable") is a plain numeric comparison. Judge pass
# (True/1.0) ranks below learner "acceptable" so a judge run alone can
# never outrank a learner's endorsement (spec 3 section 3).
LEARNER_RANK: dict[str, float] = {
    "good": 100.0,
    "acceptable": 80.0,
    "unacceptable-use-this": 40.0,
    "unacceptable-none": -1.0,
}
_JUDGE_PASS_RANK = 50.0
_JUDGE_FAIL_RANK = 0.0
_ACCEPTABLE_FLOOR = LEARNER_RANK["acceptable"]
_GOOD_RANK = LEARNER_RANK["good"]


def _matches_kind(question: Mapping, kind: str) -> bool:
    provides = question.get("provides")
    if provides is not None:
        return provides == kind
    role = question.get("role")
    if role is not None:
        return role == kind or role.startswith(kind + "-")
    return False


def _is_direction(row: Answer) -> bool:
    return bool(row.answer.get("direction")) or row.question.get("kind") == "direction"


def _judge_rank(value) -> float:
    if isinstance(value, bool):
        return _JUDGE_PASS_RANK if value else _JUDGE_FAIL_RANK
    if isinstance(value, (int, float)):
        return float(value)
    return _JUDGE_FAIL_RANK


def _rows_for(cache: CacheReader, subject: str, kind: str) -> list[Answer]:
    return [r for r in cache.assessments_of(subject) if _matches_kind(r.question, kind)]


def _judge_ranks(rows: Sequence[Answer], current_rubric: str | None) -> dict[str, float]:
    ranks: dict[str, float] = {}
    for r in rows:
        if r.port != "assess" or r.backend != "judge":
            continue
        if current_rubric is not None and r.question.get("rubric") != current_rubric:
            continue  # a stale-rubric verdict does not rank a candidate
        artifact_sha = r.question.get("artifact_sha")
        if not artifact_sha:
            continue
        rank = _judge_rank(r.answer.get("value"))
        if artifact_sha not in ranks or rank > ranks[artifact_sha]:
            ranks[artifact_sha] = rank
    return ranks


def _learner_ratings(rows: Sequence[Answer]) -> dict[str, tuple[int, str]]:
    """artifact_sha -> (latest ts, rating), newest wins per artifact."""
    out: dict[str, tuple[int, str]] = {}
    for r in rows:
        if r.port != "assess" or r.backend != "learner":
            continue
        rating = r.answer.get("value")
        if rating not in LEARNER_RANK:
            continue
        artifact_sha = r.question.get("artifact_sha") or r.answer.get("artifact_sha")
        if not artifact_sha:
            continue
        prev = out.get(artifact_sha)
        if prev is None or r.ts > prev[0]:
            out[artifact_sha] = (r.ts, rating)
    return out


# --- current_best -----------------------------------------------------------

@dataclass(frozen=True)
class CurrentBest:
    artifact_sha: str | None
    source: str          # "learner" | "judge" | "none"
    rank: float
    challenger: str | None = None  # an unrated artifact ranking higher, presented not swapped


def current_best(cache: CacheReader, subject: str, kind: str, *,
                 current_rubric: str | None = None) -> CurrentBest:
    rows = _rows_for(cache, subject, kind)
    learner_ratings = _learner_ratings(rows)
    judge_ranks = _judge_ranks(rows, current_rubric)

    latest_learner_row = max(
        (r for r in rows if r.port == "assess" and r.backend == "learner"
         and r.answer.get("value") in LEARNER_RANK),
        key=lambda r: r.ts, default=None)

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
        challenger = _best_challenger(judge_ranks, learner_ratings)
        return CurrentBest(artifact_sha=artifact_sha, source="learner", rank=rank,
                           challenger=challenger)

    # Only a genuinely passing judge verdict (rank above the fail floor)
    # counts as a usable current_best -- an all-failing history must read
    # the same as "no candidate at all" (rank -1.0), not as "improved"
    # over none, or a subsequent real pass would never register as an
    # improvement (it would already be numerically <= a failed 0.0... and
    # worse, a fail alone would look like progress over nothing).
    passing = {s: r for s, r in judge_ranks.items() if r > _JUDGE_FAIL_RANK}
    if passing:
        best_sha = max(passing, key=passing.get)
        return CurrentBest(artifact_sha=best_sha, source="judge", rank=passing[best_sha])

    return CurrentBest(artifact_sha=None, source="none", rank=-1.0)


def _best_challenger(judge_ranks: Mapping[str, float],
                     learner_ratings: Mapping[str, tuple]) -> str | None:
    """Any judge-passed artifact the learner hasn't rated is an eligible
    challenger -- presented, never silently swapped in (spec 3 section 3).
    Not gated on out-ranking the current pick: the point is to surface a
    challenger for the learner to look at, not to pre-judge it against
    their existing choice.
    """
    candidates = [(sha, rank) for sha, rank in judge_ranks.items()
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
             attempt_cap: int = 8, current_rubric: str | None = None) -> ExhaustedStatus:
    rows = _rows_for(cache, subject, kind)
    provide_rows = sorted((r for r in rows if r.port == "provide"), key=lambda r: r.ts)
    attempts = len(provide_rows)
    if attempts < attempt_cap:
        return ExhaustedStatus(exhausted=False, attempts=attempts, reason="attempt cap not reached")

    best = current_best(cache, subject, kind, current_rubric=current_rubric)
    if best.artifact_sha is None:
        return ExhaustedStatus(exhausted=False, attempts=attempts, reason="no current_best yet")

    last_k_shas: set[str] = set()
    for r in provide_rows[-k:]:
        for item in r.answer.get("items", []):
            if isinstance(item, Mapping) and item.get("sha"):
                last_k_shas.add(item["sha"])

    # Compare the last k attempts' candidates against the best that stood
    # BEFORE them, not against current_best (which already folds them in
    # -- comparing against itself would make "the last attempt IS the
    # best" look like it never out-ranked anything).
    judge_ranks = _judge_ranks(rows, current_rubric)
    learner_ratings = _learner_ratings(rows)
    prior_judge_best = max((r for s, r in judge_ranks.items() if s not in last_k_shas),
                           default=-1.0)
    prior_learner_best = max((LEARNER_RANK[rating] for s, (_, rating) in learner_ratings.items()
                              if s not in last_k_shas and rating in ("good", "acceptable")),
                             default=-1.0)
    prior_best_rank = max(prior_judge_best, prior_learner_best)

    if any(judge_ranks.get(s, -1.0) > prior_best_rank for s in last_k_shas):
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


def _has_untried_lever(rows: Sequence[Answer], current_rubric: str | None,
                       known_backends: Container[str] | None) -> bool:
    judge_rows = [r for r in rows if r.port == "assess" and r.backend == "judge"]
    if judge_rows and current_rubric is not None:
        if any(r.question.get("rubric") != current_rubric for r in judge_rows):
            return True  # rubric changed since verdicts -- machine verdicts re-rank
    provide_ts = max((r.ts for r in rows if r.port == "provide"), default=-1)
    if any(r.answer.get("suggestion") and r.ts > provide_ts for r in judge_rows):
        return True  # a judge suggestion has not been followed by a new attempt
    if known_backends is not None:
        tried = {r.backend for r in rows if r.port == "provide"}
        if set(known_backends) - tried:
            return True  # an unsearched backend remains
    return False


def queue(syllabus, cache: CacheReader, *, budgets: Mapping[str, object] | None = None,
         current_rubric: str | None = None,
         known_backends: Mapping[str, Container[str]] | None = None) -> list[QueueEntry]:
    entries: list[QueueEntry] = []
    for subject, kind in _gap_candidates(syllabus):
        rows = _rows_for(cache, subject, kind)
        best = current_best(cache, subject, kind, current_rubric=current_rubric)
        status = exhausted(cache, subject, kind, current_rubric=current_rubric)
        if best.rank >= _GOOD_RANK or status.exhausted:
            continue  # never: good/exhausted -- exhausted surfaces on the feedback screen
        directed = any(_is_direction(r) for r in rows)
        attempts = len([r for r in rows if r.port == "provide"])
        if best.artifact_sha is None or best.source != "learner" or best.rank < _ACCEPTABLE_FLOOR:
            bucket = 1
        elif _has_untried_lever(rows, current_rubric,
                                (known_backends or {}).get(kind)):
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

def confusion_weights(seed: Mapping[str, float], study: StudyReader,
                      confusion_ids: Sequence[str]) -> dict[str, float]:
    """curated seed x StudyReader lapse rates (spec 3 section 3). A
    StudyRecord grade <= 1 (Anki's "again") counts as a lapse; a
    confusion with no study history yet just keeps its seed weight.
    """
    weights: dict[str, float] = {}
    for cid in confusion_ids:
        base = seed.get(cid, 1.0)
        records = study.records(cid)
        if not records:
            weights[cid] = base
            continue
        lapses = sum(1 for r in records if r.grade <= 1)
        lapse_rate = lapses / len(records)
        weights[cid] = base * (1.0 + lapse_rate)
    return weights
