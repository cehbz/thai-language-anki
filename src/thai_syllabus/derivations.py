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

`cache`/`syllabus` are always the first parameters (the spec's prose
`current_best(subject, kind)` elides the obvious reader dependency; this
implementation makes it explicit so every function is a pure fold over an
injected CacheReader, testable with synthetic rows and no real store).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from . import record
from .authority import AUTHORITY_ORDER, ROLE_FOR_KIND
from .media import Speaker
from .ports import Answer, CacheReader, StudyReader
from .record import LEARNER_RANK
from .syllabus import Syllabus

__all__ = [
    "CurrentBest", "current_best",
    "pending",
    "attempts_since_change", "next_source",
    "ExhaustedStatus", "exhausted",
    "improved",
    "directed",
    "QueueEntry", "queue",
    "challengers",
    "reasks",
    "confusion_weights",
    "LEARNER_RANK",
    "stale",
    "DEFAULT_ATTEMPT_CAP",
]

# LEARNER_RANK (record.py): a numeric rank on the same scale judge
# verdicts use, so current_best's regression guard ("never below an
# artifact the learner rated acceptable") is a plain numeric comparison.
# Judge pass (True/1.0) ranks below learner "acceptable" so a judge run
# alone can never outrank a learner's endorsement (spec 3 section 6).
_JUDGE_PASS_RANK = 50.0
_JUDGE_FAIL_RANK = 0.0
_ACCEPTABLE_FLOOR = LEARNER_RANK["acceptable"]
_GOOD_RANK = LEARNER_RANK["good"]

# The attempt cap exhausted() enforces, read from this one place: run.py
# and reviewserver.py pass it explicitly until providers.yaml wires a
# configured value.
DEFAULT_ATTEMPT_CAP = 8


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
    stale on that account. Mechanical rows carry no rubric at all
    (`rubric: None` -- they check ground truth, e.g. recording
    duration/format, not a judge prompt); a rubric change can never make
    one stale, so a mechanical row with no rubric is exempted first.
    """
    if row.question.get("rubric") is None and row.backend == "mechanical":
        return False
    role = row.question.get("role")
    if role in current_rubric:
        return row.question.get("rubric") != current_rubric[role]
    return False


# Public export: reviewserver._judge_verdict_line's rubric filter has to
# apply the same role-scoped semantics (and the same mechanical-row
# exemption) this module's own folds use.
stale = _stale


def _machine_ranks(rows: Sequence[Answer], kind: str,
                   current_rubric: Mapping[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    """Authority-driven machine rank per artifact (spec 3 section 6): for
    the kind's role, walk AUTHORITY_ORDER[role] skipping "learner" (the
    learner is folded in separately by current_best); the first backend in
    that order with a verdict row for an artifact decides its rank.
    Returns (ranks, sources) -- sources names the deciding backend per
    sha, for CurrentBest.source. Pictures additionally fold in preference-
    row bonuses (_apply_preference).
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
    """Provenance-prior tie-break (spec 3 section 6): for artifacts still
    passing, rank += (len(prior) - index) / (len(prior) + 1), where index
    is `provenance_source(sha)`'s position in `prior` (a source absent
    from `prior`, or a sha `provenance_source` names none for, gets no
    bonus). `provenance_source` reads the media table (SyllabusDb.
    media_provenance(sha)["source"]), not a cache row: a candidate's real
    Source (e.g. "forvo") is recorded there, not on the bytes-fetch row
    that actually wrote its sha (that row's own backend is "audiofetch"/
    "imgfetch", never a Source name). Only applied to already-passing
    ranks, and always < 1.0 so it can only break ties, never outrank a
    genuinely better machine verdict.
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


def current_best(cache: CacheReader, subject: str, kind: str, *,
                 current_rubric: Mapping[str, str], prior: Sequence[str] = (),
                 provenance_source: Callable[[str], str | None]) -> CurrentBest:
    rows = record.rows_for(cache, subject, kind)
    role = ROLE_FOR_KIND.get(kind, kind)
    rating_rows = record.ratings_for_role(cache.assessments_of(subject), role)
    learner_ratings = _ratings_by_artifact(rating_rows)
    machine_ranks, machine_sources = _machine_ranks(rows, kind, current_rubric)
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
        best_sha = max(passing, key=passing.get)
        return CurrentBest(artifact_sha=best_sha, source=machine_sources.get(best_sha),
                           rank=passing[best_sha], speaker=_speaker_for(rows, best_sha))

    return CurrentBest(artifact_sha=None, source=None, rank=-1.0)


# --- pending -----------------------------------------------------------

def pending(cache: CacheReader, subject: str, kind: str) -> bool:
    """True while the newest submitted judge batch marker names `subject`
    (record.unresolved_batch): membership in the run's unresolved batch --
    nothing else is pending. Resolving the batch (Assessor.resolve)
    releases the whole marker at once, so every subject it named stops
    being pending together, whether or not each of its questions actually
    got a verdict.
    """
    found = record.unresolved_batch(cache)
    if found is None:
        return False
    _batch_id, subjects, _roles = found
    return subject in subjects


# --- next_source / attempts_since_change --------------------------------

def _no_provenance_source(artifact_sha: str) -> str | None:
    """provenance_source for a current_best() call made with an empty
    `prior` -- _apply_prior returns before ever calling it, so this exists
    only to satisfy the required keyword.
    """
    return None


def _anchor_ts(cache: CacheReader, subject: str, kind: str, rows: Sequence[Answer]) -> int:
    """The ts of the earliest provide row whose items include current-
    best's artifact -- current-best computed rubric-agnostically here
    (empty rubric mapping: never stale), since source escalation tracks
    when a candidate was last PRODUCED, not whether its verdict is still
    fresh under the current judge rubric (that is queue()'s own concern).
    -1 (every ask counts) while no artifact exists yet.
    """
    best = current_best(cache, subject, kind, current_rubric={}, prior=(),
                        provenance_source=_no_provenance_source)
    if best.artifact_sha is None:
        return -1
    producing = [r.ts for r in rows if r.port == "provide"
                and any(isinstance(i, Mapping) and i.get("sha") == best.artifact_sha
                        for i in r.answer.get("items", []))]
    return min(producing) if producing else -1


def attempts_since_change(cache: CacheReader, subject: str, kind: str) -> list[Answer]:
    """Source asks under (subject, kind) with ts greater than the ts of
    the row that produced current-best's artifact -- every ask counts
    when no artifact exists yet.
    """
    rows = record.rows_for(cache, subject, kind)
    since_ts = _anchor_ts(cache, subject, kind, rows)
    return [r for r in record.source_asks(rows) if r.ts > since_ts]


def next_source(cache: CacheReader, subject: str, kind: str,
                sources: Sequence[str]) -> str | None:
    """The first of `sources` (cheapest first) with no ask in
    attempts_since_change; None once every source has been asked since
    current-best last changed.
    """
    tried_since = {r.backend for r in attempts_since_change(cache, subject, kind)}
    for source in sources:
        if source not in tried_since:
            return source
    return None


# --- exhausted ---------------------------------------------------------

@dataclass(frozen=True)
class ExhaustedStatus:
    exhausted: bool
    attempts: int


def exhausted(cache: CacheReader, subject: str, kind: str, *,
              sources: Sequence[str], attempt_cap: int) -> ExhaustedStatus:
    """Every source in `sources` has been asked since current-best last
    changed, or the attempt count since then reached `attempt_cap` --
    reopened by a learner row or a new source that changes what
    next_source/attempts_since_change see next time.
    """
    since = attempts_since_change(cache, subject, kind)
    is_exhausted = next_source(cache, subject, kind, sources) is None or len(since) >= attempt_cap
    return ExhaustedStatus(exhausted=is_exhausted, attempts=len(since))


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


def _current_artifact_unacceptable(cache: CacheReader, subject: str, role: str,
                                   artifact_sha: str | None) -> bool:
    """True when the latest learner rating specifically naming
    `artifact_sha` (current-best's own pick) is one of the
    "unacceptable-*" values -- distinct from having no artifact at all.
    """
    if artifact_sha is None:
        return False
    ratings = record.ratings_for_role(cache.assessments_of(subject), role)
    on_artifact = [r for r in ratings
                  if (r.question.get("artifact_sha") or r.answer.get("artifact_sha")) == artifact_sha]
    if not on_artifact:
        return False
    latest = max(on_artifact, key=lambda r: r.ts)
    return latest.answer.get("value") in ("unacceptable-none", "unacceptable-use-this")


def _has_untried_lever(cache: CacheReader, subject: str, kind: str, rows: Sequence[Answer],
                       current_rubric: Mapping[str, str], sources: Sequence[str]) -> bool:
    """A rubric change left a judge verdict stale, a judge suggestion has
    not been followed by a new attempt, or an unasked source remains
    (spec 3 section 6 bucket 2).
    """
    judge_rows = [r for r in rows if r.port == "assess" and r.backend == "judge"]
    if any(_stale(r, current_rubric) for r in judge_rows):
        return True
    provide_ts = max((r.ts for r in rows if r.port == "provide"), default=-1)
    if any(r.answer.get("suggestion") and r.ts > provide_ts for r in judge_rows):
        return True
    return next_source(cache, subject, kind, sources) is not None


# --- queue: F10 order ----------------------------------------------------

@dataclass(frozen=True)
class QueueEntry:
    subject: str
    kind: str
    bucket: int   # 1 = no-artifact/unacceptable, 2 = untried lever, 3 = acceptable/unrated
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


def queue(syllabus, cache: CacheReader, *, current_rubric: Mapping[str, str],
         prior: Sequence[str], sources_for: Callable[[str], Sequence[str]],
         attempt_cap: int, provenance_source: Callable[[str], str | None]) -> list[QueueEntry]:
    entries: list[QueueEntry] = []
    for subject, kind in _gap_candidates(syllabus):
        if pending(cache, subject, kind):
            continue  # a batch is still out -- pending is reported, not queued
        best = current_best(cache, subject, kind, current_rubric=current_rubric, prior=prior,
                            provenance_source=provenance_source)
        if best.rank >= _GOOD_RANK:
            continue  # good -- never queued

        role = ROLE_FOR_KIND.get(kind, kind)
        rejected = _current_artifact_unacceptable(cache, subject, role, best.artifact_sha)
        is_directed = directed(cache, subject)
        rows = record.rows_for(cache, subject, kind)
        sources = sources_for(kind)
        attempts = len(record.source_asks(rows))

        if best.artifact_sha is None or rejected:
            status = exhausted(cache, subject, kind, sources=sources, attempt_cap=attempt_cap)
            if status.exhausted and not is_directed:
                continue  # out of machine options and nothing directs it -- excluded
            bucket = 1
        elif _has_untried_lever(cache, subject, kind, rows, current_rubric, sources):
            bucket = 2
        else:
            bucket = 3

        entries.append(QueueEntry(subject=subject, kind=kind, bucket=bucket,
                                  directed=is_directed, rank=best.rank, attempts=attempts))

    entries.sort(key=lambda e: (e.bucket, not e.directed, e.rank, e.attempts, e.subject, e.kind))
    return entries


# --- challengers -----------------------------------------------------------

def challengers(cache: CacheReader, syllabus, *, current_rubric: Mapping[str, str],
                prior: Sequence[str],
                provenance_source: Callable[[str], str | None]) -> list[tuple[str, str, str]]:
    """(subject, learner-accepted sha, challenger sha) for every subject in
    syllabus.gaps()'s universe where a machine-ranked candidate under
    `current_rubric` outranks a learner-accepted artifact -- never auto-
    switched, only ever presented.
    """
    out: list[tuple[str, str, str]] = []
    for subject, kind in _gap_candidates(syllabus):
        best = current_best(cache, subject, kind, current_rubric=current_rubric, prior=prior,
                            provenance_source=provenance_source)
        if best.source != "learner" or best.artifact_sha is None:
            continue
        rows = record.rows_for(cache, subject, kind)
        role = ROLE_FOR_KIND.get(kind, kind)
        rating_rows = record.ratings_for_role(cache.assessments_of(subject), role)
        rated = set(_ratings_by_artifact(rating_rows))
        machine_ranks, _sources = _machine_ranks(rows, kind, current_rubric)
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
        out.append((subject, best.artifact_sha, challenger_sha))
    return out


# --- reasks ----------------------------------------------------------------

def reasks(cache: CacheReader, study: StudyReader, syllabus, *, lapse_threshold: int,
          card_keys_for: Callable[[str], Sequence[str]]) -> list[tuple[str, str]]:
    """(subject, card_key) for every word/pair whose learner-rated "good"
    artifact's card has accumulated at least `lapse_threshold` lapses
    (StudyRecord grade <= 1). `card_keys_for` names the card keys a
    subject's compiled cards use -- a later task carries the card key's
    own parts as columns; for now the caller composes them.
    """
    out: list[tuple[str, str]] = []
    subjects = ([(w.id, "picture") for w in syllabus.words]
               + [(w.id, "recording") for w in syllabus.words]
               + [(p.id, "rendition") for p in syllabus.pairs])
    for subject, kind in subjects:
        role = ROLE_FOR_KIND.get(kind, kind)
        ratings = record.ratings_for_role(cache.assessments_of(subject), role)
        if not ratings:
            continue
        latest = max(ratings, key=lambda r: r.ts)
        if latest.answer.get("value") != "good":
            continue
        for card_key in card_keys_for(subject):
            lapses = sum(1 for r in study.records(card_key) if r.grade <= 1)
            if lapses >= lapse_threshold:
                out.append((subject, card_key))
    return out


# --- confusion_weights ---------------------------------------------------

def confusion_weights(seed: Mapping[str, float], syllabus: Syllabus,
                      study: StudyReader) -> dict[str, float]:
    """curated seed x the aggregate's own study grouping (spec 3 section
    6). A StudyRecord grade <= 1 (Anki's "again") counts as a lapse; a
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
