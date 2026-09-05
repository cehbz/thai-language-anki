"""The record, read through one module (spec 3 section 6): pure folds
over `cache` rows. Every provide/assess row a writer appends names its
need kind (picture | recording | rendition | sentence | grapheme-keyword)
or, for a learner row, its own row kind (rating | direction | waiver |
card-flag | note | drill | reverify) in question["kind"]; a judge-batch
marker row's kind is "batch". A fold here reads that field, `backend`,
`port`, `subject`, and `answer` only -- never an encoded key, and never a
`provides`/`role` string matched by prefix or membership.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .ports import Answer, CacheReader

__all__ = ["LEARNER_RANK", "rows_for", "source_asks", "candidate_shas", "learner_ratings",
          "ratings_for_role", "directions", "judge_verdicts", "latest_query"]

# The bytes-fetching backends write the candidate a Source ask already
# caused, not an ask of their own (spec 3 section 3: an attempt is one
# Source ask).
_BYTES_BACKENDS = ("imgfetch", "audiofetch")

# The learner rating vocabulary: every value a rating row's answer["value"]
# is allowed to carry, ranked on the same numeric scale a judge verdict
# ranks on (derivations.py's regression guard and floor compare the two
# directly). Owned here, the one place a row's rating is recognized --
# ratings_for_role's membership check and derivations.py's ranking both
# read this same table.
LEARNER_RANK: dict[str, float] = {
    "good": 100.0,
    "acceptable": 80.0,
    "unacceptable-use-this": 40.0,
    "unacceptable-none": -1.0,
}


def rows_for(cache: CacheReader, subject: str, kind: str) -> list[Answer]:
    """Every row on record for `subject` whose own question names `kind`,
    oldest first.
    """
    return [r for r in cache.assessments_of(subject) if r.question.get("kind") == kind]


def source_asks(rows: Sequence[Answer]) -> list[Answer]:
    """The provide rows among `rows` that are Source asks, oldest first."""
    return sorted((r for r in rows if r.port == "provide" and r.backend not in _BYTES_BACKENDS),
                 key=lambda r: r.ts)


def candidate_shas(rows: Sequence[Answer]) -> list[str]:
    """Every artifact sha any provide row in `rows` produced, first-seen order."""
    shas: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.port != "provide":
            continue
        for item in r.answer.get("items", []):
            sha = item.get("sha") if isinstance(item, Mapping) else None
            if sha and sha not in seen:
                seen.add(sha)
                shas.append(sha)
    return shas


def learner_ratings(rows: Sequence[Answer]) -> list[Answer]:
    """Every learner rating row in `rows`, oldest first (newest last). A
    rating row always carries an artifact_sha; one that does not is
    returned as-is, not dropped.
    """
    return sorted((r for r in rows if r.backend == "learner"
                  and r.question.get("kind") == "rating"), key=lambda r: r.ts)


def directions(rows: Sequence[Answer]) -> list[Answer]:
    """Every learner direction row in `rows`, oldest first."""
    return sorted((r for r in rows if r.backend == "learner"
                  and r.question.get("kind") == "direction"), key=lambda r: r.ts)


def judge_verdicts(rows: Sequence[Answer], role: str) -> list[Answer]:
    """Every judge verdict in `rows` under `role`, oldest first."""
    return sorted((r for r in rows if r.port == "assess" and r.backend == "judge"
                  and r.question.get("role") == role), key=lambda r: r.ts)


def ratings_for_role(rows: Sequence[Answer], role: str) -> list[Answer]:
    """Every learner rating row in `rows` under `role` and recognized by
    LEARNER_RANK, oldest first (newest last) -- the one need a subject's
    rating rows for multiple needs are told apart by (a rating row's own
    kind is always "rating", never the need kind).
    """
    return sorted((r for r in learner_ratings(rows) if r.question.get("role") == role
                  and r.answer.get("value") in LEARNER_RANK), key=lambda r: r.ts)


def latest_query(rows: Sequence[Answer]) -> str | None:
    """The phrase, url, or text the newest Source ask in `rows` carried,
    or None when `rows` holds no Source ask.
    """
    asks = source_asks(rows)
    if not asks:
        return None
    latest = max(asks, key=lambda r: r.ts)
    params = latest.question.get("params", {}) or {}
    return params.get("query") or params.get("url") or params.get("text")
