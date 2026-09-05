"""The record, read through one module (spec 3 section 6): pure folds
over `cache` rows. Every provide/assess row a writer appends names its
artifact kind (picture | recording | rendition | sentence |
grapheme-keyword) or, for a learner row, its own row kind (rating |
direction | waiver | card-flag | note | drill | reverify) in
question["kind"], and the kind of thing its subject is (word | pair |
sentence | grapheme) in question["subject_kind"]; a judge-batch marker
row's kind is "batch". A fold here reads those fields, `backend`, `port`,
`subject`, and `answer` only -- never an encoded key, and never a
`provides`/`role` string matched by prefix or membership.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .entities import text_sha
from .ports import Answer, CacheReader

__all__ = ["LEARNER_RANK", "rows_for", "source_asks", "candidate_shas", "learner_ratings",
          "ratings_for_role", "directions", "judge_verdicts", "latest_query",
          "asks_since", "spend_since", "unresolved_batch", "subject_kind_of", "DRAFT_SUBJECT", "SentenceDraft",
          "drafts_in", "sentence_drafts"]

# The subject every sentence-drafting ask is appended under: drafts are
# proposed for a run's open Targets as a set, not for one subject.
DRAFT_SUBJECT = "sentence-drafts"

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


def subject_kind_of(rows: Sequence[Answer]) -> str:
    """The kind of thing these rows' subject is, as the rows themselves
    name it (question["subject_kind"]) -- "word" for a row written before
    the field existed, and for every subject that is a word.
    """
    for r in rows:
        subject_kind = r.question.get("subject_kind")
        if subject_kind:
            return str(subject_kind)
    return "word"


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


def asks_since(cache: CacheReader, backend: str, since_ts: int) -> int:
    """How many Source asks `backend` made at or after `since_ts` -- the
    asks a per-day budget counts (a bytes-fetch row is not an ask of its
    own).
    """
    return len(source_asks(cache.rows_since("provide", backend, since_ts)))


def spend_since(cache: CacheReader, backend: str, since_ts: int) -> float:
    """What those asks cost, in `backend`'s own currency."""
    return sum(r.cost for r in source_asks(cache.rows_since("provide", backend, since_ts)))


def unresolved_batch(cache: CacheReader) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """The (batch_id, subjects, roles) of the newest judge-batch marker
    row (subject "batch") whose latest status is "submitted" -- subjects
    and roles are parallel lists aligned by index, naming every question
    that batch asked. None while no batch is out. Shared by
    Assessor.unresolved_batch and derivations.pending, so both read the
    marker rows exactly the same way.
    """
    rows = [r for r in cache.assessments_of("batch") if r.question.get("kind") == "batch"]
    latest_by_key: dict[str, Answer] = {}
    for r in rows:
        prev = latest_by_key.get(r.key_sha)
        if prev is None or r.ts > prev.ts:
            latest_by_key[r.key_sha] = r
    submitted = [r for r in latest_by_key.values() if r.answer.get("status") == "submitted"]
    if not submitted:
        return None
    newest = max(submitted, key=lambda r: r.ts)
    return (newest.question["batch_id"], tuple(newest.question.get("subjects", [])),
           tuple(newest.question.get("roles", [])))


# --- sentence drafts --------------------------------------------------------

@dataclass(frozen=True)
class SentenceDraft:
    """One drafted sentence as the LLM answered it: text, L1 gloss, and
    the Targets it claims to fill."""
    text: str
    gloss: str
    claimed: tuple[str, ...]

    @property
    def text_sha(self) -> str:
        return text_sha(self.text)


def _strip_fences(text: str) -> str:
    return re.sub(r"^```[a-z]*\n|\n```$", "", text.strip())


def drafts_in(text: str) -> list[SentenceDraft]:
    """The drafts one llm answer item carries; empty when it is not the
    JSON the drafting prompt asked for."""
    try:
        data = json.loads(_strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return []
    drafted = (data.get("sentences") if isinstance(data, Mapping) else None) or []
    return [SentenceDraft(text=str(d["text"]).strip(), gloss=str(d.get("gloss") or ""),
                          claimed=tuple(d.get("targets") or []))
            for d in drafted if isinstance(d, Mapping) and d.get("text")]


def sentence_drafts(cache: CacheReader) -> list[SentenceDraft]:
    """Every sentence draft any run's drafting ask produced, newest ask
    last -- what there is to adopt once the verdicts land."""
    return [draft for row in rows_for(cache, DRAFT_SUBJECT, "sentence") if row.port == "provide"
            for item in row.answer.get("items", []) for draft in drafts_in(str(item))]
