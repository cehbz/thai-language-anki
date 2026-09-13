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
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from .cachekeys import RunReportKey, comment_identity
from .entities import Clauses, Sentence, Word, clauses_from_json, text_sha
from .media import Provenance
from .ports import Answer, CacheReader
from .transport import strip_fences

_log = logging.getLogger(__name__)

__all__ = ["LEARNER_RANK", "rows_for", "source_asks", "last_source_ask_ts", "candidate_shas",
          "learner_ratings",
          "ratings_for_role", "latest_rating", "directions", "judge_verdicts",
          "latest_query", "tried_urls", "latest_nothing_reason",
          "latest_phrase", "drafted_phrase", "parse_phrases",
          "asks_since", "spend_since", "cost_since", "unresolved_batch", "run_reports",
          "subject_kind_of", "retired_texts",
          "DRAFT_SUBJECT", "SentenceDraft",
          "parse_drafts", "parse_no_fit", "merge_drafts", "draft_sentence", "drafts_in",
          "sentence_drafts",
          "excluded_candidates", "card_flags", "card_notes", "normalize_shown",
          "Comment", "comment_of", "comments_of", "comments",
          "PARSE_SUBJECT", "vocabulary_line", "parse_prompt", "parses_in",
          "PHRASE_SUBJECT",
          "COMMENT_SUBJECT", "COMMENT_PROMPT_VERSION", "COMMENT_ACTIONS", "CommentReading",
          "parse_comment_readings", "reading_of", "vetoed_readings", "vetoed_readings_all",
          "without_vetoed_readings",
          "action_label", "reading_view", "gloss_on_requested",
          "Retirement", "retirements", "retirement_evidence"]

# The subject every sentence-drafting ask is appended under: drafts are
# proposed for a run's open Targets as a set, not for one subject.
DRAFT_SUBJECT = "sentence-drafts"

# The subject every sentence-parsing ask is appended under (spec 2 r10
# section 4's migration parse step): the whole worklist of rows lacking
# clauses is one ask, not one subject per row.
PARSE_SUBJECT = "sentence-parses"

# The subject every phrase-drafting ask is appended under (spec 3 r24
# section 5): one batch prompt drafts a phrase for every open picture
# need lacking one, not one ask per subject. The drafted phrase itself is
# appended separately, one provide row per subject (cachekeys.PhraseKey),
# which is what `drafted_phrase`/`latest_phrase` read back.
PHRASE_SUBJECT = "picture-phrases"

# The subject every comment-reading ask is appended under (spec 3 r30
# section 5): one batch prompt reads every unread comment; the readings
# themselves are appended one row per comment under the comment's own
# subject (cachekeys.CommentReadingKey).
COMMENT_SUBJECT = "card-comments"

# The comment prompt's version, part of every reading and veto key: a
# bump re-reads every comment under the new prompt.
COMMENT_PROMPT_VERSION = "1"

# The execution side of a comment (design section 2): action name ->
# its required parameters. The parser keeps an action only when every
# parameter is present and typed; anything else is unactionable text.
COMMENT_ACTIONS: dict[str, tuple[str, ...]] = {
    "direction": ("kind", "text"),
    "retire_sentence": ("reason", "replacement_hint"),
    "replacement_sentence": ("thai", "gloss"),
    "rate": ("kind", "value"),
    "gloss_on": ("word",),
    "none": ("remark",),
}

# The artifact kinds a comment's `direction` or `rate` action may name:
# the two a card shows and the learner's own screen already rates. Named
# for the comment vocabulary it belongs to, so it is not read as
# reviewserver._ARTIFACT_KINDS (the kinds a question carries, rendition
# included).
_COMMENT_ARTIFACT_KINDS = ("picture", "recording")

# provide rows from these backends are not Source asks (spec 3 section 3
# vocabulary: an attempt is one Source ask): imgfetch/audiofetch write the
# candidate a Source ask already caused, and a learner row is a supply --
# an answer, not an ask. a legacy-current row is a migrated candidate's
# provenance (spec 2 section 4), an answer with no ask. an "llm" row is
# attempts.phrase_attempt's own per-subject phrase row (spec 3 r24 section
# 5): the phrase a Source ask already drafted, recorded under the picture
# need's own subject so record.drafted_phrase/latest_phrase and
# reviewserver's "what was tried" can read it back -- the ask itself is
# the batch prompt, appended separately under backend "llm-phrase" (spec 3
# roster), which stays a Source ask and is unaffected by this exclusion.
_NOT_SOURCE_ASK_BACKENDS = ("imgfetch", "audiofetch", "learner", "legacy-current", "llm")

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
    oldest first, minus the rows derived from a comment reading the
    learner struck (`without_vetoed_readings`).
    """
    return [r for r in without_vetoed_readings(cache.assessments_of(subject))
            if r.question.get("kind") == kind]


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
    return sorted((r for r in rows
                  if r.port == "provide" and r.backend not in _NOT_SOURCE_ASK_BACKENDS),
                 key=lambda r: r.ts)


def last_source_ask_ts(rows: Sequence[Answer]) -> int:
    """The newest Source ask's ts among `rows` (`source_asks`), or -1 when
    none is on record -- what "the last provide row" means everywhere a
    fold measures a judge suggestion's own freshness against it (spec 3
    section 5's picture query precedence, `latest_phrase`;
    derivations._has_untried_lever's bucket-2 check): the search that
    produced the judged candidate, never a provide row that is an answer
    rather than an ask -- attempts.phrase_attempt's own per-subject
    phrase row (fix round 2 finding 1) included, the same rows
    `source_asks` already excludes (imgfetch/audiofetch/learner/
    legacy-current/llm). A phrase drafted after a pending suggestion must
    not make that suggestion look stale.
    """
    asks = source_asks(rows)
    return max((r.ts for r in asks), default=-1)


def candidate_shas(rows: Sequence[Answer]) -> list[str]:
    """Every artifact sha any provide row in `rows` produced, plus every
    sha an attempt outcome row (port "attempt") names under
    answer["candidates"], first-seen order across both row kinds, no
    duplicates (spec 3 section 6): an outcome row names the artifacts its
    own attempt stored for this need -- a url-keyed fetch shared by two
    subjects appends a provide row under the first subject only (spec 3
    section 2), so the second subject's own outcome row is where its
    stored sha is named. Either way the sha is a candidate of the need;
    it ranks only by its own verdicts, never by appearing here.
    """
    shas: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.port == "provide":
            for item in r.answer.get("items", []):
                sha = item.get("sha") if isinstance(item, Mapping) else None
                if sha and sha not in seen:
                    seen.add(sha)
                    shas.append(sha)
        elif r.port == "attempt":
            for sha in r.answer.get("candidates", []):
                if sha and sha not in seen:
                    seen.add(sha)
                    shas.append(sha)
    return shas


def learner_ratings(rows: Sequence[Answer]) -> list[Answer]:
    """Every learner rating row in `rows`, oldest first (newest last),
    minus the rows derived from a struck comment reading
    (`without_vetoed_readings`). A rating row always carries an
    artifact_sha; one that does not is returned as-is, not dropped.
    """
    return sorted((r for r in without_vetoed_readings(rows) if r.backend == "learner"
                  and r.question.get("kind") == "rating"), key=lambda r: r.ts)


def directions(rows: Sequence[Answer]) -> list[Answer]:
    """Every learner direction row in `rows`, oldest first, minus the
    rows derived from a struck comment reading
    (`without_vetoed_readings`).
    """
    return sorted((r for r in without_vetoed_readings(rows) if r.backend == "learner"
                  and r.question.get("kind") == "direction"), key=lambda r: r.ts)


def judge_verdicts(rows: Sequence[Answer], role: str) -> list[Answer]:
    """Every judge verdict in `rows` under `role`, oldest first."""
    return sorted((r for r in rows if r.port == "assess" and r.backend == "judge"
                  and r.question.get("role") == role), key=lambda r: r.ts)


def ratings_for_role(rows: Sequence[Answer], role: str) -> list[Answer]:
    """Every learner rating row in `rows` under `role` whose value
    LEARNER_RANK recognizes, oldest first: how one subject's ratings for
    several needs are told apart.
    """
    return sorted((r for r in learner_ratings(rows) if r.question.get("role") == role
                  and r.answer.get("value") in LEARNER_RANK), key=lambda r: r.ts)


def latest_rating(rows: Sequence[Answer], role: str) -> str | None:
    """The value of the newest learner rating in `rows` under `role`
    (LEARNER_RANK's vocabulary), or None when `rows` holds none.
    """
    rated = ratings_for_role(rows, role)
    return rated[-1].answer["value"] if rated else None


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


def tried_urls(cache: CacheReader, subject: str, kind: str, source: str) -> frozenset[str]:
    """Every url an attempt-outcome row for (subject, kind, source) names
    as handed to imgfetch, ingested or refused (spec 3 section 5): the
    union of answer["tried"] over every such row.
    """
    return frozenset(
        url for r in rows_for(cache, subject, kind)
        if r.port == "attempt" and r.backend == source
        for url in r.answer.get("tried", ()))


def latest_nothing_reason(rows: Sequence[Answer]) -> str | None:
    """The reason the newest `nothing` attempt-outcome row in `rows`
    states, or None when none of them carries one. The sentence attempt
    is the writer that states one (spec 3 r19 section 5's no-fit answer),
    so this is the drafter's own words for declining -- what the feedback
    screen shows the learner beside its direction question.
    """
    reasons = [r for r in rows if r.port == "attempt"
              and r.answer.get("outcome") == "nothing" and r.answer.get("reason")]
    return str(max(reasons, key=lambda r: r.ts).answer["reason"]) if reasons else None


def drafted_phrase(rows: Sequence[Answer]) -> str | None:
    """The newest drafted image-search phrase on record for one subject
    (spec 3 section 5): a provide row the phrase drafter appended
    (backend "llm", question["provides"] == "phrase") whose answer
    carries a `phrase`, or None when none is on record -- what
    `attempts.phrase_attempt` tests to decide whether a picture need
    still lacks one.
    """
    drafts = [r for r in rows if r.port == "provide" and r.backend == "llm"
             and r.question.get("provides") == "phrase" and r.answer.get("phrase")]
    return str(max(drafts, key=lambda r: r.ts).answer["phrase"]) if drafts else None


def latest_phrase(rows: Sequence[Answer]) -> str | None:
    """The image-search phrase a picture attempt's query prefers (spec 3
    section 5), in precedence: the latest learner direction; else a judge
    suggestion newer than the last Source ask (`last_source_ask_ts` --
    the search that produced the judged candidate, never
    attempts.phrase_attempt's own per-subject phrase row, fix round 2
    finding 1); else the newest drafted phrase on record
    (`drafted_phrase`); else None -- no query on record, and the need
    waits (spec 3 r25 section 5).
    """
    directed = directions(rows)
    if directed:
        return str(directed[-1].answer["direction"])
    last_provide = last_source_ask_ts(rows)
    suggestions = [r for r in rows if r.port == "assess" and r.backend == "judge"
                  and r.answer.get("suggestion") and r.ts > last_provide]
    if suggestions:
        return str(max(suggestions, key=lambda r: r.ts).answer["suggestion"])
    return drafted_phrase(rows)


def asks_since(cache: CacheReader, backend: str, since_ts: int) -> int:
    """How many Source asks `backend` made at or after `since_ts` -- the
    asks a per-day budget counts (a bytes-fetch row is not an ask of its
    own).
    """
    return len(source_asks(cache.rows_since("provide", backend, since_ts)))


def fetches_since(cache: CacheReader, source: str, since_ts: int) -> int:
    """How many bytes fetches attributed to `source` landed at or after
    `since_ts`: the audiofetch rows whose question params name it. Forvo
    counts an mp3 download as a request against its daily limit (spec 3
    section 4), so its per-day budget sums these with the lookups.
    """
    rows = cache.rows_since("provide", "audiofetch", since_ts)
    return sum(1 for r in rows if _params_of(r).get("source") == source)


def _params_of(row: Answer) -> Mapping:
    params = row.question.get("params") if isinstance(row.question, Mapping) else None
    return params if isinstance(params, Mapping) else {}


def spend_since(cache: CacheReader, backend: str, since_ts: int) -> float:
    """What those asks cost, in `backend`'s own currency."""
    return sum(r.cost for r in source_asks(cache.rows_since("provide", backend, since_ts)))


def cost_since(cache: CacheReader, port: str, backend: str, since_ts: int) -> float:
    """What every row `backend` appended under `port` cost, at or after
    `since_ts` -- raw cost, with no Source-ask filter (unlike
    spend_since, which reads "provide" rows only). cli `run --spend-cap`
    (spec 3 section 7) needs this: a judge verdict's own cost is
    appended at port "assess" (assessor.py's _append_verdict), where
    spend_since never looks.
    """
    return sum(r.cost for r in cache.rows_since(port, backend, since_ts))


@dataclass(frozen=True)
class Retirement:
    """What one retirement row states (spec 3 r30 section 5): the
    sentence text when the row carries it (a learner-caused retirement
    always does; an F13 row written before r30 does not), the reason,
    and the replacement hint a learner comment gave.
    """
    text: str | None
    reason: str
    replacement_hint: str | None


def retirements(cache: CacheReader) -> dict[str, Retirement]:
    """text_sha -> the newest retirement row's facts, over every port
    "attempt" backend "run" kind "retirement" row on record
    (cachekeys.RetirementKey, attempts.retire_sentence). A retirement row
    outlives the sentences row it accompanied (deleted right after), so
    this still finds it once the sentence itself is gone.

    A retirement whose reading was struck (spec 5 r10) keeps its reason
    and loses its replacement hint: the deletion already happened and the
    text is never re-adopted, but a struck reading must stop steering the
    drafter (`retirement_evidence`, derivations.refused_drafts). The
    strike lands under the comment's subject and the retirement row under
    the sentence's own, so the two never meet in one subject's row set --
    `vetoed_readings_all` is the fold that sees it, which is why this one
    takes the whole cache rather than a row sequence. A retirement row
    from F13 names no reading and is never touched.
    """
    struck = vetoed_readings_all(cache)
    out: dict[str, Retirement] = {}
    for r in cache.rows_since("attempt", "run", 0):
        if r.question.get("kind") != "retirement":
            continue
        ref = _reading_ref(r)
        hint = None if ref in struck else (r.question.get("replacement_hint") or None)
        out[r.subject] = Retirement(text=r.question.get("text") or None,
                                    reason=str(r.question.get("reason") or "retired"),
                                    replacement_hint=hint)
    return out


def retired_texts(cache: CacheReader) -> frozenset[str]:
    """Every sentence text_sha the run has ever retired (F13 and the
    comment pass, spec 3 section 5): `retirements`'s keys --
    derivations.adoptable_drafts reads it to never re-adopt the same
    text, and refused_drafts reads it to keep telling the drafter not to
    propose it again.
    """
    return frozenset(retirements(cache))


def retirement_evidence(r: Retirement) -> str:
    """The drafter's line for a retired text (derivations.refused_drafts):
    why it was retired, and the learner's replacement hint when the
    retiring comment gave one.
    """
    line = f"retired: {r.reason}"
    return f"{line}; replacement hint: {r.replacement_hint}" if r.replacement_hint else line


def excluded_candidates(cache: CacheReader, subject: str) -> list[dict[str, str | None]]:
    """`subject`'s excluded candidates from the newest RunReport row:
    {"sha", "reason"}, in the order the run recorded them (spec 5
    section 1 kind 1's "rejected candidates").
    """
    newest = cache.latest("run", "runreport", RunReportKey())
    if newest is None:
        return []
    items = newest.answer.get("excluded_items") or []
    return [{"sha": item.get("artifact_sha"), "reason": item.get("reason")}
           for item in items if item.get("subject") == subject]


def card_flags(rows: Sequence[Answer]) -> list[str]:
    """Every card-level flag among `rows` (question["kind"] ==
    "card-flag", carrying `anchor` and `card_kind`) as
    "ANCHOR::CARD_KIND", first-seen order (spec 4 section 4).
    """
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        if r.question.get("kind") != "card-flag":
            continue
        anchor, card_kind = r.question.get("anchor"), r.question.get("card_kind")
        if not anchor or not card_kind:
            continue
        label = f"{anchor}::{card_kind}"
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def normalize_shown(shown: Mapping[str, Any]) -> dict[str, Any]:
    """A gallery note's own `shown`, in the current shape (spec 5
    section 1 r5, fix round 4): a legacy singular `recording` value
    (the shape round 1 wrote, before round 3 switched a card's own
    recordings to a list) becomes `recordings: [value]` (`[]` for a
    `recording` of None), and the legacy key is always dropped -- an
    already-list `recordings`, present alongside a stray `recording`,
    is kept as-is rather than overwritten. Without this, comparing a
    `recording`-shaped row against a `recordings`-shaped "what the card
    shows now" reads every field different on the recording account
    alone, even when nothing about the recording actually changed --
    `card_notes` applies this to every row's `shown` on the way out, so
    every caller (reviewserver.py's `_is_stale` included) only ever
    sees the one shape.
    """
    if not isinstance(shown, Mapping):
        return {}
    result = dict(shown)
    if "recording" in result:
        legacy = result.pop("recording")
        if "recordings" not in result:
            result["recordings"] = [legacy] if legacy is not None else []
    return result


def card_notes(rows: Sequence[Answer], anchor: str, card_kind: str) -> list[dict[str, Any]]:
    """One card's own notes (spec 5 section 1 r5): every card-flag row
    among `rows` whose question["anchor"] == `anchor` AND
    question["card_kind"] == `card_kind`, oldest first, each as
    {"text", "ts", "shown"} -- `shown` is the artifacts (and sentence
    text_sha, syllabus state id) that row's own append named the card
    as displaying, {} for a row written before that field existed (it
    named nothing, so reviewserver.py's staleness check has nothing to
    compare and never flags it), run through `normalize_shown` so a
    legacy row's singular `recording` never reads different from the
    current `recordings`-list shape on that account alone.

    Both `anchor` and `card_kind` are required: a subject alone does
    not identify one card -- a word note's several card kinds (reading,
    production) share a subject, and a minimal-pair note's two member
    cards share BOTH a subject and a card_kind ("recognition"), told
    apart only by their own per-card anchor (compile.py's MemberKey,
    the same `card_id` append_gallery_note already records).
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        if (r.question.get("kind") != "card-flag" or r.question.get("card_kind") != card_kind
                or r.question.get("anchor") != anchor):
            continue
        answer = r.answer if isinstance(r.answer, Mapping) else {}
        out.append({"text": answer.get("note") or "", "ts": r.ts,
                    "shown": normalize_shown(r.question.get("shown") or {}),
                    "comment_sha": comment_identity(r.key_sha, r.ts)})
    return out


@dataclass(frozen=True)
class Comment:
    """One learner comment (spec 5 r9): a card-flag row with note text,
    identified by its own row (cachekeys.comment_identity). `subject_kind`
    is None for a row written before the field existed; `question_kind`
    and `artifact_kind` are set for a session comment only."""
    comment_sha: str
    subject: str
    subject_kind: str | None
    anchor: str
    card_kind: str
    text: str
    ts: int
    shown: Mapping[str, Any]
    question_kind: str | None
    artifact_kind: str | None


def comment_of(row: Answer) -> Comment | None:
    """`row` as a Comment, or None when it is not one: a card-flag row
    whose answer carries a non-empty note (an Anki flag import row
    carries none, anki_import's `{"flagged": ...}` answer)."""
    if row.question.get("kind") != "card-flag":
        return None
    answer = row.answer if isinstance(row.answer, Mapping) else {}
    text = answer.get("note")
    if not isinstance(text, str) or not text.strip():
        return None
    q = row.question
    return Comment(comment_sha=comment_identity(row.key_sha, row.ts), subject=row.subject,
                   subject_kind=q.get("subject_kind"), anchor=str(q.get("anchor") or ""),
                   card_kind=str(q.get("card_kind") or ""), text=text, ts=row.ts,
                   shown=normalize_shown(q.get("shown") or {}),
                   question_kind=q.get("question_kind"), artifact_kind=q.get("artifact_kind"))


def comments_of(rows: Sequence[Answer]) -> list[Comment]:
    """Every comment among `rows`, oldest first."""
    return [c for r in sorted(rows, key=lambda r: r.ts) if (c := comment_of(r)) is not None]


def comments(cache: CacheReader) -> list[Comment]:
    """Every comment on record, every subject, oldest first."""
    return comments_of(cache.rows_since("assess", "learner", 0))


def reading_of(rows: Sequence[Answer], comment_sha: str,
               prompt_version: str | None = None) -> Answer | None:
    """The newest comment-reading row among `rows` for `comment_sha`,
    under `prompt_version` when given, else any version; None when none.
    """
    mine = [r for r in rows if r.question.get("kind") == "comment-reading"
            and r.question.get("comment_sha") == comment_sha
            and (prompt_version is None or r.question.get("prompt_version") == prompt_version)]
    return max(mine, key=lambda r: r.ts) if mine else None


def _reading_ref(row: Answer) -> tuple[str, str] | None:
    """The (comment_sha, prompt_version) reading `row` names, or None
    when it names neither or only half of one: both halves together are
    the reference (a prompt version bump is a new reading), so a row
    carrying one alone refers to no reading at all.
    """
    comment_sha, version = row.question.get("comment_sha"), row.question.get("prompt_version")
    if not comment_sha or not version:
        return None
    return (str(comment_sha), str(version))


def vetoed_readings(rows: Sequence[Answer]) -> frozenset[tuple[str, str]]:
    """Every (comment_sha, prompt_version) a comment-veto row among
    `rows` strikes (spec 5 r10). A veto row naming only half a reference
    strikes nothing rather than everything that half-matches it.
    """
    return frozenset(ref for r in rows if r.question.get("kind") == "comment-veto"
                     and (ref := _reading_ref(r)) is not None)


def vetoed_readings_all(cache: CacheReader) -> frozenset[tuple[str, str]]:
    """Every (comment_sha, prompt_version) struck anywhere on the record.
    `vetoed_readings` folds over one subject's rows, which is enough for
    every row the comment pass writes back under the comment's own
    subject; a replacement sentence is the exception -- it is drafted
    under DRAFT_SUBJECT while the strike lands under the comment's
    subject, so the two never meet in one subject's row set. This fold
    reads the veto rows themselves (every one is a learner assess row),
    so a reader of rows written under another subject can still see the
    strike (`sentence_drafts`).
    """
    return vetoed_readings(cache.rows_since("assess", "learner", 0))


# The two row kinds a veto never hides: they are the reading itself and
# the strike on it, which the feedback screen reads back (reading_view).
_READING_ROW_KINDS = ("comment-reading", "comment-veto")


def without_vetoed_readings(rows: Sequence[Answer]) -> list[Answer]:
    """`rows` minus every row derived from a vetoed reading (spec 3 r30
    section 5): a row whose question names a (comment_sha,
    prompt_version) a comment-veto row among `rows` strikes -- the
    reading and veto rows themselves excepted. A row naming no reading,
    or only half of one (`_reading_ref`), is never dropped -- every
    learner row written before the comment pass names none. `rows_for`,
    `learner_ratings` and `directions` apply this, so every fold over
    directions and ratings ignores a struck reading without knowing
    anything about comments; a caller must hand in the subject's whole
    row set for the veto to be seen.
    """
    struck = vetoed_readings(rows)
    if not struck:
        return list(rows)
    return [r for r in rows
            if r.question.get("kind") in _READING_ROW_KINDS
            or (ref := _reading_ref(r)) is None or ref not in struck]


def action_label(action: Mapping[str, Any]) -> str:
    """The short label the screen shows for one recorded action (spec 5
    r10): what was done, and when refused, why.
    """
    name = action.get("action")
    if name == "direction":
        label = f"direction for the {action.get('kind')} search: {action.get('text')}"
    elif name == "retire_sentence":
        label = f"sentence retired: {action.get('reason')}"
    elif name == "replacement_sentence":
        label = f"replacement drafted: {action.get('thai')} ({action.get('gloss')})"
    elif name == "rate":
        label = f"rating {action.get('value')} on the {action.get('kind')}"
    elif name == "gloss_on":
        label = f"gloss on: {action.get('word')}"
    elif name == "none":
        label = f"no action: {action.get('remark')}"
    else:
        label = str(name)
    if action.get("outcome") == "refused":
        return f"{label} (not done: {action.get('reason')})"
    return label


def reading_view(rows: Sequence[Answer], comment_sha: str) -> dict[str, Any] | None:
    """A comment's reading as the page shows it (spec 5 r10): the
    reading text, one label per action, the unactionable requests, the
    prompt version (the strike names it) and whether it is vetoed. None
    while unread. A reading with no action and one unactionable line --
    what a comment the reader passed over is recorded as -- renders as
    an empty `actions` list beside that line.
    """
    row = reading_of(rows, comment_sha)
    if row is None:
        return None
    version = str(row.question.get("prompt_version"))
    answer = row.answer if isinstance(row.answer, Mapping) else {}
    return {"reading": answer.get("reading"), "prompt_version": version,
            "vetoed": (comment_sha, version) in vetoed_readings(rows),
            "actions": [action_label(a) for a in answer.get("actions") or []
                        if isinstance(a, Mapping)],
            "unactionable": [f"no action available: {u}" for u in answer.get("unactionable") or []]}


def gloss_on_requested(rows: Sequence[Answer]) -> bool:
    """A gloss-on row on the subject (the comment pass's `gloss_on`
    action, spec 3 r30 section 5) whose reading is not vetoed.
    """
    return any(r.backend == "learner" and r.question.get("kind") == "gloss-on"
               for r in without_vetoed_readings(rows))


def run_reports(cache: CacheReader) -> list[Answer]:
    """Every run.py RunReport row (port="run", backend="runreport",
    subject="run"), oldest first: one per run() call, each carrying every
    RunReport field in its own `answer`.
    """
    return [r for r in cache.assessments_of("run") if r.question.get("kind") == "runreport"]


def unresolved_batch(
        cache: CacheReader) -> tuple[str, tuple[str, ...], tuple[str, ...],
                                     tuple[str, ...]] | None:
    """The (batch_id, subjects, roles, kinds) of the newest judge-batch
    marker row whose latest status is "submitted"; the three lists are
    parallel, one entry per question that batch asked -- a question's own
    (subject, kind) need is read off them by index. None while no batch
    is out.
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
           tuple(newest.question.get("roles", [])), tuple(newest.question["kinds"]))


# --- sentence drafts --------------------------------------------------------

@dataclass(frozen=True)
class SentenceDraft:
    """One drafted sentence as the LLM answered it: clauses of vocabulary
    ids, the rendered text, and the L1 gloss (spec 1 section 1, spec 3
    section 5). Which Targets it fills is derived, never carried here:
    fills is membership of a target's word in `clauses` (spec 1 section
    3), checked against the Sentence `draft_sentence` builds.
    """
    clauses: Clauses
    text: str
    gloss: str

    @property
    def text_sha(self) -> str:
        return text_sha(self.text)


def parse_drafts(text: str) -> list[SentenceDraft]:
    """The listings one llm answer item's JSON carries, one `SentenceDraft`
    per listing, not merged -- a text repeated in the same item comes
    back as several entries here; empty when `text` is not the JSON the
    drafting prompt asked for. An item lacking `clauses` or `text` is
    skipped; one whose `clauses` do not parse (clauses_from_json) is
    skipped with a `logging` warning naming the text's first 40
    characters. `merge_drafts` does the merging, over this item alone
    (`drafts_in`) or over a whole run's items (`sentence_attempt`).
    """
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return []
    drafted = (data.get("sentences") if isinstance(data, Mapping) else None) or []
    out: list[SentenceDraft] = []
    for d in drafted:
        if not isinstance(d, Mapping) or not d.get("text") or "clauses" not in d:
            continue
        one_text = str(d["text"]).strip()
        try:
            clauses = clauses_from_json(d["clauses"])
        except ValueError as e:
            _log.warning("parse_drafts: skipping a draft with malformed clauses (text %r): %s",
                         one_text[:40], e)
            continue
        out.append(SentenceDraft(clauses=clauses, text=one_text, gloss=str(d.get("gloss") or "")))
    return out


def parse_no_fit(text: str) -> str | None:
    """The reason one llm answer item gives for drafting nothing at all
    (spec 3 r19 section 5's no-fit answer): the stripped `reason` of a
    `{"sentences": [], "reason": "<non-empty string>"}` item, else None.
    An item listing any sentence is a draft, not a no-fit answer, and one
    with no `sentences` key, no reason, a blank reason, or a reason that
    is not a string, is not recognizable as either -- `parse_drafts`
    reads the same item for its drafts, and the two never both answer.
    """
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, Mapping) or "sentences" not in data or data.get("sentences"):
        return None
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None
    return reason.strip()


def parse_phrases(text: str) -> dict[str, str]:
    """The subject -> phrase map one phrase-drafting answer's JSON
    carries (spec 3 section 5, attempts._phrase_prompt's own {"phrases":
    [{"subject": "...", "phrase": "..."}]} shape): an item lacking a
    `subject` or a non-empty string `phrase` is skipped. Empty when
    `text` is not that JSON. A subject listed twice keeps the last one
    listed.
    """
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return {}
    items = (data.get("phrases") if isinstance(data, Mapping) else None) or []
    out: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        subject, phrase = item.get("subject"), item.get("phrase")
        if not subject or not isinstance(phrase, str) or not phrase.strip():
            continue
        out[str(subject)] = phrase.strip()
    return out


# --- the comment pass: reading one learner comment (spec 3 r30 section 5) ---

@dataclass(frozen=True)
class CommentReading:
    """One comment's reading as the reader answered it (spec 3 r30
    section 5): one line saying what the learner means, the vocabulary
    actions to take (each a mapping carrying `action` and that action's
    own parameters, COMMENT_ACTIONS), and the requests the deck cannot
    act on. Either tuple may be empty: a comment the reader passed over
    is recorded as a reading with no action and one unactionable line,
    and a `none` action is an action with no side effect -- an executed
    act, not an unactionable request, told apart by its own name.
    """
    reading: str
    actions: tuple[Mapping[str, Any], ...]
    unactionable: tuple[str, ...]


def _valid_action(a: Any) -> bool:
    """Whether `a` is one action of the COMMENT_ACTIONS vocabulary with
    every parameter present and typed. Whether a well-formed `gloss_on`
    names a word the deck can act on is a separate question the comment's
    own subject answers (`_names_another_word`).
    """
    if not isinstance(a, Mapping) or a.get("action") not in COMMENT_ACTIONS:
        return False
    for param in COMMENT_ACTIONS[a["action"]]:
        if param == "value":
            v = a.get("value")
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 4:
                return False
        elif param == "kind":
            if a.get("kind") not in _COMMENT_ARTIFACT_KINDS:
                return False
        elif param == "replacement_hint":
            if a.get(param) is not None and not isinstance(a.get(param), str):
                return False
        elif not isinstance(a.get(param), str) or not a[param].strip():
            return False
    return True


def _names_another_word(a: Any, subject: str | None) -> bool:
    """A well-formed `gloss_on` naming a word other than the comment's
    own subject (design decision 3): the deck shows the gloss on the
    commented card, and has no way to act on a different one. Always
    False while the comment's subject is unknown -- there is nothing to
    check the word against.
    """
    return a.get("action") == "gloss_on" and subject is not None and a.get("word") != subject


def parse_comment_readings(text: str,
                           subjects: Mapping[str, str] | None = None
                           ) -> dict[str, CommentReading]:
    """The comment sha -> reading map one comment-reading answer's JSON
    carries ({"readings": [{"comment", "reading", "actions",
    "unactionable"}]}). An item lacking a string `comment` or a non-empty
    string `reading` is skipped; an action outside COMMENT_ACTIONS or
    missing a parameter is demoted to `unactionable` as "unrecognized
    action: <its json>" and logged, so the item keeps its reading and is
    not re-asked.

    `subjects` is the handed comment sha -> subject map when the caller
    has it (attempts.comment_attempt hands one): only the shas it names
    are read back -- an item naming any other, a hallucinated or stale
    one, is dropped, so nothing is ever executed against a comment
    nobody asked about -- and a `gloss_on` naming a word other than that
    comment's own subject is demoted as "gloss-on names another word:
    <its json>". With no map (wiring's recognizer, which has no handed
    set) the read is shape-only. Empty when `text` is not that JSON.
    """
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return {}
    items = (data.get("readings") if isinstance(data, Mapping) else None) or []
    out: dict[str, CommentReading] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        comment, reading = item.get("comment"), item.get("reading")
        if not isinstance(comment, str) or not isinstance(reading, str) or not reading.strip():
            continue
        if subjects is not None and comment not in subjects:
            _log.warning("parse_comment_readings: dropping a reading for an unhanded comment %s",
                         comment)
            continue
        subject = subjects.get(comment) if subjects is not None else None
        actions: list[Mapping[str, Any]] = []
        unactionable = [str(u) for u in (item.get("unactionable") or []) if str(u).strip()]
        for a in item.get("actions") or []:
            well_formed = _valid_action(a)
            if well_formed and not _names_another_word(a, subject):
                actions.append(dict(a))
                continue
            rendered = json.dumps(a, ensure_ascii=False, sort_keys=True, default=str)
            why = "gloss-on names another word" if well_formed else "unrecognized action"
            _log.warning("parse_comment_readings: comment %s: %s %s", comment, why, rendered)
            unactionable.append(f"{why}: {rendered}")
        out[comment] = CommentReading(reading=reading.strip(), actions=tuple(actions),
                                      unactionable=tuple(unactionable))
    return out


def merge_drafts(drafts: Sequence[SentenceDraft]) -> list[SentenceDraft]:
    """One draft per distinct text among `drafts`, in first-seen order:
    their gloss is the first non-empty one. Differing clauses reject the
    text (a `logging` warning names its first 40 characters); a differing
    non-empty gloss does not -- spec 3 r19 section 5's "a text listed
    twice is one candidate ... differing glosses keep the first, since
    the verdict is keyed by the text and was given on that gloss" -- the
    disagreement is only logged, at DEBUG.
    """
    order: list[str] = []
    glosses: dict[str, str] = {}
    clauses: dict[str, Clauses] = {}
    conflicted: set[str] = set()
    for d in drafts:
        one_text = d.text
        if one_text not in glosses:
            order.append(one_text)
            glosses[one_text] = d.gloss
            clauses[one_text] = d.clauses
            continue
        if d.clauses != clauses[one_text]:
            conflicted.add(one_text)
        if d.gloss and glosses[one_text] and d.gloss != glosses[one_text]:
            _log.debug("merge_drafts: keeping the first gloss for a text with differing glosses: %r",
                      one_text[:40])
        elif d.gloss and not glosses[one_text]:
            glosses[one_text] = d.gloss
    for one_text in order:
        if one_text in conflicted:
            _log.warning("merge_drafts: dropping a draft with conflicting clauses: %r",
                         one_text[:40])
    return [SentenceDraft(clauses=clauses[one_text], text=one_text, gloss=glosses[one_text])
            for one_text in order if one_text not in conflicted]


def draft_sentence(draft: SentenceDraft, today: Callable[[], date]) -> Sentence:
    """`draft` as a Sentence value: its own clauses, learner_voice,
    provenance llm/draft. `today` is called once for the provenance date
    (Sourcing.today / adoptable_drafts' own `today` parameter).
    """
    return Sentence(clauses=draft.clauses, text=draft.text, gloss=draft.gloss,
                    voice="learner_voice",
                    provenance=Provenance(source="llm", origin="draft",
                                          licence="generated", acquired=today()))


def drafts_in(text: str) -> list[SentenceDraft]:
    """The drafts one llm answer item carries, `merge_drafts` applied
    over that item's own listings alone."""
    return merge_drafts(parse_drafts(text))


def sentence_drafts(cache: CacheReader) -> list[SentenceDraft]:
    """Every sentence draft any run's drafting ask produced, newest ask
    last -- what there is to adopt once the verdicts land. `merge_drafts`
    runs once over every provide row's own parsed drafts, across rows as
    well as within one: a text drafted in two separate runs is one draft
    here, the same as `sentence_attempt`'s own merge over one run's
    items.

    A drafting row the comment pass wrote -- a replacement sentence
    (attempts._draft_replacement), marked with the reading that produced
    it -- is dropped once that reading is struck (spec 5 r10): striking
    a reading unmakes what it did, and the replacement is one of the
    things it did. The strike lives under the comment's own subject, not
    DRAFT_SUBJECT, so the global fold (`vetoed_readings_all`) is what
    sees it; the filter sits here rather than in one caller so that both
    readers of the drafts -- derivations.adoptable_drafts and the run's
    own recovery step -- lose the draft together. A drafting ask's own
    row names no reading and is never touched.
    """
    struck = vetoed_readings_all(cache)
    raw: list[SentenceDraft] = []
    for row in rows_for(cache, DRAFT_SUBJECT, "sentence"):
        if row.port != "provide":
            continue
        ref = _reading_ref(row)
        if ref is not None and ref in struck:
            continue
        raw.extend(d for item in row.answer.get("items", []) for d in parse_drafts(str(item)))
    return merge_drafts(raw)


# --- sentence parsing (spec 2 r10 section 4 / spec 3 r16 section 5) --------

def vocabulary_line(word: Word) -> str:
    """One vocabulary entry's own text -- id, Thai form, English meaning
    -- with no leading marker: a caller prefixes its own list marker
    (attempts._sentence_prompt's `"- " + vocabulary_line(w)` for its
    vocabulary section, `"- target {id}: " + vocabulary_line(w)` for a
    target/introducible line; `parse_prompt` likewise).
    """
    return f"{word.id}  {word.thai}  ({word.meaning})"


def parse_prompt(texts: Sequence[str], vocabulary: Sequence[Word]) -> str:
    """The migration parse ask's prompt (spec 3 r16 section 5): the full
    registered vocabulary as `vocabulary_line`, the clause-rendering
    rule, and each of `texts` numbered. Asks
    {"parses": [{"text": "...", "clauses": [["id", ...], ...]}]}.
    """
    vocab_lines = "\n".join("- " + vocabulary_line(w) for w in vocabulary)
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, start=1))
    return (
        "Parse each numbered Thai sentence below into clauses of vocabulary ids, "
        "drawing only on this vocabulary:\n"
        f"{vocab_lines}\n"
        "A clause renders as its words' Thai concatenated in order with no "
        'separator; clauses within one sentence join with one space; a repeated '
        'word renders as its Thai form followed by ๆ and is written [id, "ๆ"]. '
        "Find the clauses whose rendering reproduces each sentence exactly.\n"
        f"Sentences:\n{numbered}\n"
        'Output JSON only: {"parses": [{"text": "...", '
        '"clauses": [["id", ...], ...]}]}')


def parses_in(text: str) -> dict[str, Clauses]:
    """The text -> Clauses map one llm-parse answer item's JSON carries
    (`parse_prompt`'s own {"parses": [...]} shape), keyed by each entry's
    exact text. An item lacking `text` or `clauses` is skipped; one whose
    `clauses` do not parse (clauses_from_json) is skipped with a
    `logging` warning naming the text's first 40 characters. Empty when
    `text` is not the JSON the parsing prompt asked for.
    """
    try:
        data = json.loads(strip_fences(text))
    except (json.JSONDecodeError, TypeError):
        return {}
    parsed = (data.get("parses") if isinstance(data, Mapping) else None) or []
    out: dict[str, Clauses] = {}
    for p in parsed:
        if not isinstance(p, Mapping) or not p.get("text") or "clauses" not in p:
            continue
        one_text = str(p["text"]).strip()
        try:
            out[one_text] = clauses_from_json(p["clauses"])
        except ValueError as e:
            _log.warning("parses_in: skipping a parse with malformed clauses (text %r): %s",
                         one_text[:40], e)
    return out
