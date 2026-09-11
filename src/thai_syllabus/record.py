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

from .cachekeys import RunReportKey
from .entities import Clauses, Sentence, Word, clauses_from_json, text_sha
from .media import Provenance
from .ports import Answer, CacheReader
from .transport import strip_fences

_log = logging.getLogger(__name__)

__all__ = ["LEARNER_RANK", "rows_for", "source_asks", "candidate_shas", "learner_ratings",
          "ratings_for_role", "latest_rating", "directions", "judge_verdicts",
          "latest_query", "tried_urls", "latest_nothing_reason",
          "asks_since", "spend_since", "cost_since", "unresolved_batch", "run_reports",
          "subject_kind_of", "retired_texts",
          "DRAFT_SUBJECT", "SentenceDraft",
          "parse_drafts", "parse_no_fit", "merge_drafts", "draft_sentence", "drafts_in",
          "sentence_drafts",
          "excluded_candidates", "card_flags",
          "PARSE_SUBJECT", "vocabulary_line", "parse_prompt", "parses_in"]

# The subject every sentence-drafting ask is appended under: drafts are
# proposed for a run's open Targets as a set, not for one subject.
DRAFT_SUBJECT = "sentence-drafts"

# The subject every sentence-parsing ask is appended under (spec 2 r10
# section 4's migration parse step): the whole worklist of rows lacking
# clauses is one ask, not one subject per row.
PARSE_SUBJECT = "sentence-parses"

# provide rows from these backends are not Source asks (spec 3 section 3
# vocabulary: an attempt is one Source ask): imgfetch/audiofetch write the
# candidate a Source ask already caused, and a learner row is a supply --
# an answer, not an ask. a legacy-current row is a migrated candidate's
# provenance (spec 2 section 4), an answer with no ask.
_NOT_SOURCE_ASK_BACKENDS = ("imgfetch", "audiofetch", "learner", "legacy-current")

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
    return sorted((r for r in rows
                  if r.port == "provide" and r.backend not in _NOT_SOURCE_ASK_BACKENDS),
                 key=lambda r: r.ts)


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


def retired_texts(cache: CacheReader) -> frozenset[str]:
    """Every sentence text_sha the run has ever retired (F13, spec 3
    section 5): the subject of every port "attempt" backend "run" kind
    "retirement" row on record (cachekeys.RetirementKey,
    run._retire_exhausted_sentence). A retirement row outlives the
    sentences row it accompanied (deleted right after), so this still
    finds it once the sentence itself is gone -- derivations.adoptable_drafts
    reads it to never re-adopt the same text, and refused_drafts reads it
    to keep telling the drafter not to propose it again.
    """
    return frozenset(r.subject for r in cache.rows_since("attempt", "run", 0)
                     if r.question.get("kind") == "retirement")


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
    """
    raw: list[SentenceDraft] = []
    for row in rows_for(cache, DRAFT_SUBJECT, "sentence"):
        if row.port != "provide":
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
