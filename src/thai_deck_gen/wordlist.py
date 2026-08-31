import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

from thai_deck_eval.data_io import load_categories

THAI_RE = re.compile(r"^[ก-๛]+$")

WORDLIST_PROMPT = """You are drafting Thai vocabulary for a spaced-repetition
deck based on the Fluent Forever 625-word list, category "{category}".

Target: {count} entries in this category, colloquial spoken register.

Top-200 Thai frequency words for anchoring word choice (most common first):
{anchors}

Instructions:
- Every noun must include a classifier (Thai measure word).
- Where a single English concept splits into multiple distinct Thai words
  (e.g. formality register, physical vs. abstract sense), emit one entry
  per Thai word and set split_of to the original English concept.
- Prefer colloquial, everyday spoken Thai over formal/written register.
- Respond with a YAML list of mappings ONLY, no prose, no code fences.
  Each mapping has keys: thai, gloss, category, part_of_speech
  (noun|verb|adjective|other), classifier (required for nouns, else omit),
  picturable (bool, default true), split_of (omit unless splitting a concept).
"""

EXTENSION_PROMPT = """You are extending the Thai vocabulary of a spaced-repetition
deck based on the Fluent Forever 625-word list, category "{category}".
The learner's everyday conversations center on: {theme}.

Target: {count} additional entries in this category that are useful in
that context, colloquial spoken register. Only words that this category
genuinely admits; skip the category's generic staples.

Already listed in this category (do not repeat):
{existing}

Top-200 Thai frequency words for anchoring word choice (most common first):
{anchors}

Instructions:
- Every noun must include a classifier (Thai measure word).
- Where a single English concept splits into multiple distinct Thai words
  (e.g. formality register, physical vs. abstract sense), emit one entry
  per Thai word and set split_of to the original English concept.
- Prefer colloquial, everyday spoken Thai over formal/written register.
- Respond with a YAML list of mappings ONLY, no prose, no code fences.
  Each mapping has keys: thai, gloss, category, part_of_speech
  (noun|verb|adjective|other), classifier (required for nouns, else omit),
  picturable (bool, default true), split_of (omit unless splitting a concept).
"""


class WordEntry(BaseModel):
    thai: str
    gloss: str
    category: str
    part_of_speech: Literal["noun", "verb", "adjective", "other"]
    classifier: str | None = None
    picturable: bool = True
    image_query: str | None = None   # what a photo of this word looks like
    split_of: str | None = None
    emphasis: bool = False         # added by the emphasis extension pass

    @field_validator("thai")
    @classmethod
    def _thai_script(cls, v: str) -> str:
        if not THAI_RE.match(v):
            raise ValueError("thai must match ^[ก-๛]+$")
        return v

    @model_validator(mode="after")
    def _classifier_required_for_noun(self) -> "WordEntry":
        if self.part_of_speech == "noun" and not self.classifier:
            raise ValueError("classifier required when part_of_speech is noun")
        return self


def load_word_list(path: Path, categories_path: Path) -> list[WordEntry]:
    categories = set(load_categories(categories_path))
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    errors = []
    entries = []
    for i, item in enumerate(raw):
        category = item.get("category")
        if category not in categories:
            errors.append(f"entry {i}: unknown category {category!r}")
            continue
        try:
            entries.append(WordEntry(**item))
        except ValueError as exc:
            errors.append(f"entry {i}: {exc}")
    if errors:
        raise ValueError("; ".join(errors))
    return entries


def _load_frequency_anchors(path: Path, n: int = 200) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    words = [w for w in lines if w and not w.startswith("#")]
    return words[:n]


def draft_word_list(llm, categories_path: Path, frequency_path: Path,
                     out_path: Path, prompt_version: str = "wl1",
                     warnings: list[str] | None = None) -> int:
    if warnings is None:
        warnings = []
    categories = load_categories(categories_path)
    anchors = _load_frequency_anchors(frequency_path)
    per_category = -(-625 // len(categories))
    entries = _load_entries(out_path)
    done = {e.category for e in entries}
    for category in categories:
        if category in done:
            continue
        prompt = WORDLIST_PROMPT.format(
            category=category, count=per_category, anchors=", ".join(anchors))
        entries.extend(_parse_entries(llm.complete(prompt), category, warnings))
        _write_entries(entries, out_path)
    _write_entries(entries, out_path)
    return len(entries)


FENCE_RE = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _strip_fences(response: str) -> str:
    """Models sometimes wrap the YAML in a markdown code fence regardless."""
    m = FENCE_RE.match(response)
    return m.group(1) if m else response


def _parse_entries(response: str, category: str, warnings: list[str],
                   emphasis: bool = False) -> list[WordEntry]:
    try:
        raw = yaml.safe_load(_strip_fences(response)) or []
    except yaml.YAMLError as exc:
        warnings.append(f"{category}: unparseable response: {exc}")
        return []
    parsed = []
    for item in raw:
        if not isinstance(item, dict):
            warnings.append(f"{category}: dropped non-mapping entry {item!r}")
            continue
        item = dict(item, category=category, emphasis=emphasis)
        try:
            parsed.append(WordEntry(**item))
        except ValueError as exc:
            warnings.append(f"{category}: dropped invalid entry: {exc}")
    return parsed


def _load_entries(out_path: Path) -> list[WordEntry]:
    if not Path(out_path).is_file():
        return []
    existing = yaml.safe_load(Path(out_path).read_text(encoding="utf-8")) or []
    return [WordEntry(**item) for item in existing]


def _write_entries(entries: list[WordEntry], out_path: Path) -> None:
    entries.sort(key=lambda e: (e.category, e.thai))
    Path(out_path).write_text(
        yaml.safe_dump([_dump(e) for e in entries], allow_unicode=True),
        encoding="utf-8")


def _dump(entry: WordEntry) -> dict:
    data = entry.model_dump(exclude_none=True)
    if not data["emphasis"]:
        del data["emphasis"]
    return data


def extend_word_list(llm, categories_path: Path, frequency_path: Path,
                     out_path: Path, emphasis, warnings: list[str] | None = None) -> int:
    """Add theme-relevant entries on top of an existing base list.

    Per category: extra = round(base_count x (weight - 1)); categories at
    weight <= 1 are skipped, as are categories that already carry
    emphasis-tagged entries (resume). Returns the number of emphasis
    entries in the file afterwards.
    """
    if warnings is None:
        warnings = []
    categories = load_categories(categories_path)
    anchors = _load_frequency_anchors(frequency_path)
    per_category = -(-625 // len(categories))
    entries = _load_entries(out_path)
    done = {e.category for e in entries if e.emphasis}
    for category in categories:
        extra = round(per_category * (emphasis.weight(category) - 1))
        if extra <= 0 or category in done:
            continue
        existing = [e.thai for e in entries if e.category == category]
        prompt = EXTENSION_PROMPT.format(
            category=category, theme=emphasis.theme, count=extra,
            existing=", ".join(existing) or "(none)", anchors=", ".join(anchors))
        entries.extend(_parse_entries(llm.complete(prompt), category, warnings,
                                      emphasis=True))
        _write_entries(entries, out_path)
    _write_entries(entries, out_path)
    return sum(1 for e in entries if e.emphasis)


IMAGE_QUERY_PROMPT = """You are writing image-search phrases for a Thai
vocabulary deck, category "{category}".

For each word below, give an English phrase that describes WHAT A PHOTOGRAPH
OF THAT CONCEPT LOOKS LIKE -- not the word itself. The phrase is fed to a
stock-photo search, so it must name something a camera can capture.

- Concrete words: name the object plus enough context to disambiguate the
  sense ("orange" is a fruit here, not a colour or a phone network).
- Abstract words, pronouns and verbs: describe a scene that depicts them.
  "I" -> a person pointing at their own chest. "go" -> a person walking away
  down a road. "tomorrow" -> a calendar page being turned.
- Never ask for text, writing, captions or logos in the image.
- 3 to 7 words. English only.

Words:
{words}

Respond with a YAML mapping ONLY (no prose, no code fences), Thai word as
the key and the phrase as the value.
"""


def draft_image_queries(llm, word_list_path: Path,
                        warnings: list[str] | None = None) -> int:
    """Fill in `image_query` for picturable entries that lack one.

    Drafted per category and written back after each, so a run killed by a
    session limit keeps everything it has already drafted. Existing phrases
    are never redrafted -- the word list stays a hand-editable artifact.
    """
    warnings = warnings if warnings is not None else []
    path = Path(word_list_path)
    rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("image_query") or not row.get("picturable", True):
            continue
        by_category.setdefault(row.get("category", ""), []).append(row)

    filled = 0
    for category, pending in by_category.items():
        listing = "\n".join(
            f"- {r['thai']}: {r.get('gloss', '')} ({r.get('part_of_speech', '')})"
            for r in pending)
        response = llm.complete(
            IMAGE_QUERY_PROMPT.format(category=category, words=listing))
        try:
            drafted = yaml.safe_load(_strip_fences(response)) or {}
        except yaml.YAMLError as exc:
            warnings.append(f"{category}: unparseable image-query response: {exc}")
            continue
        if not isinstance(drafted, dict):
            warnings.append(f"{category}: image-query response was not a mapping")
            continue
        for row in pending:
            phrase = drafted.get(row["thai"])
            if isinstance(phrase, str) and phrase.strip():
                row["image_query"] = phrase.strip()
                filled += 1
        path.write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    return filled


def apply_query_proposals(word_list_path: Path, proposals_path: Path) -> int:
    """Adopt judge-proposed image phrases, marking them as machine-written.

    A phrase you wrote by hand is never overwritten: `image_query_source:
    human` is the opt-out, and everything a judge wrote carries `judge` so
    the two are always tellable apart.
    """
    proposals_path, word_list_path = Path(proposals_path), Path(word_list_path)
    if not proposals_path.exists():
        return 0
    proposals = yaml.safe_load(proposals_path.read_text(encoding="utf-8")) or {}
    rows = yaml.safe_load(word_list_path.read_text(encoding="utf-8")) or []
    applied = 0
    for row in rows:
        proposal = proposals.get(row.get("thai"))
        if not proposal or row.get("image_query_source") == "human":
            continue
        phrase = (proposal.get("suggestion") or "").strip()
        if not phrase or phrase == row.get("image_query"):
            continue
        row["image_query"] = phrase
        row["image_query_source"] = "judge"
        applied += 1
    if applied:
        word_list_path.write_text(
            yaml.safe_dump(rows, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
    return applied
