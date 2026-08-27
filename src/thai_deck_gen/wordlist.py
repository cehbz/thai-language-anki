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


class WordEntry(BaseModel):
    thai: str
    gloss: str
    category: str
    part_of_speech: Literal["noun", "verb", "adjective", "other"]
    classifier: str | None = None
    picturable: bool = True
    split_of: str | None = None

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
    entries = []
    for category in categories:
        prompt = WORDLIST_PROMPT.format(
            category=category, count=per_category, anchors=", ".join(anchors))
        response = llm.complete(prompt)
        try:
            raw = yaml.safe_load(response) or []
        except yaml.YAMLError as exc:
            warnings.append(f"{category}: unparseable response: {exc}")
            continue
        for item in raw:
            if not isinstance(item, dict):
                warnings.append(f"{category}: dropped non-mapping entry {item!r}")
                continue
            item = dict(item, category=category)
            try:
                entries.append(WordEntry(**item))
            except ValueError as exc:
                warnings.append(f"{category}: dropped invalid entry: {exc}")
                continue
    entries.sort(key=lambda e: (e.category, e.thai))
    Path(out_path).write_text(
        yaml.safe_dump([e.model_dump(exclude_none=True) for e in entries],
                       allow_unicode=True),
        encoding="utf-8")
    return len(entries)
