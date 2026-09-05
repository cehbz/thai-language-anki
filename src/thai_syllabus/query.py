"""The picture search query (spec 3 section 5): the phrase a human or a
judge drafted, else the gloss's head term plus the word's category
qualifier.

Both image corpora index English metadata, so the query is English. The
qualifier separates the senses a gloss alone conflates ("orange" the
fruit from "orange" the colour); data/image_query_hints.yaml maps a
category name to its qualifier, and a category with no entry contributes
none.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

from .entities import Word
from .ids import CategoryName

__all__ = ["head_term", "picture_query", "load_query_hints", "QUERY_HINTS"]

# Repo data/ directory, resolved as curated.py resolves it: project input
# data outside any one deck's curated/*.yaml.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Where one alternative of a learner definition ends and the next begins.
_ALTERNATIVES = (",", ";", "/", " or ")

# A leading parenthetical is a prefix note -- "(for) a long time" -- not a
# sense note, so it is dropped rather than cut at.
_LEADING_NOTE = re.compile(r"^\s*\([^)]*\)\s*")


def head_term(meaning: str) -> str:
    """The core of a learner definition: leading note dropped, the first
    alternative before a parenthetical or an alternative separator.
    """
    head = _LEADING_NOTE.sub("", meaning).split("(")[0]
    for separator in _ALTERNATIVES:
        head = head.split(separator)[0]
    return head.strip()


def picture_query(word: Word, category: CategoryName | None, phrase: str | None,
                  hints: Mapping[str, str]) -> str:
    """`phrase` (a learner direction or a judge suggestion) verbatim; else
    the gloss head term with `category`'s qualifier where `hints` names
    one. Refuses a word whose gloss reduces to nothing: an image search
    that cannot describe its object has no query (F11).
    """
    if phrase:
        return phrase
    head = head_term(word.meaning)
    if not head:
        raise ValueError(
            f"word {word.id!r} has no gloss to search a picture for "
            f"(meaning={word.meaning!r})")
    qualifier = hints.get(category) if category is not None else None
    return f"{head} {qualifier}" if qualifier else head


def load_query_hints(path: str | Path) -> dict[str, str]:
    """category name -> search qualifier, from a YAML mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image query hints file not found: {path}")
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


# Loaded once at import, as curated.CATEGORY_NAMES is: the repo's
# category -> qualifier table every picture query draws on.
QUERY_HINTS: dict[str, str] = load_query_hints(_DATA_DIR / "image_query_hints.yaml")
