"""Identity types for the syllabus domain: plain `str` at runtime
(typing.NewType), distinct to a type checker.
"""
import re
from collections.abc import Collection
from typing import NewType

WordId = NewType("WordId", str)
ConfusionId = NewType("ConfusionId", str)
PairId = NewType("PairId", str)
TargetId = NewType("TargetId", str)

# A Grapheme's identity is its `symbol` (spec 1 section 1); this alias
# names that identity space in OrderEntry and Finding.note_id contexts.
GraphemeId = NewType("GraphemeId", str)

CategoryName = NewType("CategoryName", str)


def slug_id(text: str, taken: Collection[str] = ()) -> WordId:
    """The WordId a closure Word takes from its English gloss (design
    2026-09-12 §1): lowercased, every run of non-alphanumerics a single
    hyphen, trimmed; suffixed `-2`, `-3`, ... while `taken` holds it (the
    live convention, `delicious-2`). A gloss with no alphanumeric
    character has no id and is refused, naming it.
    """
    base = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    if not base:
        raise ValueError(f"no word id can be made from gloss {text!r}")
    if base not in taken:
        return WordId(base)
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return WordId(f"{base}-{n}")
