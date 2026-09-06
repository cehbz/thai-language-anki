"""Identity types for the syllabus domain: plain `str` at runtime
(typing.NewType), distinct to a type checker.
"""
from typing import NewType

WordId = NewType("WordId", str)
ConfusionId = NewType("ConfusionId", str)
PairId = NewType("PairId", str)
TargetId = NewType("TargetId", str)

# A Grapheme's identity is its `symbol` (spec 1 section 1); this alias
# names that identity space in OrderEntry and Finding.note_id contexts.
GraphemeId = NewType("GraphemeId", str)

CategoryName = NewType("CategoryName", str)
