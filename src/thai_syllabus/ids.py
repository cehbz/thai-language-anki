"""Identity types for the syllabus domain.

Plain `str` at runtime (via `typing.NewType`) so entities stay stdlib-only
and JSON/hash friendly; the wrapper only buys static-typing distinction
between, say, a WordId and a PairId.
"""
from typing import NewType

WordId = NewType("WordId", str)
ConfusionId = NewType("ConfusionId", str)
PairId = NewType("PairId", str)
TargetId = NewType("TargetId", str)

# Grapheme has no separate id field (spec: "symbol: str  # identity"); this
# alias names the identity space a Grapheme's `symbol` lives in, for use in
# OrderEntry and Finding.note_id contexts.
GraphemeId = NewType("GraphemeId", str)

CategoryName = NewType("CategoryName", str)
