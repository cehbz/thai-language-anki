"""thai_syllabus: the domain core (spec 1) for the redesigned deck.

Fresh, self-contained, stdlib-only (dataclasses, no pydantic). Imports
nothing from thai_deck_eval or thai_deck_gen.
"""
from .entities import (
    Grapheme,
    MinimalPair,
    Pronunciation,
    Sentence,
    SoundConfusion,
    Syllable,
    Target,
    Word,
)
from .media import Picture, Provenance, Recording, Speaker
from .profile import Profile
from .rules import Compile, Finding, Gaps, Metric, Report, Rule
from .syllabus import Syllabus, TargetLike

__all__ = [
    "Grapheme",
    "MinimalPair",
    "Pronunciation",
    "Sentence",
    "SoundConfusion",
    "Syllable",
    "Target",
    "Word",
    "Picture",
    "Provenance",
    "Recording",
    "Speaker",
    "Profile",
    "Compile",
    "Finding",
    "Gaps",
    "Metric",
    "Report",
    "Rule",
    "Syllabus",
    "TargetLike",
]
