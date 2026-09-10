"""The learner profile (spec 1 section 2): register, and per-category
emphasis. Confusion training weights are derived state (curated seed x
StudyRecord evidence), spec 2/3's territory.
"""
from dataclasses import dataclass, field
from typing import Literal

from .ids import CategoryName


@dataclass(frozen=True)
class Profile:
    register: Literal["male_colloquial"]
    emphasis: dict[CategoryName, float] = field(default_factory=dict)
    productive_cutoff: int = 2000   # spec 1 section 2, r9: the frequency
                                     # rank at or above which a categorized
                                     # Word carries a productive Target
                                     # (Nation's high-frequency line)
