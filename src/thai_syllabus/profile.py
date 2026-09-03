"""The learner profile (spec 1, section 2).

Confusion training weights are NOT stored here: derived as seed (curated
data) x StudyRecord evidence, which is spec 2/3's territory. L1 is implicit
in curated inputs.
"""
from dataclasses import dataclass, field
from typing import Literal

from .ids import Category


@dataclass(frozen=True)
class Profile:
    register: Literal["male_colloquial"]
    emphasis: dict[Category, float] = field(default_factory=dict)
