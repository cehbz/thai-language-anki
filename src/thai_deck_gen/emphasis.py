"""Learner emphasis profile: what the learner actually talks about.

Read from data/emphasis.yaml. Consumers: the word list extension pass,
themed sentence generation, and emphasis-weighted intro ordering.
"""
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class Emphasis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str
    category_weights: dict[str, float] = {}

    def weight(self, category: str) -> float:
        """Per-category weight; `default` key applies to unlisted categories."""
        return self.category_weights.get(
            category, self.category_weights.get("default", 1.0))


def load_emphasis(path: Path) -> Emphasis | None:
    path = Path(path)
    if not path.is_file():
        return None
    return Emphasis(**(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))
