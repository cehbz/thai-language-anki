"""Loaders for rulebook data files. Data lives in the repo's data/ directory
(the evaluator runs from the repo checkout, not an installed wheel)."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

@dataclass
class ContrastEntry:
    id: str
    kind: str
    weight: float

@lru_cache
def load_contrasts(path: Path | None = None) -> list[ContrastEntry]:
    raw = yaml.safe_load((path or DATA_DIR / "contrasts.yaml").read_text()) or []
    return [ContrastEntry(**e) for e in raw]

@lru_cache
def load_spelling_targets(path: Path | None = None) -> dict[str, list[str]]:
    return yaml.safe_load((path or DATA_DIR / "spelling_targets.yaml").read_text()) or {}

@lru_cache
def load_function_words(path: Path | None = None) -> set[str]:
    return set(yaml.safe_load((path or DATA_DIR / "function_words.yaml").read_text()) or [])

@lru_cache
def load_g2p_exceptions(path: Path | None = None) -> dict[str, str]:
    return yaml.safe_load((path or DATA_DIR / "g2p_exceptions.yaml").read_text()) or {}

class FileFrequencyList:
    def __init__(self, path: Path | None = None):
        lines = (path or DATA_DIR / "frequency_th.txt").read_text().splitlines()
        words = [w for w in lines if w and not w.startswith("#")]
        self._rank = {w: i + 1 for i, w in enumerate(words)}
    def rank(self, word: str) -> int | None:
        return self._rank.get(word)
