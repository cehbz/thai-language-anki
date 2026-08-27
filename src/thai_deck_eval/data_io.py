"""Loaders for rulebook data files. Data lives in the repo's data/ directory
(the evaluator runs from the repo checkout, not an installed wheel)."""
from dataclasses import dataclass
from pathlib import Path
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

@dataclass
class ContrastEntry:
    id: str
    kind: str
    weight: float

def load_contrasts(path: Path | None = None) -> list[ContrastEntry]:
    raw = yaml.safe_load((path or DATA_DIR / "contrasts.yaml").read_text())
    return [ContrastEntry(**e) for e in raw]

def load_spelling_targets(path: Path | None = None) -> dict[str, list[str]]:
    return yaml.safe_load((path or DATA_DIR / "spelling_targets.yaml").read_text())

def load_function_words(path: Path | None = None) -> set[str]:
    return set(yaml.safe_load((path or DATA_DIR / "function_words.yaml").read_text()))

def load_g2p_exceptions(path: Path | None = None) -> dict[str, str]:
    return yaml.safe_load((path or DATA_DIR / "g2p_exceptions.yaml").read_text())

class FileFrequencyList:
    def __init__(self, path: Path | None = None):
        lines = (path or DATA_DIR / "frequency_th.txt").read_text().splitlines()
        words = [w for w in lines if w and not w.startswith("#")]
        self._rank = {w: i + 1 for i, w in enumerate(words)}
    def rank(self, word: str) -> int | None:
        return self._rank.get(word)
