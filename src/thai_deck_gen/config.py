from pathlib import Path

import yaml
from pydantic import BaseModel


class GenConfig(BaseModel):
    lexicon_top_n: int = 3000
    sentence_base: int = 300
    test_spelling_rank: int = 300
    max_iterations: int = 5
    model: str = "claude"          # cache key namespace for CliBackend
    images: bool = True            # wire live image search (openverse/wikimedia)
    thai1000_apkg: str | None = None  # path to a thai1000 apkg, deck-root-relative


def load_config(deck_root: Path) -> GenConfig:
    """Load <deck_root>/gen.yaml overrides if present, else defaults."""
    path = Path(deck_root) / "gen.yaml"
    if not path.exists():
        return GenConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GenConfig(**data)
