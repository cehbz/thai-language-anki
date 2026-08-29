from pathlib import Path

import yaml
from pydantic import BaseModel


class GenConfig(BaseModel):
    lexicon_top_n: int = 3000
    sentence_base: int = 300
    test_spelling_rank: int = 300
    max_iterations: int = 5
    model: str = "claude-opus-5"   # passed to claude -p --model; also the cache key namespace
    images: bool = True            # wire live image search (openverse/wikimedia)
    imgfetch: str = "imgfetch"     # path to the imgfetch binary (tools/imgfetch); bare name = PATH lookup
    search_proxy: str | None = None   # e.g. socks5h://127.0.0.1:1080; image SEARCH only (Openverse blocks this egress)
    thai1000_apkg: str | None = None  # path to a thai1000 apkg, deck-root-relative


def load_config(deck_root: Path) -> GenConfig:
    """Load <deck_root>/gen.yaml overrides if present, else defaults."""
    path = Path(deck_root) / "gen.yaml"
    if not path.exists():
        return GenConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GenConfig(**data)
