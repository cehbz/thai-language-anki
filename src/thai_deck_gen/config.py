from pathlib import Path

import yaml
from pydantic import BaseModel


class SecretsConfig(BaseModel):
    """References to API keys, never the keys themselves.

    Each value is an `op://<vault>/<item>/<field>` 1Password reference or a
    path to an owner-only (0600) file. See thai_deck_eval.secrets.
    """
    forvo: str | None = None        # forvo pronunciations (native audio)
    google_tts: str | None = None   # google cloud text-to-speech (sentence audio)
    openai: str | None = None       # gpt-image-1 fallback for unillustrated words
    anthropic: str | None = None    # api drafting, off the subscription quota


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
    forvo_request_limit: int | None = None  # max forvo lookups per run (free tier: 500/day)
    image_candidates: int = 5      # candidates judged per picture word before one is kept
    rulebook: str | None = None    # evaluator rulebook supplying the judge for image checks
    llm_backend: str = "cli"       # cli spends subscription tokens (already paid for);
                                   # api spends cash, for work the CLI can't do
    secrets: SecretsConfig = SecretsConfig()


def load_config(deck_root: Path) -> GenConfig:
    """Load <deck_root>/gen.yaml overrides if present, else defaults."""
    path = Path(deck_root) / "gen.yaml"
    if not path.exists():
        return GenConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GenConfig(**data)
