from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field

class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["cli", "api", "fake"] = "cli"
    model: str = "claude-opus-5"
    effort: str = "medium"
    confidence_floor: float = 0.6
    prompt_version: str = "1"
    cache_path: str = ".thai-deck-eval-cache.sqlite"

class RulebookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "1"
    gates: bool = True
    taper_rank: int = 300
    sentence_base: int = 300
    target_speakers: int = 3
    deductions: dict[str, float] = Field(
        default_factory=lambda: {"error": 25.0, "warn": 2.0, "info": 0.0})
    metric_weights: dict[str, float] = Field(
        default_factory=lambda: {"coverage/minimal_pairs": 3.0,
                                 "coverage/spelling": 2.0,
                                 "coverage/frequency": 3.0,
                                 "speakers/minimal_pairs": 1.0})
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

def load_rulebook(path: Path | None) -> RulebookConfig:
    if path is None:
        return RulebookConfig()
    return RulebookConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})
