from dataclasses import dataclass, field
from typing import Any
from ..model.deck import Deck

@dataclass
class EvalContext:
    deck: Deck
    config: Any = field(default_factory=dict)
    g2p: Any = None
    g2p_second: Any = None
    tokenizer: Any = None
    freq: Any = None
    judge: Any = None
    waivers: list = field(default_factory=list)  # reviewed-and-accepted findings

    def cfg(self, key: str, default=None):
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)
