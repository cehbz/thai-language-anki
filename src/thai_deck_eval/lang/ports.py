from typing import Protocol
from .ipa import IpaSyllable

class G2P(Protocol):
    def syllables(self, word: str) -> list[IpaSyllable] | None: ...

class Tokenizer(Protocol):
    def tokens(self, text: str) -> list[str]: ...

class FrequencyList(Protocol):
    def rank(self, word: str) -> int | None: ...
