from dataclasses import dataclass, field

@dataclass
class ProducerResult:
    added: int = 0
    changed: int = 0
    blocked: list[str] = field(default_factory=list)
