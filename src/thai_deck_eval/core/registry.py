from dataclasses import dataclass
from typing import Callable
from .findings import Dimension, Finding, Severity, Stage

@dataclass
class RuleDef:
    id: str
    stage: Stage
    dimension: Dimension
    default_severity: Severity
    fn: Callable

    def finding(self, message, note_id=None, severity=None, evidence=None) -> Finding:
        return Finding(rule=self.id, severity=severity or self.default_severity,
                       dimension=self.dimension, message=message,
                       note_id=note_id, evidence=evidence or {})

_REGISTRY: dict[str, RuleDef] = {}

def rule(rule_id: str, stage: Stage, dimension: Dimension, default_severity: Severity):
    def deco(fn):
        if rule_id in _REGISTRY:
            raise ValueError(f"duplicate rule id {rule_id}")
        rd = RuleDef(rule_id, stage, dimension, default_severity, fn)
        _REGISTRY[rule_id] = rd
        fn.finding = rd.finding
        fn.rule_def = rd
        return fn
    return deco

def rules_for(stage: Stage) -> list[RuleDef]:
    return [r for r in _REGISTRY.values() if r.stage == stage]
