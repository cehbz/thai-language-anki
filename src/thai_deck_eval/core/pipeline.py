from dataclasses import dataclass, field
from pathlib import Path
from .context import EvalContext
from .findings import Dimension, Finding, Metric, Severity, Stage
from .registry import rules_for
from ..model.deck import DeckSchemaError, load_deck

ORDER = [Stage.MECHANICAL, Stage.LINGUISTIC, Stage.METHOD, Stage.JUDGE]

@dataclass
class EvalResult:
    findings: list[Finding] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    stages_run: list[Stage] = field(default_factory=list)
    stages_skipped: list[Stage] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

def run_pipeline(ctx: EvalContext, stages: list[Stage] | None = None) -> EvalResult:
    res = EvalResult()
    enabled = stages if stages is not None else ORDER
    gated = False
    for stage in ORDER:
        if stage not in enabled:
            continue
        if gated:
            res.stages_skipped.append(stage)
            continue
        for rd in rules_for(stage):
            for item in rd.fn(ctx) or []:
                (res.metrics if isinstance(item, Metric) else res.findings).append(item)
        res.stages_run.append(stage)
        if ctx.cfg("gates", True) and res.has_errors:
            gated = True
    return res

def evaluate_path(path: Path, ctx_factory, stages=None) -> EvalResult:
    try:
        deck = load_deck(path)
    except DeckSchemaError as e:
        res = EvalResult(stages_skipped=list(ORDER))
        res.findings = [Finding(rule="schema/invalid", severity=Severity.ERROR,
                                dimension=Dimension.INTEGRITY, message=i)
                        for i in e.issues]
        return res
    return run_pipeline(ctx_factory(deck), stages=stages)
