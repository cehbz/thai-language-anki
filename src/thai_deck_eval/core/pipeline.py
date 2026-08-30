from dataclasses import dataclass, field
from pathlib import Path
from .context import EvalContext
from .findings import Dimension, Finding, Metric, Severity, Stage
from .registry import rules_for
from ..model.deck import DeckSchemaError, load_deck
from ..waivers import load_waivers, partition

ORDER = [Stage.MECHANICAL, Stage.LINGUISTIC, Stage.METHOD, Stage.JUDGE]

# "schema" is a pseudo-stage: a schema failure never reaches run_pipeline
# (evaluate_path skips everything itself in that case), so a dependency on
# "schema" is always satisfied here.
DEFAULT_DEPENDS_ON: dict[str, list[str]] = {
    "mechanical": ["schema"],
    "linguistic": ["schema"],
    "method": ["schema"],
    "judge": ["schema", "mechanical", "linguistic"],
}

@dataclass
class EvalResult:
    findings: list[Finding] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    stages_run: list[Stage] = field(default_factory=list)
    stages_skipped: list[Stage] = field(default_factory=list)
    waived: list[Finding] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

def run_pipeline(ctx: EvalContext, stages: list[Stage] | None = None) -> EvalResult:
    res = EvalResult()
    enabled = stages if stages is not None else ORDER
    depends_on = ctx.cfg("depends_on", DEFAULT_DEPENDS_ON)
    # Per-stage outcomes tracked by stage-name string (Stage.value), so they
    # line up with depends_on's keys/values.
    errored: set[str] = set()
    skipped: set[str] = set()
    for stage in ORDER:
        if stage not in enabled:
            res.stages_skipped.append(stage)
            skipped.add(stage.value)
            continue
        deps = depends_on.get(stage.value, [])
        # "schema" is always satisfied inside run_pipeline (see
        # DEFAULT_DEPENDS_ON); any other dependency that errored or was
        # itself skipped (transitively unverified) blocks this stage.
        if any(d != "schema" and (d in errored or d in skipped) for d in deps):
            res.stages_skipped.append(stage)
            skipped.add(stage.value)
            continue
        produced: list[Finding] = []
        for rd in rules_for(stage):
            for item in rd.fn(ctx) or []:
                (res.metrics if isinstance(item, Metric) else produced).append(item)

        # Waivers apply before the gate check, so a reviewed-and-accepted
        # error blocks neither the gate nor the stages that depend on it.
        kept, waived, stale = partition(produced, ctx.deck, ctx.waivers)
        res.findings.extend(kept)
        res.waived.extend(waived)
        for waiver in stale:
            res.findings.append(Finding(
                rule="waiver/stale", severity=Severity.INFO,
                dimension=Dimension.INTEGRITY, note_id=waiver.note_id,
                message=(f"waiver for {waiver.rule} no longer matches this note's "
                         f"image; re-review it or drop the waiver")))
        res.stages_run.append(stage)
        if any(f.severity == Severity.ERROR for f in kept):
            errored.add(stage.value)
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
    ctx = ctx_factory(deck)
    if not ctx.waivers:
        ctx.waivers = load_waivers(deck.root)
    return run_pipeline(ctx, stages=stages)
