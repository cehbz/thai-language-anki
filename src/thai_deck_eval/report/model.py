from typing import Literal
from pydantic import BaseModel
from ..core.findings import Finding, Metric, Severity
from .scoring import Scores

class Report(BaseModel):
    deck_name: str
    deck_version: str
    rulebook_version: str
    stages_run: list[str]
    stages_skipped: list[str]
    findings: list[Finding]
    metrics: list[Metric]
    scores: Scores
    gate: Literal["pass", "fail"]

def build_report(name, version, result, scores, config) -> Report:
    return Report(
        deck_name=name, deck_version=version,
        rulebook_version=config.version,
        stages_run=[str(s) for s in result.stages_run],
        stages_skipped=[str(s) for s in result.stages_skipped],
        findings=result.findings, metrics=result.metrics, scores=scores,
        gate="fail" if result.has_errors else "pass")
