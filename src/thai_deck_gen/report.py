import hashlib, yaml
from pathlib import Path
from pydantic import BaseModel

class GapFinding(BaseModel):
    rule: str; severity: str; note_id: str | None = None; message: str

class Gaps(BaseModel):
    gate: str
    missing_contrasts: list[str]
    pair_by_note: dict[str, str]
    missing_categories: list[str]
    frequency_covered: int
    speaker_value: float
    findings: list[GapFinding]

    def findings_for(self, rule_prefix: str) -> list[GapFinding]:
        return [f for f in self.findings if f.rule.startswith(rule_prefix)]

def _metric(report: dict, name: str) -> dict:
    for m in report.get("metrics", []):
        if m["name"] == name:
            return m
    return {"value": 0.0, "detail": {}}

def parse_report(report: dict, contrasts_path: Path) -> Gaps:
    entries = yaml.safe_load(Path(contrasts_path).read_text())
    order = {e["id"]: (-e["weight"], i) for i, e in enumerate(entries)}
    pairs = _metric(report, "coverage/minimal_pairs")
    missing = sorted(pairs["detail"].get("missing", []),
                     key=lambda c: order.get(c, (0, 999)))
    return Gaps(
        gate=report["gate"],
        missing_contrasts=missing,
        pair_by_note=pairs["detail"].get("by_note", {}),
        missing_categories=_metric(report, "coverage/categories")
            ["detail"].get("missing", []),
        frequency_covered=round(_metric(report, "coverage/frequency")["value"] * 625),
        speaker_value=_metric(report, "speakers/minimal_pairs")["value"],
        findings=[GapFinding(**{k: f.get(k) for k in
                                ("rule", "severity", "note_id", "message")})
                  for f in report.get("findings", [])])

def fingerprint(gaps: Gaps) -> str:
    return hashlib.sha256(gaps.model_dump_json().encode()).hexdigest()
