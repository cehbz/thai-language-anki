from ..core.findings import Severity
from .model import Report

def render_text(rep: Report) -> str:
    lines = [f"deck: {rep.deck_name} v{rep.deck_version}   "
             f"rulebook v{rep.rulebook_version}   gate: {rep.gate.upper()}",
             f"scores  integrity {rep.scores.integrity:.0f}  "
             f"language {rep.scores.language:.0f}  "
             f"method {rep.scores.method:.0f}  content {rep.scores.content:.0f}",
             f"stages  ran: {', '.join(rep.stages_run) or '-'}"
             + (f"   skipped: {', '.join(rep.stages_skipped)}"
                if rep.stages_skipped else "")]
    for sev in (Severity.ERROR, Severity.WARN, Severity.INFO):
        fs = [f for f in rep.findings if f.severity == sev]
        if fs:
            lines.append(f"\n{str(sev).upper()} ({len(fs)}):")
            for f in fs:
                rid = f.evidence.get("rule_override", f.rule)
                where = f" [{f.note_id}]" if f.note_id else ""
                lines.append(f"  {rid}{where}: {f.message}")
    if rep.metrics:
        lines.append("\nmetrics:")
        for m in rep.metrics:
            lines.append(f"  {m.name}: {m.value:.2f}")
    return "\n".join(lines) + "\n"
