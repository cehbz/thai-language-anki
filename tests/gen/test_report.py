import json
from pathlib import Path
from thai_deck_gen.report import Gaps, fingerprint, parse_report

FIXTURE = Path(__file__).parent / "fixtures" / "report_gaps.json"
CONTRASTS = Path(__file__).parents[2] / "data" / "contrasts.yaml"

def _gaps() -> Gaps:
    return parse_report(json.loads(FIXTURE.read_text()), CONTRASTS)

def test_missing_contrasts_ordered_by_weight():
    # mid-high weight 4 > vowel_length weight 4 — tie broken by inventory order
    assert _gaps().missing_contrasts == ["tone:mid-high", "vowel_length"]

def test_parses_metrics_and_findings():
    g = _gaps()
    assert g.missing_categories == ["Animals", "Colors"]
    assert g.frequency_covered == 250
    assert g.pair_by_note == {"mp-tone-mid-low-1": "tone:mid-low"}
    assert g.findings_for("mech/")[0].note_id == "mp-tone-mid-low-1"

def test_fingerprint_stable_and_sensitive():
    a, b = _gaps(), _gaps()
    assert fingerprint(a) == fingerprint(b)
    b.missing_categories.append("Jobs")
    assert fingerprint(a) != fingerprint(b)
