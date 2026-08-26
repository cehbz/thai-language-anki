from thai_deck_eval.core.findings import Dimension, Finding, Metric, Severity, Stage
from thai_deck_eval.core.registry import _REGISTRY, rule, rules_for

def test_rule_registration_and_finding_defaults():
    @rule("mech/example", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
    def example(ctx):
        yield example.finding("boom", note_id="n1")
    try:
        rd = next(r for r in rules_for(Stage.MECHANICAL) if r.id == "mech/example")
        f = list(rd.fn(None))[0]
        assert isinstance(f, Finding)
        assert (f.rule, f.severity, f.dimension, f.note_id) == (
            "mech/example", Severity.ERROR, Dimension.INTEGRITY, "n1")
    finally:
        _REGISTRY.pop("mech/example")

def test_metric_defaults():
    m = Metric(name="coverage/pairs", value=0.5)
    assert m.dimension == Dimension.METHOD

def test_duplicate_id_rejected():
    @rule("mech/dup", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
    def a(ctx): ...
    try:
        import pytest
        with pytest.raises(ValueError):
            @rule("mech/dup", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
            def b(ctx): ...
    finally:
        _REGISTRY.pop("mech/dup")
