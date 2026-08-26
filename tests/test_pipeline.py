from thai_deck_eval.core.findings import Dimension, Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.core.registry import _REGISTRY, rule
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import DeckMeta, StagePlan

def _deck():
    return Deck(meta=DeckMeta(name="t", version="0",
                stage_plan=StagePlan(phases=["sounds"])))

def _with_rules(rules, fn):
    try:
        fn()
    finally:
        for rid in rules:
            _REGISTRY.pop(rid, None)

def test_error_gates_later_stages():
    @rule("mech/t-fail", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
    def fail(ctx):
        yield fail.finding("bad")
    ran = []
    @rule("lang/t-probe", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
    def probe(ctx):
        ran.append(1)
        return []
    def go():
        res = run_pipeline(EvalContext(deck=_deck()))
        assert Stage.LINGUISTIC in res.stages_skipped
        assert ran == []
        assert [f.rule for f in res.findings] == ["mech/t-fail"]
    _with_rules(["mech/t-fail", "lang/t-probe"], go)

def test_warn_does_not_gate_and_metrics_collected():
    @rule("mech/t-warn", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
    def w(ctx):
        yield w.finding("meh")
    @rule("meth/t-metric", Stage.METHOD, Dimension.METHOD, Severity.INFO)
    def m(ctx):
        from thai_deck_eval.core.findings import Metric
        yield Metric(name="coverage/x", value=0.5)
    def go():
        res = run_pipeline(EvalContext(deck=_deck()))
        assert res.stages_skipped == []
        assert [m.name for m in res.metrics] == ["coverage/x"]
    _with_rules(["mech/t-warn", "meth/t-metric"], go)

def test_stage_filter():
    ran = []
    @rule("judge/t-probe", Stage.JUDGE, Dimension.CONTENT, Severity.WARN)
    def p(ctx):
        ran.append(1)
        return []
    def go():
        run_pipeline(EvalContext(deck=_deck()), stages=[Stage.MECHANICAL])
        assert ran == []
    _with_rules(["judge/t-probe"], go)
