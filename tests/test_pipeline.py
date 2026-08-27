from pathlib import Path
from thai_deck_eval.core.findings import Dimension, Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.core.registry import _REGISTRY, rule
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import DeckMeta, StagePlan

def _deck():
    return Deck(meta=DeckMeta(name="t", version="0",
                stage_plan=StagePlan(phases=["sounds"])), root=Path("."))

def _with_rules(rules, fn):
    try:
        fn()
    finally:
        for rid in rules:
            _REGISTRY.pop(rid, None)

def test_error_gates_later_stages():
    # Default depends_on: mechanical/linguistic/method each depend only on
    # "schema" (always satisfied inside run_pipeline), so a mechanical error
    # must NOT skip linguistic/method -- only judge, which depends on
    # mechanical (and linguistic), is gated.
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
        assert Stage.LINGUISTIC not in res.stages_skipped
        assert Stage.METHOD not in res.stages_skipped
        assert Stage.JUDGE in res.stages_skipped
        assert ran == [1]
        assert [f.rule for f in res.findings] == ["mech/t-fail"]
    _with_rules(["mech/t-fail", "lang/t-probe"], go)

def test_transitive_skip_via_stage_filter():
    # mechanical is filtered out of `stages` -> lands in stages_skipped ->
    # judge's dependency on "mechanical" is unsatisfied (transitively
    # skipped, not just errored) -> judge is also skipped. linguistic has
    # no dependency on mechanical, so it still runs.
    ran = []
    @rule("lang/t-probe2", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
    def probe(ctx):
        ran.append(1)
        return []
    def go():
        res = run_pipeline(EvalContext(deck=_deck()),
                           stages=[Stage.LINGUISTIC, Stage.JUDGE])
        assert Stage.LINGUISTIC in res.stages_run
        assert ran == [1]
        assert Stage.MECHANICAL in res.stages_skipped
        assert Stage.JUDGE in res.stages_skipped
    _with_rules(["lang/t-probe2"], go)

def test_custom_depends_on_via_dict_config():
    # A dict config can override depends_on entirely: here linguistic is
    # made to depend on mechanical, so a mechanical error skips it (which
    # it would NOT do under the default depends_on).
    @rule("mech/t-fail2", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
    def fail(ctx):
        yield fail.finding("bad")
    ran = []
    @rule("lang/t-probe3", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
    def probe(ctx):
        ran.append(1)
        return []
    def go():
        cfg = {"depends_on": {"linguistic": ["mechanical"]}}
        res = run_pipeline(EvalContext(deck=_deck(), config=cfg))
        assert Stage.LINGUISTIC in res.stages_skipped
        assert ran == []
    _with_rules(["mech/t-fail2", "lang/t-probe3"], go)

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
        assert "coverage/x" in [m.name for m in res.metrics]
    _with_rules(["mech/t-warn", "meth/t-metric"], go)

def test_stage_filter():
    ran = []
    @rule("judge/t-probe", Stage.JUDGE, Dimension.CONTENT, Severity.WARN)
    def p(ctx):
        ran.append(1)
        return []
    def go():
        res = run_pipeline(EvalContext(deck=_deck()), stages=[Stage.MECHANICAL])
        assert ran == []
        assert Stage.JUDGE in res.stages_skipped  # filtered-out stages report as skipped
    _with_rules(["judge/t-probe"], go)
