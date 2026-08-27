from thai_deck_eval.config import RulebookConfig, load_rulebook
from thai_deck_eval.core.findings import Dimension, Finding, Metric, Severity
from thai_deck_eval.core.pipeline import EvalResult
from thai_deck_eval.report.scoring import compute_scores

def _f(dim, sev):
    return Finding(rule="x/y", severity=sev, dimension=dim, message="m")

def test_defaults_load():
    cfg = load_rulebook(None)
    assert cfg.judge.backend == "cli" and cfg.gates is True

def test_deductions():
    res = EvalResult(findings=[_f(Dimension.INTEGRITY, Severity.WARN),
                               _f(Dimension.INTEGRITY, Severity.ERROR),
                               _f(Dimension.LANGUAGE, Severity.INFO)])
    s = compute_scores(res, RulebookConfig())
    assert s.integrity == 73 and s.language == 100

def test_method_blend():
    res = EvalResult(metrics=[Metric(name="coverage/minimal_pairs", value=0.5),
                              Metric(name="coverage/frequency", value=1.0)],
                     findings=[_f(Dimension.METHOD, Severity.WARN)])
    s = compute_scores(res, RulebookConfig())
    # (3*0.5 + 3*1.0) / 6 = 0.75 → 75 - 2 = 73
    assert s.method == 73

def test_method_zero_without_metrics():
    assert compute_scores(EvalResult(), RulebookConfig()).method == 0.0

def test_rulebook_file(tmp_path):
    p = tmp_path / "rb.yaml"
    p.write_text("taper_rank: 100\njudge:\n  backend: fake\n")
    cfg = load_rulebook(p)
    assert cfg.taper_rank == 100 and cfg.judge.backend == "fake"
