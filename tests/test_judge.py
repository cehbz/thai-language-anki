import thai_deck_eval.stages.judge_rules  # noqa: F401
from thai_deck_eval.config import RulebookConfig
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.judge.core import CachedJudge, FakeJudge, JudgeRequest, Verdict
from thai_deck_eval.model.deck import load_deck
from tests.helpers import DeckBuilder

def _ctx(root, judge):
    return EvalContext(deck=load_deck(root), config=RulebookConfig(), judge=judge)

def test_confidence_floor_falls_back_on_dict_config(tmp_path):
    judge = FakeJudge({"s-1": [Verdict(rule="judge/unnatural-sentence",
                                       passed=False, confidence=0.3,
                                       rationale="maybe")]})
    ctx = EvalContext(deck=load_deck(DeckBuilder(tmp_path).build()),
                      config={"sentence_base": 2}, judge=judge)
    res = run_pipeline(ctx, stages=[Stage.JUDGE])
    f = next(f for f in res.findings if f.rule == "judge/unnatural-sentence")
    assert f.severity == Severity.INFO  # 0.3 < the 0.6 rulebook default

def test_all_pass_yields_nothing(tmp_path):
    res = run_pipeline(_ctx(DeckBuilder(tmp_path).build(), FakeJudge({})),
                       stages=[Stage.JUDGE])
    assert res.findings == []

def test_failed_verdict_becomes_finding(tmp_path):
    judge = FakeJudge({"s-1": [Verdict(rule="judge/unnatural-sentence",
                                       passed=False, confidence=0.9,
                                       rationale="word order is English-like")]})
    res = run_pipeline(_ctx(DeckBuilder(tmp_path).build(), judge),
                       stages=[Stage.JUDGE])
    f = next(f for f in res.findings if f.rule == "judge/unnatural-sentence")
    assert f.note_id == "s-1" and f.severity == Severity.ERROR
    assert "English-like" in f.message

def test_low_confidence_demoted_to_info(tmp_path):
    judge = FakeJudge({"s-1": [Verdict(rule="judge/unnatural-sentence",
                                       passed=False, confidence=0.3,
                                       rationale="maybe")]})
    res = run_pipeline(_ctx(DeckBuilder(tmp_path).build(), judge),
                       stages=[Stage.JUDGE])
    f = next(f for f in res.findings if f.rule == "judge/unnatural-sentence")
    assert f.severity == Severity.INFO

def test_cache_hits_skip_inner(tmp_path):
    class Counting:
        calls = 0
        def judge(self, req):
            Counting.calls += 1
            return [Verdict(rule=r, passed=True, confidence=1.0, rationale="")
                    for r in req.rules]
    cached = CachedJudge(Counting(), tmp_path / "c.sqlite", "m", "1")
    req = JudgeRequest(note_id="n", rules=["judge/x"], prompt="p")
    a = cached.judge(req)
    b = cached.judge(req)
    assert a == b and Counting.calls == 1
    cached2 = CachedJudge(Counting(), tmp_path / "c.sqlite", "m", "1")
    assert cached2.judge(req) == a and Counting.calls == 1  # persists
    cached3 = CachedJudge(Counting(), tmp_path / "c.sqlite", "m", "2")
    cached3.judge(req)
    assert Counting.calls == 2  # prompt_version bump invalidates
