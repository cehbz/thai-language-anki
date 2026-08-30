import thai_deck_eval.stages.judge_rules  # noqa: F401
from thai_deck_eval.config import RulebookConfig
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.judge.core import CachedJudge, FakeJudge, JudgeRequest, Verdict
from thai_deck_eval.model.deck import load_deck
from tests.helpers import DeckBuilder

# These tests exercise the JUDGE stage's rules in isolation via
# `stages=[Stage.JUDGE]`; under the default depends_on DAG, judge depends on
# mechanical+linguistic, which are excluded by that filter and would count
# as skipped dependencies (see test_pipeline.py's transitive-skip test) --
# so judge itself would be transitively skipped. Give judge no dependencies
# here so these single-stage unit tests stay isolated from whatever other
# stage rules happen to be registered globally.
_ISOLATED_JUDGE_CFG = RulebookConfig(depends_on={"judge": []})

def _ctx(root, judge):
    return EvalContext(deck=load_deck(root), config=_ISOLATED_JUDGE_CFG, judge=judge)

def test_confidence_floor_falls_back_on_dict_config(tmp_path):
    judge = FakeJudge({"s-1": [Verdict(rule="judge/unnatural-sentence",
                                       passed=False, confidence=0.3,
                                       rationale="maybe")]})
    ctx = EvalContext(deck=load_deck(DeckBuilder(tmp_path).build()),
                      config={"sentence_base": 2, "depends_on": {"judge": []}},
                      judge=judge)
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

def test_cached_judge_close_and_context_manager(tmp_path):
    import sqlite3
    import pytest

    class Inner:
        def judge(self, req):
            return [Verdict(rule=r, passed=True, confidence=1.0, rationale="")
                    for r in req.rules]

    req = JudgeRequest(note_id="n", rules=["judge/x"], prompt="p")
    with CachedJudge(Inner(), tmp_path / "c.sqlite", "m", "1") as cj:
        assert cj.judge(req)[0].passed is True
    with pytest.raises(sqlite3.ProgrammingError):
        cj.judge(req)  # connection closed on context exit


# --- batched judging: the cache must not defeat the single-submission shape ---

class _BatchSpy:
    """Inner judge exposing judge_many, recording what reached the API."""

    def __init__(self):
        self.seen: list[list[str]] = []

    def judge(self, req):
        raise AssertionError("judge_many should be preferred")

    def judge_many(self, reqs):
        self.seen.append([r.note_id for r in reqs])
        return {r.note_id: [Verdict(rule=r.rules[0], passed=True,
                                    confidence=1.0, rationale="")]
                for r in reqs}


def test_cached_judge_many_submits_once_and_caches(tmp_path):
    from thai_deck_eval.judge.core import CachedJudge, JudgeRequest
    inner = _BatchSpy()
    reqs = [JudgeRequest(note_id=f"sn-{i}", rules=["judge/unnatural-sentence"],
                         prompt=f"p{i}") for i in range(3)]
    with CachedJudge(inner, tmp_path / "c.sqlite", "m", "1") as judge:
        first = judge.judge_many(reqs)
    assert set(first) == {"sn-0", "sn-1", "sn-2"}
    assert inner.seen == [["sn-0", "sn-1", "sn-2"]]

    with CachedJudge(_BatchSpy(), tmp_path / "c.sqlite", "m", "1") as judge:
        again = judge.judge_many(reqs)          # every verdict served from cache
    assert set(again) == {"sn-0", "sn-1", "sn-2"}


def test_cached_judge_many_only_sends_uncached_cards(tmp_path):
    from thai_deck_eval.judge.core import CachedJudge, JudgeRequest
    reqs = [JudgeRequest(note_id=f"sn-{i}", rules=["judge/unnatural-sentence"],
                         prompt=f"p{i}") for i in range(3)]
    with CachedJudge(_BatchSpy(), tmp_path / "c.sqlite", "m", "1") as judge:
        judge.judge_many(reqs[:2])
    inner = _BatchSpy()
    with CachedJudge(inner, tmp_path / "c.sqlite", "m", "1") as judge:
        out = judge.judge_many(reqs)
    assert inner.seen == [["sn-2"]]
    assert set(out) == {"sn-0", "sn-1", "sn-2"}


def test_cached_judge_many_falls_back_to_per_card_backends(tmp_path):
    from thai_deck_eval.judge.core import CachedJudge, JudgeRequest

    class PerCard:
        def __init__(self): self.calls = 0
        def judge(self, req):
            self.calls += 1
            return [Verdict(rule=req.rules[0], passed=True, confidence=1.0,
                            rationale="")]

    inner = PerCard()
    reqs = [JudgeRequest(note_id=f"sn-{i}", rules=["judge/unnatural-sentence"],
                         prompt=f"p{i}") for i in range(2)]
    with CachedJudge(inner, tmp_path / "c.sqlite", "m", "1") as judge:
        out = judge.judge_many(reqs)
    assert inner.calls == 2
    assert set(out) == {"sn-0", "sn-1"}
