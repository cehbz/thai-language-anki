from pathlib import Path

import pytest

from thai_deck_gen import orchestrator
from thai_deck_gen.config import GenConfig
from thai_deck_gen.context import GenContext
from thai_deck_gen.deckio import new_deck
from thai_deck_gen.orchestrator import EvalError, generate
from thai_deck_gen.producers import ProducerResult

DATA = Path(__file__).parents[2] / "data"


def _ctx(tmp_path, max_iterations=5):
    return GenContext(
        g2p=None, tokenizer=None, freq=None, llm=None,
        word_list=[], lexicon_words=[], exceptions={}, pair_seeds={},
        grammar_points=[], exemplars=[],
        config=GenConfig(max_iterations=max_iterations),
        data_dir=DATA,
        adjudication_queue=tmp_path / "work" / "ipa_adjudication.yaml",
        targets_path=DATA / "spelling_targets.yaml",
        thai1000_apkg=None, forvo_api_key=None, tts_api_key=None, http_get=None,
    )


def _report(gate, missing_contrasts=(), missing_categories=(), findings=()):
    return {
        "gate": gate,
        "findings": list(findings),
        "metrics": [
            {"name": "coverage/minimal_pairs", "value": 0.0,
             "detail": {"missing": list(missing_contrasts), "by_note": {}}},
            {"name": "coverage/categories", "value": 0.0,
             "detail": {"missing": list(missing_categories)}},
            {"name": "coverage/frequency", "value": 0.0, "detail": {}},
            {"name": "speakers/minimal_pairs", "value": 0.0, "detail": {}},
        ],
    }


def _stub_producers(monkeypatch, calls):
    def make(name):
        def fn(gaps, deck, ctx):
            calls.append(name)
            return ProducerResult(added=1)
        return fn
    for name in ("fill_pairs", "fill_spelling", "fill_words", "fill_sentences"):
        monkeypatch.setattr(orchestrator, name, make(name))


def _stub_write(monkeypatch, writes):
    monkeypatch.setattr(orchestrator, "write_deck", lambda deck: writes.append(deck))


def test_generate_dispatch_order_and_stops_on_clean_report(tmp_path, monkeypatch):
    calls, writes = [], []
    _stub_producers(monkeypatch, calls)
    _stub_write(monkeypatch, writes)
    deck = new_deck(tmp_path / "d", "t", ["sounds", "words", "sentences"])
    ctx = _ctx(tmp_path)

    reports = iter([
        _report("fail", missing_contrasts=["tone:mid-high", "vowel_length"]),
        _report("fail", missing_contrasts=["vowel_length"]),
        _report("pass"),
    ])

    summaries = generate(deck, ctx, evaluate=lambda deck_dir: next(reports))

    assert calls == ["fill_pairs", "fill_spelling", "fill_words", "fill_sentences"] * 2
    assert len(writes) == 2                    # deck written each dispatched iteration
    assert len(summaries) == 2
    assert all(set(s.results) == {"pairs", "spelling", "words", "sentences"}
              for s in summaries)
    assert summaries[0].gaps_fingerprint != summaries[1].gaps_fingerprint


def test_generate_stops_on_no_progress(tmp_path, monkeypatch):
    calls, writes = [], []
    _stub_producers(monkeypatch, calls)
    _stub_write(monkeypatch, writes)
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    ctx = _ctx(tmp_path)

    same_report = _report("fail", missing_contrasts=["tone:mid-high"])
    n_evals = {"n": 0}

    def evaluate(deck_dir):
        n_evals["n"] += 1
        return same_report

    summaries = generate(deck, ctx, evaluate=evaluate)

    assert n_evals["n"] == 2                   # detects repeat on the 2nd eval
    assert len(summaries) == 1
    assert len(writes) == 1


def test_generate_stops_at_max_iterations(tmp_path, monkeypatch):
    calls, writes = [], []
    _stub_producers(monkeypatch, calls)
    _stub_write(monkeypatch, writes)
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    ctx = _ctx(tmp_path, max_iterations=2)

    n_evals = {"n": 0}

    def evaluate(deck_dir):
        # gaps shrink every call so fingerprints never repeat
        n_evals["n"] += 1
        return _report("fail", missing_contrasts=["tone:mid-high"] * (10 - n_evals["n"]))

    summaries = generate(deck, ctx, evaluate=evaluate)

    assert n_evals["n"] == 2
    assert len(summaries) == 2


def test_generate_propagates_eval_error(tmp_path, monkeypatch):
    _stub_producers(monkeypatch, [])
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    ctx = _ctx(tmp_path)

    def evaluate(deck_dir):
        raise EvalError("boom")

    with pytest.raises(EvalError):
        generate(deck, ctx, evaluate=evaluate)
