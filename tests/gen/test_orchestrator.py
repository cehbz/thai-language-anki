from pathlib import Path

import pytest

from thai_deck_gen import orchestrator
from thai_deck_gen.config import GenConfig
from thai_deck_gen.context import GenContext
from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.orchestrator import EvalError, generate
from thai_deck_gen.producers import ProducerResult
from tests.gen.test_pairs import _gaps
from tests.gen.test_sentences import _deck_with_words
from tests.gen.test_tts import _deck_with_sentence_and_pair

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
    assert len(writes) == 4                    # written after content and after media, each iteration
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
    assert len(writes) == 2                    # after content, after media


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


def test_dispatch_media_filters_forvo_to_native_tier_families(tmp_path, monkeypatch):
    captured = {}

    def fake_fetch_forvo(needs, deck, manifest, client, today, **kw):
        captured["needs"] = needs
        return ProducerResult()

    monkeypatch.setattr(orchestrator, "fetch_forvo", fake_fetch_forvo)
    deck = _deck_with_sentence_and_pair(tmp_path)
    write_deck(deck)
    ctx = _ctx(tmp_path)
    ctx.forvo_api_key = "KEY"

    orchestrator._dispatch_media(_gaps([]), deck, ctx)

    assert captured["needs"]
    assert all(n.family != "sentence" for n in captured["needs"])
    assert any(n.family == "minimal_pair" for n in captured["needs"])


def test_blocked_summary_mentions_leftover_native_tier_needs(tmp_path):
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    ctx = _ctx(tmp_path)

    msg = orchestrator._blocked_summary(deck, ctx)

    assert "audio commission" in msg


def test_generate_writes_content_before_media_dispatch(tmp_path, monkeypatch):
    calls, writes = [], []
    _stub_producers(monkeypatch, calls)
    _stub_write(monkeypatch, writes)
    def boom(gaps, deck, ctx):
        raise RuntimeError("media filler crashed")
    monkeypatch.setattr(orchestrator, "_dispatch_media", boom)
    deck = new_deck(tmp_path / "d", "t", ["sounds", "words", "sentences"])
    ctx = _ctx(tmp_path)
    report = _report("fail", missing_contrasts=["vowel_length"])
    with pytest.raises(RuntimeError):
        generate(deck, ctx, evaluate=lambda deck_dir: report)
    assert len(writes) == 1                    # content producers' work persisted


def test_dispatch_media_passes_word_list_glosses_to_image_scan(tmp_path, monkeypatch):
    from thai_deck_gen.wordlist import WordEntry
    seen = {}
    def fake_pending_images(deck, flagged=None, glosses=None):
        seen["glosses"] = glosses
        return []
    monkeypatch.setattr(orchestrator, "pending_images", fake_pending_images)
    monkeypatch.setattr(orchestrator, "fill_images",
                        lambda needs, gaps, deck, manifest, ctx, today: ProducerResult())
    deck = _deck_with_sentence_and_pair(tmp_path)
    write_deck(deck)
    ctx = _ctx(tmp_path)
    ctx.http_get = lambda *a, **k: None
    ctx.word_list = [WordEntry(thai="น้ำ", gloss="water", category="Beverages",
                               part_of_speech="noun", classifier="แก้ว")]
    orchestrator._dispatch_media(_gaps([]), deck, ctx)
    assert seen["glosses"] == {"น้ำ": "water"}


def test_generate_logs_blocked_reasons(tmp_path, monkeypatch, capsys):
    calls, writes = [], []
    _stub_producers(monkeypatch, calls)
    _stub_write(monkeypatch, writes)
    monkeypatch.setattr(orchestrator, "fill_sentences", lambda gaps, deck, ctx: ProducerResult(
        blocked=["ก: 2 unknown non-target tokens", "ข: unranked",
                 "llm unavailable, sentence generation halted: usage limit"]))
    deck = new_deck(tmp_path / "d", "t", ["sounds", "words", "sentences"])
    ctx = _ctx(tmp_path)
    reports = iter([_report("fail", missing_contrasts=["vowel_length"]), _report("pass")])
    generate(deck, ctx, evaluate=lambda deck_dir: next(reports))
    out = capsys.readouterr().out
    assert "ก: 2 unknown non-target tokens" in out
    assert "halted" in out


def test_dispatch_media_keeps_provenance_for_media_written_before_a_crash(tmp_path, monkeypatch):
    """The killer case: the process dies INSIDE the long image filler, after
    it has already written files to disk."""
    from thai_deck_gen.media.manifest import Manifest, MediaEntry

    def fake_tts(needs, deck, manifest, tts, today):
        manifest.record(MediaEntry(file="media/audio/sentences/a.mp3", channel="tts",
                                   origin="google", fetched=today))
        return ProducerResult(changed=1)

    def fake_images(needs, gaps, deck, manifest, ctx, today):
        manifest.record(MediaEntry(file="media/images/pw-0.jpg", channel="openverse",
                                   origin="http://x", fetched=today))
        raise RuntimeError("killed mid-images, after one image was written")

    monkeypatch.setattr(orchestrator, "fill_tts", fake_tts)
    monkeypatch.setattr(orchestrator, "fill_images", fake_images)
    deck = _deck_with_sentence_and_pair(tmp_path)
    write_deck(deck)
    ctx = _ctx(tmp_path)
    ctx.tts_api_key = "KEY"
    ctx.http_get = lambda *a, **k: None

    with pytest.raises(RuntimeError):
        orchestrator._dispatch_media(_gaps([]), deck, ctx)

    on_disk = Manifest.load(deck.root)
    assert on_disk.channel_of("media/audio/sentences/a.mp3") == "tts"
    assert on_disk.channel_of("media/images/pw-0.jpg") == "openverse"
