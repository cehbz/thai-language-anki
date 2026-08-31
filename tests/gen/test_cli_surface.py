"""Every command and every parameter of the generator CLI.

The CLI is where wiring lives, and wiring is what unit tests cannot see: a
parameter that never reaches its destination leaves every function below it
passing. Each test here fails if a specific argument stops being forwarded.
"""
from pathlib import Path

import pytest
import yaml

import thai_deck_gen.cli as cli
from thai_deck_gen.producers import ProducerResult
from tests.gen.test_images import _deck_with_pw


@pytest.fixture
def deck(tmp_path):
    return _deck_with_pw(tmp_path, term="ส้ม", gloss="orange")


@pytest.fixture
def calls(monkeypatch):
    """Record what the CLI hands to each collaborator."""
    seen: dict[str, dict] = {}

    def spy(name, result=None):
        def fn(*args, **kwargs):
            seen[name] = {"args": args, "kwargs": kwargs}
            return result if result is not None else ProducerResult()
        return fn

    monkeypatch.setattr(cli, "write_deck", spy("write_deck"))
    monkeypatch.setattr(cli, "_report_result", lambda *a, **k: None)
    return seen, spy


def _ctx(**kw):
    from thai_deck_gen.config import GenConfig
    from thai_deck_gen.wordlist import WordEntry

    class Ctx:
        word_list = [WordEntry(thai="ส้ม", gloss="orange", category="Food",
                               part_of_speech="noun", classifier="ลูก",
                               image_query="orange fruit on a stall")]
        config = GenConfig(**kw)
        http_get = None
        imagegen = None
        pexels_key = None
        image_query_hints = {}
        image_candidates = 5
    return Ctx()


# --- words / pairs / spelling: the gap-driven producers ---

@pytest.mark.parametrize("command,target", [
    ("words", "fill_words"), ("pairs", "fill_pairs"), ("spelling", "fill_spelling"),
])
def test_producer_commands_dispatch_and_write(deck, calls, monkeypatch, command, target):
    seen, spy = calls
    monkeypatch.setattr(cli, target, spy(target))
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: _ctx())
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: "GAPS")
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)

    assert cli.main([command, "--deck", str(deck.root)]) in (0, None)
    assert target in seen, f"{command} did not call {target}"
    assert seen[target]["args"][0] == "GAPS"       # gaps reach the producer
    assert "write_deck" in seen, f"{command} did not persist the deck"


def test_producer_commands_honour_data_dir(deck, calls, monkeypatch, tmp_path):
    seen, spy = calls
    captured = {}
    monkeypatch.setattr(cli, "fill_words", spy("fill_words"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_gaps_for", lambda d, data_dir: captured.setdefault(
        "data_dir", data_dir) or "GAPS")
    monkeypatch.setattr(cli, "_build_ctx", lambda d, data_dir: captured.setdefault(
        "ctx_data_dir", data_dir) or _ctx())

    cli.main(["words", "--deck", str(deck.root), "--data-dir", str(tmp_path / "dd")])
    assert captured["data_dir"] == tmp_path / "dd"
    assert captured["ctx_data_dir"] == tmp_path / "dd"


# --- images ---

def test_images_forwards_glosses_phrases_judge_and_limit(deck, calls, monkeypatch):
    seen, spy = calls
    captured = {}

    def fake_pending(deck_, flagged=None, glosses=None, image_queries=None):
        captured.update(flagged=flagged, glosses=glosses, image_queries=image_queries)
        return []

    monkeypatch.setattr(cli, "pending_images", fake_pending)
    monkeypatch.setattr(cli, "fill_images", spy("fill_images"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: _ctx())
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: "GAPS")
    monkeypatch.setattr(cli, "flagged_image_note_ids", lambda gaps: {"pw-0"})
    monkeypatch.setattr(cli, "imagegen_for", lambda ctx: "IMAGEGEN")
    monkeypatch.setattr(cli, "image_judge_for", lambda *a, **k: "JUDGE")

    cli.main(["images", "--deck", str(deck.root), "--limit", "7"])

    assert captured["glosses"] == {"ส้ม": "orange"}
    assert captured["image_queries"] == {"ส้ม": "orange fruit on a stall"}
    assert captured["flagged"] == {"pw-0"}
    assert seen["fill_images"]["kwargs"]["judge"] == "JUDGE"
    assert seen["fill_images"]["kwargs"]["limit"] == 7


def test_images_no_verify_disables_the_judge(deck, calls, monkeypatch):
    seen, spy = calls
    monkeypatch.setattr(cli, "pending_images", lambda *a, **k: [])
    monkeypatch.setattr(cli, "fill_images", spy("fill_images"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: _ctx())
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: "GAPS")
    monkeypatch.setattr(cli, "flagged_image_note_ids", lambda gaps: set())
    monkeypatch.setattr(cli, "imagegen_for", lambda ctx: None)
    monkeypatch.setattr(cli, "image_judge_for",
                        lambda *a, **k: pytest.fail("judge built despite --no-verify"))

    cli.main(["images", "--deck", str(deck.root), "--no-verify"])
    assert seen["fill_images"]["kwargs"]["judge"] is None


def test_images_stops_when_search_is_unreachable(deck, monkeypatch, capsys):
    ctx = _ctx()
    ctx.http_get = lambda *a, **k: None
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: ctx)
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: "GAPS")
    monkeypatch.setattr(cli, "flagged_image_note_ids", lambda gaps: set())
    monkeypatch.setattr(cli, "imagegen_for", lambda c: None)
    monkeypatch.setattr(cli, "search_reachable", lambda get: "proxy is down")
    monkeypatch.setattr(cli, "fill_images",
                        lambda *a, **k: pytest.fail("ran despite unreachable search"))

    assert cli.main(["images", "--deck", str(deck.root)]) == 2
    assert "proxy is down" in capsys.readouterr().err


# --- audio ---

def test_fetch_forvo_forwards_key_speakers_limit_and_memo(deck, calls, monkeypatch):
    seen, spy = calls
    monkeypatch.setattr(cli, "fetch_forvo", spy("fetch_forvo"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "pending_audio", lambda d: [])
    monkeypatch.setattr(cli, "_require_secret", lambda deck_dir, name: f"SECRET:{name}")
    monkeypatch.setattr(cli, "ForvoClient", lambda key: f"CLIENT({key})")

    cli.main(["audio", "fetch-forvo", "--deck", str(deck.root),
              "--max-speakers", "2", "--limit", "40"])

    kw = seen["fetch_forvo"]["kwargs"]
    assert seen["fetch_forvo"]["args"][3] == "CLIENT(SECRET:forvo)"
    assert kw["max_speakers"] == 2
    assert kw["limit"] == 40
    assert kw["memo"] is not None, "lookups would be re-spent without the memo"
    assert callable(kw["checkpoint"])


def test_fetch_forvo_limit_defaults_to_the_configured_request_limit(deck, calls,
                                                                   monkeypatch):
    seen, spy = calls
    monkeypatch.setattr(cli, "fetch_forvo", spy("fetch_forvo"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "pending_audio", lambda d: [])
    monkeypatch.setattr(cli, "_require_secret", lambda *a: "K")
    monkeypatch.setattr(cli, "ForvoClient", lambda key: "C")
    from thai_deck_gen.config import GenConfig
    monkeypatch.setattr(cli, "load_config", lambda p: GenConfig(forvo_request_limit=123))

    cli.main(["audio", "fetch-forvo", "--deck", str(deck.root)])
    assert seen["fetch_forvo"]["kwargs"]["limit"] == 123


def test_tts_uses_the_configured_key_and_several_voices(deck, calls, monkeypatch):
    seen, spy = calls
    monkeypatch.setattr(cli, "fill_tts", spy("fill_tts"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "pending_audio", lambda d: [])
    monkeypatch.setattr(cli, "_require_secret", lambda deck_dir, name: f"S:{name}")
    monkeypatch.setattr(cli, "GoogleTts", lambda key: f"TTS({key})")

    cli.main(["audio", "tts", "--deck", str(deck.root)])
    assert seen["fill_tts"]["args"][3] == "TTS(S:google_tts)"
    assert len(seen["fill_tts"]["kwargs"]["voices"]) > 1


def test_import_thai1000_forwards_the_apkg(deck, calls, monkeypatch, tmp_path):
    seen, spy = calls
    apkg = tmp_path / "t.apkg"
    monkeypatch.setattr(cli, "import_thai1000", spy("import_thai1000"))
    monkeypatch.setattr(cli, "audio_index", lambda p: f"INDEX({p})")
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "pending_audio", lambda d: [])

    cli.main(["audio", "import-thai1000", "--deck", str(deck.root),
              "--apkg", str(apkg)])
    assert seen["import_thai1000"]["args"][3] == f"INDEX({apkg})"


def test_import_commission_forwards_recordings_batch_and_speaker(deck, calls,
                                                                 monkeypatch, tmp_path):
    seen, spy = calls
    monkeypatch.setattr(cli, "import_commission", spy("import_commission"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)

    cli.main(["audio", "import-commission", "--deck", str(deck.root),
              "--recordings", str(tmp_path / "rec"), "--batch", str(tmp_path / "b.yaml"),
              "--speaker", "khun-somchai"])
    args = seen["import_commission"]["args"]
    assert args[0] == tmp_path / "rec"
    assert args[1] == tmp_path / "b.yaml"
    assert args[4] == "khun-somchai"


def test_commission_writes_a_batch_of_native_tier_needs(deck, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "pending_audio", lambda d: [
        type("N", (), {"family": "picture_word"})(),
        type("N", (), {"family": "sentence"})()])
    monkeypatch.setattr(cli, "write_commission_batch",
                        lambda needs, root: captured.setdefault("n", len(needs)) or
                        tmp_path / "batch.yaml")

    cli.main(["audio", "commission", "--deck", str(deck.root)])
    assert captured["n"] == 1, "sentences are TTS tier and must not be commissioned"


# --- compile ---

def test_compile_forwards_out_base_and_skip_incomplete(deck, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: type("G", (), {"pair_by_note": {}})())
    monkeypatch.setattr(cli, "compile_deck",
                        lambda *a, **k: captured.update(args=a, kwargs=k) or [])

    cli.main(["compile", "--deck", str(deck.root), "--out", str(tmp_path / "o.apkg"),
              "--base", "42", "--skip-incomplete"])
    assert captured["args"][2] == tmp_path / "o.apkg"
    assert captured["kwargs"]["base"] == 42
    assert captured["kwargs"]["skip_incomplete"] is True


def test_compile_base_defaults_to_config(deck, monkeypatch, tmp_path):
    from thai_deck_gen.config import GenConfig
    captured = {}
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: type("G", (), {"pair_by_note": {}})())
    monkeypatch.setattr(cli, "load_config", lambda p: GenConfig(sentence_base=77))
    monkeypatch.setattr(cli, "compile_deck",
                        lambda *a, **k: captured.update(kwargs=k) or [])

    cli.main(["compile", "--deck", str(deck.root), "--out", str(tmp_path / "o.apkg")])
    assert captured["kwargs"]["base"] == 77


# --- sentences ---

def test_sentences_fill_checkpoints(deck, calls, monkeypatch):
    seen, spy = calls
    monkeypatch.setattr(cli, "fill_sentences", spy("fill_sentences"))
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: _ctx())
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: "GAPS")

    cli.main(["sentences", "fill", "--deck", str(deck.root)])
    assert callable(seen["fill_sentences"]["kwargs"]["checkpoint"])


def test_fetch_exemplars_forwards_out_and_sample_size(deck, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "fetch_exemplars",
                        lambda out, sample_size: captured.update(out=out, n=sample_size) or 3)
    cli.main(["sentences", "fetch-exemplars", "--deck", str(deck.root),
              "--out", str(tmp_path / "ex.txt"), "--sample-size", "12"])
    assert captured == {"out": tmp_path / "ex.txt", "n": 12}


def test_fetch_exemplars_defaults_out_into_the_deck(deck, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "fetch_exemplars",
                        lambda out, sample_size: captured.update(out=out) or 0)
    cli.main(["sentences", "fetch-exemplars", "--deck", str(deck.root)])
    assert captured["out"] == deck.root / "work" / "exemplars.txt"


# --- approve / search-terms / init / generate ---

def test_approve_records_rule_reason_and_current_image(deck, monkeypatch):
    from thai_deck_eval.waivers import load_waivers
    img = deck.root / "media" / "images" / "pw-0.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"bytes")
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)

    cli.main(["approve", "--deck", str(deck.root), "--note", "pw-0",
              "--rule", "judge/image-embedded-text", "--reason", "the date is the point"])
    w = load_waivers(deck.root)[0]
    assert (w.note_id, w.rule, w.reason) == ("pw-0", "judge/image-embedded-text",
                                             "the date is the point")
    assert w.sha, "a waiver must name the image it approved"


def test_search_terms_apply_proposals_uses_the_deck_proposals(deck, monkeypatch,
                                                              tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "apply_query_proposals",
                        lambda wl, props: captured.update(wl=wl, props=props) or 2)
    cli.main(["search-terms", "--deck", str(deck.root), "--apply-proposals",
              "--data-dir", str(tmp_path / "dd")])
    assert captured["wl"] == tmp_path / "dd" / "word_list_th.yaml"
    assert captured["props"] == deck.root / "work" / "image_query_proposals.yaml"


def test_search_terms_drafts_into_the_word_list(deck, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "draft_image_queries",
                        lambda backend, path, warnings: captured.setdefault("path", path) or 5)
    monkeypatch.setattr(cli, "_drafting_backend", lambda d, cfg: "BACKEND")
    cli.main(["search-terms", "--deck", str(deck.root), "--data-dir", str(tmp_path / "dd")])
    assert captured["path"] == tmp_path / "dd" / "word_list_th.yaml"


def test_init_creates_a_deck_with_name_and_phases(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "new_deck",
                        lambda root, name, phases: captured.update(
                            root=root, name=name, phases=phases) or "DECK")
    monkeypatch.setattr(cli, "write_deck", lambda deck: None)
    cli.main(["init", str(tmp_path / "d"), "--name", "thai-ff",
              "--phases", "sounds,words"])
    assert captured["name"] == "thai-ff"
    assert captured["phases"] == ["sounds", "words"]


def test_generate_max_iterations_overrides_the_config(deck, monkeypatch):
    captured = {}
    ctx = _ctx(max_iterations=5)
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: ctx)
    monkeypatch.setattr(cli, "generate",
                        lambda d, c: (captured.setdefault("iters", c.config.max_iterations), [])[1])

    cli.main(["generate", str(deck.root), "--max-iterations", "9"])
    assert captured["iters"] == 9


def test_generate_without_the_flag_keeps_the_configured_iterations(deck, monkeypatch):
    captured = {}
    ctx = _ctx(max_iterations=5)
    monkeypatch.setattr(cli, "load_deck", lambda p: deck)
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: ctx)
    monkeypatch.setattr(cli, "generate",
                        lambda d, c: (captured.setdefault("iters", c.config.max_iterations), [])[1])

    cli.main(["generate", str(deck.root)])
    assert captured["iters"] == 5
