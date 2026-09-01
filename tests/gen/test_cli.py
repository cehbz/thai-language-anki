import os

import pytest
import yaml

from thai_deck_eval.model.deck import load_deck
from thai_deck_gen.cli import build_parser, main
from thai_deck_gen.deckio import write_deck
from tests.gen.test_sentences import _deck_with_words


def test_init_creates_loadable_deck(tmp_path):
    deck_dir = tmp_path / "d"
    rc = main(["init", str(deck_dir), "--name", "x", "--phases", "sounds,words"])
    assert rc == 0
    deck = load_deck(deck_dir)
    assert deck.meta.name == "x"
    assert deck.meta.stage_plan.phases == ["sounds", "words"]
    assert deck.picture_words == []


def test_parser_accepts_generate_args():
    args = build_parser().parse_args(["generate", "some/dir", "--max-iterations", "1"])
    assert args.command == "generate"
    assert args.max_iterations == 1


def test_parser_accepts_init_args():
    args = build_parser().parse_args(
        ["init", "some/dir", "--name", "x", "--phases", "sounds,words,sentences"])
    assert args.command == "init"
    assert args.name == "x"
    assert args.phases == "sounds,words,sentences"


@pytest.mark.parametrize("argv,expected_command", [
    (["pairs", "--deck", "d"], "pairs"),
    (["spelling", "--deck", "d"], "spelling"),
    (["words", "--deck", "d"], "words"),
    (["sentences", "fetch-exemplars", "--deck", "d"], "sentences"),
    (["sentences", "fill", "--deck", "d"], "sentences"),
    (["audio", "fetch-forvo", "--deck", "d"], "audio"),
    (["audio", "import-thai1000", "--deck", "d", "--apkg", "a.apkg"], "audio"),
    (["audio", "import-commission", "--deck", "d", "--recordings", "r",
     "--batch", "b.yaml", "--speaker", "joe"], "audio"),
    (["audio", "tts", "--deck", "d"], "audio"),
    (["audio", "commission", "--deck", "d"], "audio"),
    (["images", "--deck", "d"], "images"),
    (["compile", "--deck", "d", "--out", "o.apkg"], "compile"),
    (["wordlist"], "wordlist"),
])
def test_parser_accepts_thin_subcommands(argv, expected_command):
    args = build_parser().parse_args(argv)
    assert args.command == expected_command


def test_audio_commission_writes_batch(tmp_path):
    deck = _deck_with_words(tmp_path / "d", 2)
    write_deck(deck)
    rc = main(["audio", "commission", "--deck", str(deck.root)])
    assert rc == 0
    batch_path = deck.root / "work" / "commission_batch_001.yaml"
    assert batch_path.exists()
    batch = yaml.safe_load(batch_path.read_text())
    assert {item["note_id"] for item in batch["items"]} == {"pw-0", "pw-1"}


def test_audio_commission_no_needs_writes_nothing(tmp_path, capsys):
    deck_dir = tmp_path / "d"
    main(["init", str(deck_dir), "--name", "x", "--phases", "sounds"])
    rc = main(["audio", "commission", "--deck", str(deck_dir)])
    assert rc == 0
    assert not (deck_dir / "work").exists() or not list(
        (deck_dir / "work").glob("commission_batch_*.yaml"))


@pytest.mark.integration
def test_generate_cli_runs_one_iteration(tmp_path, monkeypatch):
    """Exercises the real `generate` CLI path end to end (real evaluator
    subprocess, which imports pythainlp) with a faked LLM/NLP context via
    THAI_DECK_GEN_FAKE, so an empty deck can be generated against without a
    live claude CLI or word list."""
    monkeypatch.setenv("THAI_DECK_GEN_FAKE", "1")
    deck_dir = tmp_path / "d"
    assert main(["init", str(deck_dir), "--name", "x", "--phases", "sounds,words,sentences"]) == 0
    rc = main(["generate", str(deck_dir), "--max-iterations", "1"])
    assert rc == 0
    assert (deck_dir / ".last-report.json").exists()


def test_parser_accepts_wordlist_extend():
    from thai_deck_gen.cli import build_parser
    args = build_parser().parse_args(["wordlist", "--extend"])
    assert args.command == "wordlist" and args.extend is True


def test_approve_records_a_waiver_against_the_current_image(tmp_path, monkeypatch):
    """The approval names the image it approved, so a later re-fetch cannot
    inherit it."""
    import hashlib
    from thai_deck_eval.waivers import load_waivers
    from thai_deck_gen.cli import main
    from tests.gen.test_images import _deck_with_pw

    deck = _deck_with_pw(tmp_path)
    img = deck.root / "media" / "images" / "pw-0.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"jpegbytes")

    rc = main(["approve", "--deck", str(deck.root), "--note", "pw-0",
               "--rule", "judge/image-embedded-text",
               "--reason", "the expiry date is the point of the card"])
    assert rc == 0

    waivers = load_waivers(deck.root)
    assert len(waivers) == 1
    assert waivers[0].note_id == "pw-0"
    assert waivers[0].rule == "judge/image-embedded-text"
    assert waivers[0].sha == hashlib.sha256(b"jpegbytes").hexdigest()
    assert waivers[0].reason.startswith("the expiry date")


def test_approve_replaces_an_earlier_waiver_for_the_same_rule(tmp_path):
    from thai_deck_eval.waivers import load_waivers
    from thai_deck_gen.cli import main
    from tests.gen.test_images import _deck_with_pw

    deck = _deck_with_pw(tmp_path)
    img = deck.root / "media" / "images" / "pw-0.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"first")
    main(["approve", "--deck", str(deck.root), "--note", "pw-0",
          "--rule", "judge/image-embedded-text", "--reason", "first pass"])
    img.write_bytes(b"second")
    main(["approve", "--deck", str(deck.root), "--note", "pw-0",
          "--rule", "judge/image-embedded-text", "--reason", "re-reviewed"])

    waivers = load_waivers(deck.root)
    assert len(waivers) == 1
    assert waivers[0].reason == "re-reviewed"


def test_images_command_passes_the_search_phrases(tmp_path, monkeypatch):
    """The CLI is where the phrase reaches the search and the judge; a unit
    test on fill_images cannot see this wiring, and it was missing."""
    import thai_deck_gen.cli as cli
    from thai_deck_gen.wordlist import WordEntry
    from tests.gen.test_images import _deck_with_pw

    deck = _deck_with_pw(tmp_path, term="ส้ม", gloss="orange")
    captured = {}

    def fake_pending_images(deck_, flagged=None, glosses=None, image_queries=None,
                            include_present=False):
        captured["image_queries"] = image_queries
        return []

    class Ctx:
        word_list = [WordEntry(id="orange", thai="ส้ม", gloss="orange", category="Food",
                               part_of_speech="noun", classifier="ลูก",
                               image_query="orange fruit on a market stall")]
        http_get = None
        imagegen = None
        config = cli.GenConfig()

    monkeypatch.setattr(cli, "pending_images", fake_pending_images)
    monkeypatch.setattr(cli, "_build_ctx", lambda *a, **k: Ctx())
    monkeypatch.setattr(cli, "imagegen_for", lambda ctx: None)
    monkeypatch.setattr(cli, "image_judge_for", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_gaps_for", lambda *a, **k: __import__(
        "tests.gen.test_pairs", fromlist=["_gaps"])._gaps([]))
    monkeypatch.setattr(cli, "fill_images",
                        lambda *a, **k: __import__(
                            "thai_deck_gen.producers", fromlist=["ProducerResult"]
                        ).ProducerResult())

    cli.main(["images", "--deck", str(deck.root)])
    assert captured["image_queries"] == {"ส้ม": "orange fruit on a market stall"}
