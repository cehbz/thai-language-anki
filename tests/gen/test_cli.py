import os

import pytest

from thai_deck_eval.model.deck import load_deck
from thai_deck_gen.cli import build_parser, main


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
    (["images", "--deck", "d"], "images"),
    (["compile", "--deck", "d", "--out", "o.apkg"], "compile"),
    (["wordlist"], "wordlist"),
])
def test_parser_accepts_thin_subcommands(argv, expected_command):
    args = build_parser().parse_args(argv)
    assert args.command == expected_command


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
