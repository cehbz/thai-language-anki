"""What a Fluent Forever Thai deck must be, stated as requirements.

Derived from the design spec and the doctrine decisions, not from the code.
Every test drives a public entry point -- the evaluator CLI, the compiled
.apkg -- and asserts a property of the product. If the implementation is
rewritten entirely, these should still be the tests.

Each name is the requirement it enforces.
"""
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import thai_deck_eval.cli as eval_cli
from tests.helpers import DeckBuilder


def report(deck_root, *args):
    """The evaluator's own output: the only thing a caller sees."""
    res = CliRunner().invoke(eval_cli.main,
                             [str(deck_root), "--no-judge", "--format", "json", *args])
    return json.loads(res.output)


def rules_fired(deck_root, *args):
    return {f["rule"] for f in report(deck_root, *args)["findings"]}


# --- Doctrine: no translation on picture cards ---

def test_a_picture_word_carrying_an_l1_gloss_is_reported(tmp_path):
    """The image is the meaning; a translation on the card defeats the card."""
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["gloss"] = "dog"
    assert "mech/gloss-on-picture-word" in rules_fired(b.build())


def test_a_sentence_may_carry_a_gloss(tmp_path):
    """The community correction to the book: abstract material needs one."""
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["gloss"] = "the dog comes to eat rice"
    assert "mech/gloss-on-picture-word" not in rules_fired(b.build())


# --- Doctrine: minimal pairs teach one contrast, in a native voice ---

def test_a_minimal_pair_in_a_synthetic_voice_is_an_error(tmp_path):
    """Tone-bearing cards must be a human voice: TTS teaches TTS."""
    b = DeckBuilder(tmp_path)
    b.data["minimal_pairs"][0]["members"][0]["audio"]["source"] = "tts"
    rep = report(b.build())
    assert "meth/tts-audio" in {f["rule"] for f in rep["findings"]}
    assert rep["gate"] == "fail"


def test_a_pair_differing_in_more_than_the_declared_contrast_is_reported(tmp_path):
    """Two differences teach neither."""
    b = DeckBuilder(tmp_path)
    pair = b.data["minimal_pairs"][0]
    pair["members"][1]["thai"] = "ผัด"
    pair["members"][1]["ipa"] = "pʰat˨˩"
    assert any(r.startswith("lang/") or r.startswith("meth/")
               for r in rules_fired(b.build()))


# --- Doctrine: the deck is a source directory, not an .apkg ---

def test_a_deck_missing_a_media_file_it_references_is_an_error(tmp_path):
    b = DeckBuilder(tmp_path)
    root = b.build()
    next((root / "media").rglob("*.mp3")).unlink()
    rep = report(root)
    assert "mech/media-missing" in {f["rule"] for f in rep["findings"]}
    assert rep["gate"] == "fail"


def test_a_schema_violation_stops_evaluation_rather_than_scoring_it(tmp_path):
    b = DeckBuilder(tmp_path)
    root = b.build()
    (root / "notes" / "picture_words.yaml").write_text("- not: a valid note\n")
    rep = report(root)
    assert rep["gate"] == "fail"
    assert {f["rule"] for f in rep["findings"]} == {"schema/invalid"}


# --- Doctrine: staging, sound system first ---

def test_the_report_measures_contrast_coverage(tmp_path):
    """Staging is the method's spine: the deck must know what it has yet to
    teach."""
    metrics = {m["name"] for m in report(DeckBuilder(tmp_path).build())["metrics"]}
    assert {"coverage/minimal_pairs", "coverage/spelling",
            "coverage/categories", "coverage/frequency"} <= metrics


def test_speaker_diversity_is_measured_for_pairs(tmp_path):
    """One voice teaches one voice's vowels."""
    metrics = {m["name"] for m in report(DeckBuilder(tmp_path).build())["metrics"]}
    assert "speakers/minimal_pairs" in metrics


# --- Doctrine: a sentence introduces one new thing ---

def test_a_sentence_using_a_word_not_yet_introduced_is_reported(tmp_path):
    b = DeckBuilder(tmp_path)
    for w in b.data["picture_words"]:
        w["frequency_rank"] = {"มา": 1, "หมา": 2, "ข้าว": 900}[w["thai"]]
    root = b.build()
    rulebook = tmp_path / "rb.yaml"
    rulebook.write_text(yaml.safe_dump({"sentence_base": 2}))
    assert "meth/new-elements" in rules_fired(root, "--stages", "method",
                                              "--rulebook", str(rulebook))


# --- Output contract ---

def test_the_gate_fails_on_any_error_and_passes_otherwise(tmp_path):
    clean = DeckBuilder(tmp_path / "clean").build()
    assert report(clean)["gate"] == "pass"

    b = DeckBuilder(tmp_path / "broken")
    b.data["minimal_pairs"][0]["members"][0]["audio"]["source"] = "tts"
    assert report(b.build())["gate"] == "fail"


def test_the_report_scores_four_dimensions(tmp_path):
    scores = report(DeckBuilder(tmp_path).build())["scores"]
    assert set(scores) == {"integrity", "language", "method", "content"}
    assert all(0 <= v <= 100 for v in scores.values())


def test_text_and_json_describe_the_same_run(tmp_path):
    root = DeckBuilder(tmp_path).build()
    as_json = report(root)
    text = CliRunner().invoke(eval_cli.main,
                              [str(root), "--no-judge", "--format", "text"]).output
    assert as_json["gate"] in text.lower()
