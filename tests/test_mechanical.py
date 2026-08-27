import pytest
import thai_deck_eval.stages.mechanical  # noqa: F401  (registers rules)
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.model.deck import load_deck
from tests.helpers import DeckBuilder

def _run(root):
    return run_pipeline(EvalContext(deck=load_deck(root)),
                        stages=[Stage.MECHANICAL])

def _rules(res):
    return sorted(f.rule for f in res.findings)

def test_golden_is_clean(tmp_path):
    assert _rules(_run(DeckBuilder(tmp_path).build())) == []

def test_media_missing(tmp_path):
    root = DeckBuilder(tmp_path).build()
    (root / "media" / "images" / "dog.png").unlink()
    res = _run(root)
    assert "mech/media-missing" in _rules(res)
    f = next(f for f in res.findings if f.rule == "mech/media-missing")
    assert f.note_id == "w-dog" and f.severity == Severity.ERROR

def test_media_orphan(tmp_path):
    root = DeckBuilder(tmp_path).build()
    (root / "media" / "audio" / "unused.mp3").write_bytes(b"x")
    assert "mech/media-orphan" in _rules(_run(root))

def test_latin_in_thai_field(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["thai"] = "maa หมา"
    assert "mech/latin-in-thai" in _rules(_run(b.build()))

def test_duplicate_picture_word(tmp_path):
    b = DeckBuilder(tmp_path)
    dup = dict(b.data["picture_words"][0]); dup["id"] = "w-dog2"
    b.data["picture_words"].append(dup)
    assert "mech/duplicate-note" in _rules(_run(b.build()))

def test_target_not_in_sentence(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["target"] = "วิ่ง"
    assert "mech/target-not-in-sentence" in _rules(_run(b.build()))

def test_gloss_on_picture_word(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["gloss"] = "dog"
    assert "mech/gloss-on-picture-word" in _rules(_run(b.build()))
