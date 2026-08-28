from pathlib import Path
from thai_deck_gen.deckio import new_deck
from thai_deck_gen.producers.spelling import fill_spelling, missing_patterns
from thai_deck_gen.wordlist import WordEntry
from tests.gen.test_pairs import _gaps

DATA = Path(__file__).parents[2] / "data"

def _word(thai):
    return WordEntry(thai=thai, gloss="x", category="Food",
                     part_of_speech="other")

def test_missing_patterns_lists_uncovered(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    missing = missing_patterns(deck, DATA / "spelling_targets.yaml")
    assert "ก" in missing

def test_fill_spelling_picks_example_word(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    class Ctx:
        word_list = [_word("ไก่")]              # contains ก
        targets_path = DATA / "spelling_targets.yaml"
    res = fill_spelling(_gaps([]), deck, Ctx())
    note = next(n for n in deck.spelling_sound if n.pattern == "ก")
    assert note.example_word == "ไก่"
    assert note.pattern_kind == "consonant"
    assert note.consonant_class == "mid"
    assert "ก" not in res.blocked

def test_fill_spelling_vowel_pattern_with_placeholder(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    class Ctx:
        word_list = [_word("เกา")]  # เ + ก + า; matches pattern เ-
        targets_path = DATA / "spelling_targets.yaml"
    res = fill_spelling(_gaps([]), deck, Ctx())
    note = next(n for n in deck.spelling_sound if n.pattern == "เ-")
    assert note.example_word == "เกา"
    assert note.pattern_kind == "vowel"
    assert note.consonant_class is None
    assert "เ-" not in res.blocked
