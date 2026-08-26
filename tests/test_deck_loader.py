import pytest
from thai_deck_eval.model.deck import DeckSchemaError, load_deck
from tests.helpers import DeckBuilder

def test_loads_golden(tmp_path):
    deck = load_deck(DeckBuilder(tmp_path).build())
    assert deck.meta.name == "golden"
    assert len(deck.picture_words) == 3
    assert deck.root.name == "deck"
    fams = {f for f, _ in deck.all_notes()}
    assert fams == {"minimal_pair", "spelling_sound", "picture_word", "sentence"}

def test_schema_error_reports_file_and_note(tmp_path):
    b = DeckBuilder(tmp_path)
    del b.data["picture_words"][0]["category"]
    with pytest.raises(DeckSchemaError) as e:
        load_deck(b.build())
    assert any("picture_words" in i and "w-dog" in i for i in e.value.issues)

def test_missing_notes_file_is_schema_error(tmp_path):
    root = DeckBuilder(tmp_path).build()
    (root / "notes" / "sentences.yaml").unlink()
    with pytest.raises(DeckSchemaError):
        load_deck(root)
