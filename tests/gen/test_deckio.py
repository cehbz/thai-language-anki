from pathlib import Path
from thai_deck_eval.model.deck import load_deck
from thai_deck_eval.model.notes import Audio, PictureWordNote
from thai_deck_gen.deckio import new_deck, write_deck

def _pw(i):
    return PictureWordNote(
        id=f"pw-{i}", thai="น้ำ", image=f"images/pw-{i}.jpg",
        audio=Audio(file=f"audio/picture_words/pw-{i}.mp3",
                    source="native", speaker="pending"),
        frequency_rank=i, category="Beverages", part_of_speech="noun",
        classifier="แก้ว")

def test_write_deck_round_trips(tmp_path):
    deck = new_deck(tmp_path / "d", "test", ["sounds", "words"])
    deck.picture_words.append(_pw(1))
    write_deck(deck)
    loaded = load_deck(tmp_path / "d")
    assert loaded.meta.name == "test"
    assert loaded.picture_words == [_pw(1)]

def test_write_deck_is_stable(tmp_path):
    deck = new_deck(tmp_path / "d", "test", ["sounds"])
    deck.picture_words.append(_pw(1))
    write_deck(deck)
    first = (tmp_path / "d" / "notes" / "picture_words.yaml").read_bytes()
    write_deck(deck)
    assert (tmp_path / "d" / "notes" / "picture_words.yaml").read_bytes() == first

def test_write_deck_creates_all_note_files(tmp_path):
    write_deck(new_deck(tmp_path / "d", "test", ["sounds"]))
    for fam in ["minimal_pairs", "spelling_sound", "picture_words", "sentences"]:
        assert (tmp_path / "d" / "notes" / f"{fam}.yaml").exists()
