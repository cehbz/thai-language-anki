import pytest
from pydantic import ValidationError
from thai_deck_eval.model.notes import (
    Audio, MinimalPairNote, PairMember, PictureWordNote, SentenceNote,
)

AUD = {"file": "audio/a.mp3", "source": "native", "speaker": "s1"}

def test_minimal_pair_requires_two_members():
    m = {"thai": "ขาว", "ipa": "kʰaːw˨˩˦", "audio": AUD}
    with pytest.raises(ValidationError):
        MinimalPairNote(id="mp1", contrast="tone", members=[m])
    note = MinimalPairNote(id="mp1", contrast="tone", members=[m, m])
    assert note.contrast == "tone"

def test_audio_source_restricted():
    with pytest.raises(ValidationError):
        Audio(file="a.mp3", source="robot", speaker="s1")

def test_picture_word_defaults():
    w = PictureWordNote(id="w1", thai="หมา", image="images/dog.png",
                        audio=AUD, frequency_rank=120, category="Animals")
    assert w.test_spelling is False and w.classifier is None

def test_sentence_kind_restricted():
    with pytest.raises(ValidationError):
        SentenceNote(id="s1", kind="poem", thai="…", target="…", audio=AUD)

def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        PictureWordNote(id="w1", thai="หมา", image="i.png", audio=AUD,
                        frequency_rank=1, category="Animals", bogus=1)
