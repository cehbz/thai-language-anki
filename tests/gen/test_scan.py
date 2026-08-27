from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.media.scan import pending_audio, pending_images
from tests.gen.test_sentences import _deck_with_words

def test_pending_audio_flags_missing_and_pending(tmp_path):
    deck = _deck_with_words(tmp_path, 2)
    write_deck(deck)
    needs = pending_audio(deck)
    assert {n.note_id for n in needs} == {"pw-0", "pw-1"}
    assert needs[0].native_required is True
    # materialize one file AND clear its pending speaker
    (deck.root / "media" / "audio" / "picture_words").mkdir(parents=True)
    (deck.root / "media" / "audio" / "picture_words" / "pw-0.mp3").write_bytes(b"x")
    deck.picture_words[0].audio.speaker = "thai1000:main"
    assert {n.note_id for n in pending_audio(deck)} == {"pw-1"}

def test_minimal_pair_needs_are_native_required(tmp_path):
    from thai_deck_eval.model.notes import Audio, MinimalPairNote, PairMember
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    deck.minimal_pairs.append(MinimalPairNote(
        id="mp-x-1", contrast="tone", members=[
            PairMember(thai="คา", ipa="kʰaː˧",
                       audio=Audio(file="audio/minimal_pairs/mp-x-1_0.mp3",
                                   source="native", speaker="pending")),
            PairMember(thai="ค่า", ipa="kʰaː˥˩",
                       audio=Audio(file="audio/minimal_pairs/mp-x-1_1.mp3",
                                   source="native", speaker="pending"))]))
    needs = pending_audio(deck)
    assert all(n.native_required for n in needs)
    assert needs[0].member_index == 0

def test_pending_images(tmp_path):
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    needs = pending_images(deck)
    assert needs[0].term == "w0"
