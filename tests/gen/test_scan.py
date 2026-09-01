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

def test_pending_images_includes_flagged_even_when_file_exists(tmp_path):
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    img_path = deck.root / "media" / "images" / "pw-0.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"x")
    assert pending_images(deck) == []
    needs = pending_images(deck, flagged={"pw-0"})
    assert [n.note_id for n in needs] == ["pw-0"]


def test_pending_images_can_include_the_pictures_the_deck_already_has(tmp_path):
    """A verifying run judges what is there, so it must be handed every note
    that should have a picture -- not only the ones a report flagged."""
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    img_path = deck.root / "media" / "images" / "pw-0.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    img_path.write_bytes(b"x")
    assert pending_images(deck) == []
    needs = pending_images(deck, include_present=True)
    assert [n.note_id for n in needs] == ["pw-0"]


def test_pending_images_takes_glosses_from_lookup_not_note(tmp_path):
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    assert pending_images(deck)[0].gloss is None
    needs = pending_images(deck, glosses={"w0": "word zero"})
    assert needs[0].gloss == "word zero"
