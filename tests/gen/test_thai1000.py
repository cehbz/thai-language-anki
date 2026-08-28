from thai_deck_gen.media.thai1000 import audio_index, import_thai1000
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import pending_audio
from thai_deck_gen.deckio import write_deck
from tests.gen.helpers_apkg import build_apkg
from tests.gen.test_scan import test_minimal_pair_needs_are_native_required  # noqa: F401
from tests.gen.test_sentences import _deck_with_words

def _apkg(tmp_path):
    path = tmp_path / "thai1000.apkg"
    build_apkg(path,
               notes=[["hi", "w0", "phon", "[sound:0.mp3]", "noun",
                       "sent", "sphon", "seng"]],
               media={"0.mp3": b"FAKE"})
    return path

def test_audio_index_maps_word_to_bytes(tmp_path):
    index = audio_index(_apkg(tmp_path))
    assert index["w0"] == b"FAKE"

def test_audio_index_strips_bracketed_classifier(tmp_path):
    path = tmp_path / "thai1000.apkg"
    build_apkg(path,
               notes=[["hi", "w0 [classifier]", "phon", "[sound:0.mp3]", "noun",
                       "sent", "sphon", "seng"]],
               media={"0.mp3": b"FAKE_BRACKETED"})
    index = audio_index(path)
    assert index["w0"] == b"FAKE_BRACKETED"

def _no_ffmpeg(monkeypatch):
    import thai_deck_gen.media.thai1000 as m
    monkeypatch.setattr(m, "normalize_audio",
                        lambda raw, dst, runner=None: dst.write_bytes(raw))

def test_import_thai1000_fills_words_not_pairs(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    index = audio_index(_apkg(tmp_path))
    manifest = Manifest.load(deck.root)
    res = import_thai1000(pending_audio(deck), deck, manifest, index, "2026-08-27")
    assert res.changed == 1
    note = deck.picture_words[0]
    assert note.audio.speaker == "thai1000:main"
    assert (deck.root / "media" / note.audio.file).exists()
    assert manifest.channel_of(f"media/{note.audio.file}") == "thai1000"

def test_import_thai1000_leaves_pairs_unblocked(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    from thai_deck_eval.model.notes import Audio, MinimalPairNote, PairMember
    from thai_deck_gen.deckio import new_deck
    deck = new_deck(tmp_path / "pd", "t", ["sounds"])
    deck.minimal_pairs.append(MinimalPairNote(
        id="mp-x-1", contrast="tone", members=[
            PairMember(thai="w0", ipa="kʰaː˧",
                       audio=Audio(file="audio/minimal_pairs/mp-x-1_0.mp3",
                                   source="native", speaker="pending")),
            PairMember(thai="ค่า", ipa="kʰaː˥˩",
                       audio=Audio(file="audio/minimal_pairs/mp-x-1_1.mp3",
                                   source="native", speaker="pending"))]))
    write_deck(deck)
    index = audio_index(_apkg(tmp_path))
    manifest = Manifest.load(deck.root)
    needs = pending_audio(deck)
    res = import_thai1000(needs, deck, manifest, index, "2026-08-27")
    assert res.changed == 0
    assert res.blocked == []
    assert deck.minimal_pairs[0].members[0].audio.speaker == "pending"
