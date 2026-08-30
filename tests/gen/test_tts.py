import base64
from thai_deck_gen.media.tts import GoogleTts, fill_tts
from thai_deck_gen.media.ffmpeg import AudioError
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import AudioNeed
from thai_deck_gen.deckio import new_deck
from thai_deck_gen.deckio import write_deck
from thai_deck_eval.model.notes import Audio, SentenceNote, MinimalPairNote, PairMember
import pytest

def _no_ffmpeg(monkeypatch):
    import thai_deck_gen.media.tts as m
    monkeypatch.setattr(m, "normalize_audio",
                        lambda raw, dst, runner=None: dst.write_bytes(raw))

class FakeTts:
    voice = "th-TH-Neural2-C"
    def synthesize(self, text, voice=None): return b"WAV:" + text.encode()

def _deck_with_sentence_and_pair(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sentences", "sounds"])
    deck.sentences.append(SentenceNote(
        id="s0", kind="new_word", thai="ฉันกินข้าว", target="กิน",
        audio=Audio(file="audio/sentences/s0.mp3", source="tts", speaker="pending")))
    deck.minimal_pairs.append(MinimalPairNote(
        id="mp-x-1", contrast="tone", members=[
            PairMember(thai="คา", ipa="kʰaː˧",
                       audio=Audio(file="audio/minimal_pairs/mp-x-1_0.mp3",
                                   source="native", speaker="pending")),
            PairMember(thai="ค่า", ipa="kʰaː˥˩",
                       audio=Audio(file="audio/minimal_pairs/mp-x-1_1.mp3",
                                   source="native", speaker="pending"))]))
    return deck

def test_fill_tts_fills_sentences_skips_pairs(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_sentence_and_pair(tmp_path)
    write_deck(deck)
    from thai_deck_gen.media.scan import pending_audio
    needs = pending_audio(deck)
    manifest = Manifest.load(deck.root)
    res = fill_tts(needs, deck, manifest, FakeTts(), "2026-08-27")

    assert res.changed == 1
    sentence = deck.sentences[0]
    assert sentence.audio.source == "tts"
    assert sentence.audio.speaker == "tts:th-TH-Neural2-C"
    assert (deck.root / "media" / sentence.audio.file).exists()
    assert manifest.channel_of(f"media/{sentence.audio.file}") == "tts"

    pair_member = deck.minimal_pairs[0].members[0]
    assert pair_member.audio.speaker == "pending"

def _deck_with_two_sentences(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sentences"])
    deck.sentences.append(SentenceNote(
        id="s0", kind="new_word", thai="ฉันกินข้าว", target="กิน",
        audio=Audio(file="audio/sentences/s0.mp3", source="tts", speaker="pending")))
    deck.sentences.append(SentenceNote(
        id="s1", kind="new_word", thai="ฉันนอน", target="นอน",
        audio=Audio(file="audio/sentences/s1.mp3", source="tts", speaker="pending")))
    return deck

class FailingTts:
    voice = "th-TH-Neural2-C"
    def __init__(self, fail_text): self.fail_text = fail_text
    def synthesize(self, text, voice=None):
        if text == self.fail_text:
            raise AudioError("tts boom")
        return b"WAV:" + text.encode()

def test_fill_tts_per_item_error_blocks_and_continues(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_two_sentences(tmp_path)
    write_deck(deck)
    from thai_deck_gen.media.scan import pending_audio
    needs = pending_audio(deck)
    manifest = Manifest.load(deck.root)
    res = fill_tts(needs, deck, manifest, FailingTts("ฉันกินข้าว"), "2026-08-27")

    assert "s0" in res.blocked
    assert res.changed == 1
    assert deck.sentences[0].audio.speaker == "pending"
    assert deck.sentences[1].audio.speaker == "tts:th-TH-Neural2-C"

def test_google_tts_parses_base64_response():
    payload = base64.b64encode(b"mp3-bytes").decode()
    def http_post(url, json, timeout=30):
        class R:
            status_code = 200
            def json(self): return {"audioContent": payload}
        return R()
    tts = GoogleTts("KEY", http_post=http_post)
    assert tts.synthesize("สวัสดี") == b"mp3-bytes"

def test_google_tts_raises_on_error():
    def http_post(url, json, timeout=30):
        class R:
            status_code = 400
            text = "bad request"
        return R()
    tts = GoogleTts("KEY", http_post=http_post)
    with pytest.raises(AudioError):
        tts.synthesize("สวัสดี")


def test_voice_varies_across_notes_but_is_stable_per_note(tmp_path, monkeypatch):
    """One synthetic voice for 732 listening cards trains the ear on that
    voice; the choice must still be stable so re-runs don't re-synthesize."""
    from thai_deck_gen.media.tts import fill_tts, pick_voice
    voices = ["th-TH-Neural2-C", "th-TH-Standard-A", "th-TH-Neural2-B"]
    picks = {f"sn-{i}": pick_voice(f"sn-{i}", voices) for i in range(30)}
    assert len(set(picks.values())) > 1               # not all the same
    assert all(v in voices for v in picks.values())
    assert pick_voice("sn-3", voices) == picks["sn-3"]   # stable
