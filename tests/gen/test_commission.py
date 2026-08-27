from thai_deck_gen.media.commission import write_commission_batch, import_commission
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import pending_audio
from thai_deck_gen.deckio import write_deck
from tests.gen.test_sentences import _deck_with_words

def _no_ffmpeg(monkeypatch):
    import thai_deck_gen.media.commission as m
    monkeypatch.setattr(m, "normalize_audio",
                        lambda raw, dst, runner=None: dst.write_bytes(raw))

def test_write_commission_batch_none_when_no_needs(tmp_path):
    assert write_commission_batch([], tmp_path) is None

def test_write_commission_batch_numbers_increment(tmp_path):
    deck = _deck_with_words(tmp_path / "d", 1)
    write_deck(deck)
    needs = pending_audio(deck)
    p1 = write_commission_batch(needs, deck.root)
    p2 = write_commission_batch(needs, deck.root)
    assert p1.name == "commission_batch_001.yaml"
    assert p2.name == "commission_batch_002.yaml"

def test_commission_round_trip(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path / "d", 2)
    write_deck(deck)
    needs = pending_audio(deck)
    batch_file = write_commission_batch(needs, deck.root)

    recordings = tmp_path / "recordings"
    recordings.mkdir()
    import yaml
    batch = yaml.safe_load(batch_file.read_text())
    matched_id = batch["items"][0]["id"]
    (recordings / f"{matched_id}.mp3").write_bytes(b"REC")

    manifest = Manifest.load(deck.root)
    res = import_commission(recordings, batch_file, deck, manifest, "jom", "2026-08-27")
    assert res.changed == 1
    assert res.blocked == [batch["items"][1]["id"]]

    note = deck.picture_words[0]
    assert note.audio.speaker == "commissioned:jom"
    assert (deck.root / "media" / note.audio.file).exists()
    assert manifest.channel_of(f"media/{note.audio.file}") == "commissioned"
