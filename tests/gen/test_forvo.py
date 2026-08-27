import json
from thai_deck_gen.media.forvo import ForvoClient, fetch_forvo
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import pending_audio
from thai_deck_gen.deckio import write_deck
from tests.gen.test_scan import test_minimal_pair_needs_are_native_required  # noqa: F401
from tests.gen.test_sentences import _deck_with_words

class FakeForvo:
    def __init__(self, table): self.table = table
    def pronunciations(self, word): return self.table.get(word, [])
    def download(self, url): return b"mp3" + url.encode()

def _no_ffmpeg(monkeypatch):
    import thai_deck_gen.media.forvo as m
    monkeypatch.setattr(m, "normalize_audio",
                        lambda raw, dst, runner=None: dst.write_bytes(raw))

def test_fetch_forvo_fills_word_audio(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    client = FakeForvo({"w0": [{"username": "a", "pathmp3": "http://f/a.mp3"}]})
    manifest = Manifest.load(deck.root)
    res = fetch_forvo(pending_audio(deck), deck, manifest, client, "2026-08-27")
    assert res.changed == 1
    note = deck.picture_words[0]
    assert note.audio.speaker == "forvo:a"
    assert (deck.root / "media" / note.audio.file).exists()
    assert manifest.channel_of(f"media/{note.audio.file}") == "forvo"

def test_fetch_forvo_blocks_missing_word(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    res = fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                      FakeForvo({}), "2026-08-27")
    assert res.blocked == ["w0"]

def test_client_parses_api_shape():
    def http_get(url, timeout=30):
        class R:
            status_code = 200
            def json(self):
                return {"items": [{"username": "a", "pathmp3": "u", "rate": 5}]}
        return R()
    c = ForvoClient("KEY", http_get=http_get)
    assert c.pronunciations("คา")[0]["username"] == "a"
