import json
import pytest
import requests
from thai_deck_gen.media.forvo import ForvoClient, ForvoQuotaExceeded, fetch_forvo
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import pending_audio
from thai_deck_gen.deckio import write_deck
from tests.gen.test_scan import test_minimal_pair_needs_are_native_required  # noqa: F401
from tests.gen.test_sentences import _deck_with_words

class FakeForvo:
    def __init__(self, table): self.table = table
    def pronunciations(self, word): return self.table.get(word, [])
    def download(self, url): return b"mp3" + url.encode()

def _no_ffmpeg(monkeypatch, durations_ok=True):
    import thai_deck_gen.media.forvo as m
    monkeypatch.setattr(m, "normalize_audio",
                        lambda raw, dst, runner=None: dst.write_bytes(raw))
    monkeypatch.setattr(m, "duration_ok", lambda path: durations_ok)

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

class FailingForvo:
    def __init__(self, fail_word): self.fail_word = fail_word
    def pronunciations(self, word):
        if word == self.fail_word:
            raise requests.RequestException("boom")
        return [{"username": "a", "pathmp3": "http://f/a.mp3"}]
    def download(self, url): return b"mp3" + url.encode()

def test_fetch_forvo_per_item_error_blocks_and_continues(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 2)
    write_deck(deck)
    manifest = Manifest.load(deck.root)
    res = fetch_forvo(pending_audio(deck), deck, manifest, FailingForvo("w0"),
                      "2026-08-27")
    assert "w0" in res.blocked
    assert res.changed == 1
    assert deck.picture_words[1].audio.speaker == "forvo:a"

def test_client_parses_api_shape():
    def http_get(url, timeout=30):
        class R:
            status_code = 200
            def json(self):
                return {"items": [{"username": "a", "pathmp3": "u", "rate": 5}]}
        return R()
    c = ForvoClient("KEY", http_get=http_get)
    assert c.pronunciations("คา")[0]["username"] == "a"


def test_out_of_range_clip_is_discarded_and_blocks(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch, durations_ok=False)
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    manifest = Manifest.load(deck.root)
    res = fetch_forvo(pending_audio(deck), deck, manifest, 
                      FakeForvo({"w0": [{"username": "a", "pathmp3": "http://f/a.mp3"}]}),
                      "2026-08-27")
    note = deck.picture_words[0]
    assert res.blocked == ["w0"]
    assert res.changed == 0
    assert not (deck.root / "media" / note.audio.file).exists()
    assert note.audio.speaker == "pending"
    assert manifest.channel_of(f"media/{note.audio.file}") is None


def test_next_candidate_is_used_when_first_is_out_of_range(tmp_path, monkeypatch):
    import thai_deck_gen.media.forvo as m
    _no_ffmpeg(monkeypatch)
    monkeypatch.setattr(m, "duration_ok", lambda path: path.read_bytes().endswith(b"b.mp3"))
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    client = FakeForvo({"w0": [{"username": "a", "pathmp3": "http://f/a.mp3"},
                               {"username": "b", "pathmp3": "http://f/b.mp3"}]})
    manifest = Manifest.load(deck.root)
    res = fetch_forvo(pending_audio(deck), deck, manifest, client, "2026-08-27")
    note = deck.picture_words[0]
    assert res.changed == 1
    assert note.audio.speaker == "forvo:b"
    assert (deck.root / "media" / note.audio.file).read_bytes().endswith(b"b.mp3")
    assert manifest.channel_of(f"media/{note.audio.file}") == "forvo"


def test_limit_caps_api_lookups(tmp_path, monkeypatch):
    """A daily-quota cap leaves the rest pending, not blocked."""
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 3)
    write_deck(deck)
    table = {f"w{i}": [{"username": "a", "pathmp3": f"http://f/{i}.mp3"}] for i in range(3)}

    class Counting(FakeForvo):
        lookups = 0
        def pronunciations(self, word):
            Counting.lookups += 1
            return super().pronunciations(word)

    res = fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                      Counting(table), "2026-08-27", limit=2)
    assert Counting.lookups == 2
    assert res.changed == 2
    assert res.blocked == []
    assert deck.picture_words[2].audio.speaker == "pending"


def test_quota_exceeded_halts_and_keeps_progress(tmp_path, monkeypatch):
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 3)
    write_deck(deck)

    class Quota(FakeForvo):
        def pronunciations(self, word):
            if word == "w1":
                raise ForvoQuotaExceeded("daily limit reached")
            return super().pronunciations(word)

    table = {f"w{i}": [{"username": "a", "pathmp3": f"http://f/{i}.mp3"}] for i in range(3)}
    res = fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                      Quota(table), "2026-08-27")
    assert res.changed == 1
    assert res.blocked == []
    assert deck.picture_words[0].audio.speaker == "forvo:a"
    assert deck.picture_words[2].audio.speaker == "pending"


def test_client_raises_quota_error_on_429():
    def http_get(url, timeout=30):
        class R:
            status_code = 429
            text = "Limit exceeded"
        return R()
    with pytest.raises(ForvoQuotaExceeded):
        ForvoClient("KEY", http_get=http_get).pronunciations("คา")


def test_checkpoint_runs_periodically(tmp_path, monkeypatch):
    """Deck state is flushed mid-run so a killed run does not re-spend quota."""
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 3)
    write_deck(deck)
    table = {f"w{i}": [{"username": "a", "pathmp3": f"http://f/{i}.mp3"}] for i in range(3)}
    calls = []
    fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                FakeForvo(table), "2026-08-27",
                checkpoint=lambda: calls.append(1), checkpoint_every=2)
    assert len(calls) == 2          # after the 2nd fill, and a final flush


# --- lookup memoization: a request buys an answer, not just audio ---

def test_memoized_miss_is_not_looked_up_again(tmp_path, monkeypatch):
    from thai_deck_gen.media.forvo_memo import ForvoMemo
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    memo = ForvoMemo.load(deck.root)
    memo.record("w0", [], "2026-08-29")

    class Counting(FakeForvo):
        lookups = 0
        def pronunciations(self, word):
            Counting.lookups += 1
            return super().pronunciations(word)

    res = fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                      Counting({"w0": [{"username": "a", "pathmp3": "u"}]}),
                      "2026-08-30", memo=ForvoMemo.load(deck.root))
    assert Counting.lookups == 0
    assert res.blocked == ["w0"]


def test_memoized_hit_fills_without_an_api_call(tmp_path, monkeypatch):
    from thai_deck_gen.media.forvo_memo import ForvoMemo
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 1)
    write_deck(deck)
    memo = ForvoMemo.load(deck.root)
    memo.record("w0", [{"username": "a", "pathmp3": "http://f/a.mp3"}], "2026-08-29")

    class NoLookup(FakeForvo):
        def pronunciations(self, word):
            raise AssertionError("should not spend a request")

    res = fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                      NoLookup({}), "2026-08-30", memo=ForvoMemo.load(deck.root))
    assert res.changed == 1
    assert deck.picture_words[0].audio.speaker == "forvo:a"


def test_new_lookups_are_recorded_and_reloaded(tmp_path, monkeypatch):
    from thai_deck_gen.media.forvo_memo import ForvoMemo
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 2)
    write_deck(deck)
    table = {"w0": [{"username": "a", "pathmp3": "http://f/a.mp3"}]}   # w1 misses
    fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                FakeForvo(table), "2026-08-30", memo=ForvoMemo.load(deck.root))

    reloaded = ForvoMemo.load(deck.root)
    assert reloaded.seen("w0") and reloaded.seen("w1")
    assert reloaded.items("w0")[0]["username"] == "a"
    assert reloaded.items("w1") == []


def test_memoized_words_do_not_consume_the_request_limit(tmp_path, monkeypatch):
    from thai_deck_gen.media.forvo_memo import ForvoMemo
    _no_ffmpeg(monkeypatch)
    deck = _deck_with_words(tmp_path, 3)
    write_deck(deck)
    memo = ForvoMemo.load(deck.root)
    memo.record("w0", [], "2026-08-29")          # known miss, costs nothing

    class Counting(FakeForvo):
        lookups = 0
        def pronunciations(self, word):
            Counting.lookups += 1
            return super().pronunciations(word)

    table = {f"w{i}": [{"username": "a", "pathmp3": f"http://f/{i}.mp3"}] for i in (1, 2)}
    res = fetch_forvo(pending_audio(deck), deck, Manifest.load(deck.root),
                      Counting(table), "2026-08-30", limit=2,
                      memo=ForvoMemo.load(deck.root))
    assert Counting.lookups == 2                  # the limit funds two real words
    assert res.changed == 2
