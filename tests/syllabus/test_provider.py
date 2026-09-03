"""Tests for provider.py (spec 3 sections 1-2): Provider.ask()'s
cache-first shape, and each backend's cache-key function + fetch
contract. Real SyllabusDb (tmp_path sqlite) as the cache/record for the
cache-first behavior; fake backends and fake transports everywhere else
-- no network, no subprocess, no anthropic import.
"""
from pathlib import Path

import pytest

from thai_syllabus.provider import (
    FetchBackend,
    ForvoBackend,
    HttpImageSearchBackend,
    LearnerAskNotSupported,
    LlmBackend,
    PairSearchBackend,
    Provider,
    ProviderAnswer,
    Question,
    RawAnswer,
    openverse_backend,
    pexels_backend,
    tool_fetcher,
    wikimedia_backend,
)
from thai_syllabus.store import ImageIngestResult, MediaStore, SyllabusDb
from thai_syllabus.transport import Completion, TransportError
from thai_syllabus.tts import GoogleTts, pick_voice


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _FakeBackend:
    def __init__(self, key="k", raises=None, items=("x",), cost=0.0):
        self.key = key
        self.raises = raises
        self.items = items
        self.cost = cost
        self.fetch_calls = 0

    def cache_key(self, question):
        return self.key

    def fetch(self, question):
        self.fetch_calls += 1
        if self.raises:
            raise self.raises
        return RawAnswer(items=self.items, cost=self.cost)


# --- cache-first shape (spec 3 section 1) -----------------------------

def test_a_miss_executes_and_appends_one_row(db):
    backend = _FakeBackend(items=("pic-1",), cost=0.5)
    provider = Provider(record=db, cache=db, backends={"x": backend})
    answer = provider.ask("x", Question(subject="s1", provides="picture"))
    assert isinstance(answer, ProviderAnswer)
    assert answer.items == ("pic-1",)
    assert answer.cost == 0.5
    assert backend.fetch_calls == 1
    assert len(db.assessments_of("s1")) == 1


def test_a_hit_does_not_execute_and_appends_nothing(db):
    backend = _FakeBackend(items=("pic-1",), cost=0.5)
    provider = Provider(record=db, cache=db, backends={"x": backend})
    provider.ask("x", Question(subject="s1", provides="picture"))
    assert backend.fetch_calls == 1
    answer = provider.ask("x", Question(subject="s1", provides="picture"))
    assert backend.fetch_calls == 1  # not called again
    assert answer.items == ("pic-1",)
    assert answer.cost == 0.0  # a hit costs nothing
    assert len(db.assessments_of("s1")) == 1  # nothing appended on the hit


def test_an_empty_answer_is_cached(db):
    backend = _FakeBackend(items=())
    provider = Provider(record=db, cache=db, backends={"x": backend})
    provider.ask("x", Question(subject="s1", provides="picture"))
    assert backend.fetch_calls == 1
    provider.ask("x", Question(subject="s1", provides="picture"))
    assert backend.fetch_calls == 1  # the empty answer was cached; not re-asked


def test_a_transport_error_is_not_cached_and_propagates(db):
    backend = _FakeBackend(raises=TransportError("network down"))
    provider = Provider(record=db, cache=db, backends={"x": backend})
    with pytest.raises(TransportError):
        provider.ask("x", Question(subject="s1", provides="picture"))
    assert db.assessments_of("s1") == []  # nothing appended
    # a second attempt retries the backend (subject stays queued)
    with pytest.raises(TransportError):
        provider.ask("x", Question(subject="s1", provides="picture"))
    assert backend.fetch_calls == 2


def test_learner_backend_raises_without_touching_cache_or_record(db):
    provider = Provider(record=db, cache=db, backends={})
    with pytest.raises(LearnerAskNotSupported):
        provider.ask("learner", Question(subject="s1", provides="picture"))
    assert db.assessments_of("s1") == []


def test_the_stored_cache_row_carries_the_readable_key(db):
    backend = _FakeBackend(key="openverse:rice bowl")
    provider = Provider(record=db, cache=db, backends={"openverse": backend})
    provider.ask("openverse", Question(subject="rice", provides="picture"))
    answer = db.assessments_of("rice")[0]
    assert answer.key == "openverse:rice bowl"
    assert answer.port == "provide" and answer.backend == "openverse"


# --- image search backends --------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json


def test_openverse_cache_key_is_backend_colon_query():
    backend = openverse_backend()
    key = backend.cache_key(Question(subject="rice", provides="picture",
                                     params={"query": "rice bowl"}))
    assert key == "openverse:rice bowl"


def test_openverse_fetch_parses_results_and_sets_descriptive_user_agent():
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, headers))
        return _FakeResponse(json_data={"results": [
            {"url": "https://x/img.jpg", "license": "cc0",
             "foreign_landing_url": "https://x/page"}]})

    backend = openverse_backend(get=fake_get)
    answer = backend.fetch(Question(subject="rice", provides="picture",
                                    params={"query": "rice bowl"}))
    assert answer.items[0]["url"] == "https://x/img.jpg"
    assert "openverse.org" in calls[0][0]
    assert calls[0][1]["q"] == "rice bowl"
    assert "thai-syllabus" in calls[0][2]["User-Agent"]


def test_openverse_search_proxy_replaces_the_base_url():
    backend = openverse_backend(search_proxy="https://proxy.example")

    def fake_get(url, **kwargs):
        assert url.startswith("https://proxy.example")
        return _FakeResponse(json_data={"results": []})

    backend.get = fake_get
    backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))


def test_a_non_200_response_is_a_transport_error_not_cached_by_the_backend():
    backend = openverse_backend(get=lambda *a, **k: _FakeResponse(status_code=500))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))


def test_wikimedia_and_pexels_backends_key_by_backend_name():
    wm = wikimedia_backend()
    px = pexels_backend(api_key="k")
    q = Question(subject="s", provides="picture", params={"query": "cat"})
    assert wm.cache_key(q) == "wikimedia:cat"
    assert px.cache_key(q) == "pexels:cat"


def test_wikimedia_uses_imageinfo_generator_and_returns_urls():
    seen = {}

    def get(url, params, headers, timeout):
        seen.update(params)
        return _FakeResponse(json_data={"query": {"pages": {"1": {
            "title": "File:A.jpg",
            "imageinfo": [{"url": "https://u/A.jpg"}]}}}})

    backend = wikimedia_backend(get=get)
    answer = backend.fetch(Question(subject="w", provides="picture",
                                    params={"query": "orange"}))
    assert seen["generator"] == "search" and seen["prop"] == "imageinfo"
    assert seen["iiprop"] == "url" and seen["gsrsearch"] == "orange"
    assert seen["gsrnamespace"] == "6"
    assert answer.items[0]["url"] == "https://u/A.jpg"
    assert answer.items[0]["source"] == "wikimedia"
    assert answer.items[0]["origin"] == "https://commons.wikimedia.org/wiki/File:A.jpg"


def test_wikimedia_parse_skips_pages_without_a_url():
    backend = wikimedia_backend()
    data = {"query": {"pages": {
        "1": {"title": "File:NoUrl.jpg", "imageinfo": [{}]},
        "2": {"title": "File:B.jpg", "imageinfo": [{"url": "https://u/B.jpg"}]},
    }}}
    items = backend.parse_items(data)
    assert [i["url"] for i in items] == ["https://u/B.jpg"]


def test_pexels_fetch_sends_the_api_key_as_authorization_header():
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(headers)
        return _FakeResponse(json_data={"photos": [
            {"src": {"original": "https://x/p.jpg"}, "url": "https://x/page"}]})

    backend = pexels_backend(api_key="SECRET", get=fake_get)
    answer = backend.fetch(Question(subject="s", provides="picture",
                                    params={"query": "cat"}))
    assert calls[0]["Authorization"] == "SECRET"
    assert answer.items[0]["url"] == "https://x/p.jpg"


# --- FetchBackend: url -> bytes -> media store, pictures and recordings ---

class _Media:
    def __init__(self):
        self.written, self.images = [], []

    def write(self, data, ext):
        self.written.append((data, ext))
        return "sha-" + ext

    def add_image(self, data, ext):
        self.images.append((data, ext))
        return ImageIngestResult(sha="img-" + ext, ext=ext, warning=None)


def test_fetch_backend_key_is_the_url():
    backend = FetchBackend(media=None, fetcher=lambda url: (b"x", "jpg"))
    key = backend.cache_key(Question(subject="s", provides="picture-bytes",
                                     params={"url": "https://x/y.jpg"}))
    assert key == "https://x/y.jpg"


def test_fetch_backend_stores_recording_bytes_raw_and_echoes_params():
    media = _Media()
    b = FetchBackend(media=media, fetcher=lambda url: (b"mp3bytes", "mp3"))
    q = Question(subject="w", provides="recording-bytes",
                 params={"url": "https://apifree.forvo.com/x.mp3", "speaker": "krisflyer",
                        "speaker_kind": "native"})
    assert b.cache_key(q) == "https://apifree.forvo.com/x.mp3"
    ans = b.fetch(q)
    assert media.written == [(b"mp3bytes", "mp3")] and media.images == []
    item = ans.items[0]
    assert item["sha"] == "sha-mp3" and item["speaker"] == "krisflyer"
    assert item["speaker_kind"] == "native"
    assert "url" not in item and ans.cost == 0.0


def test_fetch_backend_ingests_picture_bytes_through_add_image():
    media = _Media()
    b = FetchBackend(media=media, fetcher=lambda url: (b"jpg", "jpg"))
    ans = b.fetch(Question(subject="w", provides="picture-bytes",
                           params={"url": "https://x/a.jpg"}))
    assert media.images == [(b"jpg", "jpg")] and media.written == []
    assert ans.items[0]["sha"] == "img-jpg"


def test_fetch_backend_failure_raises_and_is_not_cached():
    media = _Media()

    def failing_fetcher(url):
        raise TransportError("404")

    backend = FetchBackend(media=media, fetcher=failing_fetcher)
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="s", provides="picture-bytes",
                               params={"url": "https://x/y.jpg"}))


# --- tool_fetcher: the Go tools' interface (binary url out-path) --------

def test_tool_fetcher_returns_bytes_and_mapped_extension_on_success():
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[2]).write_bytes(b"bytes-out")
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, '{"format":"png","bytes":9}\n', "")

    fetcher = tool_fetcher("/opt/bin/imgfetch", runner=runner)
    data, ext = fetcher("https://x/pic.png")
    assert data == b"bytes-out"
    assert ext == "png"
    assert calls[0][0] == "/opt/bin/imgfetch" and calls[0][1] == "https://x/pic.png"


def test_tool_fetcher_raises_transport_error_on_nonzero_exit():
    import subprocess as sp

    def runner(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, "", "imgfetch: refused: not an image")

    fetcher = tool_fetcher("imgfetch", runner=runner)
    with pytest.raises(TransportError):
        fetcher("https://x/missing.jpg")


def test_tool_fetcher_raises_transport_error_when_binary_is_missing():
    def runner(cmd, **kwargs):
        raise OSError("no such file")

    fetcher = tool_fetcher("imgfetch", runner=runner)
    with pytest.raises(TransportError):
        fetcher("https://x/y.jpg")


# --- forvo: never re-asked, key = forvo:WORD ----------------------------

def test_forvo_cache_key_is_forvo_colon_word():
    backend = ForvoBackend(api_key="k")
    key = backend.cache_key(Question(subject="ไก่", provides="recording"))  # chicken
    assert key == "forvo:ไก่"


def test_forvo_fetch_returns_items_and_a_transport_error_on_bad_status():
    ok = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                      _FakeResponse(json_data={"items": [{"id": 1}]}))
    answer = ok.fetch(Question(subject="ไก่", provides="recording"))
    assert answer.items == ({"id": 1},)

    bad = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                       _FakeResponse(status_code=500))
    with pytest.raises(TransportError):
        bad.fetch(Question(subject="ไก่", provides="recording"))


def test_forvo_empty_result_is_still_a_valid_answer(db):
    # "never re-asked (the answer outlives the quota)" -- an empty lookup
    # must cache too, via the generic Provider empty-answer-is-cached rule.
    backend = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                           _FakeResponse(json_data={"items": []}))
    provider = Provider(record=db, cache=db, backends={"forvo": backend})
    calls_before = backend.get
    provider.ask("forvo", Question(subject="หมา", provides="recording"))  # dog
    hit = db.latest("provide", "forvo", "forvo:หมา")
    assert hit is not None
    assert hit.answer == {"items": []}


# --- tts: deterministic voice pick, key includes sha(text) --------------

def test_tts_cache_key_includes_the_picked_voice_and_sha_of_text(tmp_path):
    from thai_syllabus.provider import TtsBackend
    media = MediaStore(tmp_path / "media")
    voices = ["voice-a", "voice-b"]
    backend = TtsBackend(tts=None, voices=voices, media=media, pick_voice=pick_voice)
    key = backend.cache_key(Question(subject="subj-1", provides="recording",
                                     params={"text": "ผมกินข้าว"}))  # I eat rice
    assert key.startswith("tts:")
    voice = pick_voice("subj-1", voices)
    assert key.split(":")[1] == voice


def test_tts_fetch_writes_synthesized_audio_content_addressed(tmp_path):
    from thai_syllabus.provider import TtsBackend

    class _FakeTts:
        def synthesize(self, text, voice):
            return f"{voice}:{text}".encode()

    media = MediaStore(tmp_path / "media")
    backend = TtsBackend(tts=_FakeTts(), voices=["v1"], media=media, pick_voice=pick_voice)
    answer = backend.fetch(Question(subject="s", provides="recording",
                                    params={"text": "hi", "voice": "v1"}))
    assert answer.items[0]["voice"] == "v1"
    assert media.has(answer.items[0]["sha"], "mp3")


def test_tts_is_deterministic_same_subject_same_voice():
    voices = ["v1", "v2", "v3"]
    assert pick_voice("subj-1", voices) == pick_voice("subj-1", voices)


def test_tts_items_carry_synthetic_speaker_kind():
    from thai_syllabus.provider import TtsBackend

    class T:
        def synthesize(self, text, voice):
            return b"audio"

    backend = TtsBackend(tts=T(), voices=["v1"], media=_Media(), pick_voice=lambda s, v: v[0])
    ans = backend.fetch(Question(subject="w", provides="recording", params={"text": "ช้า"}))
    assert ans.items[0]["speaker_kind"] == "synthetic"
    assert ans.items[0]["source"] == "tts"
    assert ans.items[0]["origin"] == "v1"


# --- llm: key = llm:PRODUCER:MODEL:sha(PROMPT) --------------------------

class _FakeTransport:
    def __init__(self, text="drafted sentence", raises=None):
        self.text = text
        self.raises = raises
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        return Completion(text=self.text)


def test_llm_cache_key_is_stable_for_the_same_prompt():
    backend = LlmBackend(producer="sentence-drafter", model="claude-opus-5",
                         transport=_FakeTransport())
    q = Question(subject="s", provides="sentence", params={"prompt": "write a sentence"})
    assert backend.cache_key(q) == backend.cache_key(q)
    assert backend.cache_key(q).startswith("llm:sentence-drafter:claude-opus-5:")


def test_llm_cache_key_changes_when_the_prompt_text_changes():
    backend = LlmBackend(producer="p", model="m", transport=_FakeTransport())
    k1 = backend.cache_key(Question(subject="s", provides="sentence",
                                    params={"prompt": "prompt A"}))
    k2 = backend.cache_key(Question(subject="s", provides="sentence",
                                    params={"prompt": "prompt B"}))
    assert k1 != k2


def test_llm_fetch_delegates_to_the_transport_and_wraps_the_completion():
    transport = _FakeTransport(text="ผมกินข้าว")  # I eat rice
    backend = LlmBackend(producer="p", model="m", transport=transport, cost_per_call=0.002)
    answer = backend.fetch(Question(subject="s", provides="sentence",
                                    params={"prompt": "write a sentence about rice"}))
    assert answer.items == ("ผมกินข้าว",)  # I eat rice
    assert answer.cost == 0.002
    assert transport.prompts == ["write a sentence about rice"]


def test_llm_transport_error_propagates_uncached(db):
    transport = _FakeTransport(raises=TransportError("cli failed"))
    backend = LlmBackend(producer="p", model="m", transport=transport)
    provider = Provider(record=db, cache=db, backends={"llm": backend})
    with pytest.raises(TransportError):
        provider.ask("llm", Question(subject="s", provides="sentence",
                                     params={"prompt": "x"}))
    assert db.assessments_of("s") == []


# --- pair-search: over a DictionaryG2P port (fake only) ------------------

class _FakeG2P:
    def __init__(self, version="v1", results=()):
        self._version = version
        self._results = results
        self.searched = []

    def version(self):
        return self._version

    def search(self, confusion_id):
        self.searched.append(confusion_id)
        return self._results


def test_pair_search_cache_key_includes_confusion_and_dictionary_version():
    g2p = _FakeG2P(version="2026-09-01")
    backend = PairSearchBackend(dictionary=g2p)
    key = backend.cache_key(Question(subject="tone:mid-low", provides="pair",
                                     params={"confusion_id": "tone:mid-low"}))
    assert key == "pairs:tone:mid-low:2026-09-01"


def test_pair_search_key_changes_when_the_dictionary_version_bumps():
    old = PairSearchBackend(dictionary=_FakeG2P(version="v1"))
    new = PairSearchBackend(dictionary=_FakeG2P(version="v2"))
    q = Question(subject="c", provides="pair", params={"confusion_id": "c"})
    assert old.cache_key(q) != new.cache_key(q)


def test_pair_search_fetch_returns_the_dictionarys_candidates():
    g2p = _FakeG2P(results=({"members": ["ใกล้", "ไกล"]},))  # near, far
    backend = PairSearchBackend(dictionary=g2p)
    answer = backend.fetch(Question(subject="c", provides="pair",
                                    params={"confusion_id": "tone:mid-low"}))
    assert answer.items[0]["members"] == ["ใกล้", "ไกล"]  # near, far
    assert g2p.searched == ["tone:mid-low"]
