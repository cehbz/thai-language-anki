"""Tests for provider.py (spec 3 sections 1-2): Provider.ask()'s
cache-first shape, and each backend's cache-key function + fetch
contract. Real SyllabusDb (tmp_path sqlite) as the cache/record for the
cache-first behavior; fake backends and fake transports everywhere else
-- no network, no subprocess, no anthropic import.
"""
from pathlib import Path

import pytest
import requests

from thai_syllabus.assessor import Price
from thai_syllabus.cachekeys import ProvideKey, sha
from thai_syllabus.provider import (
    OpenverseAuth,
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
    _redact,
    brave_backend,
    forvo_limit_body,
    openverse_backend,
    pexels_backend,
    tool_fetcher,
    wikimedia_backend,
)
from thai_syllabus.store import ImageIngestResult, MediaStore, SyllabusDb
from thai_syllabus.transport import Completion, QuotaExhausted, TransportError
from thai_syllabus.tts import GoogleTts, pick_voice


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _FakeBackend:
    def __init__(self, key=None, raises=None, items=("x",), cost=0.0):
        self.key = key if key is not None else ProvideKey(source="", kind="", query="k")
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
    answer = provider.ask("x", Question(subject="s1", provides="picture", kind="picture"))
    assert isinstance(answer, ProviderAnswer)
    assert answer.items == ("pic-1",)
    assert answer.cost == 0.5
    assert backend.fetch_calls == 1
    rows = db.assessments_of("s1")
    assert len(rows) == 1
    assert rows[0].question["kind"] == "picture"  # record.rows_for reads this back


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
    key = ProvideKey(source="openverse", kind="", query="rice bowl")
    backend = _FakeBackend(key=key)
    provider = Provider(record=db, cache=db, backends={"openverse": backend})
    provider.ask("openverse", Question(subject="rice", provides="picture"))
    answer = db.assessments_of("rice")[0]
    assert answer.key == key.encode()
    assert answer.port == "provide" and answer.backend == "openverse"


# --- reask (spec 3 section 2; section 6a's re-ask rule) ---------------

def test_reask_executes_over_a_hit_and_the_newest_row_answers(db):
    calls = []

    class _Counting:
        def cache_key(self, q):
            return ProvideKey(source="forvo", kind="", query=q.params["word"])

        def fetch(self, q):
            calls.append(q.params["word"])
            return RawAnswer(items=({"id": len(calls)},), cost=1.0)

    provider = Provider(record=db, cache=db, backends={"forvo": _Counting()})
    q = Question(subject="dog", provides="recording", params={"word": "หมา"},  # หมา: dog
                 kind="recording", subject_kind="word")
    first = provider.ask("forvo", q)
    again = provider.reask("forvo", q)
    assert first.items == ({"id": 1},) and again.items == ({"id": 2},) and again.hit is False
    assert provider.ask("forvo", q).items == ({"id": 2},)
    assert calls == ["หมา", "หมา"]


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
    assert key.encode() == "openverse::rice bowl"


def test_openverse_fetch_parses_results_and_sets_descriptive_user_agent():
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None, proxies=None):
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


def test_openverse_search_proxy_is_sent_as_a_forward_proxy_with_the_real_url():
    calls = []

    def get(url, params=None, headers=None, timeout=None, proxies=None):
        calls.append((url, proxies))
        return _FakeResponse(json_data={"results": []})

    backend = openverse_backend(get=get, search_proxy="http://10.112.227.2:8888")
    backend.fetch(Question(subject="w", provides="picture", params={"query": "orange"}))
    assert calls == [("https://api.openverse.org/v1/images/",
                      {"http": "http://10.112.227.2:8888", "https": "http://10.112.227.2:8888"})]


def test_a_search_without_a_proxy_sends_none():
    calls = []

    def get(url, params=None, headers=None, timeout=None, proxies=None):
        calls.append(proxies)
        return _FakeResponse(json_data={"results": []})

    openverse_backend(get=get).fetch(Question(subject="w", provides="picture", params={"query": "orange"}))
    assert calls == [None]


def test_a_non_200_response_is_a_transport_error_not_cached_by_the_backend():
    backend = openverse_backend(get=lambda *a, **k: _FakeResponse(status_code=500))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))


def test_openverse_a_200_body_without_results_is_a_transport_error():
    backend = openverse_backend(get=lambda url, **kwargs: _FakeResponse(json_data={"detail": "throttled"}))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="w", provides="picture", params={"query": "orange"}))


# --- spec 3 r26: pacing, the Cloudflare challenge, 429 as Quota, the bearer token

class _Clock:
    def __init__(self, *readings):
        self.readings = list(readings)

    def __call__(self):
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


def _results(*urls):
    return _FakeResponse(json_data={"results": [{"url": u, "license": "cc0"} for u in urls]})


def test_a_paced_backend_sleeps_the_remainder_of_the_interval_between_two_fetches():
    slept = []
    backend = openverse_backend(get=lambda *a, **k: _results("https://x/a.jpg"),
                                min_interval_s=4.0, sleep=slept.append,
                                clock=_Clock(10.0, 10.5, 11.0, 15.0))
    q = Question(subject="rice", provides="picture", params={"query": "q"})
    backend.fetch(q)          # first ask: no wait; last call stamped at 10.5
    backend.fetch(q)          # 11.0 now: 3.5 s of the 4 s interval remain
    assert slept == [3.5]


def test_an_unpaced_backend_never_sleeps():
    slept = []
    backend = openverse_backend(get=lambda *a, **k: _results("https://x/a.jpg"), sleep=slept.append)
    q = Question(subject="rice", provides="picture", params={"query": "q"})
    backend.fetch(q)
    backend.fetch(q)
    assert slept == []


_CHALLENGE = _FakeResponse(status_code=403,
                           text="<!DOCTYPE html><html><head><title>Just a moment...</title>")


def test_a_challenge_page_is_retried_once_after_the_wait():
    answers = [_CHALLENGE, _results("https://x/a.jpg")]
    slept = []
    backend = openverse_backend(get=lambda *a, **k: answers.pop(0), challenge_wait_s=60.0,
                                sleep=slept.append)
    answer = backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))
    assert slept == [60.0]
    assert [i["url"] for i in answer.items] == ["https://x/a.jpg"]


def test_two_challenge_pages_are_a_transport_error_naming_the_challenge():
    answers = [_CHALLENGE, _CHALLENGE]
    backend = openverse_backend(get=lambda *a, **k: answers.pop(0), challenge_wait_s=60.0,
                                sleep=lambda s: None)
    with pytest.raises(TransportError, match="challenge"):
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))
    assert answers == []


def test_a_challenge_page_with_no_wait_configured_is_a_transport_error_at_once():
    backend = openverse_backend(get=lambda *a, **k: _CHALLENGE, sleep=lambda s: None)
    with pytest.raises(TransportError, match="challenge"):
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))


def test_a_429_is_the_quota_state_not_a_source_failure():
    backend = openverse_backend(get=lambda *a, **k: _FakeResponse(status_code=429, text="throttled"))
    with pytest.raises(QuotaExhausted) as e:
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))
    assert e.value.source == "openverse"


def test_a_bearer_token_is_sent_when_the_auth_callable_gives_one():
    headers_seen = []

    def get(url, params=None, headers=None, timeout=None, proxies=None):
        headers_seen.append(dict(headers))
        return _results()

    openverse_backend(get=get, auth=lambda: "tok123").fetch(
        Question(subject="w", provides="picture", params={"query": "q"}))
    assert headers_seen[0]["Authorization"] == "Bearer tok123"
    assert "thai-syllabus" in headers_seen[0]["User-Agent"]


def test_no_authorization_header_without_an_auth_callable():
    headers_seen = []

    def get(url, params=None, headers=None, timeout=None, proxies=None):
        headers_seen.append(dict(headers))
        return _results()

    openverse_backend(get=get).fetch(Question(subject="w", provides="picture", params={"query": "q"}))
    assert "Authorization" not in headers_seen[0]


def test_openverse_auth_posts_form_encoded_client_credentials_through_the_proxy_and_caches():
    posts = []

    def post(url, data=None, headers=None, timeout=None, proxies=None):
        posts.append((url, dict(data), proxies))
        return _FakeResponse(json_data={"access_token": "tok123", "expires_in": 36000,
                                        "token_type": "Bearer", "scope": "read write"})

    auth = OpenverseAuth(client_id="cid", client_secret="sec", post=post,
                         search_proxy="http://10.112.227.2:8888", clock=_Clock(100.0))
    assert auth.token() == "tok123"
    assert auth.token() == "tok123"
    assert posts == [("https://api.openverse.org/v1/auth_tokens/token/",
                      {"client_id": "cid", "client_secret": "sec",
                       "grant_type": "client_credentials"},
                      {"http": "http://10.112.227.2:8888", "https": "http://10.112.227.2:8888"})]


def test_openverse_auth_refreshes_an_expired_token():
    posts = []

    def post(url, data=None, headers=None, timeout=None, proxies=None):
        posts.append(url)
        return _FakeResponse(json_data={"access_token": f"tok{len(posts)}", "expires_in": 100})

    clock = _Clock(0.0, 0.0, 50.0, 200.0, 200.0)
    auth = OpenverseAuth(client_id="cid", client_secret="sec", post=post, clock=clock)
    assert auth.token() == "tok1"
    assert auth.token() == "tok1"     # 50 s in: still valid
    assert auth.token() == "tok2"     # 200 s in: past expires_in, fetched again
    assert len(posts) == 2


def test_openverse_auth_failure_is_a_transport_error():
    auth = OpenverseAuth(client_id="cid", client_secret="sec",
                         post=lambda *a, **k: _FakeResponse(status_code=401, text="bad"))
    with pytest.raises(TransportError, match="openverse"):
        auth.token()


def test_wikimedia_a_200_body_without_batchcomplete_is_a_transport_error():
    backend = wikimedia_backend(image_width=1600, get=lambda url, **kwargs: _FakeResponse(json_data={"detail": "throttled"}))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="w", provides="picture", params={"query": "orange"}))


def test_pexels_a_200_body_without_photos_is_a_transport_error():
    backend = pexels_backend(api_key="k",
                             get=lambda url, **kwargs: _FakeResponse(json_data={"error": "x"}))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="w", provides="picture", params={"query": "orange"}))


def test_a_200_body_that_is_not_json_is_a_transport_error():
    class _Bad(_FakeResponse):
        def json(self):
            raise ValueError("not json")

    backend = openverse_backend(get=lambda *a, **k: _Bad(status_code=200))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))


def test_openverse_empty_result_is_still_a_valid_answer(db):
    # a zero-hit search caches too (spec 3 section 6a: no hits is `nothing`,
    # not a transport error), via the generic Provider empty-answer rule.
    backend = openverse_backend(get=lambda url, **kwargs: _FakeResponse(json_data={"results": []}))
    provider = Provider(record=db, cache=db, backends={"openverse": backend})
    provider.ask("openverse", Question(subject="rice", provides="picture", params={"query": "q"}))
    hit = db.latest("provide", "openverse", ProvideKey(source="openverse", kind="", query="q"))
    assert hit is not None
    assert hit.answer == {"items": []}


def test_wikimedia_empty_result_is_still_a_valid_answer(db):
    # a zero-hit MediaWiki search body carries "batchcomplete" and no
    # "query" key at all -- expect="batchcomplete" admits it.
    backend = wikimedia_backend(image_width=1600, get=lambda url, **kwargs: _FakeResponse(json_data={"batchcomplete": ""}))
    provider = Provider(record=db, cache=db, backends={"wikimedia": backend})
    provider.ask("wikimedia", Question(subject="rice", provides="picture", params={"query": "q"}))
    hit = db.latest("provide", "wikimedia", ProvideKey(source="wikimedia", kind="", query="q"))
    assert hit is not None
    assert hit.answer == {"items": []}


def test_pexels_empty_result_is_still_a_valid_answer(db):
    backend = pexels_backend(api_key="k",
                             get=lambda url, **kwargs: _FakeResponse(json_data={"photos": []}))
    provider = Provider(record=db, cache=db, backends={"pexels": backend})
    provider.ask("pexels", Question(subject="rice", provides="picture", params={"query": "q"}))
    hit = db.latest("provide", "pexels", ProvideKey(source="pexels", kind="", query="q"))
    assert hit is not None
    assert hit.answer == {"items": []}


def test_wikimedia_and_pexels_backends_key_by_backend_name():
    wm = wikimedia_backend(image_width=1600)
    px = pexels_backend(api_key="k")
    q = Question(subject="s", provides="picture", params={"query": "cat"})
    assert wm.cache_key(q).encode() == "wikimedia::cat"
    assert px.cache_key(q).encode() == "pexels::cat"


def test_wikimedia_uses_imageinfo_generator_and_returns_thumburl():
    seen = {}

    def get(url, params, headers, timeout, proxies=None):
        seen.update(params)
        return _FakeResponse(json_data={"batchcomplete": "", "query": {"pages": {"1": {
            "title": "File:A.jpg",
            "imageinfo": [{"url": "https://u/A.jpg",
                          "thumburl": "https://u/thumb/A-1600px.jpg"}]}}}})

    backend = wikimedia_backend(image_width=1600, get=get)
    answer = backend.fetch(Question(subject="w", provides="picture",
                                    params={"query": "orange"}))
    assert seen["generator"] == "search" and seen["prop"] == "imageinfo"
    assert seen["iiprop"] == "url"
    assert seen["gsrsearch"] == "orange filetype:bitmap"
    assert seen["gsrnamespace"] == "6"
    assert seen["iiurlwidth"] == 1600
    assert answer.items[0]["url"] == "https://u/thumb/A-1600px.jpg"
    assert answer.items[0]["source"] == "wikimedia"
    assert answer.items[0]["origin"] == "https://commons.wikimedia.org/wiki/File:A.jpg"


def test_wikimedia_image_width_bounds_the_request():
    backend = wikimedia_backend(image_width=800)
    _, params, _, _ = backend.build_request("orange")
    assert params["iiurlwidth"] == 800
    assert params["gsrsearch"] == "orange filetype:bitmap"


def test_wikimedia_parse_skips_an_info_without_a_thumburl():
    backend = wikimedia_backend(image_width=1600)
    data = {"query": {"pages": {
        "1": {"title": "File:NoThumb.jpg", "imageinfo": [{"url": "https://u/NoThumb.jpg"}]},
        "2": {"title": "File:B.jpg", "imageinfo": [{"url": "https://u/B.jpg",
                                                    "thumburl": "https://u/thumb/B-1600px.jpg"}]},
    }}}
    items = backend.parse_items(data)
    assert [i["url"] for i in items] == ["https://u/thumb/B-1600px.jpg"]


def test_pexels_fetch_sends_the_api_key_as_authorization_header():
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None, proxies=None):
        calls.append(headers)
        return _FakeResponse(json_data={"photos": [
            {"src": {"original": "https://x/p.jpg"}, "url": "https://x/page"}]})

    backend = pexels_backend(api_key="SECRET", get=fake_get)
    answer = backend.fetch(Question(subject="s", provides="picture",
                                    params={"query": "cat"}))
    assert calls[0]["Authorization"] == "SECRET"
    assert answer.items[0]["url"] == "https://x/p.jpg"


# --- brave: the last picture source, a paid search API --------------------

def _brave_results():
    return _FakeResponse(json_data={"results": [
        {"title": "A live crab on sand",
         "url": "https://x/page",
         "thumbnail": {"src": "https://thumb/c.jpg"},
         "properties": {"url": "https://x/full.jpg", "placeholder": "https://p/c.jpg"}}]})


def test_brave_builds_the_image_search_request():
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None, proxies=None):
        calls.append((url, dict(params), dict(headers)))
        return _brave_results()

    backend = brave_backend(api_key="SECRET", get=fake_get, count=5)
    backend.fetch(Question(subject="ปู", provides="picture", params={"query": "live crab"}))
    url, params, headers = calls[0]
    assert url == "https://api.search.brave.com/res/v1/images/search"
    assert params == {"q": "live crab", "count": 5, "safesearch": "strict"}
    assert headers["X-Subscription-Token"] == "SECRET"
    assert headers["Accept"] == "application/json"
    assert "thai-syllabus" in headers["User-Agent"]


def test_brave_count_is_the_deck_image_candidates():
    backend = brave_backend(api_key="k", count=3)
    _, params, _, _ = backend.build_request("live crab")
    assert params["count"] == 3


def test_brave_parses_results_into_the_shared_candidate_item_shape():
    backend = brave_backend(api_key="k", get=lambda *a, **k: _brave_results())
    answer = backend.fetch(Question(subject="ปู", provides="picture",
                                    params={"query": "live crab"}))
    assert answer.items == ({"url": "https://x/full.jpg", "licence": None,
                             "source": "brave", "origin": "https://x/page"},)


def test_brave_falls_back_to_the_thumbnail_when_properties_carries_no_url():
    data = {"results": [{"title": "t", "url": "https://x/page",
                         "thumbnail": {"src": "https://thumb/c.jpg"}, "properties": {}}]}
    backend = brave_backend(api_key="k")
    assert backend.parse_items(data)[0]["url"] == "https://thumb/c.jpg"


def test_brave_keys_by_backend_name():
    q = Question(subject="s", provides="picture", params={"query": "cat"})
    assert brave_backend(api_key="k").cache_key(q).encode() == "brave::cat"


def test_brave_a_200_body_without_results_is_a_transport_error():
    backend = brave_backend(api_key="k",
                            get=lambda url, **kwargs: _FakeResponse(json_data={"error": "x"}))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="w", provides="picture", params={"query": "orange"}))


def test_brave_empty_result_is_still_a_valid_answer(db):
    backend = brave_backend(api_key="k",
                            get=lambda url, **kwargs: _FakeResponse(json_data={"results": []}))
    provider = Provider(record=db, cache=db, backends={"brave": backend})
    provider.ask("brave", Question(subject="rice", provides="picture", params={"query": "q"}))
    hit = db.latest("provide", "brave", ProvideKey(source="brave", kind="", query="q"))
    assert hit is not None
    assert hit.answer == {"items": []}


def test_brave_over_its_credit_answers_402_which_is_the_quota_state():
    """Brave answers 402 Payment Required once the month's free credit is
    spent: the source is out of budget, not broken, so it is the Quota
    state 429 already is (spec 3 section 6a)."""
    backend = brave_backend(api_key="k",
                            get=lambda *a, **k: _FakeResponse(status_code=402, text="no credit"))
    with pytest.raises(QuotaExhausted) as e:
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))
    assert e.value.source == "brave"


def test_a_402_is_the_quota_state_for_every_http_image_source():
    # the rule lives on the shared backend, not on brave alone
    backend = openverse_backend(get=lambda *a, **k: _FakeResponse(status_code=402, text="pay"))
    with pytest.raises(QuotaExhausted) as e:
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))
    assert e.value.source == "openverse"


def test_brave_a_429_is_still_the_quota_state():
    backend = brave_backend(api_key="k",
                            get=lambda *a, **k: _FakeResponse(status_code=429, text="throttled"))
    with pytest.raises(QuotaExhausted) as e:
        backend.fetch(Question(subject="rice", provides="picture", params={"query": "q"}))
    assert e.value.source == "brave"


# --- FetchBackend: url -> bytes -> media store, pictures and recordings ---

class _Media:
    def __init__(self):
        self.written, self.images = [], []

    def write(self, data, ext):
        self.written.append((data, ext))
        return "sha-" + ext

    def add_image(self, data, ext):
        self.images.append((data, ext))
        return ImageIngestResult(sha="img-" + ext, ext=ext)


def test_fetch_backend_key_is_the_url():
    backend = FetchBackend(media=None, fetcher=lambda url: (b"x", "jpg"))
    key = backend.cache_key(Question(subject="s", provides="picture-bytes",
                                     params={"url": "https://x/y.jpg"}))
    assert key.encode() == "::https://x/y.jpg"


def test_fetch_backend_stores_recording_bytes_raw_and_echoes_params():
    media = _Media()
    b = FetchBackend(media=media, fetcher=lambda url: (b"mp3bytes", "mp3"))
    q = Question(subject="w", provides="recording-bytes",
                 params={"url": "https://apifree.forvo.com/x.mp3", "speaker": "krisflyer",
                        "speaker_kind": "native"})
    assert b.cache_key(q).encode() == "::https://apifree.forvo.com/x.mp3"
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


def test_tool_fetcher_raises_a_typed_refusal_from_the_tools_json_line():
    import subprocess as sp
    from thai_syllabus.transport import FetchRefused

    def runner(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, '{"refused":"content-type","detail":"content-type \\"application/json\\" is not allowed"}\n',
                                   "audiofetch: refused ...")

    fetcher = tool_fetcher("audiofetch", runner=runner)
    with pytest.raises(FetchRefused) as err:
        fetcher("https://x/expired.mp3")
    assert err.value.reason == "content-type" and err.value.served is True
    assert "application/json" in err.value.detail


def test_tool_fetcher_parses_the_body_field_off_the_refusal_line():
    """Task 7 brief: a content-type refusal's body (Forvo's own daily
    limit statement served at an expired mp3 url) rides the tool's JSON
    refusal line as "body" so attempts.py can recognize it (spec 3
    section 6a)."""
    import subprocess as sp
    from thai_syllabus.transport import FetchRefused

    def runner(cmd, **kwargs):
        return sp.CompletedProcess(
            cmd, 1,
            '{"refused":"content-type","detail":"content-type \\"application/json\\" is not allowed",'
            '"body":"[\\"Limit\\/day reached.\\"]"}\n',
            "audiofetch: refused ...")

    fetcher = tool_fetcher("audiofetch", runner=runner)
    with pytest.raises(FetchRefused) as err:
        fetcher("https://x/expired.mp3")
    assert err.value.body == '["Limit/day reached."]'


def test_tool_fetcher_defaults_body_to_empty_when_absent():
    import subprocess as sp
    from thai_syllabus.transport import FetchRefused

    def runner(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, '{"refused":"http","detail":"http 404"}\n', "")

    with pytest.raises(FetchRefused) as err:
        tool_fetcher("imgfetch", runner=runner)("https://x/y.jpg")
    assert err.value.body == ""


def test_tool_fetcher_wire_refusal_is_not_served():
    import subprocess as sp
    from thai_syllabus.transport import FetchRefused

    def runner(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, '{"refused":"wire","detail":"request failed: timeout"}\n', "")

    with pytest.raises(FetchRefused) as err:
        tool_fetcher("imgfetch", runner=runner)("https://x/y.jpg")
    assert err.value.reason == "wire" and err.value.served is False


def test_tool_fetcher_without_a_json_line_raises_a_plain_transport_error():
    import subprocess as sp
    from thai_syllabus.transport import FetchRefused

    def runner(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, "", "imgfetch: refused: not an image")

    with pytest.raises(TransportError) as err:
        tool_fetcher("imgfetch", runner=runner)("https://x/y.jpg")
    assert not isinstance(err.value, FetchRefused)


def test_fetch_backend_reports_an_undecodable_image_as_a_format_refusal(tmp_path):
    from thai_syllabus.transport import FetchRefused
    backend = FetchBackend(media=MediaStore(tmp_path / "media"),
                           fetcher=lambda url: (b"not an image", "jpg"))
    with pytest.raises(FetchRefused) as err:
        backend.fetch(Question(subject="w", provides="picture-bytes", params={"url": "https://x/a.jpg"},
                               kind="picture", subject_kind="word"))
    assert err.value.reason == "format" and err.value.served is True


# --- forvo: never re-asked, key = forvo:WORD ----------------------------

def test_forvo_cache_key_is_forvo_colon_word():
    backend = ForvoBackend(api_key="k")
    key = backend.cache_key(Question(subject="ไก่", provides="recording"))  # chicken
    assert key.encode() == "forvo::ไก่"


def test_forvo_fetch_returns_items_and_a_transport_error_on_bad_status():
    ok = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                      _FakeResponse(json_data={"items": [{"id": 1}]}))
    answer = ok.fetch(Question(subject="ไก่", provides="recording"))
    assert answer.items == ({"id": 1},)

    bad = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                       _FakeResponse(status_code=500))
    with pytest.raises(TransportError):
        bad.fetch(Question(subject="ไก่", provides="recording"))


@pytest.mark.parametrize("payload", [["Limit/day reached."], {"error": "x"}, "text", {"items": None}])
def test_forvo_a_200_body_without_items_is_a_transport_error(payload):
    backend = ForvoBackend(api_key="k", get=lambda url, timeout=None: _FakeResponse(json_data=payload))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="ไก่", provides="recording"))   # ไก่: chicken


def test_forvo_a_200_body_that_is_not_json_is_a_transport_error():
    class _Bad(_FakeResponse):
        def json(self):
            raise ValueError("not json")

    backend = ForvoBackend(api_key="k", get=lambda url, timeout=None: _Bad(status_code=200))
    with pytest.raises(TransportError):
        backend.fetch(Question(subject="ไก่", provides="recording"))   # ไก่: chicken


def test_forvo_400_with_limit_day_reached_raises_quota_exhausted():
    backend = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                           _FakeResponse(status_code=400, json_data=["Limit/day reached."]))
    with pytest.raises(QuotaExhausted) as err:
        backend.fetch(Question(subject="ไก่", provides="recording"))   # ไก่: chicken
    assert err.value.source == "forvo"


@pytest.mark.parametrize("payload", [
    ["Limit/day reached.", "another entry"],  # not length 1
    ["some other message"],                    # wrong text
    {"error": "bad request"},                  # not a list at all
    "Limit/day reached.",                      # a bare string, not a list
])
def test_forvo_400_with_another_body_stays_a_plain_transport_error(payload):
    backend = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                           _FakeResponse(status_code=400, json_data=payload))
    with pytest.raises(TransportError) as err:
        backend.fetch(Question(subject="ไก่", provides="recording"))   # ไก่: chicken
    assert not isinstance(err.value, QuotaExhausted)


# --- forvo_limit_body: the predicate factored for attempts.py's use on a
# fetch refusal's parsed body (task 7 brief, spec 3 section 6a) ------------

def test_forvo_limit_body_recognizes_the_lookup_and_download_bodies_alike():
    assert forvo_limit_body(["Limit/day reached."]) is True


@pytest.mark.parametrize("payload", [
    ["Limit/day reached.", "another entry"],
    ["some other message"],
    {"error": "bad request"},
    "Limit/day reached.",
    None,
])
def test_forvo_limit_body_rejects_every_other_shape(payload):
    assert forvo_limit_body(payload) is False


def test_forvo_wire_failure_redacts_the_api_key_from_the_message():
    def raise_it(url, timeout=None):
        raise requests.ConnectionError(
            "GET https://apifree.forvo.com/key/SECRET123/format/json/"
            "action/word-pronunciations/word/ไก่")

    backend = ForvoBackend(api_key="SECRET123", get=raise_it)
    with pytest.raises(TransportError) as err:
        backend.fetch(Question(subject="ไก่", provides="recording"))   # ไก่: chicken
    assert "SECRET123" not in str(err.value)
    assert "***" in str(err.value)


def test_forvo_wire_failure_has_no_chained_cause_to_leak_the_key():
    def raise_it(url, timeout=None):
        raise requests.ConnectionError(
            "GET https://apifree.forvo.com/key/SECRET123/format/json/"
            "action/word-pronunciations/word/ไก่")

    backend = ForvoBackend(api_key="SECRET123", get=raise_it)
    with pytest.raises(TransportError) as err:
        backend.fetch(Question(subject="ไก่", provides="recording"))   # ไก่: chicken
    assert err.value.__cause__ is None
    assert "SECRET123" not in repr(err.value)


def test_redact_replaces_every_occurrence_and_is_a_noop_for_an_empty_secret():
    assert _redact("key=ABC and again ABC", "ABC") == "key=*** and again ***"
    assert _redact("nothing secret here", "") == "nothing secret here"


def test_forvo_empty_result_is_still_a_valid_answer(db):
    # "never re-asked (the answer outlives the quota)" -- an empty lookup
    # must cache too, via the generic Provider empty-answer-is-cached rule.
    backend = ForvoBackend(api_key="k", get=lambda url, timeout=None:
                           _FakeResponse(json_data={"items": []}))
    provider = Provider(record=db, cache=db, backends={"forvo": backend})
    calls_before = backend.get
    provider.ask("forvo", Question(subject="หมา", provides="recording"))  # dog
    hit = db.latest("provide", "forvo", ProvideKey(source="forvo", kind="", query="หมา"))
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
    assert key.encode().startswith("tts:")
    assert key.kind == pick_voice("subj-1", voices)
    assert key.query == sha("ผมกินข้าว")  # I eat rice


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


def test_voices_for_is_removed_the_pool_comes_from_providers_yaml_only():
    import thai_syllabus.tts as tts_mod
    assert not hasattr(tts_mod, "voices_for")


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
    def __init__(self, text="drafted sentence", raises=None, input_tokens=0, output_tokens=0):
        self.text = text
        self.raises = raises
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        return Completion(text=self.text, input_tokens=self.input_tokens,
                          output_tokens=self.output_tokens)


def test_llm_cache_key_is_stable_for_the_same_prompt():
    backend = LlmBackend(producer="sentence-drafter", model="claude-opus-5",
                         transport=_FakeTransport())
    q = Question(subject="s", provides="sentence", params={"prompt": "write a sentence"})
    assert backend.cache_key(q) == backend.cache_key(q)
    assert backend.cache_key(q).encode().startswith("llm:sentence-drafter:claude-opus-5:")


def test_llm_cache_key_changes_when_the_prompt_text_changes():
    backend = LlmBackend(producer="p", model="m", transport=_FakeTransport())
    k1 = backend.cache_key(Question(subject="s", provides="sentence",
                                    params={"prompt": "prompt A"}))
    k2 = backend.cache_key(Question(subject="s", provides="sentence",
                                    params={"prompt": "prompt B"}))
    assert k1 != k2


def test_llm_fetch_delegates_to_the_transport_and_wraps_the_completion():
    transport = _FakeTransport(text="ผมกินข้าว")  # I eat rice
    backend = LlmBackend(producer="p", model="m", transport=transport, quota_cost_per_call=0.002)
    answer = backend.fetch(Question(subject="s", provides="sentence",
                                    params={"prompt": "write a sentence about rice"}))
    assert answer.items == ("ผมกินข้าว",)  # I eat rice
    assert answer.cost == 0.002
    assert transport.prompts == ["write a sentence about rice"]


def test_llm_fetch_prices_the_completions_tokens_when_a_price_is_configured():
    """Spec 3 section 2's cost contract: every Answer carries the cost the
    backend incurred, measured by the backend -- api/batch llm drafting is
    tokens times the providers.yaml price, exactly as JudgeBackend prices a
    verdict. A flat per-call quota cost applies only where no price does
    (the cli transport, which reports no usage)."""
    transport = _FakeTransport(text="drafted", input_tokens=1_000_000, output_tokens=500_000)
    backend = LlmBackend(producer="p", model="m", transport=transport,
                         price=Price(2.0, 10.0), quota_cost_per_call=1.0)
    answer = backend.fetch(Question(subject="s", provides="sentence",
                                    params={"prompt": "draft"}))
    assert answer.cost == pytest.approx(2.0 + 5.0)


def test_llm_transport_error_propagates_uncached(db):
    transport = _FakeTransport(raises=TransportError("cli failed"))
    backend = LlmBackend(producer="p", model="m", transport=transport)
    provider = Provider(record=db, cache=db, backends={"llm": backend})
    with pytest.raises(TransportError):
        provider.ask("llm", Question(subject="s", provides="sentence",
                                     params={"prompt": "x"}))


def test_llm_fetch_raises_when_recognize_rejects_the_completion_and_caches_nothing(db):
    """Spec 3 r10 section 2: an LlmBackend appends only a completion the
    producer recognizes. `fetch` raises TransportError for the rest, so
    the provider's ask() caches no row."""
    transport = _FakeTransport(text="I cannot draft this.")
    backend = LlmBackend(producer="p", model="m", transport=transport, recognize=lambda t: False)
    provider = Provider(record=db, cache=db, backends={"llm": backend})
    q = Question(subject="s", provides="sentence", params={"prompt": "x"})
    with pytest.raises(TransportError, match="recognizable answer"):
        provider.ask("llm", q)
    assert db.latest("provide", "llm", backend.cache_key(q)) is None
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
    assert key.encode() == "pairs:tone:mid-low:2026-09-01"


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


def test_a_provider_miss_is_not_a_hit_and_the_re_read_is(db):
    backend = _FakeBackend(items=({"url": "u"},))
    provider = Provider(record=db, cache=db, backends={"openverse": backend})
    q = Question(subject="s", provides="picture", params={"query": "orange"})
    assert provider.ask("openverse", q).hit is False
    assert provider.ask("openverse", q).hit is True
