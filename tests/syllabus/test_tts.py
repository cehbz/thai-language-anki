"""Tests for tts.py's GoogleTts.synthesize (spec 3 r10 section 6a): a 4xx
other than 429 is a definitive refusal of this text/voice; 429, 5xx, and
a 200 body without audioContent stay transient.
"""
import pytest
import requests

from thai_syllabus.transport import SynthesisRefused, TransportError
from thai_syllabus.tts import GoogleTts


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.mark.parametrize("status", [400, 403, 404, 413])
def test_google_tts_refuses_this_text_or_voice_on_a_4xx(status):
    tts = GoogleTts(api_key="k", http_post=lambda url, json, headers, timeout: _Resp(status, text="voice not found"))
    with pytest.raises(SynthesisRefused):
        tts.synthesize("สวัสดี", "th-TH-Chirp3-HD-Gone")   # สวัสดี: hello


@pytest.mark.parametrize("status", [429, 500, 503])
def test_google_tts_is_transient_on_429_and_5xx(status):
    tts = GoogleTts(api_key="k", http_post=lambda url, json, headers, timeout: _Resp(status))
    with pytest.raises(TransportError) as err:
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello
    assert not isinstance(err.value, SynthesisRefused)


def test_google_tts_a_200_without_audio_content_is_transient():
    tts = GoogleTts(api_key="k", http_post=lambda url, json, headers, timeout: _Resp(200, {"error": "x"}))
    with pytest.raises(TransportError) as err:
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello
    assert not isinstance(err.value, SynthesisRefused)


def test_google_tts_wraps_a_wire_failure_in_transport_error():
    def raise_timeout(url, json, headers, timeout):
        raise requests.exceptions.ReadTimeout("timed out")

    tts = GoogleTts(api_key="k", http_post=raise_timeout)
    with pytest.raises(TransportError):
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello


def test_google_tts_synthesizes_on_200_with_audio_content():
    import base64
    audio = base64.b64encode(b"mp3-bytes").decode()
    tts = GoogleTts(api_key="k", http_post=lambda url, json, headers, timeout: _Resp(200, {"audioContent": audio}))
    assert tts.synthesize("สวัสดี", "v") == b"mp3-bytes"   # สวัสดี: hello


# --- spec 3 section 2's cost/secrets contract: the key rides a header, --
# --- never the url, and never leaks through a wire-failure message ------

def test_google_tts_sends_the_api_key_as_a_header_not_a_query_param():
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append((url, headers))
        return _Resp(200, {"audioContent": "eA=="})

    tts = GoogleTts(api_key="SECRET123", http_post=fake_post)
    tts.synthesize("สวัสดี", "v")   # สวัสดี: hello

    url, headers = calls[0]
    assert "SECRET123" not in url and "key=" not in url
    assert headers["X-Goog-Api-Key"] == "SECRET123"


def test_google_tts_wire_failure_redacts_the_api_key_from_the_message():
    def raise_it(url, json, headers, timeout):
        raise requests.ConnectionError(
            "POST https://texttospeech.googleapis.com/v1/text:synthesize"
            "?key=SECRET123 failed")

    tts = GoogleTts(api_key="SECRET123", http_post=raise_it)
    with pytest.raises(TransportError) as err:
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello
    assert "SECRET123" not in str(err.value)
    assert "***" in str(err.value)


def test_google_tts_wire_failure_has_no_chained_cause_to_leak_the_key():
    def raise_it(url, json, headers, timeout):
        raise requests.ConnectionError(
            "POST https://texttospeech.googleapis.com/v1/text:synthesize"
            "?key=SECRET123 failed")

    tts = GoogleTts(api_key="SECRET123", http_post=raise_it)
    with pytest.raises(TransportError) as err:
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello
    assert err.value.__cause__ is None
    assert "SECRET123" not in repr(err.value)


def test_google_tts_missing_audio_content_has_no_chained_cause_to_leak_the_key():
    tts = GoogleTts(api_key="SECRET123",
                     http_post=lambda url, json, headers, timeout: _Resp(200, {"error": "x"}))
    with pytest.raises(TransportError) as err:
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello
    assert err.value.__cause__ is None
    assert "SECRET123" not in repr(err.value)
