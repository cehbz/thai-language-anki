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
    tts = GoogleTts(api_key="k", http_post=lambda url, json, timeout: _Resp(status, text="voice not found"))
    with pytest.raises(SynthesisRefused):
        tts.synthesize("สวัสดี", "th-TH-Chirp3-HD-Gone")   # สวัสดี: hello


@pytest.mark.parametrize("status", [429, 500, 503])
def test_google_tts_is_transient_on_429_and_5xx(status):
    tts = GoogleTts(api_key="k", http_post=lambda url, json, timeout: _Resp(status))
    with pytest.raises(TransportError) as err:
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello
    assert not isinstance(err.value, SynthesisRefused)


def test_google_tts_a_200_without_audio_content_is_transient():
    tts = GoogleTts(api_key="k", http_post=lambda url, json, timeout: _Resp(200, {"error": "x"}))
    with pytest.raises(TransportError) as err:
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello
    assert not isinstance(err.value, SynthesisRefused)


def test_google_tts_wraps_a_wire_failure_in_transport_error():
    def raise_timeout(url, json, timeout):
        raise requests.exceptions.ReadTimeout("timed out")

    tts = GoogleTts(api_key="k", http_post=raise_timeout)
    with pytest.raises(TransportError):
        tts.synthesize("สวัสดี", "v")   # สวัสดี: hello


def test_google_tts_synthesizes_on_200_with_audio_content():
    import base64
    audio = base64.b64encode(b"mp3-bytes").decode()
    tts = GoogleTts(api_key="k", http_post=lambda url, json, timeout: _Resp(200, {"audioContent": audio}))
    assert tts.synthesize("สวัสดี", "v") == b"mp3-bytes"   # สวัสดี: hello
