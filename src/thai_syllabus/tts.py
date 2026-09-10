"""Voice pools and Google TTS synthesis (spec 3 deliverable 2), consumed
by provider.py's TtsBackend.

Which pool an ask draws from is the caller's: curated.py's providers.yaml
loader supplies the male/female pools (defaulting to MALE_VOICES/
FEMALE_VOICES below) and wiring.py/run.py pick between them.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .provider import _redact
from .transport import SynthesisRefused, TransportError

# Google's Thai voices, from the live voices API 2026-09-02. Sentences
# spread across the pool, so no one synthetic voice is what gets taught.
_CHIRP = "th-TH-Chirp3-HD-"
MALE_VOICES = [_CHIRP + n for n in [
    "Achird", "Algenib", "Algieba", "Alnilam", "Charon", "Enceladus",
    "Fenrir", "Iapetus", "Orus", "Puck", "Rasalgethi", "Sadachbia",
    "Sadaltager", "Schedar", "Umbriel", "Zubenelgenubi"]]
FEMALE_VOICES = [_CHIRP + n for n in [
    "Achernar", "Aoede", "Autonoe", "Callirrhoe", "Despina", "Erinome",
    "Kore", "Laomedeia", "Leda", "Pulcherrima", "Sulafat",
    "Vindemiatrix", "Zephyr"]] + ["th-TH-Neural2-C", "th-TH-Standard-A"]
THAI_VOICES = MALE_VOICES + FEMALE_VOICES


def pick_voice(subject: str, voices: list[str]) -> str:
    """Deterministic per subject, so a re-run never re-synthesizes (the
    same subject always lands on the same voice, hence the same cache
    key -- tts's "deterministic; never re-asked" policy).
    """
    digest = hashlib.sha256(subject.encode()).digest()
    return voices[digest[0] % len(voices)]


class Tts(Protocol):
    def synthesize(self, text: str, voice: str) -> bytes: ...


@dataclass
class GoogleTts:
    api_key: str
    http_post: Callable | None = field(default=None)

    def __post_init__(self) -> None:
        if self.http_post is None:
            import requests
            self.http_post = requests.post

    def synthesize(self, text: str, voice: str) -> bytes:
        """Spec 3 section 2's cost/secrets contract: the key rides the
        `X-Goog-Api-Key` header, never the url, and every failure message
        is redacted before it can carry the key into a log.
        """
        import base64
        url = "https://texttospeech.googleapis.com/v1/text:synthesize"
        headers = {"X-Goog-Api-Key": self.api_key}
        body = {
            "input": {"text": text},
            "voice": {"languageCode": "th-TH", "name": voice},
            "audioConfig": {"audioEncoding": "MP3"},
        }
        import requests
        try:
            resp = self.http_post(url, json=body, headers=headers, timeout=30)
        except requests.RequestException as e:
            # from None: the chained cause would still carry the raw,
            # unredacted url/key, printed by any full traceback render.
            raise TransportError(f"google tts failed: {_redact(str(e), self.api_key)}") from None
        if resp.status_code != 200:
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise SynthesisRefused(
                    f"google tts refused {voice!r}: {resp.status_code} "
                    f"{_redact(resp.text[:200], self.api_key)}")
            raise TransportError(
                f"google tts failed with {resp.status_code}: "
                f"{_redact(resp.text[:200], self.api_key)}")
        try:
            content = resp.json()["audioContent"]
        except (ValueError, KeyError, TypeError) as e:
            # from None: keep a stray key out of any full traceback render,
            # same as the other redacted sites in this method.
            raise TransportError(
                f"google tts answered without audioContent: {_redact(str(e), self.api_key)}") from None
        return base64.b64decode(content)
