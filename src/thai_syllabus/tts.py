"""Voice pools and Google TTS synthesis, ported out of
thai_deck_gen/media/tts.py (spec 3 deliverable 2) -- decoupled from that
module's Deck/Manifest/ProducerResult machinery, since this package
imports nothing out of thai_deck_gen or thai_deck_eval by design (see
__init__.py). Consumed by provider.py's TtsBackend.

The male/production rule ("a text filling any productive slot gets native
audio ... receptive-only texts may stay TTS", and among TTS voices,
"production draws male only") is a CALLER decision, not enforced here:
curated.py's providers.yaml loader supplies the male/female pools
(defaulting to MALE_VOICES/FEMALE_VOICES below) and wiring.py/run.py pick
which pool a given ask draws from.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .transport import TransportError

# Google's Thai voices. A deck whose every listening card speaks in one
# synthetic voice teaches that voice, so sentences are spread across them.
# Roster from the live voices API 2026-09-02 (ported verbatim).
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
        import base64
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}"
        body = {
            "input": {"text": text},
            "voice": {"languageCode": "th-TH", "name": voice},
            "audioConfig": {"audioEncoding": "MP3"},
        }
        resp = self.http_post(url, json=body, timeout=30)
        if resp.status_code != 200:
            raise TransportError(f"google tts failed with {resp.status_code}: {resp.text}")
        return base64.b64decode(resp.json()["audioContent"])
