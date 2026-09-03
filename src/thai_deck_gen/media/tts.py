import base64
import hashlib
from pathlib import Path
from typing import Protocol
import requests
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.ffmpeg import AudioError, normalize_audio
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import AudioNeed
from thai_deck_gen.producers import ProducerResult

# Google's Thai voices. A deck whose every listening card speaks in one
# synthetic voice teaches that voice, so sentences are spread across them.
# Production cards model the learner's own register (male); comprehension
# draws from the full mixed pool. Roster from the live voices API 2026-09-02.
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


def voices_for(usage: str) -> list[str]:
    return MALE_VOICES if usage == "production" else THAI_VOICES


def pick_voice(note_id: str, voices: list[str]) -> str:
    """Deterministic per note, so a re-run never re-synthesizes."""
    digest = hashlib.sha256(note_id.encode()).digest()
    return voices[digest[0] % len(voices)]


class Tts(Protocol):
    voice: str
    def synthesize(self, text: str, voice: str | None = None) -> bytes: ...

class GoogleTts:
    def __init__(self, api_key: str, voice: str = "th-TH-Neural2-C",
                http_post=requests.post):
        self.api_key = api_key
        self.voice = voice
        self.http_post = http_post

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}"
        body = {
            "input": {"text": text},
            "voice": {"languageCode": "th-TH", "name": voice or self.voice},
            "audioConfig": {"audioEncoding": "MP3"},
        }
        resp = self.http_post(url, json=body, timeout=30)
        if resp.status_code != 200:
            raise AudioError(f"google tts failed with {resp.status_code}: {resp.text}")
        return base64.b64decode(resp.json()["audioContent"])

def _find_note(deck: Deck, need: AudioNeed):
    for family, note in deck.all_notes():
        if family == need.family and note.id == need.note_id:
            return note

def fill_tts(needs: list[AudioNeed], deck: Deck, manifest: Manifest,
            tts: Tts, today: str, voices: list[str] | None = None) -> ProducerResult:
    result = ProducerResult()
    for need in needs:
        if need.native_required or need.family != "sentence":
            continue

        try:
            note = _find_note(deck, need)
            pool = voices if voices else voices_for(getattr(note, "usage", "production"))
            voice = pick_voice(need.note_id, pool)
            raw = tts.synthesize(need.text, voice)
            dst = deck.root / "media" / need.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            normalize_audio(raw, dst)

            manifest.record(MediaEntry(
                file=f"media/{need.path}", channel="tts",
                origin=f"google-tts:{voice}",
                speaker=f"tts:{voice}", fetched=today))
            note.audio.source = "tts"
            note.audio.speaker = f"tts:{voice}"
            result.changed += 1
        except (AudioError, requests.RequestException):
            result.blocked.append(need.note_id)

    return result
