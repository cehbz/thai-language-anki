import base64
from pathlib import Path
from typing import Protocol
import requests
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.ffmpeg import AudioError, normalize_audio
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import AudioNeed
from thai_deck_gen.producers import ProducerResult

class Tts(Protocol):
    voice: str
    def synthesize(self, text: str) -> bytes: ...

class GoogleTts:
    def __init__(self, api_key: str, voice: str = "th-TH-Neural2-C",
                http_post=requests.post):
        self.api_key = api_key
        self.voice = voice
        self.http_post = http_post

    def synthesize(self, text: str) -> bytes:
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={self.api_key}"
        body = {
            "input": {"text": text},
            "voice": {"languageCode": "th-TH", "name": self.voice},
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
            tts: Tts, today: str) -> ProducerResult:
    result = ProducerResult()
    for need in needs:
        if need.native_required or need.family != "sentence":
            continue

        try:
            note = _find_note(deck, need)
            raw = tts.synthesize(need.text)
            dst = deck.root / "media" / need.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            normalize_audio(raw, dst)

            manifest.record(MediaEntry(
                file=f"media/{need.path}", channel="tts",
                origin=f"google-tts:{tts.voice}",
                speaker=f"tts:{tts.voice}", fetched=today))
            note.audio.source = "tts"
            note.audio.speaker = f"tts:{tts.voice}"
            result.changed += 1
        except (AudioError, requests.RequestException):
            result.blocked.append(need.note_id)

    return result
