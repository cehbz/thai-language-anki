from pathlib import Path
from urllib.parse import quote
import requests
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.ffmpeg import normalize_audio
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import AudioNeed
from thai_deck_gen.producers import ProducerResult

class ForvoClient:
    def __init__(self, api_key: str, http_get=requests.get):
        self.api_key = api_key
        self.http_get = http_get

    def pronunciations(self, word: str) -> list[dict]:
        url = (f"https://apifree.forvo.com/key/{self.api_key}/format/json/"
               f"action/word-pronunciations/word/{quote(word)}/language/th")
        resp = self.http_get(url, timeout=30)
        if resp.status_code != 200:
            return []
        items = resp.json().get("items", [])
        return sorted(items, key=lambda it: it.get("rate", 0), reverse=True)

    def download(self, url: str) -> bytes:
        resp = self.http_get(url, timeout=30)
        return resp.content

def _find_note(deck: Deck, need: AudioNeed):
    for family, note in deck.all_notes():
        if family != need.family or note.id != need.note_id:
            continue
        if need.member_index is not None:
            return note.members[need.member_index]
        return note

def fetch_forvo(needs: list[AudioNeed], deck: Deck, manifest: Manifest,
                client: ForvoClient, today: str,
                max_speakers: int = 3) -> ProducerResult:
    result = ProducerResult()
    for need in needs:
        items = client.pronunciations(need.text)
        if not items:
            result.blocked.append(need.text)
            continue

        target = _find_note(deck, need)
        dst_path = deck.root / "media" / need.path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if need.family == "minimal_pair":
            picks = items[:max_speakers]
        else:
            picks = items[:1]

        for k, item in enumerate(picks):
            username = item["username"]
            url = item["pathmp3"]
            raw = client.download(url)
            if k == 0:
                dst = dst_path
                ref = f"media/{need.path}"
            else:
                stem = dst_path.stem
                variant_name = f"{stem}_s{k}{dst_path.suffix}"
                dst = dst_path.with_name(variant_name)
                ref = f"media/{Path(need.path).with_name(variant_name)}"

            normalize_audio(raw, dst)
            manifest.record(MediaEntry(
                file=ref, channel="forvo", origin=url,
                speaker=f"forvo:{username}", fetched=today))

        target.audio.speaker = f"forvo:{picks[0]['username']}"
        target.audio.source = "native"
        result.changed += 1

    return result
