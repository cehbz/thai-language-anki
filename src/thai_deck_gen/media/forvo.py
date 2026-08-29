from pathlib import Path
from urllib.parse import quote
import requests
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.ffmpeg import AudioError, duration_ok, normalize_audio
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import AudioNeed
from thai_deck_gen.producers import ProducerResult

class ForvoQuotaExceeded(Exception):
    """Forvo refused the request for quota reasons; further calls are pointless."""

class ForvoClient:
    def __init__(self, api_key: str, http_get=requests.get):
        self.api_key = api_key
        self.http_get = http_get

    def pronunciations(self, word: str) -> list[dict]:
        url = (f"https://apifree.forvo.com/key/{self.api_key}/format/json/"
               f"action/word-pronunciations/word/{quote(word)}/language/th")
        resp = self.http_get(url, timeout=30)
        if resp.status_code in (403, 429):
            raise ForvoQuotaExceeded(
                f"forvo returned {resp.status_code}: {getattr(resp, 'text', '')}")
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
                max_speakers: int = 3, limit: int | None = None,
                checkpoint=None, checkpoint_every: int = 25) -> ProducerResult:
    """Fill native-tier audio from Forvo.

    `limit` caps API lookups for this run (the free tier is a daily request
    quota); needs beyond the cap, and needs after a quota refusal, are left
    pending rather than blocked, so the next run picks them up.

    `checkpoint` (typically write_deck) is called every `checkpoint_every`
    filled needs and once at the end: a long run killed mid-flight keeps the
    audio it already paid for.
    """
    result = ProducerResult()
    lookups = 0
    for need in needs:
        if limit is not None and lookups >= limit:
            print(f"  forvo: stopping at the {limit}-lookup limit; "
                  f"{len(needs) - lookups} need(s) left pending")
            break
        try:
            lookups += 1
            items = client.pronunciations(need.text)
            if not items:
                result.blocked.append(need.text)
                continue

            target = _find_note(deck, need)
            dst_path = deck.root / "media" / need.path
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            wanted = max_speakers if need.family == "minimal_pair" else 1
            accepted = []
            for item in items:
                if len(accepted) == wanted:
                    break
                url = item["pathmp3"]
                raw = client.download(url)
                k = len(accepted)
                if k == 0:
                    dst = dst_path
                    ref = f"media/{need.path}"
                else:
                    variant_name = f"{dst_path.stem}_s{k}{dst_path.suffix}"
                    dst = dst_path.with_name(variant_name)
                    ref = f"media/{Path(need.path).with_name(variant_name)}"

                normalize_audio(raw, dst)
                # Forvo clips are user-uploaded: discard anything outside a
                # plausible single-word duration rather than shipping it.
                if not duration_ok(dst):
                    dst.unlink(missing_ok=True)
                    continue

                manifest.record(MediaEntry(
                    file=ref, channel="forvo", origin=url,
                    speaker=f"forvo:{item['username']}", fetched=today))
                accepted.append(item)

            if not accepted:
                result.blocked.append(need.text)
                continue

            target.audio.speaker = f"forvo:{accepted[0]['username']}"
            target.audio.source = "native"
            result.changed += 1
            if checkpoint and result.changed % checkpoint_every == 0:
                checkpoint()
        except ForvoQuotaExceeded as exc:
            print(f"  forvo: {exc}; {len(needs) - lookups} need(s) left pending")
            break
        except (AudioError, requests.RequestException):
            result.blocked.append(need.text)

    if checkpoint and result.changed:
        checkpoint()
    return result
