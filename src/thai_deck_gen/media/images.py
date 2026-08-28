import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote
import requests
import yaml
from PIL import Image
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import ImageNeed
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps

_SEARCH_CHANNELS = ("openverse", "wikimedia")

class ImageError(Exception):
    """Raised when AI image generation fails"""
    pass

@dataclass
class ImageCandidate:
    url: str
    source: str
    license: str | None

class ImageGen(Protocol):
    def generate(self, prompt: str) -> bytes: ...

class OpenAiImageGen:
    def __init__(self, api_key: str, http_post=requests.post):
        self.api_key = api_key
        self.http_post = http_post

    def generate(self, prompt: str) -> bytes:
        import base64
        url = "https://api.openai.com/v1/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {"model": "gpt-image-1", "prompt": prompt}
        resp = self.http_post(url, json=body, headers=headers, timeout=60)
        if resp.status_code != 200:
            detail = getattr(resp, "text", "")
            raise ImageError(f"gpt-image-1 failed with {resp.status_code}: {detail}")
        b64 = resp.json()["data"][0]["b64_json"]
        return base64.b64decode(b64)

def openverse_search(term: str, http_get) -> list[ImageCandidate]:
    url = f"https://api.openverse.org/v1/images/?q={quote(term)}&page_size=5"
    resp = http_get(url, timeout=30)
    if getattr(resp, "status_code", 200) != 200:
        return []
    data = resp.json() or {}
    return [ImageCandidate(url=r["url"], source="openverse", license=r.get("license"))
            for r in data.get("results", []) if r.get("url")]

def wikimedia_search(term: str, http_get) -> list[ImageCandidate]:
    url = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search"
           f"&gsrsearch={quote(term)}&gsrnamespace=6&prop=imageinfo&iiprop=url&format=json")
    resp = http_get(url, timeout=30)
    if getattr(resp, "status_code", 200) != 200:
        return []
    data = resp.json() or {}
    pages = data.get("query", {}).get("pages", {})
    candidates = []
    for page in pages.values():
        for info in page.get("imageinfo", []):
            if info.get("url"):
                candidates.append(ImageCandidate(url=info["url"], source="wikimedia",
                                                  license=None))
    return candidates

def downscale(raw: bytes, max_px: int = 600) -> bytes:
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((max_px, max_px))
    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()

def _try_candidate(candidate: ImageCandidate, http_get) -> bytes | None:
    # Trying several search candidates is the whole point of the fallback
    # chain: any failure (network, 404, unparseable image) just means "try
    # the next candidate", surfaced overall via ProducerResult.blocked.
    try:
        resp = http_get(candidate.url, timeout=30)
        if getattr(resp, "status_code", 200) != 200:
            return None
        return downscale(resp.content)
    except Exception:
        return None

def _write_media(deck: Deck, path: str, data: bytes) -> None:
    dst = deck.root / "media" / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)

def _search_fill(need: ImageNeed, deck: Deck, manifest: Manifest, http_get,
                 today: str, result: ProducerResult) -> None:
    queries = [need.term] + ([need.gloss] if need.gloss else [])
    for query in queries:
        for search_fn in (openverse_search, wikimedia_search):
            for candidate in search_fn(query, http_get):
                image = _try_candidate(candidate, http_get)
                if image is None:
                    continue
                _write_media(deck, need.path, image)
                manifest.record(MediaEntry(
                    file=f"media/{need.path}", channel=candidate.source,
                    origin=candidate.url, license=candidate.license, fetched=today))
                result.changed += 1
                return
    result.blocked.append(need.note_id)

def _queue_review(deck_root: Path, note_id: str, term: str, tried: list[str]) -> None:
    path = Path(deck_root) / "work" / "image_review.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    items = [it for it in (data or {}).get("items", []) if it["note_id"] != note_id]
    items.append({"note_id": note_id, "term": term, "tried": tried})
    path.write_text(yaml.safe_dump({"items": items}, allow_unicode=True, sort_keys=False))

def _escalate(need: ImageNeed, deck: Deck, manifest: Manifest, imagegen: ImageGen,
             today: str, result: ProducerResult) -> None:
    subject = need.gloss or need.term
    prompt = f"simple clear illustration of {subject}, Thai cultural context, no text"
    try:
        raw = imagegen.generate(prompt)
    except ImageError:
        _queue_review(deck.root, need.note_id, need.term, ["ai"])
        result.blocked.append(need.note_id)
        return
    _write_media(deck, need.path, downscale(raw))
    manifest.record(MediaEntry(
        file=f"media/{need.path}", channel="ai", origin="gpt-image-1", fetched=today))
    result.changed += 1

def flagged_image_note_ids(gaps: Gaps) -> set[str]:
    return {f.note_id for f in gaps.findings_for("judge/")
            if f.note_id and "image" in f.rule.lower()}

def fill_images(needs: list[ImageNeed], gaps: Gaps, deck: Deck,
                manifest: Manifest, ctx, today: str) -> ProducerResult:
    result = ProducerResult()
    flagged = flagged_image_note_ids(gaps)

    for need in needs:
        channel = manifest.channel_of(f"media/{need.path}")

        if need.note_id in flagged:
            if channel in _SEARCH_CHANNELS and ctx.imagegen is not None:
                _escalate(need, deck, manifest, ctx.imagegen, today, result)
            else:
                _queue_review(deck.root, need.note_id, need.term,
                              [channel] if channel else [])
                result.blocked.append(need.note_id)
            continue

        _search_fill(need, deck, manifest, ctx.http_get, today, result)

    return result
