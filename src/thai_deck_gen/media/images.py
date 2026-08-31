import hashlib
import io
from collections import Counter
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote
import requests
import yaml
from PIL import Image
from thai_deck_eval.judge.core import JudgeRequest
from thai_deck_eval.judge.prompts import PICTURE_RULES, build_picture_prompt

def usable_corpora(ctx) -> list[str]:
    """Corpora this run can actually search.

    A library with no key was never searched, so anything memoized while it
    was unusable must not be treated as a settled answer once it is.
    """
    corpora = ["openverse", "wikimedia"]
    if getattr(ctx, "pexels_key", None):
        corpora.append("pexels")
    return sorted(corpora)


def rubric_fingerprint(pexels: bool = False, corpora: list[str] | None = None) -> str:
    """Short hash of the picture rubric.

    An exhausted search is a statement about the rubric and the corpora as
    much as the queries: relaxing what counts as disqualifying text, or
    adding a library nobody had searched, makes yesterday's rejections worth
    reconsidering.
    """
    # Usable sources, not declared ones: a library with no key was never
    # actually searched, so records written without it must not look like
    # records written with it.
    usable = corpora if corpora is not None else (
        sorted(["openverse", "wikimedia"] + (["pexels"] if pexels else [])))
    blob = json.dumps([PICTURE_RULES, usable], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.imgfetch import ImgFetchUnavailable
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import ImageNeed
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps

_SEARCH_CHANNELS = ("openverse", "wikimedia", "pexels")

# Categories whose words are served by posed concept photography rather than
# by a photograph of a thing. Openverse is amateur Flickr: nobody uploads
# "pointing at myself", and everybody uploads their lunch.
CONCEPT_CATEGORIES = {"Pronouns", "Numbers", "Math/Measurements", "Directions",
                      "Time", "Days of the week", "Months", "Society",
                      "Adjectives", "Verbs"}
# Wikimedia's robot policy 403s anonymous default agents; identify the tool and a contact.
SEARCH_HEADERS = {"User-Agent": "thai-deck-gen/1.0 (https://github.com/cehbz/thai-language-anki)",
                  "Accept": "application/json"}

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

def search_reachable(http_get) -> str | None:
    """None when image search works, else why it doesn't.

    Checked once before a long run: the search proxy is an ssh tunnel that
    dies with sleep or a change of network, and without this every note
    simply blocks, one slow failure at a time.
    """
    try:
        resp = http_get("https://api.openverse.org/v1/images/?q=test&page_size=1",
                        timeout=30, headers=SEARCH_HEADERS)
    except Exception as exc:
        return f"image search unreachable: {exc}"
    status = getattr(resp, "status_code", 200)
    if status != 200:
        return f"image search unreachable: openverse returned {status}"
    return None


def pexels_search(term: str, http_get, api_key: str | None = None) -> list[ImageCandidate]:
    """Pexels: curated, deliberately posed stock photography.

    Its licence is free for personal and commercial use but is not CC, so
    the manifest records the channel and the deck's provenance stays honest.
    """
    if not api_key:
        return []
    url = f"https://api.pexels.com/v1/search?query={quote(term)}&per_page=5"
    resp = http_get(url, timeout=30, headers={"Authorization": api_key})
    if getattr(resp, "status_code", 200) != 200:
        return []
    data = resp.json() or {}
    return [ImageCandidate(url=p["src"]["large"], source="pexels",
                           license="pexels")
            for p in data.get("photos", []) if p.get("src", {}).get("large")]


def source_order(note) -> list:
    """Search functions in the order worth trying for this word.

    Abstract words need a photograph somebody staged on purpose; concrete
    and culture-specific ones are better served by amateur photography of
    the actual thing, where a Thai street beats a studio.
    """
    category = getattr(note, "category", "") or ""
    if category in CONCEPT_CATEGORIES:
        return [pexels_search, openverse_search, wikimedia_search]
    return [openverse_search, pexels_search, wikimedia_search]


def openverse_search(term: str, http_get) -> list[ImageCandidate]:
    url = f"https://api.openverse.org/v1/images/?q={quote(term)}&page_size=5"
    resp = http_get(url, timeout=30, headers=SEARCH_HEADERS)
    if getattr(resp, "status_code", 200) != 200:
        return []
    data = resp.json() or {}
    return [ImageCandidate(url=r["url"], source="openverse", license=r.get("license"))
            for r in data.get("results", []) if r.get("url")]

def wikimedia_search(term: str, http_get) -> list[ImageCandidate]:
    url = ("https://commons.wikimedia.org/w/api.php?action=query&generator=search"
           f"&gsrsearch={quote(term)}&gsrnamespace=6&prop=imageinfo&iiprop=url&format=json")
    resp = http_get(url, timeout=30, headers=SEARCH_HEADERS)
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

def _try_candidate(candidate: ImageCandidate, fetcher) -> bytes | None:
    # Trying several search candidates is the whole point of the fallback
    # chain: a refused download or an undecodable image just means "try
    # the next candidate", surfaced overall via ProducerResult.blocked.
    # ImgFetchUnavailable is not a per-candidate condition and propagates.
    raw = fetcher.fetch(candidate.url)
    if raw is None:
        return None
    try:
        return downscale(raw)
    except Exception:
        return None

def _write_media(deck: Deck, path: str, data: bytes) -> None:
    dst = deck.root / "media" / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)

def _head_term(gloss: str) -> str:
    """The searchable core of a learner gloss.

    Glosses carry sense notes and synonyms -- "I (female speaker, or casual
    general)", "orange, mandarin" -- which match nothing in an image index.
    """
    return gloss.split("(")[0].split(",")[0].split(";")[0].strip()


def _queries(need: ImageNeed, hints: dict[str, str]) -> list[str]:
    """Search terms in the order worth trying.

    Both image corpora index English metadata, so the gloss goes first: a Thai
    query matches only the handful of Thai-*captioned* items they hold, which
    are posters and book covers, and that produced both the irrelevant images
    and the ones made of text. The category qualifier separates senses the
    gloss alone conflates ("orange" the fruit from the tabby cat). The Thai
    term still comes last, where it wins on culture-specific words.
    """
    queries = []
    if need.image_query:
        queries.append(need.image_query)
    gloss = _head_term(need.gloss) if need.gloss else ""
    if gloss:
        hint = hints.get(need.category or "")
        if hint:
            queries.append(f"{gloss} {hint}")
        queries.append(gloss)
    queries.append(need.term)
    return queries


def _search_fill(need: ImageNeed, deck: Deck, manifest: Manifest, http_get,
                 fetcher, today: str, result: ProducerResult,
                 hints: dict[str, str] | None = None) -> None:
    queries = _queries(need, hints or {})
    # source-major: exhaust the better source's queries before the weaker one,
    # whose Thai-term matches are frequently transliteration collisions
    for search_fn in (openverse_search, wikimedia_search):
        for query in queries:
            try:
                candidates = list(search_fn(query, http_get))
            except requests.RequestException:
                continue          # per-item fault tolerance: a dead source is skipped
            for candidate in candidates:
                image = _try_candidate(candidate, fetcher)
                if image is None:
                    continue
                _write_media(deck, need.path, image)
                manifest.record(MediaEntry(
                    file=f"media/{need.path}", channel=candidate.source,
                    origin=candidate.url, license=candidate.license, fetched=today))
                result.changed += 1
                return
    result.blocked.append(need.note_id)

def _review_path(deck_root: Path) -> Path:
    return Path(deck_root) / "work" / "image_review.yaml"


def _review_items(deck_root: Path) -> list[dict]:
    path = _review_path(deck_root)
    data = yaml.safe_load(path.read_text()) if path.exists() else None
    return (data or {}).get("items", [])


def _queue_review(deck_root: Path, note_id: str, term: str, tried: list[str],
                  queries: list[str] | None = None,
                  corpora: list[str] | None = None) -> None:
    """Record a note nothing could illustrate, with the queries that failed.

    The queries are the memo: re-running must not re-search and re-judge a
    note whose search has already been exhausted, but a new search phrase is
    new information and does earn another attempt.
    """
    path = _review_path(deck_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [it for it in _review_items(deck_root) if it["note_id"] != note_id]
    item = {"note_id": note_id, "term": term, "tried": tried}
    if queries:
        item["queries"] = queries
        item["rubric"] = rubric_fingerprint(corpora=corpora)
    items.append(item)
    path.write_text(yaml.safe_dump({"items": items}, allow_unicode=True, sort_keys=False))


def _exhausted(deck_root: Path, need: ImageNeed, queries: list[str],
               corpora: list[str] | None = None) -> bool:
    """True when this exact query set already came back with nothing usable."""
    for item in _review_items(deck_root):
        if (item["note_id"] == need.note_id and item.get("queries") == queries
                and item.get("rubric") == rubric_fingerprint(corpora=corpora)):
            return True
    return False

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

@dataclass
class Candidate:
    """A downloaded image awaiting judgment, one of several per note."""
    note_id: str
    index: int
    url: str
    source: str
    license: str | None
    path: Path

    @property
    def request_id(self) -> str:
        return f"{self.note_id}#{self.index}"


def _note_unsearchable(deck_root: Path, term: str, phrase: str) -> None:
    """Record a phrase the corpus has nothing for.

    Result count is a free pre-search screen: whether a phrase is findable is
    a property of the index, knowable before a download or a judgment. It
    says nothing about whether the images would have worked -- that still
    needs the pictures themselves.
    """
    path = Path(deck_root) / "work" / "image_query_proposals.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if path.exists() else {})
    if term in data:
        return                     # a judge's suggestion outranks a bare count
    data[term] = {"previous": phrase, "reason": "phrase returned no results"}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def _collect_candidates(need: ImageNeed, deck: Deck, http_get, fetcher,
                        hints: dict[str, str], limit: int,
                        note=None, pexels_key: str | None = None,
                        corpora: list[str] | None = None) -> list[Candidate]:
    """Download up to `limit` distinct candidates for one need.

    Every candidate is judged in the same batch, so breadth costs one
    submission rather than one round trip per attempt.
    """
    out: list[Candidate] = []
    seen: set[str] = set()
    found: dict[str, int] = {}
    work = deck.root / "work" / "candidates" / need.note_id
    cached = _cached_candidates(work, need, limit, corpora)
    if cached:
        return cached                 # a stopped run keeps what it downloaded
    for search_fn in source_order(note if note is not None else need):
        for query in _queries(need, hints):
            if len(out) >= limit:
                return out
            try:
                if search_fn is pexels_search:
                    candidates = list(pexels_search(query, http_get,
                                                    api_key=pexels_key))
                else:
                    candidates = list(search_fn(query, http_get))
            except requests.RequestException:
                continue
            found[query] = found.get(query, 0) + len(candidates)
            for candidate in candidates:
                if len(out) >= limit or candidate.url in seen:
                    continue
                seen.add(candidate.url)
                image = _try_candidate(candidate, fetcher)
                if image is None:
                    continue
                work.mkdir(parents=True, exist_ok=True)
                path = work / f"{len(out)}.jpg"
                path.write_bytes(image)
                out.append(Candidate(need.note_id, len(out), candidate.url,
                                     candidate.source, candidate.license, path))
    # The screen is about the phrase, not one source: only a phrase that
    # every corpus came back empty on is unsearchable.
    if need.image_query and not found.get(need.image_query):
        _note_unsearchable(deck.root, need.term, need.image_query)
    if out:
        _write_candidate_meta(work, [{"file": c.path.name, "url": c.url,
                                      "source": c.source, "license": c.license}
                                     for c in out], corpora)
    return out


def _write_candidate_meta(work: Path, rows: list[dict],
                          corpora: list[str] | None = None) -> None:
    payload = {"corpora": corpora, "candidates": rows}
    (work / "candidates.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8")


def _cached_candidates(work: Path, need: ImageNeed, limit: int,
                       corpora: list[str] | None = None) -> list[Candidate]:
    """Candidates a previous run already downloaded for this note.

    Provenance lives in candidates.yaml beside the files; without it the
    images cannot be recorded in the manifest, so they are re-fetched.
    """
    meta = work / "candidates.yaml"
    if not meta.exists():
        return []
    try:
        loaded = yaml.safe_load(meta.read_text(encoding="utf-8")) or []
    except yaml.YAMLError:
        return []
    rows = loaded.get("candidates", []) if isinstance(loaded, dict) else loaded
    searched = loaded.get("corpora") if isinstance(loaded, dict) else None
    if corpora is not None and searched != corpora:
        return []            # a corpus has appeared since; search again
    out = []
    for row in rows[:limit]:
        path = work / row["file"]
        if not path.is_file():
            return []                 # partial: start this note over
        out.append(Candidate(need.note_id, len(out), row["url"], row["source"],
                             row.get("license"), path))
    return out


def _passes(verdicts) -> bool:
    return bool(verdicts) and all(v.passed for v in verdicts)


def _fill_verified(needs: list[ImageNeed], gaps: Gaps, deck: Deck,
                   manifest: Manifest, ctx, today: str, judge,
                   limit: int | None = None) -> ProducerResult:
    """Collect several candidates per note, judge them all in one batch, keep
    the one that passes. Search quality stops being load-bearing: a bad
    result is caught before it enters the deck rather than a judge pass later."""
    result = ProducerResult()
    limit = getattr(ctx, "image_candidates", 5)
    hints = getattr(ctx, "image_query_hints", None) or {}
    notes = {n.id: n for n in deck.picture_words}

    corpora = usable_corpora(ctx)
    has_pexels = "pexels" in corpora
    if not has_pexels:
        print("warning: no pexels key configured (gen.yaml secrets.pexels); "
              "concept photography for abstract words is unavailable")

    pool: dict[str, list[Candidate]] = {}
    exhausted: list[str] = []
    # Count against the limit only what this path can actually serve.
    servable = [n for n in needs if n.note_id in notes]
    needs = servable[:limit] if limit is not None else servable
    for need in needs:
        if _exhausted(deck.root, need, _queries(need, hints), corpora=corpora):
            exhausted.append(need.note_id)
            continue
        try:
            pool[need.note_id] = _collect_candidates(
                need, deck, ctx.http_get, ctx.imgfetch, hints, limit,
                note=notes[need.note_id],
                pexels_key=getattr(ctx, "pexels_key", None),
                corpora=corpora)
        except ImgFetchUnavailable as exc:
            print(f"warning: {exc}; blocking all remaining image needs")
            result.blocked.extend(n.note_id for n in needs)
            return result

    phrases = {n.note_id: n.image_query for n in needs}
    reqs = [JudgeRequest(note_id=c.request_id, rules=list(PICTURE_RULES),
                         prompt=build_picture_prompt(notes[c.note_id],
                                                     phrases.get(c.note_id)),
                         image_path=str(c.path))
            for cands in pool.values() for c in cands]
    verdicts = judge.judge_many(reqs) if reqs else {}

    by_note = {n.note_id: n for n in needs}
    result.blocked.extend(exhausted)
    if exhausted:
        print(f"  images: {len(exhausted)} note(s) already exhausted on these "
              f"queries; skipped (see work/image_review.yaml)")
    for note_id, cands in pool.items():
        winner = next((c for c in cands if _passes(verdicts.get(c.request_id))), None)
        if winner is None:
            _record_suggestion(deck.root, by_note[note_id], cands, verdicts)
            # Only a search that actually produced candidates and had them
            # rejected is exhausted. No candidates at all means the search or
            # the network failed, and memoizing that would blacklist the note
            # on the strength of a dropped connection.
            _queue_review(deck.root, note_id, by_note[note_id].term,
                          sorted({c.source for c in cands}),
                          queries=_queries(by_note[note_id], hints) if cands else None,
                          corpora=corpora)
            result.blocked.append(note_id)
        else:
            _write_media(deck, by_note[note_id].path, winner.path.read_bytes())
            manifest.record(MediaEntry(
                file=f"media/{by_note[note_id].path}", channel=winner.source,
                origin=winner.url, license=winner.license, fetched=today))
            result.changed += 1
        _record_verdicts(deck.root / "work" / "candidates" / note_id, cands,
                         verdicts, winner, corpora)
    return result


def _record_suggestion(deck_root: Path, need: ImageNeed, cands: list[Candidate],
                       verdicts: dict) -> None:
    """Keep the phrase the judge would have searched for instead.

    A new phrase changes the query set, which expires the exhaustion memo on
    its own -- so the next run retries the word without any retry loop.
    """
    suggestions = [v.suggestion for c in cands
                   for v in (verdicts.get(c.request_id) or [])
                   if not v.passed and getattr(v, "suggestion", None)]
    if not suggestions:
        return
    best = Counter(suggestions).most_common(1)[0][0]
    path = Path(deck_root) / "work" / "image_query_proposals.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if path.exists() else {})
    data[need.term] = {"suggestion": best, "previous": need.image_query}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def _record_verdicts(work: Path, cands: list[Candidate], verdicts: dict,
                     winner: Candidate | None,
                     corpora: list[str] | None = None) -> None:
    """Keep every candidate and what the judge said about it.

    Discarding the rejects made the rubric unrevisable: a relaxed rule
    cannot reconsider images that are no longer on disk, and re-searching
    is the slow half of an image run.
    """
    if not cands:
        return
    rows = []
    for c in cands:
        vs = verdicts.get(c.request_id) or []
        rows.append({
            "file": c.path.name, "url": c.url, "source": c.source,
            "license": c.license,
            "passed": bool(vs) and all(v.passed for v in vs),
            "failed_rules": sorted({v.rule for v in vs if not v.passed}),
            "accepted": winner is not None and c.request_id == winner.request_id,
        })
    _write_candidate_meta(work, rows, corpora)


def fill_images(needs: list[ImageNeed], gaps: Gaps, deck: Deck,
                manifest: Manifest, ctx, today: str, judge=None,
                limit: int | None = None) -> ProducerResult:
    if judge is not None:
        return _fill_verified(needs, gaps, deck, manifest, ctx, today, judge, limit)
    if limit is not None:
        needs = needs[:limit]

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

        try:
            _search_fill(need, deck, manifest, ctx.http_get, ctx.imgfetch, today,
                         result, hints=getattr(ctx, "image_query_hints", None))
        except ImgFetchUnavailable as exc:
            # not per-item: without the binary no download can succeed
            print(f"warning: {exc}; blocking all remaining image needs")
            result.blocked.append(need.note_id)
            result.blocked.extend(n.note_id for n in needs[needs.index(need) + 1:])
            return result

    return result
