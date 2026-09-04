"""The Provide port (spec 3 section 1/2): Provider.ask(backend, question)
-> Answer, cache-first over spec 2's `cache` table.

Cache-first semantics (spec 3 section 1): every ask() consults the cache
first; a hit costs nothing and appends nothing; a miss executes the
backend, then appends exactly one row -- hit-or-miss, success-or-empty (an
empty answer IS cached). A transport error is NOT an answer: it propagates
as an exception and nothing is appended, so the subject stays queued and
the next run retries (spec 3 section 7: "no retry frameworks").

Backends are injectable callables/classes satisfying the `Backend`
Protocol below (`cache_key` + `fetch`); this module supplies the
per-backend key functions and cost/transport wiring spec 3's roster table
(section 2) specifies, and stays stdlib + requests (no pythainlp, no
anthropic at module scope -- transport.py guards that import).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import requests

from .assessor import Price
from .cachekeys import sha
from .ports import CacheReader, RecordWriter
from .transport import Completion, TransportError

__all__ = [
    "Question", "ProviderAnswer", "RawAnswer", "Backend", "MediaWriter",
    "Provider", "LearnerAskNotSupported",
    "HttpImageSearchBackend", "openverse_backend", "wikimedia_backend",
    "pexels_backend", "IMAGE_SEARCH_USER_AGENT",
    "FetchBackend", "tool_fetcher",
    "ForvoBackend", "TtsBackend", "LlmBackend",
    "DictionaryG2P", "PairSearchBackend",
]


# --- the port contract (spec 3 section 1) -----------------------------------

@dataclass(frozen=True)
class Question:
    subject: str
    # provides: picture | recording | sentence | pair | phrase | entry
    provides: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderAnswer:
    """items may be empty -- a miss is an answer and is cached. `hit` is
    the port's own answer to "was this served from the cache?" -- the only
    authority on it, since only ask() knows which branch it took (a
    caller comparing `ts` against its own start time counts a row this
    same caller wrote a moment ago as a fresh ask every time it re-reads
    it).
    """
    items: tuple[Any, ...]
    cost: float
    ts: int
    hit: bool = False


@dataclass(frozen=True)
class RawAnswer:
    """What a Backend.fetch() returns before Provider wraps it with a ts."""
    items: tuple[Any, ...] = ()
    cost: float = 0.0


class LearnerAskNotSupported(RuntimeError):
    """Provider.ask("learner", ...) always raises this (spec 3 roster:
    learner has "n/a (no question key); rows are acts ... asked via the
    feedback screen only"). The learner backend never executes; its rows
    arrive through RecordWriter.append from the feedback surfaces (spec 5),
    not through ask().
    """


@runtime_checkable
class Backend(Protocol):
    def cache_key(self, question: Question) -> str: ...
    def fetch(self, question: Question) -> RawAnswer: ...  # may raise -- not cached


@runtime_checkable
class MediaWriter(Protocol):
    """The slice of store.MediaStore that binary-artifact backends
    (imgfetch, audiofetch, tts) need: content-addressed bytes-in, sha-out;
    `add_image` additionally normalizes (spec 4 section 3).
    """
    def write(self, data: bytes, ext: str) -> str: ...
    def add_image(self, data: bytes, ext: str) -> "ImageIngestResult": ...


class Provider:
    """Cache-first ask() over injected backends (spec 3 section 1)."""

    def __init__(self, record: RecordWriter, cache: CacheReader,
                backends: Mapping[str, Backend]):
        self._record = record
        self._cache = cache
        self._backends = dict(backends)

    def ask(self, backend: str, question: Question) -> ProviderAnswer:
        if backend == "learner":
            raise LearnerAskNotSupported(
                "the learner Provide backend has no ask(); its rows arrive "
                "via RecordWriter from the feedback surfaces")
        impl = self._backends[backend]
        key = impl.cache_key(question)
        cached = self._cache.latest("provide", backend, key)
        if cached is not None:
            return ProviderAnswer(items=tuple(cached.answer.get("items", [])),
                                  cost=0.0, ts=cached.ts, hit=True)
        raw = impl.fetch(question)  # transport errors propagate uncached
        ts = self._record.append(
            port="provide", backend=backend, key=key, subject=question.subject,
            question={"provides": question.provides, "params": dict(question.params)},
            answer={"items": list(raw.items)}, cost=raw.cost)
        return ProviderAnswer(items=raw.items, cost=raw.cost, ts=ts)


# --- image search: openverse, wikimedia, pexels -----------------------------
# key = "backend:query" (spec 3 roster); a new query is a new key, the same
# query is re-asked only manually (nothing here re-asks automatically).

IMAGE_SEARCH_USER_AGENT = (
    "thai-syllabus-deck-builder/1.0 "
    "(personal Thai-language Anki deck project; non-commercial)"
)


@dataclass
class HttpImageSearchBackend:
    """Generic HTTP image-corpus search: one GET, JSON response, a
    corpus-specific request-builder and item-parser. `search_proxy`
    (spec 3 section 5) replaces the corpus's own base URL, e.g. to route
    through a caching/rate-limiting proxy.
    """
    name: str
    build_request: Callable[[str, str | None], tuple[str, dict, dict]]
    parse_items: Callable[[Any], list[dict]]
    get: Callable[..., Any] = field(default=requests.get)
    search_proxy: str | None = None

    def cache_key(self, question: Question) -> str:
        return f"{self.name}:{question.params['query']}"

    def fetch(self, question: Question) -> RawAnswer:
        query = question.params["query"]
        url, params, headers = self.build_request(query, self.search_proxy)
        try:
            resp = self.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            raise TransportError(f"{self.name} search failed: {e}") from e
        if resp.status_code != 200:
            raise TransportError(
                f"{self.name} search returned {resp.status_code}: {resp.text[:200]}")
        items = self.parse_items(resp.json())
        return RawAnswer(items=tuple(items), cost=0.0)


def openverse_backend(get: Callable[..., Any] = requests.get,
                      search_proxy: str | None = None) -> HttpImageSearchBackend:
    def build(query: str, proxy: str | None) -> tuple[str, dict, dict]:
        base = proxy or "https://api.openverse.org"
        return (f"{base}/v1/images/",
               {"q": query, "license_type": "commercial,modification"},
               {"User-Agent": IMAGE_SEARCH_USER_AGENT})

    def parse(data: Any) -> list[dict]:
        return [{"url": r.get("url"), "licence": r.get("license"),
                "source": "openverse",
                "origin": r.get("foreign_landing_url") or r.get("url")}
               for r in data.get("results", [])]

    return HttpImageSearchBackend(name="openverse", build_request=build,
                                  parse_items=parse, get=get, search_proxy=search_proxy)


def wikimedia_backend(get: Callable[..., Any] = requests.get,
                      search_proxy: str | None = None) -> HttpImageSearchBackend:
    def build(query: str, proxy: str | None) -> tuple[str, dict, dict]:
        base = proxy or "https://commons.wikimedia.org"
        return (f"{base}/w/api.php",
               {"action": "query", "generator": "search", "gsrsearch": query,
                "gsrnamespace": "6", "prop": "imageinfo", "iiprop": "url",
                "format": "json"},
               {"User-Agent": IMAGE_SEARCH_USER_AGENT})

    def parse(data: Any) -> list[dict]:
        out = []
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            for info in page.get("imageinfo", []) or []:
                if info.get("url"):
                    out.append({"url": info["url"], "source": "wikimedia", "licence": None,
                               "origin": f"https://commons.wikimedia.org/wiki/{page.get('title', '')}"})
        return out

    return HttpImageSearchBackend(name="wikimedia", build_request=build,
                                  parse_items=parse, get=get, search_proxy=search_proxy)


def pexels_backend(api_key: str, get: Callable[..., Any] = requests.get,
                   search_proxy: str | None = None) -> HttpImageSearchBackend:
    def build(query: str, proxy: str | None) -> tuple[str, dict, dict]:
        base = proxy or "https://api.pexels.com"
        return (f"{base}/v1/search", {"query": query},
               {"User-Agent": IMAGE_SEARCH_USER_AGENT, "Authorization": api_key})

    def parse(data: Any) -> list[dict]:
        return [{"url": p.get("src", {}).get("original"), "licence": "pexels",
                "source": "pexels", "origin": p.get("url")}
               for p in data.get("photos", [])]

    return HttpImageSearchBackend(name="pexels", build_request=build,
                                  parse_items=parse, get=get, search_proxy=search_proxy)


# --- imgfetch/audiofetch: fetch a candidate's bytes by url ------------------
# key = url (content-addressed); a fetch failure is NOT cached (raises).

_FORMAT_EXT = {"jpeg": "jpg", "png": "png", "gif": "gif", "webp": "webp", "mp3": "mp3"}


def tool_fetcher(binary: str, runner: Callable[..., Any] | None = None
                 ) -> Callable[[str], tuple[bytes, str]]:
    """Fetch through one of the Go tools (tools/mediafetch: imgfetch,
    audiofetch): `<binary> <url> <out-path>`, a JSON line {format,...} on
    stdout, non-zero exit on refusal. `runner` defaults to `subprocess.run`
    looked up at call time (not bound eagerly) so tests can monkeypatch
    `subprocess.run` on this module without passing `runner` explicitly.
    """
    def fetch(url: str) -> tuple[bytes, str]:
        run = runner if runner is not None else subprocess.run
        with tempfile.TemporaryDirectory(prefix="mediafetch-") as tmp:
            out = Path(tmp) / "object"
            try:
                proc = run([binary, url, str(out)], capture_output=True, text=True, timeout=120)
            except OSError as e:
                raise TransportError(f"cannot run {binary!r}: {e}") from e
            except subprocess.TimeoutExpired as e:
                raise TransportError(f"{binary} timed out on {url!r}") from e
            if proc.returncode != 0 or not out.is_file():
                raise TransportError(f"{binary} refused {url!r}: {(proc.stderr or '').strip()}")
            try:
                fmt = json.loads((proc.stdout or "{}").splitlines()[-1]).get("format", "")
            except (json.JSONDecodeError, IndexError):
                fmt = ""
            return out.read_bytes(), _FORMAT_EXT.get(fmt, fmt or "bin")
    return fetch


@dataclass
class FetchBackend:
    """url -> bytes -> media store -> sha, for pictures (normalized through
    add_image) and recordings (stored raw). key = the url; a fetch failure
    is not cached (cost 0 -- a download is not a lookup). Every question
    param except url is echoed into the item (speaker, speaker_kind,
    source, origin), so the attempt records provenance from the item.
    """
    media: MediaWriter
    fetcher: Callable[[str], tuple[bytes, str]]

    def cache_key(self, question: Question) -> str:
        return question.params["url"]

    def fetch(self, question: Question) -> RawAnswer:
        url = question.params["url"]
        data, ext = self.fetcher(url)
        if question.provides == "picture-bytes":
            ingest = self.media.add_image(data, ext)
            sha_, ext = ingest.sha, ingest.ext
        else:
            ext = ext if ext in ("mp3", "ogg", "wav") else "mp3"
            sha_ = self.media.write(data, ext)
        item = {k: v for k, v in question.params.items() if k != "url"}
        item.update({"sha": sha_, "ext": ext})
        return RawAnswer(items=(item,), cost=0.0)


# --- forvo: recording lookups (500/day quota; never re-asked) --------------
# key = "forvo:WORD".

@dataclass
class ForvoBackend:
    api_key: str
    get: Callable[..., Any] = field(default=requests.get)
    base_url: str = "https://apifree.forvo.com"

    def cache_key(self, question: Question) -> str:
        word = question.params.get("word", question.subject)
        return f"forvo:{word}"

    def fetch(self, question: Question) -> RawAnswer:
        word = question.params.get("word", question.subject)
        url = (f"{self.base_url}/key/{self.api_key}/format/json/"
              f"action/word-pronunciations/word/{word}")
        try:
            resp = self.get(url, timeout=30)
        except requests.RequestException as e:
            raise TransportError(f"forvo lookup of {word!r} failed: {e}") from e
        if resp.status_code != 200:
            raise TransportError(f"forvo returned {resp.status_code}")
        data = resp.json()
        items = data.get("items", [])
        return RawAnswer(items=tuple(items), cost=1.0)  # 1 lookup against the daily quota


# --- tts: Google TTS (deterministic; never re-asked) ------------------------
# key = "tts:VOICE:sha(TEXT)".

@dataclass
class TtsBackend:
    """`cost_per_char` is the configured rate (providers.yaml `tts`), not a
    per-question parameter: the rate belongs to the backend that incurs it
    (spec 3 section 2's "measured by the backend"), and reading it off the
    question let any caller that forgot to pass it record a free synthesis.
    """
    tts: Any  # thai_syllabus.tts.Tts -- synthesize(text, voice) -> bytes
    voices: Sequence[str]
    media: MediaWriter
    pick_voice: Callable[[str, Sequence[str]], str]
    cost_per_char: float = 0.0

    def _voice(self, question: Question) -> str:
        return question.params.get("voice") or self.pick_voice(question.subject, self.voices)

    def cache_key(self, question: Question) -> str:
        text = question.params["text"]
        return f"tts:{self._voice(question)}:{sha(text)}"

    def fetch(self, question: Question) -> RawAnswer:
        text = question.params["text"]
        voice = self._voice(question)
        audio = self.tts.synthesize(text, voice)
        artifact_sha = self.media.write(audio, "mp3")
        return RawAnswer(items=({"sha": artifact_sha, "ext": "mp3", "voice": voice,
                                 "speaker_kind": "synthetic", "source": "tts",
                                 "origin": voice},),
                         cost=len(text) * self.cost_per_char)


# --- llm: sentence/phrase/entry drafting -------------------------------------
# key = "llm:PRODUCER:MODEL:sha(PROMPT)"; the prompt text is the entire
# contract -- any semantic change edits the text (spec 3 roster).

@dataclass
class LlmBackend:
    """Costed exactly like assessor.JudgeBackend, on the same currencies
    (spec 3 section 2's cost contract): a `price` prices the completion's
    actual token usage (api/batch, cash), and `quota_cost_per_call` is the
    flat subscription-quota cost of the cli transport, which reports no
    usage on the wire. A transport that receives usage and drops it
    violates the contract, so `fetch` keeps the whole Completion.
    """
    producer: str
    model: str
    transport: Any  # .complete(prompt: str) -> Completion; may raise TransportError
    price: Price | None = None
    quota_cost_per_call: float = 0.0

    def cache_key(self, question: Question) -> str:
        prompt = question.params["prompt"]
        return f"llm:{self.producer}:{self.model}:{sha(prompt)}"

    def _cost(self, completion: Completion) -> float:
        if self.price is not None:
            return self.price.cost(completion)
        return self.quota_cost_per_call

    def fetch(self, question: Question) -> RawAnswer:
        prompt = question.params["prompt"]
        completion = self.transport.complete(prompt)
        items = (completion.text,) if completion.text else ()
        return RawAnswer(items=items, cost=self._cost(completion))


# --- pair-search: minimal pairs over a dictionary + G2P ---------------------
# key = "pairs:CONFUSION:DICT_VERSION"; dictionary bump = new key.

@runtime_checkable
class DictionaryG2P(Protocol):
    """A dictionary+G2P corpus searchable for minimal-pair candidates
    (domain-language doc: "corpus = Thai at large (dictionary+G2P search,
    curated seeds, LLM proposal + mechanical verification)"). No default
    implementation ships here -- pythainlp/tltk stay out of the default
    test suite; production wiring is a fake-satisfying adapter elsewhere.
    """
    def version(self) -> str: ...
    def search(self, confusion_id: str) -> Sequence[Mapping[str, Any]]: ...


@dataclass
class PairSearchBackend:
    dictionary: DictionaryG2P

    def cache_key(self, question: Question) -> str:
        confusion_id = question.params["confusion_id"]
        return f"pairs:{confusion_id}:{self.dictionary.version()}"

    def fetch(self, question: Question) -> RawAnswer:
        confusion_id = question.params["confusion_id"]
        candidates = self.dictionary.search(confusion_id)
        return RawAnswer(items=tuple(candidates), cost=0.0)
