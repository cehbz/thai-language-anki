"""The Provide port (spec 3 section 1/2): Provider.ask(backend, question)
-> Answer, cache-first over spec 2's `cache` table.

Cache-first semantics (spec 3 section 1): every ask() consults the cache
first; a hit costs nothing and appends nothing; a miss executes the
backend, then appends exactly one row, success or empty (an empty answer
is cached). A transport error is not an answer: it propagates and nothing
is appended, so the subject stays queued for the next run.

Backends satisfy the `Backend` Protocol below (`cache_key` + `fetch`);
this module holds the roster spec 3 section 2 specifies, each with its
cachekeys.py key and its cost/transport wiring, on stdlib + requests.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import requests

from .assessor import LearnerAskNotSupported, Price
from .cachekeys import CacheKey, LlmPromptKey, PairSearchKey, ProvideKey, sha
from .ports import CacheReader, RecordWriter
from .transport import Completion, FetchRefused, QuotaExhausted, TransportError

__all__ = [
    "Question", "ProviderAnswer", "RawAnswer", "Backend", "MediaWriter",
    "Provider", "LearnerAskNotSupported",
    "HttpImageSearchBackend", "openverse_backend", "wikimedia_backend",
    "pexels_backend", "brave_backend", "OpenverseAuth", "IMAGE_SEARCH_USER_AGENT",
    "FetchBackend", "tool_fetcher",
    "ForvoBackend", "forvo_limit_body", "TtsBackend", "LlmBackend",
    "DictionaryG2P", "PairSearchBackend",
]


# --- the port contract (spec 3 section 1) -----------------------------------

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Question:
    subject: str
    # provides: picture | recording | sentence | pair | phrase | entry
    provides: str
    params: Mapping[str, Any] = field(default_factory=dict)
    # The artifact kind (picture | recording | rendition | sentence |
    # grapheme-keyword) the caller is asking toward, and the kind of thing
    # `subject` is (word | pair | sentence | grapheme) -- record.py's folds
    # read both back verbatim; Provider derives neither from `provides`.
    kind: str = ""
    subject_kind: str = "word"


@dataclass(frozen=True)
class ProviderAnswer:
    """One ask's answer. `items` may be empty (a miss is an answer, and is
    cached); `hit` says whether ask() served it from the cache.
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


@runtime_checkable
class Backend(Protocol):
    def cache_key(self, question: Question) -> CacheKey: ...
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
        ts = self._append_answer(backend, key, question, raw)
        return ProviderAnswer(items=raw.items, cost=raw.cost, ts=ts)

    def reask(self, backend: str, question: Question) -> ProviderAnswer:
        """Executes and appends over a hit (spec 3 section 6a's re-ask
        rule); the newest row is the answer ask() reads next."""
        if backend == "learner":
            raise LearnerAskNotSupported("the learner Provide backend has no reask()")
        impl = self._backends[backend]
        key = impl.cache_key(question)
        raw = impl.fetch(question)  # transport errors propagate uncached
        ts = self._append_answer(backend, key, question, raw)
        return ProviderAnswer(items=raw.items, cost=raw.cost, ts=ts)

    def _append_answer(self, backend: str, key: CacheKey, question: Question,
                       raw: RawAnswer) -> int:
        return self._record.append(
            port="provide", backend=backend, key=key, subject=question.subject,
            question={"provides": question.provides, "kind": question.kind,
                     "subject_kind": question.subject_kind,
                     "params": dict(question.params)},
            answer={"items": list(raw.items)}, cost=raw.cost)


# --- image search: openverse, wikimedia, pexels -----------------------------
# key = ProvideKey(source=backend name, query=the phrase): a new query is a
# new key, and nothing here re-asks an old one.

IMAGE_SEARCH_USER_AGENT = (
    "thai-syllabus-deck-builder/1.0 "
    "(personal Thai-language Anki deck project; non-commercial)"
)


OPENVERSE_TOKEN_URL = "https://api.openverse.org/v1/auth_tokens/token/"


@dataclass
class OpenverseAuth:
    """Openverse's OAuth2 client-credentials flow (spec 3 r26 section 8:
    `secrets.openverse` is `client_id:client_secret`). `token()` posts
    the credentials form-encoded to the token endpoint -- through the
    same forward proxy the searches use -- once, then serves the cached
    access token until `expires_in` has passed (a 60 s margin), then
    fetches again. A refused or malformed token answer is a
    TransportError: the source is unreachable for the run, nothing is
    cached.
    """
    client_id: str
    client_secret: str
    # resolved per instance, not at import: a test's patched requests.post
    # is what an auth built later uses
    post: Callable[..., Any] = field(default_factory=lambda: requests.post)
    search_proxy: str | None = None
    clock: Callable[[], float] = time.monotonic
    _token: str | None = field(default=None, init=False, repr=False)
    _expires_at: float = field(default=0.0, init=False, repr=False)

    def token(self) -> str:
        now = self.clock()
        if self._token is not None and now < self._expires_at:
            return self._token
        proxies = ({"http": self.search_proxy, "https": self.search_proxy}
                  if self.search_proxy else None)
        try:
            resp = self.post(OPENVERSE_TOKEN_URL,
                             data={"client_id": self.client_id,
                                   "client_secret": self.client_secret,
                                   "grant_type": "client_credentials"},
                             headers={"User-Agent": IMAGE_SEARCH_USER_AGENT},
                             timeout=30, proxies=proxies)
        except requests.RequestException as e:
            raise TransportError(f"openverse token request failed: {e}") from e
        if resp.status_code != 200:
            raise TransportError(
                f"openverse token request returned {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except ValueError as e:
            raise TransportError(f"openverse token answer is not json: {e}") from e
        token = body.get("access_token") if isinstance(body, Mapping) else None
        if not token:
            raise TransportError(f"openverse token answer carries no access_token: {str(body)[:120]}")
        expires_in = body.get("expires_in")
        lifetime = float(expires_in) if isinstance(expires_in, (int, float)) else 3600.0
        self._token, self._expires_at = str(token), now + max(lifetime - 60.0, 0.0)
        return self._token


def _is_challenge(resp: Any) -> bool:
    """Whether a 403 answer is Cloudflare's managed challenge page rather
    than the API's own refusal: the `cf-mitigated: challenge` header, or
    the challenge page's title."""
    if resp.status_code != 403:
        return False
    headers = getattr(resp, "headers", None) or {}
    if str(headers.get("cf-mitigated", "")).lower() == "challenge":
        return True
    text = getattr(resp, "text", "") or ""
    return "Just a moment" in text or "challenge-platform" in text


@dataclass
class HttpImageSearchBackend:
    """Generic HTTP image-corpus search: one GET, JSON response, a
    corpus-specific request-builder and item-parser. `search_proxy`, when
    set, is the HTTP forward proxy every search request of this corpus is
    sent through (media sourcing: Openverse refuses a Thai egress).
    """
    name: str
    build_request: Callable[[str], tuple[str, dict, dict, str]]
    parse_items: Callable[[Any], list[dict]]
    get: Callable[..., Any] = field(default=requests.get)
    search_proxy: str | None = None
    min_interval_s: float = 0.0
    challenge_wait_s: float = 0.0
    auth: Callable[[], str | None] | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def cache_key(self, question: Question) -> ProvideKey:
        return ProvideKey(source=self.name, kind="", query=question.params["query"])

    _last_call: float | None = field(default=None, init=False, repr=False)

    def _pace(self) -> None:
        """Spec 3 r26 section 8: at least `min_interval_s` between two of
        this backend's requests within one process."""
        if self.min_interval_s > 0 and self._last_call is not None:
            remaining = self._last_call + self.min_interval_s - self.clock()
            if remaining > 0:
                self.sleep(remaining)

    def _request(self, url: str, params: Mapping, headers: Mapping, proxies: Mapping | None):
        self._pace()
        try:
            return self.get(url, params=params, headers=headers, timeout=30, proxies=proxies)
        except requests.RequestException as e:
            raise TransportError(f"{self.name} search failed: {e}") from e
        finally:
            self._last_call = self.clock()

    def fetch(self, question: Question) -> RawAnswer:
        query = question.params["query"]
        url, params, headers, expect = self.build_request(query)
        headers = dict(headers)
        if self.auth is not None:
            token = self.auth()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        proxies = ({"http": self.search_proxy, "https": self.search_proxy}
                  if self.search_proxy else None)
        resp = self._request(url, params, headers, proxies)
        if _is_challenge(resp) and self.challenge_wait_s > 0:
            # Spec 3 r26 section 6a: one wait, one retry; a second
            # challenge is the transport failure.
            _log.warning("%s: challenge page; waiting %.0f s before one retry",
                         self.name, self.challenge_wait_s)
            self.sleep(self.challenge_wait_s)
            resp = self._request(url, params, headers, proxies)
        if _is_challenge(resp):
            raise TransportError(f"{self.name} search answered a challenge page (403)")
        if resp.status_code in (429, 402):
            # Spec 3 r26 section 6a: the source's own throttle is the
            # Quota state, budgeted for the run, never a source failure.
            # 402 Payment Required is the same state on a metered source
            # (Brave, once the month's free credit is spent): out of
            # budget, not broken.
            raise QuotaExhausted(self.name)
        if resp.status_code != 200:
            raise TransportError(
                f"{self.name} search returned {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as e:
            raise TransportError(
                f"{self.name} search answered a body that is not json: {e}") from e
        if not isinstance(data, Mapping) or expect not in data:
            raise TransportError(
                f"{self.name} search answered a body without {expect!r}: {str(data)[:120]}")
        items = self.parse_items(data)
        return RawAnswer(items=tuple(items), cost=0.0)


def openverse_backend(get: Callable[..., Any] = requests.get,
                      search_proxy: str | None = None, **pacing: Any) -> HttpImageSearchBackend:
    def build(query: str) -> tuple[str, dict, dict, str]:
        return ("https://api.openverse.org/v1/images/",
               {"q": query, "license_type": "commercial,modification"},
               {"User-Agent": IMAGE_SEARCH_USER_AGENT}, "results")

    def parse(data: Any) -> list[dict]:
        return [{"url": r.get("url"), "licence": r.get("license"),
                "source": "openverse",
                "origin": r.get("foreign_landing_url") or r.get("url")}
               for r in data.get("results", [])]

    return HttpImageSearchBackend(name="openverse", build_request=build,
                                  parse_items=parse, get=get, search_proxy=search_proxy,
                                  **pacing)


def wikimedia_backend(get: Callable[..., Any] = requests.get, *,
                      image_width: int, **pacing: Any) -> HttpImageSearchBackend:
    def build(query: str) -> tuple[str, dict, dict, str]:
        # "batchcomplete" is on every MediaWiki action-API search reply,
        # zero hits included (zero hits omits "query" entirely); an
        # {"error": {...}} body carries neither. gsrsearch's "filetype:
        # bitmap" excludes PDF/DjVu hits; iiurlwidth bounds the scaled
        # rendition imgfetch is handed (its thumburl), instead of the
        # source file's full-resolution bytes.
        return ("https://commons.wikimedia.org/w/api.php",
               {"action": "query", "generator": "search",
                "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": "6", "prop": "imageinfo", "iiprop": "url",
                "iiurlwidth": image_width, "format": "json"},
               {"User-Agent": IMAGE_SEARCH_USER_AGENT}, "batchcomplete")

    def parse(data: Any) -> list[dict]:
        out = []
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            for info in page.get("imageinfo", []) or []:
                if info.get("thumburl"):
                    out.append({"url": info["thumburl"], "source": "wikimedia", "licence": None,
                               "origin": f"https://commons.wikimedia.org/wiki/{page.get('title', '')}"})
        return out

    return HttpImageSearchBackend(name="wikimedia", build_request=build,
                                  parse_items=parse, get=get, **pacing)


def pexels_backend(api_key: str, get: Callable[..., Any] = requests.get,
                   **pacing: Any) -> HttpImageSearchBackend:
    def build(query: str) -> tuple[str, dict, dict, str]:
        return ("https://api.pexels.com/v1/search", {"query": query},
               {"User-Agent": IMAGE_SEARCH_USER_AGENT, "Authorization": api_key}, "photos")

    def parse(data: Any) -> list[dict]:
        return [{"url": p.get("src", {}).get("original"), "licence": "pexels",
                "source": "pexels", "origin": p.get("url")}
               for p in data.get("photos", [])]

    return HttpImageSearchBackend(name="pexels", build_request=build,
                                  parse_items=parse, get=get, **pacing)


def brave_backend(api_key: str, get: Callable[..., Any] = requests.get, *,
                  count: int = 5, **pacing: Any) -> HttpImageSearchBackend:
    """Brave's image search: the last picture source (spec 3 section 5), a
    metered web index asked only for the needs the free corpora failed.
    `count` is the deck's image_candidates -- the attempt fetches no more
    than that many hits, so asking for more spends credit on hits nothing
    reads. A search carries no licence: the item's `licence` is None, the
    unknown the media row records.
    """
    def build(query: str) -> tuple[str, dict, dict, str]:
        return ("https://api.search.brave.com/res/v1/images/search",
               {"q": query, "count": count, "safesearch": "strict"},
               {"X-Subscription-Token": api_key, "Accept": "application/json",
                "User-Agent": IMAGE_SEARCH_USER_AGENT}, "results")

    def parse(data: Any) -> list[dict]:
        out = []
        for r in data.get("results", []):
            # `properties.url` is the full-size image; `thumbnail.src` is
            # Brave's own cached rendition, the fallback when the result
            # carries no full-size url.
            url = (r.get("properties") or {}).get("url") or (r.get("thumbnail") or {}).get("src")
            if not url:
                continue
            out.append({"url": url, "licence": None, "source": "brave",
                       "origin": r.get("url")})
        return out

    return HttpImageSearchBackend(name="brave", build_request=build,
                                  parse_items=parse, get=get, **pacing)


# --- imgfetch/audiofetch: fetch a candidate's bytes by url ------------------
# key = the url alone; a fetch failure raises and is not cached.

_FORMAT_EXT = {"jpeg": "jpg", "png": "png", "gif": "gif", "webp": "webp", "mp3": "mp3"}


def tool_fetcher(binary: str, runner: Callable[..., Any] | None = None
                 ) -> Callable[[str], tuple[bytes, str]]:
    """Fetch through one of the Go tools (tools/mediafetch: imgfetch,
    audiofetch): `<binary> <url> <out-path>`, a JSON line {format,...} on
    stdout, non-zero exit on refusal. On refusal the tool prints a JSON
    line {refused, detail} on stdout; this parses that line into a
    `FetchRefused`. `runner` defaults to this module's `subprocess.run`,
    looked up at call time.
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
                refusal = _refusal_line(proc.stdout or "")
                if refusal is not None:
                    raise FetchRefused(reason=str(refusal.get("refused") or "io"),
                                       detail=str(refusal.get("detail") or ""),
                                       body=str(refusal.get("body") or ""))
                raise TransportError(f"{binary} refused {url!r}: {(proc.stderr or '').strip()}")
            try:
                fmt = json.loads((proc.stdout or "{}").splitlines()[-1]).get("format", "")
            except (json.JSONDecodeError, IndexError):
                fmt = ""
            return out.read_bytes(), _FORMAT_EXT.get(fmt, fmt or "bin")
    return fetch


def _refusal_line(stdout: str) -> dict | None:
    """The tool's {"refused", "detail"} line, when its last stdout line
    is one."""
    lines = stdout.strip().splitlines()
    if not lines:
        return None
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "refused" in data else None


@dataclass
class FetchBackend:
    """url -> bytes -> media store -> sha, for pictures (normalized
    through add_image) and recordings (stored raw), at cost 0. Every
    question param but url is echoed into the item (speaker,
    speaker_kind, source, origin), the attempt's provenance.
    """
    media: MediaWriter
    fetcher: Callable[[str], tuple[bytes, str]]

    def cache_key(self, question: Question) -> ProvideKey:
        return ProvideKey(source="", kind="", query=question.params["url"])

    def fetch(self, question: Question) -> RawAnswer:
        url = question.params["url"]
        data, ext = self.fetcher(url)
        if question.provides == "picture-bytes":
            try:
                ingest = self.media.add_image(data, ext)
            except ValueError as e:
                raise FetchRefused(reason="format", detail=str(e)) from e
            sha_, ext = ingest.sha, ingest.ext
        else:
            ext = ext if ext in ("mp3", "ogg", "wav") else "mp3"
            sha_ = self.media.write(data, ext)
        item = {k: v for k, v in question.params.items() if k != "url"}
        item.update({"sha": sha_, "ext": ext})
        return RawAnswer(items=(item,), cost=0.0)


def _redact(text: str, secret: str) -> str:
    """Every occurrence of `secret` in `text` replaced with "***" (spec 3
    section 2's cost/secrets contract, extended to logs: a credential
    never appears by value outside config, including in a wire-failure
    message a caller might log). tts.py imports this rather than
    duplicating it.
    """
    if not secret:
        return text
    return text.replace(secret, "***")


# --- forvo: recording lookups (450/day quota; re-asked once per attempt
# when a url has expired, spec 3 section 6a) --------------------------------

def forvo_limit_body(body: Any) -> bool:
    """Forvo's own statement that today's allowance is spent (spec 3
    section 6a): a body that is a list of exactly one element, the
    string "Limit/day reached." -- the shape of both the lookup's 400
    body (`_forvo_quota_exhausted`) and a download's refused body
    (attempts._fetch_forvo_item), so both call this one predicate."""
    return isinstance(body, list) and len(body) == 1 and body[0] == "Limit/day reached."


def _forvo_quota_exhausted(resp: Any) -> bool:
    """Forvo's own statement that today's allowance is spent (spec 3
    section 6a): a 400 body that is a list of exactly one element, the
    string "Limit/day reached." -- any other 400 body stays a plain
    TransportError."""
    try:
        body = resp.json()
    except ValueError:
        return False
    return forvo_limit_body(body)


@dataclass
class ForvoBackend:
    api_key: str
    get: Callable[..., Any] = field(default=requests.get)
    base_url: str = "https://apifree.forvo.com"

    def cache_key(self, question: Question) -> ProvideKey:
        return ProvideKey(source="forvo", kind="",
                          query=question.params.get("word", question.subject))

    def fetch(self, question: Question) -> RawAnswer:
        word = question.params.get("word", question.subject)
        url = (f"{self.base_url}/key/{self.api_key}/format/json/"
              f"action/word-pronunciations/word/{word}")
        try:
            resp = self.get(url, timeout=30)
            if resp.status_code == 400 and _forvo_quota_exhausted(resp):
                raise QuotaExhausted("forvo")
            if resp.status_code != 200:
                raise TransportError(f"forvo returned {resp.status_code}")
            data = resp.json()
        except requests.RequestException as e:
            # from None: the chained cause would still carry the raw,
            # unredacted url/key, printed by any full traceback render.
            raise TransportError(
                f"forvo lookup of {word!r} failed: {_redact(str(e), self.api_key)}") from None
        except ValueError as e:
            raise TransportError(f"forvo answered {word!r} with a body that is not json: {e}") from e
        if not isinstance(data, Mapping) or not isinstance(data.get("items"), list):
            raise TransportError(f"forvo answered {word!r} with a body without items: {str(data)[:120]}")
        return RawAnswer(items=tuple(data["items"]), cost=1.0)  # 1 lookup against the daily quota


# --- tts: Google TTS (deterministic; never re-asked) ------------------------

@dataclass
class TtsBackend:
    """Synthesizes one text into the media store. `cost_per_char` is the
    configured rate (providers.yaml `tts`), carried by the backend that
    incurs it (spec 3 section 2's "measured by the backend").
    """
    tts: Any  # thai_syllabus.tts.Tts -- synthesize(text, voice) -> bytes
    voices: Sequence[str]
    media: MediaWriter
    pick_voice: Callable[[str, Sequence[str]], str]
    cost_per_char: float = 0.0

    def _voice(self, question: Question) -> str:
        return question.params.get("voice") or self.pick_voice(question.subject, self.voices)

    def cache_key(self, question: Question) -> ProvideKey:
        return ProvideKey(source="tts", kind=self._voice(question),
                          query=sha(question.params["text"]))

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
# The prompt text is the whole contract: any semantic change edits the
# text, and so keys a new ask (spec 3 roster).

@dataclass
class LlmBackend:
    """Costed on the same currencies as assessor.JudgeBackend (spec 3
    section 2): `price` prices the completion's actual token usage
    (api/batch, cash); `quota_cost_per_call` is the cli transport's flat
    subscription-quota cost, which reports no usage on the wire.
    `recognize` is the producer's own check on a completion's text (spec
    3 r10 section 2); the default recognizes any text. `fetch` appends
    nothing for a completion `recognize` rejects.
    """
    producer: str
    model: str
    transport: Any  # .complete(prompt: str) -> Completion; may raise TransportError
    price: Price | None = None
    quota_cost_per_call: float = 0.0
    recognize: Callable[[str], bool] = lambda text: True

    def cache_key(self, question: Question) -> LlmPromptKey:
        return LlmPromptKey(producer=self.producer, model=self.model,
                            prompt_sha=sha(question.params["prompt"]))

    def _cost(self, completion: Completion) -> float:
        if self.price is not None:
            return self.price.cost(completion)
        return self.quota_cost_per_call

    def fetch(self, question: Question) -> RawAnswer:
        prompt = question.params["prompt"]
        completion = self.transport.complete(prompt)
        if not self.recognize(completion.text):
            raise TransportError(
                f"{self.producer} answered without a recognizable answer: "
                f"{completion.text[:80]!r}")
        items = (completion.text,) if completion.text else ()
        return RawAnswer(items=items, cost=self._cost(completion))


# --- pair-search: minimal pairs over a dictionary + G2P ---------------------

@runtime_checkable
class DictionaryG2P(Protocol):
    """A dictionary+G2P corpus searchable for minimal-pair candidates.
    The implementation is the caller's: this module ships none.
    """
    def version(self) -> str: ...
    def search(self, confusion_id: str) -> Sequence[Mapping[str, Any]]: ...


@dataclass
class PairSearchBackend:
    dictionary: DictionaryG2P

    def cache_key(self, question: Question) -> PairSearchKey:
        return PairSearchKey(confusion_id=question.params["confusion_id"],
                             dictionary_version=self.dictionary.version())

    def fetch(self, question: Question) -> RawAnswer:
        confusion_id = question.params["confusion_id"]
        candidates = self.dictionary.search(confusion_id)
        return RawAnswer(items=tuple(candidates), cost=0.0)
