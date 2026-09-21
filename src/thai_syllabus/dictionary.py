"""The dictionary oracle (design 2026-09-20 §2): English Wiktionary's
generated IPA for a Thai form, read from the record before the wire.
Engines compute; a dictionary looks up -- so this is a Provide-port
backend with I/O, beside provider.py, and never an `Engines.g2p` entry.
`phonology.Engines.dictionary` consults it lazily.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

import requests

from .cachekeys import DictionaryKey
from .engines import convert_wiktionary, extract_standard_ipa
from .entities import Syllable, is_phrase

_log = logging.getLogger(__name__)

API_URL = "https://en.wiktionary.org/w/api.php"
SOURCE = "wiktionary"
DEFAULT_USER_AGENT = "thai-syllabus/0.1"

# The default ceiling on wire fetches in one run (spec 3 section 8,
# `quotas.wiktionary.max_asks`). A run adopting a fresh vocabulary would
# otherwise look up every unknown form in it one second apart; 200 is
# some minutes of wire time, and what the cap does not reach this run
# the next run picks up -- every answer is on the record for good.
DEFAULT_MAX_ASKS = 200

# The back-off ceiling when Wikimedia asks for patience (429) or is
# unwell (5xx): the interval doubles per such answer, but never past
# this, since a run that waits a minute between lookups has effectively
# stopped asking anyway.
MAX_BACKOFF_S = 60.0

Reading = tuple[Syllable, ...]


class DictionaryStore(Protocol):
    """The two record operations a dictionary needs -- the read side's
    `latest` (ports.CacheReader) and the write side's `append`
    (ports.RecordWriter), named together because one store implements
    both and a dictionary needs exactly these two. Spelling it
    `CacheReader | RecordWriter` said "either one will do", which is
    false: a reader alone could not record what it fetched.
    """

    def latest(self, port: str, backend: str, key: Any) -> Any: ...

    def append(self, port: str, backend: str, key: Any, subject: str,
               question: Any, answer: Any, cost: float = 0.0) -> int: ...


class Wiktionary:
    """One lookup per form ever: the record row (`DictionaryKey`) is read
    first; on a miss the API's rendered page is fetched once, its
    "(standard) IPA" readings recorded, and the row answers every later
    ask. A phrase (whitespace) is never looked up (§3).

    What a failed fetch means depends on the failure (design 2026-09-20
    §2, spec 3 r50 §6a):

    - a transport error, a non-JSON or non-dict body, an API error or a
      malformed parse result kills the source for the rest of the run
      (`dead`) and writes nothing, so the next run asks again;
    - a 429 (Wikimedia asking for patience -- the live probe met one
      after about 19 lookups at one request a second, 2026-09-21) or a
      5xx (Wikimedia unwell) is no reading this time and nothing
      recorded, but the oracle stays alive at a longer interval: a
      numeric `Retry-After` when the response carries one, else double
      the current interval, capped at `MAX_BACKOFF_S`.

    At most `max_asks` wire fetches per run, attempts counted whether
    they succeed or not; past the cap a record miss is `()` with no
    request. `begin_run()` starts the run's count, interval, `dead` and
    once-per-run logging over.
    """

    def __init__(self, cache: DictionaryStore, *,
                 get: Callable[..., Any] = requests.get,
                 min_interval_s: float = 1.0,
                 max_asks: int | None = DEFAULT_MAX_ASKS,
                 user_agent: str = DEFAULT_USER_AGENT,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._cache = cache
        self._get = get
        self._min_interval_s = min_interval_s
        self._max_asks = max_asks
        self._user_agent = user_agent
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self.dead = False
        self._interval_s = min_interval_s
        self._asks = 0
        self._logged_backoff = False
        self._logged_cap = False

    def begin_run(self) -> None:
        """Start a run: the ask count, the back-off interval, `dead` and
        the once-per-run warnings are all per-run state, and one process
        runs the pipeline many times over (cli's cycle loop). Without
        this a 429 in cycle 1 would still be slowing cycle 9, and a dead
        oracle would stay dead for the whole invocation rather than for
        the run that lost it.
        """
        self.dead = False
        self._asks = 0
        self._interval_s = self._min_interval_s
        self._logged_backoff = False
        self._logged_cap = False

    def __call__(self, form: str) -> tuple[Reading, ...]:
        if is_phrase(form):
            return ()
        key = DictionaryKey(source=SOURCE, form=form)
        row = self._cache.latest("provide", SOURCE, key)
        if row is None:
            answer = self._fetch(form)
            if answer is None:
                return ()
            # `dictionary:<form>` namespaces the row the way the pair
            # search namespaces its own outside-the-vocabulary subjects
            # (`candidate:<thai>`): the subject column holds deck
            # subjects, and a looked-up form need not be one.
            self._cache.append(port="provide", backend=SOURCE, key=key,
                               subject=f"dictionary:{form}",
                               question={"form": form}, answer=answer, cost=0.0)
        else:
            answer = row.answer
        readings = [convert_wiktionary(ipa) for ipa in answer.get("ipa", [])]
        return tuple(r for r in readings if r)

    def _fetch(self, form: str) -> dict | None:
        if self.dead:
            return None
        if self._max_asks is not None and self._asks >= self._max_asks:
            if not self._logged_cap:
                self._logged_cap = True
                _log.warning("%s: max_asks (%d) reached; no more lookups this run",
                             SOURCE, self._max_asks)
            return None
        self._asks += 1
        self._pace()
        try:
            resp = self._get(API_URL,
                             params={"action": "parse", "page": form,
                                     "prop": "text|revid", "format": "json",
                                     "formatversion": "1"},
                             headers={"User-Agent": self._user_agent}, timeout=30)
        except requests.RequestException as e:
            return self._die(f"{SOURCE}: request failed: {e}")
        finally:
            self._last_call = self._clock()
        if resp.status_code == 429 or resp.status_code >= 500:
            return self._back_off(resp)
        if resp.status_code != 200:
            return self._die(f"{SOURCE}: HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError as e:
            return self._die(f"{SOURCE}: invalid JSON body: {e}")
        if not isinstance(body, dict):
            return self._die(f"{SOURCE}: unexpected JSON body shape")
        if "error" in body:
            error = body["error"]
            code = error.get("code") if isinstance(error, dict) else error
            if code == "missingtitle":
                return {"absent": True}
            return self._die(f"{SOURCE}: API error {code}")
        parse = body.get("parse")
        if (not isinstance(parse, dict) or not isinstance(parse.get("revid"), int)
                or not isinstance(parse.get("text"), dict)):
            return self._die(f"{SOURCE}: malformed parse result")
        html = parse["text"].get("*", "")
        return {"revid": parse["revid"], "ipa": extract_standard_ipa(html)}

    def _die(self, reason: str) -> None:
        self.dead = True
        _log.warning("%s; the dictionary is dead for the rest of this run", reason)
        return None

    def _back_off(self, resp: Any) -> None:
        """A 429 or a 5xx: no reading, nothing recorded, still alive, and
        a longer interval for the rest of the run."""
        self._interval_s = min(self._retry_after(resp) or self._interval_s * 2,
                               MAX_BACKOFF_S)
        if not self._logged_backoff:
            self._logged_backoff = True
            _log.warning("%s: HTTP %s; backing off to %.3g s between lookups",
                         SOURCE, resp.status_code, self._interval_s)
        return None

    @staticmethod
    def _retry_after(resp: Any) -> float | None:
        """The response's `Retry-After` as seconds, when it carries one as
        a plain number. The HTTP-date form is not honoured: it needs a
        wall clock to read, and the doubling below is a safe fallback."""
        value = getattr(resp, "headers", None) or {}
        try:
            seconds = float(value.get("Retry-After", ""))
        except (AttributeError, TypeError, ValueError):
            return None
        return seconds if seconds > 0 else None

    def _pace(self) -> None:
        if self._interval_s > 0 and self._last_call is not None:
            remaining = self._last_call + self._interval_s - self._clock()
            if remaining > 0:
                self._sleep(remaining)
