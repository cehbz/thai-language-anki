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
from typing import Any

import requests

from .cachekeys import DictionaryKey
from .engines import _convert_wiktionary, extract_standard_ipa
from .entities import Syllable
from .ports import CacheReader, RecordWriter

_log = logging.getLogger(__name__)

API_URL = "https://en.wiktionary.org/w/api.php"
SOURCE = "wiktionary"
DEFAULT_USER_AGENT = "thai-syllabus/0.1"

Reading = tuple[Syllable, ...]


class Wiktionary:
    """One lookup per form ever: the record row (`DictionaryKey`) is read
    first; on a miss the API's rendered page is fetched once, its
    "(standard) IPA" readings recorded, and the row answers every later
    ask. A wire failure (transport error or non-200) kills the source for
    the rest of the process (`dead`) and writes nothing, so the next run
    asks again. A phrase (whitespace) is never looked up (§3).
    """

    def __init__(self, cache: CacheReader | RecordWriter, *,
                 get: Callable[..., Any] = requests.get,
                 min_interval_s: float = 1.0,
                 user_agent: str = DEFAULT_USER_AGENT,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._cache = cache
        self._get = get
        self._min_interval_s = min_interval_s
        self._user_agent = user_agent
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self.dead = False

    def __call__(self, form: str) -> tuple[Reading, ...]:
        if " " in form:
            return ()
        key = DictionaryKey(source=SOURCE, form=form)
        row = self._cache.latest("provide", SOURCE, key)
        if row is None:
            answer = self._fetch(form)
            if answer is None:
                return ()
            self._cache.append(port="provide", backend=SOURCE, key=key, subject=form,
                               question={"form": form}, answer=answer, cost=0.0)
        else:
            answer = row.answer
        readings = [_convert_wiktionary(ipa) for ipa in answer.get("ipa", [])]
        return tuple(r for r in readings if r)

    def _fetch(self, form: str) -> dict | None:
        if self.dead:
            return None
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

    def _pace(self) -> None:
        if self._min_interval_s > 0 and self._last_call is not None:
            remaining = self._last_call + self._min_interval_s - self._clock()
            if remaining > 0:
                self._sleep(remaining)
