"""The thai-language.com dictionary oracle: the site's IPA for a Thai
form, read from the record before the wire. A Provide-port backend with
I/O beside `dictionary.Wiktionary`, with the same shape: one lookup per
form ever, `max_asks` forms per run, a phrase never asked.

A lookup is a dictionary search for the form (every exactly-matching
headword's entry id and Paiboon reading), then one request per entry for
its IPA. The IPA is kept only where the entry page's title is the form:
a compound listed as a sense of its base word (การแสดง under แสดง) links
to the base word's page, whose IPA is the base word's. The record row
keeps the raw strings, so the conversion can be re-run without the wire.

The site's control panel picks the romanizations and its choice travels
in the `ctrl` cookie, so a session saves Paiboon with IPA (tone bars,
ʔ shown) first. It serves a self-signed certificate, hence plain HTTP;
its robots.txt asks for a 10 s crawl delay between requests.
"""
from __future__ import annotations

import html
import http.cookiejar
import logging
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable

from .cachekeys import DictionaryKey
from .dictionary import DEFAULT_USER_AGENT, DictionaryStore, Reading
from .engines import convert_thai_language_com
from .entities import is_phrase

_log = logging.getLogger(__name__)

SITE = "http://www.thai-language.com"
SOURCE = "thai-language.com"
CRAWL_DELAY_S = 10.0

# Per run, in forms (spec 3 section 8, `quotas.thai-language.com.max_asks`):
# a form costs a search and an entry page per match, 10 s apart.
DEFAULT_MAX_ASKS = 30

SETTINGS_PATH = "/default.aspx?nav=control"
SETTINGS = {"xlitsystem": "8", "xs0": "on", "xs2": "on", "xs8": "on",
            "ipa-mode": "2", "ipa-init-glottal": "on", "ipa-final-glottal": "on",
            "xlitshowmode": "0", "submitted": "save changes"}

RESULT = re.compile(
    r"<td class=th[^>]*><a href='/id/(\d+)[^']*'>(?:<img[^>]*>&nbsp;)?([^<]+)</a>"
    r".*?</td><td[^>]*><span class='xf'>([^<]*)</span>", re.S)
IPA = re.compile(r"<td style='font-size:x-small;'>IPA</td><td><span class='ipa'>([^<]*)</span>")
TITLE = re.compile(r"<title>thai-language.com - ([^<]*)")

# post(path, fields): the page body; GET when fields is None.
Post = Callable[[str, "dict | None"], str]


def search_entries(page: str, form: str) -> list[tuple[str, str]]:
    """(entry id, Paiboon) of each search result whose headword is `form`."""
    seen = []
    for entry, head, paiboon in RESULT.findall(page):
        pair = (entry, html.unescape(paiboon).strip())
        if html.unescape(head).strip() == form and pair not in seen:
            seen.append(pair)
    return seen


def entry_ipa(page: str, form: str) -> list[str]:
    """The IPA strings of an entry page, or none when its title is not `form`."""
    title = TITLE.search(page)
    if title is None or html.unescape(title.group(1)).strip() != form:
        return []
    return list(dict.fromkeys(html.unescape(m).strip() for m in IPA.findall(page)))


def entry_row(search_page: str, form: str,
              page_of: Callable[[str], str | None]) -> dict | None:
    """The record answer for `form` from its search page: one entry per
    exactly-matching headword, `{"id", "paiboon", "ipa"}`, no entries for a
    form the site lacks. `page_of(id)` gives an entry's page; None when it
    cannot be had, which leaves the form without an answer."""
    entries = []
    for entry, paiboon in search_entries(search_page, form):
        page = page_of(entry)
        if page is None:
            return None
        entries.append({"id": entry, "paiboon": paiboon, "ipa": entry_ipa(page, form)})
    return {"entries": entries}


def readings(answer: dict) -> tuple[Reading, ...]:
    """The readings a record answer converts to, deduplicated."""
    converted = (convert_thai_language_com(ipa)
                 for entry in answer.get("entries", []) for ipa in entry.get("ipa", []))
    return tuple(dict.fromkeys(r for r in converted if r))


class Site:
    """The wire: cookies kept across requests, plain HTTP."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self._open = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())).open
        self._user_agent = user_agent

    def __call__(self, path: str, fields: dict | None = None) -> str:
        data = urllib.parse.urlencode(fields).encode() if fields is not None else None
        req = urllib.request.Request(SITE + path, data=data,
                                     headers={"User-Agent": self._user_agent})
        with self._open(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")


class ThaiLanguageCom:
    """One lookup per form ever: the record row (`DictionaryKey`) is read
    first; on a miss the site is searched and each matching entry fetched,
    and the row answers every later ask. A phrase is never looked up.

    A transport error kills the source for the rest of the run (`dead`)
    and writes nothing, so the next run asks again. At most `max_asks`
    forms are fetched per run, attempts counted whether they succeed or
    not; past the cap a record miss is `()` with no request.
    `begin_run()` starts the count and `dead` over.
    """

    def __init__(self, cache: DictionaryStore, *,
                 post: Post | None = None,
                 max_asks: int | None = DEFAULT_MAX_ASKS,
                 user_agent: str = DEFAULT_USER_AGENT,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._cache = cache
        self._post = post
        self._max_asks = max_asks
        self.user_agent = user_agent
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self._session = False
        self.dead = False
        self._asks = 0
        self._logged_cap = False

    def begin_run(self) -> None:
        self.dead = False
        self._asks = 0
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
            self._cache.append(port="provide", backend=SOURCE, key=key,
                               subject=f"dictionary:{form}",
                               question={"form": form}, answer=answer, cost=0.0)
        else:
            answer = row.answer
        return readings(answer)

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
        try:
            if not self._session:
                self._request(SETTINGS_PATH, SETTINGS)
                self._session = True
            search = self._request("/dict", {"search": form, "emode": "1", "tmode": "2"})
            return entry_row(search, form, lambda entry: self._request(f"/id/{entry}"))
        except OSError as e:
            _log.warning("%s: request failed: %s; the dictionary is dead for the rest "
                         "of this run", SOURCE, e)
            self.dead = True
            return None

    def _request(self, path: str, fields: dict | None = None) -> str:
        if self._post is None:
            self._post = Site(self.user_agent)
        if self._last_call is not None:
            remaining = self._last_call + CRAWL_DELAY_S - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        try:
            return self._post(path, fields)
        finally:
            self._last_call = self._clock()
