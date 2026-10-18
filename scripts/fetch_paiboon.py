"""Fetch thai-language.com's Paiboon reading of every word form in a deck.

    uv run python scripts/fetch_paiboon.py <deck-dir> [<out-dir>]

<out-dir> defaults to data/paiboon/ (gitignored: the site's content is
copyrighted). One dictionary search per form, at the site's robots.txt
crawl delay of 10 s; each answer is kept raw in <out-dir>/html/ and a form
already fetched is not fetched again, so an interrupted run resumes.
<out-dir>/paiboon.tsv holds `form, entry id, paiboon` for every result whose
headword is exactly the form, and a row with empty id and reading for a
form the site has no entry for.

The site's control panel picks the romanization; its choice travels in the
`ctrl` cookie, so the script saves Paiboon (system 8) once per session.
The site serves a self-signed certificate, hence plain HTTP.
"""
import html
import http.cookiejar
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from thai_syllabus.wiring import load_syllabus

SITE = "http://www.thai-language.com"
CRAWL_DELAY = 10.0
USER_AGENT = "thai-syllabus/0.1"
SETTINGS = {"xlitsystem": "8", "xs0": "on", "xs8": "on", "xlitshowmode": "0",
            "submitted": "save changes"}
RESULT = re.compile(
    r"<td class=th[^>]*><a href='/id/(\d+)[^']*'>(?:<img[^>]*>&nbsp;)?([^<]+)</a>"
    r".*?</td><td[^>]*><span class='xf'>([^<]*)</span>", re.S)


class Site:
    def __init__(self):
        jar = http.cookiejar.CookieJar()
        self._open = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar)).open
        self._last = 0.0

    def post(self, path: str, fields: dict) -> str:
        wait = self._last + CRAWL_DELAY - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            SITE + path, data=urllib.parse.urlencode(fields).encode(),
            headers={"User-Agent": USER_AGENT})
        try:
            with self._open(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        finally:
            self._last = time.monotonic()


def readings(page: str, form: str) -> list[tuple[str, str]]:
    seen = []
    for entry, head, paiboon in RESULT.findall(page):
        pair = (entry, html.unescape(paiboon).strip())
        if html.unescape(head).strip() == form and pair not in seen:
            seen.append(pair)
    return seen


def main(deck: str, out: str = "data/paiboon") -> None:
    out_dir = Path(out)
    raw = out_dir / "html"
    raw.mkdir(parents=True, exist_ok=True)
    forms = sorted({w.thai for w in load_syllabus(deck).words})
    site = None
    for i, form in enumerate(forms, 1):
        page = raw / f"{form}.html"
        if page.exists():
            continue
        if site is None:
            site = Site()
            site.post("/default.aspx?nav=control", SETTINGS)
        page.write_text(site.post("/dict", {"search": form, "emode": "1", "tmode": "2"}),
                        encoding="utf-8")
        print(f"{i}/{len(forms)} {form}", flush=True)
    rows = []
    for form in forms:
        found = readings((raw / f"{form}.html").read_text(encoding="utf-8"), form)
        rows += [f"{form}\t{e}\t{p}" for e, p in found] or [f"{form}\t\t"]
    (out_dir / "paiboon.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    hit = sum(1 for f in forms
              if readings((raw / f"{f}.html").read_text(encoding="utf-8"), f))
    print(f"{hit} of {len(forms)} forms have an entry")


if __name__ == "__main__":
    main(*sys.argv[1:])
