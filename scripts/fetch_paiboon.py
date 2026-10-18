"""Fetch thai-language.com's Paiboon and IPA readings of every word form in
a deck.

    uv run python scripts/fetch_paiboon.py <deck-dir> [<out-dir>]

<out-dir> defaults to data/paiboon/ (gitignored: the site's content is
copyrighted). One dictionary search per form gives each exact headword
match's entry id and Paiboon reading; one request per matched entry gives
its IPA (tone bars, the deck's Chao letters). Requests keep the site's
robots.txt crawl delay of 10 s; each answer is kept raw in <out-dir>/html/
(entries under html/id/) and nothing already fetched is fetched again, so
an interrupted run resumes. <out-dir>/paiboon.tsv holds `form, entry id,
paiboon, ipa` for every match, the entry's readings joined by ` | `, and a
row with only the form for a form the site has no entry for.

The site's control panel picks the romanizations; its choice travels in
the `ctrl` cookie, so the script saves Paiboon (system 8) with IPA shown
(system 2, tone bars) once per session. The site serves a self-signed
certificate, hence plain HTTP.
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
SETTINGS = {"xlitsystem": "8", "xs0": "on", "xs2": "on", "xs8": "on",
            "ipa-mode": "2", "ipa-init-glottal": "on", "ipa-final-glottal": "on",
            "xlitshowmode": "0", "submitted": "save changes"}
RESULT = re.compile(
    r"<td class=th[^>]*><a href='/id/(\d+)[^']*'>(?:<img[^>]*>&nbsp;)?([^<]+)</a>"
    r".*?</td><td[^>]*><span class='xf'>([^<]*)</span>", re.S)
IPA = re.compile(r"<td style='font-size:x-small;'>IPA</td><td><span class='ipa'>([^<]*)</span>")


class Site:
    def __init__(self):
        jar = http.cookiejar.CookieJar()
        self._open = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar)).open
        self._last = 0.0
        self.post("/default.aspx?nav=control", SETTINGS)

    def post(self, path: str, fields: dict | None = None) -> str:
        wait = self._last + CRAWL_DELAY - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        data = urllib.parse.urlencode(fields).encode() if fields is not None else None
        req = urllib.request.Request(SITE + path, data=data,
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


def ipa(page: str) -> list[str]:
    return list(dict.fromkeys(html.unescape(m).strip() for m in IPA.findall(page)))


class Cache:
    def __init__(self, raw: Path):
        self._raw = raw
        self._site = None

    def page(self, name: str, path: str, fields: dict | None = None) -> str:
        file = self._raw / f"{name}.html"
        if not file.exists():
            if self._site is None:
                self._site = Site()
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(self._site.post(path, fields), encoding="utf-8")
            print(name, flush=True)
        return file.read_text(encoding="utf-8")


def main(deck: str, out: str = "data/paiboon") -> None:
    out_dir = Path(out)
    cache = Cache(out_dir / "html")
    forms = sorted({w.thai for w in load_syllabus(deck).words})
    found = {f: readings(cache.page(f, "/dict", {"search": f, "emode": "1", "tmode": "2"}), f)
             for f in forms}
    rows = []
    for form in forms:
        for entry, paiboon in found[form]:
            rows.append(f"{form}\t{entry}\t{paiboon}\t"
                        + " | ".join(ipa(cache.page(f"id/{entry}", f"/id/{entry}"))))
        if not found[form]:
            rows.append(f"{form}\t\t\t")
    (out_dir / "paiboon.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{sum(1 for f in forms if found[f])} of {len(forms)} forms have an entry")


if __name__ == "__main__":
    main(*sys.argv[1:])
