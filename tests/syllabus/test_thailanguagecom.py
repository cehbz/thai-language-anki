"""thailanguagecom.ThaiLanguageCom: thai-language.com as a dictionary
beside Wiktionary. The record row is read before the wire; a miss
searches the form, fetches each exactly-matching entry's page and keeps
its IPA only where the page's title is the form (a compound listed under
its base word links to the base word's page). The row keeps the raw IPA
and entry ids so conversion can be re-run; a form with no entry is a row
too. 10 s between requests, `max_asks` forms per run, a phrase never.
"""
import pytest

from thai_syllabus.cachekeys import DictionaryKey
from thai_syllabus.engines import convert_thai_language_com
from thai_syllabus.entities import Syllable
from thai_syllabus.store import SyllabusDb
from thai_syllabus.thailanguagecom import (CRAWL_DELAY_S, SOURCE, ThaiLanguageCom,
                                           entry_ipa, entry_row, search_entries)


def _search(*hits):
    """A search page: (entry id, headword, paiboon) rows, the first of an
    entry carrying the image the site puts on its headline hit."""
    rows = "".join(
        f"<tr><td class=th rowspan=2><a href='/id/{i}'><img src='/img/star.gif'>&nbsp;{h}</a>"
        f" <a href='/mp3/E{i}.mp3'><img src='/img/speaker_sm.gif' /></a></td>"
        f"<td rowspan=2><span class='xf'>{p}</span></td></tr>" for i, h, p in hits)
    return f"<html><body><table>{rows}</table></body></html>"


def _entry(title, *ipa):
    cells = "".join(f"<tr><td style='font-size:x-small;'>IPA</td>"
                    f"<td><span class='ipa'>{i}</span></td></tr>" for i in ipa)
    return (f"<html><head><title>thai-language.com - {title}</title></head>"
            f"<body><table>{cells}</table></body></html>")


def syl(onset, vowel, coda, length, tone):
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


class Site:
    """A fake transport: `pages` maps a path to a page; every call is kept."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def __call__(self, path, fields=None):
        self.calls.append((path, fields))
        if path == "/dict":
            return self.pages[fields["search"]]
        return self.pages.get(path, "")

    @property
    def lookups(self):
        return [c for c in self.calls if c[0] == "/dict" or c[0].startswith("/id/")]


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


def _backend(db, pages, **kw):
    site = Site(pages)
    kw.setdefault("sleep", lambda s: None)
    return ThaiLanguageCom(db, post=site, **kw), site


# --- parsing -------------------------------------------------------------

def test_search_keeps_only_exact_headword_matches_once_per_entry_and_reading():
    page = _search((1, "กบ", "gòp"), (2, "กบฏ", "gà-bòt"), (1, "กบ", "gòp"), (3, "กบ", "gòp-2"))
    assert search_entries(page, "กบ") == [("1", "gòp"), ("3", "gòp-2")]


def test_search_reads_the_id_of_a_sense_link():
    page = ("<td class=th><a href='/id/216956#def2'>ก็ได้</a></td>"
            "<td><span class='xf'>gɔ̂ɔ dâi</span>")
    assert search_entries(page, "ก็ได้") == [("216956", "gɔ̂ɔ dâi")]


def test_entry_ipa_needs_the_page_title_to_be_the_form():
    assert entry_ipa(_entry("กบ", "kop̚˨˩", "kop̚˨˩"), "กบ") == ["kop̚˨˩"]
    assert entry_ipa(_entry("แสดง", "sa˩˥.dɛːŋ˧"), "การแสดง") == []


def test_entry_row_drops_the_ipa_of_a_compounds_base_word_page():
    page = _search((7, "การแสดง", "gaan sa-dɛɛng"))
    row = entry_row(page, "การแสดง", lambda i: _entry("แสดง", "sa˩˥.dɛːŋ˧"))
    assert row == {"entries": [{"id": "7", "paiboon": "gaan sa-dɛɛng", "ipa": []}]}


# --- conversion ----------------------------------------------------------

@pytest.mark.parametrize("ipa,tone", [("maː˧", "mid"), ("maː˨", "low"), ("maː˥˩", "falling"),
                                      ("maː˦", "high"), ("maː˩˥", "rising")])
def test_the_five_tone_bars_convert(ipa, tone):
    assert convert_thai_language_com(ipa) == (syl("m", "a", "", "long", tone),)


def test_length_and_codas_convert_and_a_glottal_coda_is_dropped():
    assert convert_thai_language_com("kop̚˨") == (syl("k", "o", "p", "short", "low"),)
    assert convert_thai_language_com("t͡ɕaʔ˨ kʰaːw˩˥") == (
        syl("tɕ", "a", "", "short", "low"), syl("kʰ", "a", "w", "long", "rising"))


def test_unmappable_ipa_converts_to_none():
    assert convert_thai_language_com("maː˧˧˧") is None
    assert convert_thai_language_com("") is None


# --- the backend ---------------------------------------------------------

def test_a_form_is_searched_then_its_entry_fetched_and_recorded(db):
    pages = {"กบ": _search((11, "กบ", "gòp"), (12, "กบฏ", "gà-bòt")),
             "/id/11": _entry("กบ", "kop̚˨")}
    d, site = _backend(db, pages)
    assert d("กบ") == ((syl("k", "o", "p", "short", "low"),),)
    assert [c[0] for c in site.lookups] == ["/dict", "/id/11"]
    key = DictionaryKey(source=SOURCE, form="กบ")
    row = db.latest("provide", SOURCE, key)
    assert row.subject == "dictionary:กบ" and row.cost == 0
    assert row.answer == {"entries": [{"id": "11", "paiboon": "gòp", "ipa": ["kop̚˨"]}]}


def test_the_search_asks_the_site_for_a_thai_headword_search(db):
    d, site = _backend(db, {"กบ": _search()})
    d("กบ")
    assert site.lookups[0] == ("/dict", {"search": "กบ", "emode": "1", "tmode": "2"})


def test_the_record_is_read_before_the_wire(db):
    pages = {"กบ": _search((11, "กบ", "gòp")), "/id/11": _entry("กบ", "kop̚˨")}
    d, site = _backend(db, pages)
    d("กบ")
    n = len(site.calls)
    fresh, site2 = _backend(db, {})
    assert fresh("กบ") == d("กบ")
    assert len(site.calls) == n and site2.calls == []


def test_a_form_with_no_entry_is_recorded_and_never_refetched(db):
    d, site = _backend(db, {"ฟอยล์": _search()})
    assert d("ฟอยล์") == ()
    assert d("ฟอยล์") == ()
    assert len(site.lookups) == 1
    row = db.latest("provide", SOURCE, DictionaryKey(source=SOURCE, form="ฟอยล์"))
    assert row.answer == {"entries": []}


def test_a_compounds_base_word_ipa_is_not_its_reading(db):
    pages = {"การแสดง": _search((7, "การแสดง", "gaan sa-dɛɛng")),
             "/id/7": _entry("แสดง", "sa˩˥.dɛːŋ˧")}
    d, _ = _backend(db, pages)
    assert d("การแสดง") == ()


def test_a_phrase_is_never_looked_up(db):
    d, site = _backend(db, {})
    assert d("กิน ข้าว") == () and d("กิน ข้าว") == ()
    assert site.calls == []


def test_requests_keep_the_crawl_delay_and_the_user_agent_is_plain(db):
    clock = [100.0]
    slept = []
    def sleep(s):
        slept.append(s); clock[0] += s
    pages = {"กบ": _search((11, "กบ", "gòp")), "/id/11": _entry("กบ", "kop̚˨")}
    d, site = _backend(db, pages, sleep=sleep, clock=lambda: clock[0])
    d("กบ")
    assert CRAWL_DELAY_S == 10.0
    # settings POST, search, entry: two waits, none before the first
    assert slept == [CRAWL_DELAY_S, CRAWL_DELAY_S]
    assert site.calls[0][0].startswith("/default.aspx?nav=control")
    assert d.user_agent == "thai-syllabus/0.1"


def test_at_most_max_asks_forms_are_fetched_per_run(db):
    pages = {f: _search() for f in ("กบ", "ขอ", "คน")}
    d, site = _backend(db, pages, max_asks=2)
    assert [d(f) for f in pages] == [(), (), ()]
    assert len(site.lookups) == 2
    assert db.latest("provide", SOURCE, DictionaryKey(source=SOURCE, form="คน")) is None
    d.begin_run()
    d("คน")
    assert len(site.lookups) == 3


def test_a_transport_failure_records_nothing_and_ends_the_run(db):
    def post(path, fields=None):
        raise OSError("down")
    d = ThaiLanguageCom(db, post=post, sleep=lambda s: None)
    assert d("กบ") == () and d.dead
    assert db.latest("provide", SOURCE, DictionaryKey(source=SOURCE, form="กบ")) is None
