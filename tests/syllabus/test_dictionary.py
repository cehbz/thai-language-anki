"""dictionary.Wiktionary (design 2026-09-20 §2): the record is read before
the wire, one row per form ever, readings converted through
engines.convert_wiktionary; a transport failure kills the source for the
rest of the run and writes nothing, while a 429 or a 5xx only backs off;
at most `max_asks` wire fetches per run, and `begin_run()` starts the
count, the back-off and `dead` over.
"""
from types import SimpleNamespace

import pytest

from thai_syllabus.cachekeys import DictionaryKey
from thai_syllabus.dictionary import Wiktionary
from thai_syllabus.entities import Syllable
from thai_syllabus.store import SyllabusDb

_ROW = ('<tr><th>(<i><a>standard</a></i>) <a>IPA</a><sup>(<a>key</a>)</sup></th>'
        '<td><span class="IPA">/ma˦˥.ka˨˩.raː˧/</span></td>'
        '<td><span class="IPA">/mok̚˦˥.ka˨˩.raː˧/</span></td></tr>')

MA = (Syllable(segments=("m", "a", ""), vowel_length="short", tone="high"),
      Syllable(segments=("k", "a", ""), vowel_length="short", tone="low"),
      Syllable(segments=("r", "a", ""), vowel_length="long", tone="mid"))
MOK = (Syllable(segments=("m", "o", "k"), vowel_length="short", tone="high"),) + MA[1:]


def _page(html, revid=64969808):
    return SimpleNamespace(status_code=200,
                           json=lambda: {"parse": {"revid": revid, "text": {"*": html}}})


def _missing():
    return SimpleNamespace(status_code=200,
                           json=lambda: {"error": {"code": "missingtitle"}})


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


def test_a_form_is_fetched_once_and_its_readings_recorded(db):
    calls = []
    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, params, headers))
        return _page("<table>" + _ROW + "</table>")
    w = Wiktionary(db, get=get, min_interval_s=0)
    assert w("มกรา") == (MA, MOK)
    assert w("มกรา") == (MA, MOK)
    assert len(calls) == 1
    url, params, headers = calls[0]
    assert url == "https://en.wiktionary.org/w/api.php"
    assert params["page"] == "มกรา" and params["action"] == "parse"
    assert "text" in params["prop"] and "revid" in params["prop"]
    assert headers["User-Agent"].startswith("thai-syllabus/")
    row = db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="มกรา"))
    assert row.subject == "dictionary:มกรา" and row.cost == 0
    assert row.answer == {"revid": 64969808, "ipa": ["ma˦˥.ka˨˩.raː˧", "mok̚˦˥.ka˨˩.raː˧"]}


def test_an_absent_entry_is_recorded_and_never_refetched(db):
    calls = []
    def get(url, params=None, headers=None, timeout=None):
        calls.append(1); return _missing()
    w = Wiktionary(db, get=get, min_interval_s=0)
    assert w("ฟอยล์") == ()
    assert w("ฟอยล์") == ()
    assert len(calls) == 1
    row = db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="ฟอยล์"))
    assert row.answer == {"absent": True}


def test_an_entry_without_the_ipa_row_is_recorded_as_no_readings(db):
    w = Wiktionary(db, get=lambda *a, **k: _page("<table><tr><th>Paiboon</th></tr></table>"),
                   min_interval_s=0)
    assert w("วันอังคาร") == ()
    row = db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="วันอังคาร"))
    assert row.answer == {"revid": 64969808, "ipa": []}


def test_a_recorded_row_is_read_before_the_wire(db):
    db.append(port="provide", backend="wiktionary",
              key=DictionaryKey(source="wiktionary", form="หุง"), subject="หุง",
              question={"form": "หุง"}, answer={"revid": 1, "ipa": ["huŋ˩˩˦"]})
    def get(*a, **k):
        raise AssertionError("the wire must not be touched")
    w = Wiktionary(db, get=get, min_interval_s=0)
    assert w("หุง") == ((Syllable(segments=("h", "u", "ŋ"), vowel_length="short", tone="rising"),),)


def test_a_wire_failure_kills_the_source_and_writes_nothing(db, caplog):
    import requests
    calls = []
    def get(*a, **k):
        calls.append(1); raise requests.ConnectionError("down")
    w = Wiktionary(db, get=get, min_interval_s=0)
    assert w("มกรา") == ()
    assert w.dead is True
    assert w("หุง") == ()
    assert len(calls) == 1
    assert db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="มกรา")) is None
    assert "wiktionary" in caplog.text.lower()


def test_an_unexpected_status_is_a_wire_failure(db):
    """A 404 (or any 4xx but 429) is the request itself being wrong, not
    the server asking for patience: it kills the source as before."""
    w = Wiktionary(db, get=lambda *a, **k: _status(404), min_interval_s=0)
    assert w("มกรา") == ()
    assert w.dead is True


def test_a_non_json_body_is_a_wire_failure(db):
    def bad_json():
        raise ValueError("not JSON")
    w = Wiktionary(db, get=lambda *a, **k: SimpleNamespace(status_code=200, json=bad_json),
                   min_interval_s=0)
    assert w("มกรา") == ()
    assert w.dead is True
    assert db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="มกรา")) is None


def test_a_body_without_a_well_formed_parse_result_is_a_wire_failure(db):
    w = Wiktionary(db, get=lambda *a, **k: SimpleNamespace(status_code=200,
                                                            json=lambda: {"unexpected": 1}),
                   min_interval_s=0)
    assert w("มกรา") == ()
    assert w.dead is True
    assert db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="มกรา")) is None


def test_requests_are_paced(db):
    slept, now = [], [0.0]
    w = Wiktionary(db, get=lambda *a, **k: _missing(), min_interval_s=1.0,
                   sleep=lambda s: slept.append(s), clock=lambda: now[0])
    w("ก"); w("ข")
    assert slept == [1.0]


def test_a_phrase_is_never_looked_up(db):
    def get(*a, **k):
        raise AssertionError("no lookup for a phrase")
    w = Wiktionary(db, get=get, min_interval_s=0)
    assert w("งอ งู") == ()


def _status(code, headers=None):
    return SimpleNamespace(status_code=code, headers=headers or {}, json=lambda: {})


def test_a_429_backs_off_without_dying_and_the_next_form_is_still_fetched(db, caplog):
    """The live probe hit a 429 after about 19 lookups at one request a
    second (2026-09-21): Wikimedia asking for patience, not a broken
    source. Nothing is recorded (a 429 carries no readings) and the
    oracle stays alive, so the run's later forms are still looked up --
    at a doubled interval."""
    seen = []
    def get(*a, **k):
        seen.append(k["params"]["page"])
        return _status(429) if len(seen) == 1 else _page("<table>" + _ROW + "</table>")
    slept, now = [], [0.0]
    w = Wiktionary(db, get=get, min_interval_s=1.0,
                   sleep=lambda s: slept.append(s), clock=lambda: now[0])
    assert w("น้ำ") == ()
    assert w.dead is False
    assert db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="น้ำ")) is None
    assert w("มกรา") == (MA, MOK)
    assert seen == ["น้ำ", "มกรา"]
    assert slept == [2.0]                     # the interval doubled for the next ask
    assert "429" in caplog.text


def test_a_numeric_retry_after_sets_the_next_wait(db):
    slept, now = [], [0.0]
    w = Wiktionary(db, get=lambda *a, **k: _status(429, {"Retry-After": "5"}),
                   min_interval_s=1.0, sleep=lambda s: slept.append(s), clock=lambda: now[0])
    assert w("น้ำ") == ()
    assert w("มกรา") == ()
    assert slept and slept[0] >= 5.0


def test_a_5xx_backs_off_without_dying(db):
    w = Wiktionary(db, get=lambda *a, **k: _status(503), min_interval_s=0)
    assert w("มกรา") == ()
    assert w.dead is False


def test_the_run_s_wire_fetches_are_capped(db, caplog):
    """A per-run ceiling on how much of the run's time the dictionary may
    spend on the wire (and how hard the run leans on Wikimedia): past it
    a record miss is simply no reading, with no request and no row."""
    calls = []
    def get(*a, **k):
        calls.append(k["params"]["page"])
        return _page("<table>" + _ROW + "</table>")
    w = Wiktionary(db, get=get, min_interval_s=0, max_asks=2)
    assert w("มกรา") == (MA, MOK)
    assert w("หุง") == (MA, MOK)
    assert w("น้ำ") == ()
    assert calls == ["มกรา", "หุง"]
    assert db.latest("provide", "wiktionary", DictionaryKey(source="wiktionary", form="น้ำ")) is None
    assert "max_asks" in caplog.text or "cap" in caplog.text
    assert w("มกรา") == (MA, MOK)             # the record still answers
    assert calls == ["มกรา", "หุง"]


def test_begin_run_starts_the_count_the_backoff_and_dead_over(db):
    import requests
    calls = []
    def get(*a, **k):
        calls.append(1); raise requests.ConnectionError("down")
    w = Wiktionary(db, get=get, min_interval_s=0)
    assert w("มกรา") == ()
    assert w.dead is True
    w.begin_run()
    assert w.dead is False
    assert w("มกรา") == ()
    assert len(calls) == 2
