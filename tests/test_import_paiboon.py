"""scripts/import_paiboon.py: the cached thai-language.com pages become
the same record rows the live backend writes, once; a form already on
record is skipped, and a form whose entry page was never cached is left
for the live backend.
"""
import importlib.util
from pathlib import Path

from thai_syllabus.cachekeys import DictionaryKey
from thai_syllabus.store import SyllabusDb
from thai_syllabus.thailanguagecom import SOURCE

from tests.syllabus.test_thailanguagecom import _entry, _search

_spec = importlib.util.spec_from_file_location(
    "import_paiboon", Path(__file__).parent.parent / "scripts" / "import_paiboon.py")
import_paiboon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_paiboon)


def _html(tmp_path):
    html = tmp_path / "html"
    (html / "id").mkdir(parents=True)
    (html / "กบ.html").write_text(_search((11, "กบ", "gòp")), encoding="utf-8")
    (html / "id" / "11.html").write_text(_entry("กบ", "kop̚˨"), encoding="utf-8")
    (html / "ฟอยล์.html").write_text(_search(), encoding="utf-8")
    (html / "คน.html").write_text(_search((21, "คน", "kon")), encoding="utf-8")  # entry page missing
    (html / "ขอ.html").write_text(_search((31, "ขอ", "kɔ̌ɔ")), encoding="utf-8")
    (html / "id" / "31.html").write_text(_entry("ขอ", "kʰɔːw˩˥"), encoding="utf-8")
    return html


def test_cached_pages_become_record_rows_and_counts_are_reported(tmp_path):
    db = SyllabusDb(tmp_path / "syllabus.db")
    db.append(port="provide", backend=SOURCE, key=DictionaryKey(source=SOURCE, form="ขอ"),
              subject="dictionary:ขอ", question={"form": "ขอ"}, answer={"entries": []}, cost=0.0)
    report = import_paiboon.import_pages(db, _html(tmp_path))
    assert report == {"imported": 2, "with_ipa": 1, "no_entry": 1, "skipped": 1, "incomplete": 1}
    row = db.latest("provide", SOURCE, DictionaryKey(source=SOURCE, form="กบ"))
    assert row.subject == "dictionary:กบ"
    assert row.answer == {"entries": [{"id": "11", "paiboon": "gòp", "ipa": ["kop̚˨"]}]}
    row = db.latest("provide", SOURCE, DictionaryKey(source=SOURCE, form="ฟอยล์"))
    assert row.answer == {"entries": []}
    assert db.latest("provide", SOURCE, DictionaryKey(source=SOURCE, form="คน")) is None
    assert db.latest("provide", SOURCE, DictionaryKey(source=SOURCE, form="ขอ")).answer == {"entries": []}


def test_a_second_import_adds_nothing(tmp_path):
    db = SyllabusDb(tmp_path / "syllabus.db")
    html = _html(tmp_path)
    import_paiboon.import_pages(db, html)
    report = import_paiboon.import_pages(db, html)
    assert report["imported"] == 0 and report["skipped"] == 3
