"""scripts/strip_pron.py: words.yaml loses its stored pronunciations
(spec 2 r23), every row kept; each word's old reading is reported
against its settled one where they differ, with counts.
"""
import importlib.util
from pathlib import Path

import yaml

from thai_syllabus.store import SyllabusDb

from tests.syllabus.builders import on_record, syl

_spec = importlib.util.spec_from_file_location(
    "strip_pron", Path(__file__).parent.parent / "scripts" / "strip_pron.py")
strip_pron = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strip_pron)


def _old(segments, length, tone):
    return {"syllables": [{"segments": list(segments), "vowel_length": length, "tone": tone}],
            "corroboration": "curated_exception"}


def _deck(tmp_path):
    deck = tmp_path / "deck"
    (deck / "curated").mkdir(parents=True)
    rows = [
        {"id": "rice", "thai": "ข้าว", "meaning": "rice", "classifier": None,       # ข้าว: rice
         "category": "Food", "pron": _old(("kʰ", "a", "w"), "long", "falling")},
        {"id": "good", "thai": "ดี", "meaning": "good", "classifier": None,         # ดี: good
         "category": "Adjectives", "pron": _old(("d", "i", ""), "long", "low"),
         "speaker": "male"},
        {"id": "fish", "thai": "ปลา", "meaning": "fish", "classifier": None,       # ปลา: fish
         "category": "Food", "pron": _old(("pl", "a", ""), "long", "mid")},
    ]
    (deck / "curated" / "words.yaml").write_text(yaml.safe_dump(rows, allow_unicode=True),
                                                 encoding="utf-8")
    db = SyllabusDb(deck / "syllabus.db")
    on_record(db, "ข้าว", (syl("kʰ", "a", "w", "long", "falling"),))
    on_record(db, "ดี", (syl("d", "i", "", "long", "mid"),))
    db.close()
    return deck


def test_words_yaml_keeps_every_row_and_no_pronunciation(tmp_path):
    deck = _deck(tmp_path)
    strip_pron.strip(deck)
    rows = yaml.safe_load((deck / "curated" / "words.yaml").read_text(encoding="utf-8"))
    assert [r["id"] for r in rows] == ["rice", "good", "fish"]
    assert not any("pron" in r for r in rows)
    assert rows[1]["speaker"] == "male"


def test_each_difference_is_reported_with_the_counts(tmp_path):
    lines, counts = strip_pron.strip(_deck(tmp_path))
    assert lines == ["good\tดี\tdiː˨˩\t->\tdiː˧", "fish\tปลา\tplaː˧\t->\t(none)"]
    assert counts == {"same": 1, "differs": 1, "now none": 1}
