"""Tests for inventory.py (spec 1 r14 design 2026-09-12 SS1): the repo
data files listing the 44 Thai consonants and the vowel signs, tone marks
and diacritics, each with the recited name Thai speakers spell with.
"""
from pathlib import Path

from thai_syllabus.inventory import load_consonants, load_vowels

DATA = Path(__file__).resolve().parents[2] / "data"


def test_the_consonant_table_has_the_forty_four_letters_once_each():
    rows = load_consonants(DATA / "thai_consonants.yaml")
    assert len(rows) == 44 and len({r.symbol for r in rows}) == 44
    # ก..ฮ is 46 code points; ฤ and ฦ are vocalic letters, not consonants
    assert {r.symbol for r in rows} == {chr(c) for c in range(0x0E01, 0x0E2F)} - {"ฤ", "ฦ"}
    by = {r.symbol: r for r in rows}
    assert by["ก"].consonant_class == "mid" and by["ก"].keyword_thai == "ไก่" and by["ก"].name_thai == "กอ ไก่"
    assert by["ข"].consonant_class == "high" and by["ค"].consonant_class == "low"
    assert all(r.keyword_thai and r.keyword_gloss and r.name_thai.startswith(r.symbol) for r in rows)


def test_the_vowel_table_covers_the_signs_and_the_four_tone_marks():
    rows = load_vowels(DATA / "thai_vowels.yaml")
    kinds = {r.symbol: r.kind for r in rows}
    assert {kinds[m] for m in "่้๊๋"} == {"tone_mark"}
    assert kinds["า"] == "vowel_sign" and kinds["์"] == "diacritic"
    assert all(r.name_thai for r in rows)
