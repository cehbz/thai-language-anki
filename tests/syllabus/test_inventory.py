"""Tests for inventory.py (spec 1 r14 design 2026-09-12 SS1): the repo
data files listing the 42 Thai consonants in use and the vowel signs, tone marks
and diacritics, each with the recited name Thai speakers spell with.
"""
from pathlib import Path

import pytest

from thai_syllabus.inventory import (ConsonantRow, DATA_DIR, InventoryError, VowelRow, consonants,
                                     load_consonants, load_vowels, vowels)

DATA = Path(__file__).resolve().parents[2] / "data"


def test_the_consonant_table_has_the_forty_two_letters_once_each():
    rows = load_consonants(DATA / "thai_consonants.yaml")
    assert len(rows) == 42 and len({r.symbol for r in rows}) == 42
    # ก..ฮ is 46 code points; ฤ and ฦ are vocalic letters, and ฃ and ฅ are
    # obsolete
    assert {r.symbol for r in rows} == {chr(c) for c in range(0x0E01, 0x0E2F)} - {"ฤ", "ฦ", "ฃ", "ฅ"}
    by = {r.symbol: r for r in rows}
    assert by["ก"].consonant_class == "mid" and by["ก"].keyword_thai == "ไก่" and by["ก"].name_thai == "กอ ไก่"
    assert by["ข"].consonant_class == "high" and by["ค"].consonant_class == "low"
    assert all(r.keyword_thai and r.keyword_gloss and r.name_thai.startswith(r.symbol) for r in rows)


def test_the_vowel_table_covers_the_signs_and_the_four_tone_marks():
    rows = load_vowels(DATA / "thai_vowels.yaml")
    kinds = {r.symbol: r.kind for r in rows}
    assert len(rows) == 22
    assert {kinds[m] for m in "่้๊๋"} == {"tone_mark"}
    assert kinds["า"] == "vowel_sign" and kinds["์"] == "diacritic" and kinds["็"] == "diacritic"
    assert all(r.name_thai for r in rows)
    # spec 3 r72: every row names its keyword, spelled with the symbol
    assert all(r.symbol in r.keyword_thai and r.keyword_gloss for r in rows)


def _vowel_table(tmp_path, text):
    path = tmp_path / "vowels.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_vowel_row_names_its_keyword_and_gloss(tmp_path):
    """Spec 3 r72: the keyword of a sign or mark is named in the table,
    as a consonant's is."""
    path = _vowel_table(tmp_path, "- {symbol: า, kind: vowel_sign, sound: aː, name: สระ อา,"
                                  " keyword: กา, gloss: crow}\n")   # า: aː; กา: crow
    assert load_vowels(path) == (VowelRow(symbol="า", kind="vowel_sign", sound="aː",
                                          name_thai="สระ อา", keyword_thai="กา",
                                          keyword_gloss="crow"),)


def test_a_vowel_row_without_a_keyword_is_refused(tmp_path):
    path = _vowel_table(tmp_path, "- {symbol: า, kind: vowel_sign, sound: aː, name: สระ อา}\n")
    with pytest.raises(InventoryError, match="malformed"):
        load_vowels(path)


def test_vowels_reads_the_repo_table():
    assert vowels() == load_vowels(DATA / "thai_vowels.yaml")


# --- the repo table, at the package-relative path (spec 3 r40 §5) ---------

def test_consonants_reads_the_repo_table_of_42():
    """The adoption pass takes the inventory from the repo's own data/,
    resolved relative to the package -- no deck configuration, no cwd."""
    rows = consonants()
    assert len(rows) == 42
    assert rows[0] == ConsonantRow(symbol="ก", consonant_class="mid", sound="k",
                                   name_thai="กอ ไก่", keyword_thai="ไก่",
                                   keyword_gloss="chicken")   # ก: k; ไก่: chicken


def test_the_data_dir_holds_the_consonant_table():
    assert (DATA_DIR / "thai_consonants.yaml").exists()
