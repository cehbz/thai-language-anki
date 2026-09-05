"""Tests for syllabus.py's study_by_confusion (spec 1 section 3, spec 2
section 3): grouping StudyReader rows by confusion through the
aggregate's own pairs, not a store-owned map.
"""
import pytest

from thai_syllabus.entities import MinimalPair, SoundConfusion
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.store import SyllabusDb
from thai_syllabus.syllabus import Syllabus

from .builders import syl, word
from .fakes import FakeTokenizer


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


def _pair(pair_id: str, confusion: SoundConfusion) -> MinimalPair:
    mid_w = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    low_w = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    return MinimalPair.create(id=PairId(pair_id), confusion=confusion,
                              members=(mid_w, low_w))


def test_study_groups_study_records_by_confusion(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(card_key="p1:s1:0::recognition", compile_id="c", ts=1,
                    grade=1, time_ms=1)

    assert list(syllabus.study_by_confusion(db)) == ["tone:mid-low"]


def test_study_by_confusion_accepts_both_pair_card_key_shapes(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(card_key="p1::recognition", compile_id="c", ts=1, grade=2,
                    time_ms=10)
    db.append_study(card_key="p1:speaker-a:0::recognition", compile_id="c", ts=2,
                    grade=3, time_ms=20)

    grouped = syllabus.study_by_confusion(db)
    assert {r.card_key for r in grouped["tone:mid-low"]} == {
        "p1::recognition", "p1:speaker-a:0::recognition",
    }


def test_study_by_confusion_resolves_a_colon_bearing_pair_id_exact_shape(db):
    # Real pair ids embed the confusion id, which itself contains ":"
    # ("tone:mid-low/klai") -- the anchor parse must not cut on the first
    # ":" it sees.
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("tone:mid-low/klai", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(card_key="tone:mid-low/klai::recognition", compile_id="c",
                    ts=1, grade=1, time_ms=1)

    grouped = syllabus.study_by_confusion(db)
    assert {r.card_key for r in grouped["tone:mid-low"]} == {"tone:mid-low/klai::recognition"}


def test_study_by_confusion_resolves_a_colon_bearing_pair_id_memberkey_shape(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("tone:mid-low/klai", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(card_key="tone:mid-low/klai:s1:0::recognition", compile_id="c",
                    ts=1, grade=1, time_ms=1)

    grouped = syllabus.study_by_confusion(db)
    assert {r.card_key for r in grouped["tone:mid-low"]} == {
        "tone:mid-low/klai:s1:0::recognition",
    }


def test_study_by_confusion_prefers_the_longest_matching_pair_id(db):
    # A decoy pair id that is a proper prefix of another's, sharing the
    # ":" component the confusion id already contributes -- the anchor
    # must resolve to the pair it actually names, not the shorter decoy.
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    decoy_confusion = SoundConfusion(id=ConfusionId("tone:mid-low-decoy"), dimension="tone",
                                     sounds=("mid", "low"))
    pair = _pair("tone:mid-low/klai", confusion)
    decoy_pair = _pair("tone:mid-low/klai-long", decoy_confusion)
    syllabus = Syllabus(pairs=(pair, decoy_pair), confusions=(confusion, decoy_confusion),
                        tokenizer=FakeTokenizer())

    db.append_study(card_key="tone:mid-low/klai-long:s1:0::recognition", compile_id="c",
                    ts=1, grade=1, time_ms=1)

    grouped = syllabus.study_by_confusion(db)
    assert "tone:mid-low" not in grouped
    assert {r.card_key for r in grouped["tone:mid-low-decoy"]} == {
        "tone:mid-low/klai-long:s1:0::recognition",
    }


def test_study_by_confusion_skips_a_card_key_naming_no_known_pair(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(card_key="unrelated-word::listening", compile_id="c", ts=1,
                    grade=1, time_ms=1)

    assert syllabus.study_by_confusion(db) == {}
