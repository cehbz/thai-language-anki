"""Tests for syllabus.py's study_by_confusion (spec 1 section 3, spec 2
section 3): grouping StudyReader rows by confusion through the
aggregate's own pairs, not a store-owned map.
"""
import dataclasses

import pytest

from thai_syllabus.entities import MinimalPair, SoundConfusion
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.store import SyllabusDb
from thai_syllabus.syllabus import Syllabus

from .builders import sentence, syl, target, word
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


# --- cover(): the fewest drafts that fill the still-unfilled Targets -------

def _open_syllabus() -> Syllabus:
    """Three receptive Targets, no sentences: every Target unfilled."""
    return Syllabus(words=(word("a", "ก", "ay"), word("b", "ข", "bee"),  # ก/ข/ค: letter names
                           word("c", "ค", "see")),
                    targets=(target("a/r", "a"), target("b/r", "b"), target("c/r", "c")),
                    tokenizer=FakeTokenizer())


def test_cover_adopts_the_fewest_drafts_that_fill_the_unfilled_targets():
    syl = _open_syllabus()
    ta, tb, tc = syl.targets
    s_a, s_ab, s_c, s_b = sentence("a"), sentence("ab"), sentence("c"), sentence("b")
    chosen = syl.cover([(s_a, [ta]), (s_ab, [ta, tb]), (s_c, [tc]), (s_b, [tb])])
    assert [s.text for s, _ in chosen] == ["ab", "c"]
    assert [tuple(t.id for t in ts) for _, ts in chosen] == [("a/r", "b/r"), ("c/r",)]


def test_cover_prefers_the_shorter_text_when_two_drafts_fill_the_same_targets():
    syl = _open_syllabus()
    ta = syl.targets[0]
    chosen = syl.cover([(sentence("a longer draft"), [ta]), (sentence("short"), [ta])])
    assert [s.text for s, _ in chosen] == ["short"]


def test_cover_skips_a_draft_that_fills_nothing_still_unfilled():
    syl = _open_syllabus().with_sentences([sentence("ก")])  # ก: the letter's name
    ta = syl.targets[0]
    assert syl.gaps().unfilled_targets == ("b/r", "c/r")
    assert syl.cover([(sentence("another a"), [ta])]) == []


# --- lookups and the voice a recording may draw (E2, E7) -------------------

def _voice_syllabus(skill="receptive") -> Syllabus:
    return Syllabus(words=(word("rice", "ข้าว", "rice"), word("news", "ข่าว", "news")),  # rice/news
                    targets=(target(f"rice/{skill}", "rice", skill=skill),
                             target("news/receptive", "news")),
                    tokenizer=FakeTokenizer())


def test_the_syllabus_names_the_sentence_and_pair_it_cannot_find():
    syllabus = _voice_syllabus()
    with pytest.raises(KeyError, match="deadbeef"):
        syllabus.sentence("deadbeef")
    with pytest.raises(KeyError, match="no-such-pair"):
        syllabus.pair("no-such-pair")


def test_the_syllabus_finds_an_adopted_sentence_by_its_text_sha():
    adopted = sentence("ข้าว")   # ข้าว: rice
    syllabus = _voice_syllabus().with_sentences([adopted])
    assert syllabus.sentence(adopted.text_sha) is adopted


def test_a_word_serves_productive_only_with_a_productive_target():
    assert _voice_syllabus("productive").serves_productive("rice")
    assert not _voice_syllabus("receptive").serves_productive("rice")
    assert not _voice_syllabus("productive").serves_productive("news")


def test_a_sentence_serves_productive_when_it_fills_a_productive_target():
    productive = _voice_syllabus("productive")
    assert productive.sentence_serves_productive(sentence("ข้าว"))       # ข้าว: rice
    assert not productive.sentence_serves_productive(sentence("ข่าว"))   # ข่าว: news
    assert not _voice_syllabus().sentence_serves_productive(sentence("ข้าว"))


def test_a_pair_takes_the_strictest_of_its_members_voice_constraints():
    confusion = SoundConfusion(id=ConfusionId("tone:falling-low"), dimension="tone",
                               sounds=("falling", "low"))
    pair = MinimalPair(id=PairId("p1"), confusion=confusion.id, members=("rice", "news"))
    strict = dataclasses.replace(_voice_syllabus("productive"), pairs=(pair,),
                                 confusions=(confusion,))
    loose = dataclasses.replace(_voice_syllabus("receptive"), pairs=(pair,),
                                confusions=(confusion,))
    assert strict.pair_voice_constraint("p1") == "male"
    assert loose.pair_voice_constraint("p1") == "any"
