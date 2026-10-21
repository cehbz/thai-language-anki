"""blockedpairs: each pairs.yaml pair neither pair rule lets stand
(Syllabus.usable_pairs), with the rules blocking it and its cure (spec 5
r23)."""
from __future__ import annotations

import pytest

from thai_syllabus.blockedpairs import Cure, blocked_pairs
from thai_syllabus.cachekeys import DictionaryKey
from thai_syllabus.entities import MinimalPair, SoundConfusion
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.store import SyllabusDb
from thai_syllabus.syllabus import Syllabus

from .builders import pron, syl, word


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


TONE = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone", sounds=("mid", "low"))
MID, LOW, HIGH = (syl(onset="kh", tone=t) for t in ("mid", "low", "high"))


def _pair(pid, *members):
    return MinimalPair.create(id=PairId(pid), confusion=TONE, members=members)


def _syllabus(db, words, pairs, confusion=TONE):
    return Syllabus(words=tuple(words), targets=(), pairs=tuple(pairs), graphemes=(),
                    confusions=(confusion,), assessments=db)


def _answered(db, form, *sources):
    for source in sources:
        db.append(port="provide", backend=source, key=DictionaryKey(source, form),
                  subject=f"dictionary:{form}", question={"form": form}, answer={})


A, B = word("a", "ก", "a", syllables=(MID,)), word("b", "ข", "b", syllables=(LOW,))
C, D = word("c", "ค", "c", syllables=(MID,)), word("d", "ง", "d", syllables=(LOW,))


def _unsettled(w):
    return word(str(w.id), w.thai, w.meaning, syllables=w.pron.syllables, posterior=0.6)


def test_a_usable_pair_is_not_blocked(db):
    assert blocked_pairs(_syllabus(db, [A, B], [_pair("p1", A, B)]), db) == ()


def test_a_blocked_pair_names_each_rule_blocking_it_with_the_findings_evidence(db):
    p = _pair("p1", A, B)
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(A), B], [p]), db)
    assert blocked.pair == p
    assert [(f.rule, f.note_id) for f in blocked.findings] == [("pair/pronunciation-settled", "a")]
    assert "posterior 0.600" in blocked.findings[0].evidence


def test_a_pair_drifted_out_of_its_contrast_carries_the_exact_confusion_finding(db):
    p = _pair("p1", A, B)
    drifted = word("b", "ข", "b", syllables=(HIGH,))
    blocked, = blocked_pairs(_syllabus(db, [A, drifted], [p]), db)
    assert [(f.rule, f.note_id) for f in blocked.findings] == [("pair/exact-confusion", "p1")]


def test_a_blocked_pair_whose_confusion_has_a_usable_pair_adopted_since_is_replaced(db):
    old, new = _pair("p1", A, B), _pair("p2", C, D)
    tone = SoundConfusion(id=TONE.id, dimension="tone", sounds=TONE.sounds, weight=3)
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(A), B, C, D], [old, new], tone), db)
    assert blocked.cure == Cure("replaced", pairs=(PairId("p2"),))
    assert blocked.cure.text() == "replaced by p2"


def test_a_blocked_pair_whose_confusions_need_is_met_without_it_is_replaced(db):
    usable, old = _pair("p2", C, D), _pair("p1", A, B)
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(A), B, C, D], [usable, old]), db)
    assert blocked.cure == Cure("replaced", pairs=(PairId("p2"),))


def test_a_usable_pair_adopted_before_it_does_not_replace_it_while_the_need_is_short(db):
    usable, old = _pair("p2", C, D), _pair("p1", A, B)
    tone = SoundConfusion(id=TONE.id, dimension="tone", sounds=TONE.sounds, weight=3)
    _answered(db, "ก", "wiktionary", "thai-language.com")
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(A), B, C, D], [usable, old], tone), db)
    assert blocked.cure.kind == "ruling"


def test_a_member_a_dictionary_has_not_answered_awaits_that_family(db):
    _answered(db, "ก", "thai-language.com")
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(A), B], [_pair("p1", A, B)]), db)
    assert blocked.cure == Cure("awaiting", families=("wiktionary",))
    assert blocked.cure.text() == "awaiting evidence from wiktionary"


def test_only_a_blocking_members_dictionaries_are_awaited(db):
    _answered(db, "ก", "wiktionary", "thai-language.com")
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(A), B], [_pair("p1", A, B)]), db)
    assert blocked.cure == Cure("ruling")
    assert blocked.cure.text() == "needs a ruling"


def test_a_drifted_pair_awaits_any_members_unanswered_dictionary(db):
    _answered(db, "ก", "wiktionary", "thai-language.com")
    drifted = word("b", "ข", "b", syllables=(HIGH,))
    blocked, = blocked_pairs(_syllabus(db, [A, drifted], [_pair("p1", A, B)]), db)
    assert blocked.cure == Cure("awaiting", families=("wiktionary", "thai-language.com"))


def test_a_phrase_member_awaits_no_dictionary(db):
    phrase = word("a", "ก ก", "a", syllables=(MID,))
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(phrase), B], [_pair("p1", phrase, B)]), db)
    assert blocked.cure == Cure("ruling")


def test_a_dissenting_spoken_witness_with_all_evidence_in_needs_a_ruling(db):
    _answered(db, "ก", "wiktionary", "thai-language.com")
    dissenting = word("a", "ก", "a", syllables=(MID,))
    dissenting = type(dissenting)(id=dissenting.id, thai=dissenting.thai, meaning="a",
                                  pron=pron(MID, spoken_dissent=True))
    blocked, = blocked_pairs(_syllabus(db, [dissenting, B], [_pair("p1", A, B)]), db)
    assert "dissents" in blocked.findings[0].evidence
    assert blocked.cure == Cure("ruling")


def test_an_unsettled_member_is_the_one_blocking_its_pair(db):
    blocked, = blocked_pairs(_syllabus(db, [_unsettled(A), B], [_pair("p1", A, B)]), db)
    assert blocked.blocking == ("a",)


def test_every_member_of_a_drifted_pair_is_blocking(db):
    drifted = word("b", "ข", "b", syllables=(HIGH,))
    blocked, = blocked_pairs(_syllabus(db, [A, drifted], [_pair("p1", A, B)]), db)
    assert blocked.blocking == ("a", "b")
