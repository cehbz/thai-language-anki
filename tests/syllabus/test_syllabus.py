"""Tests for syllabus.py's study_by_confusion (spec 1 section 3, spec 2
section 3): grouping StudyReader rows by confusion through the
aggregate's own pairs, not a store-owned map.
"""
import dataclasses

import pytest

from thai_syllabus.entities import MinimalPair, SoundConfusion
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.ports import StudyRecord
from thai_syllabus.store import SyllabusDb
from thai_syllabus.syllabus import Syllabus, decompose

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


def _study(**overrides) -> StudyRecord:
    fields = {"family": "minimal_pair", "anchor": "p1", "card_kind": "recognition",
             "compile_id": "c", "ts": 1, "grade": 1, "time_ms": 1}
    fields.update(overrides)
    return StudyRecord(**fields)


def test_study_groups_study_records_by_confusion(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(_study(member_index="0", speaker_id="s1"))

    assert list(syllabus.study_by_confusion(db)) == ["tone:mid-low"]


def test_study_by_confusion_groups_every_member_row_under_the_pair_id(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(_study(ts=1, grade=2, time_ms=10, member_index="0", speaker_id="speaker-a"))
    db.append_study(_study(ts=2, grade=3, time_ms=20, member_index="1", speaker_id="speaker-b"))

    grouped = syllabus.study_by_confusion(db)
    assert len(grouped["tone:mid-low"]) == 2


def test_study_by_confusion_resolves_a_colon_bearing_pair_id(db):
    # Real pair ids embed the confusion id, which itself contains ":"
    # ("tone:mid-low/klai") -- an exact anchor match must not mistreat it.
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("tone:mid-low/klai", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(_study(anchor="tone:mid-low/klai"))

    grouped = syllabus.study_by_confusion(db)
    assert len(grouped["tone:mid-low"]) == 1


def test_study_by_confusion_skips_an_anchor_naming_no_known_pair(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(_study(anchor="unrelated-pair"))

    assert syllabus.study_by_confusion(db) == {}


def test_study_by_confusion_ignores_a_non_pair_family_row(db):
    # A word row whose anchor happens to equal a pair id must not be
    # folded into that pair's confusion.
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,), tokenizer=FakeTokenizer())

    db.append_study(_study(family="word", anchor="p1", card_kind="listening"))

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


def test_adopting_a_draft_cover_chose_still_has_that_target_in_its_own_fill_set():
    """Adopt-then-gate agreement (spec 1 r6): a draft cover() adopts for
    a Target still has that Target in its own live fill_set once
    with_sentences actually adopts it, agreeing with what fill_set()
    computed for it as a candidate before adoption -- adoption must not
    change what a sentence itself fills, and a sentence-introduced
    Target's own novelty check must still exclude the sentence from its
    own "other adopted" check once it is one of self.sentences.
    """
    a = word("a", "ก")  # a
    ta = target("a/r", "a", "receptive", introduction="sentence")
    syl = Syllabus(words=(a,), targets=(ta,), tokenizer=FakeTokenizer())
    draft = sentence("ก")   # ก: the letter a
    chosen = syl.cover([(draft, [ta])])
    adopted_sentence, gained = chosen[0]
    assert ta in gained
    before = syl.fill_set(draft)
    new_syllabus = syl.with_sentences([adopted_sentence])
    after = new_syllabus.fill_set(adopted_sentence)
    assert before == after == (ta,)


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


# --- orthographic marks carry no vocabulary (spec 1 section 3) -------------

def test_has_lexical_content_is_false_for_orthographic_marks_true_for_a_word():
    assert Syllabus._has_lexical_content("ๆ") is False  # repetition mark
    assert Syllabus._has_lexical_content("ฯ") is False  # abbreviation mark
    assert Syllabus._has_lexical_content("วิ่ง") is True  # run -- a real word


def test_a_repetition_mark_is_not_a_new_word_and_the_sentence_fills_its_target():
    run = word("run", "วิ่ง", "run")  # วิ่ง: run
    t = target("run/receptive", "run", "receptive")
    s = sentence("วิ่งๆ", voice="learner_voice")  # run repeatedly -- ๆ repeats "run"
    tok = FakeTokenizer({s.text: ["วิ่ง", "ๆ"]})
    syllabus = Syllabus(words=(run,), targets=(t,), tokenizer=tok)
    assert syllabus.fills(s, t) is True


def test_an_attached_repetition_mark_still_resolves_to_the_bare_registered_word():
    """newmm keeps some reduplications as one token ("ช้าๆ") rather than
    splitting the mark off; the attached form still resolves against a
    registration of the bare word ("ช้า")."""
    slow = word("slow", "ช้า", "slow")  # ช้า: slow
    t = target("slow/receptive", "slow", "receptive")
    s = sentence("ช้าๆ", voice="learner_voice")  # slowly -- ๆ attached to "ช้า"
    tok = FakeTokenizer({s.text: ["ช้าๆ"]})
    syllabus = Syllabus(words=(slow,), targets=(t,), tokenizer=tok)
    assert syllabus.fills(s, t) is True


def test_unknown_tokens_is_empty_for_an_attached_abbreviation_mark_over_a_registered_word():
    bangkok = word("bangkok", "กรุงเทพ", "Bangkok")  # กรุงเทพ: Bangkok
    syllabus = Syllabus(words=(bangkok,), tokenizer=FakeTokenizer())
    assert syllabus._unknown_tokens(["กรุงเทพฯ"]) == []  # กรุงเทพฯ: Bangkok, with the abbreviation mark attached


def test_tokens_of_returns_an_immutable_tuple():
    rice = word("rice", "ข้าว", "rice")  # ข้าว: rice
    s = sentence("ข้าว")
    tok = FakeTokenizer({s.text: ["ข้าว"]})
    syllabus = Syllabus(words=(rice,), tokenizer=tok)
    assert isinstance(syllabus.tokens_of(s), tuple)


def test_gaps_excludes_a_sentence_introduced_word_from_words_missing_pictures():
    rice = word("rice", "ข้าว", "rice")  # ข้าว: rice -- picture-introduced
    glue = word("with", "กับ", "with")  # กับ: with -- glue word, sentence-introduced
    syllabus = Syllabus(words=(rice, glue),
                        targets=(target("rice/r", "rice"),
                                 target("with/r", "with", introduction="sentence")),
                        tokenizer=FakeTokenizer())
    assert syllabus.gaps().words_missing_pictures == ("rice",)


# --- decompose: the fills clause 1 boundary rule's split (spec 1 section 3) --

def test_decompose_of_a_registered_word_is_itself():
    assert decompose("มาก", {"มาก", "มา"}) == ("มาก",)  # มาก: very -- registered outright


def test_decompose_splits_a_compound_into_its_registered_components():
    # โรงพยาบาล: hospital; โรง: building/hall; พยาบาล: nurse
    assert decompose("โรงพยาบาล", {"โรง", "พยาบาล"}) == ("โรง", "พยาบาล")


def test_decompose_finds_the_split_even_when_the_whole_compound_is_also_registered():
    """Spec 1 section 3 fills clause 1's own example: โรงพยาบาล
    (hospital) registered alongside its own parts โรง (building/hall)
    and พยาบาล (nurse) still splits into the two parts -- a proper
    split is tried before the whole-token fallback, so registering the
    compound does not hide its parts.
    """
    known = {"โรง", "พยาบาล", "โรงพยาบาล"}
    assert decompose("โรงพยาบาล", known) == ("โรง", "พยาบาล")


def test_decompose_is_none_when_the_registered_prefix_leaves_an_unregistered_remainder():
    assert decompose("มาก", {"มา"}) is None  # มาก: very; มา: come -- "ก" alone is not registered


def test_decompose_is_none_with_no_matching_prefix_at_all():
    assert decompose("มาก", set()) is None  # มาก: very -- nothing registered


# --- mentions_at/mentions_in: the fills clause 1 boundary rule itself -------

def test_mentions_at_is_true_for_an_exact_token():
    rice = word("rice", "ข้าว", "rice")  # ข้าว: rice
    syllabus = Syllabus(words=(rice,), tokenizer=FakeTokenizer())
    assert syllabus.mentions_at(["ข้าว"], "ข้าว") is True


def test_mentions_at_is_false_for_a_registered_word_that_is_only_a_token_prefix():
    """มาก (very) is one token; มา (come) is registered but not
    mentioned -- the remainder "ก" is not itself registered, so มาก has
    no full decomposition into มา plus a registered remainder (spec 1
    section 3 fills clause 1).
    """
    come = word("come", "มา", "come")  # มา: come
    syllabus = Syllabus(words=(come,), tokenizer=FakeTokenizer())
    assert syllabus.mentions_at(["มาก"], "มา") is False


def test_mentions_at_is_true_for_a_word_that_is_a_component_of_a_full_decomposition():
    """Spec 1 section 3 fills clause 1's own example, with the compound
    itself registered too (not just its parts): โรงพยาบาล (hospital)
    still mentions โรง (building/hall) as a component of its
    decomposition, and mentions itself outright by token identity --
    registering the whole compound does not suppress its parts.
    """
    building = word("building", "โรง", "building/hall")  # โรง: building/hall
    nurse = word("nurse", "พยาบาล", "nurse")  # พยาบาล: nurse
    hospital = word("hospital", "โรงพยาบาล", "hospital")  # โรงพยาบาล: hospital
    syllabus = Syllabus(words=(building, nurse, hospital), tokenizer=FakeTokenizer())
    assert syllabus.mentions_at(["โรงพยาบาล"], "โรง") is True
    assert syllabus.mentions_at(["โรงพยาบาล"], "พยาบาล") is True
    assert syllabus.mentions_at(["โรงพยาบาล"], "โรงพยาบาล") is True


def test_mentions_at_is_true_for_every_component_of_a_three_part_registration():
    """ห้องน้ำ ("hông náam", bathroom), ห้อง ("room") and น้ำ ("water")
    all registered: the token mentions all three -- itself outright, and
    both parts as its full decomposition.
    """
    room = word("room", "ห้อง", "room")  # ห้อง: room
    water = word("water", "น้ำ", "water")  # น้ำ: water
    bathroom = word("bathroom", "ห้องน้ำ", "bathroom")  # ห้องน้ำ: bathroom
    syllabus = Syllabus(words=(room, water, bathroom), tokenizer=FakeTokenizer())
    assert syllabus.mentions_at(["ห้องน้ำ"], "ห้อง") is True
    assert syllabus.mentions_at(["ห้องน้ำ"], "น้ำ") is True
    assert syllabus.mentions_at(["ห้องน้ำ"], "ห้องน้ำ") is True


def test_mentions_at_is_false_for_a_partial_decomposition():
    """Only โรง (building) is registered; พยาบาล (nurse) is not, so
    โรงพยาบาล (hospital) has no full decomposition and mentions nothing
    beyond its own exact token.
    """
    building = word("building", "โรง", "building/hall")
    syllabus = Syllabus(words=(building,), tokenizer=FakeTokenizer())
    assert syllabus.mentions_at(["โรงพยาบาล"], "โรง") is False


def test_known_words_is_every_registered_words_thai_text():
    rice = word("rice", "ข้าว", "rice")
    eat = word("eat", "กิน", "eat")
    syllabus = Syllabus(words=(rice, eat), tokenizer=FakeTokenizer())
    assert syllabus.known_words == frozenset({"ข้าว", "กิน"})


def test_mentions_in_is_mentions_at_against_an_explicit_known_set():
    """mentions_in is mentions_at's staticmethod form, for a caller with
    no Syllabus instance to hand (compile.thai_cloze).
    """
    assert Syllabus.mentions_in(["โรงพยาบาล"], "โรง", {"โรง", "พยาบาล"}) is True
    assert Syllabus.mentions_in(["มาก"], "มา", {"มา"}) is False
