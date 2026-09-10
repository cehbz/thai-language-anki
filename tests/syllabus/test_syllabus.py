"""Tests for syllabus.py's study_by_confusion (spec 1 section 3, spec 2
section 3): grouping StudyReader rows by confusion through the
aggregate's own pairs, not a store-owned map.
"""
import dataclasses

import pytest

from thai_syllabus.entities import Category, MinimalPair, SoundConfusion
from thai_syllabus.ids import CategoryName, ConfusionId, PairId, WordId
from thai_syllabus.ports import StudyRecord
from thai_syllabus.store import SyllabusDb
from thai_syllabus.syllabus import Syllabus, derive_productive_targets

from .builders import sentence, syl, target, thai_of, word


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
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,))

    db.append_study(_study(member_index="0", speaker_id="s1"))

    assert list(syllabus.study_by_confusion(db)) == ["tone:mid-low"]


def test_study_by_confusion_groups_every_member_row_under_the_pair_id(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,))

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
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,))

    db.append_study(_study(anchor="tone:mid-low/klai"))

    grouped = syllabus.study_by_confusion(db)
    assert len(grouped["tone:mid-low"]) == 1


def test_study_by_confusion_skips_an_anchor_naming_no_known_pair(db):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,))

    db.append_study(_study(anchor="unrelated-pair"))

    assert syllabus.study_by_confusion(db) == {}


def test_study_by_confusion_ignores_a_non_pair_family_row(db):
    # A word row whose anchor happens to equal a pair id must not be
    # folded into that pair's confusion.
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = _pair("p1", confusion)
    syllabus = Syllabus(pairs=(pair,), confusions=(confusion,))

    db.append_study(_study(family="word", anchor="p1", card_kind="listening"))

    assert syllabus.study_by_confusion(db) == {}


# --- cover(): the fewest drafts that fill the still-unfilled Targets -------

def _open_syllabus() -> Syllabus:
    """Three receptive Targets, no sentences: every Target unfilled."""
    return Syllabus(words=(word("a", "ก", "ay"), word("b", "ข", "bee"),  # ก/ข/ค: letter names
                           word("c", "ค", "see")),
                    targets=(target("a/r", "a"), target("b/r", "b"), target("c/r", "c")))


def test_cover_adopts_the_fewest_drafts_that_fill_the_unfilled_targets():
    syl = _open_syllabus()
    ta, tb, tc = syl.targets
    to = thai_of(*syl.words)
    s_a = sentence(((WordId("a"),),), to)
    s_ab = sentence(((WordId("a"), WordId("b")),), to)
    s_c = sentence(((WordId("c"),),), to)
    s_b = sentence(((WordId("b"),),), to)
    chosen = syl.cover([(s_a, [ta]), (s_ab, [ta, tb]), (s_c, [tc]), (s_b, [tb])])
    assert [s.text for s, _ in chosen] == ["กข", "ค"]  # letter names a+b, c
    assert [tuple(t.id for t in ts) for _, ts in chosen] == [("a/r", "b/r"), ("c/r",)]


def test_cover_prefers_the_shorter_text_when_two_drafts_fill_the_same_targets():
    syl = _open_syllabus()
    ta = syl.targets[0]
    to = thai_of(*syl.words)
    long_draft = sentence(((WordId("a"), WordId("b"), WordId("c")),), to)  # a longer draft
    short_draft = sentence(((WordId("a"),),), to)  # short
    chosen = syl.cover([(long_draft, [ta]), (short_draft, [ta])])
    assert [s.text for s, _ in chosen] == [short_draft.text]


def test_cover_skips_a_draft_that_fills_nothing_still_unfilled():
    syl = _open_syllabus()
    to = thai_of(*syl.words)
    adopted = sentence(((WordId("a"),),), to)  # ก: the letter's name
    syl = syl.with_sentences([adopted])
    ta = syl.targets[0]
    assert syl.gaps().unfilled_targets == ("b/r", "c/r")
    assert syl.cover([(adopted, [ta])]) == []  # ta is no longer uncovered


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
    syl = Syllabus(words=(a,), targets=(ta,))
    draft = sentence(((a.id,),), thai_of(a))   # ก: the letter a
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
                             target("news/receptive", "news")))


def test_the_syllabus_names_the_sentence_and_pair_it_cannot_find():
    syllabus = _voice_syllabus()
    with pytest.raises(KeyError, match="deadbeef"):
        syllabus.sentence("deadbeef")
    with pytest.raises(KeyError, match="no-such-pair"):
        syllabus.pair("no-such-pair")


def test_the_syllabus_finds_an_adopted_sentence_by_its_text_sha():
    rice = word("rice", "ข้าว")  # rice
    adopted = sentence(((rice.id,),), thai_of(rice))
    syllabus = _voice_syllabus().with_sentences([adopted])
    assert syllabus.sentence(adopted.text_sha) is adopted


def test_a_word_serves_productive_only_with_a_productive_target():
    assert _voice_syllabus("productive").serves_productive("rice")
    assert not _voice_syllabus("receptive").serves_productive("rice")
    assert not _voice_syllabus("productive").serves_productive("news")


def test_a_sentence_serves_productive_when_it_fills_a_productive_target():
    rice = word("rice", "ข้าว")   # rice
    news = word("news", "ข่าว")  # news
    to = thai_of(rice, news)
    rice_sentence = sentence(((rice.id,),), to)     # rice
    news_sentence = sentence(((news.id,),), to)     # news
    productive = _voice_syllabus("productive")
    assert productive.sentence_serves_productive(rice_sentence)
    assert not productive.sentence_serves_productive(news_sentence)
    assert not _voice_syllabus().sentence_serves_productive(rice_sentence)


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


def test_gaps_excludes_a_sentence_introduced_word_from_words_missing_pictures():
    rice = word("rice", "ข้าว")  # rice
    glue = word("with", "กับ")  # กับ: with -- glue word, sentence-introduced
    syllabus = Syllabus(words=(rice, glue),
                        targets=(target("rice/r", "rice"),
                                 target("with/r", "with", introduction="sentence")))
    assert syllabus.gaps().words_missing_pictures == ("rice",)


# --- derive_productive_targets (spec 1 r9) --------------------------------

def _food(*word_ids: str) -> Category:
    return Category(name=CategoryName("Food"), members=frozenset(word_ids))


def test_derive_productive_targets_derives_for_a_categorized_word_at_the_cutoff():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [_food("rice")], {rice.id: 2000}, 2000)
    assert [t.id for t in derived] == ["rice/productive"]
    d = derived[0]
    assert (d.word, d.skill, d.introduction) == (rice.id, "productive", "picture_card")


def test_derive_productive_targets_excludes_a_word_ranked_past_the_cutoff():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [_food("rice")], {rice.id: 2001}, 2000)
    assert derived == ()


def test_derive_productive_targets_excludes_an_uncategorized_word():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [], {rice.id: 1}, 2000)
    assert derived == ()


def test_derive_productive_targets_excludes_a_no_productive_word():
    rice = dataclasses.replace(word("rice", "ข้าว"), no_productive=True)
    derived = derive_productive_targets([rice], [], [_food("rice")], {rice.id: 1}, 2000)
    assert derived == ()


def test_derive_productive_targets_excludes_an_unranked_word():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [_food("rice")], {}, 2000)
    assert derived == ()


def test_derive_productive_targets_keeps_a_listed_exception_below_cutoff():
    rice = word("rice", "ข้าว")
    listed = target("rice/productive", "rice", skill="productive")
    derived = derive_productive_targets(
        [rice], [listed], [_food("rice")], {rice.id: 5000}, 2000)
    assert derived == ()


def test_derive_productive_targets_raises_on_a_listed_target_for_an_eligible_word():
    rice = word("rice", "ข้าว")
    listed = target("rice/productive", "rice", skill="productive")
    with pytest.raises(ValueError, match="rice/productive"):
        derive_productive_targets([rice], [listed], [_food("rice")], {rice.id: 10}, 2000)


def test_order_places_the_derived_productive_target_after_the_receptive_one():
    rice = word("rice", "ข้าว")
    receptive = target("rice/receptive", "rice")
    derived = derive_productive_targets(
        [rice], [receptive], [_food("rice")], {rice.id: 1}, 2000)
    syllabus = Syllabus(words=(rice,), targets=(receptive,) + derived,
                        categories=(_food("rice"),))
    ids = [e.id for e in syllabus.order() if e.kind == "word_target"]
    assert ids.index("rice/receptive") < ids.index("rice/productive")
