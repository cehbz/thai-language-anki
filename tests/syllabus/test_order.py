"""Syllabus.order(): sounds before words; receptive before productive per
word; ties by frequency rank / emphasis weight (spec 1, section 3).
"""
from thai_syllabus.entities import Category, Grapheme, MinimalPair, SoundConfusion, Target
from thai_syllabus.ids import CategoryName, ConfusionId, PairId
from thai_syllabus.profile import Profile
from thai_syllabus.syllabus import Syllabus

from .builders import target, word
from .fakes import FakeTokenizer


def make_pair(id_, confusion_id, member_words) -> MinimalPair:
    confusion = SoundConfusion(id=ConfusionId(confusion_id), dimension="tone",
                               sounds=("mid", "low"))
    return MinimalPair.create(id=PairId(id_), confusion=confusion,
                              members=member_words)


def test_sounds_stage_precedes_every_word_target():
    rice = word("rice", "ข้าว")  # rice
    dog = word("dog", "หมา")  # dog
    from .builders import syl
    mid_w = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    low_w = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    pair = make_pair("p1", "tone:mid-low", (mid_w, low_w))
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                               consonant_class="mid", keyword_word=low_w)  # ไกล contains ก
    t1 = target("rice/receptive", "rice", "receptive")
    t2 = target("dog/receptive", "dog", "receptive")

    syllabus = Syllabus(
        words=(rice, dog, mid_w, low_w), targets=(t1, t2), pairs=(pair,),
        graphemes=(grapheme,),
        profile=Profile(register="male_colloquial"), tokenizer=FakeTokenizer(),
    )
    ordering = syllabus.order()
    positions = {item if isinstance(item, str) else item.id: i
                for i, item in enumerate(ordering)}
    assert positions[pair.id] < positions[t1.id]
    assert positions[grapheme.symbol] < positions[t1.id]
    assert positions[pair.id] < positions[t2.id]


def test_receptive_precedes_productive_for_the_same_word():
    rice = word("rice", "ข้าว")  # rice
    receptive = target("rice/receptive", "rice", "receptive")
    productive = target("rice/productive", "rice", "productive")
    syllabus = Syllabus(words=(rice,), targets=(productive, receptive),
                        profile=Profile(register="male_colloquial"),
                        tokenizer=FakeTokenizer())
    ordering = [t.id for t in syllabus.order() if isinstance(t, Target)]
    assert ordering.index(receptive.id) < ordering.index(productive.id)


def test_ties_are_broken_by_frequency_rank_over_emphasis_weight():
    common = word("common", "บ้าน")  # house -- frequent
    rare = word("rare", "ปราสาท")  # palace -- infrequent
    t_common = target("common/receptive", "common", "receptive")
    t_rare = target("rare/receptive", "rare", "receptive")
    syllabus = Syllabus(
        words=(common, rare), targets=(t_rare, t_common),
        profile=Profile(register="male_colloquial"), tokenizer=FakeTokenizer(),
        frequency={common.id: 10, rare.id: 5000},
    )
    ordering = [t.id for t in syllabus.order() if isinstance(t, Target)]
    assert ordering.index(t_common.id) < ordering.index(t_rare.id)


def test_emphasis_weight_can_move_a_lower_frequency_word_earlier():
    common = word("common", "บ้าน")  # house -- frequent, "other" category
    rare = word("rare", "ปราสาท")  # palace -- infrequent, but emphasized "food"
    t_common = target("common/receptive", "common", "receptive")
    t_rare = target("rare/receptive", "rare", "receptive")
    syllabus = Syllabus(
        words=(common, rare), targets=(t_rare, t_common),
        profile=Profile(register="male_colloquial",
                        emphasis={"food": 100.0}),
        tokenizer=FakeTokenizer(),
        frequency={common.id: 10, rare.id: 100},
        categories=(Category(name=CategoryName("food"), members=frozenset({rare.id})),),
    )
    ordering = [t.id for t in syllabus.order() if isinstance(t, Target)]
    assert ordering.index(t_rare.id) < ordering.index(t_common.id)


# --- Syllabus.category_of ------------------------------------------------

def test_category_of_returns_the_owning_categorys_name():
    rice = word("rice", "ข้าว")  # rice
    syllabus = Syllabus(words=(rice,),
                        categories=(Category(name=CategoryName("Food"),
                                            members=frozenset({rice.id})),))
    assert syllabus.category_of(rice.id) == "Food"


def test_category_of_is_none_for_a_word_in_no_category():
    # a closure word (spec 1: pair members and grapheme keywords are in no
    # category)
    keyword = word("chicken", "ไก่")  # chicken
    syllabus = Syllabus(words=(keyword,), categories=())
    assert syllabus.category_of(keyword.id) is None
