"""Syllabus.order(): sounds before words; receptive before productive per
word; ties by frequency rank / emphasis weight; a sentence dealt right
after its last used word's last Target, a last-word group shorter first
then by text_sha -- the key the fill set's placement reads too; a
sentence using no targeted word last (spec 1, section 3, r24).
"""
import pytest

from thai_syllabus.entities import Category, Grapheme, MinimalPair, SoundConfusion
from thai_syllabus.ids import CategoryName, ConfusionId, PairId
from thai_syllabus.profile import Profile
from thai_syllabus.syllabus import Syllabus

from .builders import sentence, target, thai_of, word


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
        profile=Profile(register="male_colloquial"),
    )
    ordering = syllabus.order()
    positions = {(e.kind, e.id): i for i, e in enumerate(ordering)}
    assert positions[("pair", pair.id)] < positions[("word_target", t1.id)]
    assert positions[("grapheme", grapheme.symbol)] < positions[("word_target", t1.id)]
    assert positions[("pair", pair.id)] < positions[("word_target", t2.id)]


def test_receptive_precedes_productive_for_the_same_word():
    rice = word("rice", "ข้าว")  # rice
    receptive = target("rice/receptive", "rice", "receptive")
    productive = target("rice/productive", "rice", "productive")
    syllabus = Syllabus(words=(rice,), targets=(productive, receptive),
                        profile=Profile(register="male_colloquial"))
    ordering = [e.id for e in syllabus.order() if e.kind == "word_target"]
    assert ordering.index(receptive.id) < ordering.index(productive.id)


def test_ties_are_broken_by_frequency_rank_over_emphasis_weight():
    common = word("common", "บ้าน")  # house -- frequent
    rare = word("rare", "ปราสาท")  # palace -- infrequent
    t_common = target("common/receptive", "common", "receptive")
    t_rare = target("rare/receptive", "rare", "receptive")
    syllabus = Syllabus(
        words=(common, rare), targets=(t_rare, t_common),
        profile=Profile(register="male_colloquial"),
        frequency={common.id: 10, rare.id: 5000},
    )
    ordering = [e.id for e in syllabus.order() if e.kind == "word_target"]
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
        frequency={common.id: 10, rare.id: 100},
        categories=(Category(name=CategoryName("food"), members=frozenset({rare.id})),),
    )
    ordering = [e.id for e in syllabus.order() if e.kind == "word_target"]
    assert ordering.index(t_rare.id) < ordering.index(t_common.id)


# --- sentence entries ------------------------------------------------------

def test_order_places_a_sentence_after_every_word_it_uses():
    rice = word("rice", "ข้าว")  # rice
    eat = word("eat", "กิน")  # eat
    t1 = target("t1", "rice")
    t2 = target("t2", "eat")
    to = thai_of(rice, eat)
    s = sentence(((eat.id, rice.id),), to, gloss="eat rice")  # eat rice
    syllabus = Syllabus(words=(rice, eat), targets=(t1, t2), sentences=(s,))
    entries = syllabus.order()
    pos = {(e.kind, e.id): i for i, e in enumerate(entries)}
    s = next(e for e in entries if e.kind == "sentence")
    assert pos[("sentence", s.id)] > max(pos[("word_target", "t1")], pos[("word_target", "t2")])


def test_sentence_is_dealt_directly_after_its_last_used_words_last_target():
    """Three words, one sentence using the first two: the sentence lands
    right after its last used word's own last Target entry, and before
    the next word's first entry (r24) -- no longer after every word."""
    w1 = word("w1", "หนึ่ง")  # one
    w2 = word("w2", "สอง")  # two
    w3 = word("w3", "สาม")  # three
    t1 = target("t1", "w1")
    t2 = target("t2", "w2")
    t3 = target("t3", "w3")
    to = thai_of(w1, w2, w3)
    s = sentence(((w1.id, w2.id),), to, gloss="one two")
    syllabus = Syllabus(words=(w1, w2, w3), targets=(t1, t2, t3), sentences=(s,))
    entries = syllabus.order()
    positions = {(e.kind, e.id): i for i, e in enumerate(entries)}
    assert syllabus.last_used_word(s) == w2.id
    assert positions[("sentence", s.text_sha)] == positions[("word_target", "t2")] + 1
    assert positions[("sentence", s.text_sha)] < positions[("word_target", "t3")]


def test_sentences_sharing_a_last_word_place_the_shorter_first():
    # These thai strings are chosen so that the *long* sentence's text_sha
    # sorts before the short one's -- a text_sha-only tie-break would put
    # long first, so this only passes when length is compared first.
    w1 = word("w1", "หมา")  # dog
    w2 = word("w2", "แมว")  # cat
    w3 = word("w3", "สอง")  # two
    t1 = target("t1", "w1")
    t2 = target("t2", "w2")
    t3 = target("t3", "w3")
    to = thai_of(w1, w2, w3)
    short = sentence(((w1.id, w3.id),), to, gloss="dog two")  # length 2
    long = sentence(((w1.id, w2.id, w3.id),), to, gloss="dog cat two")  # length 3
    syllabus = Syllabus(words=(w1, w2, w3), targets=(t1, t2, t3), sentences=(long, short))
    assert syllabus.last_used_word(short) == w3.id
    assert syllabus.last_used_word(long) == w3.id
    entries = syllabus.order()
    sentence_shas = [e.id for e in entries if e.kind == "sentence"]
    assert sentence_shas == [short.text_sha, long.text_sha]


def test_sentences_tied_on_length_sharing_a_last_word_tie_on_text_sha():
    w1 = word("w1", "หนึ่ง")  # one
    w2 = word("w2", "สอง")  # two
    t1 = target("t1", "w1")
    t2 = target("t2", "w2")
    to = thai_of(w1, w2)
    a = sentence(((w1.id, w2.id),), to, gloss="one two")
    b = sentence(((w2.id, w1.id),), to, gloss="two one")
    syllabus = Syllabus(words=(w1, w2), targets=(t1, t2), sentences=(a, b))
    assert syllabus.last_used_word(a) == w2.id
    assert syllabus.last_used_word(b) == w2.id
    assert a.text_sha != b.text_sha
    entries = syllabus.order()
    sentence_shas = [e.id for e in entries if e.kind == "sentence"]
    assert sentence_shas == sorted([a.text_sha, b.text_sha])


def test_order_and_the_fill_set_placement_share_one_key_within_a_last_word_group():
    """Fix wave (final review): order() deals a last-word group shorter
    first, so the fill set's placement (spec 1 section 3 clause 3's "an
    adopted sentence placed at or before this one") must read the same
    (last-word position, word count, text_sha) key. A (long, lower
    text_sha) and B (short) share last word w and both use the
    sentence-introduced x; B also uses the sentence-introduced y. B is
    dealt first, so A cannot have met x for B: B carries two unmet
    sentence-introduced Targets and fills nothing, and A -- placed
    after B, whose fill set is empty -- fills x alone.
    """
    x = word("a_x", "ข้าว")  # rice
    y = word("b_y", "ปลา")  # fish
    f = word("c_f", "หมา")  # dog
    g = word("d_g", "แมว")  # cat
    w = word("z_w", "กิน")  # eat
    t_x = target("x", "a_x", introduction="sentence")
    t_y = target("y", "b_y", introduction="sentence")
    t_f = target("f", "c_f")
    t_g = target("g", "d_g")
    t_w = target("w", "z_w")
    to = thai_of(x, y, f, g, w)
    long_a = sentence(((f.id, x.id, g.id, w.id),), to, gloss="A")
    short_b = sentence(((y.id, x.id, w.id),), to, gloss="B")
    syllabus = Syllabus(words=(x, y, f, g, w), targets=(t_x, t_y, t_f, t_g, t_w),
                        sentences=(long_a, short_b))
    assert long_a.text_sha < short_b.text_sha
    assert syllabus.last_used_word(long_a) == syllabus.last_used_word(short_b) == w.id

    sentence_shas = [e.id for e in syllabus.order() if e.kind == "sentence"]
    assert sentence_shas == [short_b.text_sha, long_a.text_sha]
    assert t_x not in syllabus.fill_set(short_b)
    assert syllabus.fill_set(short_b) == ()
    assert t_x in syllabus.fill_set(long_a)


def test_a_sentence_whose_last_used_word_has_no_target_is_placed_last():
    w1 = word("w1", "หนึ่ง")  # one
    orphan = word("orphan", "เอก")  # a word with no Target
    t1 = target("t1", "w1")
    to = thai_of(w1, orphan)
    s = sentence(((orphan.id,),), to, gloss="orphan only")
    syllabus = Syllabus(words=(w1, orphan), targets=(t1,), sentences=(s,))
    with pytest.raises(ValueError, match=s.text_sha):
        syllabus.last_used_word(s)
    entries = syllabus.order()
    assert entries[-1].kind == "sentence"
    assert entries[-1].id == s.text_sha


# --- Syllabus.last_used_word ------------------------------------------------

def test_last_used_word_picks_the_word_with_the_greatest_last_target_position():
    # "eat" < "rice" by word id, so eat's target sorts before rice's
    # (order()'s tie-break: frequency tied at inf for both, then word id)
    # -- rice's target position is the greater of the two.
    rice = word("rice", "ข้าว")  # rice
    eat = word("eat", "กิน")  # eat
    t_rice = target("t1", "rice")
    t_eat = target("t2", "eat")
    to = thai_of(rice, eat)
    s = sentence(((eat.id, rice.id),), to, gloss="eat rice")  # eat rice
    syllabus = Syllabus(words=(rice, eat), targets=(t_rice, t_eat), sentences=(s,))
    assert syllabus.last_used_word(s) == rice.id


def test_last_used_word_raises_naming_the_text_sha_when_no_used_word_has_a_target():
    rice = word("rice", "ข้าว")  # rice
    to = thai_of(rice)
    s = sentence(((rice.id,),), to, gloss="rice")  # rice, no Target on rice
    syllabus = Syllabus(words=(rice,), targets=(), sentences=(s,))
    with pytest.raises(ValueError, match=s.text_sha):
        syllabus.last_used_word(s)


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


# --- a name word's Targets sit inside the sounds block (spec 1 r16) -------

def _grapheme_syllabus():
    chicken = word("chicken", "ไก่", "chicken")            # ไก่: chicken
    name = word("name-chicken", "กอ ไก่", "the letter ก's recited name")  # กอ ไก่
    rice = word("rice", "ข้าว", "rice")                    # ข้าว: rice
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    return Syllabus(
        words=(chicken, name, rice), graphemes=(g,),
        targets=(target("rice/receptive", "rice"),
                 target("name-chicken/productive", "name-chicken", skill="productive"),
                 target("name-chicken/receptive", "name-chicken")),
        profile=Profile(register="male_colloquial"),
        frequency={"rice": 1})


def test_a_name_words_targets_follow_its_grapheme_receptive_first():
    ordering = _grapheme_syllabus().order()
    kinds_ids = [(e.kind, e.id) for e in ordering]
    assert kinds_ids[:3] == [("grapheme", "ก"),
                             ("word_target", "name-chicken/receptive"),
                             ("word_target", "name-chicken/productive")]


def test_a_name_words_targets_precede_every_ordinary_word_target():
    ordering = _grapheme_syllabus().order()
    positions = {e.id: i for i, e in enumerate(ordering)}
    assert positions["name-chicken/productive"] < positions["rice/receptive"]


def test_a_name_words_targets_are_placed_exactly_once():
    ids = [e.id for e in _grapheme_syllabus().order() if e.kind == "word_target"]
    assert ids.count("name-chicken/receptive") == 1
    assert ids.count("name-chicken/productive") == 1
    assert ids == ["name-chicken/receptive", "name-chicken/productive", "rice/receptive"]


def test_a_sentence_using_a_name_word_is_still_placed():
    """`_word_last_position` keeps order() total: a name word's Targets
    are not in _ordered_targets, so without a seeded position a sentence
    naming one would raise. No drafted sentence uses a letter name in
    practice; the function must not depend on that."""
    chicken = word("chicken", "ไก่", "chicken")            # ไก่: chicken
    name = word("name-chicken", "กอ ไก่", "the letter ก's recited name")  # กอ ไก่
    rice = word("rice", "ข้าว", "rice")                    # ข้าว: rice
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    s = sentence(((name.id,),), thai_of(chicken, name, rice), gloss="the name of ก")
    syllabus = Syllabus(
        words=(chicken, name, rice), graphemes=(g,), sentences=(s,),
        targets=(target("rice/receptive", "rice"),
                 target("name-chicken/receptive", "name-chicken")),
        profile=Profile(register="male_colloquial"))
    ordering = syllabus.order()
    positions = {(e.kind, e.id): i for i, e in enumerate(ordering)}
    assert ("sentence", s.text_sha) in positions
    assert syllabus._word_last_position["name-chicken"] == -1


def test_two_name_words_sentence_groups_are_ordered_by_word_id_not_hash_order():
    """`_word_last_position` seeds every name word at -1 from
    `name_word_ids`, a frozenset, so iterating `_word_last_position.items()`
    to find the -1 words visits them in the frozenset's hash-dependent
    order (varies with PYTHONHASHSEED). order() must instead visit them in
    a fixed (sorted-by-word-id) order, so two sentences each keyed to a
    different name word always land in the same relative order."""
    keyword_a = word("keyword-a", "กา")
    keyword_b = word("keyword-b", "ขา")
    name_a = word("name-a", "กอ ไก่", "the letter ก's recited name")
    name_b = word("name-b", "ขอ ไข่", "the letter ข's recited name")
    g_a = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                          keyword_word=keyword_a, name_word=name_a)
    g_b = Grapheme.create(symbol="ข", kind="consonant", sound="kh", consonant_class="high",
                          keyword_word=keyword_b, name_word=name_b)
    to = thai_of(keyword_a, keyword_b, name_a, name_b)
    # Built out of word-id order (b before a) so a hash-order bug would
    # not be masked by construction order coinciding with the fix.
    s_b = sentence(((name_b.id,),), to, gloss="name b only")
    s_a = sentence(((name_a.id,),), to, gloss="name a only")
    syllabus = Syllabus(
        words=(keyword_a, keyword_b, name_a, name_b), graphemes=(g_a, g_b),
        sentences=(s_b, s_a),
        targets=(target("name-a/receptive", "name-a"),
                 target("name-b/receptive", "name-b")),
        profile=Profile(register="male_colloquial"))
    assert syllabus._word_last_position["name-a"] == -1
    assert syllabus._word_last_position["name-b"] == -1
    entries = syllabus.order()
    sentence_shas = [e.id for e in entries if e.kind == "sentence"]
    assert sentence_shas == [s_a.text_sha, s_b.text_sha]
