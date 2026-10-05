"""Syllabus.order(): sounds before words; receptive before productive per
word; ties by frequency rank / emphasis weight; a sentence dealt right
after its last used word's last Target, a last-word group shorter first
then by text_sha -- the key the fill set's placement reads too; a
sentence using no targeted word last (spec 1, section 3, r24); a
sentence using a word met only by a later-placed sentence dealt directly
after that sentence (r31).
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
    assert positions[("pair", pair.id)] < positions[("word_target", t2.id)]
    # spec 1 r32: a grapheme is no entry; compile deals its card
    assert not [e for e in ordering if e.kind == "grapheme"]


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
    """order() deals a last-word group shorter first, and the fill set's
    placement (spec 1 section 3 clause 3's "an adopted sentence placed
    at or before this one") reads the same key. A (long, lower
    text_sha) and B (short) share last word w and both use the
    sentence-introduced x; B also uses the sentence-introduced y. At its
    own key B, shorter, comes first with two unmet Targets; A introduces
    x at its own key, and B is placed directly after A (r31), where y is
    its one unmet Target: order() deals A then B, and B fills x and y.
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
    assert sentence_shas == [long_a.text_sha, short_b.text_sha]
    assert syllabus._placement_key(short_b) == (syllabus._placement_key(long_a)
                                                + (short_b.word_count, short_b.text_sha))
    assert syllabus.fill_set(short_b) == (t_w, t_x, t_y)
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


# --- placement by met position (r31) -----------------------------------------

def _sentence_shas(syllabus):
    return [e.id for e in syllabus.order() if e.kind == "sentence"]


def test_order_deals_a_sentence_directly_after_the_sentence_that_met_its_word():
    """gun and very are sentence-introduced; `met` (market, very) is the
    only other sentence filling very and is placed at market's entry.
    `late` (gun, very) has very's entry for its own, yet is dealt right
    after `met`, before old's Target and the sentence placed there."""
    gun = word("gun", "กระบอก")     # classifier for guns -- sentence-introduced
    very = word("very", "มาก")       # very -- sentence-introduced
    market = word("market", "ตลาด")  # market
    old = word("old", "เก่า")         # old
    to = thai_of(gun, very, market, old)
    met = sentence(((market.id, very.id),), to)   # market (is) very ...
    late = sentence(((gun.id, very.id),), to)     # (this) gun (is) very ...
    after = sentence(((old.id,),), to)            # old
    syllabus = Syllabus(
        words=(gun, very, market, old),
        targets=(target("gun/receptive", "gun", introduction="sentence"),
                 target("very/receptive", "very", introduction="sentence"),
                 target("market/receptive", "market"), target("old/receptive", "old")),
        sentences=(late, after, met),
        frequency={gun.id: 1, very.id: 2, market.id: 3, old.id: 4})
    entries = syllabus.order()
    positions = {(e.kind, e.id): i for i, e in enumerate(entries)}
    assert _sentence_shas(syllabus) == [met.text_sha, late.text_sha, after.text_sha]
    assert positions[("sentence", late.text_sha)] == positions[("sentence", met.text_sha)] + 1
    assert positions[("sentence", late.text_sha)] < positions[("word_target", "old/receptive")]


def _chain():
    """w2, w3 and w1 are sentence-introduced, in that order, then late.
    A (w1, late) introduces w1 at late's entry. B (w2, w1) has w1's
    entry for its own and two unmet words there; A meets w1, so B
    follows A and introduces w2. C (w3, w2) likewise follows B."""
    w2 = word("w2", "สอง")    # two -- sentence-introduced
    w3 = word("w3", "สาม")    # three -- sentence-introduced
    w1 = word("w1", "หนึ่ง")  # one -- sentence-introduced
    late = word("late", "สาย")  # late
    to = thai_of(w1, w2, w3, late)
    a = sentence(((w1.id, late.id),), to)  # one late
    b = sentence(((w2.id, w1.id),), to)    # two one
    c = sentence(((w3.id, w2.id),), to)    # three two
    syllabus = Syllabus(
        words=(w1, w2, w3, late),
        targets=(target("w1/receptive", "w1", introduction="sentence"),
                 target("w2/receptive", "w2", introduction="sentence"),
                 target("w3/receptive", "w3", introduction="sentence"),
                 target("late/receptive", "late")),
        sentences=(c, b, a),
        frequency={w2.id: 1, w3.id: 2, w1.id: 3, late.id: 4})
    return syllabus, a, b, c


def test_a_chain_of_met_sentences_orders_each_after_the_one_that_met_its_word():
    syllabus, a, b, c = _chain()
    key = syllabus._placement_key
    assert key(b) == key(a) + (b.word_count, b.text_sha)
    assert key(c) == key(b) + (c.word_count, c.text_sha)
    assert _sentence_shas(syllabus) == [a.text_sha, b.text_sha, c.text_sha]
    assert [len(syllabus.fill_set(s)) for s in (a, b, c)] == [2, 2, 2]


def test_sentences_anchored_on_one_sentence_order_by_word_count_then_text_sha():
    """Three sentences each pair very with a word of their own and are
    met by `met` alone: they follow it shortest first, ties by text_sha."""
    small = word("small", "เล็ก")    # small
    very = word("very", "มาก")       # very -- sentence-introduced
    a = word("a", "ก")              # a -- sentence-introduced
    b = word("b", "ข")              # b -- sentence-introduced
    c = word("c", "ค")              # c -- sentence-introduced
    market = word("market", "ตลาด")  # market
    words = (small, very, a, b, c, market)
    to = thai_of(*words)
    met = sentence(((market.id, very.id),), to)             # market (is) very ...
    two_a = sentence(((a.id, very.id),), to)
    two_b = sentence(((b.id, very.id),), to)
    three = sentence(((small.id, c.id, very.id),), to)
    syllabus = Syllabus(
        words=words,
        targets=(target("small/receptive", "small"), target("market/receptive", "market"),
                 *(target(f"{w.id}/receptive", w.id, introduction="sentence")
                   for w in (very, a, b, c))),
        sentences=(three, two_b, two_a, met),
        frequency={w.id: rank for rank, w in enumerate(words, 1)})
    anchored = sorted((two_a, two_b, three), key=lambda s: (s.word_count, s.text_sha))
    assert anchored[-1] is three
    assert _sentence_shas(syllabus) == [met.text_sha] + [s.text_sha for s in anchored]


def test_a_sentence_using_no_targeted_word_is_still_placed_after_an_anchored_one():
    syllabus, a, b, c = _chain()
    orphan = word("orphan", "เอก")  # a word with no Target
    syllabus = Syllabus(words=syllabus.words + (orphan,), targets=syllabus.targets,
                        frequency=syllabus.frequency,
                        sentences=syllabus.sentences
                        + (sentence(((orphan.id,),), thai_of(orphan)),))
    assert _sentence_shas(syllabus)[:3] == [a.text_sha, b.text_sha, c.text_sha]
    assert syllabus.order()[-1].id == syllabus.sentences[-1].text_sha


def test_a_sentence_with_no_sentence_introduced_word_keeps_its_own_key():
    w1 = word("w1", "หมา")  # dog
    w2 = word("w2", "แมว")  # cat
    w3 = word("w3", "สอง")  # two
    to = thai_of(w1, w2, w3)
    sentences = (sentence(((w1.id, w3.id),), to), sentence(((w1.id, w2.id, w3.id),), to),
                 sentence(((w2.id,),), to))
    syllabus = Syllabus(words=(w1, w2, w3),
                        targets=(target("t1", "w1"), target("t2", "w2"), target("t3", "w3")),
                        sentences=sentences)
    for s in sentences:
        assert syllabus._placement_key(s) == (
            syllabus._word_last_position[syllabus.last_used_word(s)], s.word_count, s.text_sha)


def test_a_sentence_anchored_on_a_letter_name_sentence_stays_ahead_of_every_word_target():
    """Four letter names; n1 and n2 are sentence-introduced. A (n1, n3,
    n4) introduces n1; B (n2, n1) is shorter, so its own key precedes
    A's, with two unmet words there. B follows A, and both stay after
    the sounds block, ahead of rice's Target."""
    keywords = [word(f"k{i}", t) for i, t in enumerate(("กา", "ขา", "คา", "งา"), 1)]
    names = [word(f"n{i}", t) for i, t in enumerate(("กอ ไก่", "ขอ ไข่", "คอ ควาย", "งอ งู"), 1)]
    graphemes = [Grapheme.create(symbol=sym, kind="consonant", sound=snd, consonant_class=cls,
                                 keyword_word=k, name_word=n)
                 for sym, snd, cls, k, n in zip("กขคง", ("k", "kh", "kh", "ng"),
                                                ("mid", "high", "low", "low"), keywords, names)]
    rice = word("rice", "ข้าว")  # rice
    n1, n2, n3, n4 = names
    to = thai_of(*names)
    a = sentence(((n1.id, n3.id, n4.id),), to)
    b = sentence(((n2.id, n1.id),), to)
    syllabus = Syllabus(
        words=(*keywords, *names, rice), graphemes=tuple(graphemes), sentences=(b, a),
        targets=(target("n1/receptive", "n1", introduction="sentence"),
                 target("n2/receptive", "n2", introduction="sentence"),
                 target("n3/receptive", "n3"), target("n4/receptive", "n4"),
                 target("rice/receptive", "rice")),
        profile=Profile(register="male_colloquial"))
    entries = syllabus.order()
    positions = {(e.kind, e.id): i for i, e in enumerate(entries)}
    assert syllabus._placement_key(b) == syllabus._placement_key(a) + (b.word_count, b.text_sha)
    assert positions[("sentence", b.text_sha)] == positions[("sentence", a.text_sha)] + 1
    assert positions[("sentence", b.text_sha)] < positions[("word_target", "rice/receptive")]
    assert positions[("sentence", a.text_sha)] > max(
        positions[("word_target", t)] for t in ("n3/receptive", "n4/receptive"))


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


# --- a name word's Targets sit inside the sounds block (spec 1 r16; r32
# gives a name word none and takes graphemes out of order()) -------------

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


def test_a_name_words_targets_open_the_order_receptive_first():
    ordering = _grapheme_syllabus().order()
    kinds_ids = [(e.kind, e.id) for e in ordering]
    assert kinds_ids[:2] == [("word_target", "name-chicken/receptive"),
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


def test_two_name_words_sentence_groups_are_ordered_by_placement_key_not_hash_order():
    """`_word_last_position` seeds every name word at -1 from
    `name_word_ids`, a frozenset whose iteration order varies with
    PYTHONHASHSEED. order() deals the sentences at -1 by placement key
    (word count, then text_sha), not by that iteration, so two sentences
    each keyed to a different name word always land in the same relative
    order -- here name b's first, the shorter, against word-id order."""
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
    s_a = sentence(((name_a.id, name_a.id),), to, gloss="name a twice")
    syllabus = Syllabus(
        words=(keyword_a, keyword_b, name_a, name_b), graphemes=(g_a, g_b),
        sentences=(s_b, s_a),
        targets=(target("name-a/receptive", "name-a"),
                 target("name-b/receptive", "name-b")),
        profile=Profile(register="male_colloquial"))
    assert syllabus._word_last_position["name-a"] == -1
    assert syllabus._word_last_position["name-b"] == -1
    assert s_b.word_count < s_a.word_count
    entries = syllabus.order()
    sentence_shas = [e.id for e in entries if e.kind == "sentence"]
    assert sentence_shas == [s_b.text_sha, s_a.text_sha]
