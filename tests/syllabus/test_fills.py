"""Syllabus.fills(sentence, target): the one definition of "this text serves
that target" (spec 1, section 3): word at a token boundary, voice satisfies
skill, strict i+1 with a novelty budget of 1 for a sentence-introducing
target, 0 otherwise.
"""
from thai_syllabus.profile import Profile
from thai_syllabus.syllabus import Syllabus

from .builders import sentence, target, word
from .fakes import FakeTokenizer


def base_syllabus(words, targets, tokenizer, frequency=None):
    return Syllabus(words=words, targets=targets,
                    profile=Profile(register="male_colloquial"),
                    tokenizer=tokenizer, frequency=frequency or {})


# --- clause 1: token boundary membership -----------------------------------

def test_fills_when_the_word_is_the_whole_token():
    rice = word("rice", "ข้าว")  # rice
    t = target("rice/receptive", "rice", "receptive")
    s = sentence("ผมกินข้าว", voice="learner_voice")  # I eat rice
    tok = FakeTokenizer({s.text: ["ผม", "กิน", "ข้าว"]})
    syllabus = base_syllabus((rice,), (t,), tok)
    assert syllabus.fills(s, t) is True


def test_does_not_fill_when_the_word_is_only_a_substring_not_a_token():
    rice = word("rice", "ข้าว")  # rice
    t = target("rice/receptive", "rice", "receptive")
    s = sentence("มีข้าวของเยอะ", voice="learner_voice")  # lots of stuff
    # tokenizer splits it as one unrelated token that merely contains the
    # substring, not a boundary match
    tok = FakeTokenizer({s.text: ["มีข้าวของเยอะ"]})
    syllabus = base_syllabus((rice,), (t,), tok)
    assert syllabus.fills(s, t) is False


def test_fills_on_a_compound_token_that_is_two_known_words_joined():
    """A token that is the concatenation of two known words counts as a
    boundary match for each of them (startswith/endswith): the compound
    token "ตัวอย่าง" (example) is both a startswith-match for "ตัว" (body)
    and an endswith-match for "อย่าง" (kind), so each is a used word.
    """
    body = word("body", "ตัว")  # body/classifier -- already met
    kind = word("kind", "อย่าง")  # kind/sort -- the target under test, only
                                 # appears as the suffix of the compound token
    t_body = target("body/receptive", "body", "receptive")
    t_kind = target("kind/receptive", "kind", "receptive")
    s = sentence("ตัวอย่าง", voice="learner_voice")  # example
    tok = FakeTokenizer({s.text: ["ตัวอย่าง"]})
    syllabus = base_syllabus((body, kind), (t_body, t_kind), tok,
                             frequency={body.id: 1, kind.id: 2})
    assert syllabus.fills(s, t_kind) is True


# --- clause 2: voice satisfies skill -----------------------------------------

def test_other_voice_fills_a_receptive_target():
    dog = word("dog", "หมา")  # dog
    t = target("dog/receptive", "dog", "receptive")
    s = sentence("หมาน่ารักนะคะ", voice="other_voice")  # the dog is cute (female speaker)
    tok = FakeTokenizer({s.text: ["หมา", "น่ารัก", "นะ", "คะ"]})
    syllabus = base_syllabus((dog,), (t,), tok)
    assert syllabus.fills(s, t) is True


def test_other_voice_does_not_fill_a_productive_target():
    dog = word("dog", "หมา")  # dog
    t = target("dog/productive", "dog", "productive")
    s = sentence("หมาน่ารักนะคะ", voice="other_voice")  # the dog is cute (female speaker)
    tok = FakeTokenizer({s.text: ["หมา", "น่ารัก", "นะ", "คะ"]})
    syllabus = base_syllabus((dog,), (t,), tok)
    assert syllabus.fills(s, t) is False


def test_learner_voice_fills_a_productive_target():
    dog = word("dog", "หมา")  # dog
    t = target("dog/productive", "dog", "productive")
    s = sentence("ผมมีหมา", voice="learner_voice")  # I have a dog
    tok = FakeTokenizer({s.text: ["ผม", "มี", "หมา"]})
    syllabus = base_syllabus((dog,), (t,), tok)
    assert syllabus.fills(s, t) is True


# --- clause 3: strict i+1 / novelty budget -----------------------------------

def test_a_sentence_using_only_already_met_words_fills_its_target():
    rice = word("rice", "ข้าว")  # rice
    eat = word("eat", "กิน")  # eat
    t_eat_earlier = target("eat/receptive", "eat", "receptive")
    t_rice = target("rice/receptive", "rice", "receptive")
    s = sentence("กินข้าว", voice="learner_voice")  # eat rice
    tok = FakeTokenizer({s.text: ["กิน", "ข้าว"]})
    # "eat" ordered before "rice" by frequency so it is already met.
    syllabus = base_syllabus((rice, eat), (t_eat_earlier, t_rice), tok,
                             frequency={eat.id: 1, rice.id: 2})
    assert syllabus.fills(s, t_rice) is True


def test_a_sentence_using_a_word_with_no_earlier_target_does_not_fill_a_picture_card_target():
    rice = word("rice", "ข้าว")  # rice
    unmet = word("unmet", "จาน")  # plate -- no target at all
    t_rice = target("rice/receptive", "rice", "receptive",
                    introduction="picture_card")
    s = sentence("ข้าวอยู่ในจาน", voice="learner_voice")  # the rice is on the plate
    tok = FakeTokenizer({s.text: ["ข้าว", "อยู่ใน", "จาน"]})
    syllabus = base_syllabus((rice, unmet), (t_rice,), tok)
    assert syllabus.fills(s, t_rice) is False


def test_one_new_word_is_permitted_when_the_target_is_sentence_introduced():
    """A sentence-introduced target gets a novelty budget of 1: one
    companion word may ride along without an earlier target of its own.
    """
    rice = word("rice", "ข้าว")  # rice -- being introduced by this sentence
    companion = word("companion", "จาน")  # plate -- also new, no target
    t_rice = target("rice/receptive", "rice", "receptive",
                    introduction="sentence")
    s = sentence("ข้าวอยู่ในจาน", voice="learner_voice")  # the rice is on the plate
    tok = FakeTokenizer({s.text: ["ข้าว", "อยู่ใน", "จาน"]})
    syllabus = base_syllabus((rice, companion), (t_rice,), tok)
    assert syllabus.fills(s, t_rice) is True


def test_the_novelty_budget_does_not_cover_a_second_new_word():
    rice = word("rice", "ข้าว")  # rice -- being introduced by this sentence
    unmet1 = word("unmet1", "จาน")  # plate -- new, no target
    unmet2 = word("unmet2", "ช้อน")  # spoon -- also new, no target
    t_rice = target("rice/receptive", "rice", "receptive",
                    introduction="sentence")
    s = sentence("ข้าวอยู่ในจานกับช้อน", voice="learner_voice")  # the rice is on the plate with a spoon
    tok = FakeTokenizer({s.text: ["ข้าว", "อยู่ใน", "จาน", "กับ", "ช้อน"]})
    syllabus = base_syllabus((rice, unmet1, unmet2), (t_rice,), tok)
    assert syllabus.fills(s, t_rice) is False


def test_a_word_whose_target_comes_later_still_lets_the_sentence_fill():
    """The sentence enters the order after its LAST word's target; a used
    word targeted later than the exercised target is met by then. The
    parsimony case: multi-fill texts enter late, they do not fail."""
    rice = word("rice", "ข้าว")     # rice
    spoon = word("spoon", "ช้อน")   # spoon -- targeted, but later than rice
    t_rice = target("rice/receptive", "rice", "receptive",
                    introduction="picture_card")
    t_spoon = target("spoon/receptive", "spoon", "receptive",
                     introduction="picture_card")
    s = sentence("ข้าวกับช้อน", voice="learner_voice")  # rice with a spoon
    tok = FakeTokenizer({s.text: ["ข้าว", "กับ", "ช้อน"]})
    syllabus = base_syllabus((rice, spoon), (t_rice, t_spoon), tok,
                             frequency={rice.id: 1, spoon.id: 99})
    assert syllabus.fills(s, t_rice) is True
