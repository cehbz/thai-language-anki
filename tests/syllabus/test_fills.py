"""Syllabus.fills(sentence, target) and Syllabus.fill_set(sentence): the one
definition of "this text serves that target" (spec 1, section 3): word at a
token boundary, voice satisfies skill, then a sentence-level gate (every
content token a registered word, every used word carrying a Target) and a
novelty rule over the whole fill set -- at most one candidate may be a
sentence-introduced Target no other adopted sentence, placed at or before
this one, already contains.
"""
import pytest

from thai_syllabus.profile import Profile
from thai_syllabus.syllabus import Syllabus, token_is_known

from .builders import sentence, target, word
from .fakes import FakeTokenizer


def base_syllabus(words, targets, tokenizer, frequency=None):
    return Syllabus(words=words, targets=targets,
                    profile=Profile(register="male_colloquial"),
                    tokenizer=tokenizer, frequency=frequency or {})


# --- clause 1: token boundary membership -----------------------------------

def test_fills_when_the_word_is_the_whole_token():
    rice = word("rice", "ข้าว")  # rice
    i_word = word("i", "ผม")  # I -- registered with a Target so it doesn't empty the fill set
    eat = word("eat", "กิน")  # eat -- registered with a Target so it doesn't empty the fill set
    t = target("rice/receptive", "rice", "receptive")
    s = sentence("ผมกินข้าว", voice="learner_voice")  # I eat rice
    tok = FakeTokenizer({s.text: ["ผม", "กิน", "ข้าว"]})
    syllabus = base_syllabus((rice, i_word, eat),
                             (t, target("i/receptive", "i"), target("eat/receptive", "eat")), tok)
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
    """A token that is the concatenation of two known words decomposes
    wholly into them (Syllabus.decompose): the compound token "ตัวอย่าง"
    (example) splits into "ตัว" (body) and "อย่าง" (kind), and each is a
    used word.
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


def test_compound_with_unregistered_remainder_is_new():
    # token "กินข้าว" (eat-rice) with "กิน" (eat) registered and "ข้าว" (rice) not
    eat = word("eat", "กิน")  # eat
    t = target("eat/receptive", "eat", "receptive")
    s = sentence("กินข้าว", voice="learner_voice", gloss="eat rice")  # eat rice
    tok = FakeTokenizer({s.text: ["กินข้าว"]})
    syllabus = base_syllabus((eat,), (t,), tok)
    assert syllabus.fills(s, t) is False  # "rice" is unknown and has no Target


@pytest.mark.parametrize("token,known,expected", [
    ("กินข้าว", {"กิน", "ข้าว"}, True),        # eat-rice: both halves known
    ("กินข้าว", {"กิน"}, False),               # eat-rice: the "rice" remainder is unknown
    ("โรงพยาบาล", {"ยา"}, False),              # hospital contains medicine mid-token; not a boundary match
    ("ตัวอย่าง", {"ตัวอย่าง"}, True),           # example: the compound itself is a registered Word
    ("", {"กิน"}, False),                      # no token at all: nothing to know
    ("กิน", set(), False),                     # eat, with no known words at all
])
def test_token_known_table(token, known, expected):
    assert token_is_known(token, known) is expected


# --- clause 2: voice satisfies skill -----------------------------------------

def test_other_voice_fills_a_receptive_target():
    dog = word("dog", "หมา")  # dog
    cute = word("cute", "น่ารัก")  # registered with a Target so it doesn't empty the fill set
    na = word("na", "นะ")  # particle, registered likewise
    kha = word("kha", "คะ")  # polite female particle, registered likewise
    t = target("dog/receptive", "dog", "receptive")
    s = sentence("หมาน่ารักนะคะ", voice="other_voice")  # the dog is cute (female speaker)
    tok = FakeTokenizer({s.text: ["หมา", "น่ารัก", "นะ", "คะ"]})
    syllabus = base_syllabus((dog, cute, na, kha),
                             (t, target("cute/receptive", "cute"), target("na/receptive", "na"),
                              target("kha/receptive", "kha")), tok)
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
    i_word = word("i", "ผม")  # registered with a Target so it doesn't empty the fill set
    have = word("have", "มี")  # registered with a Target so it doesn't empty the fill set
    t = target("dog/productive", "dog", "productive")
    s = sentence("ผมมีหมา", voice="learner_voice")  # I have a dog
    tok = FakeTokenizer({s.text: ["ผม", "มี", "หมา"]})
    syllabus = base_syllabus((dog, i_word, have),
                             (t, target("i/receptive", "i"), target("have/receptive", "have")), tok)
    assert syllabus.fills(s, t) is True


# --- clause 3: the sentence-level gate ----------------------------------

def test_a_sentence_using_only_targeted_words_fills_its_target():
    rice = word("rice", "ข้าว")  # rice
    eat = word("eat", "กิน")  # eat
    t_eat = target("eat/receptive", "eat", "receptive")
    t_rice = target("rice/receptive", "rice", "receptive")
    s = sentence("กินข้าว", voice="learner_voice")  # eat rice
    tok = FakeTokenizer({s.text: ["กิน", "ข้าว"]})
    syllabus = base_syllabus((rice, eat), (t_eat, t_rice), tok,
                             frequency={eat.id: 1, rice.id: 2})
    assert syllabus.fills(s, t_rice) is True


def test_a_word_with_no_target_at_all_empties_the_fill_set():
    rice = word("rice", "ข้าว")  # rice
    untargeted = word("untargeted", "จาน")  # plate -- registered, no Target
    t_rice = target("rice/receptive", "rice", "receptive",
                    introduction="picture_card")
    s = sentence("ข้าวจาน", voice="learner_voice")  # rice, plate
    tok = FakeTokenizer({s.text: ["ข้าว", "จาน"]})
    syllabus = base_syllabus((rice, untargeted), (t_rice,), tok)
    assert syllabus.fills(s, t_rice) is False


# --- clause 3: the sentence-introduced novelty rule, over the fill set ------

def test_two_unmet_sentence_introduced_targets_empty_the_fill_set():
    """Two candidate Targets are both sentence-introduced and neither is
    met by any other adopted sentence: more than one unmet
    sentence-introduced Target empties the fill set (spec 1 section 3).
    """
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    s = sentence("ข้าวช้อน", voice="learner_voice")  # rice, spoon
    tok = FakeTokenizer({s.text: ["ข้าว", "ช้อน"]})
    syllabus = base_syllabus((rice, spoon), (t_rice, t_spoon), tok)
    assert syllabus.fill_set(s) == ()
    assert syllabus.fills(s, t_rice) is False
    assert syllabus.fills(s, t_spoon) is False


def test_one_unmet_sentence_introduced_target_fills_all_its_candidates():
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- picture_card, not counted against the novelty rule
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive")
    s = sentence("ข้าวช้อน", voice="learner_voice")  # rice, spoon
    tok = FakeTokenizer({s.text: ["ข้าว", "ช้อน"]})
    syllabus = base_syllabus((rice, spoon), (t_rice, t_spoon), tok)
    assert syllabus.fill_set(s) == (t_rice, t_spoon)


def test_a_gate_failing_adopted_sentence_does_not_meet_a_glue_target():
    """"Met" means the target is IN the other adopted sentence's OWN
    fill set, not merely clauses 1 and 2 (spec 1 section 3): an adopted
    sentence mentioning "rice" at a token boundary, in the right voice,
    but whose own fill set is empty (an unregistered token) does not
    meet rice's glue Target for a later sentence -- both of the later
    sentence's candidates stay unmet and its own fill set empties too.
    """
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    gate_failing = sentence("ข้าวอร่อย", voice="learner_voice")  # rice, tasty (unregistered) -- adopted
    s = sentence("ข้าวช้อน", voice="learner_voice")  # rice, spoon -- the candidate, placed after
    tok = FakeTokenizer({gate_failing.text: ["ข้าว", "อร่อย"], s.text: ["ข้าว", "ช้อน"]})
    syllabus = Syllabus(words=(rice, spoon), targets=(t_rice, t_spoon),
                        sentences=(gate_failing,), profile=Profile(register="male_colloquial"),
                        tokenizer=tok, frequency={rice.id: 1, spoon.id: 2})
    assert syllabus.fill_set(gate_failing) == ()   # "อร่อย" is unregistered
    assert syllabus.fill_set(s) == ()


def test_an_earlier_adopted_sentence_meeting_one_target_lets_the_other_fill():
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    earlier = sentence("ข้าว", voice="learner_voice")  # rice -- adopted, mentions only rice
    s = sentence("ข้าวช้อน", voice="learner_voice")  # rice, spoon -- the candidate
    tok = FakeTokenizer({earlier.text: ["ข้าว"], s.text: ["ข้าว", "ช้อน"]})
    syllabus = Syllabus(words=(rice, spoon), targets=(t_rice, t_spoon),
                        sentences=(earlier,), profile=Profile(register="male_colloquial"),
                        tokenizer=tok, frequency={rice.id: 1, spoon.id: 2})
    assert syllabus.fill_set(s) == (t_rice, t_spoon)


def test_a_later_adopted_sentence_does_not_count_as_meeting():
    """`later` mentions spoon's word, but its own order() position (set
    by "bowl", the word it also uses) is placed after the candidate's --
    it does not count, so both candidates stay unmet and the fill set
    empties."""
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    bowl = word("bowl", "ชาม")  # bowl -- gives `later` a later order() position
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    t_bowl = target("bowl/receptive", "bowl", "receptive")
    later = sentence("ช้อนชาม", voice="learner_voice")  # spoon, bowl -- adopted, placed AFTER s
    s = sentence("ข้าวช้อน", voice="learner_voice")  # rice, spoon -- the candidate
    tok = FakeTokenizer({later.text: ["ช้อน", "ชาม"], s.text: ["ข้าว", "ช้อน"]})
    syllabus = Syllabus(words=(rice, spoon, bowl), targets=(t_rice, t_spoon, t_bowl),
                        sentences=(later,), profile=Profile(register="male_colloquial"),
                        tokenizer=tok, frequency={rice.id: 1, spoon.id: 2, bowl.id: 3})
    assert syllabus.fill_set(s) == ()


def test_an_adopted_sentence_is_not_its_own_meeting_sentence():
    """The candidate `s` is itself already adopted; "other adopted
    sentence" excludes it, so it cannot vacuously satisfy its own
    novelty rule."""
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    s = sentence("ข้าวช้อน", voice="learner_voice")  # rice, spoon -- adopted, uses both itself
    tok = FakeTokenizer({s.text: ["ข้าว", "ช้อน"]})
    syllabus = Syllabus(words=(rice, spoon), targets=(t_rice, t_spoon), sentences=(s,),
                        profile=Profile(register="male_colloquial"), tokenizer=tok,
                        frequency={rice.id: 1, spoon.id: 2})
    assert syllabus.fill_set(s) == ()


def test_an_unregistered_token_empties_the_fill_set_for_every_target():
    """No per-target pass: the sentence-level gate applies once, over the
    whole fill set, not per target checked."""
    rice = word("rice", "ข้าว")  # rice
    spoon = word("spoon", "ช้อน")  # spoon
    t_rice = target("rice/receptive", "rice", "receptive")
    t_spoon = target("spoon/receptive", "spoon", "receptive")
    s = sentence("ข้าวช้อนอร่อย", voice="learner_voice")  # rice, spoon, tasty (unregistered)
    tok = FakeTokenizer({s.text: ["ข้าว", "ช้อน", "อร่อย"]})
    syllabus = base_syllabus((rice, spoon), (t_rice, t_spoon), tok)
    assert syllabus.fill_set(s) == ()
    assert syllabus.fills(s, t_rice) is False
    assert syllabus.fills(s, t_spoon) is False


def test_vocabulary_met_by_includes_words_targeted_at_or_before():
    s = Syllabus(words=(word("a", "ก"), word("b", "ข"), word("c", "ค")),
                targets=(target("a/r", "a"), target("b/r", "b"), target("c/r", "c")),
                frequency={"a": 1, "b": 2, "c": 3}, tokenizer=FakeTokenizer())
    met = s.vocabulary_met_by(s.targets[1])
    assert [w.id for w in met] == ["a", "b"]
    assert len(s.with_sentences([]).sentences) == 0


def test_a_word_targeted_anywhere_in_the_order_still_lets_the_sentence_fill():
    """A used word's Target position does not matter to the sentence-level
    gate: it need only exist (`_words_used` subset of
    `_word_target_positions`), whatever order() places it at. "กับ"
    ("with") is a registered glue Word with its own receptive Target --
    glue words carry a Target like any other (spec 1 section 3)."""
    rice = word("rice", "ข้าว")     # rice
    spoon = word("spoon", "ช้อน")   # spoon -- a much later Target position than rice's
    with_word = word("with", "กับ")  # glue word, registered with its own Target
    t_rice = target("rice/receptive", "rice", "receptive",
                    introduction="picture_card")
    t_spoon = target("spoon/receptive", "spoon", "receptive",
                     introduction="picture_card")
    t_with = target("with/receptive", "with", "receptive")
    s = sentence("ข้าวกับช้อน", voice="learner_voice")  # rice with a spoon
    tok = FakeTokenizer({s.text: ["ข้าว", "กับ", "ช้อน"]})
    syllabus = base_syllabus((rice, spoon, with_word), (t_rice, t_spoon, t_with), tok,
                             frequency={rice.id: 1, with_word.id: 2, spoon.id: 99})
    assert syllabus.fills(s, t_rice) is True


def test_an_unregistered_content_token_empties_the_fill_set_but_a_whitespace_token_does_not():
    """A content token matching no registered Word empties the fill set --
    no per-target exemption. A whitespace-only token carries no
    vocabulary and never counts."""
    rice = word("rice", "ข้าว")  # rice
    t = target("rice/receptive", "rice", "receptive")
    syllabus = base_syllabus((rice,), (t,),
                             FakeTokenizer({"ข้าวอร่อย": ["ข้าว", "อร่อย"], "ข้าว ": ["ข้าว", " "]}))
    unknown_content = sentence("ข้าวอร่อย", voice="learner_voice")  # rice is delicious
    assert syllabus.fills(unknown_content, t) is False  # "อร่อย" matches no registered Word
    whitespace_only = sentence("ข้าว ", voice="learner_voice")
    assert syllabus.fills(whitespace_only, t) is True  # a bare space carries no lexical content
