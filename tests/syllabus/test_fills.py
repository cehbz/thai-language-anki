"""Syllabus.fills(sentence, target) and Syllabus.fill_set(sentence): the one
definition of "this text serves that target" (spec 1, section 3): target.word
among the sentence's own words (a repeated element counted once), voice
satisfies skill, then a sentence-level gate (every used word carrying a
Target) and a novelty rule over the whole fill set -- at most one candidate
may be a sentence-introduced Target no other adopted sentence, placed at or
before this one, already contains.
"""
import dataclasses

import pytest

from thai_syllabus.entities import REPEAT_MARK, Sentence
from thai_syllabus.ids import WordId
from thai_syllabus.profile import Profile
from thai_syllabus.syllabus import Syllabus

from .builders import PROV, sentence, target, thai_of, word


def base_syllabus(words, targets, frequency=None):
    return Syllabus(words=words, targets=targets,
                    profile=Profile(register="male_colloquial"),
                    frequency=frequency or {})


# --- clause 1: element membership -------------------------------------------

def test_fills_when_the_word_is_an_element():
    rice = word("rice", "ข้าว")  # rice
    i_word = word("i", "ผม")  # I -- registered with a Target so it doesn't empty the fill set
    eat = word("eat", "กิน")  # eat -- registered with a Target so it doesn't empty the fill set
    t = target("rice/receptive", "rice", "receptive")
    to = thai_of(rice, i_word, eat)
    s = sentence(((i_word.id, eat.id, rice.id),), to, voice="learner_voice")  # I eat rice
    syllabus = base_syllabus((rice, i_word, eat),
                             (t, target("i/receptive", "i"), target("eat/receptive", "eat")))
    assert syllabus.fills(s, t) is True


def test_does_not_fill_when_the_word_is_only_a_substring_of_another_words_form():
    """Two Words share one Thai form ("ผม") -- "i-male-speaker" (I) and
    "hair" (hair on the head), a real Thai homograph. A clause naming
    "hair" fills hair's Target, never "i-male-speaker"'s -- fill_set
    reads element word identity, not the rendered text (spec 1 section
    1: a shared form is a homograph, and a sentence names the sense).
    """
    i_male_speaker = word("i-male-speaker", "ผม", "I (male speaker)")
    hair = word("hair-on-the-head", "ผม", "hair (on the head)")
    t_i = target("i/receptive", "i-male-speaker", "receptive")
    t_hair = target("hair/receptive", "hair-on-the-head", "receptive")
    to = thai_of(hair)
    s = sentence(((hair.id,),), to, voice="learner_voice")  # hair
    syllabus = base_syllabus((i_male_speaker, hair), (t_i, t_hair))
    assert syllabus.fills(s, t_hair) is True
    assert syllabus.fills(s, t_i) is False


def test_a_repeated_element_fills_its_target_once():
    fast = word("fast", "เร็ว")  # fast
    run = word("run", "วิ่ง")  # run -- registered with a Target so it doesn't empty the fill set
    t_fast = target("fast/receptive", "fast", "receptive")
    t_run = target("run/receptive", "run", "receptive")
    to = thai_of(fast, run)
    s = sentence(((run.id, (fast.id, REPEAT_MARK)),), to,
                voice="learner_voice")  # run fast-fast (reduplicated)
    syllabus = base_syllabus((fast, run), (t_fast, t_run))
    assert syllabus.fills(s, t_fast) is True


# --- clause 2: voice satisfies skill -----------------------------------------

def test_other_voice_fills_a_receptive_target():
    dog = word("dog", "หมา")  # dog
    cute = word("cute", "น่ารัก")  # registered with a Target so it doesn't empty the fill set
    na = word("na", "นะ")  # particle, registered likewise
    kha = word("kha", "คะ")  # polite female particle, registered likewise
    t = target("dog/receptive", "dog", "receptive")
    to = thai_of(dog, cute, na, kha)
    s = sentence(((dog.id, cute.id, na.id, kha.id),), to,
                voice="other_voice")  # the dog is cute (female speaker)
    syllabus = base_syllabus((dog, cute, na, kha),
                             (t, target("cute/receptive", "cute"), target("na/receptive", "na"),
                              target("kha/receptive", "kha")))
    assert syllabus.fills(s, t) is True


def test_other_voice_does_not_fill_a_productive_target():
    dog = word("dog", "หมา")  # dog
    t = target("dog/productive", "dog", "productive")
    to = thai_of(dog)
    s = sentence(((dog.id,),), to, voice="other_voice")  # the dog
    syllabus = base_syllabus((dog,), (t,))
    assert syllabus.fills(s, t) is False


def test_learner_voice_fills_a_productive_target():
    dog = word("dog", "หมา")  # dog
    i_word = word("i", "ผม")  # registered with a Target so it doesn't empty the fill set
    have = word("have", "มี")  # registered with a Target so it doesn't empty the fill set
    t = target("dog/productive", "dog", "productive")
    to = thai_of(dog, i_word, have)
    s = sentence(((i_word.id, have.id, dog.id),), to, voice="learner_voice")  # I have a dog
    syllabus = base_syllabus((dog, i_word, have),
                             (t, target("i/receptive", "i"), target("have/receptive", "have")),
                             frequency={i_word.id: 1, have.id: 1, dog.id: 2})
    assert syllabus.fills(s, t) is True


# --- clause 2: a productive fill's voice and marking conditions (r10, r25) --

def test_a_productive_target_fills_from_any_word_the_sentence_uses():
    """Spec 1 section 3, clause 2 (r25): a productive Target is filled by a
    learner-voice sentence using its word, wherever that word falls in
    order -- eat is not the sentence's last used word (rice is), and
    eat/productive fills all the same."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice -- last used word
    t_a_r = target("eat/receptive", "eat", "receptive")
    t_a_p = target("eat/productive", "eat", "productive")
    t_b_r = target("rice/receptive", "rice", "receptive")
    t_b_p = target("rice/productive", "rice", "productive")
    to = thai_of(a, b)
    s = sentence(((a.id, b.id),), to, voice="learner_voice")  # eat rice
    syllabus = base_syllabus((a, b), (t_a_r, t_a_p, t_b_r, t_b_p),
                             frequency={a.id: 1, b.id: 2})
    assert syllabus.last_used_word(s) == b.id   # the fixture's own premise
    assert syllabus.fills(s, t_a_p) is True
    assert syllabus.fills(s, t_b_p) is True


def test_other_voice_fills_no_productive_target_off_the_last_used_word():
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice -- last used word
    t_a_p = target("eat/productive", "eat", "productive")
    t_b_r = target("rice/receptive", "rice", "receptive")
    to = thai_of(a, b)
    s = sentence(((a.id, b.id),), to, voice="other_voice")  # eat rice
    syllabus = base_syllabus((a, b), (t_a_p, t_b_r), frequency={a.id: 1, b.id: 2})
    assert syllabus.fills(s, t_a_p) is False
    assert syllabus.fills(s, t_b_r) is True


def test_marking_that_does_not_admit_the_learner_fills_no_productive_target():
    """Spec 1 section 3, clause 2 (r10): the sentence's marking must be
    empty or the Profile's own sex; ค่ะ marks a female speaker, which the
    male_colloquial profile's learner_speaker does not admit -- neither
    productive Target fills, but both receptive Targets still do (clause
    2's voice-satisfies-skill test alone, unaffected by marking)."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice
    kha = word("kha", "ค่ะ", "female politeness particle", speaker="female")
    t_a_r = target("eat/receptive", "eat", "receptive")
    t_a_p = target("eat/productive", "eat", "productive")
    t_b_r = target("rice/receptive", "rice", "receptive")
    t_b_p = target("rice/productive", "rice", "productive")
    t_kha_r = target("kha/receptive", "kha", "receptive")
    to = thai_of(a, b, kha)
    s = sentence(((a.id, b.id, kha.id),), to, voice="learner_voice")  # eat rice (female speaker)
    syllabus = base_syllabus((a, b, kha), (t_a_r, t_a_p, t_b_r, t_b_p, t_kha_r),
                             frequency={a.id: 1, kha.id: 2, b.id: 3})
    assert syllabus.marking(s) == frozenset({"female"})
    assert syllabus.fills(s, t_a_p) is False
    assert syllabus.fills(s, t_b_p) is False
    assert syllabus.fills(s, t_a_r) is True
    assert syllabus.fills(s, t_b_r) is True


def test_an_other_voice_sentence_fills_its_words_receptive_targets_and_no_productive_one():
    """Spec 1 r28: a drafted sentence whose marking does not admit the
    learner is other_voice; it fills every receptive Target of its words
    and no productive Target (clause 2)."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice
    kha = word("kha", "ค่ะ", "female politeness particle", speaker="female")
    receptive = (target("eat/receptive", "eat"), target("rice/receptive", "rice"),
                 target("kha/receptive", "kha", introduction="sentence"))
    productive = (target("eat/productive", "eat", "productive"),
                  target("rice/productive", "rice", "productive"))
    s = sentence(((a.id, b.id, kha.id),), thai_of(a, b, kha),
                 voice="other_voice")  # eat rice (female speaker)
    syllabus = base_syllabus((a, b, kha), receptive + productive,
                             frequency={a.id: 1, b.id: 2, kha.id: 3})
    assert set(syllabus.fill_set(s)) == set(receptive)
    assert syllabus.productive_fills(s) == ()


def test_marking_that_admits_the_learner_fills_every_used_words_productive_target():
    """ครับ marks a male speaker, the male_colloquial profile's own sex
    (r10): admitted, so both used words' productive Targets fill."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice
    khrap = word("khrap", "ครับ", "male politeness particle", speaker="male")
    t_a_r = target("eat/receptive", "eat", "receptive")
    t_a_p = target("eat/productive", "eat", "productive")
    t_b_r = target("rice/receptive", "rice", "receptive")
    t_b_p = target("rice/productive", "rice", "productive")
    t_khrap_r = target("khrap/receptive", "khrap", "receptive")
    to = thai_of(a, b, khrap)
    s = sentence(((a.id, b.id, khrap.id),), to, voice="learner_voice")  # eat rice (male speaker)
    syllabus = base_syllabus((a, b, khrap), (t_a_r, t_a_p, t_b_r, t_b_p, t_khrap_r),
                             frequency={a.id: 1, khrap.id: 2, b.id: 3})
    assert syllabus.marking(s) == frozenset({"male"})
    assert syllabus.fills(s, t_a_p) is True
    assert syllabus.fills(s, t_b_p) is True


def test_gaps_counts_a_productive_target_off_the_last_used_word_as_filled():
    """gaps().unfilled_targets (spec 1 section 3, clause 2/r25 via
    fill_set): the adopted sentence fills eat/productive though eat is
    not its last used word, so nothing stays open."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice -- last used word
    t_a_r = target("eat/receptive", "eat", "receptive")
    t_a_p = target("eat/productive", "eat", "productive")
    t_b_r = target("rice/receptive", "rice", "receptive")
    t_b_p = target("rice/productive", "rice", "productive")
    to = thai_of(a, b)
    s = sentence(((a.id, b.id),), to, voice="learner_voice")  # eat rice
    syllabus = base_syllabus((a, b), (t_a_r, t_a_p, t_b_r, t_b_p),
                             frequency={a.id: 1, b.id: 2}).with_sentences([s])
    assert syllabus.gaps().unfilled_targets == ()


# --- candidate_targets: clauses 1-2 alone, apart from clause 3 -------------

def test_candidate_targets_returns_the_clause_1_and_2_targets_in_id_order():
    """candidate_targets (spec 1 section 3, clauses 1-2): every Target on
    a used word passes in a learner-voice sentence admitting the
    learner, in target-id order."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice -- last used word
    t_a_r = target("eat/receptive", "eat", "receptive")
    t_a_p = target("eat/productive", "eat", "productive")
    t_b_r = target("rice/receptive", "rice", "receptive")
    t_b_p = target("rice/productive", "rice", "productive")
    to = thai_of(a, b)
    s = sentence(((a.id, b.id),), to, voice="learner_voice")  # eat rice
    syllabus = base_syllabus((a, b), (t_a_r, t_a_p, t_b_r, t_b_p),
                             frequency={a.id: 1, b.id: 2})
    assert syllabus.candidate_targets(s) == (t_a_p, t_a_r, t_b_p, t_b_r)


def test_candidate_targets_is_empty_when_the_sentence_uses_no_targeted_word():
    """() when the sentence uses no targeted word at all (spec 1 section 3)."""
    untargeted = word("untargeted", "จาน")  # plate -- registered, no Target
    to = thai_of(untargeted)
    s = sentence(((untargeted.id,),), to, voice="learner_voice")  # plate
    syllabus = base_syllabus((untargeted,), ())
    assert syllabus.candidate_targets(s) == ()


# --- clause 3: the sentence-level gate ----------------------------------

def test_a_sentence_using_only_targeted_words_fills_its_target():
    rice = word("rice", "ข้าว")  # rice
    eat = word("eat", "กิน")  # eat
    t_eat = target("eat/receptive", "eat", "receptive")
    t_rice = target("rice/receptive", "rice", "receptive")
    to = thai_of(rice, eat)
    s = sentence(((eat.id, rice.id),), to, voice="learner_voice")  # eat rice
    syllabus = base_syllabus((rice, eat), (t_eat, t_rice), frequency={eat.id: 1, rice.id: 2})
    assert syllabus.fills(s, t_rice) is True


def test_a_word_with_no_target_at_all_empties_the_fill_set():
    rice = word("rice", "ข้าว")  # rice
    untargeted = word("untargeted", "จาน")  # plate -- registered, no Target
    t_rice = target("rice/receptive", "rice", "receptive", introduction="picture_card")
    to = thai_of(rice, untargeted)
    s = sentence(((rice.id, untargeted.id),), to, voice="learner_voice")  # rice, plate
    syllabus = base_syllabus((rice, untargeted), (t_rice,))
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
    to = thai_of(rice, spoon)
    s = sentence(((rice.id, spoon.id),), to, voice="learner_voice")  # rice, spoon
    syllabus = base_syllabus((rice, spoon), (t_rice, t_spoon))
    assert syllabus.fill_set(s) == ()
    assert syllabus.fills(s, t_rice) is False
    assert syllabus.fills(s, t_spoon) is False


def test_two_unmet_sentence_introduced_productive_targets_off_the_last_word_empty_the_fill_set():
    """The novelty rule (spec 1 section 3, clause 3) counts every
    candidate: with productive Targets filled off the last used word
    (r25), two unmet sentence-introduced ones -- on eat and rice, neither
    the last used word (fish is) -- still empty the fill set."""
    eat = word("eat", "กิน")  # eat -- sentence-introduced
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    fish = word("fish", "ปลา")  # fish -- last used word
    t_eat = target("eat/productive", "eat", "productive", introduction="sentence")
    t_rice = target("rice/productive", "rice", "productive", introduction="sentence")
    t_fish = target("fish/receptive", "fish", "receptive")
    to = thai_of(eat, rice, fish)
    s = sentence(((eat.id, rice.id, fish.id),), to,
                 voice="learner_voice")  # eat rice (and) fish
    syllabus = base_syllabus((eat, rice, fish), (t_eat, t_rice, t_fish),
                             frequency={eat.id: 1, rice.id: 2, fish.id: 3})
    assert syllabus.last_used_word(s) == fish.id   # the fixture's own premise
    assert syllabus.fill_set(s) == ()


def test_one_unmet_sentence_introduced_target_fills_all_its_candidates():
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- picture_card, not counted against the novelty rule
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive")
    to = thai_of(rice, spoon)
    s = sentence(((rice.id, spoon.id),), to, voice="learner_voice")  # rice, spoon
    syllabus = base_syllabus((rice, spoon), (t_rice, t_spoon))
    assert syllabus.fill_set(s) == (t_rice, t_spoon)


def test_a_gate_failing_adopted_sentence_does_not_meet_a_glue_target():
    """"Met" means the target is IN the other adopted sentence's OWN
    fill set, not merely clauses 1 and 2 (spec 1 section 3): an adopted
    sentence naming "rice" as an element, in the right voice, but whose
    own fill set is empty (an untargeted word) does not meet rice's glue
    Target for a later sentence -- both of the later sentence's
    candidates stay unmet and its own fill set empties too.
    """
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    tasty = word("tasty", "อร่อย")  # tasty -- registered, no Target
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    to = thai_of(rice, spoon, tasty)
    gate_failing = sentence(((rice.id, tasty.id),), to,
                            voice="learner_voice")  # rice, tasty (untargeted) -- adopted
    s = sentence(((rice.id, spoon.id),), to,
                voice="learner_voice")  # rice, spoon -- the candidate, placed after
    syllabus = Syllabus(words=(rice, spoon, tasty), targets=(t_rice, t_spoon),
                        sentences=(gate_failing,), profile=Profile(register="male_colloquial"),
                        frequency={rice.id: 1, spoon.id: 2})
    assert syllabus.fill_set(gate_failing) == ()   # "tasty" is untargeted
    assert syllabus.fill_set(s) == ()


def test_an_earlier_adopted_sentence_meeting_one_target_lets_the_other_fill():
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    to = thai_of(rice, spoon)
    earlier = sentence(((rice.id,),), to, voice="learner_voice")  # rice -- adopted, names only rice
    s = sentence(((rice.id, spoon.id),), to, voice="learner_voice")  # rice, spoon -- the candidate
    syllabus = Syllabus(words=(rice, spoon), targets=(t_rice, t_spoon),
                        sentences=(earlier,), profile=Profile(register="male_colloquial"),
                        frequency={rice.id: 1, spoon.id: 2})
    assert syllabus.fill_set(s) == (t_rice, t_spoon)


def test_a_later_adopted_sentence_does_not_count_as_meeting():
    """`later` names spoon's word, but its own order() position (set by
    "bowl", the word it also uses) is placed after the candidate's -- it
    does not count, so both candidates stay unmet and the fill set
    empties."""
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    bowl = word("bowl", "ชาม")  # bowl -- gives `later` a later order() position
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    t_bowl = target("bowl/receptive", "bowl", "receptive")
    to = thai_of(rice, spoon, bowl)
    later = sentence(((spoon.id, bowl.id),), to,
                     voice="learner_voice")  # spoon, bowl -- adopted, placed AFTER s
    s = sentence(((rice.id, spoon.id),), to, voice="learner_voice")  # rice, spoon -- the candidate
    syllabus = Syllabus(words=(rice, spoon, bowl), targets=(t_rice, t_spoon, t_bowl),
                        sentences=(later,), profile=Profile(register="male_colloquial"),
                        frequency={rice.id: 1, spoon.id: 2, bowl.id: 3})
    assert syllabus.fill_set(s) == ()


def test_an_adopted_sentence_is_not_its_own_meeting_sentence():
    """The candidate `s` is itself already adopted; "other adopted
    sentence" excludes it, so it cannot vacuously satisfy its own
    novelty rule."""
    rice = word("rice", "ข้าว")  # rice -- sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- sentence-introduced
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    to = thai_of(rice, spoon)
    s = sentence(((rice.id, spoon.id),), to, voice="learner_voice")  # rice, spoon -- adopted, uses both itself
    syllabus = Syllabus(words=(rice, spoon), targets=(t_rice, t_spoon), sentences=(s,),
                        profile=Profile(register="male_colloquial"),
                        frequency={rice.id: 1, spoon.id: 2})
    assert syllabus.fill_set(s) == ()


def test_vocabulary_met_by_includes_words_targeted_at_or_before():
    s = Syllabus(words=(word("a", "ก"), word("b", "ข"), word("c", "ค")),
                targets=(target("a/r", "a"), target("b/r", "b"), target("c/r", "c")),
                frequency={"a": 1, "b": 2, "c": 3})
    met = s.vocabulary_met_by(s.targets[1])
    assert [w.id for w in met] == ["a", "b"]
    assert len(s.with_sentences([]).sentences) == 0


def test_a_word_targeted_anywhere_in_the_order_still_lets_the_sentence_fill():
    """A used word's Target position does not matter to the sentence-level
    gate: it need only exist (the sentence's words a subset of
    _word_target_positions), whatever order() places it at. "กับ" ("with")
    is a registered glue Word with its own receptive Target -- glue words
    carry a Target like any other (spec 1 section 3)."""
    rice = word("rice", "ข้าว")     # rice
    spoon = word("spoon", "ช้อน")   # spoon -- a much later Target position than rice's
    with_word = word("with", "กับ")  # glue word, registered with its own Target
    t_rice = target("rice/receptive", "rice", "receptive", introduction="picture_card")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="picture_card")
    t_with = target("with/receptive", "with", "receptive")
    to = thai_of(rice, spoon, with_word)
    s = sentence(((rice.id, with_word.id, spoon.id),), to,
                voice="learner_voice")  # rice with a spoon
    syllabus = base_syllabus((rice, spoon, with_word), (t_rice, t_spoon, t_with),
                             frequency={rice.id: 1, with_word.id: 2, spoon.id: 99})
    assert syllabus.fills(s, t_rice) is True


# --- the per-word production cap (r26) ---------------------------------------

def _four_rice_sentences(studied=frozenset()):
    """rice (ข้าว) carries a receptive and a productive Target; four
    adopted learner-voice sentences use it, placed in this order by their
    last used words: ข้าว (rice), ข้าวไก่ (rice, chicken), ข้าวหมู (rice,
    pork), ข้าวปลา (rice, fish). Each of chicken/pork/fish carries a
    receptive Target only, so rice/productive is the one capped Target.
    """
    rice = word("rice", "ข้าว")      # rice
    chicken = word("chicken", "ไก่")  # chicken
    pork = word("pork", "หมู")        # pork
    fish = word("fish", "ปลา")        # fish
    t_rice_p = target("rice/productive", "rice", "productive")
    targets = (target("rice/receptive", "rice"), t_rice_p, target("chicken/receptive", "chicken"),
               target("pork/receptive", "pork"), target("fish/receptive", "fish"))
    to = thai_of(rice, chicken, pork, fish)
    s1 = sentence(((rice.id,),), to)                # rice
    s2 = sentence(((rice.id, chicken.id),), to)     # rice (with) chicken
    s3 = sentence(((rice.id, pork.id),), to)        # rice (with) pork
    s4 = sentence(((rice.id, fish.id),), to)        # rice (with) fish
    syllabus = Syllabus(words=(rice, chicken, pork, fish), targets=targets,
                        sentences=(s4, s3, s2, s1), profile=Profile(register="male_colloquial"),
                        frequency={rice.id: 1, chicken.id: 2, pork.id: 3, fish.id: 4},
                        **({"studied_cloze_pairs": frozenset(studied)} if studied else {}))
    return syllabus, t_rice_p, (s1, s2, s3, s4)


def test_a_productive_target_is_filled_by_the_first_three_sentences_in_placement_order():
    """Spec 1 r26: at most `production_sentences_per_word` (Profile, 3)
    sentences fill a productive Target, the first in placement order; the
    fourth keeps its receptive fills."""
    syllabus, t_rice_p, (s1, s2, s3, s4) = _four_rice_sentences()
    assert [syllabus.fills(s, t_rice_p) for s in (s1, s2, s3, s4)] == [True, True, True, False]
    assert syllabus.profile.production_sentences_per_word == 3
    assert syllabus.productive_fills(s4) == ()
    assert {t.id for t in syllabus.fill_set(s4)} == {"rice/receptive", "fish/receptive"}
    assert syllabus.target_words(s4) == ("fish", "rice")   # no productive fill: every fill's word


def test_a_studied_pair_keeps_filling_and_counts_toward_the_cap():
    """A (sentence, Target) pair with a study record on its Cloze card
    keeps filling whatever its position and takes one of the three: the
    latest-placed unstudied sentence (s3) gives way."""
    syllabus, t_rice_p, (s1, s2, s3, s4) = _four_rice_sentences()
    studied = {(s4.text_sha, t_rice_p.id)}
    syllabus, t_rice_p, (s1, s2, s3, s4) = _four_rice_sentences(studied)
    assert [syllabus.fills(s, t_rice_p) for s in (s1, s2, s3, s4)] == [True, True, False, True]


def test_the_cap_reads_the_profile():
    syllabus, t_rice_p, sentences = _four_rice_sentences()
    syllabus = dataclasses.replace(
        syllabus, profile=Profile(register="male_colloquial", production_sentences_per_word=2))
    assert [syllabus.fills(s, t_rice_p) for s in sentences] == [True, True, False, False]


def test_a_draft_fills_a_productive_target_only_below_the_cap_before_it():
    """A draft (not adopted) fills a productive Target only when fewer
    than the cap adopted sentences placed before it fill that Target."""
    syllabus, t_rice_p, (s1, s2, s3, s4) = _four_rice_sentences()
    syllabus = dataclasses.replace(syllabus, sentences=(s1, s2, s3))
    rice, chicken, fish = (syllabus.word(w) for w in ("rice", "chicken", "fish"))
    to = thai_of(*syllabus.words)
    after_all = sentence(((rice.id, fish.id),), to)                # rice (with) fish: after s3
    before_s3 = sentence(((rice.id, chicken.id), (rice.id,)), to)  # rice (with) chicken, rice:
                                                                   # chicken's group, 3 words
    assert syllabus._placement_key(s3) < syllabus._placement_key(after_all)
    assert syllabus._placement_key(s2) < syllabus._placement_key(before_s3) \
        < syllabus._placement_key(s3)
    assert syllabus.fills(after_all, t_rice_p) is False
    assert syllabus.fills(before_s3, t_rice_p) is True


def test_a_studied_pair_clause_3_refuses_holds_no_place_under_the_cap():
    """Three studied (sentence, rice/productive) pairs whose sentences
    fail clause 3's gate (salt carries no Target) fill nothing and hold
    none of the three: the one unstudied adopted sentence fills
    rice/productive, and a draft predicted to fill it does once adopted."""
    rice = word("rice", "ข้าว")      # rice
    chicken = word("chicken", "ไก่")  # chicken
    salt = word("salt", "เกลือ")      # salt -- no Target
    t_rice_p = target("rice/productive", "rice", "productive")
    to = thai_of(rice, chicken, salt)
    gated = (sentence(((rice.id, salt.id),), to),             # rice (with) salt
             sentence(((salt.id, rice.id),), to),             # salt (with) rice
             sentence(((rice.id, chicken.id, salt.id),), to))  # rice, chicken, salt
    unstudied = sentence(((rice.id,),), to)                    # rice
    draft = sentence(((rice.id, chicken.id),), to)             # rice (with) chicken
    syllabus = Syllabus(
        words=(rice, chicken, salt),
        targets=(target("rice/receptive", "rice"), t_rice_p, target("chicken/receptive", "chicken")),
        sentences=gated + (unstudied,), profile=Profile(register="male_colloquial"),
        frequency={rice.id: 1, chicken.id: 2},
        studied_cloze_pairs=frozenset((s.text_sha, t_rice_p.id) for s in gated))
    assert [syllabus.fill_set(s) for s in gated] == [(), (), ()]
    assert syllabus.fills(unstudied, t_rice_p) is True
    assert syllabus.fills(draft, t_rice_p) is True
    adopted = dataclasses.replace(syllabus, sentences=syllabus.sentences + (draft,))
    assert adopted.fills(draft, t_rice_p) is True


def test_more_studied_pairs_than_the_cap_all_keep_filling_and_no_unstudied_sentence_fills():
    """Four studied pairs on rice/productive at cap 3: all four keep
    filling; the earlier-placed unstudied sentence does not, and a draft
    placed before the studied ones is not predicted to fill."""
    rice = word("rice", "ข้าว")      # rice
    chicken = word("chicken", "ไก่")  # chicken
    pork = word("pork", "หมู")        # pork
    fish = word("fish", "ปลา")        # fish
    egg = word("egg", "ไข่")          # egg
    duck = word("duck", "เป็ด")       # duck
    t_rice_p = target("rice/productive", "rice", "productive")
    to = thai_of(rice, chicken, pork, fish, egg, duck)
    unstudied = sentence(((rice.id,),), to)  # rice
    draft = sentence(((rice.id, chicken.id),), to)  # rice (with) chicken
    studied = tuple(sentence(((rice.id, w.id),), to) for w in (pork, fish, egg, duck))
    words = (rice, chicken, pork, fish, egg, duck)
    syllabus = Syllabus(
        words=words,
        targets=(target("rice/receptive", "rice"), t_rice_p)
        + tuple(target(f"{w.id}/receptive", w.id) for w in words[1:]),
        sentences=(unstudied,) + studied, profile=Profile(register="male_colloquial"),
        frequency={w.id: rank for rank, w in enumerate(words, 1)},
        studied_cloze_pairs=frozenset((s.text_sha, t_rice_p.id) for s in studied))
    assert syllabus._placement_key(draft) < min(syllabus._placement_key(s) for s in studied)
    assert [syllabus.fills(s, t_rice_p) for s in studied] == [True] * 4
    assert syllabus.fills(unstudied, t_rice_p) is False
    assert syllabus.fills(draft, t_rice_p) is False


def test_a_draft_is_predicted_to_fill_only_as_the_fold_would_once_adopted():
    """Cap 3: A (s1) unstudied and early, D and E (s3, s4) studied and
    late. A draft placed between A and D (s2) would be the fourth --
    two studied plus one unstudied before it -- so it is not predicted
    to fill rice/productive, and does not fill it once adopted."""
    _, t_rice_p, (s1, s2, s3, s4) = _four_rice_sentences()
    syllabus, t_rice_p, (s1, s2, s3, s4) = _four_rice_sentences(
        {(s3.text_sha, t_rice_p.id), (s4.text_sha, t_rice_p.id)})
    pending = dataclasses.replace(syllabus, sentences=(s1, s3, s4))
    assert pending._placement_key(s1) < pending._placement_key(s2) < pending._placement_key(s3)
    assert [pending.fills(s, t_rice_p) for s in (s1, s3, s4)] == [True, True, True]
    assert pending.fills(s2, t_rice_p) is False
    assert syllabus.fills(s2, t_rice_p) is False


def test_without_studied_pairs_the_fold_evaluates_clause_3_once_per_adopted_sentence(monkeypatch):
    """No studied pairs: the fold runs one clause-3 evaluation per adopted
    sentence and no other pass over the sentences."""
    syllabus, t_rice_p, sentences = _four_rice_sentences()
    calls = {"clause_3": 0, "candidates": 0}
    clause_3, candidates = Syllabus._compute_fill_set, Syllabus.candidate_targets

    def counting_clause_3(self, *args):
        calls["clause_3"] += 1
        return clause_3(self, *args)

    def counting_candidates(self, *args):
        calls["candidates"] += 1
        return candidates(self, *args)

    monkeypatch.setattr(Syllabus, "_compute_fill_set", counting_clause_3)
    monkeypatch.setattr(Syllabus, "candidate_targets", counting_candidates)
    assert [syllabus.fills(s, t_rice_p) for s in sentences] == [True, True, True, False]
    assert calls == {"clause_3": len(sentences), "candidates": 0}


# --- target_words: the words of the Targets a sentence fills --------------

def test_target_words_are_the_words_of_its_productive_fills_in_target_id_order():
    """Syllabus.target_words (spec 3 r54): the word of each productive
    Target in the fill set, once each, in target-id order -- eat and
    rice, both filled productively, eat first though rice is the last
    used word and leads the sentence; i, filled only receptively, is
    not named."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice -- last used word
    i_word = word("i", "ผม")  # I -- receptive only
    to = thai_of(a, b, i_word)
    s = sentence(((b.id, i_word.id, a.id),), to, voice="learner_voice")  # rice, I eat
    syllabus = base_syllabus(
        (a, b, i_word),
        (target("rice/productive", "rice", "productive"), target("rice/receptive", "rice"),
         target("i/receptive", "i"), target("eat/receptive", "eat"),
         target("eat/productive", "eat", "productive")),
        frequency={a.id: 1, i_word.id: 1, b.id: 2})
    assert syllabus.target_words(s) == (a.id, b.id)


def test_target_words_fall_back_to_every_filled_targets_word_with_no_productive_fill():
    """An other-voice sentence fills no productive Target: its target
    words are the words of every Target it fills, in target-id order."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice
    to = thai_of(a, b)
    s = sentence(((b.id, a.id),), to, voice="other_voice")  # rice, eat
    syllabus = base_syllabus(
        (a, b),
        (target("rice/receptive", "rice"), target("eat/receptive", "eat"),
         target("eat/productive", "eat", "productive")),
        frequency={a.id: 1, b.id: 2})
    assert syllabus.target_words(s) == (a.id, b.id)


def test_target_words_are_empty_when_the_sentence_fills_nothing():
    rice = word("rice", "ข้าว")  # rice
    untargeted = word("untargeted", "จาน")  # plate -- registered, no Target
    to = thai_of(rice, untargeted)
    s = sentence(((rice.id, untargeted.id),), to, voice="learner_voice")  # rice, plate
    syllabus = base_syllabus((rice, untargeted), (target("rice/receptive", "rice"),))
    assert syllabus.target_words(s) == ()


# --- check_sentence: the vocabulary-dependent half of the Sentence invariant

def test_check_sentence_refuses_an_element_naming_an_unregistered_word():
    rice = word("rice", "ข้าว")  # rice
    syllabus = base_syllabus((rice,), (target("rice/receptive", "rice"),))
    ghost = Sentence(clauses=((WordId("ghost"),),), text="ผี",  # ghost -- never registered
                     gloss="ghost", voice="learner_voice", provenance=PROV)
    with pytest.raises(ValueError, match=f"{ghost.text_sha}.*ghost"):
        syllabus.check_sentence(ghost)


def test_check_sentence_refuses_a_text_that_does_not_match_its_clauses_rendering():
    rice = word("rice", "ข้าว")  # rice
    syllabus = base_syllabus((rice,), (target("rice/receptive", "rice"),))
    mismatched = Sentence(clauses=((rice.id,),), text="ข้าวข้าว",  # doubled, does not match
                          gloss="rice", voice="learner_voice", provenance=PROV)
    with pytest.raises(ValueError, match=f"{mismatched.text_sha}.*ข้าวข้าว.*ข้าว"):
        syllabus.check_sentence(mismatched)


def test_with_sentences_calls_check_sentence_and_refuses_a_bad_sentence():
    rice = word("rice", "ข้าว")  # rice
    syllabus = base_syllabus((rice,), (target("rice/receptive", "rice"),))
    ghost = Sentence(clauses=((WordId("ghost"),),), text="ผี",  # ghost -- never registered
                     gloss="ghost", voice="learner_voice", provenance=PROV)
    with pytest.raises(ValueError):
        syllabus.with_sentences([ghost])
