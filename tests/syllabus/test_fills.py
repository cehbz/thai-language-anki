"""Syllabus.fills(sentence, target) and Syllabus.fill_set(sentence): the one
definition of "this text serves that target" (spec 1, section 3): target.word
among the sentence's own words (a repeated element counted once), voice
satisfies skill, then a sentence-level gate (every used word carrying a
Target) and a novelty rule over the whole fill set -- at most one candidate
may be a sentence-introduced Target no other adopted sentence, placed at or
before this one, already contains.
"""
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
    """dog is also the sentence's last used word (frequency puts its
    Target after i's and have's) -- clause 2's last-word requirement
    (§3 clause 2, r10) holds here, distinct from the tests below that
    exercise it directly."""
    dog = word("dog", "หมา")  # dog -- last used word
    i_word = word("i", "ผม")  # registered with a Target so it doesn't empty the fill set
    have = word("have", "มี")  # registered with a Target so it doesn't empty the fill set
    t = target("dog/productive", "dog", "productive")
    to = thai_of(dog, i_word, have)
    s = sentence(((i_word.id, have.id, dog.id),), to, voice="learner_voice")  # I have a dog
    syllabus = base_syllabus((dog, i_word, have),
                             (t, target("i/receptive", "i"), target("have/receptive", "have")),
                             frequency={i_word.id: 1, have.id: 1, dog.id: 2})
    assert syllabus.last_used_word(s) == dog.id
    assert syllabus.fills(s, t) is True


# --- clause 2: the productive fill's last-word and marking conditions (r10) -

def test_a_productive_target_fills_only_from_the_sentences_last_used_word():
    """Spec 1 section 3, clause 2 (r10): a productive Target is filled
    only when the sentence's last used word is the target's own word --
    here rice (b), not eat (a); today (before this rule) both fill."""
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
    assert syllabus.last_used_word(s) == b.id
    assert syllabus.fills(s, t_b_p) is True
    assert syllabus.fills(s, t_a_p) is False


def test_marking_that_does_not_admit_the_learner_fills_no_productive_target():
    """Spec 1 section 3, clause 2 (r10): the sentence's marking must be
    empty or the Profile's own sex; ค่ะ marks a female speaker, which the
    male_colloquial profile's learner_speaker does not admit -- neither
    productive Target fills, but both receptive Targets still do (clause
    2's voice-satisfies-skill test alone, unaffected by marking)."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice -- last used word among a/b
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
    assert syllabus.last_used_word(s) == b.id
    assert syllabus.fills(s, t_a_p) is False
    assert syllabus.fills(s, t_b_p) is False
    assert syllabus.fills(s, t_a_r) is True
    assert syllabus.fills(s, t_b_r) is True


def test_marking_that_admits_the_learner_fills_the_last_words_productive_target():
    """ครับ marks a male speaker, the male_colloquial profile's own sex
    (r10): admitted, so the productive Target on the sentence's last
    used word still fills."""
    a = word("eat", "กิน")  # eat
    b = word("rice", "ข้าว")  # rice -- last used word among a/b
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
    assert syllabus.last_used_word(s) == b.id
    assert syllabus.fills(s, t_b_p) is True


def test_gaps_unfilled_targets_reflects_the_last_word_condition():
    """gaps().unfilled_targets (spec 1 section 3, clause 2/r10 via
    fill_set): eat's productive Target stays unfilled because eat is not
    the sentence's last used word, even though the sentence is adopted."""
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
    assert syllabus.gaps().unfilled_targets == ("eat/productive",)


# --- candidate_targets: clauses 1-2 alone, apart from clause 3 -------------

def test_candidate_targets_returns_the_clause_1_and_2_targets_in_id_order():
    """candidate_targets (spec 1 section 3, clauses 1-2): eat/productive
    fails clause 2 -- rice, not eat, is the sentence's last used word --
    so it is not a candidate; the rest pass, in target-id order."""
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
    assert syllabus.candidate_targets(s) == (t_a_r, t_b_p, t_b_r)


def test_candidate_targets_is_empty_when_the_sentence_uses_no_targeted_word():
    """() when the sentence uses no targeted word at all -- last_used_word
    would raise (spec 1 section 3)."""
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
