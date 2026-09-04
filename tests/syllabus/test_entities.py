"""Entity construction and invariants (spec 1, section 1).

Thai strings always carry an English gloss alongside them (project rule).
"""
import dataclasses
from datetime import date

import pytest

from thai_syllabus.entities import (
    Category,
    Grapheme,
    MinimalPair,
    Pronunciation,
    SoundConfusion,
    Syllable,
    Target,
    Sentence,
    Word,
)
from thai_syllabus.media import Provenance
from thai_syllabus.ids import CategoryName, ConfusionId, PairId, TargetId, WordId


def syl(onset="m", vowel="a", coda="", length="short", tone="mid") -> Syllable:
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def pron(*syllables: Syllable, corroboration="engines_agree") -> Pronunciation:
    return Pronunciation(syllables=tuple(syllables), corroboration=corroboration)


PROV = Provenance(source="test", origin="fixture", licence="cc0",
                   acquired=date(2026, 1, 1))


# --- Word / Pronunciation / Syllable -------------------------------------

def test_word_is_frozen_and_holds_its_pronunciation():
    w = Word(id=WordId("rice"), thai="ข้าว", pron=pron(syl("kh", "aː", "w")),
              meaning="cooked rice", classifier=None)
    assert w.thai == "ข้าว"  # rice
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.meaning = "something else"


def test_word_classifier_references_another_word_id():
    plate = WordId("plate")
    w = Word(id=WordId("rice"), thai="ข้าว", pron=pron(syl("kh", "aː", "w")),
              meaning="cooked rice", classifier=plate)
    assert w.classifier == plate


# --- SoundConfusion --------------------------------------------------------

def test_sound_confusion_holds_the_two_opposed_values():
    c = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                        sounds=("mid", "low"))
    assert c.sounds == ("mid", "low")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.dimension = "length"


# --- Grapheme: keyword-containment invariant -------------------------------

def test_grapheme_create_accepts_a_keyword_word_whose_spelling_contains_the_symbol():
    keyword = Word(id=WordId("chicken"), thai="ไก่", pron=pron(syl("k", "ai", "")),
                    meaning="chicken", classifier=None)
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                        consonant_class="mid", keyword_word=keyword)
    assert g.keyword == keyword.id


def test_grapheme_create_rejects_a_keyword_word_missing_the_symbol():
    keyword = Word(id=WordId("dog"), thai="หมา", pron=pron(syl("m", "aː", "")),
                    meaning="dog", classifier=None)
    with pytest.raises(ValueError):
        Grapheme.create(symbol="ก", kind="consonant", sound="k",
                        consonant_class="mid", keyword_word=keyword)


def test_grapheme_name_word_defaults_to_none():
    keyword = Word(id=WordId("chicken"), thai="ไก่", pron=pron(syl("k", "ai", "")),
                    meaning="chicken", classifier=None)
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                        consonant_class="mid", keyword_word=keyword)
    assert g.name_word is None


def test_grapheme_create_accepts_an_explicit_name_word():
    keyword = Word(id=WordId("chicken"), thai="ไก่", pron=pron(syl("k", "ai", "")),
                    meaning="chicken", classifier=None)
    # กอ "gɔɔ" -- the recited name-syllable for the letter ก, distinct from
    # the keyword ไก่ "chicken".
    name_word = Word(id=WordId("letter-name:ko"), thai="กอ",
                     pron=pron(syl("k", "ɔː", "")), meaning="name of the letter ก",
                     classifier=None)
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                        consonant_class="mid", keyword_word=keyword,
                        name_word=name_word)
    assert g.name_word == name_word.id


def test_grapheme_is_still_constructible_directly_without_name_word():
    # Plain construction (bypassing create()) is what curated-data loading
    # does before re-validating -- name_word must default cleanly there too.
    g = Grapheme(symbol="ก", kind="consonant", sound="k",
                consonant_class="mid", keyword=WordId("chicken"))
    assert g.name_word is None


# --- Target -----------------------------------------------------------------

def test_target_defaults_to_picture_card_introduction():
    t = Target(id=TargetId("rice/receptive"), word=WordId("rice"),
               skill="receptive")
    assert t.introduction == "picture_card"


# --- MinimalPair: exact-confusion invariant --------------------------------

def test_minimal_pair_create_accepts_members_differing_in_exactly_the_confusion():
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = Word(id=WordId("near"), thai="ใกล้", pron=pron(syl("k", "ai", "", tone="mid")),
                    meaning="near", classifier=None)
    low_word = Word(id=WordId("far"), thai="ไกล", pron=pron(syl("k", "ai", "", tone="low")),
                    meaning="far", classifier=None)
    pair = MinimalPair.create(id=PairId("tone:mid-low/kai"), confusion=confusion,
                              members=(mid_word, low_word))
    assert pair.members == (mid_word.id, low_word.id)


def test_minimal_pair_create_rejects_members_differing_in_more_than_the_confusion():
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = Word(id=WordId("near"), thai="ใกล้", pron=pron(syl("k", "ai", "", tone="mid")),
                    meaning="near", classifier=None)
    # Differs in tone AND vowel -- not an exact confusion.
    other_vowel = Word(id=WordId("other"), thai="กลอ", pron=pron(syl("k", "ɔː", "", tone="low")),
                       meaning="(not a real word, for the test)", classifier=None)
    with pytest.raises(ValueError):
        MinimalPair.create(id=PairId("bad"), confusion=confusion,
                           members=(mid_word, other_vowel))


def test_minimal_pair_create_rejects_members_using_sounds_outside_the_confusion():
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = Word(id=WordId("near"), thai="ใกล้", pron=pron(syl("k", "ai", "", tone="mid")),
                    meaning="near", classifier=None)
    # Tone-only difference, but "rising" is not one of the confusion's sounds.
    rising_word = Word(id=WordId("rising"), thai="ไก่", pron=pron(syl("k", "ai", "", tone="rising")),
                       meaning="(not a real word, for the test)", classifier=None)
    with pytest.raises(ValueError):
        MinimalPair.create(id=PairId("bad"), confusion=confusion,
                           members=(mid_word, rising_word))


# --- Category -------------------------------------------------------------

def test_category_holds_a_name_and_its_member_word_ids():
    cat = Category(name=CategoryName("Food"), members=frozenset({WordId("rice"), WordId("fish")}))
    assert cat.name == "Food"
    assert cat.members == {"rice", "fish"}


def test_category_is_frozen():
    cat = Category(name=CategoryName("Food"), members=frozenset({WordId("rice")}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cat.name = CategoryName("Colors")


# --- Sentence -----------------------------------------------------------------

def test_sentence_identity_is_text_together_with_provenance():
    a = Sentence(text="ผมกินข้าว", voice="learner_voice", provenance=PROV)
    b = Sentence(text="ผมกินข้าว", voice="learner_voice", provenance=PROV)
    assert a == b  # "I eat rice" (learner voice), same provenance
