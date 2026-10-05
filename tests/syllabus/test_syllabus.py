"""Tests for syllabus.py's study_by_confusion (spec 1 section 3, spec 2
section 3): grouping StudyReader rows by confusion through the
aggregate's own pairs, not a store-owned map.
"""
import dataclasses

import pytest

from thai_syllabus.entities import Category, Grapheme, MinimalPair, Pronunciation, SoundConfusion
from thai_syllabus.ids import CategoryName, ConfusionId, PairId, WordId
from thai_syllabus.ports import StudyRecord
from thai_syllabus.store import SyllabusDb
from thai_syllabus.rulebook import RULES
from thai_syllabus.syllabus import Syllabus, derive_productive_targets

from .builders import sentence, syl, target, thai_of, word
from .fakes import FakeMediaIndex


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


# --- segmental_confusions_of: the confusions a word's reading waits on -----

_D_T = SoundConfusion(id=ConfusionId("consonant:d-t"), dimension="consonant", sounds=("d", "t"))
_T_K = SoundConfusion(id=ConfusionId("final:place-t-k"), dimension="final", sounds=("t", "k"))
_VELAR = SoundConfusion(id=ConfusionId("aspiration:velar"), dimension="aspiration",
                        sounds=("k", "kʰ"))
_E_AE = SoundConfusion(id=ConfusionId("vowel_quality:e-ɛ"), dimension="vowel_quality",
                       sounds=("e", "ɛ"))
_LOW_FALL = SoundConfusion(id=ConfusionId("tone:low-falling"), dimension="tone",
                           sounds=("low", "falling"))
_LENGTH = SoundConfusion(id=ConfusionId("vowel_length:short-long"), dimension="length",
                         sounds=("short", "long"))

_DII = word("dii", "ดี", syllables=(syl("d", "iː", "", "long", "mid"),))       # ดี: good
_TII = word("tii", "ตี", syllables=(syl("t", "iː", "", "long", "mid"),))       # ตี: hit
_TAAK = word("taak", "ตาก", syllables=(syl("t", "aː", "k", "long", "low"),))    # ตาก: dry in the sun
_KAT = word("kat", "กัด", syllables=(syl("k", "a", "t", "short", "low"),))      # กัด: bite
_KHAT = word("khat", "ขัด", syllables=(syl("kʰ", "a", "t", "short", "low"),))   # ขัด: scrub
_PEN = word("pen", "เป็น", syllables=(syl("p", "e", "n", "short", "mid"),))     # เป็น: to be
_TRAA = word("traa", "ตรา", syllables=(syl("tr", "aː", "", "long", "mid"),))    # ตรา: seal, brand
_MAA = word("maa", "มา", syllables=(syl("m", "aː", "", "long", "mid"),))        # มา: come


def _staged_syllabus(*extra_words) -> Syllabus:
    """d/t, t/k, k/kʰ, tone and length trained by pairs; e/ɛ curated with
    no pair in the deck."""
    pairs = (MinimalPair.create(id=PairId("consonant:d-t/dii"), confusion=_D_T,
                                members=(_DII, _TII)),
             MinimalPair.create(id=PairId("aspiration:velar/kat"), confusion=_VELAR,
                                members=(_KAT, _KHAT)),
             MinimalPair(id=PairId("final:place-t-k/x"), confusion=_T_K.id,
                         members=("kat", "taak")),
             MinimalPair(id=PairId("tone:low-falling/x"), confusion=_LOW_FALL.id,
                         members=("kat", "maa")),
             MinimalPair(id=PairId("vowel_length:short-long/x"), confusion=_LENGTH.id,
                         members=("kat", "taak")))
    return Syllabus(words=(_DII, _TII, _TAAK, _KAT, _KHAT, _PEN, _TRAA, _MAA, *extra_words),
                    pairs=pairs, confusions=(_D_T, _T_K, _VELAR, _E_AE, _LOW_FALL, _LENGTH))


def test_a_words_segmental_confusions_are_the_trained_ones_its_sounds_touch():
    syllabus = _staged_syllabus()
    assert syllabus.segmental_confusions_of("taak") == {"consonant:d-t", "final:place-t-k"}
    assert syllabus.segmental_confusions_of("kat") == {"aspiration:velar", "final:place-t-k"}


def test_tone_and_vowel_length_are_not_segmental_confusions():
    # มา (come) touches only the length and tone pairs' sounds
    assert _staged_syllabus().segmental_confusions_of("maa") == frozenset()


def test_a_confusion_with_no_pair_is_not_among_a_words_segmental_confusions():
    # เป็น (to be) carries e, a sound of the untrained e/ɛ confusion
    assert _staged_syllabus().segmental_confusions_of("pen") == frozenset()


def test_a_cluster_onset_touches_the_segmental_confusions_of_its_head():
    # ตรา (seal): onset tr's head is t
    assert _staged_syllabus().segmental_confusions_of("traa") == {"consonant:d-t"}


def test_a_word_with_no_pronunciation_touches_no_segmental_confusion():
    bare = dataclasses.replace(word("bare", "ดุ"),                          # ดุ: fierce
                               pron=Pronunciation(syllables=(), corroboration="disputed"))
    assert _staged_syllabus(bare).segmental_confusions_of("bare") == frozenset()


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


# --- marking() and check_sentence's speaker-marking refusal (spec 1
# section 1, r10) --------------------------------------------------------

def test_marking_of_a_sentence_using_male_marked_words_is_male():
    khrap = word("khrap", "ครับ", "male politeness particle", speaker="male")
    phom = word("phom", "ผม", "I (male speaker)", speaker="male")
    to = thai_of(khrap, phom)
    s = sentence(((phom.id, khrap.id),), to)
    syllabus = Syllabus(words=(khrap, phom))
    assert syllabus.marking(s) == frozenset({"male"})


def test_marking_of_a_sentence_with_no_speaker_marked_words_is_empty():
    rice = word("rice", "ข้าว")
    s = sentence(((rice.id,),), thai_of(rice))
    syllabus = Syllabus(words=(rice,))
    assert syllabus.marking(s) == frozenset()


def test_check_sentence_refuses_a_sentence_marking_both_a_male_and_a_female_speaker():
    khrap = word("khrap", "ครับ", "male politeness particle", speaker="male")
    kha = word("kha", "ค่ะ", "female politeness particle", speaker="female")
    to = thai_of(khrap, kha)
    mixed = sentence(((khrap.id, kha.id),), to)
    syllabus = Syllabus(words=(khrap, kha),
                        targets=(target("khrap/receptive", "khrap"),
                                 target("kha/receptive", "kha")))
    with pytest.raises(ValueError, match=f"{mixed.text_sha}.*both a male and a female"):
        syllabus.check_sentence(mixed)


def test_derive_productive_targets_derives_for_a_categorized_word_at_the_cutoff():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [_food("rice")], {rice.id: 2000}, 2000,
                                        learner_speaker="male")
    assert [t.id for t in derived] == ["rice/productive"]
    d = derived[0]
    assert (d.word, d.skill, d.introduction) == (rice.id, "productive", "picture_card")


def test_derive_productive_targets_excludes_a_word_ranked_past_the_cutoff():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [_food("rice")], {rice.id: 2001}, 2000,
                                        learner_speaker="male")
    assert derived == ()


def test_derive_productive_targets_excludes_an_uncategorized_word():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [], {rice.id: 1}, 2000, learner_speaker="male")
    assert derived == ()


def test_derive_productive_targets_excludes_a_no_productive_word():
    rice = dataclasses.replace(word("rice", "ข้าว"), no_productive=True)
    derived = derive_productive_targets([rice], [], [_food("rice")], {rice.id: 1}, 2000,
                                        learner_speaker="male")
    assert derived == ()


def test_derive_productive_targets_excludes_an_unranked_word():
    rice = word("rice", "ข้าว")
    derived = derive_productive_targets([rice], [], [_food("rice")], {}, 2000,
                                        learner_speaker="male")
    assert derived == ()


def test_derive_productive_targets_keeps_a_listed_exception_below_cutoff():
    """Spec 1 r9: a word ranked below the cutoff (rank 5000 against a
    2000 cutoff, so NOT eligible on its own) is exactly the case
    targets.yaml's own "exception" wording describes -- a listed
    productive Target for a word that would not have earned a derived
    one. `derived == ()` alone proves only that the function adds no
    *second* row; it says nothing about whether the listed row itself
    still reaches Syllabus.targets. Combined the way wiring.py's
    load_syllabus actually combines them (`tuple(bundle.targets) +
    derive_productive_targets(...)`), the listed Target must survive
    into Syllabus.targets, and rice must carry exactly one
    "rice/productive" -- not zero (dropped) and not two (re-derived on
    top of it).
    """
    rice = word("rice", "ข้าว")
    listed = target("rice/productive", "rice", skill="productive")
    derived = derive_productive_targets(
        [rice], [listed], [_food("rice")], {rice.id: 5000}, 2000, learner_speaker="male")
    assert derived == ()

    syllabus = Syllabus(words=(rice,), targets=(listed,) + derived,
                        categories=(_food("rice"),))

    productive = [t for t in syllabus.targets if t.skill == "productive"]
    assert productive == [listed]


def test_derive_productive_targets_raises_on_a_listed_target_for_an_eligible_word():
    rice = word("rice", "ข้าว")
    listed = target("rice/productive", "rice", skill="productive")
    with pytest.raises(ValueError, match="rice/productive"):
        derive_productive_targets([rice], [listed], [_food("rice")], {rice.id: 10}, 2000,
                                  learner_speaker="male")


def test_derive_productive_targets_skips_a_name_word_whose_target_adoption_listed():
    """I4: the grapheme pass lists a recited-name Word's two Targets in
    targets.yaml (spec 1 r16), and `Letter names` is a category, so a name
    word whose Thai happens to carry a frequency rank at or above the
    cutoff would be a listed productive Target on an "eligible" word --
    the one case r9 raises on. The pass owns those rows, so the rule skips
    the word instead: the deck must load.
    """
    name = word("name-chicken", "กอ ไก่", "recited name of the letter ก")   # กอ ไก่
    letters = Category(name=CategoryName("Letter names"), members=frozenset({name.id}))
    listed = target("name-chicken/productive", "name-chicken", skill="productive")

    derived = derive_productive_targets([name], [listed], [letters], {name.id: 10}, 2000,
                                        name_word_ids=frozenset({name.id}), learner_speaker="male")

    assert derived == ()


def test_derive_productive_targets_still_derives_for_an_ordinary_word_beside_a_name_word():
    """I4 is a skip of the named words only -- every other eligible word
    still derives."""
    rice = word("rice", "ข้าว")
    name = word("name-chicken", "กอ ไก่", "recited name of the letter ก")   # กอ ไก่
    letters = Category(name=CategoryName("Letter names"), members=frozenset({name.id}))

    derived = derive_productive_targets(
        [rice, name], [target("name-chicken/productive", "name-chicken", skill="productive")],
        [_food("rice"), letters], {rice.id: 10, name.id: 10}, 2000,
        name_word_ids=frozenset({name.id}), learner_speaker="male")

    assert [t.id for t in derived] == ["rice/productive"]


# --- the speaker marking (spec 1 r27, principle E3) -----------------------

def _pronouns(*word_ids: str) -> Category:
    return Category(name=CategoryName("Pronouns"), members=frozenset(word_ids))


def test_derive_productive_targets_gives_a_female_marked_word_none_for_a_male_learner():
    dichan = word("dichan", "ดิฉัน", "I (female, polite)", speaker="female")   # ดิฉัน
    derived = derive_productive_targets([dichan], [], [_pronouns("dichan")],
                                        {dichan.id: 918}, 2000, learner_speaker="male")
    assert derived == ()


def test_derive_productive_targets_derives_for_a_female_marked_word_for_a_female_learner():
    dichan = word("dichan", "ดิฉัน", "I (female, polite)", speaker="female")   # ดิฉัน
    derived = derive_productive_targets([dichan], [], [_pronouns("dichan")],
                                        {dichan.id: 918}, 2000, learner_speaker="female")
    assert [t.id for t in derived] == ["dichan/productive"]


def test_derive_productive_targets_keeps_male_marked_and_unmarked_words_for_a_male_learner():
    phom = word("phom", "ผม", "I (male)", speaker="male")   # ผม
    rice = word("rice", "ข้าว")                               # ข้าว
    derived = derive_productive_targets(
        [phom, rice], [], [_pronouns("phom"), _food("rice")],
        {phom.id: 20, rice.id: 100}, 2000, learner_speaker="male")
    assert [t.id for t in derived] == ["phom/productive", "rice/productive"]


def test_derive_productive_targets_refuses_a_listed_productive_target_on_an_other_sex_word():
    dichan = word("dichan", "ดิฉัน", "I (female, polite)", speaker="female")   # ดิฉัน
    listed = target("dichan-says-i", "dichan", skill="productive")
    with pytest.raises(ValueError, match="dichan-says-i.*speaker"):
        derive_productive_targets([dichan], [listed], [], {}, 2000, learner_speaker="male")


def test_order_places_the_derived_productive_target_after_the_receptive_one():
    rice = word("rice", "ข้าว")
    receptive = target("rice/receptive", "rice")
    derived = derive_productive_targets(
        [rice], [receptive], [_food("rice")], {rice.id: 1}, 2000, learner_speaker="male")
    syllabus = Syllabus(words=(rice,), targets=(receptive,) + derived,
                        categories=(_food("rice"),))
    ids = [e.id for e in syllabus.order() if e.kind == "word_target"]
    assert ids.index("rice/receptive") < ids.index("rice/productive")


# --- spelling_group: the targeted Words of one written form (spec 1 r29) ---

def test_a_spelling_group_is_the_words_of_one_form_in_introduction_order():
    # หนัง: movie (colloquial) / leather; ดี: good, another form
    movie = word("movie", "หนัง", "movie")
    leather = word("leather", "หนัง", "leather")
    good = word("good", "ดี", "good")
    syllabus = Syllabus(words=(leather, good, movie),
                        targets=(target("leather/r", "leather"), target("good/r", "good"),
                                 target("movie/r", "movie")),
                        frequency={movie.id: 1, good.id: 2, leather.id: 3})
    assert syllabus.spelling_group("leather") == (movie, leather)
    assert syllabus.spelling_group("movie") == (movie, leather)


def test_a_word_alone_in_its_form_is_a_group_of_itself():
    rice = word("rice", "ข้าว", "rice")      # ข้าว: rice
    news = word("news", "ข่าว", "news")      # ข่าว: news, a different form
    syllabus = Syllabus(words=(rice, news),
                        targets=(target("rice/r", "rice"), target("news/r", "news")))
    assert syllabus.spelling_group("rice") == (rice,)


def test_a_word_with_no_target_is_left_out_of_its_forms_group():
    # หลัง: back (of the body) / the classifier หลัง, which no Target names
    back = word("back", "หลัง", "back")
    classifier = word("classifier:หลัง", "หลัง", "(classifier)")
    syllabus = Syllabus(words=(classifier, back), targets=(target("back/r", "back"),))
    assert syllabus.spelling_group("back") == (back,)
    assert syllabus.spelling_group("classifier:หลัง") == (back,)


def test_a_sentence_introduced_word_is_a_member_of_its_forms_group():
    # ที่: serving (counted order) / "at", met only through sentences
    serving = word("serving", "ที่", "serving")
    at = word("at", "ที่", "at")
    syllabus = Syllabus(words=(serving, at),
                        targets=(target("serving/r", "serving"),
                                 target("at/r", "at", introduction="sentence")),
                        frequency={at.id: 1, serving.id: 2})
    assert syllabus.spelling_group("serving") == (at, serving)


# --- name_word_ids: the Words that are a grapheme's recited name ----------

def test_name_word_ids_names_every_graphemes_name_word():
    """Spec 1 r16: a name word's Targets are placed by order() inside the
    sounds block, met by the chart cell rather than by a sentence, and
    served by the glyph source alone -- three folds over one set."""
    chicken = word("chicken", "ไก่", "chicken")            # ไก่: chicken
    name = word("name-chicken", "กอ ไก่", "the letter ก's recited name")  # กอ ไก่
    egg = word("egg", "ไข่", "egg")                        # ไข่: egg
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    syllabus = Syllabus(words=(chicken, name, egg), graphemes=(g,))
    assert syllabus.name_word_ids == frozenset({"name-chicken"})


def test_a_grapheme_with_no_name_word_contributes_none():
    chicken = word("chicken", "ไก่", "chicken")            # ไก่: chicken
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken)
    assert Syllabus(words=(chicken,), graphemes=(g,)).name_word_ids == frozenset()


# --- with_adoptions: the aggregate the grapheme pass leaves behind --------

def test_with_adoptions_replaces_the_four_curated_collections():
    """Spec 3 r40 §5: the run's grapheme pass writes words.yaml,
    targets.yaml and graphemes.yaml, then reads its own result back
    through this -- a fresh instance, so every cached_property is
    recomputed rather than carried over stale (as with_words is)."""
    rice = word("rice", "ข้าว", "rice")                    # ข้าว: rice
    before = Syllabus(words=(rice,), targets=(target("rice/receptive", "rice"),))
    chicken = word("chicken", "ไก่", "chicken")            # ไก่: chicken
    name = word("name-chicken", "กอ ไก่", "the letter ก's recited name")  # กอ ไก่
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    names = Category(name="Letter names", members=frozenset({"name-chicken"}))

    after = before.with_adoptions(
        words=(rice, chicken, name),
        targets=(target("rice/receptive", "rice"),
                 target("name-chicken/receptive", "name-chicken")),
        graphemes=(g,), categories=(names,))

    assert after is not before
    assert [w.id for w in after.words] == ["rice", "chicken", "name-chicken"]
    assert [t.id for t in after.targets] == ["rice/receptive", "name-chicken/receptive"]
    assert after.graphemes == (g,)
    assert after.category_of("name-chicken") == "Letter names"
    assert after.name_word_ids == frozenset({"name-chicken"})
    assert after.find_word("chicken") is chicken       # the word index is rebuilt
    assert before.find_word("chicken") is None         # the old instance is untouched


# --- a Target that wants several sentences (spec 1 r30) ----------------------

_NOT = word("not", "ไม่", "not")       # ไม่: not
_EAT = word("eat", "กิน", "eat")       # กิน: eat
_RICE = word("rice", "ข้าว", "rice")   # ข้าว: rice
_TO = thai_of(_NOT, _EAT, _RICE)
_NOT_EAT = sentence(((_NOT.id, _EAT.id),), _TO)                          # ไม่กิน: (I) don't eat
_NOT_EAT_RICE = sentence(((_NOT.id, _EAT.id, _RICE.id),), _TO)           # ไม่กินข้าว: don't eat rice
_RICE_NOT_EAT = sentence(((_RICE.id,), (_NOT.id, _EAT.id)), _TO)         # ข้าว ไม่กิน: rice, I don't eat
_NOT_EAT_THEN_RICE = sentence(((_NOT.id, _EAT.id), (_RICE.id,)), _TO)    # ไม่กิน ข้าว: don't eat, rice


def _wanting(n: int, adopted=()) -> Syllabus:
    """ไม่ (not), sentence-introduced, wanting `n` sentences, before two
    picture words; `adopted` the sentences on record."""
    return Syllabus(words=(_NOT, _EAT, _RICE),
                    targets=(target("not/receptive", "not", introduction="sentence",
                                    sentences=n),
                             target("eat/receptive", "eat"), target("rice/receptive", "rice")),
                    sentences=tuple(adopted), frequency={"not": 1, "eat": 2, "rice": 3})


def _sentence_required(syllabus: Syllabus, rule: str = "target/sentence-required"
                       ) -> dict[str, str]:
    return {f.note_id: f.evidence for f in syllabus.report().findings if f.rule == rule}


def test_a_target_wanting_three_sentences_is_open_while_two_fill_it():
    """Fix round 1 ruling: a Target with a sentence and short of its count
    is a warn-severity `target/sentences-wanted` finding, not the
    gate-closing `target/sentence-required`; it is still open."""
    syllabus = _wanting(3, (_NOT_EAT, _NOT_EAT_RICE))
    assert sum(syllabus.targets[0] in syllabus.fill_set(s) for s in syllabus.sentences) == 2
    assert _sentence_required(syllabus) == {}
    assert _sentence_required(syllabus, "target/sentences-wanted") == {
        "not/receptive": "2 of 3 adopted sentences fill it"}
    assert "not/receptive" in syllabus.gaps().unfilled_targets


def test_a_target_wanting_three_sentences_with_none_is_sentence_required():
    syllabus = _wanting(3)
    assert _sentence_required(syllabus) == {"not/receptive": "no adopted sentence fills it",
                                            "eat/receptive": "no adopted sentence fills it",
                                            "rice/receptive": "no adopted sentence fills it"}
    assert _sentence_required(syllabus, "target/sentences-wanted") == {}


def test_a_deck_short_only_of_wanted_sentences_has_an_open_gate():
    rules = tuple(r for r in RULES if r.id.startswith("target/sentence"))
    syllabus = dataclasses.replace(_wanting(3, (_NOT_EAT, _NOT_EAT_RICE)), rules=rules)
    assert [r.severity for r in rules if r.id == "target/sentences-wanted"] == ["warn"]
    assert syllabus.report().gate is True
    assert dataclasses.replace(_wanting(3), rules=rules).report().gate is False


def test_unfilled_targets_lists_both_kinds_of_open_target_in_target_order():
    """ไม่ (not) is short with one sentence; ข้าว (rice) has none: both are
    open, in the Syllabus's target order."""
    syllabus = _wanting(3, (_NOT_EAT,))
    assert syllabus.gaps().unfilled_targets == ("not/receptive", "rice/receptive")


def test_a_target_wanting_three_sentences_is_filled_by_the_third():
    syllabus = _wanting(3, (_NOT_EAT, _NOT_EAT_RICE, _RICE_NOT_EAT))
    assert _sentence_required(syllabus) == {}
    assert syllabus.gaps().unfilled_targets == ()


def test_a_target_wanting_one_sentence_reads_as_before():
    assert _sentence_required(_wanting(1))["not/receptive"] == "no adopted sentence fills it"
    assert "not/receptive" not in _sentence_required(_wanting(1, (_NOT_EAT,)))


def test_a_sentence_introduced_target_is_met_by_its_first_sentence_while_still_open():
    syllabus = _wanting(3, (_NOT_EAT,))
    assert "not/receptive" in syllabus.met_sentence_introduced_targets()
    assert "not/receptive" in syllabus.gaps().unfilled_targets


def test_cover_adopts_drafts_up_to_the_sentences_a_target_still_wants():
    """Spec 3 r61: adoption supplies the count -- three drafts fill ไม่
    (not), which wants two more, and the third gains nothing still
    wanted."""
    syllabus = _wanting(3, (_NOT_EAT,))
    drafts = [(s, syllabus.fill_set(s))
              for s in (_NOT_EAT_RICE, _RICE_NOT_EAT, _NOT_EAT_THEN_RICE)]
    chosen = syllabus.cover(drafts)
    assert len(chosen) == 2
    assert [tuple(t.id for t in gained) for _, gained in chosen] == [
        ("not/receptive", "rice/receptive"), ("not/receptive",)]


def test_gaps_lists_a_scene_picture_only_for_a_sentence_carrying_a_cloze_card():
    """Spec 3 r61: a scene picture is sourced only for a sentence with a
    productive fill (its Cloze card shows the picture)."""
    syllabus = Syllabus(words=(_EAT, _RICE),
                        targets=(target("eat/receptive", "eat"),
                                 target("eat/productive", "eat", "productive"),
                                 target("rice/receptive", "rice")),
                        sentences=(sentence(((_EAT.id, _RICE.id),), _TO),   # กินข้าว: eat rice
                                   sentence(((_RICE.id,),), _TO)),          # ข้าว: rice
                        frequency={"eat": 1, "rice": 2})
    eat_rice, rice = syllabus.sentences
    assert syllabus.productive_fills(eat_rice) and not syllabus.productive_fills(rice)
    assert syllabus.gaps().scene_pictures == (eat_rice.text_sha,)


def _eat_rice_slots(media=None) -> Syllabus:
    """กินข้าว (eat rice) fills eat/productive and rice/productive; ข้าว
    (rice) alone fills rice/productive."""
    return Syllabus(words=(_EAT, _RICE),
                    targets=(target("eat/receptive", "eat"),
                             target("eat/productive", "eat", "productive"),
                             target("rice/receptive", "rice"),
                             target("rice/productive", "rice", "productive")),
                    sentences=(sentence(((_EAT.id, _RICE.id),), _TO),   # กินข้าว: eat rice
                               sentence(((_RICE.id, (_RICE.id, "ๆ")),), _TO)),  # ข้าวข้าวๆ
                    frequency={"eat": 1, "rice": 2},
                    **({"media": media} if media is not None else {}))


def test_a_filled_cloze_slot_per_productive_fill_keyed_as_its_cards_anchor():
    """Spec 4 section 1, spec 3 r65: one slot per productive Target a
    sentence fills, keyed TEXT_SHA:TARGET_ID (the Cloze card's anchor)."""
    syllabus = _eat_rice_slots()
    eat_rice, rice_rice = syllabus.sentences
    assert [(slot.sentence, slot.target.id) for slot in syllabus.cloze_slots] == [
        (eat_rice, "eat/productive"), (eat_rice, "rice/productive"),
        (rice_rice, "rice/productive")]
    assert [slot.key for slot in syllabus.cloze_slots] == [
        f"{eat_rice.text_sha}:eat/productive", f"{eat_rice.text_sha}:rice/productive",
        f"{rice_rice.text_sha}:rice/productive"]
    assert syllabus.cloze_slot(f"{rice_rice.text_sha}:rice/productive").breaks == 2
    with pytest.raises(KeyError, match="no filled Cloze slot"):
        syllabus.cloze_slot(f"{rice_rice.text_sha}:eat/productive")


def test_gaps_lists_every_filled_cloze_slot_lacking_a_gapped_recording():
    """Spec 1 r32: filled Cloze slots lacking a gapped recording are a gap,
    once their sentence has a recording (spec 3 r65)."""
    eat_rice = sentence(((_EAT.id, _RICE.id),), _TO)
    rice_rice = sentence(((_RICE.id, (_RICE.id, "ๆ")),), _TO)
    has_one = f"{eat_rice.text_sha}:eat/productive"
    syllabus = _eat_rice_slots(FakeMediaIndex(recording_provenance={
        has_one: {"source": "tts"}, eat_rice.text_sha: {"source": "tts"},
        rice_rice.text_sha: {"source": "forvo"}}))
    assert syllabus.gaps().gapped_recordings == tuple(
        slot.key for slot in syllabus.cloze_slots if slot.key != has_one)
    assert len(syllabus.gaps().gapped_recordings) == 2


def test_a_slot_whose_sentence_has_no_recording_has_no_gapped_need_yet():
    """Spec 3 r65: the gapped clip takes its voice from the sentence's
    recording, so its need waits for one."""
    eat_rice = sentence(((_EAT.id, _RICE.id),), _TO)
    syllabus = _eat_rice_slots(FakeMediaIndex(recording_provenance={
        eat_rice.text_sha: {"source": "tts"}}))
    assert syllabus.gaps().gapped_recordings == (
        f"{eat_rice.text_sha}:eat/productive", f"{eat_rice.text_sha}:rice/productive")


def test_a_target_wanting_one_sentence_leaves_the_state_id_as_it_was():
    """`sentences` at its default contributes nothing to the syllabus
    state id, so a deck that sets it nowhere keeps its identity (the
    literal is the id this fixture had before `sentences` existed); a
    Target wanting more is a different state."""
    rice = word("rice", "ข้าว")   # ข้าว: rice
    one = Syllabus(words=(rice,), targets=(target("rice/receptive", "rice",
                                                  introduction="sentence"),))
    assert one.state_id() == "f06d1777ad98c0c0d33080ba54422e3581c2c1fa668c51a5a1643787295d4ce5"
    two = dataclasses.replace(one, targets=(target("rice/receptive", "rice",
                                                   introduction="sentence", sentences=2),))
    assert two.state_id() != one.state_id()
