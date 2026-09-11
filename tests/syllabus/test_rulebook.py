"""The rulebook (spec 1, section 4), enumerated against the spec table:
every registered rule, the locked principles, and the traceability measure
derived from both.
"""
import re
from pathlib import Path

from thai_syllabus.curated import RulebookConfig
from thai_syllabus.entities import Category, Grapheme, MinimalPair, SoundConfusion, Word
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.media import Speaker
from thai_syllabus.profile import Profile
from thai_syllabus.rules import Rule
from thai_syllabus.rulebook import (ENFORCEMENT_PRINCIPLES, PICTURE_FIT, PICTURE_FIT_RUBRIC,
                                    PICTURE_PREFERENCE, PRINCIPLES, RULES, SCENE_FIT_RUBRIC,
                                    SENTENCE_FOR_TARGET_RUBRIC, SENTENCE_REGISTER_NATURAL,
                                    apply_overlay, rubrics_for, sentence_note_id,
                                    traceability_metric)
from thai_syllabus.syllabus import Syllabus

from .builders import pron, sentence, syl, target, thai_of, word
from .fakes import FakeAssessmentReader, FakeMediaIndex


def make_syllabus(**kwargs):
    kwargs.setdefault("profile", Profile(register="male_colloquial"))
    kwargs.setdefault("rules", RULES)
    return Syllabus(**kwargs)


def rule_ids(report):
    return {f.rule for f in report.findings}


# --- traceability -------------------------------------------------------

def test_traceability_metric_flags_a_rule_with_no_live_principle():
    orphan = Rule(id="orphan/rule", principle="Z9", severity="error",
                  shape="check", check=lambda s: [])
    metric = traceability_metric(rules=[orphan], known_principles={"F1"},
                                 enforcement_principles={"F1"})
    assert metric.detail["orphan_rules"] == ["orphan/rule"]
    assert metric.value == 0.0


def test_traceability_metric_flags_an_enforcement_principle_with_no_rule():
    rule = Rule(id="a/rule", principle="F1", severity="error", shape="check",
               check=lambda s: [])
    metric = traceability_metric(rules=[rule], known_principles={"F1", "F2"},
                                 enforcement_principles={"F1", "F2"})
    assert metric.detail["unenforced_principles"] == ["F2"]
    assert metric.value == 0.0


def test_traceability_metric_is_clean_for_a_well_formed_registry():
    rule = Rule(id="a/rule", principle="F1", severity="error", shape="check",
               check=lambda s: [])
    metric = traceability_metric(rules=[rule], known_principles={"F1"},
                                 enforcement_principles={"F1"})
    assert metric.detail == {"orphan_rules": [], "unenforced_principles": []}
    assert metric.value == 1.0


def test_the_shipped_rulebook_is_internally_traceable():
    metric = traceability_metric(rules=RULES, known_principles=PRINCIPLES,
                                 enforcement_principles=ENFORCEMENT_PRINCIPLES)
    assert metric.detail == {"orphan_rules": [], "unenforced_principles": []}


def test_traceability_reports_every_principle_without_a_rule():
    metric = traceability_metric([r for r in RULES if r.principle != "E7"], PRINCIPLES,
                                 ENFORCEMENT_PRINCIPLES)
    assert metric.value == 0.0
    assert metric.detail["unenforced_principles"] == ["E7"]


def test_rulebook_traceability_rule_is_itself_in_the_registry():
    assert any(r.id == "rulebook/traceability" for r in RULES)


# --- rule enumeration against the spec table --------------------------------

def test_registered_rules_match_the_spec_table():
    # order/sounds-first, order/reading-after-graphemes,
    # order/receptive-before-productive, order/sentence-after-words,
    # category/single-membership, syllabus/closure and
    # rendition/mixed-speakers are retired spec 1 section 4 r10: they held
    # by construction (order(), the loader's registration checks, the
    # rendition attempt's one-speaker intersection).
    expected = {
        "pair/exact-confusion", "pair/rendition-required", "rendition/synthetic",
        "coverage/confusions",
        "coverage/categories", "picture/fit",
        "picture/preference", "scene/fit", "target/picture-required",
        "sentence/fills-novelty",
        "target/sentence-required", "grapheme/keyword-picture-required",
        "grapheme/keyword-contains-symbol", "target/recording-required",
        "sentence/recording-required", "recording/synthetic",
        "sentence/synthetic-productive",
        "sentence/register-natural",
        "word/pronunciation-corroborated", "word/classifier-known",
        "coverage/speakers", "card/unique-front", "rulebook/traceability",
    }
    assert {r.id for r in RULES} == expected


def test_rules_holds_none_of_the_seven_ids_retired_by_construction():
    """Spec 1 section 4 r10: these seven hold by construction (order(),
    the loader's registration checks, the rendition attempt's one-speaker
    intersection) and carry no registered rule.
    """
    retired = {
        "order/sounds-first", "order/reading-after-graphemes",
        "order/receptive-before-productive", "order/sentence-after-words",
        "category/single-membership", "syllabus/closure",
        "rendition/mixed-speakers",
    }
    assert {r.id for r in RULES} & retired == set()


def test_principles_matches_the_locked_principles_doc():
    text = Path("docs/principles.md").read_text(encoding="utf-8")
    ids = set(re.findall(r"\*\*([AFE]\d+[ab]?)\.\*\*", text))
    ids.add("F6a")    # named only in prose ("(F6a: ...)"), never its own heading
    ids.add("META-1")  # the charter's traceability meta-rule; not numbered in the doc
    assert ids == PRINCIPLES


# --- pair/exact-confusion ----------------------------------------------------

def test_pair_exact_confusion_flags_a_pair_loaded_with_a_mismatched_dimension():
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    # Constructed with the WRONG tone (rising, not one of the confusion's
    # sounds) via the plain constructor, bypassing MinimalPair.create --
    # simulating data loaded straight off disk.
    other_word = word("other", "ไกล", syllables=(syl(tone="rising"),))  # far (test fixture)
    bad_pair = MinimalPair(id=PairId("bad"), confusion=confusion.id,
                           members=(mid_word.id, other_word.id))
    syllabus = make_syllabus(words=(mid_word, other_word), pairs=(bad_pair,),
                             confusions=(confusion,))
    findings = [f for f in syllabus.report().findings if f.rule == "pair/exact-confusion"]
    assert len(findings) == 1
    assert findings[0].note_id == bad_pair.id


def test_pair_exact_confusion_passes_a_well_formed_pair():
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    low_word = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    good_pair = MinimalPair.create(id=PairId("good"), confusion=confusion,
                                   members=(mid_word, low_word))
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(good_pair,),
                             confusions=(confusion,))
    findings = [f for f in syllabus.report().findings if f.rule == "pair/exact-confusion"]
    assert findings == []


# --- grapheme/keyword-contains-symbol ---------------------------------------

def test_grapheme_keyword_rule_flags_a_grapheme_loaded_with_a_bad_keyword():
    dog = word("dog", "หมา")  # dog, does not contain "ก"
    bad_grapheme = Grapheme(symbol="ก", kind="consonant", sound="k",
                            consonant_class="mid", keyword=dog.id)
    syllabus = make_syllabus(words=(dog,), graphemes=(bad_grapheme,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "grapheme/keyword-contains-symbol"]
    assert len(findings) == 1
    assert findings[0].note_id == "ก"  # the grapheme symbol


# --- sentence/fills-novelty --------------------------------------------------

def test_sentence_fills_novelty_flags_a_sentence_using_an_untargeted_word():
    rice = word("rice", "ข้าว")  # rice
    unmet1 = word("unmet1", "จาน")  # plate
    unmet2 = word("unmet2", "ช้อน")  # spoon
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    to = thai_of(rice, unmet1, unmet2)
    s = sentence(((rice.id, unmet1.id, unmet2.id),), to,
                voice="learner_voice")  # rice, plate, spoon
    syllabus = make_syllabus(words=(rice, unmet1, unmet2), targets=(t_rice,), sentences=(s,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/fills-novelty"]
    assert len(findings) == 1


def test_sentence_fills_novelty_flags_every_mentioned_target_with_two_unmet_glue_targets():
    """Two sentence-introduced (glue) Targets in one adopted sentence,
    neither met by any other adopted sentence: the fill set is empty, so
    every mentioned Target is flagged, not just one of them."""
    rice = word("rice", "ข้าว")  # rice -- glue, sentence-introduced
    spoon = word("spoon", "ช้อน")  # spoon -- glue, sentence-introduced
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    t_spoon = target("spoon/receptive", "spoon", "receptive", introduction="sentence")
    to = thai_of(rice, spoon)
    s = sentence(((rice.id, spoon.id),), to, voice="learner_voice")  # rice, spoon
    syllabus = make_syllabus(words=(rice, spoon), targets=(t_rice, t_spoon), sentences=(s,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/fills-novelty"]
    assert len(findings) == 2


def test_sentence_fills_novelty_does_not_flag_a_word_that_is_not_used():
    """A registered Target whose word is not among the sentence's own
    words is not a candidate at all: the loop that gathers candidate
    Targets (the sentence's own words) skips it, so it raises no finding even though
    its own Target has no adopted sentence to fill it.
    """
    very = word("very", "มาก")  # very -- the only word actually used
    come = word("come", "มา")   # come -- registered, but never used here
    t_very = target("very/receptive", "very", "receptive", introduction="sentence")
    t_come = target("come/receptive", "come", "receptive", introduction="sentence")
    to = thai_of(very, come)
    s = sentence(((very.id,),), to, voice="learner_voice")  # very
    syllabus = make_syllabus(words=(very, come), targets=(t_very, t_come), sentences=(s,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/fills-novelty"]
    assert findings == []


def test_sentence_fills_novelty_is_silent_when_the_sentence_fills_its_target():
    rice = word("rice", "ข้าว")  # rice
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    to = thai_of(rice)
    s = sentence(((rice.id,),), to, voice="learner_voice")  # rice
    syllabus = make_syllabus(words=(rice,), targets=(t_rice,), sentences=(s,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/fills-novelty"]
    assert findings == []


def test_sentence_fills_novelty_does_not_flag_a_productive_target_off_the_last_used_word():
    """Spec 1 section 3, clause 2 (r10): eat/productive is not a
    candidate at all (rice, not eat, is the sentence's last used word) --
    the rule flags a candidate_targets() member not in the fill set,
    never re-implementing clauses 1-2 to catch a Target clause 2 itself
    already excludes."""
    eat = word("eat", "กิน")  # eat
    rice = word("rice", "ข้าว")  # rice -- last used word
    t_eat_r = target("eat/receptive", "eat", "receptive")
    t_eat_p = target("eat/productive", "eat", "productive")
    t_rice_r = target("rice/receptive", "rice", "receptive")
    to = thai_of(eat, rice)
    s = sentence(((eat.id, rice.id),), to, voice="learner_voice")  # eat rice
    syllabus = make_syllabus(words=(eat, rice), targets=(t_eat_r, t_eat_p, t_rice_r),
                             sentences=(s,), frequency={eat.id: 1, rice.id: 2})
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/fills-novelty"]
    assert findings == []


# --- coverage/categories ---------------------------------------------------

def test_coverage_categories_counts_categories_with_a_target():
    rice = word("rice", "ข้าว")  # rice
    t = target("rice/receptive", "rice", "receptive")
    syllabus = make_syllabus(words=(rice,), targets=(t,), categories=(
        Category(name="Food", members=frozenset({"rice"})),
        Category(name="Colors", members=frozenset({"red"}))))
    metric = next(m for m in syllabus.report().metrics if m.rule == "coverage/categories")
    assert metric.value == 0.5
    assert metric.detail["covered"] == ["Food"]


def test_coverage_categories_is_full_with_no_categories_at_all():
    syllabus = make_syllabus()
    metric = next(m for m in syllabus.report().metrics if m.rule == "coverage/categories")
    assert metric.value == 1.0


# --- coverage/confusions ------------------------------------------------

def test_coverage_confusions_measures_pairs_and_speakers_per_confusion():
    from .fakes import FakeMediaIndex
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    low_word = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    pair = MinimalPair.create(id=PairId("p"), confusion=confusion,
                              members=(mid_word, low_word))
    media = FakeMediaIndex(rendition_speakers={confusion.id: frozenset({"speaker-a"})})
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,),
                             confusions=(confusion,), media=media)
    metrics = {m.rule: m for m in syllabus.report().metrics}
    detail = metrics["coverage/confusions"].detail[confusion.id]
    assert detail == {"pairs": 1, "speakers": 1, "covered": True}


# --- sentence/register-natural (judged) --------------------------------------

def _pom_gin_khaao():
    pom = word("pom", "ผม")  # I (male speaker)
    gin = word("gin", "กิน")  # eat
    rice = word("rice", "ข้าว")  # rice
    to = thai_of(pom, gin, rice)
    s = sentence(((pom.id, gin.id, rice.id),), to, voice="learner_voice")  # I eat rice
    return (pom, gin, rice), s


def test_sentence_register_rule_reads_a_cached_verdict_and_never_a_live_judge():
    words, s = _pom_gin_khaao()
    reader = FakeAssessmentReader()
    identity = None
    for r in RULES:
        if r.id == "sentence/register-natural":
            identity = r.judged_subjects(make_syllabus(words=words, sentences=(s,)))[0]
    assert identity is not None
    note_id, sha = identity
    reader = FakeAssessmentReader(verdicts={("sentence/register-natural", note_id, sha): False})
    syllabus = make_syllabus(words=words, sentences=(s,), assessments=reader)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/register-natural"]
    assert len(findings) == 1


def test_sentence_register_rule_is_silent_with_no_cached_verdict():
    words, s = _pom_gin_khaao()
    syllabus = make_syllabus(words=words, sentences=(s,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/register-natural"]
    assert findings == []


# --- rulebook overlay (spec 3) -----------------------------------------------

def test_overlay_changes_severity_and_rubric_only_where_configured():
    cfg = RulebookConfig(severities={"sentence/register-natural": "error"},
                         rubrics={"sentence/register-natural": "new text"})
    out = apply_overlay(RULES, cfg)
    by_id = {r.id: r for r in out}
    assert by_id["sentence/register-natural"].severity == "error"
    assert by_id["sentence/register-natural"].rubric == "new text"
    assert by_id["card/unique-front"].severity == "error"
    assert SENTENCE_REGISTER_NATURAL.severity == "warn"  # registry untouched


def test_overlay_ignores_a_rubric_override_targeting_a_non_judged_rule():
    cfg = RulebookConfig(rubrics={"card/unique-front": "should be ignored"})
    out = apply_overlay(RULES, cfg)
    by_id = {r.id: r for r in out}
    assert by_id["card/unique-front"].rubric is None


def test_rule_role_defaults_to_id():
    assert SENTENCE_REGISTER_NATURAL.role == "sentence/register-natural"


# --- completeness errors, synthetic/mixed-speaker warnings, picture fit -----

_SLOW = word("slow", "ช้า", "slow")


def _slow_sentence():
    return sentence(((_SLOW.id,),), thai_of(_SLOW))  # slow


def _syl(media=None, sentences=(), targets=None):
    t = targets or (target("slow/receptive", "slow"),)
    return Syllabus(words=(_SLOW,), targets=t, sentences=tuple(sentences),
                    media=media or FakeMediaIndex())


def _rules(rid):
    return [r for r in RULES if r.id == rid]


def test_target_without_picture_is_an_error_finding():
    findings = _rules("target/picture-required")[0].check(_syl())
    assert [f.note_id for f in findings] == ["slow"]
    assert _rules("target/picture-required")[0].severity == "error"


def test_sentence_introduced_target_needs_no_picture_but_still_needs_a_recording():
    # A glue word introduced only by a sentence is abstract -- no picture
    # can depict it, so it never joins target/picture-required's subjects
    # (spec 1 section 3, r5). Its recording is still required.
    s = _syl(targets=(target("slow/receptive", "slow", introduction="sentence"),))
    assert _rules("target/picture-required")[0].check(s) == []
    assert [f.note_id for f in _rules("target/recording-required")[0].check(s)] == ["slow"]


def test_target_with_picture_recording_and_sentence_has_no_completeness_findings():
    # recording_speakers deliberately left empty -- target/recording-required
    # must key off recording_provenance, not recording_speakers (see the
    # next test): speaker_id is nullable, so a real recording can have no
    # speakers and still be a current-best recording.
    media = FakeMediaIndex(pictures={"slow"},
                           recording_provenance={"slow": {"source": "forvo", "speaker": Speaker("s", "native")}})
    s = _syl(media=media, sentences=[_slow_sentence()])
    for rid in ("target/picture-required", "target/recording-required", "target/sentence-required"):
        assert _rules(rid)[0].check(s) == []


def test_target_recording_required_is_silent_for_a_recording_with_no_speaker_id():
    # media row's speaker_id is nullable (store.py schema): a real
    # current-best recording with none on file must not close the gate.
    media = FakeMediaIndex(recording_speakers={"slow": frozenset()},
                           recording_provenance={"slow": {"source": "forvo", "speaker": Speaker("s", "native")}})
    assert _rules("target/recording-required")[0].check(_syl(media=media)) == []


def test_target_sentence_required_is_silent_when_a_sentence_fills_the_target():
    s = _syl(sentences=[_slow_sentence()])
    assert _rules("target/sentence-required")[0].check(s) == []


def test_synthetic_recording_is_a_warning():
    media = FakeMediaIndex(recording_speakers={"slow": frozenset({"tts:v"})},
                           recording_provenance={"slow": {"source": "tts", "speaker": Speaker("tts:v", "synthetic")}})
    r = _rules("recording/synthetic")[0]
    assert r.severity == "warn"
    assert [f.note_id for f in r.check(_syl(media=media))] == ["slow"]


def _mid_low_pair():
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    low_word = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    pair = MinimalPair.create(id=PairId("tone:mid-low/klai"), confusion=confusion,
                              members=(mid_word, low_word))
    return confusion, mid_word, low_word, pair


# --- pair/rendition-required --------------------------------------------

def test_pair_rendition_required_flags_a_pair_with_no_rendition():
    confusion, mid_word, low_word, pair = _mid_low_pair()
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,), confusions=(confusion,))
    findings = [f for f in syllabus.report().findings if f.rule == "pair/rendition-required"]
    assert [f.note_id for f in findings] == [pair.id]


def test_pair_rendition_required_flags_a_half_recorded_pair():
    confusion, mid_word, low_word, pair = _mid_low_pair()
    media = FakeMediaIndex(rendition_provenance={
        pair.id: ({"speaker_id": "a", "speaker": Speaker("a", "native")},)})
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,), confusions=(confusion,),
                             media=media)
    findings = [f for f in syllabus.report().findings if f.rule == "pair/rendition-required"]
    assert [f.note_id for f in findings] == [pair.id]


def test_pair_rendition_required_is_silent_when_every_member_has_a_recording():
    confusion, mid_word, low_word, pair = _mid_low_pair()
    media = FakeMediaIndex(rendition_provenance={pair.id: (
        {"speaker_id": "a", "speaker": Speaker("a", "native")},
        {"speaker_id": "b", "speaker": Speaker("b", "native")})})
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,), confusions=(confusion,),
                             media=media)
    findings = [f for f in syllabus.report().findings if f.rule == "pair/rendition-required"]
    assert findings == []


# --- grapheme/keyword-picture-required --------------------------------------

def test_grapheme_keyword_picture_required_flags_a_missing_picture():
    chicken = word("chicken", "ไก่", "chicken")
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                               consonant_class="mid", keyword_word=chicken)
    syllabus = make_syllabus(words=(chicken,), graphemes=(grapheme,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "grapheme/keyword-picture-required"]
    assert [f.note_id for f in findings] == ["ก"]


def test_grapheme_keyword_picture_required_is_silent_when_the_keyword_has_a_picture():
    chicken = word("chicken", "ไก่", "chicken")
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                               consonant_class="mid", keyword_word=chicken)
    media = FakeMediaIndex(pictures={"chicken"})
    syllabus = make_syllabus(words=(chicken,), graphemes=(grapheme,), media=media)
    findings = [f for f in syllabus.report().findings
               if f.rule == "grapheme/keyword-picture-required"]
    assert findings == []


# --- rendition/synthetic -----------------------------------------------------

def test_rendition_synthetic_flags_a_tts_rendition():
    confusion, mid_word, low_word, pair = _mid_low_pair()
    media = FakeMediaIndex(rendition_provenance={pair.id: (
        {"speaker_id": "a", "speaker": Speaker("a", "native")},
        {"speaker_id": "tts", "speaker": Speaker("tts", "synthetic")})})
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,), confusions=(confusion,),
                             media=media)
    findings = [f for f in syllabus.report().findings if f.rule == "rendition/synthetic"]
    assert [f.note_id for f in findings] == [pair.id]


# --- sentence/synthetic-productive ------------------------------------------

def test_sentence_synthetic_productive_flags_tts_audio_on_a_filling_productive_sentence():
    rice = word("rice", "ข้าว")  # rice
    t_rice = target("rice/productive", "rice", "productive")
    s = sentence(((rice.id,),), thai_of(rice), voice="learner_voice")  # rice
    media = FakeMediaIndex(recording_provenance={
        sentence_note_id(s): {"source": "tts", "speaker": Speaker("tts:v", "synthetic")}})
    syllabus = make_syllabus(words=(rice,), targets=(t_rice,), sentences=(s,), media=media)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/synthetic-productive"]
    assert [f.note_id for f in findings] == [sentence_note_id(s)]


def test_sentence_synthetic_productive_is_silent_for_a_receptive_only_sentence():
    rice = word("rice", "ข้าว")  # rice
    t_rice = target("rice/receptive", "rice", "receptive")
    s = sentence(((rice.id,),), thai_of(rice), voice="learner_voice")  # rice
    media = FakeMediaIndex(recording_provenance={
        sentence_note_id(s): {"source": "tts", "speaker": Speaker("tts:v", "synthetic")}})
    syllabus = make_syllabus(words=(rice,), targets=(t_rice,), sentences=(s,), media=media)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/synthetic-productive"]
    assert findings == []


# --- coverage/speakers (E7) --------------------------------------------

def test_coverage_speakers_ignores_unknown_attributes():
    index = FakeMediaIndex(speakers={
        "recording": (Speaker("a", "native", sex="male"), Speaker("b", "native"))})
    syllabus = make_syllabus(media=index)
    metric = next(m for m in syllabus.report().metrics if m.rule == "coverage/speakers")
    assert metric.detail["recording"]["speakers"] == 2
    assert metric.detail["recording"]["sex"] == {"male": 1}
    assert metric.detail["recording"]["age_band"] == {}
    assert metric.detail["recording"]["region"] == {}


def test_coverage_speakers_is_empty_for_a_corpus_with_no_current_best_speakers():
    syllabus = make_syllabus(media=FakeMediaIndex())
    metric = next(m for m in syllabus.report().metrics if m.rule == "coverage/speakers")
    assert metric.detail["rendition"] == {"speakers": 0, "sex": {}, "age_band": {}, "region": {}}


# --- picture/fit judged_subjects --------------------------------------------

def test_picture_fit_judged_subjects_includes_a_targeted_word_with_a_picture():
    media = FakeMediaIndex(pictures={"slow"})
    assert PICTURE_FIT.judged_subjects(_syl(media=media)) == [("slow", "sha-slow")]


def test_picture_fit_judged_subjects_excludes_a_targeted_word_with_no_picture():
    assert PICTURE_FIT.judged_subjects(_syl()) == []


def test_picture_fit_judged_subjects_excludes_a_sentence_introduced_word_even_with_a_picture():
    # A sentence-introduced word compiles to no picture card, so judging a
    # picture attached to one anyway buys nothing.
    media = FakeMediaIndex(pictures={"slow"})
    s = _syl(media=media, targets=(target("slow/receptive", "slow", introduction="sentence"),))
    assert PICTURE_FIT.judged_subjects(s) == []


def test_picture_fit_rubric_is_the_old_text_verbatim():
    from thai_deck_eval.judge.prompts import PICTURE_RULES
    for rid in ("judge/image-off-phrase", "judge/image-irrelevant", "judge/image-embedded-text"):
        assert PICTURE_RULES[rid] in PICTURE_FIT_RUBRIC


def test_every_rubric_comes_from_a_judged_rule():
    assert set(rubrics_for(RULES)) == {r.role for r in RULES if r.shape == "judged"}


def test_rubrics_for_maps_a_judged_rules_own_rubric():
    r = rubrics_for(RULES)
    assert r["picture-for-word"] == PICTURE_FIT_RUBRIC
    assert r["picture-preference"] == PICTURE_PREFERENCE.rubric


# --- picture/preference (judged) --------------------------------------------

def test_picture_preference_role_and_shape():
    assert PICTURE_PREFERENCE.shape == "judged"
    assert PICTURE_PREFERENCE.role == "picture-preference"
    assert PICTURE_PREFERENCE.judged_subjects(_syl()) == []


# --- word/pronunciation-corroborated (E4) -----------------------------------

def test_disputed_pronunciation_is_an_error_on_the_word():
    rice = word("rice", "ข้าว", corroboration="disputed")  # rice
    t = target("rice/receptive", "rice", "receptive")
    syllabus = make_syllabus(words=(rice,), targets=(t,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "word/pronunciation-corroborated"]
    assert [f.note_id for f in findings] == ["rice"]


def test_corroborated_pronunciation_is_silent():
    rice = word("rice", "ข้าว", corroboration="engines_agree")  # rice
    t = target("rice/receptive", "rice", "receptive")
    syllabus = make_syllabus(words=(rice,), targets=(t,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "word/pronunciation-corroborated"]
    assert findings == []


# --- word/classifier-known (E5) ---------------------------------------------

def test_classifier_known_flags_an_unresolvable_classifier():
    dog = word("dog", "หมา", classifier="ghost-classifier")  # dog
    syllabus = make_syllabus(words=(dog,))
    findings = [f for f in syllabus.report().findings if f.rule == "word/classifier-known"]
    assert [f.note_id for f in findings] == ["dog"]
    assert _rules("word/classifier-known")[0].severity == "warn"


def test_classifier_known_is_silent_when_the_classifier_resolves():
    unit = word("unit", "ตัว")  # classifier word
    dog = word("dog", "หมา", classifier="unit")  # dog
    syllabus = make_syllabus(words=(dog, unit))
    findings = [f for f in syllabus.report().findings if f.rule == "word/classifier-known"]
    assert findings == []


# --- sentence/recording-required (F7) ---------------------------------------

def test_sentence_recording_required_flags_a_sentence_with_no_recording():
    rice = word("rice", "ข้าว")  # rice
    s = sentence(((rice.id,),), thai_of(rice), voice="learner_voice")  # rice
    syllabus = make_syllabus(words=(rice,), sentences=(s,))
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/recording-required"]
    assert [f.note_id for f in findings] == [sentence_note_id(s)]


def test_sentence_recording_required_is_silent_with_a_current_best_recording():
    rice = word("rice", "ข้าว")  # rice
    s = sentence(((rice.id,),), thai_of(rice), voice="learner_voice")  # rice
    media = FakeMediaIndex(recording_provenance={
        sentence_note_id(s): {"source": "forvo", "speaker": Speaker("s", "native")}})
    syllabus = make_syllabus(words=(rice,), sentences=(s,), media=media)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/recording-required"]
    assert findings == []


# --- scene/fit: a sentence's scene picture has its own rubric ---------------

def test_the_scene_rubric_is_registered_for_the_scene_role():
    assert rubrics_for(RULES)["scene-for-sentence"] == SCENE_FIT_RUBRIC
    assert "scene the sentence describes" in SCENE_FIT_RUBRIC


def test_the_sentence_rubric_asks_whether_the_gloss_states_the_meaning():
    assert "gloss" in SENTENCE_FOR_TARGET_RUBRIC
