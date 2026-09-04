"""The seeded rulebook (spec 1, section 4): pair/exact-confusion,
grapheme/keyword-contains-symbol, sentence/fills-novelty, syllabus/closure,
media/picture-required, coverage/confusions, rulebook/traceability, and one
judged rule (sentence/register-natural) exercising the AssessmentReader path.
"""
from thai_syllabus.curated import RulebookConfig
from thai_syllabus.entities import Category, Grapheme, MinimalPair, SoundConfusion, Word
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.media import Speaker
from thai_syllabus.profile import Profile
from thai_syllabus.rules import Rule
from thai_syllabus.rulebook import (ENFORCEMENT_PRINCIPLES, PICTURE_FIT, PICTURE_FIT_RUBRIC,
                                    PRINCIPLES, RUBRICS_BY_ROLE, RULES, SENTENCE_REGISTER_NATURAL,
                                    apply_overlay, rubrics_for, sentence_note_id,
                                    traceability_metric)
from thai_syllabus.syllabus import Syllabus

from .builders import pron, sentence, syl, target, word
from .fakes import FakeAssessmentReader, FakeMediaIndex, FakeTokenizer


def make_syllabus(**kwargs):
    kwargs.setdefault("profile", Profile(register="male_colloquial"))
    kwargs.setdefault("tokenizer", FakeTokenizer())
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


def test_rulebook_traceability_rule_is_itself_in_the_registry():
    assert any(r.id == "rulebook/traceability" for r in RULES)


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

def test_sentence_fills_novelty_flags_a_sentence_exceeding_its_budget():
    rice = word("rice", "ข้าว")  # rice
    unmet1 = word("unmet1", "จาน")  # plate
    unmet2 = word("unmet2", "ช้อน")  # spoon
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    s = sentence("ข้าวอยู่ในจานกับช้อน", voice="learner_voice")  # the rice is on the plate with a spoon
    tok = FakeTokenizer({s.text: ["ข้าว", "อยู่ใน", "จาน", "กับ", "ช้อน"]})
    syllabus = make_syllabus(words=(rice, unmet1, unmet2), targets=(t_rice,),
                             sentences=(s,), tokenizer=tok)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/fills-novelty"]
    assert len(findings) == 1


def test_sentence_fills_novelty_is_silent_when_the_sentence_stays_within_budget():
    rice = word("rice", "ข้าว")  # rice
    t_rice = target("rice/receptive", "rice", "receptive", introduction="sentence")
    s = sentence("ข้าว", voice="learner_voice")  # rice
    tok = FakeTokenizer({s.text: ["ข้าว"]})
    syllabus = make_syllabus(words=(rice,), targets=(t_rice,), sentences=(s,),
                             tokenizer=tok)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/fills-novelty"]
    assert findings == []


# --- syllabus/closure ---------------------------------------------------

def test_closure_flags_a_target_whose_word_does_not_resolve():
    t = target("ghost/receptive", "ghost", "receptive")  # no matching Word
    syllabus = make_syllabus(words=(), targets=(t,))
    findings = [f for f in syllabus.report().findings if f.rule == "syllabus/closure"]
    assert len(findings) == 1
    assert findings[0].note_id == t.id


def test_closure_is_silent_when_every_reference_resolves():
    rice = word("rice", "ข้าว")  # rice
    t = target("rice/receptive", "rice", "receptive")
    syllabus = make_syllabus(words=(rice,), targets=(t,))
    findings = [f for f in syllabus.report().findings if f.rule == "syllabus/closure"]
    assert findings == []


# --- category/single-membership -----------------------------------------

def test_single_membership_flags_a_word_in_two_categories():
    syllabus = make_syllabus(categories=(
        Category(name="Food", members=frozenset({"rice"})),
        Category(name="Verbs", members=frozenset({"rice"}))))
    findings = [f for f in syllabus.report().findings
               if f.rule == "category/single-membership"]
    assert [f.note_id for f in findings] == ["rice"]


def test_single_membership_is_silent_when_every_word_is_in_at_most_one_category():
    syllabus = make_syllabus(categories=(
        Category(name="Food", members=frozenset({"rice"})),
        Category(name="Colors", members=frozenset({"red"}))))
    findings = [f for f in syllabus.report().findings
               if f.rule == "category/single-membership"]
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


# --- media/picture-required (gap-metric) -------------------------------------

def test_media_picture_required_counts_targeted_words_without_a_picture():
    from .fakes import FakeMediaIndex
    rice = word("rice", "ข้าว")  # rice
    dog = word("dog", "หมา")  # dog
    t1 = target("rice/receptive", "rice", "receptive")
    t2 = target("dog/receptive", "dog", "receptive")
    media = FakeMediaIndex(pictures={rice.id})
    syllabus = make_syllabus(words=(rice, dog), targets=(t1, t2), media=media)
    metrics = {m.rule: m for m in syllabus.report().metrics}
    assert metrics["media/picture-required"].detail["missing"] == [dog.id]


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
    assert detail == {"pairs": 1, "speakers": 1}


# --- sentence/register-natural (judged) --------------------------------------

def test_sentence_register_rule_reads_a_cached_verdict_and_never_a_live_judge():
    s = sentence("ผมกินข้าว", voice="learner_voice")  # I eat rice
    reader = FakeAssessmentReader()
    identity = None
    for r in RULES:
        if r.id == "sentence/register-natural":
            identity = r.judged_subjects(make_syllabus(sentences=(s,)))[0]
    assert identity is not None
    note_id, sha = identity
    reader = FakeAssessmentReader(verdicts={("sentence/register-natural", note_id, sha): False})
    syllabus = make_syllabus(sentences=(s,), assessments=reader)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/register-natural"]
    assert len(findings) == 1


def test_sentence_register_rule_is_silent_with_no_cached_verdict():
    s = sentence("ผมกินข้าว", voice="learner_voice")  # I eat rice
    syllabus = make_syllabus(sentences=(s,))
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
    assert by_id["syllabus/closure"].severity == "error"
    assert SENTENCE_REGISTER_NATURAL.severity == "warn"  # registry untouched


def test_overlay_ignores_a_rubric_override_targeting_a_non_judged_rule():
    cfg = RulebookConfig(rubrics={"syllabus/closure": "should be ignored"})
    out = apply_overlay(RULES, cfg)
    by_id = {r.id: r for r in out}
    assert by_id["syllabus/closure"].rubric is None


def test_rule_role_defaults_to_id():
    assert SENTENCE_REGISTER_NATURAL.role == "sentence/register-natural"


# --- completeness errors, synthetic/mixed-speaker warnings, picture fit -----

def _syl(media=None, sentences=(), targets=None):
    w = word("slow", "ช้า", "slow")
    t = targets or (target("slow/receptive", "slow"),)
    # A bare single-word sentence -- FakeTokenizer's default (no mapping)
    # treats unmapped text as one whole token, so this fills "slow" with
    # no companion tokens to register (a strict clause-3 novelty budget
    # would otherwise need every other token registered as its own Word).
    return Syllabus(words=(w,), targets=t, sentences=tuple(sentences),
                    media=media or FakeMediaIndex(), tokenizer=FakeTokenizer())


def _rules(rid):
    return [r for r in RULES if r.id == rid]


def test_target_without_picture_is_an_error_finding():
    findings = _rules("target/picture-required")[0].check(_syl())
    assert [f.note_id for f in findings] == ["slow"]
    assert _rules("target/picture-required")[0].severity == "error"


def test_target_with_picture_recording_and_sentence_has_no_completeness_findings():
    # recording_speakers deliberately left empty -- target/recording-required
    # must key off recording_provenance, not recording_speakers (see the
    # next test): speaker_id is nullable, so a real recording can have no
    # speakers and still be a current-best recording.
    media = FakeMediaIndex(pictures={"slow"},
                           recording_provenance={"slow": {"source": "forvo", "speaker": Speaker("s", "native")}})
    s = _syl(media=media, sentences=[sentence("ช้า")])
    for rid in ("target/picture-required", "target/recording-required", "target/sentence-required"):
        assert _rules(rid)[0].check(s) == []


def test_target_recording_required_is_silent_for_a_recording_with_no_speaker_id():
    # media row's speaker_id is nullable (store.py schema): a real
    # current-best recording with none on file must not close the gate.
    media = FakeMediaIndex(recording_speakers={"slow": frozenset()},
                           recording_provenance={"slow": {"source": "forvo", "speaker": Speaker("s", "native")}})
    assert _rules("target/recording-required")[0].check(_syl(media=media)) == []


def test_target_sentence_required_is_silent_when_a_sentence_fills_the_target():
    s = _syl(sentences=[sentence("ช้า")])
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


# --- rendition/synthetic, rendition/mixed-speakers --------------------------

def test_rendition_synthetic_flags_a_tts_rendition():
    confusion, mid_word, low_word, pair = _mid_low_pair()
    media = FakeMediaIndex(rendition_provenance={pair.id: (
        {"speaker_id": "a", "speaker": Speaker("a", "native")},
        {"speaker_id": "tts", "speaker": Speaker("tts", "synthetic")})})
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,), confusions=(confusion,),
                             media=media)
    findings = [f for f in syllabus.report().findings if f.rule == "rendition/synthetic"]
    assert [f.note_id for f in findings] == [pair.id]


def test_rendition_mixed_speakers_flags_different_speaker_ids():
    confusion, mid_word, low_word, pair = _mid_low_pair()
    media = FakeMediaIndex(rendition_provenance={pair.id: (
        {"speaker_id": "a", "speaker": Speaker("a", "native")},
        {"speaker_id": "b", "speaker": Speaker("b", "native")})})
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,), confusions=(confusion,),
                             media=media)
    findings = [f for f in syllabus.report().findings if f.rule == "rendition/mixed-speakers"]
    assert [f.note_id for f in findings] == [pair.id]


def test_rendition_mixed_speakers_ignores_a_null_speaker_id():
    # both rows present (so the count check passes) but one has no
    # speaker_id on file -- must not be counted as a second, distinct
    # speaker.
    confusion, mid_word, low_word, pair = _mid_low_pair()
    media = FakeMediaIndex(rendition_provenance={pair.id: (
        {"speaker_id": "a", "speaker": Speaker("a", "native")},
        {"speaker_id": None, "speaker": Speaker("a2", "native")})})
    syllabus = make_syllabus(words=(mid_word, low_word), pairs=(pair,), confusions=(confusion,),
                             media=media)
    findings = [f for f in syllabus.report().findings if f.rule == "rendition/mixed-speakers"]
    assert findings == []


# --- sentence/synthetic-productive ------------------------------------------

def test_sentence_synthetic_productive_flags_tts_audio_on_a_filling_productive_sentence():
    rice = word("rice", "ข้าว")  # rice
    t_rice = target("rice/productive", "rice", "productive")
    s = sentence("ข้าว", voice="learner_voice")  # rice
    tok = FakeTokenizer({s.text: ["ข้าว"]})
    media = FakeMediaIndex(recording_provenance={
        sentence_note_id(s): {"source": "tts", "speaker": Speaker("tts:v", "synthetic")}})
    syllabus = make_syllabus(words=(rice,), targets=(t_rice,), sentences=(s,), tokenizer=tok,
                             media=media)
    findings = [f for f in syllabus.report().findings
               if f.rule == "sentence/synthetic-productive"]
    assert [f.note_id for f in findings] == [sentence_note_id(s)]


def test_sentence_synthetic_productive_is_silent_for_a_receptive_only_sentence():
    rice = word("rice", "ข้าว")  # rice
    t_rice = target("rice/receptive", "rice", "receptive")
    s = sentence("ข้าว", voice="learner_voice")  # rice
    tok = FakeTokenizer({s.text: ["ข้าว"]})
    media = FakeMediaIndex(recording_provenance={
        sentence_note_id(s): {"source": "tts", "speaker": Speaker("tts:v", "synthetic")}})
    syllabus = make_syllabus(words=(rice,), targets=(t_rice,), sentences=(s,), tokenizer=tok,
                             media=media)
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


def test_picture_fit_rubric_is_the_old_text_verbatim():
    from thai_deck_eval.judge.prompts import PICTURE_RULES
    for rid in ("judge/image-off-phrase", "judge/image-irrelevant", "judge/image-embedded-text"):
        assert PICTURE_RULES[rid] in PICTURE_FIT_RUBRIC


def test_rubrics_for_maps_roles():
    r = rubrics_for(RULES)
    assert r["picture-for-word"] == PICTURE_FIT_RUBRIC
    assert "sentence-for-target" in r and "picture-preference" in r
