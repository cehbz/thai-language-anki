"""The seeded rulebook (spec 1, section 4): pair/exact-confusion,
grapheme/keyword-contains-symbol, sentence/fills-novelty, syllabus/closure,
media/picture-required, coverage/confusions, rulebook/traceability, and one
judged rule (sentence/register-natural) exercising the AssessmentReader path.
"""
from thai_syllabus.entities import Grapheme, MinimalPair, SoundConfusion, Word
from thai_syllabus.ids import ConfusionId, PairId
from thai_syllabus.profile import Profile
from thai_syllabus.rules import Rule
from thai_syllabus.rulebook import ENFORCEMENT_PRINCIPLES, PRINCIPLES, RULES, traceability_metric
from thai_syllabus.syllabus import Syllabus

from .builders import pron, sentence, syl, target, word
from .fakes import FakeAssessmentReader, FakeTokenizer


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
