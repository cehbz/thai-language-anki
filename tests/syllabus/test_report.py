"""Syllabus.report(): runs check-shaped and measure-shaped rules, reads
judged-rule verdicts and waivers from the AssessmentReader (never calls a
judge), gates on unwaived error findings, and stamps a content hash that
goes stale the moment the aggregate's content changes (spec 1, section 3).
"""
from thai_syllabus.profile import Profile
from thai_syllabus.rules import Finding, Metric, Rule
from thai_syllabus.syllabus import Syllabus

from .builders import target, word
from .fakes import FakeAssessmentReader, FakeTokenizer


def always_fails(syllabus) -> list[Finding]:
    return [Finding(rule="test/always-fails", note_id=w.id, evidence="bad")
           for w in syllabus.words]


def count_words(syllabus) -> Metric:
    return Metric(rule="test/word-count", value=float(len(syllabus.words)))


ERROR_CHECK = Rule(id="test/always-fails", principle="F1", severity="error",
                   shape="check", check=always_fails)
WORD_COUNT = Rule(id="test/word-count", principle="F1", severity="info",
                  shape="measure", measure=count_words)


def make_syllabus(words=(), rules=(), assessments=None):
    return Syllabus(words=words, profile=Profile(register="male_colloquial"),
                    tokenizer=FakeTokenizer(), rules=rules,
                    assessments=assessments or FakeAssessmentReader())


def test_report_runs_check_rules_and_collects_findings():
    rice = word("rice", "ข้าว")  # rice
    rep = make_syllabus((rice,), rules=(ERROR_CHECK,)).report()
    assert {f.rule for f in rep.findings} == {"test/always-fails"}
    assert len(rep.findings) == 1


def test_report_runs_measure_rules_and_collects_metrics():
    rice = word("rice", "ข้าว")  # rice
    dog = word("dog", "หมา")  # dog
    rep = make_syllabus((rice, dog), rules=(WORD_COUNT,)).report()
    assert rep.metrics == (Metric(rule="test/word-count", value=2.0),)


def test_gate_fails_on_an_unwaived_error_finding():
    rice = word("rice", "ข้าว")  # rice
    rep = make_syllabus((rice,), rules=(ERROR_CHECK,)).report()
    assert rep.gate is False


def test_gate_passes_when_the_only_error_finding_is_waived():
    rice = word("rice", "ข้าว")  # rice
    waived_identity = ("test/always-fails", rice.id, None)
    reader = FakeAssessmentReader(waived={waived_identity})
    rep = make_syllabus((rice,), rules=(ERROR_CHECK,), assessments=reader).report()
    assert rep.gate is True
    # The finding is still reported -- waiving suppresses the gate, not the record.
    assert len(rep.findings) == 1


def test_gate_passes_with_no_error_findings():
    rep = make_syllabus((), rules=()).report()
    assert rep.gate is True


def test_a_warn_finding_does_not_fail_the_gate():
    warn_rule = Rule(id="test/warns", principle="F1", severity="warn",
                     shape="check",
                     check=lambda s: [Finding(rule="test/warns", note_id="x",
                                              evidence="meh")])
    rep = make_syllabus((), rules=(warn_rule,)).report()
    assert rep.gate is True
    assert len(rep.findings) == 1


def test_judged_rule_findings_come_from_cached_verdicts_never_a_live_judge():
    rice = word("rice", "ข้าว")  # rice
    calls = []

    def subjects(s):
        return [(w.id, None) for w in s.words]

    judged_rule = Rule(id="test/judged", principle="F1", severity="error",
                       shape="judged", rubric="is this word natural?",
                       judged_subjects=subjects)
    reader = FakeAssessmentReader(verdicts={("test/judged", rice.id, None): False})
    rep = make_syllabus((rice,), rules=(judged_rule,), assessments=reader).report()
    assert len(rep.findings) == 1
    assert rep.findings[0].rule == "test/judged"
    assert rep.gate is False


def test_an_unassessed_judged_subject_produces_no_finding():
    rice = word("rice", "ข้าว")  # rice

    def subjects(s):
        return [(w.id, None) for w in s.words]

    judged_rule = Rule(id="test/judged", principle="F1", severity="error",
                       shape="judged", rubric="?", judged_subjects=subjects)
    rep = make_syllabus((rice,), rules=(judged_rule,)).report()
    assert rep.findings == ()
    assert rep.gate is True


# --- syllabus_state_id: structural staleness ---------------------------------

def test_state_id_is_stable_for_identical_content():
    rice = word("rice", "ข้าว")  # rice
    a = make_syllabus((rice,)).state_id()
    b = make_syllabus((rice,)).state_id()
    assert a == b


def test_state_id_changes_when_a_word_is_added():
    rice = word("rice", "ข้าว")  # rice
    dog = word("dog", "หมา")  # dog
    a = make_syllabus((rice,)).state_id()
    b = make_syllabus((rice, dog)).state_id()
    assert a != b


def test_report_carries_the_live_state_id():
    rice = word("rice", "ข้าว")  # rice
    syllabus = make_syllabus((rice,))
    assert syllabus.report().syllabus_state_id == syllabus.state_id()


# --- rulebook_id: staleness = EITHER differs (spec 3 section 6) --------------

def test_report_carries_the_live_rulebook_id():
    syllabus = make_syllabus(())
    assert syllabus.report().rulebook_id == syllabus.rulebook_id()


def test_rulebook_id_is_stable_for_identical_rulebook_text_and_rules():
    a = Syllabus(rules=(ERROR_CHECK,), rulebook_text="severities: {}").rulebook_id()
    b = Syllabus(rules=(ERROR_CHECK,), rulebook_text="severities: {}").rulebook_id()
    assert a == b


def test_rulebook_id_changes_when_the_rulebook_text_changes_but_content_does_not():
    rice = word("rice", "ข้าว")  # rice
    a = Syllabus(words=(rice,), rulebook_text="severities: {}")
    b = Syllabus(words=(rice,), rulebook_text="severities: {pair/exact-confusion: warn}")
    assert a.state_id() == b.state_id()          # same content
    assert a.rulebook_id() != b.rulebook_id()     # different rulebook


def test_rulebook_id_changes_when_the_registrys_rule_ids_change():
    a = Syllabus(rules=(ERROR_CHECK,)).rulebook_id()
    b = Syllabus(rules=(ERROR_CHECK, WORD_COUNT)).rulebook_id()
    assert a != b


# --- compile(): spec 4 -- see tests/syllabus/test_compile.py for the real
# implementation's coverage (models/fields/guids/tags/due/gate/dropped-
# cards); Syllabus.compile() now delegates to compile.compile_syllabus and
# needs a SyllabusDb/MediaStore this module's bare-bones make_syllabus()
# fixture doesn't build.
