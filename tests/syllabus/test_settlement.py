"""settlement.py: each form's pronunciation settled from every reading on
record (spec 3 section 6 settlement(), r68; spec 1 r34) by a family-level
Dawid-Skene one-coin fit per feature. Evidence sets are small and
synthetic; the gatherer reads a real SyllabusDb.
"""
import pytest

from thai_syllabus.cachekeys import DictionaryKey, EngineKey, JudgeKey
from thai_syllabus.entities import Pronunciation, Syllable
from thai_syllabus.ids import WordId
from thai_syllabus.settlement import (Evidence, Member, evidence_on_record, judge_subjects,
                                      settle)
from thai_syllabus.store import SyllabusDb


def S(on, v, co, ln="long", t="mid"):
    return Syllable(segments=(on, v, co), vowel_length=ln, tone=t)


WIKT = Member("wiktionary")
G2P = Member("thaig2p", "1")
TLTK = Member("tltk", "1")
RULE = Member("rule-tone", "1")
JUDGE = Member("judge")
TLC = Member("thai-language.com")

KHAAW = (S("kʰ", "a", "w", "long", "falling"),)
KHAW = (S("kʰ", "a", "w", "short", "falling"),)


def agreeing(n, *members):
    """`n` background forms every member reads alike."""
    return {f"f{i}": {m: [(S("p", "a", str(i)),)] for m in members} for i in range(n)}


def test_three_families_agreeing_settle_the_form():
    ev = Evidence.of({"ข้าว": {WIKT: [KHAAW], TLTK: [KHAAW], TLC: [KHAAW]}})
    got = settle(ev).readings["ข้าว"]
    assert got == Pronunciation(syllables=KHAAW, posterior=pytest.approx(1.0),
                                 spoken_dissent=False)


def test_members_of_one_family_add_no_votes():
    """Wiktionary and thaig2p agreeing are one family's report: the two
    other families outvote it."""
    ev = Evidence.of({"ข้าว": {WIKT: [KHAW], G2P: [KHAW], TLTK: [KHAAW], TLC: [KHAAW]}})
    assert settle(ev).readings["ข้าว"].syllables == KHAAW


def test_a_family_reports_its_most_accurate_member_present():
    """Wiktionary's length is wrong on the background forms where
    thaig2p's agrees with the other families, so on the split form the
    Wiktionary family reports thaig2p's length, which thai-language.com
    joins."""
    by_form = agreeing(10, G2P, TLTK, TLC)
    for i in range(10):
        by_form[f"f{i}"][WIKT] = [(S("p", "a", str(i), "short"),)]
    by_form["ข้าว"] = {WIKT: [KHAAW], G2P: [KHAW], TLTK: [KHAAW], TLC: [KHAW]}
    assert settle(Evidence.of(by_form)).readings["ข้าว"].syllables == KHAW


def test_a_member_is_scored_only_against_the_other_families():
    """thaig2p is right wherever Wiktionary is wrong: scored against a
    posterior its own family helped make, the two would vouch for each
    other; against the other families alone thaig2p alone is accurate."""
    by_form = agreeing(10, G2P, TLTK, TLC)
    for i in range(10):
        by_form[f"f{i}"][WIKT] = [(S("b", "o", str(i)),)]
    got = settle(Evidence.of(by_form))
    assert got.member_accuracy["segments"][G2P] > 0.95
    assert got.member_accuracy["segments"][WIKT] < 0.05


def test_a_family_that_disagrees_is_fitted_less_accurate():
    by_form = agreeing(10, WIKT, TLTK, TLC)
    for i in range(3):
        by_form[f"f{i}"][WIKT] = [(S("p", "a", str(i), "short"),)]
    got = settle(Evidence.of(by_form))
    accuracy = got.accuracy["length"]
    assert accuracy["wiktionary"] < accuracy["spelling"]
    assert accuracy["wiktionary"] < accuracy["thai-language.com"]
    assert got.accuracy["segments"]["wiktionary"] == pytest.approx(accuracy["spelling"])


def test_a_multi_reading_report_matches_any_of_its_values():
    ev = Evidence.of({"ข้าว": {WIKT: [KHAAW, KHAW], TLTK: [KHAAW], TLC: [KHAAW]}})
    got = settle(ev).readings["ข้าว"]
    assert got.syllables == KHAAW and not got.spoken_dissent


def test_the_spoken_witness_dissents_when_no_reading_it_lists_matches_on_a_feature():
    ev = Evidence.of({"ข้าว": {WIKT: [KHAW], TLTK: [KHAAW], TLC: [KHAAW]},
                      "ดี": {TLTK: [(S("d", "i", ""),)], TLC: [(S("d", "i", ""),)]}})
    got = settle(ev).readings
    assert got["ข้าว"].syllables == KHAAW and got["ข้าว"].spoken_dissent
    assert not got["ดี"].spoken_dissent      # no Wiktionary reading: no witness


def test_the_posterior_is_the_least_of_the_features_winning_posteriors():
    """Segments and length are unanimous; a three-way split on tone
    alone sets it."""
    by_form = agreeing(10, WIKT, TLTK, TLC)
    by_form["ข้าว"] = {WIKT: [KHAAW], TLTK: [(S("kʰ", "a", "w", "long", "low"),)],
                       TLC: [(S("kʰ", "a", "w", "long", "rising"),)]}
    got = settle(Evidence.of(by_form)).readings
    assert got["f0"].posterior > 0.99
    assert [s.segments for s in got["ข้าว"].syllables] == [KHAAW[0].segments]
    assert got["ข้าว"].posterior < 0.5


def test_length_and_tone_winners_take_the_segments_winners_syllable_count():
    """A tone vector of another syllable count cannot be assembled onto
    the winning segments: the winner is taken among those that can."""
    two = (S("p", "a", ""), S("k", "a", ""))
    ev = Evidence.of({"ปากะ": {WIKT: [(S("p", "a", "k", "long", "high"),)],
                                TLTK: [(S("p", "a", "", "long", "mid"), S("k", "a", "", "long", "low"))],
                                TLC: [(S("p", "a", "", "long", "mid"), S("k", "a", "", "long", "falling"))]}})
    got = settle(ev).readings["ปากะ"]
    assert [s.segments for s in got.syllables] == [s.segments for s in two]
    assert [s.tone for s in got.syllables][0] == "mid"


def test_the_rule_tone_engine_reports_a_monosyllables_tone():
    low = (S("kʰ", "a", "w", "long", "low"),)
    ev = Evidence.of({"ข้าว": {WIKT: [KHAAW], RULE: ["low"], TLC: [low]}})
    assert settle(ev).readings["ข้าว"].syllables == low


def test_a_form_with_no_reading_has_no_settlement():
    ev = Evidence.of({"ข้าว": {RULE: ["falling"]}, "ดี": {WIKT: [], TLC: []}})
    assert settle(ev).readings == {}


def test_the_settlement_answers_a_forms_pronunciation_or_none():
    """The Settlement port (spec 2 r23): `reading(form)`."""
    settled = settle(Evidence.of({"ข้าว": {WIKT: [KHAAW], TLC: [KHAAW]}}))
    assert settled.reading("ข้าว") == settled.readings["ข้าว"]
    assert isinstance(settled.reading("ข้าว"), Pronunciation)
    assert settled.reading("ดี") is None


def test_a_forms_judge_subjects_are_its_words_and_its_candidate_subject():
    """A Word minted from an outside form keeps the verdict asked under
    `candidate:<form>`, so both subjects are the form's."""
    from .builders import word
    words = [word("rice", "ข้าว"), word("rice-2", "ข้าว"), word("good", "ดี")]
    assert judge_subjects(words, ["ปา"]) == {
        "ข้าว": [WordId("rice"), WordId("rice-2"), "candidate:ข้าว"],
        "ดี": [WordId("good"), "candidate:ดี"],
        "ปา": ["candidate:ปา"]}


def test_the_settlement_is_memoised_on_the_evidence():
    first = settle(Evidence.of({"ข้าว": {WIKT: [KHAAW], TLC: [KHAAW]}}))
    assert settle(Evidence.of({"ข้าว": {TLC: [KHAAW], WIKT: [KHAAW]}})) is first


# --- the evidence on record ----------------------------------------------

RUBRIC = "R"


def _judge(db, subject, thai, tone, *, rubric=RUBRIC):
    value = {"syllables": [{"segments": ["kʰ", "a", "w"], "vowel_length": "long",
                            "tone": tone}], "gloss": "rice"}
    db.append(port="assess", backend="judge",
              key=JudgeKey(rubric_sha=rubric, subject=subject, identity="",
                           role="pronunciation-for-word"),
              subject=subject,
              question={"kind": "pronunciation", "role": "pronunciation-for-word",
                        "rubric": rubric, "params": {"thai": thai}},
              answer={"value": value})


def _engine(db, name, version, form, value):
    db.append(port="provide", backend=name, key=EngineKey(name, version, form),
              subject=f"engine:{form}", question={"form": form}, answer={"value": value})


def _dictionary(db, source, form, answer):
    db.append(port="provide", backend=source, key=DictionaryKey(source, form),
              subject=f"dictionary:{form}", question={"form": form}, answer=answer)


def test_the_evidence_is_every_reading_on_record_of_the_form(tmp_path):
    db = SyllabusDb(tmp_path / "syllabus.db")
    syl = [{"segments": ["kʰ", "a", "w"], "vowel_length": "long", "tone": "falling"}]
    _engine(db, "thaig2p", "1", "ข้าว", syl)
    _engine(db, "tltk", "0", "ข้าว", [{**syl[0], "tone": "low"}])   # not the installed version
    _engine(db, "tltk", "1", "ข้าว", syl)
    _engine(db, "rule-tone", "1", "ข้าว", "falling")
    _dictionary(db, "wiktionary", "ข้าว", {"revid": 1, "ipa": ["/kʰaːw˥˩/", "/kʰaw˥˩/", "?"]})
    _dictionary(db, "thai-language.com", "ข้าว",
                {"entries": [{"id": "1", "paiboon": "kâao", "ipa": ["kʰaːw˥˩"]}]})
    _judge(db, "rice", "ข้าว", "falling")
    _judge(db, "candidate:ข้าว", "ข้าว", "high", rubric="OLD")    # stale
    got = evidence_on_record(db, {"ข้าว": ["rice", "candidate:ข้าว"]},
                             current_rubric={"pronunciation-for-word": RUBRIC})
    assert got == Evidence.of({"ข้าว": {G2P: [KHAAW], TLTK: [KHAAW], RULE: ["falling"],
                                        WIKT: [KHAAW, KHAW], TLC: [KHAAW], JUDGE: [KHAAW]}})


def test_the_judges_reading_is_its_newest_fresh_verdict_across_the_forms_subjects(tmp_path):
    db = SyllabusDb(tmp_path / "syllabus.db")
    _judge(db, "rice", "ข้าว", "low")
    _judge(db, "candidate:ข้าว", "ข้าว", "falling")
    _engine(db, "thaig2p", "1", "ดี", None)               # "no reading" is a row too
    got = evidence_on_record(db, {"ข้าว": ["rice", "candidate:ข้าว"], "ดี": ["good"]},
                             current_rubric={"pronunciation-for-word": RUBRIC})
    assert got == Evidence.of({"ข้าว": {JUDGE: [KHAAW]}})
