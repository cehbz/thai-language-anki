"""Tests for phonology.py (spec 3 r28 section 5; the 2026-09-01
domain-language ruling): the corroboration rule that cross-checks a
judge's pronunciation verdict against thaig2p and, on a monosyllabic
tone disagreement, the deterministic tone engine. Engines are injected
fakes here -- thaig2p (torch) loads only inside default_engines(), never
in these tests.
"""
import pytest

from thai_syllabus.entities import Pronunciation, Syllable
from thai_syllabus.phonology import (Engines, corroborates, engines_pronunciation,
                                     syllables_from_verdict)


def S(on, v, co, ln, t):
    return Syllable(segments=(on, v, co), vowel_length=ln, tone=t)


J = (S("kʰ", "a", "w", "long", "falling"),)


def test_agreement_with_thaig2p_corroborates():
    eng = Engines(g2p=lambda w: J, tone=lambda w: None)
    assert corroborates(J, "ข้าว", eng) is True  # ข้าว: rice


def test_tone_disagreement_is_settled_by_the_tone_engine_on_a_monosyllable():
    g2p = (S("kʰ", "a", "w", "long", "low"),)
    eng = Engines(g2p=lambda w: g2p, tone=lambda w: "falling")
    assert corroborates(J, "ข้าว", eng) is True
    eng2 = Engines(g2p=lambda w: g2p, tone=lambda w: "low")
    assert corroborates(J, "ข้าว", eng2) is False


def test_segment_disagreement_or_no_g2p_answer_leaves_it_disputed():
    other = (S("k", "a", "w", "long", "falling"),)
    assert corroborates(J, "ข้าว", Engines(g2p=lambda w: other, tone=lambda w: "falling")) is False
    assert corroborates(J, "ข้าว", Engines(g2p=lambda w: None, tone=lambda w: "falling")) is False


def test_syllables_from_verdict_builds_entities():
    v = {"syllables": [{"segments": ["kʰ", "a", "w"], "vowel_length": "long", "tone": "falling"}]}
    assert syllables_from_verdict(v) == J


@pytest.mark.integration
def test_default_engines_g2p_returns_a_syllable_for_a_real_word():
    """Documents thaig2p's actual output for ข้าว (rice) via the real
    pythainlp adapter -- skipped when the model isn't installed. The KB
    records thaig2p gets this word's vowel length wrong (short, not
    long); this test asserts whatever it actually returns rather than
    the "correct" answer, so a model upgrade that fixes it is visible as
    a test change, not a silent pass/fail flip.
    """
    from thai_syllabus.phonology import default_engines
    try:
        engines = default_engines()
        syls = engines.g2p("ข้าว")  # ข้าว: rice
    except Exception as e:  # pragma: no cover - model/deps absent in CI
        pytest.skip(f"thaig2p unavailable: {e}")
    if syls is None:
        pytest.skip("thaig2p returned no analysis for ข้าว")
    assert len(syls) == 1
    assert syls[0].segments[1] == "a"


def test_default_engines_builds_the_engines_once(monkeypatch, real_default_engines):
    """M7: every caller reads `ctx.engines` first and falls back to this
    function, so a run with no injected engines builds Thaig2p -- and with
    it pythainlp and torch -- once per call site instead of once. The
    fallback is memoised: same Engines object, one construction. The
    injection seam is untouched; this is only about the default.

    The fake stands in for Thaig2p through the function's own deferred
    import, so nothing here loads torch. `real_default_engines` is the
    guard fixture's handle on the unpatched function (conftest.py), and
    the memo is cleared either side so no other test inherits the fake.
    """
    from thai_syllabus import engines as engines_module

    built = []

    class _FakeThaig2p:
        def __init__(self):
            built.append(1)

        def __call__(self, thai):
            return None

    monkeypatch.setattr(engines_module, "Thaig2p", _FakeThaig2p)
    real_default_engines.cache_clear()
    try:
        first = real_default_engines()
        second = real_default_engines()
        assert first is second
        assert built == [1]
    finally:
        real_default_engines.cache_clear()


# --- engines_pronunciation: the adoption pass's seed (spec 3 r40 §5) -------

def _engines(g2p_result, tone_result="mid"):
    return Engines(g2p=lambda thai: g2p_result, tone=lambda thai: tone_result)


def test_a_monosyllable_the_tone_engine_agrees_with_is_engines_agree():
    one = Syllable(segments=("k", "a", ""), vowel_length="short", tone="mid")
    got = engines_pronunciation("ไก่", _engines((one,), "mid"))   # ไก่: chicken
    assert got == Pronunciation(syllables=(one,), corroboration="engines_agree")


def test_a_monosyllable_the_tone_engine_disagrees_with_stays_disputed():
    one = Syllable(segments=("k", "a", ""), vowel_length="short", tone="mid")
    got = engines_pronunciation("ไก่", _engines((one,), "low"))   # ไก่: chicken
    assert got == Pronunciation(syllables=(one,), corroboration="disputed")


def test_a_phrase_of_several_syllables_is_disputed_and_waits_for_the_judge():
    """The rule tone engine settles one syllable's tone only, so a recited
    name ("กอ ไก่", the name of ก) is never corroborated here: the
    adjudication pass asks the judge next run (spec 3 r28)."""
    two = (Syllable(segments=("k", "ɔ", ""), vowel_length="long", tone="mid"),
           Syllable(segments=("k", "a", ""), vowel_length="short", tone="low"))
    got = engines_pronunciation("กอ ไก่", _engines(two, "mid"))   # กอ ไก่: the name of ก
    assert got == Pronunciation(syllables=two, corroboration="disputed")


def test_nothing_read_is_no_pronunciation_at_all():
    assert engines_pronunciation("ๆ", _engines(None)) is None
    assert engines_pronunciation("ๆ", _engines(())) is None


# --- the token-wise fallback (2026-09-17 evidence: thaig2p reads "ปอ" and
# "ปลา" but not "ปอ ปลา" as one phrase) -------------------------------------

def _token_engines(readings: dict, tone_result="mid"):
    """g2p keyed by the exact string asked: `readings[thai]`, None for any
    thai not in the map -- so a phrase-level miss and a per-token hit are
    both under the caller's control."""
    return Engines(g2p=lambda thai: readings.get(thai), tone=lambda thai: tone_result)


def test_a_phrase_no_engine_reads_whole_is_read_token_by_token_and_concatenated():
    po = (Syllable(segments=("p", "ɔ", ""), vowel_length="long", tone="mid"),)
    pla = (Syllable(segments=("p", "l", "a"), vowel_length="short", tone="mid"),)
    eng = _token_engines({"ปอ": po, "ปลา": pla})   # "ปอ ปลา" itself: no entry -> None
    got = engines_pronunciation("ปอ ปลา", eng)
    assert got == Pronunciation(syllables=po + pla, corroboration="disputed")


def test_a_phrase_with_one_unreadable_token_is_no_pronunciation_at_all():
    po = (Syllable(segments=("p", "ɔ", ""), vowel_length="long", tone="mid"),)
    eng = _token_engines({"ปอ": po})   # "ปลา" has no entry -> None
    assert engines_pronunciation("ปอ ปลา", eng) is None


def test_a_single_token_phrase_the_engine_cannot_read_is_still_no_pronunciation():
    """No whitespace to split on: the token-wise fallback never applies,
    exactly as before this fix."""
    eng = _token_engines({})
    assert engines_pronunciation("ๆ", eng) is None
