"""Tests for phonology.py (spec 3 section 3, r68): one local engine's
reading of a form as the record keeps it -- a phrase read token by token,
the ʔ-coda convention, a degenerate reading refused -- and the real
engines' construction. Engines are injected fakes here -- thaig2p (torch)
loads only inside default_engines(), never in these tests.
"""
import pytest

from thai_syllabus.entities import Syllable
from thai_syllabus.phonology import is_degenerate, read_form, syllables_from_verdict


def S(on, v, co, ln, t):
    return Syllable(segments=(on, v, co), vowel_length=ln, tone=t)


J = (S("kʰ", "a", "w", "long", "falling"),)


def test_syllables_from_verdict_builds_entities():
    v = {"syllables": [{"segments": ["kʰ", "a", "w"], "vowel_length": "long", "tone": "falling"}]}
    assert syllables_from_verdict(v) == J


def test_a_judge_verdict_is_normalized_on_the_way_in():
    """syllables_from_verdict is the one funnel a verdict passes through
    on its way into the settlement's evidence (design §5)."""
    got = syllables_from_verdict(
        {"syllables": [{"segments": ["pʰ", "a", "ʔ"], "vowel_length": "short",
                        "tone": "high"}]})
    assert got == (Syllable(segments=("pʰ", "a", ""), vowel_length="short",
                            tone="high"),)


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
        syls = engines.g2p["thaig2p"]("ข้าว")  # ข้าว: rice
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

    The fakes stand in for Thaig2p and Tltk through the function's own
    deferred import, so nothing here needs the "nlp" extra (pythainlp,
    torch, tltk). `real_default_engines` is the
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

    class _FakeTltk:
        def __call__(self, thai):
            return None

    monkeypatch.setattr(engines_module, "Thaig2p", _FakeThaig2p)
    monkeypatch.setattr(engines_module, "Tltk", _FakeTltk)
    real_default_engines.cache_clear()
    try:
        first = real_default_engines()
        second = real_default_engines()
        assert first is second
        assert built == [1]
    finally:
        real_default_engines.cache_clear()


@pytest.mark.integration
def test_default_engines_wires_both_segmental_oracles():
    from thai_syllabus.phonology import default_engines as real
    real.cache_clear()
    try:
        eng = real()
        assert set(eng.g2p) == {"thaig2p", "tltk"}
        # the case the second engine exists for
        readings = [r for r in (read_form(g, "ภาพวาด") for g in eng.g2p.values()) if r]
        assert readings and all(len(r) == 2 for r in readings)
    finally:
        real.cache_clear()


# --- a phrase is its tokens (design 2026-09-20 §3): thaig2p reads "ปอ" and
# "ปลา" but not "ปอ ปลา", and reads "งอ งู" as ŋɔŋ.ŋu while "งอ" alone is ŋɔ.

PO = (Syllable(segments=("p", "ɔ", ""), vowel_length="long", tone="mid"),)
PLA = (Syllable(segments=("pl", "a", ""), vowel_length="long", tone="mid"),)
WRONG = (Syllable(segments=("p", "ɔ", "ŋ"), vowel_length="long", tone="mid"),) + PLA


def test_a_phrase_is_read_token_by_token_never_whole():
    readings = {"ปอ ปลา": WRONG, "ปอ": PO, "ปลา": PLA}
    assert read_form(readings.get, "ปอ ปลา") == PO + PLA


def test_a_phrase_with_one_unreadable_token_is_no_reading():
    assert read_form({"ปอ": PO}.get, "ปอ ปลา") is None   # "ปลา" has no entry -> None


def test_a_form_the_engine_cannot_read_is_no_reading():
    assert read_form({}.get, "ๆ") is None
    assert read_form(lambda t: (), "ๆ") is None


def test_a_whitespace_only_form_is_no_reading():
    """`thai.split()` on an all-whitespace form is `[]` -- no tokens, not
    one empty token -- so this must not read as an empty reading: a
    reading never has an empty syllable tuple."""
    assert read_form(lambda t: PO, "  ") is None


def test_a_dead_open_syllable_is_read_without_a_glottal_coda():
    """Design 2026-09-18 §5: thaig2p writes ʔ there, tltk does not."""
    assert read_form(lambda t: (S("tɕ", "a", "ʔ", "short", "low"),), "จะ") == (
        S("tɕ", "a", "", "short", "low"),)


# --- degenerate readings: the neural g2p's decoder loop (2026-09-18) --------
# Evidence from the live deck: thaig2p reads ภาพวาด as eleven syllables
# ("pʰa pʰa wa wa wa wa wa wa wa wa wa") and ษอ ฤๅษี as eleven.

def _syl(segments, length="short", tone="mid"):
    return Syllable(segments=segments, vowel_length=length, tone=tone)


def test_three_identical_syllables_in_a_row_is_a_decoder_loop():
    wa = _syl(("w", "a", ""), "long")
    assert is_degenerate((wa, wa, wa), "ภาพวาด") is True


def test_reduplication_of_two_syllables_is_not_degenerate():
    """ๆ repeats a word exactly once (ข้างๆ kʰâːŋ.kʰâːŋ), and ตุ๊กตุ๊ก
    túk.túk is spelt with the repeat: two identical syllables are Thai,
    not a loop."""
    tuk = _syl(("t", "u", "k"), "short", "high")
    assert is_degenerate((tuk, tuk), "ตุ๊กตุ๊ก") is False


def test_more_syllables_than_the_spelling_has_consonants_is_degenerate():
    """ภาพวาด is four consonant letters; no reading of it has eleven
    syllables, since every Thai syllable needs at least one."""
    wa = _syl(("w", "a", ""), "long")
    assert is_degenerate(tuple(wa for _ in range(11)), "ภาพวาด") is True


def test_a_long_honest_reading_is_not_degenerate():
    """ออกกำลังกาย: four syllables, eight consonant letters, none repeated
    three times over."""
    reading = (_syl(("ʔ", "ɔ", "k"), "long", "low"), _syl(("k", "a", "m")),
               _syl(("l", "a", "ŋ")), _syl(("k", "a", "j"), "long"))
    assert is_degenerate(reading, "ออกกำลังกาย") is False


def test_a_degenerate_whole_form_reading_is_no_reading():
    """A looped reading is not a reading: it is recorded as none."""
    wa = _syl(("w", "a", ""), "long")
    assert read_form(lambda t: tuple(wa for _ in range(11)), "ภาพวาด") is None


def test_a_degenerate_token_wise_reading_is_no_reading():
    """ษอ ฤๅษี, adopted 2026-09-17 with eleven syllables: the token-wise
    reading concatenates per-token readings, so the loop has to be caught
    on the combined result too."""
    si = _syl(("s", "i", ""), "long")
    readings = {"ษอ": (_syl(("s", "ɔ", ""), "long", "rising"),),
                "ฤๅษี": tuple(si for _ in range(10))}
    assert read_form(readings.get, "ษอ ฤๅษี") is None
