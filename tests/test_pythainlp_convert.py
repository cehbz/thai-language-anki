"""Offline unit tests for the pythainlp/tltk raw-string converters.

These feed hardcoded raw strings (from the pythainlp_adapter module
docstring's observed/verified examples) through the pure `_convert` /
`_convert_compact` functions directly. No pythainlp/tltk/torch import is
triggered: `thai_deck_eval.lang.pythainlp_adapter` only imports those
inside the port classes' methods, never at module level, so these
functions are safely importable and callable without the "nlp" extra
installed. NOT integration-marked.
"""
from thai_deck_eval.lang.pythainlp_adapter import _convert, _convert_compact
from thai_deck_eval.lang.tone import Tone


# --- thaig2p-shaped (token-per-phone, Chao tone letters) ------------------

def test_convert_simple_syllable():
    syl = _convert("m aː ˧")[0]
    assert (syl.onset, syl.vowel, syl.long, syl.coda, syl.tone) == (
        "m", "a", True, None, Tone.MID)

def test_convert_short_vowel_with_coda():
    syl = _convert("k a j ˨˩")[0]
    assert (syl.onset, syl.vowel, syl.long, syl.coda, syl.tone) == (
        "k", "a", False, "j", Tone.LOW)

def test_convert_rising_long_vowel_coda():
    syl = _convert("kʰ aː w ˩˩˦")[0]
    assert (syl.onset, syl.vowel, syl.long, syl.coda, syl.tone) == (
        "kʰ", "a", True, "w", Tone.RISING)

def test_convert_strips_no_release_diacritic_on_coda():
    # สามารถ, second syllable: stop coda carries a no-audible-release mark
    syl = _convert("m aː t̚ ˥˩")[0]
    assert syl.coda == "t" and syl.tone == Tone.FALLING

def test_convert_multisyllable_word():
    syls = _convert("s aː ˩˩˦ . m aː t̚ ˥˩")
    assert len(syls) == 2
    assert syls[0].tone == Tone.RISING
    assert syls[1].tone == Tone.FALLING

def test_convert_merges_diphthong_glide():
    # เมีย -> "ia" vowel: head "i" + non-syllabic "a" (combining inverted
    # breve below) merge into a single "ia" phone.
    syl = _convert("m i a̯ ˧")[0]
    assert syl.vowel == "ia" and syl.coda is None

def test_convert_strips_affricate_tie_bar():
    # ช้าง -> aspirated affricate onset, tie bar stripped to match ipa.py's
    # "tɕʰ" spelling.
    syl = _convert("t͡ɕʰ aː ŋ ˦˥")[0]
    assert syl.onset == "tɕʰ" and syl.coda == "ŋ" and syl.tone == Tone.HIGH

def test_convert_glottal_onset_and_coda():
    # จะ -> unaspirated affricate onset, explicit glottal-stop coda
    syl = _convert("t͡ɕ a ʔ ˨˩")[0]
    assert syl.onset == "tɕ" and syl.coda == "ʔ" and syl.tone == Tone.LOW

def test_convert_vowel_initial_syllable():
    syl = _convert("ʔ aː ˧")[0]
    assert syl.onset == "ʔ" and syl.long is True

def test_convert_unparseable_cluster_returns_none():
    # consonant-cluster onsets aren't representable; must not raise.
    assert _convert("k l uə ˧") is None

def test_convert_empty_returns_none():
    assert _convert("") is None


# --- tltk_ipa-shaped (compact, digit tone) ---------------------------------

def test_convert_compact_multisyllable_word():
    # pythainlp's own docstring: transliterate("สามารถ", engine="tltk_ipa")
    # == 'saː5.maːt3'. สามารถ is independently known (from the thaig2p
    # example above) to be rising+falling; digit 5=rising, 3=falling
    # confirms the derived digit->tone map (see module docstring).
    syls = _convert_compact("saː5.maːt3")
    assert len(syls) == 2
    a, b = syls
    assert (a.onset, a.vowel, a.long, a.coda, a.tone) == (
        "s", "a", True, None, Tone.RISING)
    assert (b.onset, b.vowel, b.long, b.coda, b.tone) == (
        "m", "a", True, "t", Tone.FALLING)

def test_convert_compact_all_tone_digits():
    assert _convert_compact("ma1")[0].tone == Tone.MID
    assert _convert_compact("ma2")[0].tone == Tone.LOW
    assert _convert_compact("ma3")[0].tone == Tone.FALLING
    assert _convert_compact("ma4")[0].tone == Tone.HIGH
    assert _convert_compact("ma5")[0].tone == Tone.RISING

def test_convert_compact_affricate_remap():
    # Synthetic: not an observed tltk output, but a direct exercise of the
    # 'c'/'cʰ' -> 'tɕ'/'tɕʰ' remap, grounded in tltk.nlp's own stable['X']
    # consonant table (จ -> 'c', ช/ฉ/ฌ -> 'ch' -> 'cʰ' post-aspiration-pass).
    assert _convert_compact("ca1")[0].onset == "tɕ"
    assert _convert_compact("cʰa1")[0].onset == "tɕʰ"

def test_convert_compact_no_digit_tone_returns_none():
    assert _convert_compact("maa") is None

def test_convert_compact_unparseable_returns_none():
    assert _convert_compact("xyz9") is None

def test_convert_compact_empty_returns_none():
    assert _convert_compact("") is None
