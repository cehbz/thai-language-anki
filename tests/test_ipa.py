import pytest
from thai_deck_eval.lang.ipa import IpaParseError, diff_features, parse_ipa
from thai_deck_eval.lang.tone import Tone

def test_parse_single_syllable():
    (s,) = parse_ipa("kʰaːw˥˩")
    assert (s.onset, s.vowel, s.long, s.coda, s.tone) == (
        "kʰ", "a", True, "w", Tone.FALLING)

def test_parse_no_coda_short():
    (s,) = parse_ipa("tɕa˨˩")
    assert (s.onset, s.vowel, s.long, s.coda, s.tone) == ("tɕ", "a", False, None, Tone.LOW)

def test_parse_multisyllable():
    syls = parse_ipa("maː˧.kʰaj˨˩")
    assert len(syls) == 2 and syls[1].tone == Tone.LOW

def test_parse_error():
    with pytest.raises(IpaParseError):
        parse_ipa("hello")

def test_diff_tone_only():
    a, b = parse_ipa("kʰaːw˨˩˦")[0], parse_ipa("kʰaːw˨˩")[0]
    assert diff_features(a, b) == {"tone"}

def test_diff_aspiration():
    a, b = parse_ipa("kaj˨˩")[0], parse_ipa("kʰaj˨˩")[0]
    assert diff_features(a, b) == {"aspiration"}

def test_diff_length_and_tone():
    a, b = parse_ipa("kʰaːw˥˩")[0], parse_ipa("kʰaw˨˩")[0]
    assert diff_features(a, b) == {"length", "tone"}

def test_render_ipa_round_trips():
    from thai_deck_eval.lang.ipa import parse_ipa, render_ipa
    for s in ["kʰaːw˥˩", "naːm˦˥", "ma˧.naːw˧", "kin˧"]:
        assert render_ipa(parse_ipa(s)) == s
