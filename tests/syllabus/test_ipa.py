"""ipa.render (spec 4 section 1): a Pronunciation renders as onset, vowel
(plus length mark when long), coda, and Chao tone letters per syllable,
joined with ".".
"""
import pytest

from thai_syllabus.entities import Pronunciation, Syllable
from thai_syllabus.ipa import render


def _syl(onset: str, vowel: str, coda: str, *, length: str = "short",
        tone: str = "mid") -> Syllable:
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def _pron(*syllables: Syllable) -> Pronunciation:
    return Pronunciation(syllables=tuple(syllables), corroboration="engines_agree")


def test_ipa_renders_tone_and_length():
    # "khaːw˥˩" -- long falling "khaaw" (rice): the vowel is spelled once,
    # with the length mark, not doubled.
    assert render(_pron(_syl("kh", "a", "w", length="long", tone="falling"))) == "khaːw˥˩"


def test_ipa_short_vowel_has_no_length_mark():
    assert render(_pron(_syl("m", "a", "", length="short", tone="mid"))) == "ma˧"


def test_ipa_joins_syllables_with_a_dot():
    assert render(_pron(_syl("p", "o", "m", tone="mid"),
                        _syl("k", "a", "t", tone="low"))) == "pom˧.kat˨˩"


@pytest.mark.parametrize("tone,letters", [
    ("mid", "˧"),
    ("low", "˨˩"),
    ("falling", "˥˩"),
    ("high", "˦˥"),
    ("rising", "˨˩˦"),
])
def test_ipa_tone_letters_match_the_deck_convention(tone, letters):
    assert render(_pron(_syl("k", "a", "", tone=tone))) == f"ka{letters}"
