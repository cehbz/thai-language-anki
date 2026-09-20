"""Renders a Pronunciation as the deck's authored IPA string (spec 4
section 1: "Ipa renders the Pronunciation value with tone and length").

Per syllable: onset, then vowel (followed by the length mark when
vowel_length is long and the vowel is not a centering diphthong -- the
vowel itself is spelled once, not doubled), then coda, then the Chao tone
letters; syllables join with ".".
"""
from .entities import DIPHTHONGS, Pronunciation, Syllable, Tone

_TONE_LETTERS: dict[Tone, str] = {
    "mid": "˧",
    "low": "˨˩",
    "falling": "˥˩",
    "high": "˦˥",
    "rising": "˨˩˦",
}

LENGTH_MARK = "ː"


def render(pron: Pronunciation) -> str:
    """`onset + vowel [+ length mark] + coda + tone letters` per syllable,
    joined with ".".
    """
    return ".".join(_render_syllable(s) for s in pron.syllables)


def _render_syllable(syllable: Syllable) -> str:
    # A centering diphthong is long by convention and written unmarked
    # (design 2026-09-20 §1).
    long_mark = (syllable.vowel_length == "long" and syllable.vowel not in DIPHTHONGS)
    length_mark = LENGTH_MARK if long_mark else ""
    return (syllable.onset + syllable.vowel + length_mark + syllable.coda
           + _TONE_LETTERS[syllable.tone])
