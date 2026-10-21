"""A reading rendered in Paiboon romanization, the system of
thai-language.com's own Paiboon strings (research: thai-nlp-tooling
"Romanization"): unaspirated stops g, bp, dt and aspirated k, p, t; long
vowels doubled; the tone marked on the syllable's first vowel letter (mid
none, low grave, falling circumflex, high acute, rising caron); the glottal
coda dropped; syllables joined by hyphens.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from .entities import DIPHTHONGS, Syllable, Tone

_ONSETS = {"k": "g", "kʰ": "k", "p": "bp", "pʰ": "p", "t": "dt", "tʰ": "t",
           "tɕ": "j", "tɕʰ": "ch", "ʔ": "", "ŋ": "ng", "j": "y"}
_VOWELS = {"ɯ": "ʉ", "ɤ": "ə", "ɯa": "ʉa"}
_CODAS = {"ŋ": "ng", "ʔ": "", "j": "i", "w": "o"}
_TONE_MARKS: dict[Tone, str] = {"mid": "", "low": "̀", "falling": "̂",
                                "high": "́", "rising": "̌"}


def _onset(onset: str) -> str:
    # A cluster is a stop plus r, l or w, each kept as written.
    if len(onset) > 1 and onset[-1] in "rlw":
        return _onset(onset[:-1]) + onset[-1]
    return _ONSETS.get(onset, onset)


def _syllable(s: Syllable) -> str:
    onset, vowel, coda = s.segments
    v = _VOWELS.get(vowel, vowel)
    if s.vowel_length == "long" and vowel not in DIPHTHONGS:
        v += v
    v = v[0] + _TONE_MARKS[s.tone] + v[1:]
    # w after i is written u (หิว hǐu), o elsewhere (ข้าว kâao).
    c = "u" if coda == "w" and vowel == "i" else _CODAS.get(coda, coda)
    return _onset(onset) + v + c


def render(syllables: Sequence[Syllable]) -> str:
    """`syllables` in Paiboon, hyphen-joined, NFC."""
    return unicodedata.normalize("NFC", "-".join(_syllable(s) for s in syllables))
