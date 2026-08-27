"""Deterministic Thai tone rules: consonant class x live/dead x mark.

Sources: thai-language.com/ref/tone-rules; thaiwithgrace.com/thai-tones
(class split 9 mid / 11 high / 24 low).
"""
from dataclasses import dataclass
from enum import StrEnum


class Tone(StrEnum):
    MID = "mid"
    LOW = "low"
    FALLING = "falling"
    HIGH = "high"
    RISING = "rising"


class ConsClass(StrEnum):
    MID = "mid"
    HIGH = "high"
    LOW = "low"


_MID = "กจฎฏดตบปอ"
_HIGH = "ขฃฉฐถผฝศษสห"
_LOW = "คฅฆงชซฌญฑฒณทธนพฟภมยรลวฬฮ"
CONSONANT_CLASS: dict[str, ConsClass] = (
    {c: ConsClass.MID for c in _MID}
    | {c: ConsClass.HIGH for c in _HIGH}
    | {c: ConsClass.LOW for c in _LOW})

MAI_EK, MAI_THO, MAI_TRI, MAI_CHATTAWA = "่", "้", "๊", "๋"
_MARKS = {MAI_EK, MAI_THO, MAI_TRI, MAI_CHATTAWA}

_SONORANT_FINALS = set("งนมณญยรลฬว")
_STOP_FINALS = set("กขคฆจชซฌฎฏฐฑฒดตถทธบปพฟภศษส")
_LOW_SONORANTS = set("งญณนมยรลวฬ")


def tone_of(cls: ConsClass, live: bool, long_vowel: bool, mark: str | None) -> Tone:
    if mark == MAI_EK:
        return Tone.FALLING if cls == ConsClass.LOW else Tone.LOW
    if mark == MAI_THO:
        return Tone.HIGH if cls == ConsClass.LOW else Tone.FALLING
    if mark == MAI_TRI:
        return Tone.HIGH
    if mark == MAI_CHATTAWA:
        return Tone.RISING
    if live:
        return Tone.RISING if cls == ConsClass.HIGH else Tone.MID
    if cls == ConsClass.LOW:
        return Tone.FALLING if long_vowel else Tone.HIGH
    return Tone.LOW


@dataclass
class SyllableAnalysis:
    initial: str
    cls: ConsClass
    vowel: str
    long_vowel: bool
    final: str | None
    live: bool
    mark: str | None
    tone: Tone


# (template, vowel name, long?, allows_final?) — pre-vowels use "-" for the
# initial slot; combining vowels/marks are stripped before template matching.
_PRE_VOWELS = {"เ": ("e", True), "แ": ("ɛ", True), "โ": ("o", True),
               "ไ": ("aj", False), "ใ": ("aj", False)}
_POST_LONG = {"า": ("a", True)}
_ABOVE_BELOW = {"ิ": ("i", False), "ี": ("i", True),
                "ึ": ("ɯ", False), "ื": ("ɯ", True),
                "ุ": ("u", False), "ู": ("u", True),
                "ั": ("a", False)}   # ◌ั
_SARA_A = "ะ"


def analyze_syllable(word: str) -> SyllableAnalysis | None:
    # (original_index, char) pairs — the index lets us verify that a leftover
    # consonant actually sits *after* the vowel in writing order before we
    # accept it as a final. Without this, a CCV two-syllable word's second
    # initial (e.g. the น in ธ-น-า) would be misread as a final.
    raw = [c for c in word if c not in _MARKS]
    mark = next((c for c in word if c in _MARKS), None)
    chars = list(enumerate(raw))

    pre = None
    if chars and chars[0][1] in _PRE_VOWELS:
        pre = chars.pop(0)[1]

    if not chars or chars[0][1] not in CONSONANT_CLASS:
        return None
    initial = chars.pop(0)[1]
    cls = CONSONANT_CLASS[initial]
    # ห นำ / อ นำ: leading silent ห (or อ) + low sonorant → leader's class
    if chars and chars[0][1] in CONSONANT_CLASS:
        nxt = chars[0][1]
        if initial == "ห" and nxt in _LOW_SONORANTS:
            initial, cls = chars.pop(0)[1], ConsClass.HIGH
        elif initial == "อ" and nxt == "ย":
            initial, cls = chars.pop(0)[1], ConsClass.MID
        elif pre is None and nxt not in _SONORANT_FINALS | _STOP_FINALS:
            return None

    vowel = long_v = vowel_idx = None
    if pre is not None:
        if any(c in _ABOVE_BELOW or c == _SARA_A or c == "า" for _, c in chars):
            return None  # complex เ-ือ / เ-าะ / เ-ีย forms: out of scope
        vowel, long_v = _PRE_VOWELS[pre]
    else:
        for idx, c in list(chars):
            if c in _ABOVE_BELOW:
                vowel, long_v = _ABOVE_BELOW[c]
                chars.remove((idx, c))
                vowel_idx = idx
                break
            if c in _POST_LONG:
                vowel, long_v = _POST_LONG[c]
                chars.remove((idx, c))
                vowel_idx = idx
                break
            if c == _SARA_A:
                vowel, long_v = "a", False
                chars.remove((idx, c))
                vowel_idx = idx
                break
    if vowel is None:
        return None

    final = None
    if chars:
        if len(chars) > 1 or chars[0][1] not in _SONORANT_FINALS | _STOP_FINALS:
            return None
        final_idx, final = chars[0]
        if pre is None and final_idx < vowel_idx:
            return None  # leftover consonant precedes the vowel: it's the
            # next syllable's initial (CCV), not this syllable's final

    if pre in ("ไ", "ใ"):
        live = True          # -aj diphthong behaves live
    elif final is None:
        live = long_v
    else:
        live = final in _SONORANT_FINALS
    return SyllableAnalysis(initial, cls, vowel, bool(long_v), final, live,
                            mark, tone_of(cls, live, bool(long_v), mark))
