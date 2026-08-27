import re
from dataclasses import dataclass
from .tone import Tone

class IpaParseError(ValueError):
    pass

_TONES = {"˧": Tone.MID, "˨˩˦": Tone.RISING, "˨˩": Tone.LOW,
          "˥˩": Tone.FALLING, "˦˥": Tone.HIGH}
_ONSETS = ["tɕʰ", "tɕ", "pʰ", "tʰ", "kʰ", "b", "d", "p", "t", "k", "ʔ",
           "m", "n", "ŋ", "f", "s", "h", "w", "l", "j", "r"]
_VOWELS = ["ɯa", "ia", "ua", "ɯ", "ɤ", "ɛ", "ɔ", "i", "e", "a", "o", "u"]
_CODAS = ["p", "t", "k", "ʔ", "m", "n", "ŋ", "j", "w"]

@dataclass
class IpaSyllable:
    onset: str
    vowel: str
    long: bool
    coda: str | None
    tone: Tone

def _take(s: str, options: list[str]) -> tuple[str | None, str]:
    for o in options:
        if s.startswith(o):
            return o, s[len(o):]
    return None, s

def _parse_one(s: str) -> IpaSyllable:
    tone = None
    for mark in sorted(_TONES, key=len, reverse=True):
        if s.endswith(mark):
            tone, s = _TONES[mark], s.removesuffix(mark)
            break
    if tone is None:
        raise IpaParseError(f"no tone letters in {s!r}")
    onset, s = _take(s, _ONSETS)
    if onset is None:
        raise IpaParseError(f"unknown onset in {s!r}")
    vowel, s = _take(s, _VOWELS)
    if vowel is None:
        raise IpaParseError(f"unknown vowel in {s!r}")
    long = s.startswith("ː")
    s = s.removeprefix("ː")
    coda, s = _take(s, _CODAS)
    if s:
        raise IpaParseError(f"trailing {s!r}")
    return IpaSyllable(onset, vowel, long, coda, tone)

def parse_ipa(s: str) -> list[IpaSyllable]:
    parts = [p for p in re.split(r"[.\s]+", s.strip()) if p]
    if not parts:
        raise IpaParseError("empty")
    return [_parse_one(p) for p in parts]

def diff_features(a: IpaSyllable, b: IpaSyllable) -> set[str]:
    diffs: set[str] = set()
    if a.onset != b.onset:
        bare = {a.onset.replace("ʰ", ""), b.onset.replace("ʰ", "")}
        diffs.add("aspiration" if len(bare) == 1 else "onset")
    if a.vowel != b.vowel:
        diffs.add("vowel")
    if a.long != b.long:
        diffs.add("length")
    if a.coda != b.coda:
        diffs.add("coda")
    if a.tone != b.tone:
        diffs.add("tone")
    return diffs
