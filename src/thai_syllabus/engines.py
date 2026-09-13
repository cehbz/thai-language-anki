"""The two pronunciation engines (spec 3 r28 section 5): pythainlp's
thaig2p converted into `Syllable`s, and the deterministic tone-rule
engine. Both are pure string work apart from the one pythainlp call,
whose heavy import (pythainlp/torch) stays inside `Thaig2p.__init__` so
that importing this module -- or the run that wires it up -- never pulls
it in; constructing a `Thaig2p` is what needs the "nlp" extra.

Observed thaig2p output format (pythainlp 5.3.7, engine="thaig2p", via
`transliterate(word, engine="thaig2p")`), e.g.:

    ขาว   -> 'kʰ aː w ˩˩˦'
    ข่าว  -> 'kʰ aː w ˨˩'
    ข้าว  -> 'kʰ a w ˥˩'
    ไก่   -> 'k a j ˨˩'
    มา    -> 'm aː ˧'
    สามารถ -> 's aː ˩˩˦ . m aː t̚ ˥˩'
    เมีย  -> 'm i a̯ ˧'
    ช้าง  -> 't͡ɕʰ aː ŋ ˦˥'
    จะ    -> 't͡ɕ a ʔ ˨˩'
    อา    -> 'ʔ aː ˧'

Shape of the raw string:
  * Syllables are separated by " . " (a literal "." surrounded by spaces).
  * Within a syllable, EVERY phone is its own whitespace-separated token.
    The final token of a syllable is always its tone, rendered as Chao
    tone-letters (not digits).
  * Tone letters observed: "˧" (mid), "˩˩˦" (rising), "˨˩" (low),
    "˥˩" (falling), "˦˥" (high). Note thaig2p's rising is "˩˩˦" while
    ipa.py renders rising as "˨˩˦"; the tone reaches Syllable as a name,
    so the two spellings never meet. Length (ː, U+02D0) is embedded in
    the vowel's own token (e.g. "aː") and becomes `vowel_length`.
  * Stop codas carry a "no audible release" diacritic (t̚/k̚/p̚,
    U+031A) that has no equivalent field on Syllable and is stripped.
  * Affricates carry a tie bar (t͡ɕ, U+0361) that the deck's segment
    inventory does not use (it spells them "tɕ"/"tɕʰ"); stripped.
  * The ia/ɯa/ua diphthongs are emitted as TWO tokens: the head vowel
    (i/ɯ/u) followed by a non-syllabic "a" carrying a COMBINING INVERTED
    BREVE BELOW (a̯, U+032F) marking it as an offglide, e.g. "i a̯" for
    the vowel of เมีย. These two tokens are merged into a single
    "ia"/"ɯa"/"ua" phone.
  * Vowel-initial syllables (อา, เอา) get an explicit "ʔ" onset token;
    dead syllables with no written final consonant (จะ) get an explicit
    "ʔ" coda token. Both are in the onset/coda inventories below, so no
    special-casing is required for them.
  * Consonant-cluster onsets (e.g. "กลัว" -> tokens "k", "l", ...) are a
    stop/fricative followed by r/l/w (kr kl kw, kʰr kʰl kʰw, pr pl, pʰr
    pʰl, tr, and loanword br bl dr fr fl); the two tokens merge into one
    onset string (e.g. "kl") when a vowel follows, since `Syllable`'s
    onset is a plain string with nothing downstream that splits it. When
    no vowel follows the r/l/w token (so it's actually a coda or the next
    syllable's initial), no merge happens and the syllable still fails to
    convert as before.

The tone-rule engine is the second opinion on a monosyllabic tone
disagreement. Sources: thai-language.com/ref/tone-rules;
thaiwithgrace.com/thai-tones (class split 9 mid / 11 high / 24 low).
"""
from __future__ import annotations

import re
from typing import Literal

from .entities import Syllable, Tone, VowelLength

# --- thaig2p's raw output -> Syllable ------------------------------------

_TONE_MAP: dict[str, Tone] = {
    "˧": "mid",
    "˩˩˦": "rising",
    "˨˩": "low",
    "˥˩": "falling",
    "˦˥": "high",
}

_TIE_BAR = "͡"       # combining double inverted breve, e.g. t͡ɕ
_NO_RELEASE = "̚"    # combining left angle above, e.g. k̚
_NONSYLLABIC = "̯"   # combining inverted breve below, e.g. a̯ (glide)
_LONG_MARK = "ː"

_ONSETS = frozenset({"tɕʰ", "tɕ", "pʰ", "tʰ", "kʰ", "b", "d", "p", "t", "k",
                     "ʔ", "m", "n", "ŋ", "f", "s", "h", "w", "l", "j", "r"})
_VOWELS = frozenset({"ɯa", "ia", "ua", "ɯ", "ɤ", "ɛ", "ɔ", "i", "e", "a",
                     "o", "u"})
_CODAS = frozenset({"p", "t", "k", "ʔ", "m", "n", "ŋ", "j", "w"})
_DIPHTHONG_HEADS = frozenset({"i", "ɯ", "u"})
_CLUSTER_SEMIVOWELS = frozenset({"r", "l", "w"})

_SYLLABLE_SEP = re.compile(r"\s*\.\s*")


class _ConvertError(Exception):
    """Internal: a syllable couldn't be mapped. Never escapes _convert."""


def _strip_marks(tok: str) -> str:
    return tok.replace(_TIE_BAR, "").replace(_NO_RELEASE, "")


def _merge_phones(tokens: list[str]) -> list[str]:
    """Join a diphthong head (i/ɯ/u) with a following non-syllabic glide
    token into a single ia/ɯa/ua phone; pass everything else through."""
    cleaned = [_strip_marks(t) for t in tokens]
    merged: list[str] = []
    i = 0
    while i < len(cleaned):
        tok = cleaned[i]
        nxt = cleaned[i + 1] if i + 1 < len(cleaned) else None
        if tok in _DIPHTHONG_HEADS and nxt is not None and nxt.endswith(_NONSYLLABIC):
            merged.append(tok + nxt.rstrip(_NONSYLLABIC))
            i += 2
        else:
            merged.append(tok.rstrip(_NONSYLLABIC))
            i += 1
    return merged


def _convert_syllable(group: str) -> Syllable:
    tokens = group.split()
    if len(tokens) < 3:
        raise _ConvertError(f"too few phones in {group!r}")
    *phone_toks, tone_tok = tokens
    tone = _TONE_MAP.get(tone_tok)
    if tone is None:
        raise _ConvertError(f"unknown tone letters {tone_tok!r}")

    phones = _merge_phones(phone_toks)
    if len(phones) < 2:
        raise _ConvertError(f"missing onset/vowel in {group!r}")
    onset, *rest = phones

    if onset not in _ONSETS:
        raise _ConvertError(f"unknown onset {onset!r}")

    # A Thai initial cluster (stop/fricative + r/l/w) arrives as two phone
    # tokens; merge them into one onset string, but only when a vowel
    # follows the r/l/w token -- otherwise it's a coda (w/j) or the next
    # syllable's initial, not part of this onset.
    if (rest and rest[0] in _CLUSTER_SEMIVOWELS and len(rest) > 1
            and rest[1].removesuffix(_LONG_MARK) in _VOWELS):
        onset += rest[0]
        rest = rest[1:]

    if not rest:
        raise _ConvertError(f"missing vowel in {group!r}")
    vowel_tok, *coda_toks = rest

    length: VowelLength = "long" if vowel_tok.endswith(_LONG_MARK) else "short"
    vowel = vowel_tok.removesuffix(_LONG_MARK)
    if vowel not in _VOWELS:
        raise _ConvertError(f"unknown vowel {vowel!r}")

    if len(coda_toks) > 1:
        raise _ConvertError(f"unsupported coda cluster {coda_toks!r}")
    coda = coda_toks[0] if coda_toks else ""
    if coda and coda not in _CODAS:
        raise _ConvertError(f"unknown coda {coda!r}")

    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def _convert(raw: str) -> tuple[Syllable, ...] | None:
    """Convert thaig2p's raw string to Syllables. Never raises: returns
    None for anything unmappable (an open syllable's coda is "")."""
    groups = [g for g in _SYLLABLE_SEP.split(raw.strip()) if g.strip()]
    if not groups:
        return None
    try:
        return tuple(_convert_syllable(g) for g in groups)
    except _ConvertError:
        return None


class Thaig2p:
    """pythainlp's thaig2p as an `Engines.g2p` callable. The pythainlp
    import (which pulls torch) happens when one is constructed, not at
    module import, so constructing it is what fails when the "nlp" extra
    isn't installed."""

    def __init__(self) -> None:
        from pythainlp.transliterate import transliterate
        self._transliterate = transliterate

    def syllables(self, word: str) -> tuple[Syllable, ...] | None:
        """Never raises: None for a word thaig2p can't produce a mappable
        analysis for (a consonant-cluster onset, say)."""
        try:
            return _convert(self._transliterate(word, engine="thaig2p"))
        except Exception:
            return None

    __call__ = syllables


# --- the deterministic tone rules: consonant class x live/dead x mark -----

ConsClass = Literal["mid", "high", "low"]

_MID = "กจฎฏดตบปอ"
_HIGH = "ขฃฉฐถผฝศษสห"
_LOW = "คฅฆงชซฌญฑฒณทธนพฟภมยรลวฬฮ"
CONSONANT_CLASS: dict[str, ConsClass] = (
    {c: "mid" for c in _MID}
    | {c: "high" for c in _HIGH}
    | {c: "low" for c in _LOW})

MAI_EK, MAI_THO, MAI_TRI, MAI_CHATTAWA = "่", "้", "๊", "๋"
_MARKS = {MAI_EK, MAI_THO, MAI_TRI, MAI_CHATTAWA}

_SONORANT_FINALS = set("งนมณญยรลฬว")
_STOP_FINALS = set("กขคฆจชซฌฎฏฐฑฒดตถทธบปพฟภศษส")
_LOW_SONORANTS = set("งญณนมยรลวฬ")


def tone_of(cls: ConsClass, live: bool, long_vowel: bool, mark: str | None) -> Tone:
    if mark == MAI_EK:
        return "falling" if cls == "low" else "low"
    if mark == MAI_THO:
        return "high" if cls == "low" else "falling"
    if mark == MAI_TRI:
        return "high"
    if mark == MAI_CHATTAWA:
        return "rising"
    if live:
        return "rising" if cls == "high" else "mid"
    if cls == "low":
        return "falling" if long_vowel else "high"
    return "low"


# (vowel name, long?) — pre-vowels sit before the initial in writing order;
# combining vowels/marks are stripped before matching.
_PRE_VOWELS = {"เ": ("e", True), "แ": ("ɛ", True), "โ": ("o", True),
               "ไ": ("aj", False), "ใ": ("aj", False)}
_POST_LONG = {"า": ("a", True)}
_ABOVE_BELOW = {"ิ": ("i", False), "ี": ("i", True),
                "ึ": ("ɯ", False), "ื": ("ɯ", True),
                "ุ": ("u", False), "ู": ("u", True),
                "ั": ("a", False)}   # ◌ั
_SARA_A = "ะ"


def rule_tone(word: str) -> Tone | None:
    """The written syllable's tone by rule, or None when the word isn't a
    single syllable this analysis covers (multi-syllable words, complex
    vowel forms, unrecognised finals)."""
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
            initial, cls = chars.pop(0)[1], "high"
        elif initial == "อ" and nxt == "ย":
            initial, cls = chars.pop(0)[1], "mid"
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
    return tone_of(cls, live, bool(long_v), mark)
