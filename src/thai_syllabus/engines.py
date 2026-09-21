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
    the vowel of เมีย. These two tokens are merged into a single ia/ɯa/ua
    phone, long.
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

from .entities import DIPHTHONGS, Syllable, Tone, VowelLength, without_glottal_coda

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

    vowel = vowel_tok.removesuffix(_LONG_MARK)
    length: VowelLength = ("long" if vowel_tok.endswith(_LONG_MARK) or vowel in DIPHTHONGS
                           else "short")
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
        return without_glottal_coda(tuple(_convert_syllable(g) for g in groups))
    except _ConvertError:
        return None


# --- tltk's raw output -> Syllable (design 2026-09-18 §4) ------------------
#
# Observed output of tltk.nlp.th2ipa (tltk as installed 2026-09-18):
#
#     มา        -> 'maː1 <s/>'
#     ไก่       -> 'kaj2 <s/>'
#     ข้าว      -> 'kʰaːw3 <s/>'
#     ช้าง      -> 'cʰaːŋ4 <s/>'
#     ขาว       -> 'kʰaːw5 <s/>'
#     ภาพวาด    -> 'pʰaːp3.waːt3 <s/>'
#     ข้างๆ     -> 'kʰaːŋ3 kʰaːŋ3 <s/>'
#
# Unlike thaig2p, a syllable is one compact string with no separators
# between its phones, so the inventories are matched longest-first.

_TLTK_TONES: dict[str, Tone] = {"1": "mid", "2": "low", "3": "falling",
                                "4": "high", "5": "rising"}

# tltk's spellings that differ from the deck's segment inventory. Applied
# longest-first, so cʰ is read before c.
_TLTK_SUBSTITUTIONS = (("ᴐ", "ɔ"), ("cʰ", "tɕʰ"), ("c", "tɕ"))

_TLTK_MARKUP = re.compile(r"<[^>]*>")
_TLTK_SYLLABLE_SEP = re.compile(r"[.\s~|]+")

# tltk marks a syllable it could not read as an EMPTY SYLLABLE SLOT: a "."
# with nothing (or only more separator whitespace) on one side of it, where
# a real syllable's phones would otherwise sit. Two dots in a row (with or
# without whitespace between them) is one empty slot between two real
# syllables (สิบเอ็ด -> "cʰaːn1..muk4"-shaped: 5 syllables truncated to 2,
# the middle three collapsed to nothing); a leading or trailing dot is an
# empty slot at that edge (สิบเอ็ด -> "si2." itself: 2 syllables, got 1).
# `_TLTK_SYLLABLE_SEP`'s "+" quantifier collapses a run of separators into
# one, which would otherwise hide exactly this signal. A lone space is
# never this signal -- "ข้างๆ" -> "kʰaːŋ3 kʰaːŋ3" is two ordinary
# syllables, not a slot -- so only "." is checked here, and only after
# `_TLTK_MARKUP` and the outer `.strip()` have already removed the markup
# and edge whitespace that would otherwise make a real dot look edge-most.
_TLTK_EMPTY_SLOT = re.compile(r"\.\s*\.")


def _tltk_has_empty_slot(cleaned: str) -> bool:
    return bool(cleaned) and (
        cleaned[0] == "." or cleaned[-1] == "."
        or _TLTK_EMPTY_SLOT.search(cleaned) is not None)

# longest-first so "tɕʰ" wins over "tɕ" and "ɯa" over "ɯ"
_ONSETS_LONGEST_FIRST = tuple(sorted(_ONSETS, key=len, reverse=True))
_VOWELS_LONGEST_FIRST = tuple(sorted(_VOWELS, key=len, reverse=True))
_CODAS_LONGEST_FIRST = tuple(sorted(_CODAS, key=len, reverse=True))


def _take(s: str, options: tuple[str, ...]) -> tuple[str | None, str]:
    for o in options:
        if s.startswith(o):
            return o, s[len(o):]
    return None, s


def _convert_compact_syllable(body: str, tone: Tone, group: str) -> Syllable:
    """One compact syllable string (no separators between phones), matched
    longest-first: onset [cluster] vowel [ː] [coda]. `group` is the raw
    text for error messages only. Shared by the tltk and Wiktionary
    converters, whose notations meet here once each has mapped its own
    tone marks and spellings."""
    s = body
    onset, s = _take(s, _ONSETS_LONGEST_FIRST)
    if onset is None:
        raise _ConvertError(f"unknown onset in {group!r}")
    # A Thai initial cluster (stop/fricative + r/l/w) is one onset, the
    # same merge _convert makes for thaig2p -- but only when a vowel
    # follows, so the w of kʰaːw stays a coda.
    if s[:1] in _CLUSTER_SEMIVOWELS and _take(s[1:], _VOWELS_LONGEST_FIRST)[0] is not None:
        onset, s = onset + s[0], s[1:]

    vowel, s = _take(s, _VOWELS_LONGEST_FIRST)
    if vowel is None:
        raise _ConvertError(f"unknown vowel in {group!r}")
    length: VowelLength = "long" if s.startswith(_LONG_MARK) else "short"
    s = s.removeprefix(_LONG_MARK)
    # tltk writes the diphthongs with the length mark inside (iːa); the
    # deck spells them ia/ɯa/ua, long (design 2026-09-20 §1).
    if s.startswith("a") and vowel in _DIPHTHONG_HEADS:
        vowel, s = vowel + "a", s[1:]
    if vowel in DIPHTHONGS:
        length = "long"

    coda, s = _take(s, _CODAS_LONGEST_FIRST)
    if s:
        raise _ConvertError(f"trailing {s!r} in {group!r}")
    return Syllable(segments=(onset, vowel, coda or ""), vowel_length=length, tone=tone)


def _convert_tltk_syllable(group: str) -> Syllable:
    tone = _TLTK_TONES.get(group[-1:])
    if tone is None:
        raise _ConvertError(f"no tone digit in {group!r}")
    s = group[:-1]
    for old, new in _TLTK_SUBSTITUTIONS:
        s = s.replace(old, new)
    return _convert_compact_syllable(s, tone, group)


def _convert_tltk(raw: str) -> tuple[Syllable, ...] | None:
    """Convert tltk's raw th2ipa string to Syllables. Never raises:
    returns None for anything unmappable, including a reading tltk itself
    marked incomplete with an empty syllable slot (see
    `_tltk_has_empty_slot`) -- a truncated reading is refused rather than
    returned short, the same contract as any other unmappable input."""
    cleaned = _TLTK_MARKUP.sub(" ", raw).strip()
    if _tltk_has_empty_slot(cleaned):
        return None
    groups = [g for g in _TLTK_SYLLABLE_SEP.split(cleaned) if g]
    if not groups:
        return None
    try:
        return without_glottal_coda(tuple(_convert_tltk_syllable(g) for g in groups))
    except _ConvertError:
        return None


# --- Wiktionary's rendered IPA -> Syllable (design 2026-09-20 §2) ----------
#
# Observed on en.wiktionary.org 2026-09-20 (Template:th-pron):
#
#     มกรา     -> /ma˦˥.ka˨˩.raː˧/ and /mok̚˦˥.ka˨˩.raː˧/ (two readings)
#     กล้วย    -> /klua̯j˥˩/
#     สิบเอ็ด  -> /sip̚˨˩.ʔet̚˨˩/
#     จะ       -> /t͡ɕaʔ˨˩/
#     หุง      -> /huŋ˩˩˦/
#
# Syllables separated by "."; tone as Chao letters at the end of the
# syllable (the same five strings thaig2p uses); affricates with a tie
# bar; unreleased stops with the no-release mark; the diphthong offglide
# with the non-syllabic mark; a ʔ coda on a dead open syllable.

_WIKTIONARY_TONE_LETTERS = "˥˦˧˨˩"
_WIKTIONARY_SYLLABLE_SEP = re.compile(r"\.")


def _convert_wiktionary_syllable(group: str) -> Syllable:
    body = group.rstrip(_WIKTIONARY_TONE_LETTERS)
    tone_letters = group[len(body):]
    tone = _TONE_MAP.get(tone_letters)
    if tone is None:
        raise _ConvertError(f"unknown tone letters {tone_letters!r} in {group!r}")
    body = _strip_marks(body).replace(_NONSYLLABIC, "")
    return _convert_compact_syllable(body, tone, group)


def convert_wiktionary(ipa: str) -> tuple[Syllable, ...] | None:
    """Convert one Wiktionary IPA string (with or without the enclosing
    slashes) to Syllables. Never raises: None for anything unmappable,
    including a reading with an empty syllable slot -- the same refusal
    `_tltk_has_empty_slot` makes for tltk.

    An empty slot here is Wiktionary's own notation, not a defect:
    a compound-linking variant is written with a TRAILING "." (observed
    2026-09-21: น้ำ lists /naːm˦˥/, /naːm˦˥./ and /nam˦˥./; นรก lists
    /na˦˥.rok̚˦˥./ and /na˦˥.rok̚˦˥.ka˨˩./). The dot says the form links
    onward into a compound, so the string is not a whole word's reading;
    dropping the empty group would record it as one.
    """
    cleaned = ipa.strip().strip("/").strip()
    groups = _WIKTIONARY_SYLLABLE_SEP.split(cleaned)
    if any(not g for g in groups):
        return None
    try:
        return without_glottal_coda(tuple(_convert_wiktionary_syllable(g) for g in groups))
    except _ConvertError:
        return None


_WIKTIONARY_ROW = re.compile(r"<tr>.*?</tr>", re.DOTALL)
_WIKTIONARY_IPA_SPAN = re.compile(r'<span class="IPA">/([^/<]+)/</span>')
# The rendered page's language headings: `<h2 id="Thai">`, inside a
# `<div class="mw-heading mw-heading2">` wrapper in today's markup.
_WIKTIONARY_THAI_HEADING = re.compile(r'<h2 id="Thai"')
_WIKTIONARY_LANGUAGE_HEADING = re.compile(r"<h2")


def _thai_section(html: str) -> str:
    """The Thai language section of a rendered Wiktionary page -- from its
    `<h2 id="Thai">` heading to the next `<h2` (or the end). The whole
    page when it has no such heading, which is what a fragment (a
    fixture's bare table) is.
    """
    heading = _WIKTIONARY_THAI_HEADING.search(html)
    if heading is None:
        return html
    following = _WIKTIONARY_LANGUAGE_HEADING.search(html, heading.end())
    return html[heading.start():following.start()] if following else html[heading.start():]


def extract_standard_ipa(html: str) -> list[str]:
    """The IPA readings of the "(standard) IPA" row of a rendered Thai
    entry's pronunciation table, in page order, without the slashes; []
    when the page has no such row (an entry with no th-pron table, or a
    layout this does not recognize).

    One Wiktionary page is one spelling, not one language: น้ำ carries
    Northern Thai, Nyaw and Isan sections beside the Thai one, each with
    its own "(standard) IPA" row. Only the Thai section is read
    (`_thai_section`); a page with no language heading at all is read
    whole.
    """
    for row in _WIKTIONARY_ROW.findall(_thai_section(html)):
        head, _, _cells = row.partition("</th>")
        if ">standard<" in head and ">IPA<" in head:
            return _WIKTIONARY_IPA_SPAN.findall(row)
    return []


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


class Tltk:
    """tltk's rule-based g2p as an `Engines.g2p` callable, the second
    segmental oracle beside `Thaig2p` (design 2026-09-18). Rule-based, so
    it cannot produce the decoder loop thaig2p does on long compounds; it
    fails the other way instead, by truncating, which is why agreement
    between the two is what corroborates rather than either alone.

    tltk imports no torch, so this is cheap, but the import stays in
    __init__ for symmetry with Thaig2p and so that importing this module
    never needs the "nlp" extra.
    """

    def __init__(self) -> None:
        from tltk import nlp
        self._th2ipa = nlp.th2ipa

    def syllables(self, word: str) -> tuple[Syllable, ...] | None:
        """Never raises: None for a word tltk can't read. th2ipa itself
        raises ValueError on some two-token phrases (ธอ ธง, ฝอ ฝา)."""
        try:
            return _convert_tltk(self._th2ipa(word))
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
        # อ straight after the initial is never a genuine final consonant --
        # it is the -อ vowel (คอ, ขอ) -- so it must not trip the generic
        # "leftover consonant we don't understand" bail below.
        elif pre is None and nxt not in _SONORANT_FINALS | _STOP_FINALS and nxt != "อ":
            return None

    vowel = long_v = vowel_idx = None
    if pre is not None:
        if pre == "เ" and len(chars) == 1 and chars[0][1] == "า":
            vowel, long_v = "aw", False   # เ-า: live diphthong (เขา, เรา)
            chars.pop(0)
        elif pre == "เ" and len(chars) == 1 and chars[0][1] == "อ":
            vowel, long_v = "ɤ", True     # เ-อ: the อ completes the vowel,
            chars.pop(0)                  # it is not a final consonant
        elif any(c in _ABOVE_BELOW or c == _SARA_A or c == "า" for _, c in chars):
            return None  # complex เ-ือ / เ-าะ / เ-ีย forms: out of scope
        else:
            vowel, long_v = _PRE_VOWELS[pre]
    else:
        for idx, c in list(chars):
            if c in _ABOVE_BELOW:
                vowel, long_v = _ABOVE_BELOW[c]
                chars.remove((idx, c))
                vowel_idx = idx
                if c == "ื" and chars and chars[0] == (idx + 1, "อ"):
                    chars.pop(0)  # -ือ: the อ completes the vowel, not a final
                break
            if c in _POST_LONG:
                vowel, long_v = _POST_LONG[c]
                chars.remove((idx, c))
                vowel_idx = idx
                break
            if c == "อ":
                vowel, long_v = "ɔ", True   # -อ: the vowel, not a final
                chars.remove((idx, c))
                vowel_idx = idx
                break
            if c == "ำ":
                vowel, long_v = "am", False  # -ำ: /am/, inherently live
                chars.remove((idx, c))
                vowel_idx = idx
                break
            if c == _SARA_A:
                vowel, long_v = "a", False
                chars.remove((idx, c))
                vowel_idx = idx
                break
        if (vowel is None and len(chars) == 1
                and chars[0][1] in _SONORANT_FINALS | _STOP_FINALS):
            # implicit vowel: two bare consonants, unwritten short /o/
            # (คน, นก) -- live/dead follows the written final as usual.
            vowel, long_v, vowel_idx = "o", False, -1
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

    if pre in ("ไ", "ใ") or vowel in ("aw", "am"):
        live = True          # -aj/-aw diphthongs and -am are inherently live
    elif final is None:
        live = long_v
    else:
        live = final in _SONORANT_FINALS
    return tone_of(cls, live, bool(long_v), mark)
