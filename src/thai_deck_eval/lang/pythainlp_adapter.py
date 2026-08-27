"""pythainlp/tltk adapters. Heavy imports (pythainlp, tltk, torch) stay
inside methods so that importing this module — or the pipeline that may
wire it up — never pulls them in by default.

Observed thaig2p output format (pythainlp 5.3.7, engine="thaig2p", via
`transliterate(word, engine="thaig2p")`), recorded from Task 10 Step 1
discovery, e.g.:

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
  * Within a syllable, EVERY phone is its own whitespace-separated token
    (unlike this project's `ipa.parse_ipa`, which expects phones
    concatenated together within a syllable). The final token of a
    syllable is always its tone, rendered as Chao tone-letters (not
    digits).
  * Tone letters observed: "˧" (mid), "˩˩˦" (rising), "˨˩" (low),
    "˥˩" (falling), "˦˥" (high). Length (ː, U+02D0) is embedded in the
    vowel's own token (e.g. "aː"), matching parse_ipa's convention.
  * Stop codas carry a "no audible release" diacritic (t̚/k̚/p̚,
    U+031A) that has no equivalent field on IpaSyllable and is stripped.
  * Affricates carry a tie bar (t͡ɕ, U+0361) that ipa.py's onset
    inventory does not use (it spells them "tɕ"/"tɕʰ"); stripped.
  * The ia/ɯa/ua diphthongs are emitted as TWO tokens: the head vowel
    (i/ɯ/u) followed by a non-syllabic "a" carrying a
    COMBINING INVERTED BREVE BELOW (a̯, U+032F) marking it as an
    offglide, e.g. "i a̯" for the vowel of เมีย. These two tokens are
    merged into a single "ia"/"ɯa"/"ua" phone.
  * Vowel-initial syllables (อา, เอา) get an explicit "ʔ" onset token;
    dead syllables with no written final consonant (จะ) get an explicit
    "ʔ" coda token. Both already exist in ipa.py's onset/coda
    inventories, so no special-casing is required for them.
  * Consonant-cluster onsets (e.g. "กลัว" -> tokens "k", "l", ...) are
    NOT representable by IpaSyllable's single-onset field; such
    syllables fail to convert (and the whole word's `syllables()` call
    returns None) rather than raising.

pythainlp's own docstring for `transliterate(..., engine="tltk_ipa")`
shows a materially different raw shape: compact per-syllable strings
with no spaces between phones and a digit tone (e.g. "saː5.maːt3"), not
the token-per-phone/Chao-letter shape thaig2p emits. In this environment
tltk itself is broken (`import tltk` fails: `ModuleNotFoundError: No
module named 'pandas'`, a transitive dependency tltk needs but does not
declare/that is not part of this project's "nlp" extra), so `tltk_ipa`
could not be empirically exercised (run) here. TltkG2P instead routes
through `_convert_compact` below, whose digit-tone mapping and phone
alphabet were verified by reading tltk's own installed source
(`tltk/nlp.py`, present in this environment despite being unimportable)
rather than by guessing:

  * Digit->tone mapping. `tltk.nlp.th2ipa`'s `NORMALIZE_IPA` table applies
    the replacements `('4','5'), ('3','4'), ('2','3'), ('1','2'), ('0','1')`
    in that order (each on the *previous* step's output), which nets out to
    "every internal tone digit is incremented by one" (0->1, 1->2, 2->3,
    3->4, 4->5) since later replacements can't re-touch digits already
    written by earlier ones. `tltk.nlp.ToneAssign` shows the internal digit
    for a MID-class, LIVE syllable with no tone mark — the textbook "no
    mark -> mid tone" case — is `'0'`. Internal digits increment through
    the mid-class live-syllable branch in mai-ek/mai-tho/mai-tri/
    mai-chattawa order, i.e. `'1'`=low, `'2'`=falling, `'3'`=high,
    `'4'`=rising (standard Thai tone-mark rules for mid-class consonants).
    So internal->printed is 0->1, 1->2, 2->3, 3->4, 4->5, giving the
    printed-digit->tone map used below: 1=mid, 2=low, 3=falling, 4=high,
    5=rising. Cross-checked against pythainlp's own docstring example
    `transliterate("สามารถ", engine="tltk_ipa") == 'saː5.maːt3'`: สามารถ is
    independently known (from the thaig2p example above) to be
    rising+falling, and 5=rising, 3=falling — confirms the mapping.
  * Onset/vowel/coda alphabet. `th2ipa` runs two regex passes before its
    NORMALIZE_IPA character table: doubled vowel letters collapse to
    `<vowel>ː` (matching `parse_ipa`'s long-vowel convention), and a
    trailing `h` after p/t/k/c becomes the aspiration diacritic `ʰ`. The
    NORMALIZE_IPA table itself maps tltk's internal ASCII phone letters to
    the same Unicode IPA characters `ipa.py` already uses (`O`->ɔ-like,
    `x`->ɛ, `@`->ɤ, `N`->ŋ, `?`->ʔ, `U`->ɯ), so after normalization most
    onsets/vowels/codas already match `ipa.py`'s `_ONSETS`/`_VOWELS`/
    `_CODAS` inventories directly. The one gap: `tltk.nlp`'s own
    `stable['X']` consonant table maps จ (unaspirated affricate) to ASCII
    `'c'` and ฉ/ช/ฌ (aspirated) to `'ch'` (-> `'cʰ'` after the aspiration
    regex), neither of which is in `ipa.py`'s onset set (which spells the
    affricate `"tɕ"`/`"tɕʰ"` per the thaig2p convention above); the
    compact-format converter remaps `c`/`cʰ` -> `tɕ`/`tɕʰ` before parsing.
    Consonant-cluster onsets are unrepresented in this alphabet's design
    and, like the thaig2p path, simply fail to convert (return None)
    rather than raising.

Because tltk cannot actually run in this environment, this mapping is
verified against source + one worked example, not against a broad live
sample; `test_tltk_g2p_does_not_raise` (integration) only checks the "never
raise" contract, not conversion correctness, for that reason.
"""
import re

from .ipa import _CODAS, _ONSETS, _VOWELS, IpaParseError, IpaSyllable, parse_ipa
from .tone import Tone

_TONE_MAP = {
    "˧": Tone.MID,
    "˩˩˦": Tone.RISING,
    "˨˩": Tone.LOW,
    "˥˩": Tone.FALLING,
    "˦˥": Tone.HIGH,
}

_TIE_BAR = "͡"       # combining double inverted breve, e.g. t͡ɕ
_NO_RELEASE = "̚"    # combining left angle above, e.g. k̚
_NONSYLLABIC = "̯"   # combining inverted breve below, e.g. a̯ (glide)
_LONG_MARK = "ː"

_ONSET_SET = frozenset(_ONSETS)
_VOWEL_SET = frozenset(_VOWELS)
_CODA_SET = frozenset(_CODAS)
_DIPHTHONG_HEADS = frozenset({"i", "ɯ", "u"})

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


def _convert_syllable(group: str) -> IpaSyllable:
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
    onset, vowel_tok, *coda_toks = phones

    if onset not in _ONSET_SET:
        raise _ConvertError(f"unknown onset {onset!r}")

    long = vowel_tok.endswith(_LONG_MARK)
    vowel = vowel_tok.removesuffix(_LONG_MARK)
    if vowel not in _VOWEL_SET:
        raise _ConvertError(f"unknown vowel {vowel!r}")

    if len(coda_toks) > 1:
        raise _ConvertError(f"unsupported coda cluster {coda_toks!r}")
    coda = coda_toks[0] if coda_toks else None
    if coda is not None and coda not in _CODA_SET:
        raise _ConvertError(f"unknown coda {coda!r}")

    return IpaSyllable(onset, vowel, long, coda, tone)


def _convert(raw: str) -> list[IpaSyllable] | None:
    """Convert pythainlp's thaig2p-shaped raw string to IpaSyllable list.
    Never raises: returns None for anything unmappable."""
    groups = [g for g in _SYLLABLE_SEP.split(raw.strip()) if g.strip()]
    if not groups:
        return None
    try:
        return [_convert_syllable(g) for g in groups]
    except _ConvertError:
        return None


# --- tltk_ipa's compact/digit-tone format --------------------------------
# See the module docstring for how this digit->tone map and the affricate
# remap were derived (verified against tltk's own source, not guessed).

_COMPACT_TONE_DIGIT = {"1": Tone.MID, "2": Tone.LOW, "3": Tone.FALLING,
                       "4": Tone.HIGH, "5": Tone.RISING}
_COMPACT_SYLLABLE_RE = re.compile(r"^(?P<body>.+?)(?P<tone>[1-5])$")
# reverse of ipa.py's own letters->Tone map, to hand off to parse_ipa
_TONE_TO_CHAO_LETTERS = {tone: letters for letters, tone in
                         {"˧": Tone.MID, "˨˩˦": Tone.RISING, "˨˩": Tone.LOW,
                          "˥˩": Tone.FALLING, "˦˥": Tone.HIGH}.items()}


def _convert_compact_syllable(group: str) -> IpaSyllable:
    m = _COMPACT_SYLLABLE_RE.match(group)
    if m is None:
        raise _ConvertError(f"no digit tone in {group!r}")
    tone = _COMPACT_TONE_DIGIT.get(m.group("tone"))
    if tone is None:
        raise _ConvertError(f"unknown tone digit in {group!r}")
    # tltk's ASCII affricate codes ('c' unaspirated, 'ch'->'cʰ' aspirated
    # after the aspiration-diacritic pass) aren't in ipa.py's onset set,
    # which spells the affricate "tɕ"/"tɕʰ" (matching the thaig2p path).
    body = m.group("body").replace("cʰ", "tɕʰ").replace("c", "tɕ")
    try:
        parsed = parse_ipa(body + _TONE_TO_CHAO_LETTERS[tone])
    except IpaParseError as e:
        raise _ConvertError(str(e)) from e
    if len(parsed) != 1:
        raise _ConvertError(f"expected one syllable in {group!r}")
    return parsed[0]


def _convert_compact(raw: str) -> list[IpaSyllable] | None:
    """Convert pythainlp's tltk_ipa-shaped raw string to IpaSyllable list.
    Never raises: returns None for anything unmappable."""
    groups = [g for g in _SYLLABLE_SEP.split(raw.strip()) if g.strip()]
    if not groups:
        return None
    try:
        return [_convert_compact_syllable(g) for g in groups]
    except _ConvertError:
        return None


class PyThaiNLPG2P:
    def __init__(self):
        from pythainlp.transliterate import transliterate
        self._transliterate = transliterate

    def syllables(self, word: str) -> list[IpaSyllable] | None:
        try:
            raw = self._transliterate(word, engine="thaig2p")
            return _convert(raw)
        except Exception:
            return None


class TltkG2P:
    def syllables(self, word: str) -> list[IpaSyllable] | None:
        try:
            from pythainlp.transliterate import transliterate
            return _convert_compact(transliterate(word, engine="tltk_ipa"))
        except Exception:
            return None


class PyThaiNLPTokenizer:
    def __init__(self, extra_words: set[str] | None = None):
        self._trie = None
        if extra_words:
            from pythainlp.corpus import thai_words
            from pythainlp.util import dict_trie
            self._trie = dict_trie(set(thai_words()) | extra_words)

    def tokens(self, text: str) -> list[str]:
        from pythainlp.tokenize import word_tokenize
        if self._trie is not None:
            toks = word_tokenize(text, custom_dict=self._trie)
        else:
            toks = word_tokenize(text)
        return [t for t in toks if t.strip()]
