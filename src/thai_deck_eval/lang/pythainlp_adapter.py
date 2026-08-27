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
could not be empirically exercised here. TltkG2P still routes through
the shared `_convert` below per the task brief's sketch; per the task's
instructions this is acceptable — the port's contract is "never raise",
not "always succeed" — but note that if tltk becomes usable in this
environment in the future, `_convert` will likely need a second code
path for its compact/digit-tone format.
"""
import re

from .ipa import _CODAS, _ONSETS, _VOWELS, IpaSyllable
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
            return _convert(transliterate(word, engine="tltk_ipa"))
        except Exception:
            return None


class PyThaiNLPTokenizer:
    def tokens(self, text: str) -> list[str]:
        from pythainlp.tokenize import word_tokenize
        return [t for t in word_tokenize(text) if t.strip()]
