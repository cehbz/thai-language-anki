"""Paiboon (this deck's specific romanization variant) -> this project's
authored-IPA string format (see lang/ipa.py's parse_ipa grammar).

Built empirically from the ~1000 word_phonetic values in the "Thai 1000
Common Words" Anki deck (scripts/import_apkg.py's source data), then
cross-checked against independently-known Thai pronunciations. Per-word
conversion is per-syllable (split on spaces/hyphens): every syllable must
convert with confidence or the whole word returns None. The result is
also validated by round-tripping through parse_ipa before being returned.

Tone diacritics (on the FIRST vowel letter of each syllable):
    grave (à)      -> low
    acute (á)      -> high
    circumflex (â) -> falling
    breve (ă) / caron (ǎ) -> rising
    unmarked       -> mid

Onsets (this deck's Paiboon; only single Thai-consonant onsets are
representable -- ipa.py's IpaSyllable has one onset slot, so genuine
consonant CLUSTER onsets (gl-, gr-, gw-, kl-, kr-, kw-, pl-, pr-, bpl-,
bpr-, dtr-, ...) are never mapped: the onset-matching step below only
ever consumes a single Paiboon consonant letter/digraph, so a cluster's
second consonant (l/r/w) is left in the vowel position, fails to match
any vowel spelling, and the syllable correctly comes back unmapped. This
matches src/thai_deck_eval/lang/pythainlp_adapter.py's own documented
behavior for thaig2p's cluster onsets ("fail to convert ... rather than
raising"):
    bp->p  dt->t  ch->tɕʰ  ng->ŋ  g->k  j->tɕ  k->kʰ  p->pʰ  t->tʰ  y->j
    b->b d->d f->f h->h l->l m->m n->n r->r s->s w->w
A syllable with no consonant letter at all (starts directly with a
vowel, e.g. "u", "or", "àan") gets onset ʔ, matching how thaig2p itself
emits an explicit ʔ onset for vowel-initial syllables (see
pythainlp_adapter.py's module docstring, e.g. อา -> "ʔ aː ˧").

Vowels: built from the empirical syllable inventory
(scratchpad/syllables.txt, 779 unique forms across the deck), verified
against real Thai spellings pulled from the deck's own word_tha field for
representative members of each group. Two families use a genuine
short/long PAIR of distinct spellings (safe to map exactly):
    a/aa i/ee u/oo o/oh(+or/oi for the open/definite-long ɔ forms)
    e/ay
plus single-form diphthongs ia/ua/ɯa (eua) and ɤ (er), and fixed
glide-final compounds (aai, aao, ai, ao, oi, ui, iw, eo, uay, ieow, oie)
whose glide occupies the coda slot.

Two vowel families in this specific deck's romanization have NO
distinguishing short/long spelling at all, and the deck's own data proves
the collision is real (not just theoretical) -- these NEVER convert,
regardless of what follows, per "never guess":

  * ae (แ): แต่ "but" (long ɛː, forced low by an explicit mai ek mark)
    and แตะ "touch" (short ɛ, low by the unmarked-dead-syllable default)
    both romanize identically as "dtàe" in this deck.

  * bare "o" + a following consonant letter: Thai's rule that an
    unmarked closed syllable with no vowel sign gets an implicit SHORT o
    (e.g. จบ "finish" -> short o) collides with this deck's use of the
    same bare "o" spelling for explicit long ออ closed syllables (e.g.
    มอบ "give" -> long ɔː); both come out as "op" here. Note open bare
    "o" (no coda at all, e.g. "dtó" for โต๊ะ) IS unambiguous (short) --
    the deck's separate "oh" spelling is reserved for definite long o in
    that position -- so only the +coda case is dropped.

  * "eu" + a following consonant letter: ลึก "deep" (short ɯ, explicit
    ึ mark) and ดื่ม/ปืน/คืน (long ɯː, ื mark) both romanize as "eu" +
    coda letter here. Open "eu" (e.g. "meu" for มือ) is unambiguous
    (long) and does convert.

Finals: p/t/k as literal codas, ng->ŋ, m/n as-is. A short vowel with no
written final gets an explicit ʔ coda (matching thaig2p's own convention
for open dead syllables, e.g. จะ -> "... a ʔ ...") rather than being left
coda-less; long vowels and diphthongs with no written final get no coda.
"""
import re
import unicodedata

from .ipa import IpaParseError, parse_ipa

_TONE_MARKS = {
    "̀": "˨˩",   # grave -> low
    "́": "˦˥",   # acute -> high
    "̂": "˥˩",   # circumflex -> falling
    "̆": "˨˩˦",  # breve -> rising
    "̌": "˨˩˦",  # caron -> rising
}
_MID_TONE = "˧"

# Longest spelling first so e.g. "bp" is tried before "b".
_ONSETS = ["bp", "dt", "ch", "ng",
           "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r",
           "s", "t", "w", "y"]
_ONSET_IPA = {
    "bp": "p", "dt": "t", "ch": "tɕʰ", "ng": "ŋ",
    "b": "b", "d": "d", "f": "f", "g": "k", "h": "h", "j": "tɕ",
    "k": "kʰ", "l": "l", "m": "m", "n": "n", "p": "pʰ", "r": "r",
    "s": "s", "t": "tʰ", "w": "w", "y": "j",
}
_VOWEL_STARTS = "aeiou"

_CODAS = {"ng": "ŋ", "k": "k", "t": "t", "p": "p", "m": "m", "n": "n"}

# (spelling, ipa vowel, length, forced_coda, final_only)
#   length: "long" | "short" | "diph" | "ambig"
#   forced_coda: a fixed coda that's already baked into the spelling
#     (the glide of a diphthong-plus-glide compound); when set, nothing
#     may follow this spelling in the syllable.
#   final_only: this spelling is only unambiguous when it's the whole
#     rest of the syllable (see "eu"/"o" doc above); if anything follows,
#     the entry does not match at all (falls through to unmapped).
# Ordered longest-spelling-first so matching is greedy/unambiguous.
_VOWELS = [
    ("ieow", "ia", "diph", "w", False),
    ("eua", "ɯa", "diph", None, False),
    ("aai", "a", "long", "j", False),
    ("aao", "a", "long", "w", False),
    ("uay", "ua", "diph", "j", False),
    ("oie", "ɤ", "long", "j", False),
    ("ae", "ɛ", "ambig", None, False),
    ("ai", "a", "short", "j", False),
    ("ao", "a", "short", "w", False),
    ("ay", "e", "long", None, False),
    ("aa", "a", "long", None, False),
    ("ee", "i", "long", None, False),
    ("oo", "u", "long", None, False),
    ("oh", "o", "long", None, False),
    ("or", "ɔ", "long", None, False),
    ("oi", "ɔ", "long", "j", False),
    ("ui", "u", "short", "j", False),
    ("iw", "i", "short", "w", False),
    ("eo", "e", "short", "w", False),
    ("er", "ɤ", "long", None, False),
    ("ia", "ia", "diph", None, False),
    ("ua", "ua", "diph", None, False),
    ("eu", "ɯ", "long", None, True),   # unambiguous only when syllable-final
    ("o", "o", "short", None, True),   # unambiguous only when syllable-final
    ("a", "a", "short", None, False),
    ("e", "e", "short", None, False),
    ("i", "i", "short", None, False),
    ("u", "u", "short", None, False),
]


def _decompose(ch: str) -> tuple[str, str | None]:
    d = unicodedata.normalize("NFD", ch)
    return (d[0], d[1]) if len(d) > 1 else (d, None)


def _extract_tone(token: str) -> tuple[str, str] | None:
    """Return (tone_letters, plain_ascii_syllable), or None if the token
    carries conflicting diacritics."""
    plain_chars = []
    tone = None
    for ch in token:
        base, mark = _decompose(ch)
        plain_chars.append(base)
        mark_tone = _TONE_MARKS.get(mark) if mark else None
        if mark_tone is not None:
            if tone is not None and tone != mark_tone:
                return None
            tone = mark_tone
    return (tone or _MID_TONE), "".join(plain_chars)


def _match_onset(rest: str) -> tuple[str, str] | None:
    for spelling in _ONSETS:
        if rest.startswith(spelling):
            return _ONSET_IPA[spelling], rest[len(spelling):]
    if rest[:1] in _VOWEL_STARTS:
        return "ʔ", rest
    return None


def _match_vowel(rest: str):
    for spelling, ipa_vowel, length, forced_coda, final_only in _VOWELS:
        if not rest.startswith(spelling):
            continue
        tail = rest[len(spelling):]
        if final_only and tail:
            continue
        if length == "ambig":
            return None
        return ipa_vowel, tail, length, forced_coda
    return None


def _convert_syllable(token: str) -> str | None:
    extracted = _extract_tone(token)
    if extracted is None:
        return None
    tone_letters, plain = extracted

    onset_match = _match_onset(plain)
    if onset_match is None:
        return None
    onset_ipa, rest = onset_match

    vowel_match = _match_vowel(rest)
    if vowel_match is None:
        return None
    ipa_vowel, tail, length, forced_coda = vowel_match

    if forced_coda is not None:
        if tail:
            return None  # a glide-final vowel can't take a further coda
        coda_ipa = forced_coda
        long_flag = length == "long"
    elif tail == "":
        if length == "diph":
            coda_ipa, long_flag = None, False
        elif length == "long":
            coda_ipa, long_flag = None, True
        else:  # short, no written final -> implicit glottal stop
            coda_ipa, long_flag = "ʔ", False
    else:
        coda_ipa = _CODAS.get(tail)
        if coda_ipa is None:
            return None
        long_flag = length == "long"

    long_marker = "ː" if long_flag else ""
    return f"{onset_ipa}{ipa_vowel}{long_marker}{coda_ipa or ''}{tone_letters}"


def paiboon_to_ipa(s: str) -> str | None:
    syllables = [tok for tok in re.split(r"[\s-]+", s.strip()) if tok]
    if not syllables:
        return None
    converted = [_convert_syllable(syl) for syl in syllables]
    if any(c is None for c in converted):
        return None
    joined = ".".join(converted)
    try:
        parsed = parse_ipa(joined)
    except IpaParseError:
        return None
    if len(parsed) != len(converted):
        return None
    return joined
