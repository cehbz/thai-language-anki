"""Pronunciation engines and the corroboration rule (spec 3 r28 section 5;
the 2026-09-01 domain-language ruling: engines are cheap oracles, the
judge the better-read oracle when they disagree, its verdict plus one
engine is corroboration). Engines are callables so the run injects the
real ones and tests inject fakes; the engines themselves live in
engines.py, and thaig2p (torch) loads only inside default_engines()."""
from __future__ import annotations
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from .entities import Pronunciation, Syllable, Tone


@dataclass(frozen=True)
class Engines:
    g2p: Callable[[str], tuple[Syllable, ...] | None]
    tone: Callable[[str], Tone | None]


def syllables_from_verdict(value: Mapping) -> tuple[Syllable, ...]:
    return tuple(Syllable(segments=tuple(s["segments"]), vowel_length=s["vowel_length"],
                          tone=s["tone"]) for s in value["syllables"])


def _same_segments(a: Syllable, b: Syllable) -> bool:
    return a.segments == b.segments and a.vowel_length == b.vowel_length


def corroborates(judge: tuple[Syllable, ...], thai: str, engines: Engines) -> bool:
    g2p = engines.g2p(thai)
    if g2p is None or len(g2p) != len(judge):
        return False
    if not all(_same_segments(a, b) for a, b in zip(judge, g2p)):
        return False
    if all(a.tone == b.tone for a, b in zip(judge, g2p)):
        return True
    if len(judge) != 1:
        return False
    return engines.tone(thai) == judge[0].tone


# --- degenerate readings: the neural g2p's decoder loop --------------------

# ก..ฮ: every Thai letter that can open a syllable. A syllable needs at
# least one of them, so their count bounds any honest reading's length.
_CONSONANT_LETTERS = frozenset(chr(c) for c in range(0x0E01, 0x0E2F))

# Two identical syllables in a row are Thai (ๆ repeats a word once --
# ข้างๆ kʰâːŋ.kʰâːŋ -- and ตุ๊กตุ๊ก túk.túk is spelt with the repeat);
# three are the decoder emitting the same token until it is cut off.
_MAX_REPEAT = 2


def is_degenerate(syllables: tuple[Syllable, ...], thai: str) -> bool:
    """Whether a reading is the neural g2p's decoder loop rather than a
    reading (2026-09-18 evidence: thaig2p reads ภาพวาด, four consonant
    letters, as eleven syllables "pʰa pʰa wa wa wa wa wa wa wa wa wa", and
    reads ษอ ฤๅษี as eleven).

    Two independent checks, either of which condemns the reading; on the
    live deck's 890 words neither fires on a word that is not corrupt:

    - more syllables than `thai` has consonant letters, which no reading
      can honestly have;
    - the same syllable three times in a row, which no Thai word has.

    The predicate is shared with migrate, so an old deck's looped IPA is
    refused at the boundary instead of entering words.yaml as a
    `curated_exception` the adjudication pass never revisits.
    """
    if not syllables:
        return False
    if len(syllables) > sum(1 for c in thai if c in _CONSONANT_LETTERS):
        return True
    run = 1
    for earlier, later in zip(syllables, syllables[1:]):
        run = run + 1 if later == earlier else 1
        if run > _MAX_REPEAT:
            return True
    return False


def engines_pronunciation(thai: str, engines: Engines) -> Pronunciation | None:
    """The two engines' own reading of `thai`, the seed every Word the run
    adopts is written with (spec 3 r40/r43 section 5; design 2026-09-12
    §2): thaig2p's syllables, corroboration `engines_agree` when the rule
    tone engine agrees with a monosyllable's tone -- the same agreement
    rule `corroborates` falls back on -- and `disputed` otherwise,
    including every multi-syllable form, which the adjudication pass then
    asks the judge about (r28) while E4 blocks that word's cards.

    A recited name the engines cannot read as a phrase is read token by
    token (r43, 2026-09-17 evidence: thaig2p reads "ปอ" and "ปลา" but not
    "ปอ ปลา"): when thaig2p yields nothing for the whole of `thai` and
    `thai` contains whitespace, each whitespace-separated token is read on
    its own and the syllables concatenated in order, always `disputed` --
    a token-wise reading is never engine-agreed, the phrase having failed
    the whole-form ask that `corroborates` itself would still make.

    A reading `is_degenerate` condemns is treated as no reading at all
    (2026-09-18): the whole-form loop falls through to the token-wise
    path, and a loop in the combined result is refused too.

    None when thaig2p reads nothing and there is no token to fall back on,
    when any one token itself reads nothing, or when every reading it
    could build is degenerate: a Word is never written with an empty
    syllable tuple, and the caller reports the row it could not adopt.
    """
    syllables = engines.g2p(thai)
    if syllables and not is_degenerate(tuple(syllables), thai):
        agrees = len(syllables) == 1 and engines.tone(thai) == syllables[0].tone
        return Pronunciation(syllables=tuple(syllables),
                             corroboration="engines_agree" if agrees else "disputed")
    if " " not in thai:
        return None
    combined: list[Syllable] = []
    for token in thai.split():
        token_syllables = engines.g2p(token)
        if not token_syllables:
            return None
        combined.extend(token_syllables)
    if is_degenerate(tuple(combined), thai):
        return None
    return Pronunciation(syllables=tuple(combined), corroboration="disputed")


@functools.cache
def default_engines() -> Engines:
    """The real engines (spec 3 r28 section 5): pythainlp's thaig2p for
    segments/length/tone, and the deterministic tone-rule engine as the
    second opinion on a monosyllabic tone disagreement. Constructing the
    thaig2p engine pulls in pythainlp/torch, so it happens inside this
    function -- unit tests inject fake Engines and never reach here.

    Memoised: every caller reads its own injected Engines first and falls
    back to this (`ctx.engines or default_engines()`, the seam that stays
    as it is), so a run with none injected would otherwise load the model
    once per call site -- the adjudication pass and the grapheme pass at
    least. The engines are stateless callables, so one instance serves
    the whole process.
    """
    from .engines import Thaig2p, rule_tone
    return Engines(g2p=Thaig2p(), tone=rule_tone)
