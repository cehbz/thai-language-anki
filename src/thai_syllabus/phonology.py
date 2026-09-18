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
from .entities import Pronunciation, Syllable, Tone, without_glottal_coda


@dataclass(frozen=True)
class Engines:
    """The segmental oracles, in order, plus the tone-rule engine
    (design 2026-09-18 §1). `g2p` is a tuple because corroboration is
    "the judge's verdict plus one engine" -- which engine is not fixed,
    and a third oracle costs nothing to add. Order decides only which
    reading is written when they disagree, and a word they disagree on is
    `disputed` and blocked anyway.
    """
    g2p: tuple[Callable[[str], tuple[Syllable, ...] | None], ...]
    tone: Callable[[str], Tone | None]

    def readings(self, thai: str) -> tuple[tuple[Syllable, ...], ...]:
        """Each engine's reading of `thai`, in order, with the empty and
        the degenerate ones (spec 3 r44) dropped."""
        out = []
        for engine in self.g2p:
            got = engine(thai)
            if got and not is_degenerate(tuple(got), thai):
                out.append(tuple(got))
        return tuple(out)


def syllables_from_verdict(value: Mapping) -> tuple[Syllable, ...]:
    return without_glottal_coda(tuple(
        Syllable(segments=tuple(s["segments"]), vowel_length=s["vowel_length"],
                 tone=s["tone"]) for s in value["syllables"]))


def _same_segments(a: Syllable, b: Syllable) -> bool:
    return a.segments == b.segments and a.vowel_length == b.vowel_length


def corroborates(judge: tuple[Syllable, ...], thai: str, engines: Engines) -> bool:
    """Whether a judge verdict is corroborated by an engine (design
    2026-09-18 §2): it must match ANY engine on segments and vowel
    length, and then on tone -- or, for a single syllable, the rule tone
    engine decides. Before this there was one segmental oracle, so a word
    thaig2p read badly could never leave `disputed`: the live run logged
    "141 of 141 verdicts not corroborated" on three consecutive cycles.
    """
    for reading in engines.readings(thai):
        if len(reading) != len(judge):
            continue
        if not all(_same_segments(a, b) for a, b in zip(judge, reading)):
            continue
        if all(a.tone == b.tone for a, b in zip(judge, reading)):
            return True
        if len(judge) == 1 and engines.tone(thai) == judge[0].tone:
            return True
    return False


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
    """The engines' own reading of `thai`, the seed every Word the run
    adopts is written with (spec 3 r40/r43 section 5; design 2026-09-12
    §2, 2026-09-18 §3): the first engine's syllables in tuple order,
    corroboration `engines_agree` when a second engine's whole reading
    matches it, or when the rule tone engine agrees with a monosyllable's
    tone -- the same agreement rule `corroborates` falls back on -- and
    `disputed` otherwise, including every multi-syllable form no other
    engine confirms, which the adjudication pass then asks the judge
    about (r28) while E4 blocks that word's cards.

    Two engines agreeing is `engines_agree` (design 2026-09-18 §3) --
    evidence a single engine plus the one-syllable tone rule could never
    give for a multi-syllable form. Otherwise the first sound reading in
    tuple order is written, `engines_agree` when the rule tone engine
    settles its single syllable's tone, else `disputed`.

    A recited name the engines cannot read as a phrase is read token by
    token (r43, 2026-09-17 evidence: thaig2p reads "ปอ" and "ปลา" but not
    "ปอ ปลา"): when no engine yields anything for the whole of `thai` and
    `thai` contains whitespace, each whitespace-separated token is read on
    its own (its first engine reading) and the syllables concatenated in
    order, always `disputed` -- a token-wise reading is never
    engine-agreed, the phrase having failed the whole-form ask that
    `corroborates` itself would still make.

    A reading `is_degenerate` condemns is treated as no reading at all
    (2026-09-18): the whole-form loop falls through to the token-wise
    path, and a loop in the combined result is refused too.

    None when no engine reads anything and there is no token to fall back
    on, when any one token itself reads nothing, or when every reading it
    could build is degenerate: a Word is never written with an empty
    syllable tuple, and the caller reports the row it could not adopt.
    """
    readings = engines.readings(thai)
    if readings:
        first = readings[0]
        if any(other == first for other in readings[1:]):
            return Pronunciation(syllables=first, corroboration="engines_agree")
        agrees = len(first) == 1 and engines.tone(thai) == first[0].tone
        return Pronunciation(syllables=first,
                             corroboration="engines_agree" if agrees else "disputed")
    if " " not in thai:
        return None
    combined: list[Syllable] = []
    for token in thai.split():
        token_readings = engines.readings(token)
        if not token_readings:
            return None
        combined.extend(token_readings[0])
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
    return Engines(g2p=(Thaig2p(),), tone=rule_tone)
