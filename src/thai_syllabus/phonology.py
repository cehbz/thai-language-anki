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
    "the judge's verdict plus one engine" -- which engine is not fixed.
    Order decides only which reading is written when they disagree, and a
    word they disagree on is `disputed` and blocked anyway.

    `engines_pronunciation`'s whole-reading agreement check (below) is
    defined for exactly two engines: it compares every other reading only
    against `readings[0]`, so a third oracle is NOT free to add as-is --
    with three engines where the first is unique and the other two agree
    with each other but not the first, that pairwise agreement would go
    undetected and the word would stay `disputed`. `corroborates`, which
    checks a judge verdict against every reading in turn, has no such
    limit. Widening the first check to real pairwise agreement is
    speculative generality with no second consumer today (YAGNI); add it
    when a third engine actually arrives.
    """
    g2p: tuple[Callable[[str], tuple[Syllable, ...] | None], ...]
    tone: Callable[[str], Tone | None]

    def readings(self, thai: str) -> tuple[tuple[Syllable, ...], ...]:
        """Each engine's reading of `thai`, in order, with the ʔ-coda
        convention (design 2026-09-18 §5) applied and the empty and the
        degenerate ones (spec 3 r44) dropped.

        `without_glottal_coda` is applied here, at the port boundary, not
        only inside `_convert`/`_convert_tltk`: those converters already
        apply it, so this is redundant for the two real engines, but it is
        what makes the guarantee belong to the port rather than to each
        converter's discipline. A third engine, or a test fake injected
        without it, would otherwise silently fail to corroborate on every
        dead-open-syllable word -- a failure that looks exactly like "the
        engines disagree" and is very hard to diagnose from the outside.
        """
        out = []
        for engine in self.g2p:
            got = engine(thai)
            if not got:
                continue
            reading = without_glottal_coda(tuple(got))
            if not is_degenerate(reading, thai):
                out.append(reading)
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
    "141 of 141 verdicts not corroborated" on five consecutive cycles.

    Precondition: `judge` must already carry the ʔ-coda convention (design
    2026-09-18 §5) -- i.e. be `without_glottal_coda`'d, as `readings()`
    guarantees every engine reading is. The sole caller passes it through
    `syllables_from_verdict`, which normalizes on the way in; an
    unnormalized verdict silently fails to corroborate against a dead
    open syllable rather than raising.
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
    matches it, or -- only when no other engine produced a reading at all
    -- when the rule tone engine agrees with a monosyllable's tone; and
    `disputed` otherwise, including every multi-syllable form no other
    engine confirms, which the adjudication pass then asks the judge
    about (r28) while E4 blocks that word's cards.

    That tone fallback is NARROWER here than in `corroborates`, which
    applies it per reading however many engines spoke. The asymmetry is
    deliberate: `engines_agree` is terminal (`is_corroborated`, so the
    word is never adjudicated again), while `corroborates` only accepts a
    verdict the judge already produced.

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
        # The rule-tone fallback is single-engine behaviour: it only ever
        # confirms a reading no other engine was there to contradict. Once
        # a second reading exists it necessarily differs from `first` (the
        # whole-match check above already failed), so it IS a differing
        # engine reading of the same word -- and the tone rule agreeing
        # with `first` on its own would not be confirmation, it would be
        # overriding that disagreement on the rule engine's say-so alone.
        # (design 2026-09-18: this is what would let a truncated tltk
        # reading be promoted to engines_agree.)
        agrees = (len(readings) == 1 and len(first) == 1
                 and engines.tone(thai) == first[0].tone)
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
    """The real engines (spec 3 r28 section 5): pythainlp's thaig2p and
    tltk's rule-based g2p for segments/length/tone, and the deterministic
    tone-rule engine as the second opinion on a monosyllabic tone
    disagreement. Constructing the thaig2p engine pulls in pythainlp/torch,
    so it happens inside this function -- unit tests inject fake Engines
    and never reach here.

    Memoised: every caller reads its own injected Engines first and falls
    back to this (`ctx.engines or default_engines()`, the seam that stays
    as it is), so a run with none injected would otherwise load the model
    once per call site -- the adjudication pass and the grapheme pass at
    least. The engines are stateless callables, so one instance serves
    the whole process.
    """
    from .engines import Thaig2p, Tltk, rule_tone
    return Engines(g2p=(Thaig2p(), Tltk()), tone=rule_tone)
