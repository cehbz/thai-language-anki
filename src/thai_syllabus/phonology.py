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
    """The local segmental oracles, in order; the tone-rule engine
    (design 2026-09-18 §1); and an optional dictionary (design
    2026-09-20 §2). `g2p` is a tuple because corroboration is "the
    judge's verdict plus one engine" -- which engine is not fixed.
    Order decides only which reading is written when they disagree, and a
    word they disagree on is `disputed` and blocked anyway.

    Engines compute, a dictionary looks up -- so the dictionary is
    consulted lazily, only where the local engines fail to agree
    (`engines_pronunciation`) or to corroborate a verdict
    (`corroborates`).

    `engines_pronunciation`'s whole-reading agreement check (below) is
    real pairwise agreement (`_agreed`, design 2026-09-20 §3): any two
    readings equal, whichever positions they hold, so a third oracle is
    free to add as-is.
    """
    g2p: tuple[Callable[[str], tuple[Syllable, ...] | None], ...]
    tone: Callable[[str], Tone | None]
    dictionary: Callable[[str], tuple[tuple[Syllable, ...], ...]] | None = None

    def readings(self, thai: str) -> tuple[tuple[Syllable, ...], ...]:
        """Each local engine's reading of `thai`, in order, with the ʔ-coda
        convention (design 2026-09-18 §5) applied and the empty and the
        degenerate ones (spec 3 r44) dropped.

        A form containing whitespace is a phrase and is read token by
        token (design 2026-09-20 §3): the engine's reading is the
        concatenation of its token readings, and a token it cannot read
        makes the phrase unread by it. No whole-form read of a phrase is
        ever made: thaig2p reads "ปอ" and "ปลา" but not "ปอ ปลา", and
        reads "งอ งู" with a spurious ŋ coda that "งอ" alone does not
        have.

        `without_glottal_coda` is applied here, at the port boundary, not
        only inside `_convert`/`_convert_tltk`: those converters already
        apply it, so this is redundant for the two real engines, but it is
        what makes the guarantee belong to the port rather than to each
        converter's discipline. A third engine, or a test fake injected
        without it, would otherwise silently fail to corroborate on every
        dead-open-syllable word -- a failure that looks exactly like "the
        engines disagree" and is very hard to diagnose from the outside.
        """
        tokens = thai.split() if " " in thai else [thai]
        out = []
        for engine in self.g2p:
            reading = _read_tokens(engine, tokens)
            if reading is not None and not is_degenerate(reading, thai):
                out.append(reading)
        return tuple(out)

    def dictionary_readings(self, thai: str) -> tuple[tuple[Syllable, ...], ...]:
        """The dictionary's readings of `thai`, normalized and
        degeneracy-checked exactly as any engine's are (design 2026-09-20
        §2): a looked-up reading is no more trusted than a computed one.

        `()` when this Engines has no dictionary, and `()` for a phrase:
        a phrase is its tokens (design 2026-09-20 §3), and a dictionary
        has no entry for a token sequence, so it is never asked one.

        DEDUPLICATED, order preserved, and that is load-bearing: an entry
        lists several IPA spans, two of which can normalize to the same
        reading (a /tɕaʔ/ span and a /tɕa/ span collapse under the ʔ-coda
        convention). `_agreed` returns any reading two entries of the
        tuple share, so an undeduplicated pair would pairwise-agree with
        ITSELF and seal `engines_agree` -- which is terminal
        (`is_corroborated`) -- on the dictionary's say-so alone. The
        dictionary's variants are one opinion, and it may reach
        `engines_agree` only by agreeing with a local engine.
        """
        if self.dictionary is None or " " in thai:
            return ()
        out = []
        for got in self.dictionary(thai):
            reading = without_glottal_coda(tuple(got))
            if reading and not is_degenerate(reading, thai):
                out.append(reading)
        return tuple(dict.fromkeys(out))


def _read_tokens(engine, tokens: list[str]) -> tuple[Syllable, ...] | None:
    if not tokens:
        # A whitespace-only (or all-separator) form splits to no tokens at
        # all -- not a form with one empty token, a form with none. An
        # empty `combined` here would read as "the engine agreed on
        # nothing", so this is no reading, not an empty one (a Word is
        # never written with an empty syllable tuple).
        return None
    combined: list[Syllable] = []
    for token in tokens:
        got = engine(token)
        if not got:
            return None
        combined.extend(without_glottal_coda(tuple(got)))
    return tuple(combined)


def _agreed(readings: tuple[tuple[Syllable, ...], ...]) -> tuple[Syllable, ...] | None:
    """The first reading some later reading equals -- pairwise agreement
    (design 2026-09-20 §3), so two oracles agreeing is found whichever
    positions they hold."""
    for i, reading in enumerate(readings):
        if any(other == reading for other in readings[i + 1:]):
            return reading
    return None


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

    The dictionary is consulted only when no local reading corroborates
    (design 2026-09-20 §2): a verdict the engines already back costs no
    lookup.
    """
    if _matches_any(judge, thai, engines.readings(thai), engines):
        return True
    return _matches_any(judge, thai, engines.dictionary_readings(thai), engines)


def _matches_any(judge: tuple[Syllable, ...], thai: str,
                 readings: tuple[tuple[Syllable, ...], ...], engines: Engines) -> bool:
    """Whether any of `readings` corroborates `judge` under the rule
    above -- the same test for a local engine's reading and the
    dictionary's."""
    for reading in readings:
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
    §2, 2026-09-18 §3, 2026-09-20 §3): pairwise agreement between any two
    of `engines.readings(thai)` is `engines_agree` (the first such
    reading found); otherwise, only when no other engine produced a
    reading at all, a lone monosyllable the rule tone engine agrees with
    is also `engines_agree`; else the first reading in tuple order is
    written, `disputed` -- including every multi-syllable form no other
    engine confirms, which the adjudication pass then asks the judge
    about (r28) while E4 blocks that word's cards. None when no engine
    reads `thai` at all: a Word is never written with an empty syllable
    tuple, and the caller reports the row it could not adopt.

    That tone fallback is NARROWER here than in `corroborates`, which
    applies it per reading however many engines spoke. The asymmetry is
    deliberate: `engines_agree` is terminal (`is_corroborated`, so the
    word is never adjudicated again), while `corroborates` only accepts a
    verdict the judge already produced.

    A form no local engine reads but the dictionary does is written with
    the dictionary's first reading, `disputed` (design 2026-09-20 §2),
    for the judge to corroborate against -- a seed where there was none,
    never a reading that stands on the dictionary's say-so alone.
    """
    readings = engines.readings(thai)
    agreed = _agreed(readings)
    if agreed is None:
        # Local engines did not agree (or did not read): the dictionary is
        # consulted now, and only now (design 2026-09-20 §2).
        readings = readings + engines.dictionary_readings(thai)
        agreed = _agreed(readings)
    if agreed is not None:
        return Pronunciation(syllables=agreed, corroboration="engines_agree")
    if not readings:
        return None
    first = readings[0]
    # The rule-tone fallback is single-engine behaviour: it only ever
    # confirms a reading no other engine was there to contradict.
    settles = (len(readings) == 1 and len(first) == 1
               and engines.tone(thai) == first[0].tone)
    return Pronunciation(syllables=first,
                         corroboration="engines_agree" if settles else "disputed")


@functools.cache
def default_engines(dictionary: Callable[[str], tuple[tuple[Syllable, ...], ...]] | None = None
                    ) -> Engines:
    """The real engines (spec 3 r28 section 5): pythainlp's thaig2p and
    tltk's rule-based g2p for segments/length/tone, and the deterministic
    tone-rule engine as the second opinion on a monosyllabic tone
    disagreement. Constructing the thaig2p engine pulls in pythainlp/torch,
    so it happens inside this function -- unit tests inject fake Engines
    and never reach here.

    `dictionary` is the deck's Wiktionary backend
    (wiring.build_sourcing), threaded through by every caller as
    `default_engines(ctx.dictionary)`; None leaves the Engines with no
    dictionary at all.

    Memoised: every caller reads its own injected Engines first and falls
    back to this (`ctx.engines or default_engines(ctx.dictionary)`, the
    seam that stays as it is), so a run with none injected would
    otherwise load the model once per call site -- the adjudication pass
    and the grapheme pass at least. Memoised PER DICTIONARY OBJECT: one
    run has one backend, so its call sites share one Engines, and a
    second deck in the same process gets its own rather than the first
    deck's record. The engines are stateless callables, so one instance
    serves the whole process.
    """
    from .engines import Thaig2p, Tltk, rule_tone
    return Engines(g2p=(Thaig2p(), Tltk()), tone=rule_tone, dictionary=dictionary)
