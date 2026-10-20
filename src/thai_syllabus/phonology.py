"""The local pronunciation engines as the settlement's oracles (spec 3
section 3, r68): thaig2p and tltk read a form's syllables, the rule tone
engine a monosyllable's tone; each reading is a record row
(recorded.py) the settlement folds (settlement.py). Engines are callables
so the run injects the real ones and tests inject fakes; the engines
themselves live in engines.py, and thaig2p (torch) loads only inside
default_engines()."""
from __future__ import annotations
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from .entities import Syllable, Tone, without_glottal_coda


@dataclass(frozen=True)
class Engines:
    """The local engines by member name (settlement.FAMILY): the g2p
    engines (thaig2p, tltk) reading a form's syllables, and the rule tone
    engine (`rule-tone`, design 2026-09-18 §1) a monosyllable's tone."""
    g2p: Mapping[str, Callable[[str], tuple[Syllable, ...] | None]]
    tone: Callable[[str], Tone | None]


def read_form(engine, thai: str) -> tuple[Syllable, ...] | None:
    """One engine's reading of `thai`, ʔ-coda convention applied (design
    2026-09-18 §5), None when unread or degenerate (spec 3 r44).

    A form containing whitespace is a phrase and is read token by token
    (design 2026-09-20 §3): the engine's reading is the concatenation of
    its token readings, and a token it cannot read makes the phrase
    unread by it. thaig2p reads "ปอ" and "ปลา" but not "ปอ ปลา", and
    reads "งอ งู" with a spurious ŋ coda that "งอ" alone does not have.
    `str.split()` is the tokenizer: the same whitespace
    `entities.is_phrase` tests."""
    reading = _read_tokens(engine, thai.split())
    if reading is None or is_degenerate(reading, thai):
        return None
    return reading


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


def syllables_from_verdict(value: Mapping) -> tuple[Syllable, ...]:
    return without_glottal_coda(tuple(
        Syllable(segments=tuple(s["segments"]), vowel_length=s["vowel_length"],
                 tone=s["tone"]) for s in value["syllables"]))


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


@functools.cache
def default_engines() -> Engines:
    """The real engines (spec 3 section 3): pythainlp's thaig2p and tltk's
    rule-based g2p for segments/length/tone, and the deterministic
    tone-rule engine. Constructing the thaig2p engine pulls in
    pythainlp/torch, so it happens inside this function -- unit tests
    inject fake Engines and never reach here. Memoised: the engines are
    stateless callables, so one instance serves the whole process.
    """
    from .engines import Thaig2p, Tltk, rule_tone
    return Engines(g2p={"thaig2p": Thaig2p(), "tltk": Tltk()}, tone=rule_tone)
