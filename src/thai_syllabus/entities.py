"""Language model and teaching-material entities (spec 1, section 1).

Frozen dataclasses; identity fields are noted per entity. Thai strings are
always accompanied by an English gloss in comments/docstrings (project rule).

Construction invariants that need a resolved Word (Grapheme's keyword
containment, MinimalPair's exact-confusion check) are enforced by
classmethod factories -- `Grapheme.create` / `MinimalPair.create` -- rather
than by the plain constructor, since the plain dataclass only holds ids.
Loaded data is re-checked the same way by the `grapheme/keyword-contains-
symbol` and `pair/exact-confusion` rules (see rulebook.py), which reuse the
same pure diff functions.
"""
import hashlib
from dataclasses import dataclass
from typing import Literal

from .ids import CategoryName, ConfusionId, PairId, TargetId, WordId
from .media import Provenance

Dimension = Literal["tone", "length", "aspiration", "vowel_quality", "consonant"]
Skill = Literal["receptive", "productive"]
Introduction = Literal["picture_card", "sentence"]
Voice = Literal["learner_voice", "other_voice"]
Tone = Literal["mid", "low", "falling", "high", "rising"]
VowelLength = Literal["short", "long"]

# A word's pronunciation is a lexical fact, adjudicated by knowledge (engines
# as cheap oracles, an LLM as a better-read oracle on disagreement). Only
# "disputed" blocks card emission (rule R-PRON); the other two values both
# count as corroborated. Not specified further by the spec -- see the
# domain-language doc's "Smell 3 resolved" note.
Corroboration = Literal["engines_agree", "curated_exception", "disputed"]


def is_corroborated(c: Corroboration) -> bool:
    return c != "disputed"


@dataclass(frozen=True)
class Syllable:
    """One syllable's segments, vowel length, and tone.

    `segments` is a fixed (onset, vowel, coda) triple of phonemic segment
    strings; coda is "" for an open syllable. The spec names only "segments,
    vowel length, Chao tone" as what a Syllable holds -- this 3-tuple shape
    is a design choice made here (see the implementation report).
    """
    segments: tuple[str, str, str]
    vowel_length: VowelLength
    tone: Tone

    @property
    def onset(self) -> str:
        return self.segments[0]

    @property
    def vowel(self) -> str:
        return self.segments[1]

    @property
    def coda(self) -> str:
        return self.segments[2]


@dataclass(frozen=True)
class Pronunciation:
    syllables: tuple[Syllable, ...]
    corroboration: Corroboration


@dataclass(frozen=True)
class Word:
    """One sense of a Thai lexical item. Identity: id."""
    id: WordId
    thai: str
    pron: Pronunciation
    meaning: str
    classifier: WordId | None = None


@dataclass(frozen=True)
class SoundConfusion:
    """Two Thai sounds liable to be mistaken for each other. Identity: id."""
    id: ConfusionId
    dimension: Dimension
    sounds: tuple[str, str]


def _dimension_value(syllable: Syllable, dimension: Dimension) -> str:
    if dimension == "tone":
        return syllable.tone
    if dimension == "length":
        return syllable.vowel_length
    if dimension in ("aspiration", "consonant"):
        return syllable.onset
    if dimension == "vowel_quality":
        return syllable.vowel
    raise ValueError(f"unknown dimension: {dimension!r}")


def _segment_diff(a: Syllable, b: Syllable) -> set[Dimension]:
    """Every dimension on which two syllables differ."""
    diffs: set[Dimension] = set()
    if a.tone != b.tone:
        diffs.add("tone")
    if a.vowel_length != b.vowel_length:
        diffs.add("length")
    if a.onset != b.onset:
        bare_a, bare_b = a.onset.rstrip("hʰ"), b.onset.rstrip("hʰ")
        diffs.add("aspiration" if bare_a == bare_b else "consonant")
    if a.vowel != b.vowel:
        diffs.add("vowel_quality")
    if a.coda != b.coda:
        diffs.add("consonant")
    return diffs


def pronunciation_diff(a: Pronunciation, b: Pronunciation) -> set[Dimension]:
    """Every dimension on which two Pronunciations differ, across syllables."""
    if len(a.syllables) != len(b.syllables):
        return {"tone", "length", "aspiration", "vowel_quality", "consonant"}
    diffs: set[Dimension] = set()
    for sa, sb in zip(a.syllables, b.syllables):
        diffs |= _segment_diff(sa, sb)
    return diffs


def exact_confusion_violation(confusion: SoundConfusion,
                               pronunciations: tuple[Pronunciation, ...]) -> str | None:
    """None if every pair of `pronunciations` differs in exactly
    `confusion.dimension`, using only `confusion.sounds`' two values.
    Otherwise a human-readable reason.
    """
    if len(pronunciations) < 2:
        return "a minimal pair needs at least two members"
    for i in range(len(pronunciations)):
        for j in range(i + 1, len(pronunciations)):
            diff = pronunciation_diff(pronunciations[i], pronunciations[j])
            if diff != {confusion.dimension}:
                return (f"members {i} and {j} differ in {sorted(diff)}, "
                        f"not exactly {{{confusion.dimension!r}}}")
    allowed = set(confusion.sounds)
    for i, p in enumerate(pronunciations):
        # Compare against the first member that actually differs on this
        # dimension, so the check works even for a syllable count mismatch
        # (pronunciation_diff already rejected those above).
        value = _dimension_value(p.syllables[-1], confusion.dimension)
        if value not in allowed:
            return (f"member {i}'s {confusion.dimension} value {value!r} "
                    f"is not one of the confusion's sounds {confusion.sounds!r}")
    return None


@dataclass(frozen=True)
class Grapheme:
    """A spelling unit with its sound facts and an exemplar (keyword) Word.

    Identity: symbol. Invariant: keyword's thai contains symbol -- enforced
    by `create`, which needs the resolved keyword Word (not just its id).

    `name_word` (spec 4, section 1) is the recited letter name as its own
    Word -- e.g. for ก the name-word is "กอ" ("gɔɔ"), distinct from the
    keyword "ไก่" ("gài", chicken): the grapheme/Reading card's NameThai
    field is the two words' `thai` concatenated ("กอ ไก่"). No containment
    invariant applies to it (unlike keyword, a name-word need not spell out
    the symbol itself -- consonant names substitute a vowel, e.g. ก -> กอ,
    not ก-something containing ก verbatim in every class). Defaults to
    None: curated data may not carry it yet, and compile()'s NameThai
    rendering degrades gracefully (falls back to symbol + keyword) when
    absent.
    """
    symbol: str
    kind: Literal["consonant", "vowel_sign", "tone_mark"]
    sound: str
    consonant_class: Literal["mid", "high", "low"] | None
    keyword: WordId
    name_word: WordId | None = None

    @classmethod
    def create(cls, *, symbol: str, kind: Literal["consonant", "vowel_sign", "tone_mark"],
               sound: str, consonant_class: Literal["mid", "high", "low"] | None,
               keyword_word: Word, name_word: Word | None = None) -> "Grapheme":
        if symbol not in keyword_word.thai:
            raise ValueError(
                f"grapheme {symbol!r} is not contained in keyword word "
                f"{keyword_word.id!r} ({keyword_word.thai!r})")
        return cls(symbol=symbol, kind=kind, sound=sound,
                   consonant_class=consonant_class, keyword=keyword_word.id,
                   name_word=name_word.id if name_word is not None else None)


@dataclass(frozen=True)
class Target:
    """(word, skill): a learning target. Identity: id."""
    id: TargetId
    word: WordId
    skill: Skill
    introduction: Introduction = "picture_card"


@dataclass(frozen=True)
class Category:
    """A curated theme grouping Words (the FF 625 list). Identity: name.

    A word belongs to at most one Category (rule category/single-
    membership); closure words (pair members, keywords) belong to none.
    """
    name: CategoryName
    members: frozenset[WordId]


@dataclass(frozen=True)
class MinimalPair:
    """2-3 Words exhibiting exactly one SoundConfusion. Identity: id."""
    id: PairId
    confusion: ConfusionId
    members: tuple[WordId, ...]

    @classmethod
    def create(cls, *, id: PairId, confusion: SoundConfusion,
               members: tuple[Word, ...]) -> "MinimalPair":
        if not (2 <= len(members) <= 3):
            raise ValueError("a minimal pair has 2 or 3 members")
        reason = exact_confusion_violation(
            confusion, tuple(m.pron for m in members))
        if reason is not None:
            raise ValueError(reason)
        return cls(id=id, confusion=confusion.id,
                   members=tuple(m.id for m in members))


def text_sha(text: str) -> str:
    """sha256 hex digest of text, the one sentence id."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Sentence:
    """An artifact, not a fact of the language. Identity: text_sha, the sha256
    of text; provenance is a fact of the row, not identity.

    Which Targets it fills is derived (Syllabus.fills), never stored.
    """
    text: str
    gloss: str
    voice: Voice
    provenance: Provenance

    @property
    def text_sha(self) -> str:
        return text_sha(self.text)
