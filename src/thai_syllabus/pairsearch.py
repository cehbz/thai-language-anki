"""The pair search's pure core (design 2026-09-12 section 2; spec 3 r47
section 5): which pairs a confusion still wants, and the best exact pairs
a candidate pool can form for it. No record, no engines, no I/O -- the run
pass (attempts.pair_search_attempt) builds the candidates and adopts what
this returns.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from itertools import combinations

from .entities import (Dimension, MinimalPair, Pronunciation, SoundConfusion,
                       exact_confusion_violation, is_corroborated)
from .ids import ConfusionId, PairId, WordId

_UNRANKED = 10**6


@dataclass(frozen=True)
class Candidate:
    """One form the search may use as a member: its corroborated
    pronunciation, the Word it already is (None for an outside form) and
    its frequency rank."""
    thai: str
    pron: Pronunciation
    word_id: WordId | None
    rank: int | None


def wanted(confusions: Sequence[SoundConfusion], pairs: Sequence[MinimalPair]
           ) -> dict[ConfusionId, int]:
    """Pairs each confusion still wants: pair_count (weight-proportional,
    design ruling 2) minus the pairs adopted for it, never below zero."""
    have: dict[ConfusionId, int] = {}
    for p in pairs:
        have[p.confusion] = have.get(p.confusion, 0) + 1
    return {c.id: max(0, c.pair_count - have.get(c.id, 0)) for c in confusions}


def pair_id_for(confusion: SoundConfusion, a: Candidate, b: Candidate) -> PairId:
    """`<confusion id>/<x>-<y>`: the members' word ids when known, else
    their Thai forms, sorted (the live convention: tone:mid-low/far-near)."""
    x, y = sorted(str(m.word_id) if m.word_id is not None else m.thai for m in (a, b))
    return PairId(f"{confusion.id}/{x}-{y}")


def _score(a: Candidate, b: Candidate, shares: Callable[[Candidate, Candidate], bool]) -> tuple:
    words = sum(1 for m in (a, b) if m.word_id is not None)
    ranks = sum(m.rank if m.rank is not None else _UNRANKED for m in (a, b))
    return (-words, not shares(a, b), ranks, a.thai, b.thai)


def _masked(pron: Pronunciation, dimension: Dimension) -> tuple:
    """The pronunciation with `dimension`'s feature blanked in every
    syllable: two candidates can form an exact pair for that dimension
    only if their masked forms are equal (same syllable count, every other
    feature the same), so pairing happens within a bucket, not across
    the pool. A coda difference is `"final"` (spec 1 r20), not
    `"consonant"`, so only `"final"` blanks the coda; `"consonant"` and
    `"aspiration"` blank the onset alone."""
    out = []
    for s in pron.syllables:
        onset, vowel, coda = s.segments
        out.append((
            "" if dimension in ("aspiration", "consonant") else onset,
            "" if dimension == "vowel_quality" else vowel,
            "" if dimension == "final" else coda,
            "" if dimension == "length" else s.vowel_length,
            "" if dimension == "tone" else s.tone,
        ))
    return tuple(out)


def select_pairs(confusion: SoundConfusion, candidates: Sequence[Candidate], *, wanted: int,
                 taken: Collection[str],
                 shares: Callable[[Candidate, Candidate], bool] = lambda a, b: False
                 ) -> list[tuple[Candidate, Candidate]]:
    """At most `wanted` exact pairs for `confusion` out of `candidates`,
    best first: both members Words, then one, then none; a pair `shares`
    says shares a Forvo speaker on the record first; then the lower rank
    sum. A candidate whose pronunciation is not corroborated is never a
    member (E4); no form is used twice, and a form in `taken` (already a
    member of this confusion's pair) is skipped.

    Candidates are bucketed by `_masked` before pairing, so this is
    linear in the pool rather than quadratic: an exact pair can only
    come from within one bucket, and `exact_confusion_violation` still
    has the final say over anything the bucket admits."""
    pool = [c for c in candidates if is_corroborated(c.pron.corroboration) and c.thai not in taken]
    buckets: dict[tuple, list[Candidate]] = {}
    for c in pool:
        buckets.setdefault(_masked(c.pron, confusion.dimension), []).append(c)
    exact = [(a, b) for bucket in buckets.values() if len(bucket) >= 2
             for a, b in combinations(bucket, 2)
             if a.thai != b.thai and exact_confusion_violation(confusion, (a.pron, b.pron)) is None]
    exact.sort(key=lambda ab: _score(ab[0], ab[1], shares))
    chosen: list[tuple[Candidate, Candidate]] = []
    used: set[str] = set()
    for a, b in exact:
        if len(chosen) >= wanted:
            break
        if a.thai in used or b.thai in used:
            continue
        chosen.append((a, b))
        used.update((a.thai, b.thai))
    return chosen
