"""Blocked pairs and their cures (spec 5 r23): each pairs.yaml pair
neither pair rule lets stand (`Syllabus.usable_pairs`), the findings
blocking it and what would cure it. Derived, never stored.

A cure is the first that holds of: `replaced` -- the pair search has
adopted a usable pair for the same confusion since (later in pairs.yaml),
or the confusion's pair need is met without it; `awaiting` -- a
dictionary has not answered a blocking member's form, so that family's
reading can still arrive; `ruling` -- the evidence is all in and the
member stays unsettled, or its spoken witness dissents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .cachekeys import DictionaryKey
from .entities import MinimalPair, is_phrase
from .ids import PairId, WordId
from .pairsearch import wanted
from .ports import CacheReader
from .rulebook import PAIR_EXACT_CONFUSION, PAIR_PRONUNCIATION_SETTLED
from .rules import Finding
from .syllabus import Syllabus

# The dictionaries a form's reading can still arrive from, in the
# settlement's family order (each is its own family's source).
DICTIONARIES = ("wiktionary", "thai-language.com")


@dataclass(frozen=True)
class Cure:
    """What would unblock a pair: `pairs` replace it, `families` are
    awaited, or a ruling is needed."""
    kind: Literal["replaced", "awaiting", "ruling"]
    pairs: tuple[PairId, ...] = ()
    families: tuple[str, ...] = ()

    def text(self) -> str:
        if self.kind == "replaced":
            return f"replaced by {', '.join(self.pairs)}" if self.pairs else "not needed"
        if self.kind == "awaiting":
            return f"awaiting evidence from {', '.join(self.families)}"
        return "needs a ruling"


@dataclass(frozen=True)
class BlockedPair:
    """A blocked pair, its findings and cure, and its blocking members:
    the unsettled or dissenting ones, or every member of a pair whose
    readings left its contrast (any member's reading may be the one
    wrong)."""
    pair: MinimalPair
    findings: tuple[Finding, ...]
    cure: Cure
    blocking: tuple[WordId, ...] = ()


def unanswered(cache: CacheReader, form: str) -> tuple[str, ...]:
    """The dictionaries with no answer on record for `form`; none for a
    phrase, which is never looked up."""
    if is_phrase(form):
        return ()
    return tuple(source for source in DICTIONARIES
                 if cache.latest("provide", source, DictionaryKey(source, form)) is None)


def blocked_pairs(syllabus: Syllabus, cache: CacheReader) -> tuple[BlockedPair, ...]:
    """Every pair not in `syllabus.usable_pairs`, in pairs.yaml order, with
    the pair rules' findings on it and its cure."""
    usable = syllabus.usable_pairs
    if len(usable) == len(syllabus.pairs):
        return ()
    usable_ids = {p.id for p in usable}
    by_member: dict[str, list[Finding]] = {}
    for f in PAIR_PRONUNCIATION_SETTLED.check(syllabus):
        by_member.setdefault(f.note_id, []).append(f)
    by_pair = {f.note_id: f for f in PAIR_EXACT_CONFUSION.check(syllabus)}
    need = wanted(syllabus.confusions, usable)
    out = []
    for i, pair in enumerate(syllabus.pairs):
        if pair.id in usable_ids:
            continue
        findings = [f for m in pair.members for f in by_member.get(m, ())]
        if pair.id in by_pair:
            findings.append(by_pair[pair.id])
        blocking = pair.members if pair.id in by_pair else tuple(
            m for m in pair.members if m in by_member)
        same = [p for p in usable if p.confusion == pair.confusion]
        since = tuple(p.id for p in syllabus.pairs[i + 1:]
                      if p.id in usable_ids and p.confusion == pair.confusion)
        if since or need.get(pair.confusion, 0) == 0:
            cure = Cure("replaced", pairs=since or tuple(p.id for p in same))
        else:
            awaited = {source for m in blocking
                       if (w := syllabus.find_word(m)) is not None
                       for source in unanswered(cache, w.thai)}
            families = tuple(s for s in DICTIONARIES if s in awaited)
            cure = Cure("awaiting", families=families) if families else Cure("ruling")
        out.append(BlockedPair(pair=pair, findings=tuple(findings), cure=cure,
                               blocking=blocking))
    return tuple(out)
