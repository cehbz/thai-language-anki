"""The pronunciation settlement (spec 3 section 6 settlement(), r68; spec
1 r34): each form's reading folded from every reading of it on record --
the local engines, Wiktionary, thai-language.com and the judge's newest
fresh pronunciation verdict. Derived, never written.

Per feature (segments, vowel length, tone) a family-level Dawid-Skene
one-coin fit over three error families: Wiktionary (Wiktionary,
thaig2p), the spelling rules (tltk, the rule tone engine, the judge) and
thai-language.com. Members of a family share their errors, so they add
no votes: the family reports what its most accurate member present
reports. A family's accuracy is fitted by EM over every form two or more
families read; a member's accuracy is scored only against the posterior
from the other families, so co-family members never vouch for each
other. A report of several readings (a Wiktionary entry listing spoken
variants) is a set that matches any of its values. Every form's fit
carries an implicit value no member reported (OTHER), so a value one
family alone reports settles at about that family's accuracy.

`evidence_on_record` gathers the evidence; `settle` is the pure fold,
memoised in-process on the evidence; `settlement_of` is the two over a
set of Words (and any outside forms), what the loader and the run read
(spec 2 r23's Settlement interface: `Settlement.reading(form)`).
"""
from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Union

from .cachekeys import DictionaryKey, EngineKey
from .derivations import CANDIDATE_SUBJECT_PREFIX, stale
from .engines import RULE_TONE_VERSION, THAIG2P_VERSION, TLTK_VERSION, convert_wiktionary
from .entities import Pronunciation, Syllable, Tone, Word
from .phonology import syllables_from_verdict
from .ports import CacheReader
from .thailanguagecom import readings as thai_language_com_readings

Reading = tuple[Syllable, ...]
# A member's report of a form: a reading, or the rule tone engine's tone
# of a monosyllable (its only feature).
Report = Union[Reading, Tone]

FEATURES = ("segments", "length", "tone")

# Member source -> error family. The order is the members' order: it
# breaks the first iteration's ties, when every member's accuracy is the
# prior.
FAMILY = {"wiktionary": "wiktionary", "thaig2p": "wiktionary",
          "judge": "spelling", "tltk": "spelling", "rule-tone": "spelling",
          "thai-language.com": "thai-language.com"}
_ORDER = {source: i for i, source in enumerate(FAMILY)}

# The one source that writes the spoken form (spec 3 section 6).
SPOKEN_WITNESS = "wiktionary"

# The installed local engines' versions, the members `evidence_on_record` reads.
ENGINE_VERSIONS = {"thaig2p": THAIG2P_VERSION, "tltk": TLTK_VERSION,
                   "rule-tone": RULE_TONE_VERSION}

ITERATIONS = 50
# The implicit candidate every fit carries: a value no member reported.
# A family's error mass then always has somewhere to go, so agreement
# scores by the agreeing families' accuracies (one family alone at p
# gives about p). It never wins: a family's accuracy is at least 0.5, so
# a reported value scores at least OTHER's, and a tie goes to the
# reported value.
OTHER = None
PRIOR = 0.8
# A family's fitted accuracy stays within these: below 0.5 it would be
# read as an anti-oracle, at 1 one family would decide alone.
FAMILY_FLOOR, FAMILY_CEILING = 0.5, 0.995


@dataclass(frozen=True)
class Member:
    """One oracle: a source, and for a local engine its version (each
    engine version is a member)."""
    source: str
    version: str = ""

    @property
    def family(self) -> str:
        return FAMILY[self.source]

    def rank(self) -> tuple[int, str]:
        """The member order (FAMILY's, then version)."""
        return _ORDER[self.source], self.version


@dataclass(frozen=True)
class Evidence:
    """Every member's reports of every form, frozen and canonically
    ordered so equal evidence hashes equal (the memo's key). A member
    with no report of a form is absent; so is a form with none."""
    reports: tuple[tuple[str, tuple[tuple[Member, tuple[Report, ...]], ...]], ...]

    @classmethod
    def of(cls, by_form: Mapping[str, Mapping[Member, Iterable[Report]]]) -> Evidence:
        rows = []
        for form in sorted(by_form):
            members = tuple(sorted(
                ((m, tuple(dict.fromkeys(rs))) for m, rs in by_form[form].items()),
                key=lambda mr: mr[0].rank()))
            members = tuple((m, rs) for m, rs in members if rs)
            if members:
                rows.append((form, members))
        return cls(tuple(rows))


@dataclass(frozen=True)
class Settlement:
    """The fold's result: a reading per form with any (the entity
    Pronunciation: each feature's winner, the least winning posterior,
    and whether the spoken witness dissents), and the fitted accuracies
    per feature (family name -> accuracy, member -> accuracy). Shared
    through the memo, so read-only by convention."""
    readings: Mapping[str, Pronunciation]
    accuracy: Mapping[str, Mapping[str, float]]
    member_accuracy: Mapping[str, Mapping[Member, float]]

    def reading(self, form: str) -> Pronunciation | None:
        """The form's settled reading; None when no evidence reads it."""
        return self.readings.get(form)


def _values(reports: tuple[Report, ...], feature: str) -> frozenset[tuple]:
    """The feature's values among a member's reports of one form."""
    out = set()
    for r in reports:
        if isinstance(r, str):
            if feature == "tone":
                out.add((r,))
        elif feature == "segments":
            out.add(tuple(s.segments for s in r))
        elif feature == "length":
            out.add(tuple(s.vowel_length for s in r))
        else:
            out.add(tuple(s.tone for s in r))
    return frozenset(out)


class _Fit:
    """One feature's EM fit over the forms' reports (`rep`: form ->
    member -> values, members in member order)."""

    def __init__(self, rep: dict[str, dict[Member, frozenset[tuple]]]) -> None:
        self.rep = rep
        self.cand = {f: sorted(set().union(*vs.values())) for f, vs in rep.items()}
        members = sorted({m for vs in rep.values() for m in vs}, key=Member.rank)
        self.families = sorted({m.family for m in members})
        # form -> family -> its members reading the form, families in order
        self.present = {f: {F: [m for m in vs if m.family == F]
                            for F in self.families if any(m.family == F for m in vs)}
                        for f, vs in rep.items()}
        # Only a form two or more families read says anything about accuracy.
        shared = [f for f in rep if len(self.present[f]) >= 2]
        read_by = {m: [f for f in shared if m in rep[f]] for m in members}
        self.acc = {m: PRIOR for m in members}
        self.p = {F: PRIOR for F in self.families}
        for _ in range(ITERATIONS):
            post = {f: self.posterior(f, self.families) for f in shared}
            for F in self.families:
                hits = [max(post[f].get(v, 0.0) for v in self.family_report(f, F))
                        for f in shared if F in self.present[f]]
                if hits:
                    self.p[F] = min(max(sum(hits) / len(hits), FAMILY_FLOOR), FAMILY_CEILING)
            for m in members:
                others = [G for G in self.families if G != m.family]
                hits = []
                for f in read_by[m]:
                    left_out = self.posterior(f, others)
                    hits.append(max(left_out.get(v, 0.0) for v in rep[f][m]))
                if hits:
                    self.acc[m] = sum(hits) / len(hits)
        self.post = {f: self.posterior(f, self.families) for f in rep}

    def family_report(self, f: str, F: str) -> frozenset[tuple]:
        """What the family's most accurate member present reports."""
        present = self.present[f][F]
        if len(present) == 1:
            return self.rep[f][present[0]]
        return self.rep[f][max(present, key=lambda m: self.acc[m])]

    def posterior(self, f: str, families: Iterable[str]) -> dict[tuple | None, float]:
        """Each candidate value's posterior, and OTHER's (None)."""
        C = self.cand[f]
        if not C:
            return {}
        miss = len(C)    # K - 1, K counting OTHER
        lp = dict.fromkeys([*C, OTHER], 0.0)
        for F in families:
            if F not in self.present[f]:
                continue
            r = self.family_report(f, F)
            hit, wrong = math.log(self.p[F]), math.log((1 - self.p[F]) / miss)
            for v in lp:
                lp[v] += hit if v in r else wrong
        top = max(lp.values())
        z = sum(math.exp(x - top) for x in lp.values())
        return {v: math.exp(x - top) / z for v, x in lp.items()}


@lru_cache(maxsize=8)
def settle(evidence: Evidence) -> Settlement:
    """Each form's settled reading, from a fit per feature over all the
    evidence. The segments winner fixes the syllable count; the length
    and tone winners are taken among the values of that count, so the
    three assemble into syllables. A form whose reports carry no segments
    (the rule tone engine's alone) has no reading."""
    fits = {feature: _Fit({form: {m: vs for m, rs in members
                                  if (vs := _values(rs, feature))}
                           for form, members in evidence.reports})
            for feature in FEATURES}
    readings = {}
    for form, members in evidence.reports:
        segments = _winner(fits["segments"].post[form], None)
        if segments is None:
            continue
        seg, p_seg = segments
        lengths, p_len = _winner(fits["length"].post[form], len(seg))
        tones, p_tone = _winner(fits["tone"].post[form], len(seg))
        syllables = tuple(Syllable(segments=s, vowel_length=ln, tone=t)
                          for s, ln, t in zip(seg, lengths, tones))
        settled = {"segments": seg, "length": lengths, "tone": tones}
        witness = [rs for m, rs in members if m.source == SPOKEN_WITNESS]
        dissent = bool(witness) and any(
            settled[feature] not in _values(witness[0], feature) for feature in FEATURES)
        readings[form] = Pronunciation(syllables=syllables,
                                       posterior=min(p_seg, p_len, p_tone),
                                       spoken_dissent=dissent)
    return Settlement(readings=readings,
                      accuracy={feature: dict(fit.p) for feature, fit in fits.items()},
                      member_accuracy={feature: dict(fit.acc) for feature, fit in fits.items()})


def _winner(post: Mapping[tuple | None, float], count: int | None) -> tuple[tuple, float] | None:
    """The value of highest posterior (the first in value order on a
    tie), among those of `count` syllables when given: a length or tone
    vector is assembled onto the winning segments, so it must have their
    syllable count."""
    eligible = [(v, p) for v, p in post.items()
                if v is not OTHER and (count is None or len(v) == count)]
    if not eligible:
        return None
    return max(eligible, key=lambda vp: vp[1])


def evidence_on_record(cache: CacheReader, judge_subjects: Mapping[str, Collection[str]], *,
                       current_rubric: Mapping[str, str],
                       versions: Mapping[str, str] = ENGINE_VERSIONS) -> Evidence:
    """Every reading on record of each form in `judge_subjects`: the local
    engines' rows at `versions`, Wiktionary's and thai-language.com's
    rows, and the newest fresh pronunciation-for-word verdict among the
    form's subjects (`judge_subjects[form]`: its Words' ids, its
    `candidate:` subject)."""
    by_form: dict[str, dict[Member, list[Report]]] = {}
    for form, subjects in judge_subjects.items():
        reports: dict[Member, list[Report]] = {}
        for engine, version in versions.items():
            row = cache.latest("provide", engine, EngineKey(engine, version, form))
            value = None if row is None else row.answer.get("value")
            if value is not None:
                reports[Member(engine, version)] = [
                    value if isinstance(value, str) else syllables_from_verdict({"syllables": value})]
        row = cache.latest("provide", "wiktionary", DictionaryKey("wiktionary", form))
        if row is not None:
            reports[Member("wiktionary")] = [
                r for r in (convert_wiktionary(ipa) for ipa in row.answer.get("ipa", [])) if r]
        row = cache.latest("provide", "thai-language.com", DictionaryKey("thai-language.com", form))
        if row is not None:
            reports[Member("thai-language.com")] = list(thai_language_com_readings(row.answer))
        verdicts = [r for s in subjects for r in cache.assessments_of(s)
                    if r.port == "assess" and r.backend == "judge"
                    and r.question.get("role") == "pronunciation-for-word"
                    and not stale(r, current_rubric)]
        if verdicts:
            value = max(verdicts, key=lambda r: r.ts).answer.get("value")
            if isinstance(value, Mapping) and value.get("syllables"):
                reports[Member("judge")] = [syllables_from_verdict(value)]
        by_form[form] = reports
    return Evidence.of(by_form)


def judge_subjects(words: Iterable[Word], forms: Iterable[str] = ()
                   ) -> dict[str, list[str]]:
    """Each form's judge subjects, in order: its Words' ids, then the
    `candidate:<form>` subject the pair search asks an outside form under
    (a Word minted from one keeps that verdict); `forms` adds outside
    forms with the candidate subject alone."""
    out: dict[str, list[str]] = {}
    for w in words:
        out.setdefault(w.thai, []).append(w.id)
    for form in forms:
        out.setdefault(form, [])
    for form, subjects in out.items():
        subjects.append(f"{CANDIDATE_SUBJECT_PREFIX}{form}")
    return out


def settlement_of(cache: CacheReader, words: Iterable[Word], forms: Iterable[str] = (), *,
                  current_rubric: Mapping[str, str]) -> Settlement:
    """The settlement over the readings on record of `words`' forms and
    `forms`; reads the record only, never runs an engine."""
    return settle(evidence_on_record(cache, judge_subjects(words, forms),
                                     current_rubric=current_rubric))


def with_readings(words: Iterable[Word], settlement: Settlement) -> tuple[Word, ...]:
    """`words`, each given its form's settled reading (spec 1 r34)."""
    return tuple(replace(w, pron=settlement.reading(w.thai)) for w in words)
