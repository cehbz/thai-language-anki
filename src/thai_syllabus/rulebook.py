"""The rule registry (spec 1, section 4): an explicit module-level list, no
import side effects.

Seeded with 8 rules against the principles draft (2026-09-01-principles-
draft.md): a representative slice, not the full doctrine -- spec section 4
says the exact list is "enumerated at implementation-plan time against the
locked principles." `ENFORCEMENT_PRINCIPLES` here is therefore scoped to
just the principles this seed actually covers, not the whole draft's A/F/E
list; `test_the_shipped_rulebook_is_internally_traceable` checks that
scoped claim, and `traceability_metric` itself is exercised independently
against synthetic registries in tests/syllabus/test_rulebook.py.
"""
import dataclasses
import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .entities import exact_confusion_violation
from .rules import Finding, Metric, Rule

if TYPE_CHECKING:
    from .curated import RulebookConfig
    from .entities import Sentence
    from .syllabus import Syllabus

# Every principle id from the principles draft with enforcement intent
# (i.e. not a pure [study]/[ask] research item). "META-1" is this
# implementation's own id for the charter's traceability meta-rule, which
# the draft states in prose (section on Rule) but does not number.
PRINCIPLES: frozenset[str] = frozenset({
    "META-1",
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "F1", "F2", "F3", "F4", "F5", "F6", "F6a", "F6b", "F7", "F8", "F9",
    "F10", "F11", "F12",
    "E1", "E2", "E3", "E4", "E5", "E6",
})

# The subset of PRINCIPLES this seed actually implements a rule for.
ENFORCEMENT_PRINCIPLES: frozenset[str] = frozenset({
    "META-1", "F1", "F2", "F3", "F5", "F6", "E3",
})


def sentence_note_id(sentence: "Sentence") -> str:
    """Sentence has no id field (identity = text + provenance); this is a
    stable, content-derived note_id for Findings and judged subjects.
    """
    basis = json.dumps({
        "text": sentence.text,
        "source": sentence.provenance.source,
        "origin": sentence.provenance.origin,
        "acquired": str(sentence.provenance.acquired),
    }, sort_keys=True)
    return "sentence:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


# --- pair/exact-confusion ---------------------------------------------------
# Re-checks MinimalPair.create's invariant against loaded data (pairs built
# directly, bypassing the factory -- e.g. read from a store in spec 2).

def _check_pair_exact_confusion(syllabus: "Syllabus") -> list[Finding]:
    confusions = {c.id: c for c in syllabus.confusions}
    findings: list[Finding] = []
    for pair in syllabus.pairs:
        confusion = confusions.get(pair.confusion)
        if confusion is None:
            findings.append(Finding(rule="pair/exact-confusion", note_id=pair.id,
                                    evidence=f"unknown confusion {pair.confusion!r}"))
            continue
        members = [syllabus.find_word(m) for m in pair.members]
        if any(m is None for m in members):
            findings.append(Finding(rule="pair/exact-confusion", note_id=pair.id,
                                    evidence="a member word does not resolve"))
            continue
        reason = exact_confusion_violation(confusion, tuple(m.pron for m in members))
        if reason is not None:
            findings.append(Finding(rule="pair/exact-confusion", note_id=pair.id,
                                    evidence=reason))
    return findings


PAIR_EXACT_CONFUSION = Rule(id="pair/exact-confusion", principle="F1",
                            severity="error", shape="check",
                            check=_check_pair_exact_confusion)


# --- grapheme/keyword-contains-symbol ---------------------------------------

def _check_grapheme_keyword(syllabus: "Syllabus") -> list[Finding]:
    findings: list[Finding] = []
    for g in syllabus.graphemes:
        keyword = syllabus.find_word(g.keyword)
        if keyword is None:
            findings.append(Finding(rule="grapheme/keyword-contains-symbol",
                                    note_id=g.symbol,
                                    evidence=f"keyword {g.keyword!r} does not resolve"))
        elif g.symbol not in keyword.thai:
            findings.append(Finding(rule="grapheme/keyword-contains-symbol",
                                    note_id=g.symbol,
                                    evidence=f"{g.symbol!r} not in {keyword.thai!r}"))
    return findings


GRAPHEME_KEYWORD_CONTAINS_SYMBOL = Rule(id="grapheme/keyword-contains-symbol",
                                        principle="F6", severity="error",
                                        shape="check", check=_check_grapheme_keyword)


# --- sentence/fills-novelty --------------------------------------------------
# F5: a sentence's permitted new-word count. Flags every (sentence, target)
# where the sentence mentions the target's word in the right voice but
# fills() rejects it on the novelty clause.

def _check_sentence_fills_novelty(syllabus: "Syllabus") -> list[Finding]:
    findings: list[Finding] = []
    for s in syllabus.sentences:
        for t in syllabus.targets:
            target_word = syllabus.find_word(t.word)
            if target_word is None or not syllabus.mentions(s, target_word.thai):
                continue
            if t.skill == "productive" and s.voice != "learner_voice":
                continue
            if not syllabus.fills(s, t):
                findings.append(Finding(
                    rule="sentence/fills-novelty", note_id=sentence_note_id(s),
                    evidence=f"exceeds the novelty budget for target {t.id!r}"))
    return findings


SENTENCE_FILLS_NOVELTY = Rule(id="sentence/fills-novelty", principle="F5",
                              severity="error", shape="check",
                              check=_check_sentence_fills_novelty)


# --- syllabus/closure ---------------------------------------------------
# Every referenced WordId resolves: target words, word classifiers, pair
# members, grapheme keywords.

def _check_closure(syllabus: "Syllabus") -> list[Finding]:
    known = {w.id for w in syllabus.words}
    findings: list[Finding] = []
    for t in syllabus.targets:
        if t.word not in known:
            findings.append(Finding(rule="syllabus/closure", note_id=t.id,
                                    evidence=f"target references unknown word {t.word!r}"))
    for w in syllabus.words:
        if w.classifier is not None and w.classifier not in known:
            findings.append(Finding(rule="syllabus/closure", note_id=w.id,
                                    evidence=f"classifier references unknown word {w.classifier!r}"))
    for p in syllabus.pairs:
        for m in p.members:
            if m not in known:
                findings.append(Finding(rule="syllabus/closure", note_id=p.id,
                                        evidence=f"pair references unknown word {m!r}"))
    for g in syllabus.graphemes:
        if g.keyword not in known:
            findings.append(Finding(rule="syllabus/closure", note_id=g.symbol,
                                    evidence=f"grapheme keyword references unknown word {g.keyword!r}"))
    return findings


SYLLABUS_CLOSURE = Rule(id="syllabus/closure", principle="F2", severity="error",
                        shape="check", check=_check_closure)


# --- media/picture-required (gap-metric) -------------------------------------
# F3: a picture carries meaning. Measures how many targeted words still
# lack one; gaps() reads the same underlying facts.

def _measure_picture_required(syllabus: "Syllabus") -> Metric:
    targeted = {t.word for t in syllabus.targets}
    missing = sorted(w for w in targeted if not syllabus.media.has_picture(w))
    total = len(targeted) or 1
    return Metric(rule="media/picture-required", value=len(missing) / total,
                 detail={"missing": missing})


MEDIA_PICTURE_REQUIRED = Rule(id="media/picture-required", principle="F3",
                              severity="info", shape="measure",
                              measure=_measure_picture_required)


# --- coverage/confusions --------------------------------------------------
# F1: sound system first. Coverage per trained confusion = pairs x distinct
# speakers.

def _measure_coverage_confusions(syllabus: "Syllabus") -> Metric:
    detail: dict[str, dict[str, int]] = {}
    for c in syllabus.confusions:
        pair_count = sum(1 for p in syllabus.pairs if p.confusion == c.id)
        speakers = syllabus.media.rendition_speakers(c.id)
        detail[c.id] = {"pairs": pair_count, "speakers": len(speakers)}
    covered = sum(1 for d in detail.values() if d["pairs"] >= 1 and d["speakers"] >= 1)
    value = covered / len(detail) if detail else 1.0
    return Metric(rule="coverage/confusions", value=value, detail=detail)


COVERAGE_CONFUSIONS = Rule(id="coverage/confusions", principle="F1",
                           severity="info", shape="measure",
                           measure=_measure_coverage_confusions)


# --- sentence/register-natural (judged) --------------------------------------
# E3: register (male colloquial). A judged rule: report() only reads cached
# verdicts through the AssessmentReader, never calls the judge itself.

SENTENCE_REGISTER_RUBRIC = (
    "Does this sentence read as natural male colloquial Central Thai, in "
    "the learner's register?"
)


def _sentence_register_subjects(syllabus: "Syllabus") -> list[tuple[str, str | None]]:
    return [(sentence_note_id(s), None) for s in syllabus.sentences
           if s.voice == "learner_voice"]


SENTENCE_REGISTER_NATURAL = Rule(id="sentence/register-natural", principle="E3",
                                 severity="warn", shape="judged",
                                 rubric=SENTENCE_REGISTER_RUBRIC,
                                 judged_subjects=_sentence_register_subjects)


# --- rulebook/traceability -------------------------------------------------
# "Traceability is itself a measure": every rule names a live principle;
# every principle with enforcement intent names >=1 rule.

def traceability_metric(rules: Sequence[Rule], known_principles: frozenset[str],
                        enforcement_principles: frozenset[str]) -> Metric:
    orphan_rules = sorted(r.id for r in rules if r.principle not in known_principles)
    used = {r.principle for r in rules}
    unenforced_principles = sorted(p for p in enforcement_principles if p not in used)
    value = 1.0 if not orphan_rules and not unenforced_principles else 0.0
    return Metric(rule="rulebook/traceability", value=value,
                 detail={"orphan_rules": orphan_rules,
                         "unenforced_principles": unenforced_principles})


def _measure_traceability(syllabus: "Syllabus") -> Metric:
    del syllabus  # traceability is a property of the registry, not the aggregate
    return traceability_metric(RULES, PRINCIPLES, ENFORCEMENT_PRINCIPLES)


RULEBOOK_TRACEABILITY = Rule(id="rulebook/traceability", principle="META-1",
                             severity="info", shape="measure",
                             measure=_measure_traceability)


# --- rulebook overlay (spec 3) ----------------------------------------------
# Applies a curated RulebookConfig's severity/rubric overrides on top of the
# code-defined registry, leaving RULES itself untouched.

def apply_overlay(rules: Sequence[Rule], config: "RulebookConfig") -> tuple[Rule, ...]:
    out = []
    for r in rules:
        changes: dict = {}
        if r.id in config.severities:
            changes["severity"] = config.severities[r.id]
        if r.shape == "judged" and r.id in config.rubrics:
            changes["rubric"] = config.rubrics[r.id]
        out.append(dataclasses.replace(r, **changes) if changes else r)
    return tuple(out)


RULES: list[Rule] = [
    PAIR_EXACT_CONFUSION,
    GRAPHEME_KEYWORD_CONTAINS_SYMBOL,
    SENTENCE_FILLS_NOVELTY,
    SYLLABUS_CLOSURE,
    MEDIA_PICTURE_REQUIRED,
    COVERAGE_CONFUSIONS,
    SENTENCE_REGISTER_NATURAL,
    RULEBOOK_TRACEABILITY,
]
