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

from thai_deck_eval.judge.prompts import PICTURE_RULES as _OLD_PICTURE_RULES  # texts kept verbatim

from .entities import exact_confusion_violation
from .rules import Finding, Metric, Rule

if TYPE_CHECKING:
    from .curated import RulebookConfig
    from .entities import Sentence
    from .ids import WordId
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
    "META-1", "F1", "F2", "F3", "F5", "F6", "F7", "E3",
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


# --- rubric constants (judged rules) ----------------------------------------
# PICTURE_FIT_RUBRIC is the three old thai_deck_eval judge rubric texts,
# concatenated verbatim (not paraphrased) so a migrated verdict, cached under
# the old text's hash, still hits the cache under this rule.

PICTURE_FIT_RUBRIC = "\n\n".join(_OLD_PICTURE_RULES[k] for k in (
    "judge/image-off-phrase", "judge/image-irrelevant", "judge/image-embedded-text"))
PICTURE_PREFERENCE_RUBRIC = ("Rank the attached candidates by how well each, as the only picture "
                             "on a flashcard, evokes the word for a learner: concrete, "
                             "unambiguous, no answer-revealing text.")
SENTENCE_FOR_TARGET_RUBRIC = (SENTENCE_REGISTER_RUBRIC
                              + " Is it natural, grammatical, something a native speaker would say?")
RUBRICS_BY_ROLE = {"picture-for-word": PICTURE_FIT_RUBRIC,
                   "picture-preference": PICTURE_PREFERENCE_RUBRIC,
                   "sentence-for-target": SENTENCE_FOR_TARGET_RUBRIC}


def rubrics_for(rules: Sequence[Rule]) -> dict[str, str]:
    """role -> rubric for every judged rule, overlay-aware (a rule's own
    `rubric` wins over RUBRICS_BY_ROLE's default for that role), merged
    over RUBRICS_BY_ROLE so every known role has an entry even if `rules`
    doesn't carry a judged rule for it.
    """
    out = dict(RUBRICS_BY_ROLE)
    for r in rules:
        if r.shape == "judged" and r.rubric:
            out[r.role] = r.rubric
    return out


# --- completeness errors (F3/F5/F6/F7/F1): a targeted word or pair with no
# current-best artifact at all is an error, not a gap metric -- these close
# report().gate the way media/picture-required (an info-severity measure)
# never did.

def _targeted_words(syllabus: "Syllabus") -> list["WordId"]:
    seen, out = set(), []
    for t in syllabus.targets:
        if t.word not in seen:
            seen.add(t.word)
            out.append(t.word)
    return out


def _check_target_picture(syllabus: "Syllabus") -> list[Finding]:
    return [Finding(rule="target/picture-required", note_id=w, evidence="no current-best picture")
           for w in _targeted_words(syllabus) if not syllabus.media.has_picture(w)]


TARGET_PICTURE_REQUIRED = Rule(id="target/picture-required", principle="F3",
                               severity="error", shape="check",
                               check=_check_target_picture)


def _check_target_recording(syllabus: "Syllabus") -> list[Finding]:
    # recording_speakers() alone would false-positive: a media row's
    # speaker_id is nullable, so a real current-best recording with no
    # speaker_id on file yields an empty speaker set. recording_provenance()
    # is None only when there is no current-best recording at all.
    return [Finding(rule="target/recording-required", note_id=w, evidence="no current-best recording")
           for w in _targeted_words(syllabus) if syllabus.media.recording_provenance(w) is None]


TARGET_RECORDING_REQUIRED = Rule(id="target/recording-required", principle="F7",
                                 severity="error", shape="check",
                                 check=_check_target_recording)


def _check_target_sentence(syllabus: "Syllabus") -> list[Finding]:
    return [Finding(rule="target/sentence-required", note_id=t.id, evidence="no adopted sentence fills it")
           for t in syllabus.targets if not any(syllabus.fills(s, t) for s in syllabus.sentences)]


TARGET_SENTENCE_REQUIRED = Rule(id="target/sentence-required", principle="F5",
                                severity="error", shape="check",
                                check=_check_target_sentence)


def _check_pair_rendition(syllabus: "Syllabus") -> list[Finding]:
    # A half-recorded pair (fewer rows than members) has no rendition either
    # -- every member needs its own current-best recording.
    return [Finding(rule="pair/rendition-required", note_id=p.id, evidence="no rendition")
           for p in syllabus.pairs
           if len(syllabus.media.rendition_provenance(p.id)) < len(p.members)]


PAIR_RENDITION_REQUIRED = Rule(id="pair/rendition-required", principle="F1",
                               severity="error", shape="check",
                               check=_check_pair_rendition)


def _check_grapheme_keyword_picture(syllabus: "Syllabus") -> list[Finding]:
    return [Finding(rule="grapheme/keyword-picture-required", note_id=g.symbol,
                    evidence=f"keyword {g.keyword!r} has no picture")
           for g in syllabus.graphemes if not syllabus.media.has_picture(g.keyword)]


GRAPHEME_KEYWORD_PICTURE_REQUIRED = Rule(id="grapheme/keyword-picture-required", principle="F6",
                                         severity="error", shape="check",
                                         check=_check_grapheme_keyword_picture)


# --- synthetic / mixed-speaker warnings (F7/F1) -----------------------------
# A completeness error only asks "is there a current-best artifact at all";
# these ask "is what's current-best actually good enough" -- TTS standing in
# for a human voice, or a minimal pair's two members voiced by different
# speakers (a third confound on top of the sound contrast itself).

def _check_recording_synthetic(syllabus: "Syllabus") -> list[Finding]:
    out = []
    for w in _targeted_words(syllabus):
        prov = syllabus.media.recording_provenance(w)
        if prov and prov.get("speaker_kind") == "synthetic":
            out.append(Finding(rule="recording/synthetic", note_id=w,
                               evidence=f"current-best recording is {prov.get('source')}"))
    return out


RECORDING_SYNTHETIC = Rule(id="recording/synthetic", principle="F7",
                           severity="warn", shape="check",
                           check=_check_recording_synthetic)


def _check_rendition_synthetic(syllabus: "Syllabus") -> list[Finding]:
    out = []
    for p in syllabus.pairs:
        rows = syllabus.media.rendition_provenance(p.id)
        if rows and any(r.get("speaker_kind") == "synthetic" for r in rows):
            out.append(Finding(rule="rendition/synthetic", note_id=p.id, evidence="TTS rendition"))
    return out


RENDITION_SYNTHETIC = Rule(id="rendition/synthetic", principle="F1",
                           severity="warn", shape="check",
                           check=_check_rendition_synthetic)


def _check_rendition_mixed(syllabus: "Syllabus") -> list[Finding]:
    out = []
    for p in syllabus.pairs:
        rows = syllabus.media.rendition_provenance(p.id)
        speakers = {r.get("speaker_id") for r in rows if r and r.get("speaker_id") is not None}
        if len(rows) == len(p.members) and len(speakers) > 1:
            out.append(Finding(rule="rendition/mixed-speakers", note_id=p.id,
                               evidence=f"speakers {sorted(map(str, speakers))}"))
    return out


RENDITION_MIXED_SPEAKERS = Rule(id="rendition/mixed-speakers", principle="F1",
                                severity="warn", shape="check",
                                check=_check_rendition_mixed)


def _check_sentence_synthetic_productive(syllabus: "Syllabus") -> list[Finding]:
    out = []
    for s in syllabus.sentences:
        productive = [t for t in syllabus.targets if t.skill == "productive" and syllabus.fills(s, t)]
        if not productive:
            continue
        prov = syllabus.media.recording_provenance(sentence_note_id(s))
        if prov and prov.get("speaker_kind") == "synthetic":
            out.append(Finding(rule="sentence/synthetic-productive", note_id=sentence_note_id(s),
                               evidence="productive sentence carries TTS audio"))
    return out


SENTENCE_SYNTHETIC_PRODUCTIVE = Rule(id="sentence/synthetic-productive", principle="F7",
                                     severity="warn", shape="check",
                                     check=_check_sentence_synthetic_productive)


# --- picture/fit (judged, F3) ------------------------------------------------
# Whether the CURRENT picture actually fits the word -- distinct from the
# completeness question (is there one at all). role="picture-for-word"
# names the AssessmentReader verdict this reads (spec 4's judge role).

def _picture_fit_subjects(syllabus: "Syllabus") -> list[tuple[str, str | None]]:
    return [(w, syllabus.media.picture_sha(w)) for w in _targeted_words(syllabus)
           if syllabus.media.has_picture(w)]


PICTURE_FIT = Rule(id="picture/fit", principle="F3", severity="warn", shape="judged",
                   rubric=PICTURE_FIT_RUBRIC, role="picture-for-word",
                   judged_subjects=_picture_fit_subjects)


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
    TARGET_PICTURE_REQUIRED,
    TARGET_RECORDING_REQUIRED,
    TARGET_SENTENCE_REQUIRED,
    PAIR_RENDITION_REQUIRED,
    GRAPHEME_KEYWORD_PICTURE_REQUIRED,
    RECORDING_SYNTHETIC,
    RENDITION_SYNTHETIC,
    RENDITION_MIXED_SPEAKERS,
    SENTENCE_SYNTHETIC_PRODUCTIVE,
    PICTURE_FIT,
]
