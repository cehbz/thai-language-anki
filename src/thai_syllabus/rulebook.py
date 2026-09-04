"""The rule registry (spec 1, section 4): an explicit module-level list, no
import side effects. Every rule spec 1 r2 section 4's table names against
docs/principles.md r2 is registered here; `RULES` is the enumeration.
"""
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

from .entities import Target, exact_confusion_violation, is_corroborated
from .rules import Finding, Metric, Rule

if TYPE_CHECKING:
    from .curated import RulebookConfig
    from .entities import Sentence
    from .ids import WordId
    from .syllabus import Syllabus

# Every principle id docs/principles.md r2 numbers, plus "META-1", this
# implementation's own id for the charter's traceability meta-rule (stated
# in prose, never numbered by the doc).
PRINCIPLES: frozenset[str] = frozenset({
    "META-1",
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "F1", "F2", "F3", "F4", "F5", "F6", "F6a", "F6b", "F7", "F8", "F9",
    "F10", "F11", "F12",
    "E1", "E2", "E3", "E4", "E5", "E6", "E7",
})


def sentence_note_id(sentence: "Sentence") -> str:
    """The note_id for Findings and judged subjects on a sentence."""
    return sentence.text_sha


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


# --- category/single-membership ---------------------------------------------
# F2: a word belongs to at most one Category.

def _check_category_single_membership(syllabus: "Syllabus") -> list[Finding]:
    seen: set["WordId"] = set()
    duplicates: list["WordId"] = []
    for cat in syllabus.categories:
        for word_id in cat.members:
            if word_id in seen and word_id not in duplicates:
                duplicates.append(word_id)
            seen.add(word_id)
    return [Finding(rule="category/single-membership", note_id=word_id,
                    evidence="word is a member of more than one category")
           for word_id in duplicates]


CATEGORY_SINGLE_MEMBERSHIP = Rule(id="category/single-membership", principle="F2",
                                  severity="error", shape="check",
                                  check=_check_category_single_membership)


# --- coverage/categories ----------------------------------------------------
# F2: fraction of categories with at least one targeted word.

def _measure_coverage_categories(syllabus: "Syllabus") -> Metric:
    targeted = {t.word for t in syllabus.targets}
    covered = sorted(cat.name for cat in syllabus.categories
                     if cat.members & targeted)
    value = len(covered) / len(syllabus.categories) if syllabus.categories else 1.0
    return Metric(rule="coverage/categories", value=value, detail={"covered": covered})


COVERAGE_CATEGORIES = Rule(id="coverage/categories", principle="F2",
                           severity="info", shape="measure",
                           measure=_measure_coverage_categories)


# --- coverage/confusions --------------------------------------------------
# F1: sound system first. Coverage per trained confusion = pairs x distinct
# speakers.

def _measure_coverage_confusions(syllabus: "Syllabus") -> Metric:
    detail: dict[str, dict[str, int | bool]] = {}
    for c in syllabus.confusions:
        pair_count = sum(1 for p in syllabus.pairs if p.confusion == c.id)
        speakers = syllabus.media.rendition_speakers(c.id)
        covered = pair_count >= 1 and len(speakers) >= 1
        detail[c.id] = {"pairs": pair_count, "speakers": len(speakers), "covered": covered}
    covered_count = sum(1 for d in detail.values() if d["covered"])
    value = covered_count / len(detail) if detail else 1.0
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
# PICTURE_FIT_RUBRIC is the three old picture/fit judge rubric texts,
# concatenated verbatim (not paraphrased) so a migrated verdict, cached under
# the old text's hash, still hits the cache under this rule.

PICTURE_FIT_RUBRIC = (
    "Does the image show what the intended phrase describes? This asks "
    "only whether the search found what it was looking for. Pass if no "
    "phrase is given.\n\n"
    "Would this image, as a picture on a flashcard, evoke the word for a "
    "learner? An abstract word is served by a scene that cues it, not by "
    "a literal depiction -- a person pointing at their own chest evokes "
    "\"I\", two apples evoke \"two\". Scale the bar to the card: when a "
    "gloss is shown the image only has to support it, so an image that "
    "fits the glossed sense passes even if it would not have evoked the "
    "word unaided; when no gloss is shown the image carries the meaning "
    "alone and must evoke the word by itself. If it fails, give a "
    "`suggestion`: the search phrase that would have found a better "
    "picture.\n\n"
    "Fail only if text in the image reveals the answer: the Thai word "
    "itself, its English translation, or a romanized spelling of it. "
    "Incidental text passes -- watermarks, photographer credits, shop "
    "signage, product packaging, text in unrelated languages. The rule "
    "exists so the picture cannot give away the word, not to require a "
    "text-free photograph."
)
PICTURE_PREFERENCE_RUBRIC = ("Rank the attached candidates by how well each, as the only picture "
                             "on a flashcard, evokes the word for a learner: concrete, "
                             "unambiguous, no answer-revealing text.")
SENTENCE_FOR_TARGET_RUBRIC = (SENTENCE_REGISTER_RUBRIC
                              + " Is it natural, grammatical, something a native speaker would say?")


def rubrics_for(rules: Sequence[Rule]) -> dict[str, str]:
    """role -> rubric text, for every judged rule in `rules`."""
    return {r.role: r.rubric for r in rules if r.shape == "judged" and r.rubric}


# --- completeness errors (F3/F5/F6/F7/F1): a targeted word or pair with no
# current-best artifact at all is an error, not a gap metric.

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

def _speaker_kind(prov: Mapping[str, Any] | None) -> str | None:
    speaker = prov.get("speaker") if prov else None
    return speaker.kind if speaker is not None else None


def _check_recording_synthetic(syllabus: "Syllabus") -> list[Finding]:
    out = []
    for w in _targeted_words(syllabus):
        prov = syllabus.media.recording_provenance(w)
        if _speaker_kind(prov) == "synthetic":
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
        if rows and any(_speaker_kind(r) == "synthetic" for r in rows):
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
        if _speaker_kind(prov) == "synthetic":
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


# --- picture/preference (judged, F3) -----------------------------------------
# Ranks candidate pictures against each other (derivations._apply_preference
# reads the cached ranking directly); report() asks no subject about it.

def _no_judged_subjects(syllabus: "Syllabus") -> list[tuple[str, str | None]]:
    del syllabus
    return []


PICTURE_PREFERENCE = Rule(id="picture/preference", principle="F3", severity="info",
                          shape="judged", rubric=PICTURE_PREFERENCE_RUBRIC,
                          role="picture-preference", judged_subjects=_no_judged_subjects)


# --- coverage/speakers (E7) --------------------------------------------------
# Speaker diversity per audio corpus: distinct speakers, and how many of them
# carry each known sex/age_band/region -- unknown attributes never count.

_SPEAKER_CORPORA: tuple[str, ...] = ("recording", "rendition", "sentence")
_SPEAKER_ATTRIBUTES: tuple[str, ...] = ("sex", "age_band", "region")


def _measure_coverage_speakers(syllabus: "Syllabus") -> Metric:
    detail: dict[str, dict[str, Any]] = {}
    covered = 0
    for corpus in _SPEAKER_CORPORA:
        speakers = syllabus.media.speakers_of(corpus)
        counts: dict[str, dict[str, int]] = {attr: {} for attr in _SPEAKER_ATTRIBUTES}
        for speaker in speakers:
            for attr in _SPEAKER_ATTRIBUTES:
                value = getattr(speaker, attr)
                if value != "unknown":
                    counts[attr][value] = counts[attr].get(value, 0) + 1
        detail[corpus] = {"speakers": len(speakers), **counts}
        if speakers:
            covered += 1
    value = covered / len(_SPEAKER_CORPORA)
    return Metric(rule="coverage/speakers", value=value, detail=detail)


COVERAGE_SPEAKERS = Rule(id="coverage/speakers", principle="E7", severity="info",
                         shape="measure", measure=_measure_coverage_speakers)


# --- word/pronunciation-corroborated (E4) ------------------------------------
# A disputed pronunciation blocks the word's card.

def _check_word_pronunciation_corroborated(syllabus: "Syllabus") -> list[Finding]:
    findings = []
    for word_id in _targeted_words(syllabus):
        target_word = syllabus.find_word(word_id)
        if target_word is not None and not is_corroborated(target_word.pron.corroboration):
            findings.append(Finding(rule="word/pronunciation-corroborated", note_id=target_word.id,
                                    evidence=f"pronunciation is {target_word.pron.corroboration!r}"))
    return findings


WORD_PRONUNCIATION_CORROBORATED = Rule(id="word/pronunciation-corroborated", principle="E4",
                                       severity="error", shape="check",
                                       check=_check_word_pronunciation_corroborated)


# --- word/classifier-known (E5) ----------------------------------------------

def _check_word_classifier_known(syllabus: "Syllabus") -> list[Finding]:
    known = {w.id for w in syllabus.words}
    return [Finding(rule="word/classifier-known", note_id=w.id,
                    evidence=f"classifier references unknown word {w.classifier!r}")
           for w in syllabus.words if w.classifier is not None and w.classifier not in known]


WORD_CLASSIFIER_KNOWN = Rule(id="word/classifier-known", principle="E5", severity="warn",
                             shape="check", check=_check_word_classifier_known)


# --- sentence/recording-required (F7) ----------------------------------------
# Mirrors target/recording-required's completeness shape, subject = text_sha.

def _check_sentence_recording(syllabus: "Syllabus") -> list[Finding]:
    return [Finding(rule="sentence/recording-required", note_id=sentence_note_id(s),
                    evidence="no current-best recording")
           for s in syllabus.sentences
           if syllabus.media.recording_provenance(sentence_note_id(s)) is None]


SENTENCE_RECORDING_REQUIRED = Rule(id="sentence/recording-required", principle="F7",
                                   severity="error", shape="check",
                                   check=_check_sentence_recording)


# --- order constraint checks (F8, E1) ----------------------------------------
# Checks over order()'s own shape: OrderEntry.kind names pair, grapheme,
# word_target and sentence entries.

def _check_order_sounds_first(syllabus: "Syllabus") -> list[Finding]:
    entries = list(syllabus.order())
    sound_idxs = [i for i, e in enumerate(entries) if e.kind in ("pair", "grapheme")]
    target_idxs = [i for i, e in enumerate(entries) if e.kind == "word_target"]
    if sound_idxs and target_idxs and max(sound_idxs) > min(target_idxs):
        return [Finding(rule="order/sounds-first", note_id="order",
                        evidence="a pair or grapheme entry follows a word target")]
    return []


ORDER_SOUNDS_FIRST = Rule(id="order/sounds-first", principle="F8", severity="error",
                          shape="check", check=_check_order_sounds_first)


def _check_order_reading_after_graphemes(syllabus: "Syllabus") -> list[Finding]:
    entries = list(syllabus.order())
    grapheme_idxs = [i for i, e in enumerate(entries) if e.kind == "grapheme"]
    target_idxs = [i for i, e in enumerate(entries) if e.kind == "word_target"]
    if grapheme_idxs and target_idxs and max(grapheme_idxs) > min(target_idxs):
        return [Finding(rule="order/reading-after-graphemes", note_id="order",
                        evidence="a grapheme entry follows a word target")]
    return []


ORDER_READING_AFTER_GRAPHEMES = Rule(id="order/reading-after-graphemes", principle="E1",
                                     severity="error", shape="check",
                                     check=_check_order_reading_after_graphemes)


def _check_order_receptive_first(syllabus: "Syllabus") -> list[Finding]:
    positions = {e.id: i for i, e in enumerate(syllabus.order()) if e.kind == "word_target"}
    by_skill: dict["WordId", dict[str, Target]] = {}
    for t in syllabus.targets:
        by_skill.setdefault(t.word, {})[t.skill] = t
    findings = []
    for word_id, skills in by_skill.items():
        receptive, productive = skills.get("receptive"), skills.get("productive")
        if receptive is not None and productive is not None \
              and positions[receptive.id] > positions[productive.id]:
            findings.append(Finding(rule="order/receptive-before-productive", note_id=word_id,
                                    evidence="productive Target precedes its receptive Target"))
    return findings


ORDER_RECEPTIVE_FIRST = Rule(id="order/receptive-before-productive", principle="F8",
                             severity="error", shape="check",
                             check=_check_order_receptive_first)


def _check_order_sentence_after_words(syllabus: "Syllabus") -> list[Finding]:
    """Every sentence entry must sit after every word_target entry of a
    word it uses: a used word with no Target at all is flagged directly;
    a used word's Target that is not before the sentence's own position
    is flagged too (should not arise from order()'s own construction).
    """
    positions = {(e.kind, e.id): i for i, e in enumerate(syllabus.order())}
    findings = []
    for s in syllabus.sentences:
        note_id = sentence_note_id(s)
        sentence_pos = positions.get(("sentence", note_id))
        if sentence_pos is None:
            continue
        for w in syllabus.words:
            if not syllabus.mentions(s, w.thai):
                continue
            word_targets = [t for t in syllabus.targets if t.word == w.id]
            if not word_targets:
                findings.append(Finding(rule="order/sentence-after-words", note_id=note_id,
                                        evidence=f"uses word {w.id!r} with no Target"))
                continue
            for t in word_targets:
                target_pos = positions.get(("word_target", t.id))
                if target_pos is not None and target_pos >= sentence_pos:
                    findings.append(Finding(
                        rule="order/sentence-after-words", note_id=note_id,
                        evidence=f"target {t.id!r} for word {w.id!r} is not before the sentence"))
    return findings


ORDER_SENTENCE_AFTER_WORDS = Rule(id="order/sentence-after-words", principle="F8",
                                  severity="error", shape="check",
                                  check=_check_order_sentence_after_words)


# --- card/unique-front (A3) --------------------------------------------------
# shape="compile": evaluated by compile.py against compiled notes, not by
# report(); carries no check/measure/judged_subjects function.

CARD_UNIQUE_FRONT = Rule(id="card/unique-front", principle="A3", severity="error",
                         shape="compile")


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
    CATEGORY_SINGLE_MEMBERSHIP,
    COVERAGE_CATEGORIES,
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
    PICTURE_PREFERENCE,
    COVERAGE_SPEAKERS,
    WORD_PRONUNCIATION_CORROBORATED,
    WORD_CLASSIFIER_KNOWN,
    SENTENCE_RECORDING_REQUIRED,
    ORDER_SOUNDS_FIRST,
    ORDER_READING_AFTER_GRAPHEMES,
    ORDER_RECEPTIVE_FIRST,
    ORDER_SENTENCE_AFTER_WORDS,
    CARD_UNIQUE_FRONT,
]

# Every principle actually cited by a registered rule -- "the set of
# principles with enforcement intent" (spec 1 section 4): a principle
# enforced only by compile_syllabus() mechanics (the table's "compile"
# rows) carries no rule and is not counted here.
ENFORCEMENT_PRINCIPLES: frozenset[str] = frozenset(r.principle for r in RULES)
