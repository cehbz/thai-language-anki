"""The Syllabus aggregate (spec 1, section 3): the learner's course of
study, and every piece of cross-entity behavior.

Pure: order() is recomputed each call and consults no study history;
report() identifies the state it judged so a stale report steers nothing.
"""
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Union

from .entities import Grapheme, MinimalPair, Sentence, SoundConfusion, Target, Word
from .ids import Category, WordId
from .ports import AssessmentReader, MediaIndex, NullAssessmentReader, NullMediaIndex, Tokenizer
from .profile import Profile
from .rulebook import RULES
from .rules import Compile, Finding, Gaps, Metric, Report, Rule

TargetLike = Union[Target, str]  # str covers PairId and GraphemeId (both NewType(str))


class _WhitespaceTokenizer:
    """A tokenizer of last resort: real Thai has no spaces, so this only
    exists so Syllabus() is constructible without a tokenizer for tests
    that never call fills(). Real callers pass a real port.
    """
    def tokens(self, text: str) -> list[str]:
        return text.split()


@dataclass(frozen=True)
class Syllabus:
    words: tuple[Word, ...] = ()
    targets: tuple[Target, ...] = ()
    pairs: tuple[MinimalPair, ...] = ()
    graphemes: tuple[Grapheme, ...] = ()
    sentences: tuple[Sentence, ...] = ()
    confusions: tuple[SoundConfusion, ...] = ()
    profile: Profile = field(default_factory=lambda: Profile(register="male_colloquial"))
    tokenizer: Tokenizer = field(default_factory=_WhitespaceTokenizer)
    # Storage-owned by spec 2; taken as constructor input for now (spec 1
    # note): rank per word (lower = more frequent) and each word's emphasis
    # category.
    frequency: Mapping[WordId, int] = field(default_factory=dict)
    categories: Mapping[WordId, Category] = field(default_factory=dict)
    media: MediaIndex = field(default_factory=NullMediaIndex)
    assessments: AssessmentReader = field(default_factory=NullAssessmentReader)
    rules: Sequence[Rule] = field(default_factory=lambda: tuple(RULES))
    # spec 3 section 6: rulebook_id = sha(rulebook.yaml text + registry rule
    # ids). Raw text (curated.rulebook_file_text's output), not the parsed
    # RulebookConfig -- a config change with no severity/threshold/rubric
    # change still edits the file, and that edit must still show up here.
    rulebook_text: str = ""

    # --- lookups -------------------------------------------------------

    @cached_property
    def _word_index(self) -> dict[WordId, Word]:
        return {w.id: w for w in self.words}

    def word(self, word_id: WordId) -> Word:
        return self._word_index[word_id]

    def find_word(self, word_id: WordId) -> Word | None:
        return self._word_index.get(word_id)

    def _emphasis_weight(self, word_id: WordId) -> float:
        category = self.categories.get(word_id)
        if category is None:
            return 1.0
        return self.profile.emphasis.get(category, 1.0)

    # --- order() ---------------------------------------------------------

    def order(self) -> list[TargetLike]:
        sounds: list[TargetLike] = (
            [p.id for p in sorted(self.pairs, key=lambda p: p.id)]
            + [g.symbol for g in sorted(self.graphemes, key=lambda g: g.symbol)]
        )

        def key(t: Target) -> tuple[float, str, int]:
            freq = self.frequency.get(t.word, float("inf"))
            weight = self._emphasis_weight(t.word)
            skill_rank = 0 if t.skill == "receptive" else 1
            return (freq / weight if weight else float("inf"), str(t.word), skill_rank)

        words: list[TargetLike] = sorted(self.targets, key=key)
        return [*sounds, *words]

    @cached_property
    def _target_positions(self) -> dict[str, int]:
        return {t.id: i for i, t in enumerate(self.order()) if isinstance(t, Target)}

    @cached_property
    def _word_target_positions(self) -> dict[WordId, list[int]]:
        positions: dict[WordId, list[int]] = {}
        for t in self.order():
            if isinstance(t, Target):
                positions.setdefault(t.word, []).append(self._target_positions[t.id])
        return positions

    # --- fills() -----------------------------------------------------------

    @staticmethod
    def _boundary_match(tokens: list[str], thai: str) -> bool:
        return any(tok == thai or tok.startswith(thai) or tok.endswith(thai)
                  for tok in tokens)

    def _words_used(self, tokens: list[str]) -> set[WordId]:
        return {w.id for w in self.words if self._boundary_match(tokens, w.thai)}

    def mentions(self, sentence: Sentence, thai: str) -> bool:
        """Whether `thai` appears in `sentence.text` at a token boundary
        (rulebook helper: exposes the same boundary rule fills() uses).
        """
        return self._boundary_match(self.tokenizer.tokens(sentence.text), thai)

    def fills(self, sentence: Sentence, target: Target) -> bool:
        tokens = self.tokenizer.tokens(sentence.text)
        target_word = self.word(target.word)

        # clause 1: word at a token boundary
        if not self._boundary_match(tokens, target_word.thai):
            return False

        # clause 2: voice satisfies skill (other_voice fills receptive only)
        if target.skill == "productive" and sentence.voice != "learner_voice":
            return False

        # clause 3: strict i+1 with a novelty budget. The sentence enters
        # the order after its LAST used word's target, so any used word
        # with a target anywhere is met by entry; only words with no
        # target at all are new (spec 1 §3).
        if target.id not in self._target_positions:
            return False
        used_other_words = self._words_used(tokens) - {target.word}
        new_words = [w for w in used_other_words
                    if w not in self._word_target_positions]
        budget = 1 if target.introduction == "sentence" else 0
        return len(new_words) <= budget

    # --- report() ------------------------------------------------------

    def _judged_findings(self, rule: Rule) -> list[Finding]:
        findings = []
        for note_id, artifact_sha in rule.judged_subjects(self):
            verdict = self.assessments.verdict(rule.id, note_id, artifact_sha)
            if verdict is False:
                findings.append(Finding(rule=rule.id, note_id=note_id,
                                        artifact_sha=artifact_sha,
                                        evidence="judged: fail"))
        return findings

    def _severity(self, rule_id: str) -> str | None:
        for r in self.rules:
            if r.id == rule_id:
                return r.severity
        return None

    def report(self) -> Report:
        findings: list[Finding] = []
        metrics: list[Metric] = []
        for rule in self.rules:
            if rule.shape == "check":
                findings.extend(rule.check(self))
            elif rule.shape == "measure":
                metrics.append(rule.measure(self))
            elif rule.shape == "judged":
                findings.extend(self._judged_findings(rule))

        def blocks_gate(f: Finding) -> bool:
            return (self._severity(f.rule) == "error"
                   and not self.assessments.is_waived(f))

        gate = not any(blocks_gate(f) for f in findings)
        return Report(syllabus_state_id=self.state_id(), rulebook_id=self.rulebook_id(),
                     findings=tuple(findings), metrics=tuple(metrics), gate=gate)

    # --- gaps() ----------------------------------------------------------

    def gaps(self) -> Gaps:
        unfilled_targets = tuple(
            t.id for t in self.targets
            if not any(self.fills(s, t) for s in self.sentences)
        )
        words_with_targets = {t.word for t in self.targets}
        words_missing_pictures = tuple(
            w.id for w in self.words
            if w.id in words_with_targets and not self.media.has_picture(w.id)
        )
        words_missing_recordings = tuple(
            w.id for w in self.words
            if w.id in words_with_targets and not self.media.recording_speakers(w.id)
        )
        graphemes_missing_keyword_data = tuple(
            g.symbol for g in self.graphemes if g.keyword not in self._word_index
        )
        missing_renditions = tuple(
            c.id for c in self.confusions
            if not self.media.rendition_speakers(c.id)
        )
        return Gaps(missing_renditions=missing_renditions,
                    unfilled_targets=unfilled_targets,
                    words_missing_pictures=words_missing_pictures,
                    words_missing_recordings=words_missing_recordings,
                    graphemes_missing_keyword_data=graphemes_missing_keyword_data)

    # --- compile() ---------------------------------------------------------

    def compile(self) -> Compile:
        raise NotImplementedError("compile() is spec 4's territory")

    # --- content-hash staleness marker ------------------------------------

    def state_id(self) -> str:
        def canon(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: canon(getattr(obj, f.name))
                       for f in dataclasses.fields(obj)}
            if isinstance(obj, Mapping):
                return {str(k): canon(v) for k, v in sorted(obj.items(),
                                                             key=lambda kv: str(kv[0]))}
            if isinstance(obj, (list, tuple)):
                return [canon(v) for v in obj]
            return obj

        payload = {
            "words": sorted((canon(w) for w in self.words), key=lambda d: d["id"]),
            "targets": sorted((canon(t) for t in self.targets), key=lambda d: d["id"]),
            "pairs": sorted((canon(p) for p in self.pairs), key=lambda d: d["id"]),
            "graphemes": sorted((canon(g) for g in self.graphemes), key=lambda d: d["symbol"]),
            "sentences": sorted((canon(s) for s in self.sentences),
                                key=lambda d: json.dumps(d, sort_keys=True, default=str)),
            "confusions": sorted((canon(c) for c in self.confusions), key=lambda d: d["id"]),
            "profile": canon(self.profile),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def rulebook_id(self) -> str:
        """sha(rulebook.yaml text + registry rule ids) -- spec 3 section 6.
        Differs from state_id(): a rulebook edit (severity/threshold/rubric
        change, or the registry itself gaining/losing a rule) changes this
        without necessarily changing the aggregate's own content.
        """
        payload = {"rulebook_text": self.rulebook_text,
                  "rule_ids": sorted(r.id for r in self.rules)}
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
