"""The Syllabus aggregate (spec 1, section 3): the learner's course of
study, and every piece of cross-entity behavior.

Pure: order() is recomputed each call and consults no study history;
report() identifies the state it judged so a stale report steers nothing.
"""
import dataclasses
import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from .entities import Category, Grapheme, MinimalPair, Sentence, SoundConfusion, Target, Word
from .ids import CategoryName, WordId
from .ports import AssessmentReader, MediaIndex, NullAssessmentReader, NullMediaIndex, Tokenizer
from .profile import Profile
from .rulebook import RULES
from .rules import Finding, Gaps, Metric, OrderEntry, Report, Rule


def token_is_known(token: str, known: Collection[str]) -> bool:
    """Whether `token` is a known word, or a known word is its prefix or
    suffix with a remainder that is itself known, recursively. A known
    word occurring mid-token, matching neither end, is not a boundary.
    """
    if token in known:
        return True
    for candidate in known:
        if not candidate or candidate == token:
            continue
        if token.startswith(candidate) and token_is_known(token[len(candidate):], known):
            return True
        if token.endswith(candidate) and token_is_known(token[:len(token) - len(candidate)], known):
            return True
    return False


@dataclass(frozen=True)
class Syllabus:
    words: tuple[Word, ...] = ()
    targets: tuple[Target, ...] = ()
    pairs: tuple[MinimalPair, ...] = ()
    graphemes: tuple[Grapheme, ...] = ()
    sentences: tuple[Sentence, ...] = ()
    confusions: tuple[SoundConfusion, ...] = ()
    profile: Profile = field(default_factory=lambda: Profile(register="male_colloquial"))
    tokenizer: Tokenizer = field(kw_only=True)
    # Storage-owned by spec 2; taken as constructor input for now (spec 1
    # note): rank per word (lower = more frequent).
    frequency: Mapping[WordId, int] = field(default_factory=dict)
    # The Syllabus's curated Category collections; category_of derives the
    # reverse lookup from word id to category name.
    categories: tuple[Category, ...] = ()
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

    @cached_property
    def _category_by_word(self) -> dict[WordId, CategoryName]:
        return {word_id: cat.name for cat in self.categories for word_id in cat.members}

    def category_of(self, word_id: WordId) -> CategoryName | None:
        return self._category_by_word.get(word_id)

    def _emphasis_weight(self, word_id: WordId) -> float:
        category = self.category_of(word_id)
        if category is None:
            return 1.0
        return self.profile.emphasis.get(category, 1.0)

    # --- order() ---------------------------------------------------------

    def order(self) -> list[OrderEntry]:
        sounds = ([OrderEntry("pair", p.id) for p in sorted(self.pairs, key=lambda p: p.id)]
                 + [OrderEntry("grapheme", g.symbol)
                   for g in sorted(self.graphemes, key=lambda g: g.symbol)])

        def key(t: Target) -> tuple[float, str, int]:
            freq = self.frequency.get(t.word, float("inf"))
            weight = self._emphasis_weight(t.word)
            skill_rank = 0 if t.skill == "receptive" else 1
            return (freq / weight if weight else float("inf"), str(t.word), skill_rank)

        ordered_targets = sorted(self.targets, key=key)
        target_entries = [OrderEntry("word_target", t.id) for t in ordered_targets]

        # A word's position for sentence placement: the LAST of its own
        # targets' positions (receptive and productive both included), so a
        # sentence using that word is placed after every target it has.
        word_last_position: dict[WordId, int] = {}
        for i, t in enumerate(ordered_targets):
            position = len(sounds) + i
            word_last_position[t.word] = max(word_last_position.get(t.word, position), position)

        def sentence_after(sentence: Sentence) -> int:
            used = self._words_used(self.tokenizer.tokens(sentence.text))
            return max((word_last_position[w] for w in used if w in word_last_position),
                      default=-1)

        ordered_sentences = sorted(self.sentences, key=lambda s: (sentence_after(s), s.text_sha))
        sentence_entries = [OrderEntry("sentence", s.text_sha) for s in ordered_sentences]

        return [*sounds, *target_entries, *sentence_entries]

    @cached_property
    def _target_positions(self) -> dict[str, int]:
        return {e.id: i for i, e in enumerate(self.order()) if e.kind == "word_target"}

    @cached_property
    def _word_target_positions(self) -> dict[WordId, list[int]]:
        positions: dict[WordId, list[int]] = {}
        for t in self.targets:
            positions.setdefault(t.word, []).append(self._target_positions[t.id])
        return positions

    # --- fills() -----------------------------------------------------------

    @staticmethod
    def _boundary_match(tokens: list[str], thai: str) -> bool:
        return any(tok == thai or tok.startswith(thai) or tok.endswith(thai)
                  for tok in tokens)

    def _words_used(self, tokens: list[str]) -> set[WordId]:
        return {w.id for w in self.words if self._boundary_match(tokens, w.thai)}

    @staticmethod
    def _has_lexical_content(tok: str) -> bool:
        """Whitespace-only or punctuation/digit-only tokens carry no
        vocabulary of their own (a real tokenizer keeps whitespace
        tokens) -- never novel, never budget-consuming."""
        return any(ch.isalpha() for ch in tok)

    def _unknown_tokens(self, tokens: list[str]) -> list[str]:
        """Content tokens that do not decompose into registered Words at
        a boundary (token_is_known). Such a token has no Target by
        construction, so it always counts against the novelty budget
        (spec 1 §3: "every word it uses has an earlier Target") -- no
        exemption list for function/glue words; those must be registered
        with an early receptive Target instead. A known prefix does not
        excuse an unregistered remainder.
        """
        known = {w.thai for w in self.words}
        return [tok for tok in tokens
               if self._has_lexical_content(tok) and not token_is_known(tok, known)]

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
        # with a target anywhere is met by entry; a known word with no
        # target at all, or any content token matching no registered word
        # at all, is new (spec 1 §3; no exemption for unregistered
        # function/glue words -- they must carry an early Target).
        if target.id not in self._target_positions:
            return False
        used_other_words = self._words_used(tokens) - {target.word}
        untargeted = [w for w in used_other_words if w not in self._word_target_positions]
        new_words = untargeted + self._unknown_tokens(tokens)
        budget = 1 if target.introduction == "sentence" else 0
        return len(new_words) <= budget

    def vocabulary_met_by(self, target: Target) -> tuple[Word, ...]:
        """Every Word with a Target at or before `target`'s order()
        position (the target's own word included).
        """
        position = self._target_positions.get(target.id)
        if position is None:
            return ()
        met = [t.word for t in self.targets if self._target_positions[t.id] <= position]
        seen: set[WordId] = set()
        out: list[Word] = []
        for w in met:
            if w not in seen:
                seen.add(w)
                out.append(self.word(w))
        return tuple(out)

    def with_sentences(self, new: Sequence[Sentence]) -> "Syllabus":
        return dataclasses.replace(self, sentences=self.sentences + tuple(new))

    # --- report() ------------------------------------------------------

    def _judged_findings(self, rule: Rule) -> list[Finding]:
        findings = []
        for note_id, artifact_sha in rule.judged_subjects(self):
            # rubric passed through: the merged spec-3 key convention
            # (spec 4 "key-convention debt"; see store.py's module
            # docstring) keys a judged rule's verdict exactly like
            # assessor.JudgeBackend does -- report() reads verdicts under
            # rule.role, which defaults to the rule id.
            verdict = self.assessments.verdict(rule.role, note_id, artifact_sha,
                                               rubric=rule.rubric)
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
        """Folds report()'s completeness findings and measures (spec 1,
        section 3) by rule id. Scene pictures are optional and carry no
        rule finding; that one field reads the media index directly.
        """
        report = self.report()

        def note_ids(rule_id: str) -> tuple[str, ...]:
            return tuple(f.note_id for f in report.findings if f.rule == rule_id)

        coverage = next((m for m in report.metrics if m.rule == "coverage/confusions"), None)
        if coverage is None:
            raise RuntimeError("gaps() needs the coverage/confusions rule registered")
        missing_renditions = tuple(
            confusion_id for confusion_id, detail in coverage.detail.items()
            if not detail["covered"]
        )
        scene_pictures = tuple(
            s.text_sha for s in self.sentences if self.media.picture_sha(s.text_sha) is None
        )
        return Gaps(missing_renditions=missing_renditions,
                    unfilled_targets=note_ids("target/sentence-required"),
                    words_missing_pictures=note_ids("target/picture-required"),
                    words_missing_recordings=note_ids("target/recording-required"),
                    graphemes_missing_keyword_data=note_ids("grapheme/keyword-picture-required"),
                    sentence_recordings=note_ids("sentence/recording-required"),
                    scene_pictures=scene_pictures)

    # --- content-hash staleness marker ------------------------------------

    def state_id(self) -> str:
        def canon(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: canon(getattr(obj, f.name))
                       for f in dataclasses.fields(obj)}
            if isinstance(obj, Mapping):
                return {str(k): canon(v) for k, v in sorted(obj.items(),
                                                             key=lambda kv: str(kv[0]))}
            if isinstance(obj, frozenset):
                return sorted(str(v) for v in obj)
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
            "categories": sorted((canon(c) for c in self.categories), key=lambda d: d["name"]),
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
