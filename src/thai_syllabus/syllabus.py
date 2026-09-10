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
from typing import Any

from .cachekeys import JudgeKey
from .entities import (
    Category, Grapheme, MinimalPair, Sentence, SoundConfusion, Target, Word, render,
)
from .ids import CategoryName, ConfusionId, PairId, TargetId, WordId
from .ports import (
    AssessmentReader, MediaIndex, NullAssessmentReader, NullMediaIndex, StudyReader,
    StudyRecord,
)
from .profile import Profile
from .rulebook import RULES
from .rules import Finding, Gaps, Metric, OrderEntry, Report, Rule


@dataclass(frozen=True)
class Syllabus:
    words: tuple[Word, ...] = ()
    targets: tuple[Target, ...] = ()
    pairs: tuple[MinimalPair, ...] = ()
    graphemes: tuple[Grapheme, ...] = ()
    sentences: tuple[Sentence, ...] = ()
    confusions: tuple[SoundConfusion, ...] = ()
    profile: Profile = field(default_factory=lambda: Profile(register="male_colloquial"))
    # Rank per word, lower = more frequent; loaded from spec 2's storage
    # and handed in here.
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

    @cached_property
    def _sentence_index(self) -> dict[str, Sentence]:
        return {s.text_sha: s for s in self.sentences}

    def sentence(self, text_sha: str) -> Sentence:
        """The adopted Sentence with that text_sha; KeyError names it."""
        found = self._sentence_index.get(text_sha)
        if found is None:
            raise KeyError(f"no sentence with text_sha {text_sha!r} in the syllabus")
        return found

    @cached_property
    def _pair_index(self) -> dict[PairId, MinimalPair]:
        return {p.id: p for p in self.pairs}

    def pair(self, pair_id: PairId) -> MinimalPair:
        """The MinimalPair with that id; KeyError names it."""
        found = self._pair_index.get(pair_id)
        if found is None:
            raise KeyError(f"no minimal pair {pair_id!r} in the syllabus")
        return found

    # --- the voice a recording may draw (E2, E7) ------------------------

    def serves_productive(self, word_id: WordId) -> bool:
        """Whether anything recorded for this word plays on a productive
        back: it has a productive Target.
        """
        return any(t.word == word_id and t.skill == "productive" for t in self.targets)

    def sentence_serves_productive(self, sentence: Sentence) -> bool:
        """Whether this sentence fills a productive Target, so its own
        recording plays on a productive back.
        """
        return any(t.skill == "productive" for t in self.fill_set(sentence))

    def pair_voice_constraint(self, pair_id: PairId) -> str:
        """The strictest of the members' voice constraints, a rendition
        speaking for every member at once: "male" if any member serves a
        productive Target, else "any".
        """
        return ("male" if any(self.serves_productive(m) for m in self.pair(pair_id).members)
                else "any")

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

        target_entries = [OrderEntry("word_target", t.id) for t in self._ordered_targets]

        def sentence_after(sentence: Sentence) -> int:
            try:
                word = self.last_used_word(sentence)
            except ValueError:
                return -1
            return self._word_last_position[word]

        ordered_sentences = sorted(self.sentences, key=lambda s: (sentence_after(s), s.text_sha))
        sentence_entries = [OrderEntry("sentence", s.text_sha) for s in ordered_sentences]

        return [*sounds, *target_entries, *sentence_entries]

    @cached_property
    def _ordered_targets(self) -> tuple[Target, ...]:
        """Targets sorted by (frequency/emphasis, word id, skill), the
        basis of order()'s word_target block and of _word_last_position;
        order() builds that block from this same tuple, with no
        recursion through order() itself.
        """
        def key(t: Target) -> tuple[float, str, int]:
            freq = self.frequency.get(t.word, float("inf"))
            weight = self._emphasis_weight(t.word)
            skill_rank = 0 if t.skill == "receptive" else 1
            return (freq / weight if weight else float("inf"), str(t.word), skill_rank)

        return tuple(sorted(self.targets, key=key))

    @cached_property
    def _word_last_position(self) -> dict[WordId, int]:
        """Each word's greatest index among its own targets in
        _ordered_targets (receptive and productive both included) --
        the relative position a sentence using that word is placed
        after in order(); shared by last_used_word.
        """
        positions: dict[WordId, int] = {}
        for i, t in enumerate(self._ordered_targets):
            positions[t.word] = max(positions.get(t.word, i), i)
        return positions

    def check_sentence(self, sentence: Sentence) -> None:
        """The Sentence invariant a syllabus's own vocabulary decides
        (spec 1 section 1): every element's word registered here, and
        render(sentence.clauses, ...) equal to sentence.text. Raises
        ValueError naming the sentence's text_sha and the offending id,
        or the two texts, on the first violation found.
        """
        known = {w.id for w in self.words}
        for word_id in sentence.words:
            if word_id not in known:
                raise ValueError(
                    f"sentence {sentence.text_sha!r} names unregistered word {word_id!r}")
        rendered = render(sentence.clauses, lambda w: self.word(w).thai)
        if rendered != sentence.text:
            raise ValueError(
                f"sentence {sentence.text_sha!r} text {sentence.text!r} does not match "
                f"its clauses' rendering {rendered!r}")

    def last_used_word(self, sentence: Sentence) -> WordId:
        """The word `sentence` uses whose own target position
        (_word_last_position) is greatest -- two words can never tie
        (each target belongs to one word; every word's own last
        position is a distinct index), but the (position, word) key
        still orders any hypothetical tie to the greater word id.
        order()'s sentence_after shares this computation to place the
        sentence. Raises ValueError naming the sentence's text_sha when
        it uses no targeted word.
        """
        used = frozenset(sentence.words)
        candidates = [w for w in used if w in self._word_last_position]
        if not candidates:
            raise ValueError(f"sentence {sentence.text_sha!r} uses no targeted word")
        return max(candidates, key=lambda w: (self._word_last_position[w], w))

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

    def _target_satisfies_clauses_1_and_2(self, words: frozenset[WordId], voice: str,
                                          target: Target) -> bool:
        """Clauses 1 and 2 alone: the target's word among `words` (a
        sentence's own words as a set), the voice satisfying the skill --
        the "contains" test a fill set is built from, distinct from
        membership in one (spec 1 section 3).
        """
        if target.word not in words:
            return False
        return not (target.skill == "productive" and voice != "learner_voice")

    def _sentence_order_key(self, sentence: Sentence) -> tuple[int, str] | None:
        """(last_used_word's order() position, text_sha): the key
        order()'s own sentence_after sorts sentences by, and clause 3's
        novelty rule compares to place one adopted sentence at or before
        another. None when the sentence uses no targeted word at all.
        """
        try:
            word = self.last_used_word(sentence)
        except ValueError:
            return None
        return (self._word_last_position[word], sentence.text_sha)

    @cached_property
    def _adopted_order_keys(self) -> dict[str, tuple[int, str] | None]:
        """Every adopted sentence's own `_sentence_order_key`, computed
        once per instance and keyed by text_sha -- `_order_key_of` reads
        this for an adopted sentence instead of recomputing
        `last_used_word` (an O(words) scan) on every comparison in the
        novelty check's inner loop over self.sentences.
        """
        return {s.text_sha: self._sentence_order_key(s) for s in self.sentences}

    def _order_key_of(self, sentence: Sentence) -> tuple[int, str] | None:
        """`_sentence_order_key`, from `_adopted_order_keys` for an
        adopted sentence, computed fresh for any other sentence.
        """
        if sentence.text_sha in self._adopted_order_keys:
            return self._adopted_order_keys[sentence.text_sha]
        return self._sentence_order_key(sentence)

    @cached_property
    def _adopted_placement_order(self) -> tuple[Sentence, ...]:
        """self.sentences sorted by placement order (spec 1 section 3,
        clause 3): `_order_key_of`, or (-1, text_sha) for a sentence
        with no order key at all (order()'s own fallback) -- a strict
        total order over a finite set, the basis the fill-set recursion
        below is well-founded on.
        """
        def key(s: Sentence) -> tuple[int, str]:
            return self._order_key_of(s) or (-1, s.text_sha)
        return tuple(sorted(self.sentences, key=key))

    @cached_property
    def _adopted_fill_sets(self) -> dict[str, tuple[Target, ...]]:
        """Every adopted sentence's own fill set (spec 1 section 3,
        clause 3), computed once per instance in placement order
        (`_adopted_placement_order`): each sentence's novelty rule reads
        only the fill sets already computed here for sentences placed
        strictly before it in that same order: well-founded, placement
        order being a strict total order over a finite set. `fill_set`
        looks an adopted sentence up here directly, memoized
        for the life of this instance (`with_sentences` returns a new
        one -- this cannot go stale); a candidate not itself adopted is
        computed fresh against this completed map.
        """
        computed: dict[str, tuple[Target, ...]] = {}
        for s in self._adopted_placement_order:
            computed[s.text_sha] = self._compute_fill_set(s, computed)
        return computed

    def _compute_fill_set(self, sentence: Sentence,
                          adopted_fill_sets: Mapping[str, tuple[Target, ...]]
                          ) -> tuple[Target, ...]:
        """fill_set's body (spec 1 section 3, clause 3): a sentence-level
        gate first -- every word the sentence names carries a Target --
        then the candidates passing clauses 1 and 2, in target-id order.
        Among the candidates, a
        sentence-introduced Target is unmet unless some other adopted
        sentence, placed at or before this one, already has it in ITS
        OWN fill set (read from `adopted_fill_sets`, not clauses 1 and 2
        alone -- a sentence whose own fill set is empty, an untargeted
        word, meets nothing for anyone); more than one unmet candidate
        empties the fill set; zero or one leaves every candidate as the
        fill set. `adopted_fill_sets` carries every sentence placed
        strictly before `sentence` in placement order, when called from
        `_adopted_fill_sets`'s own recursive build, or the complete map,
        for a candidate that is not itself adopted.
        """
        used = frozenset(sentence.words)
        if not used <= self._word_target_positions.keys():
            return ()

        candidates = tuple(sorted(
            (t for t in self.targets
            if self._target_satisfies_clauses_1_and_2(used, sentence.voice, t)),
            key=lambda t: t.id))
        if not candidates:
            return ()

        this_key = self._order_key_of(sentence)

        def met_by_another_adopted_sentence(target: Target) -> bool:
            for other in self.sentences:
                if other.text_sha == sentence.text_sha:
                    continue
                other_key = self._order_key_of(other)
                if other_key is None or other_key > this_key:
                    continue
                if target in adopted_fill_sets.get(other.text_sha, ()):
                    return True
            return False

        unmet = [t for t in candidates
                if t.introduction == "sentence" and not met_by_another_adopted_sentence(t)]
        if len(unmet) > 1:
            return ()
        return candidates

    def fill_set(self, sentence: Sentence) -> tuple[Target, ...]:
        """Every Target `sentence` fills (spec 1 section 3, clause 3):
        `_compute_fill_set`'s result, memoized per instance for an
        adopted sentence (`_adopted_fill_sets`), computed fresh for a
        candidate that is not itself adopted.
        """
        adopted = self._adopted_fill_sets
        cached = adopted.get(sentence.text_sha)
        if cached is not None:
            return cached
        return self._compute_fill_set(sentence, adopted)

    def fills(self, sentence: Sentence, target: Target) -> bool:
        return target in self.fill_set(sentence)

    def met_sentence_introduced_targets(self) -> frozenset[TargetId]:
        """Every sentence-introduced Target some adopted sentence's own
        fill set (`_adopted_fill_sets`) contains: the Targets those
        sentences have already put in front of the learner, for the
        sentence prompt's vocabulary and Introducible sections.
        """
        return frozenset(t.id for fills in self._adopted_fill_sets.values()
                         for t in fills if t.introduction == "sentence")

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
        for s in new:
            self.check_sentence(s)
        return dataclasses.replace(self, sentences=self.sentences + tuple(new))

    def cover(self, drafts: Sequence[tuple[Sentence, Sequence[Target]]]
              ) -> list[tuple[Sentence, tuple[Target, ...]]]:
        """The drafts worth adopting, greedily: the one filling the most
        still-unfilled Targets, then the next, until none fills one, each
        with the Targets it is adopted for. Ties go to the shorter text,
        then the lower text_sha.
        """
        uncovered = set(self.gaps().unfilled_targets)
        remaining = sorted(drafts, key=lambda d: (len(d[0].text), d[0].text_sha))
        chosen: list[tuple[Sentence, tuple[Target, ...]]] = []
        while remaining:
            best = max(remaining, key=lambda d: len({t.id for t in d[1]} & uncovered))
            gained = tuple(t for t in best[1] if t.id in uncovered)
            if not gained:
                break
            chosen.append((best[0], gained))
            uncovered -= {t.id for t in gained}
            remaining.remove(best)
        return chosen

    # --- report() ------------------------------------------------------

    def _judged_findings(self, rule: Rule) -> list[Finding]:
        findings = []
        for note_id, artifact_sha in rule.judged_subjects(self):
            # for_rule() builds the key assessor.JudgeBackend.cache_key
            # builds for the same rubric/artifact/role: one cache row.
            key = JudgeKey.for_rule(rule.rubric, artifact_sha, note_id, rule.role)
            answer = self.assessments.verdict("judge", key)
            if answer is not None and answer.answer.get("value") is False:
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
        """report()'s completeness findings and measures (spec 1 section
        3), folded by rule id. Scene pictures carry no rule finding, so
        that one field reads the media index directly.
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

    # --- study_by_confusion -------------------------------------------------

    def study_by_confusion(self, study: StudyReader) -> dict[ConfusionId, list[StudyRecord]]:
        """Every minimal_pair-family StudyRecord grouped by the confusion
        of the pair its anchor names (spec 2 section 2). An anchor
        matching no pair is skipped.
        """
        confusion_by_pair = {p.id: p.confusion for p in self.pairs}
        grouped: dict[ConfusionId, list[StudyRecord]] = {}
        for record in study.study_rows():
            if record.family != "minimal_pair":
                continue
            confusion = confusion_by_pair.get(record.anchor)
            if confusion is None:
                continue
            grouped.setdefault(confusion, []).append(record)
        return grouped

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
        """sha(rulebook.yaml text + registry rule ids), spec 3 section 6:
        a rulebook edit changes this where state_id() stays put.
        """
        payload = {"rulebook_text": self.rulebook_text,
                  "rule_ids": sorted(r.id for r in self.rules)}
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
