"""Ports the Syllabus reads through (spec 1 defines them; spec 3 owns the
real backends). All three are read-only from the aggregate's point of view:
report() never calls a judge, fills() never calls a live tokenizer service.

Spec 2 (durable state) section 3 adds three more interfaces -- FrequencyMap,
RecordWriter, StudyReader -- not present when spec 1 was implemented.
AssessmentReader/Tokenizer are spec 1's contract, unchanged since; MediaIndex
gained recording_provenance/rendition_provenance/picture_sha here (spec 4)
for rulebook.py's completeness, synthetic/mixed-speaker, and picture/fit
rules, and speakers_of (spec 1 section 1, E7) for coverage/speakers.
store.py's SyllabusDb satisfies AssessmentReader directly (isinstance
checks against that Protocol pass) and additionally offers `assessments_of`
(spec 2's fuller read surface over the same cache table -- not part of the
Protocol spec 1 already shipped, so it is not declared here, only
implemented); MediaIndex is satisfied by wiring.py's `_DbMediaIndex`, a
separate adapter over SyllabusDb and the loaded pairs, not by SyllabusDb
itself.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .ids import ConfusionId, PairId, WordId
    from .media import Speaker
    from .rules import Finding


@runtime_checkable
class Tokenizer(Protocol):
    """Splits Thai text into tokens. Boundary membership (used by
    Syllabus.fills) is token == word.thai or token.startswith/endswith
    (word.thai) -- so a compound token naturally counts for each of the
    known words it starts or ends with.
    """
    def tokens(self, text: str) -> list[str]: ...


@runtime_checkable
class AssessmentReader(Protocol):
    """Cached verdicts only -- report() never calls a judge. Also the one
    channel waivers arrive through: a waiver is an assessment of a finding's
    identity (rule, note_id, artifact_sha).
    """
    def verdict(self, rule_id: str, note_id: str,
                artifact_sha: str | None = None,
                rubric: str | None = None) -> bool | None:
        """True/False for a cached judged-rule verdict; None if the
        (rule, note, artifact) has not been assessed yet. `rubric` is the
        judged Rule's rubric text (spec 4's merged key convention --
        store.py's module docstring: this reads the SAME
        judge:sha(RUBRIC):IDENTITY:ROLE row shape assessor.py's
        JudgeBackend writes, with ROLE=rule_id).
        """
        ...

    def is_waived(self, finding: "Finding") -> bool:
        ...


@runtime_checkable
class MediaIndex(Protocol):
    """Read access to spec 2's media relationships. Not named by spec 1's
    text; added here because gap/coverage measures (media/picture-required,
    coverage/confusions) need to know what media exists, and architecture.md
    lists that as record-owned, spec-2 territory the Syllabus reads through.
    has_picture/recording_speakers/rendition_speakers are spec 1's original
    three; recording_provenance/rendition_provenance/picture_sha (spec 4)
    add provenance-row and artifact-sha access for the rulebook's
    completeness, synthetic/mixed-speaker, and picture/fit rules;
    speakers_of (E7) backs coverage/speakers.
    """
    def has_picture(self, word: "WordId") -> bool: ...
    def recording_speakers(self, word: "WordId") -> frozenset[str]: ...
    def rendition_speakers(self, pair_confusion: "ConfusionId") -> frozenset[str]: ...

    def recording_provenance(self, word: "WordId") -> Mapping[str, Any] | None:
        """The current-best recording's `media` row: source, speaker_id,
        and `speaker` (the resolved Speaker, or None when speaker_id is
        absent); None if there is no current-best recording.
        """
        ...

    def rendition_provenance(self, pair_id: "PairId") -> tuple[Mapping[str, Any], ...]:
        """One provenance row per pair member's current-best recording; a
        member with none is skipped.
        """
        ...

    def picture_sha(self, word: "WordId") -> str | None:
        """The current-best picture's artifact sha, or None."""
        ...

    def speakers_of(self, corpus: Literal["recording", "rendition", "sentence"]) -> tuple["Speaker", ...]:
        """Distinct speakers behind that audio corpus's current-best
        artifacts. corpus is "recording" (word recordings), "rendition"
        (pair renditions), or "sentence" (sentence recordings).
        """
        ...


class NullAssessmentReader:
    """No cached verdicts, no waivers -- the default when a caller has no
    AssessmentReader to plug in yet.
    """
    def verdict(self, rule_id: str, note_id: str,
                artifact_sha: str | None = None,
                rubric: str | None = None) -> bool | None:
        return None

    def is_waived(self, finding: "Finding") -> bool:
        return False


class NullMediaIndex:
    """No media known -- the conservative default (everything reads as a gap)."""
    def has_picture(self, word: "WordId") -> bool:
        return False

    def recording_speakers(self, word: "WordId") -> frozenset[str]:
        return frozenset()

    def rendition_speakers(self, pair_confusion: "ConfusionId") -> frozenset[str]:
        return frozenset()

    def recording_provenance(self, word: "WordId") -> Mapping[str, Any] | None:
        return None

    def rendition_provenance(self, pair_id: "PairId") -> tuple[Mapping[str, Any], ...]:
        return ()

    def picture_sha(self, word: "WordId") -> str | None:
        return None

    def speakers_of(self, corpus: Literal["recording", "rendition", "sentence"]) -> tuple["Speaker", ...]:
        return ()


# --- spec 2 section 3 additions -------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One `cache` table row, read back and decoded. Not named by spec 1;
    spec 2 section 3 names it as AssessmentReader.assessments_of's element
    type. `question`/`answer` are already-decoded JSON (whatever shape the
    writing backend used); `ts` is nanoseconds since the epoch (store.py's
    sortable, collision-resistant substitute for the cache table's `ts`
    column -- see store.py's docstring for why).

    `key` is spec 3's readable canonical cache key (e.g. "forvo:WORD",
    "judge:sha(RUBRIC):sha(ARTIFACT):ROLE" with the real shas substituted
    in) -- what the backend actually asked. `key_sha` is its indexed
    digest (spec 2's `cache.key_sha` column); the two always correspond
    (key_sha = sha256(key)), `key` is kept alongside it for readability
    and for derivations that need to parse/prefix-match the raw key.
    """
    port: str
    backend: str
    key_sha: str
    key: str
    subject: str
    question: Any
    answer: Any
    cost: float
    ts: int


@dataclass(frozen=True)
class StudyRecord:
    """One `study` table row (spec 2 section 2): an imported Anki review.
    card_key = the compiled card's content identity (target/pair/grapheme
    id + card kind, spec 2's own words); compile_id identifies which
    Compile produced that card.
    """
    card_key: str
    compile_id: str
    ts: int
    grade: int
    time_ms: int


@runtime_checkable
class FrequencyMap(Protocol):
    """Word-frequency corpus lookup (spec 2 section 3). Not one of the five
    durable stores (spec 2 section 2 lists exactly five sqlite tables and no
    frequency table) -- this reads a static, unchanging project resource
    (data/frequency_th.txt), not deck state the Syllabus writes.
    """
    def rank(self, word_thai: str) -> int | None: ...


@runtime_checkable
class RecordWriter(Protocol):
    """Append-only write side of the `cache` table (spec 2 section 2,
    spec 3 section 2). `key` is the backend's own canonical, readable
    cache-key string (spec 3's per-backend key functions, e.g.
    "forvo:WORD"); the store keeps it verbatim in the `key` column AND
    hashes it into `key_sha`, the column the table indexes on. Every
    append is one transaction (the checkpoint rule); never an update,
    never a delete. Returns the row's `ts` (nanoseconds since the epoch)
    so callers can stamp the Answer/Verdict they hand back to their
    caller with the same timestamp the row was actually written under.
    """
    def append(self, port: str, backend: str, key: str, subject: str,
               question: Any, answer: Any, cost: float = 0.0) -> int: ...


@runtime_checkable
class CacheReader(Protocol):
    """Read side of the `cache` table that spec 3's ports consume (Provider/
    Assessor's cache-first ask(), and the derivations' folds over one
    subject's history). Not spec 1's AssessmentReader (that one is scoped
    to judged-rule verdicts and waivers) -- this is the general port+backend
    cache surface spec 2 section 3 alludes to but leaves to spec 3 to name.
    """
    def latest(self, port: str, backend: str, key: str) -> "Answer | None":
        """The newest row exactly matching (port, backend, key) -- the
        cache-first hit lookup every backend's ask() consults before
        executing. None on a cache miss (nothing asked yet).
        """
        ...

    def assessments_of(self, subject: str) -> list["Answer"]:
        """Every cache row (any port/backend) for one subject, oldest
        first -- the attempt record derivations.py folds over.
        """
        ...


@runtime_checkable
class StudyReader(Protocol):
    """Read side of the `study` table (spec 2 section 3). `records` takes
    either a card_key (exact match against the table) or a ConfusionId
    (aggregated over every pair card_key carrying that confusion -- see
    store.py's SyllabusDb.records for how that aggregation is resolved,
    since the study table itself only stores card_key, not confusion).
    """
    def records(self, card_key_or_confusion: str) -> list["StudyRecord"]: ...
