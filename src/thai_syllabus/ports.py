"""The ports the Syllabus reads through: AssessmentReader and MediaIndex
(spec 1), plus FrequencyMap, RecordWriter, CacheReader and StudyReader
(spec 2 section 3). All are read-only from the aggregate's point of
view: report() never calls a judge, fills() reads a sentence's own
clauses.

store.py's SyllabusDb satisfies AssessmentReader, RecordWriter,
CacheReader and StudyReader; MediaIndex is satisfied by wiring.py's
`_DbMediaIndex`, an adapter over SyllabusDb and the loaded pairs.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .cachekeys import CacheKey
    from .ids import ConfusionId, PairId, WordId
    from .media import Recording, Speaker
    from .rules import Finding


@runtime_checkable
class AssessmentReader(Protocol):
    """Cached verdicts only -- report() never calls a judge. Also the one
    channel waivers arrive through: a waiver is an assessment of a finding's
    identity (rule, note_id, artifact_sha).
    """
    def verdict(self, backend: str, key: Any) -> "Answer | None":
        """The newest cache row for (backend, key), or None if it has not
        been assessed yet. `key` is built by the caller through spec 3's
        cachekeys.py (a judged Rule's verdict: cachekeys.JudgeKey with
        role=rule.role); the reader reads what it is handed.
        """
        ...

    def is_waived(self, finding: "Finding") -> bool:
        ...


@runtime_checkable
class MediaIndex(Protocol):
    """Read access to spec 2's media relationships: what media a subject
    has (has_picture/recording_speakers/rendition_speakers), the
    provenance rows and artifact shas the rulebook's completeness,
    synthetic/mixed-speaker and picture/fit rules read
    (recording_provenance/rendition_provenance/picture_sha), and the
    speakers behind one audio corpus (speakers_of, for coverage/speakers).
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

    def rendition(self, pair_id: "PairId") -> "tuple[Recording, ...] | None":
        """The pair's current-best rendition (spec 3 section 5): one
        Recording per member, in member order; None when the pair has no
        current-best rendition.
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
    def verdict(self, backend: str, key: Any) -> "Answer | None":
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

    def rendition(self, pair_id: "PairId") -> "tuple[Recording, ...] | None":
        return None

    def picture_sha(self, word: "WordId") -> str | None:
        return None

    def speakers_of(self, corpus: Literal["recording", "rendition", "sentence"]) -> tuple["Speaker", ...]:
        return ()


# --- spec 2 section 3 additions -------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One `cache` table row, read back and decoded (spec 2 section 3).
    `question`/`answer` are already-decoded JSON in whatever shape the
    writing backend used; `ts` is nanoseconds since the epoch.

    `key` is the readable string a cachekeys.py CacheKey's encode()
    produced, kept for inspection only -- no module reads it back to
    rebuild a key. `key_sha` is its indexed digest (sha256 of `key`), the
    column every lookup matches on.
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
    family is word|minimal_pair|grapheme|sentence; anchor is the entity id
    for the family (word id, grapheme symbol, sentence text_sha), or a
    pair id for family "minimal_pair" (member_index/speaker_id then name
    the reviewed member); card_kind is the card's template name, lowered;
    compile_id identifies which Compile produced that card.
    """
    family: str
    anchor: str
    card_kind: str
    compile_id: str
    ts: int
    grade: int
    time_ms: int
    member_index: str | None = None
    speaker_id: str | None = None


@runtime_checkable
class FrequencyMap(Protocol):
    """Word-frequency corpus lookup (spec 2 sections 1 and 3) over
    curated/frequency_th.txt, copied into the deck from the repo's data/
    at migration time; read-only, never hand-edited.
    """
    def rank(self, word_thai: str) -> int | None: ...


@runtime_checkable
class RecordWriter(Protocol):
    """Append-only write side of the `cache` table (spec 2 section 2,
    spec 3 section 2). `key` is a cachekeys.py CacheKey; the store writes
    `key.encode()` to the `key` column and its sha256 to the indexed
    `key_sha`. One transaction per append; never an update, never a
    delete. Returns the row's `ts` (nanoseconds since the epoch), the
    timestamp callers stamp their own Answer/Verdict with.
    """
    def append(self, port: str, backend: str, key: "CacheKey", subject: str,
               question: Any, answer: Any, cost: float = 0.0) -> int: ...


@runtime_checkable
class CacheReader(Protocol):
    """The general read side of the `cache` table (spec 3): Provider/
    Assessor's cache-first ask(), and the derivations' folds over one
    subject's history.
    """
    def latest(self, port: str, backend: str, key: "CacheKey") -> "Answer | None":
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

    def rows_since(self, port: str, backend: str, since_ts: int) -> list["Answer"]:
        """Every row of (port, backend) at or after `since_ts`, oldest
        first -- the window a per-day budget is summed over (spec 3
        section 7: spend is summed from the record).
        """
        ...


@runtime_checkable
class StudyReader(Protocol):
    """Read side of the `study` table (spec 2 section 3). `records` is an
    exact match on one (family, anchor, card_kind); `study_rows` returns
    every row, ordered by ts, so a caller (the Syllabus aggregate's
    study_by_confusion) can group study history over its own pairs
    without querying one (family, anchor, card_kind) at a time.
    """
    def records(self, family: str, anchor: str, card_kind: str) -> list["StudyRecord"]: ...
    def study_rows(self) -> list["StudyRecord"]: ...
