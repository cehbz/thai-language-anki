"""Ports the Syllabus reads through (spec 1 defines them; spec 3 owns the
real backends). All three are read-only from the aggregate's point of view:
report() never calls a judge, fills() never calls a live tokenizer service.

Spec 2 (durable state) section 3 adds three more interfaces -- FrequencyMap,
RecordWriter, StudyReader -- not present when spec 1 was implemented.
AssessmentReader/MediaIndex/Tokenizer above are spec 1's contract and are
left untouched; store.py's SyllabusDb satisfies them exactly (isinstance
checks against these Protocols still pass) and additionally offers
`assessments_of` (spec 2's fuller read surface over the same cache table --
not part of the Protocol spec 1 already shipped, so it is not declared
here, only implemented).
"""
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .ids import ConfusionId, WordId
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
                artifact_sha: str | None = None) -> bool | None:
        """True/False for a cached judged-rule verdict; None if the
        (rule, note, artifact) has not been assessed yet.
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
    """
    def has_picture(self, word: "WordId") -> bool: ...
    def recording_speakers(self, word: "WordId") -> frozenset[str]: ...
    def rendition_speakers(self, pair_confusion: "ConfusionId") -> frozenset[str]: ...


class NullAssessmentReader:
    """No cached verdicts, no waivers -- the default when a caller has no
    AssessmentReader to plug in yet.
    """
    def verdict(self, rule_id: str, note_id: str,
                artifact_sha: str | None = None) -> bool | None:
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


# --- spec 2 section 3 additions -------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One `cache` table row, read back and decoded. Not named by spec 1;
    spec 2 section 3 names it as AssessmentReader.assessments_of's element
    type. `question`/`answer` are already-decoded JSON (whatever shape the
    writing backend used); `ts` is nanoseconds since the epoch (store.py's
    sortable, collision-resistant substitute for the cache table's `ts`
    column -- see store.py's docstring for why).
    """
    port: str
    backend: str
    key_sha: str
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
    """Word-frequency corpus lookup (spec 2 section 3). Not one of the four
    durable stores (spec 2 section 2 lists exactly four sqlite tables and no
    frequency table) -- this reads a static, unchanging project resource
    (data/frequency_th.txt), not deck state the Syllabus writes.
    """
    def rank(self, word_thai: str) -> int | None: ...


@runtime_checkable
class RecordWriter(Protocol):
    """Append-only write side of the `cache` table (spec 2 section 2/3).
    `key` is the backend's raw cache-key content (a string); the store
    hashes it to the `key_sha` the table actually indexes -- the table has
    no `key` column, only `key_sha`, so the raw key is not retained beyond
    its hash. Every append is one transaction (the checkpoint rule); never
    an update, never a delete.
    """
    def append(self, port: str, backend: str, key: str, subject: str,
               question: Any, answer: Any, cost: float = 0.0) -> None: ...


@runtime_checkable
class StudyReader(Protocol):
    """Read side of the `study` table (spec 2 section 3). `records` takes
    either a card_key (exact match against the table) or a ConfusionId
    (aggregated over every pair card_key carrying that confusion -- see
    store.py's SyllabusDb.records for how that aggregation is resolved,
    since the study table itself only stores card_key, not confusion).
    """
    def records(self, card_key_or_confusion: str) -> list["StudyRecord"]: ...
