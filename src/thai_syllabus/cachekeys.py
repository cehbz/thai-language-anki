"""Every cache key (spec 3 section 1 "Key"): one frozen dataclass per
backend key shape, defined here and used by every writer and reader; no
other module builds a key. `encode()` renders the canonical readable
string a `cache` row's `key` column stores (spec 2 section 2); `sha()`
truncates a large component (a rubric, a text) to 16 hex chars before it
goes into a key -- a word, a backend name, a role go in verbatim.
"""
import hashlib
from dataclasses import dataclass


def sha(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def preference_identity(candidates) -> str:
    """The judge backend's picture-preference identity: one candidate set,
    order-independent (sorted before hashing).
    """
    return sha(",".join(sorted(candidates)))


def rendition_identity(members) -> str:
    """A rendition's artifact identity: the member recordings it is made
    of, as one sha (member -> artifact sha, ordered by member).
    """
    return sha(",".join(members[m] for m in sorted(members)))


class CacheKey:
    """Base for the key dataclasses below. `encode()` is the canonical
    string a `cache` row's `key` column stores -- computed, never parsed
    back.
    """
    def encode(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class JudgeKey(CacheKey):
    """judge:RUBRIC_SHA:IDENTITY:ROLE. `identity` is an artifact sha, a
    note id (no artifact), or preference_identity()'s candidate-set sha.
    """
    rubric_sha: str
    identity: str
    role: str

    def encode(self) -> str:
        return f"judge:{self.rubric_sha}:{self.identity}:{self.role}"

    @classmethod
    def for_question(cls, question) -> "JudgeKey":
        """The key an AssessQuestion resolves to: identity is
        preference_identity() for picture-preference, else artifact_sha
        falling back to subject.
        """
        if question.role == "picture-preference":
            identity = preference_identity(question.params.get("candidates", []))
        else:
            identity = question.artifact_sha or question.subject
        return cls(rubric_sha=sha(question.rubric or ""), identity=identity,
                   role=question.role)

    @classmethod
    def for_rule(cls, rubric: str | None, artifact_sha: str | None, note_id: str,
                role: str) -> "JudgeKey":
        """A judged Rule's verdict key: identity is artifact_sha, falling
        back to note_id.
        """
        return cls(rubric_sha=sha(rubric or ""), identity=artifact_sha or note_id,
                   role=role)


@dataclass(frozen=True)
class LearnerKey(CacheKey):
    """learner:ARTIFACT_SHA:ROLE. artifact_sha is "-" in the string when
    absent; no rubric (the learner backend carries none).
    """
    artifact_sha: str | None
    role: str

    def encode(self) -> str:
        return f"learner:{self.artifact_sha or '-'}:{self.role}"


@dataclass(frozen=True)
class FlagKey(CacheKey):
    """flag:FAMILY:ANCHOR:CARD_KIND:FLAGS -- one Anki flag import fact,
    the (card, flags) idempotence key spec 4 section 4 defines.
    """
    family: str
    anchor: str
    card_kind: str
    flags: int

    def encode(self) -> str:
        return f"flag:{self.family}:{self.anchor}:{self.card_kind}:{self.flags}"


@dataclass(frozen=True)
class LearnerNoteKey(CacheKey):
    """learner-note:ANCHOR:TEXT_SHA -- one ReviewNote harvest row."""
    anchor: str
    text_sha: str

    def encode(self) -> str:
        return f"learner-note:{self.anchor}:{self.text_sha}"


@dataclass(frozen=True)
class DrillKey(CacheKey):
    """learner:drill:PAIR_ID:CONFUSION -- one gallery pair-drill result."""
    pair_id: str
    confusion: str

    def encode(self) -> str:
        return f"learner:drill:{self.pair_id}:{self.confusion}"


@dataclass(frozen=True)
class ReverifyKey(CacheKey):
    """learner:reverify:IDENTITY:ROLE -- a flag on a tone-correctness
    role, which queues machine re-verification. IDENTITY is artifact_sha,
    falling back to anchor when absent.
    """
    artifact_sha: str | None
    anchor: str
    role: str

    def encode(self) -> str:
        identity = self.artifact_sha or self.anchor
        return f"learner:reverify:{identity}:{self.role}"


@dataclass(frozen=True)
class DirectionKey(CacheKey):
    """direction:SUBJECT:ROLE:TEXT_SHA -- one typed learner direction (spec
    5 section 1 kind 2: "a typed direction is recorded as a direction, not
    a rating"). TEXT_SHA is sha() of the direction text, so two different
    directions on the same (subject, role) are two rows, not one
    overwritten key.
    """
    subject: str
    role: str
    text_sha: str

    def encode(self) -> str:
        return f"direction:{self.subject}:{self.role}:{self.text_sha}"


@dataclass(frozen=True)
class WaiverKey(CacheKey):
    """waiver:RULE_ID:NOTE_ID:ARTIFACT_SHA -- a learner waiver over one
    Finding's identity. artifact_sha is "-" in the string when absent.
    """
    rule_id: str
    note_id: str
    artifact_sha: str | None

    def encode(self) -> str:
        return f"waiver:{self.rule_id}:{self.note_id}:{self.artifact_sha or '-'}"


@dataclass(frozen=True)
class MechanicalKey(CacheKey):
    """mech:CHECK:PARAMS:ARTIFACT_SHA, parameter-explicit: the checked
    thresholds (or the code version, where no parameter expresses the
    check) go in PARAMS, so a parameter change is a new key.
    """
    check: str
    params: str
    artifact_sha: str

    def encode(self) -> str:
        return f"mech:{self.check}:{self.params}:{self.artifact_sha}"


@dataclass(frozen=True)
class ProvideKey(CacheKey):
    """SOURCE:KIND:QUERY -- one Source ask. `source` is the backend's own
    name (empty when the query alone identifies the ask, as a fetch by
    url does), `kind` what varies the ask within that source (a tts
    voice; empty where the source has one ask shape), `query` what was
    asked for. Every field renders at its own fixed position, empty or
    not: encode() is injective over the three fields. No field's value is
    ever read back as another's, whichever fields are empty.
    """
    source: str
    kind: str
    query: str

    def encode(self) -> str:
        return f"{self.source}:{self.kind}:{self.query}"


@dataclass(frozen=True)
class LlmPromptKey(CacheKey):
    """llm:PRODUCER:MODEL:PROMPT_SHA -- one drafting ask. The prompt text
    is the whole contract, so it goes in as sha() of itself.
    """
    producer: str
    model: str
    prompt_sha: str

    def encode(self) -> str:
        return f"llm:{self.producer}:{self.model}:{self.prompt_sha}"


@dataclass(frozen=True)
class PairSearchKey(CacheKey):
    """pairs:CONFUSION:DICTIONARY_VERSION -- one minimal-pair search over
    one dictionary+G2P corpus; a dictionary bump is a new key.
    """
    confusion_id: str
    dictionary_version: str

    def encode(self) -> str:
        return f"pairs:{self.confusion_id}:{self.dictionary_version}"


@dataclass(frozen=True)
class RenditionAskKey(CacheKey):
    """provide:SOURCE:rendition:PAIR_ID -- one Source's rendition ask for
    one MinimalPair, appended under the pair even though the per-member
    lookups it made are cached under the members.
    """
    source: str
    pair_id: str

    def encode(self) -> str:
        return f"provide:{self.source}:rendition:{self.pair_id}"


@dataclass(frozen=True)
class AttemptOutcomeKey(CacheKey):
    """attempt:SUBJECT:KIND:SOURCE -- one attempt outcome row (spec 3
    section 6): what one Source produced for one need. Every field is
    required and never empty, so encode() is injective over the three.
    """
    subject: str
    kind: str
    source: str

    def __post_init__(self) -> None:
        if not self.subject or not self.kind or not self.source:
            raise ValueError("AttemptOutcomeKey requires a non-empty subject, kind and source")

    def encode(self) -> str:
        return f"attempt:{self.subject}:{self.kind}:{self.source}"


@dataclass(frozen=True)
class BatchMarkerKey(CacheKey):
    """batch-marker:BATCH_ID -- one marker row per run's judge batch
    (spec 3 section 4), released when the batch resolves, expires, or
    fails.
    """
    batch_id: str

    def encode(self) -> str:
        return f"batch-marker:{self.batch_id}"


@dataclass(frozen=True)
class RunReportKey(CacheKey):
    """runreport -- the constant label under which run.py appends one
    summary row per run() call. The `cache` table's primary key is
    (key_sha, ts), so every call still lands its own row.
    """
    def encode(self) -> str:
        return "runreport"
