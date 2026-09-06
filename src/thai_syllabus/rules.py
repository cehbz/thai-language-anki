"""The rule model and its outputs (spec 1, section 4): Rule, Finding,
Metric, Report, plus the Gaps and Compile values that report()/gaps()/
compile() produce.

Rule shapes and what they return:
  check(syllabus)   -> list[Finding]   -- iterates its own notes internally
  measure(syllabus) -> Metric
  judged rules carry rubric text; report() reads their cached verdicts
  through the AssessmentReader port.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warn", "info"]
RuleShape = Literal["check", "measure", "judged", "compile"]
OrderKind = Literal["word_target", "pair", "grapheme", "sentence"]


@dataclass(frozen=True)
class OrderEntry:
    """One entry in Syllabus.order(): id is a TargetId for word_target, a
    PairId for pair, a grapheme symbol for grapheme, a text_sha for
    sentence.
    """
    kind: OrderKind
    id: str


@dataclass(frozen=True)
class Finding:
    """One rule failing for one note. (rule, note_id, artifact_sha) is the
    identity waivers reference.
    """
    rule: str
    note_id: str
    evidence: str
    artifact_sha: str | None = None

    def identity(self) -> tuple[str, str, str | None]:
        return (self.rule, self.note_id, self.artifact_sha)


@dataclass(frozen=True)
class Metric:
    rule: str
    value: float
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Report:
    """syllabus_state_id identifies the aggregate's content; rulebook_id
    (spec 3 section 6) identifies what judged it -- sha of rulebook.yaml's
    text plus the registry's rule ids. The report is stale when either
    differs from the live value (Syllabus.state_id() /
    Syllabus.rulebook_id()).
    """
    syllabus_state_id: str
    rulebook_id: str
    findings: tuple[Finding, ...]
    metrics: tuple[Metric, ...]
    gate: bool


@dataclass(frozen=True)
class Gaps:
    """What sourcing should produce next (input to spec 3's batch run)."""
    missing_renditions: tuple[str, ...]        # ConfusionId, undercovered
    unfilled_targets: tuple[str, ...]           # TargetId
    words_missing_pictures: tuple[str, ...]     # WordId
    words_missing_recordings: tuple[str, ...]   # WordId
    graphemes_missing_keyword_data: tuple[str, ...]  # symbol
    sentence_recordings: tuple[str, ...] = ()   # text_sha, no recording
    scene_pictures: tuple[str, ...] = ()        # text_sha, no scene picture


@dataclass(frozen=True)
class DroppedCard:
    """One (subject, template) compile.py left out of the package, and
    why: a gate field left the card with no content (reason starts
    "gated: ..."), or a current-best artifact its front depends on is
    missing (spec 4 section 3's "never an empty front" -- reason starts
    "no current-best ", or is "no name word"/"no name recording" for a
    grapheme, "no rendition" for a minimal pair).
    """
    family: str    # "word" | "minimal_pair" | "grapheme" | "sentence"
    kind: str      # the template name, e.g. "Listening", "Production"
    subject: str   # word id / MemberKey / grapheme symbol / sentence key
    reason: str


@dataclass(frozen=True)
class CompileReport:
    """What compile() produced beyond the .apkg file itself (spec 4).
    `findings` carries card/unique-front Findings, computed over the
    compiled notes themselves (report() cannot see them -- they don't
    exist until compile builds the notes).
    """
    compile_id: str
    gate: bool
    forced: bool
    warnings: tuple[str, ...]
    notes_written: int
    cards_written: int
    dropped: tuple[DroppedCard, ...]
    out_path: str
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class Compile:
    """Spec 4's result value. `compile_id` = syllabus_state_id + a
    timestamp, the value stamped into every note's CompileId field (spec 4
    section 2); `report` carries the compile-time detail (dropped cards,
    gate/force status, counts). The translation itself is compile.py's.
    """
    label: str
    syllabus_state_id: str
    compile_id: str
    report: CompileReport


@dataclass(frozen=True)
class Rule:
    id: str
    principle: str
    severity: Severity
    shape: RuleShape
    # exactly one of these is set, matching `shape`
    check: Callable[[Any], list[Finding]] | None = None
    measure: Callable[[Any], Metric] | None = None
    rubric: str | None = None
    # judged only: enumerates the (note_id, artifact_sha) pairs this rule
    # asks the Assess port about.
    judged_subjects: Callable[[Any], list[tuple[str, str | None]]] | None = None
    # judged only: the Assess role whose verdicts report() reads, defaults
    # to `id`.
    role: str | None = None

    def __post_init__(self) -> None:
        # shape="compile" is evaluated by compile.py directly against
        # compiled notes -- it carries no check/measure/judged_subjects
        # function and report() never dispatches it.
        if self.shape != "compile":
            shape_field = {"check": self.check, "measure": self.measure,
                           "judged": self.judged_subjects}[self.shape]
            if shape_field is None:
                raise ValueError(f"rule {self.id!r} is shape={self.shape!r} but "
                                 f"has no matching function")
            if self.shape == "judged" and not self.rubric:
                raise ValueError(f"judged rule {self.id!r} needs rubric text")
        if self.role is None:
            object.__setattr__(self, "role", self.id)
