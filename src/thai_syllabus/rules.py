"""The rule model and its outputs (spec 1, section 4): Rule, Finding,
Metric, Report, plus the Gaps and Compile values that report()/gaps()/
compile() produce.

Rule shapes and what they return:
  check(syllabus)   -> list[Finding]   -- iterates its own notes internally
  measure(syllabus) -> Metric
  judged rules carry rubric text; report() reads cached verdicts through
  the AssessmentReader port instead of calling the judge.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warn", "info"]
RuleShape = Literal["check", "measure", "judged"]


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
    syllabus_state_id: str
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


@dataclass(frozen=True)
class Compile:
    """Spec 4's stub: compile() itself raises NotImplementedError here, but
    the value type is named so callers can reference it.
    """
    label: str
    syllabus_state_id: str


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

    def __post_init__(self) -> None:
        shape_field = {"check": self.check, "measure": self.measure,
                       "judged": self.judged_subjects}[self.shape]
        if shape_field is None:
            raise ValueError(f"rule {self.id!r} is shape={self.shape!r} but "
                             f"has no matching function")
        if self.shape == "judged" and not self.rubric:
            raise ValueError(f"judged rule {self.id!r} needs rubric text")
