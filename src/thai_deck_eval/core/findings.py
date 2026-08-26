from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field

class Severity(StrEnum):
    ERROR = "error"; WARN = "warn"; INFO = "info"

class Dimension(StrEnum):
    INTEGRITY = "integrity"; LANGUAGE = "language"
    METHOD = "method"; CONTENT = "content"

class Stage(StrEnum):
    SCHEMA = "schema"; MECHANICAL = "mechanical"
    LINGUISTIC = "linguistic"; METHOD = "method"; JUDGE = "judge"

class Finding(BaseModel):
    rule: str
    severity: Severity
    dimension: Dimension
    message: str
    note_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

class Metric(BaseModel):
    name: str
    value: float
    dimension: Dimension = Dimension.METHOD
    detail: dict[str, Any] = Field(default_factory=dict)
