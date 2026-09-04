"""Content-addressed media values (spec 1, section 1).

Picture and Recording are dumb files: bytes (identified by hash) plus
provenance. All learning semantics live in the relationships that consume
them (word -> picture, sentence -> scene picture, pair -> renditions, ...),
which are spec 2's record, not fields here.
"""
from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class Provenance:
    source: str
    origin: str
    licence: str
    acquired: date


@dataclass(frozen=True)
class Speaker:
    id: str
    kind: Literal["native", "synthetic"]
    sex: Literal["male", "female", "unknown"] = "unknown"
    age_band: Literal["child", "adult", "older", "unknown"] = "unknown"
    region: str = "unknown"


@dataclass(frozen=True)
class Picture:
    sha: str
    provenance: Provenance


@dataclass(frozen=True)
class Recording:
    sha: str
    provenance: Provenance
    speaker: Speaker
