"""Content-addressed media values (spec 1 section 1): a Picture and a
Recording are bytes (identified by their sha) plus provenance. The
relationships that consume them (word -> picture, sentence -> scene
picture, pair -> renditions) live in spec 2's record.
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
