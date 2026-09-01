"""What has been tried to picture a subject, and how it ended.

Artifact provenance -- where bytes came from, licence, date -- is a property
of a file and lives in the media manifest. This is a property of the subject:
many attempts, one outcome. Conflating them is what produced four stores that
between them could not answer "what has been tried for this word".

Append-only JSONL, folded on load, compacted by save(). A run killed inside a
long image filler is the normal case, and appending is what survives it --
the lesson the manifest learned by losing 445 files their provenance.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

SOURCING_PATH = Path("work") / "image_sourcing.jsonl"

DECISION_KINDS = ("judge-accepted", "human-accepted", "human-supplied",
                  "human-unpicturable")
#: Decisions an automated run must not reopen.
HUMAN_DECISIONS = ("human-accepted", "human-supplied", "human-unpicturable")


@dataclass(frozen=True)
class Candidate:
    url: str
    source: str
    license: str | None
    file: str
    passed: bool
    failed_rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class Attempt:
    """A query and what came of it. A query is constitutive of an attempt."""
    query: str
    query_source: str            # phrase | gloss | judge | human
    corpora: tuple[str, ...]
    rubric: str
    candidates: tuple[Candidate, ...]
    dated: str


@dataclass(frozen=True)
class Decision:
    kind: str                    # one of DECISION_KINDS
    file: str | None
    reason: str | None
    dated: str


@dataclass
class Record:
    family: str
    subject: str
    attempts: list[Attempt] = field(default_factory=list)
    decision: Decision | None = None

    @property
    def queries_tried(self) -> list[str]:
        return [a.query for a in self.attempts]

    @property
    def decided_by_human(self) -> bool:
        return self.decision is not None and self.decision.kind in HUMAN_DECISIONS


def _candidate_from(raw: dict) -> Candidate:
    return Candidate(url=raw["url"], source=raw["source"],
                     license=raw.get("license"), file=raw["file"],
                     passed=bool(raw.get("passed")),
                     failed_rules=tuple(raw.get("failed_rules", ())))


def _attempt_from(raw: dict) -> Attempt:
    return Attempt(query=raw["query"], query_source=raw["query_source"],
                   corpora=tuple(raw.get("corpora", ())),
                   rubric=raw.get("rubric", ""),
                   candidates=tuple(_candidate_from(c)
                                    for c in raw.get("candidates", ())),
                   dated=raw["dated"])


def _attempt_as_dict(attempt: Attempt) -> dict:
    return {
        "query": attempt.query, "query_source": attempt.query_source,
        "corpora": list(attempt.corpora), "rubric": attempt.rubric,
        "dated": attempt.dated,
        "candidates": [{"url": c.url, "source": c.source, "license": c.license,
                        "file": c.file, "passed": c.passed,
                        "failed_rules": list(c.failed_rules)}
                       for c in attempt.candidates],
    }


def _decision_as_dict(decision: Decision) -> dict:
    return {"kind": decision.kind, "file": decision.file,
            "reason": decision.reason, "dated": decision.dated}


class SourcingLog:
    """Records keyed on (family, subject), backed by an append-only log."""

    def __init__(self, root: Path,
                 records: dict[tuple[str, str], Record] | None = None):
        self.root = Path(root)
        self.records_by_key = records if records is not None else {}

    @classmethod
    def load(cls, root: Path) -> "SourcingLog":
        path = Path(root) / SOURCING_PATH
        records: dict[tuple[str, str], Record] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue      # a killed write leaves one torn line
                key = (event.get("family"), event.get("subject"))
                if key[0] is None or key[1] is None:
                    continue
                record = records.setdefault(key, Record(*key))
                if "attempt" in event:
                    record.attempts.append(_attempt_from(event["attempt"]))
                elif "decision" in event:
                    raw = event["decision"]
                    record.decision = Decision(
                        kind=raw["kind"], file=raw.get("file"),
                        reason=raw.get("reason"), dated=raw["dated"])
        return cls(root, records)

    def get(self, family: str, subject: str) -> Record:
        """The record for a subject; an empty one when nothing is known."""
        return self.records_by_key.get((family, subject), Record(family, subject))

    def records(self) -> list[Record]:
        return list(self.records_by_key.values())

    def _append(self, event: dict) -> None:
        path = self.root / SOURCING_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _record_for(self, family: str, subject: str) -> Record:
        return self.records_by_key.setdefault((family, subject),
                                              Record(family, subject))

    def record_attempt(self, family: str, subject: str,
                       attempt: Attempt) -> None:
        self._record_for(family, subject).attempts.append(attempt)
        self._append({"family": family, "subject": subject,
                      "attempt": _attempt_as_dict(attempt)})

    def record_decision(self, family: str, subject: str,
                        decision: Decision) -> None:
        if decision.kind not in DECISION_KINDS:
            raise ValueError(f"unknown decision kind: {decision.kind}")
        self._record_for(family, subject).decision = decision
        self._append({"family": family, "subject": subject,
                      "decision": _decision_as_dict(decision)})

    def save(self, root: Path) -> None:
        """Rewrite the log with one event per attempt and decision.

        Compaction only: the fold of the result equals the fold of what it
        replaced, which the round-trip test pins.
        """
        path = Path(root) / SOURCING_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        events = []
        for (family, subject), record in self.records_by_key.items():
            for attempt in record.attempts:
                events.append({"family": family, "subject": subject,
                               "attempt": _attempt_as_dict(attempt)})
            if record.decision is not None:
                events.append({"family": family, "subject": subject,
                               "decision": _decision_as_dict(record.decision)})
        path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
            encoding="utf-8")


def next_mechanism(record: Record, queries: list[str], rubric: str) -> str:
    """Which mechanism a subject is owed, derived from its record.

    Never stored: a status field beside the evidence is a field that can
    disagree with it. `queries` is the current query sequence and `rubric`
    the current rubric fingerprint, so a new phrase or a relaxed rule reopens
    a subject without anyone resetting anything.
    """
    if record.decision is not None:
        return "settled"

    # Attempts made under a different rubric say nothing about this one: a
    # relaxed rule or a new corpus makes those rejections worth revisiting.
    current = [a for a in record.attempts if a.rubric == rubric]
    tried = {a.query for a in current}
    if not current or any(query not in tried for query in queries):
        return "search"
    if not any(a.query_source == "judge" for a in current):
        return "rephrase"
    if not any(a.query_source == "human" for a in current):
        return "consult"
    return "waiting"
