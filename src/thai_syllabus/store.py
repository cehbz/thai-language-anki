"""SyllabusDb: sqlite-backed durable state (spec 2 section 2), plus
MediaStore, the content-addressed writer for media/objects/ (spec 2
section 1).

Ground rules from the spec: four tables and nothing else; WAL mode; one
transaction per append; caches are never evicted -- a re-ask appends, it
never updates or deletes. `ts` is stored as an integer count of
nanoseconds since the epoch (not the ISO string spec 2's prose examples
might suggest) because the `cache` table's primary key is (key_sha, ts):
nanosecond resolution combined with a per-connection monotonic bump (see
`_next_ts`) makes same-microsecond collisions impossible without needing
a synthetic surrogate key.

AssessmentReader.verdict / .is_waived / RecordWriter.append /
StudyReader.records are all implemented here exactly as ports.py declares
them; SyllabusDb also exposes some extra, non-Protocol methods (
assessments_of, append_judge_verdict, append_waiver, append_study,
add_sentence, add_media, set_pair_confusions) that spec 2 section 3 or the
migration/testing surface needs but spec 1's frozen Protocols do not
declare.

Cache-row conventions used by the higher-level convenience methods (judge
verdicts, waivers) -- these are spec 1/2's rule-level verdict/waiver
convention, distinct from spec 3's per-backend Provider/Assessor key
functions (provider.py/assessor.py own those; see their module
docstrings). Kept readable per spec 3's "canonical readable strings"
rule even though they predate it:
  - judge verdict:  port="assess", backend="judge",
                     key = "rule-verdict:RULE_ID:NOTE_ID:ARTIFACT_SHA"
                     (ARTIFACT_SHA is "-" when absent),
                     subject = note_id, question = {rule, note_id,
                     artifact_sha}, answer = {"verdict": bool}.
                     verdict() is an EXACT key_sha match, newest row wins.
  - learner waiver:  port="assess", backend="learner",
                     key = "waiver:RULE_ID:NOTE_ID:ARTIFACT_SHA"
                     (ARTIFACT_SHA is "-" when absent), subject = same
                     finding identity, question = {"kind": "waiver",
                     rule, note_id, artifact_sha},
                     answer = {"waived": bool, "reason"}.
                     is_waived() folds newest-wins over matching rows
                     (the learner backend's standard cache policy).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .ports import Answer, StudyRecord

if False:  # TYPE_CHECKING without importing at runtime
    from .rules import Finding


_SCHEMA = """
create table if not exists sentences (
    text_sha text primary key,
    text text not null,
    voice text not null,
    source text not null,
    origin text not null,
    licence text not null,
    acquired text not null
);

create table if not exists media (
    sha text primary key,
    kind text not null,
    ext text not null,
    source text not null,
    origin text not null,
    licence text not null,
    acquired text not null,
    speaker_id text,
    speaker_kind text
);

create table if not exists cache (
    port text not null,
    backend text not null,
    key text not null,
    key_sha text not null,
    subject text not null,
    question text not null,
    answer text not null,
    cost real not null default 0,
    ts integer not null,
    primary key (key_sha, ts)
);
create index if not exists cache_key_sha on cache (key_sha);
create index if not exists cache_subject on cache (subject);
create index if not exists cache_port_backend_key_sha on cache (port, backend, key_sha);

create table if not exists study (
    card_key text not null,
    compile_id text not null,
    ts integer not null,
    grade integer not null,
    time_ms integer not null
);
create index if not exists study_card_key on study (card_key);
"""


def _key_sha(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _rule_verdict_key(rule_id: str, note_id: str, artifact_sha: str | None) -> str:
    return f"rule-verdict:{rule_id}:{note_id}:{artifact_sha or '-'}"


def _finding_key(rule_id: str, note_id: str, artifact_sha: str | None) -> str:
    return f"waiver:{rule_id}:{note_id}:{artifact_sha or '-'}"


def _finding_subject(rule_id: str, note_id: str, artifact_sha: str | None) -> str:
    return json.dumps([rule_id, note_id, artifact_sha], sort_keys=True)


def _row_to_answer(row: tuple) -> Answer:
    port, backend, key, key_sha, subject, question, answer, cost, ts = row
    return Answer(port=port, backend=backend, key_sha=key_sha, key=key,
                 subject=subject, question=json.loads(question),
                 answer=json.loads(answer), cost=cost, ts=ts)


class SyllabusDb:
    """The append-only sqlite store: sentences, media provenance, cache,
    study (spec 2 section 2). One connection, WAL mode, one transaction
    per append.
    """

    def __init__(self, path: str | Path,
                pair_confusions: Mapping[str, str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path, isolation_level=None)
        self._con.execute("pragma journal_mode=WAL")
        self._con.executescript(_SCHEMA)
        self._last_ts = 0
        # card_key -> confusion resolution for StudyReader.records(confusion):
        # the study table only stores card_key, so aggregating "every pair
        # card under this confusion" needs curated pairs' confusion field,
        # which this store doesn't own (curated.py does). Callers that want
        # confusion-level study queries supply the mapping explicitly.
        self._pair_confusions: dict[str, str] = dict(pair_confusions or {})

    def close(self) -> None:
        self._con.close()

    def set_pair_confusions(self, pair_confusions: Mapping[str, str]) -> None:
        """pair_id -> confusion_id, from curated/pairs.yaml. Used only by
        StudyReader.records() when asked for a confusion rather than a
        literal card_key.
        """
        self._pair_confusions = dict(pair_confusions)

    def _next_ts(self, requested: int | None = None) -> int:
        base = requested if requested is not None else time.time_ns()
        self._last_ts = base if base > self._last_ts else self._last_ts + 1
        return self._last_ts

    # --- RecordWriter --------------------------------------------------

    def append(self, port: str, backend: str, key: str, subject: str,
               question: Any, answer: Any, cost: float = 0.0,
               ts: int | None = None) -> int:
        ts = self._next_ts(ts)
        with self._con:
            self._con.execute(
                "insert into cache (port, backend, key, key_sha, subject, "
                "question, answer, cost, ts) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (port, backend, key, _key_sha(key), subject,
                 json.dumps(question, sort_keys=True),
                 json.dumps(answer, sort_keys=True), cost, ts))
        return ts

    # --- CacheReader (spec 3): the general cache-first read surface -------

    def latest(self, port: str, backend: str, key: str) -> Answer | None:
        row = self._con.execute(
            "select port, backend, key, key_sha, subject, question, answer, "
            "cost, ts from cache where port=? and backend=? and key_sha=? "
            "order by ts desc limit 1", (port, backend, _key_sha(key))
        ).fetchone()
        if row is None:
            return None
        return _row_to_answer(row)

    # --- AssessmentReader ------------------------------------------------

    def verdict(self, rule_id: str, note_id: str,
                artifact_sha: str | None = None) -> bool | None:
        key = _rule_verdict_key(rule_id, note_id, artifact_sha)
        row = self._con.execute(
            "select answer from cache where port='assess' and backend='judge' "
            "and key_sha=? order by ts desc limit 1",
            (_key_sha(key),)).fetchone()
        if row is None:
            return None
        return bool(json.loads(row[0])["verdict"])

    def is_waived(self, finding: "Finding") -> bool:
        subject = _finding_subject(finding.rule, finding.note_id,
                                   finding.artifact_sha)
        row = self._con.execute(
            "select answer from cache where port='assess' and backend='learner' "
            "and subject=? order by ts desc limit 1", (subject,)).fetchone()
        if row is None:
            return False
        return bool(json.loads(row[0])["waived"])

    def assessments_of(self, subject: str) -> list[Answer]:
        rows = self._con.execute(
            "select port, backend, key, key_sha, subject, question, answer, "
            "cost, ts from cache where subject=? order by ts asc",
            (subject,)).fetchall()
        return [_row_to_answer(r) for r in rows]

    # --- convenience writers for the judge/waiver conventions -------------
    #
    # Readable per spec 3's "canonical readable strings, sha() only on
    # large/binary components" rule -- but NOT spec 3's own judge-backend
    # key (judge:sha(RUBRIC):sha(ARTIFACT):ROLE, owned by assessor.py). This
    # is spec 1/2's separate, already-shipped convention for judged-*rule*
    # verdicts consumed by Syllabus.report() through AssessmentReader.verdict
    # (keyed on rule_id/note_id/artifact_sha, not rubric text/role) -- see
    # the module docstring above.

    def append_judge_verdict(self, *, rule_id: str, note_id: str,
                             verdict: bool, artifact_sha: str | None = None,
                             evidence: str | None = None,
                             cost: float = 0.0) -> None:
        key = _rule_verdict_key(rule_id, note_id, artifact_sha)
        subject = _finding_subject(rule_id, note_id, artifact_sha)
        question = {"rule": rule_id, "note_id": note_id,
                    "artifact_sha": artifact_sha}
        answer: dict[str, Any] = {"verdict": verdict}
        if evidence is not None:
            answer["evidence"] = evidence
        self.append(port="assess", backend="judge", key=key, subject=subject,
                    question=question, answer=answer, cost=cost)

    def append_waiver(self, *, rule_id: str, note_id: str,
                      artifact_sha: str | None, waived: bool,
                      reason: str = "") -> None:
        key = _finding_key(rule_id, note_id, artifact_sha)
        subject = _finding_subject(rule_id, note_id, artifact_sha)
        question = {"kind": "waiver", "rule": rule_id, "note_id": note_id,
                    "artifact_sha": artifact_sha}
        answer = {"waived": waived, "reason": reason}
        self.append(port="assess", backend="learner", key=key, subject=subject,
                    question=question, answer=answer)

    # --- study / StudyReader ----------------------------------------------

    def append_study(self, *, card_key: str, compile_id: str, grade: int,
                     time_ms: int) -> None:
        ts = self._next_ts()
        with self._con:
            self._con.execute(
                "insert into study (card_key, compile_id, ts, grade, time_ms) "
                "values (?, ?, ?, ?, ?)", (card_key, compile_id, ts, grade, time_ms))

    def records(self, card_key_or_confusion: str) -> list[StudyRecord]:
        key = card_key_or_confusion
        card_keys: list[str]
        if key in self._pair_confusions.values():
            card_keys = [ck for ck, pair_confusion
                        in self._card_keys_by_pair_confusion(key)]
        else:
            card_keys = [key]
        if not card_keys:
            return []
        placeholders = ",".join("?" for _ in card_keys)
        rows = self._con.execute(
            f"select card_key, compile_id, ts, grade, time_ms from study "
            f"where card_key in ({placeholders}) order by ts asc", card_keys
        ).fetchall()
        return [StudyRecord(card_key=r[0], compile_id=r[1], ts=r[2],
                            grade=r[3], time_ms=r[4]) for r in rows]

    def _card_keys_by_pair_confusion(self, confusion: str):
        """Every study card_key belonging to a pair whose confusion is
        `confusion`. card_key convention: "<pair_id>::<card kind>" -- the
        pair id is the segment before the first "::".
        """
        pair_ids = {pid for pid, c in self._pair_confusions.items()
                   if c == confusion}
        if not pair_ids:
            return []
        all_card_keys = self._con.execute(
            "select distinct card_key from study").fetchall()
        matches = []
        for (ck,) in all_card_keys:
            pair_id = ck.split("::", 1)[0]
            if pair_id in pair_ids:
                matches.append((ck, confusion))
        return matches

    # --- sentences ----------------------------------------------------

    def add_sentence(self, *, text_sha: str, text: str, voice: str,
                     source: str, origin: str, licence: str,
                     acquired: date) -> None:
        with self._con:
            self._con.execute(
                "insert or ignore into sentences (text_sha, text, voice, "
                "source, origin, licence, acquired) values (?, ?, ?, ?, ?, ?, ?)",
                (text_sha, text, voice, source, origin, licence,
                 acquired.isoformat()))

    # --- media provenance ----------------------------------------------

    def add_media(self, *, sha: str, kind: str, ext: str, source: str,
                 origin: str, licence: str, acquired: date,
                 speaker_id: str | None = None,
                 speaker_kind: str | None = None) -> bool:
        """Idempotent: returns True if a new provenance row was inserted,
        False if `sha` already had one (so callers can count actual rows,
        not attempted writes).
        """
        with self._con:
            cur = self._con.execute(
                "insert or ignore into media (sha, kind, ext, source, origin, "
                "licence, acquired, speaker_id, speaker_kind) "
                "values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sha, kind, ext, source, origin, licence, acquired.isoformat(),
                 speaker_id, speaker_kind))
            return cur.rowcount > 0

    def has_media(self, sha: str) -> bool:
        row = self._con.execute("select 1 from media where sha=?", (sha,)).fetchone()
        return row is not None


@dataclass
class MediaStore:
    """Content-addressed writer for media/objects/<sha>.<ext> (spec 2
    section 1). Dumb bytes-in, sha-out; provenance is SyllabusDb.add_media's
    job, kept separate so tests can exercise the CAS write without a db.
    """
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)

    def _object_path(self, sha: str, ext: str) -> Path:
        return self.root / "objects" / f"{sha}.{ext}"

    def write(self, data: bytes, ext: str) -> str:
        sha = hashlib.sha256(data).hexdigest()
        path = self._object_path(sha, ext)
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return sha

    def has(self, sha: str, ext: str) -> bool:
        return self._object_path(sha, ext).exists()

    def path_for(self, sha: str, ext: str) -> Path:
        return self._object_path(sha, ext)
