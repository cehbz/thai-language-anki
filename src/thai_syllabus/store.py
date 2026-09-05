"""SyllabusDb: sqlite-backed durable state (spec 2 section 2), plus
MediaStore, the content-addressed writer for media/objects/ (spec 2
section 1).

Ground rules from the spec: five tables and nothing else; WAL mode; one
transaction per append; caches are never evicted -- a re-ask appends, it
never updates or deletes. `ts` is stored as an integer count of
nanoseconds since the epoch (not the ISO string spec 2's prose examples
might suggest) because the `cache` table's primary key is (key_sha, ts):
nanosecond resolution combined with a per-connection monotonic bump (see
`_next_ts`) makes same-microsecond collisions impossible without needing
a synthetic surrogate key.

AssessmentReader.verdict / .is_waived / RecordWriter.append /
StudyReader.records / StudyReader.study_rows are all implemented here
exactly as ports.py declares them; SyllabusDb also exposes some extra,
non-Protocol methods (assessments_of, append_judge_verdict, append_waiver,
append_study, add_sentence, add_media) that spec 2 section 3 or the
migration/testing surface needs but spec 1's frozen Protocols do not
declare.

Cache-row conventions used by the higher-level convenience methods (judge
verdicts, waivers) -- these predate spec 3's per-backend Provider/Assessor
key functions (provider.py/assessor.py own those; see their module
docstrings) but, per spec 4's "key-convention debt" item, the judge-verdict
one is no longer a separate convention:

  - judge verdict (MERGED into spec 3's judge-backend key, spec 4):
                     port="assess", backend="judge",
                     key = "judge:sha(RUBRIC):IDENTITY:ROLE" -- exactly
                     assessor.JudgeBackend.cache_key's shape, with ROLE =
                     the judged Rule's id (a judged Rule has no separate
                     Assessor "role" of its own; the rule id fills that
                     slot) and IDENTITY = artifact_sha when the judged
                     subject is an artifact, else note_id verbatim (per
                     cachekeys.sha's "only sha large/binary components"
                     rule -- note_id is already short and readable, so it
                     is not re-hashed). Falling back to note_id rather
                     than a bare placeholder matters here specifically:
                     report() judges MANY distinct non-artifact subjects
                     (e.g. one register check per sentence) under ONE
                     shared rubric+role, so a placeholder would silently
                     collide their verdicts (assessor.py's JudgeBackend
                     makes the same fallback, to its own `subject`, for
                     the identical reason -- see its module comment).
                     subject = note_id (spec 3's AssessQuestion.subject
                     shape), NOT the old finding-identity JSON blob --
                     that blob remains the WAIVER convention's subject
                     only (below), a deliberately separate, unmerged
                     concept (learner authority over findings, not judge
                     verdicts). question = {"role": rule_id,
                     "artifact_sha": ..., "rubric": ...} and
                     answer = {"value": bool, ...} -- both exactly
                     Assessor._append_verdict's shape, so a judged Rule's
                     verdict and a direct Assessor.ask("judge", ...) call
                     under the same (rubric, role, identity) land on the
                     SAME row: one convention, not two. verdict() is an
                     EXACT key_sha match, newest row wins.

                     The now-retired convention was
                     "rule-verdict:RULE_ID:NOTE_ID:ARTIFACT_SHA" with
                     answer={"verdict": bool} and subject = the
                     finding-identity JSON blob; rows written under it
                     (e.g. by an older migration run) will not be found
                     by the merged verdict() -- acceptable per spec 2
                     section 4 item 5's "changed questions must re-judge
                     anyway".
  - learner waiver:  port="assess", backend="learner",
                     key = "waiver:RULE_ID:NOTE_ID:ARTIFACT_SHA"
                     (ARTIFACT_SHA is "-" when absent), subject = same
                     finding identity, question = {"kind": "waiver",
                     rule, note_id, artifact_sha},
                     answer = {"waived": bool, "reason"}.
                     is_waived() folds newest-wins over matching rows
                     (the learner backend's standard cache policy).
                     Explicitly OUT of the spec-4 merge (item 4 scopes it
                     to "the judge verdict path" only): a waiver is a
                     learner-authority fact about a Finding's identity,
                     not a judge verdict, and spec 3's roster gives the
                     learner backend its own separate key shape
                     ("learner:sha(ARTIFACT):ROLE, no rubric") that this
                     convention does not claim to match either -- left as
                     its own, pre-existing thing.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .cachekeys import sha
from .entities import Sentence
from .media import Provenance, Speaker
from .ports import Answer, StudyRecord

if False:  # TYPE_CHECKING without importing at runtime
    from .rules import Finding

# Pillow is a hard dependency of picture ingest (spec 4 section 3):
# MediaStore.add_image normalizes at ingest (bounded long edge, metadata
# stripped, re-encoded); the stored, sha'd bytes are the normalized file.
from PIL import Image


_SCHEMA = """
create table if not exists sentences (
    text_sha text primary key,
    text text not null,
    gloss text not null,
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
    speaker_id text
);

create table if not exists speakers (
    id text primary key,
    kind text not null,
    sex text not null default 'unknown',
    age_band text not null default 'unknown',
    region text not null default 'unknown'
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
    time_ms integer not null,
    primary key (card_key, ts)
);
"""


def _key_sha(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _judge_verdict_key(rule_id: str, note_id: str, artifact_sha: str | None,
                       rubric: str | None) -> str:
    """The merged spec-3 judge-backend key shape (see module docstring):
    role = rule_id; identity = artifact_sha, falling back to note_id.
    Exactly assessor.JudgeBackend.cache_key's formula.
    """
    identity = artifact_sha or note_id
    return f"judge:{sha(rubric or '')}:{identity}:{rule_id}"


def _finding_key(rule_id: str, note_id: str, artifact_sha: str | None) -> str:
    return f"waiver:{rule_id}:{note_id}:{artifact_sha or '-'}"


def _finding_subject(rule_id: str, note_id: str, artifact_sha: str | None) -> str:
    return json.dumps([rule_id, note_id, artifact_sha], sort_keys=True)


def _row_to_answer(row: tuple) -> Answer:
    port, backend, key, key_sha, subject, question, answer, cost, ts = row
    return Answer(port=port, backend=backend, key_sha=key_sha, key=key,
                 subject=subject, question=json.loads(question),
                 answer=json.loads(answer), cost=cost, ts=ts)


def _row_to_study_record(row: tuple) -> StudyRecord:
    card_key, compile_id, ts, grade, time_ms = row
    return StudyRecord(card_key=card_key, compile_id=compile_id, ts=ts,
                       grade=grade, time_ms=time_ms)


class SyllabusDb:
    """The append-only sqlite store: sentences, media provenance, cache,
    study (spec 2 section 2). One connection, WAL mode, one transaction
    per append.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path, isolation_level=None)
        self._con.execute("pragma journal_mode=WAL")
        self._con.executescript(_SCHEMA)
        self._last_ts = 0

    def close(self) -> None:
        self._con.close()

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
                artifact_sha: str | None = None,
                rubric: str | None = None) -> bool | None:
        key = _judge_verdict_key(rule_id, note_id, artifact_sha, rubric)
        row = self._con.execute(
            "select answer from cache where port='assess' and backend='judge' "
            "and key_sha=? order by ts desc limit 1",
            (_key_sha(key),)).fetchone()
        if row is None:
            return None
        return bool(json.loads(row[0])["value"])

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
                             rubric: str | None = None,
                             evidence: str | None = None,
                             cost: float = 0.0) -> None:
        key = _judge_verdict_key(rule_id, note_id, artifact_sha, rubric)
        # subject = note_id, matching spec 3's AssessQuestion.subject shape
        # (the merged convention -- see module docstring); NOT the old
        # finding-identity blob, which stays the waiver convention's own.
        question = {"role": rule_id, "artifact_sha": artifact_sha,
                    "rubric": rubric}
        answer: dict[str, Any] = {"value": verdict}
        if evidence is not None:
            answer["evidence"] = evidence
        self.append(port="assess", backend="judge", key=key, subject=note_id,
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
                     time_ms: int, ts: int | None = None) -> bool:
        """Insert-or-ignore on the table's primary key (card_key, ts).
        Returns True if a new row was inserted, False if that exact
        (card_key, ts) pair already had one.

        `ts` defaults to an auto-generated, collision-avoided value (via
        `_next_ts`, the cache table's own convention) for callers with no
        externally meaningful timestamp of their own. An EXPLICIT `ts` --
        anki_import.py's revlog import passes the revlog row's own
        (already-unique) epoch-ms review id -- is stored VERBATIM instead:
        `_next_ts`'s monotonic bump-past-the-last-seen-value exists to
        avoid same-instant collisions among freshly generated nanosecond
        timestamps, and would silently renumber a real, externally unique
        id drawn from a completely different, smaller scale (ms) -- which
        would break "idempotent by (card_key, ts)" (spec 4 section 4) on
        any reimport after this store has also done nanosecond-scale
        cache writes.
        """
        if ts is None:
            ts = self._next_ts()
        with self._con:
            cur = self._con.execute(
                "insert or ignore into study (card_key, compile_id, ts, grade, "
                "time_ms) values (?, ?, ?, ?, ?)",
                (card_key, compile_id, ts, grade, time_ms))
            return cur.rowcount > 0

    def records(self, card_key: str) -> list[StudyRecord]:
        rows = self._con.execute(
            "select card_key, compile_id, ts, grade, time_ms from study "
            "where card_key=? order by ts asc", (card_key,)).fetchall()
        return [_row_to_study_record(r) for r in rows]

    def study_rows(self) -> list[StudyRecord]:
        """Every `study` row, ordered by ts, for callers (the Syllabus
        aggregate) that group study history themselves rather than
        querying one card_key at a time.
        """
        rows = self._con.execute(
            "select card_key, compile_id, ts, grade, time_ms from study "
            "order by ts asc").fetchall()
        return [_row_to_study_record(r) for r in rows]

    # --- sentences ----------------------------------------------------

    def add_sentence(self, *, text_sha: str, text: str, gloss: str, voice: str,
                     source: str, origin: str, licence: str,
                     acquired: date) -> bool:
        """Insert-or-ignore on text_sha. Returns True if a new row was
        inserted, False if text_sha already had one.
        """
        with self._con:
            cur = self._con.execute(
                "insert or ignore into sentences (text_sha, text, gloss, voice, "
                "source, origin, licence, acquired) values (?, ?, ?, ?, ?, ?, ?, ?)",
                (text_sha, text, gloss, voice, source, origin, licence,
                 acquired.isoformat()))
            return cur.rowcount > 0

    # --- speakers -------------------------------------------------------

    def add_speaker(self, speaker: Speaker) -> None:
        """Insert-or-ignore on id (spec 2 section 2): a speaker's
        attributes are never overwritten once recorded.
        """
        with self._con:
            self._con.execute(
                "insert or ignore into speakers (id, kind, sex, age_band, region) "
                "values (?, ?, ?, ?, ?)",
                (speaker.id, speaker.kind, speaker.sex, speaker.age_band, speaker.region))

    def speaker(self, speaker_id: str) -> Speaker | None:
        row = self._con.execute(
            "select id, kind, sex, age_band, region from speakers where id=?",
            (speaker_id,)).fetchone()
        if row is None:
            return None
        id_, kind, sex, age_band, region = row
        return Speaker(id=id_, kind=kind, sex=sex, age_band=age_band, region=region)

    # --- media provenance ----------------------------------------------

    def add_media(self, *, sha: str, kind: str, ext: str, source: str,
                 origin: str, licence: str, acquired: date,
                 speaker_id: str | None = None) -> bool:
        """Idempotent: returns True if a new provenance row was inserted,
        False if `sha` already had one (so callers can count actual rows,
        not attempted writes). `speaker_id`, when given, must already name
        a row in `speakers` (add_speaker first) -- fails fast otherwise.
        """
        if speaker_id is not None and self.speaker(speaker_id) is None:
            raise ValueError(
                f"add_media: speaker_id {speaker_id!r} names no speaker "
                "(call add_speaker first)")
        with self._con:
            cur = self._con.execute(
                "insert or ignore into media (sha, kind, ext, source, origin, "
                "licence, acquired, speaker_id) "
                "values (?, ?, ?, ?, ?, ?, ?, ?)",
                (sha, kind, ext, source, origin, licence, acquired.isoformat(),
                 speaker_id))
            return cur.rowcount > 0

    def all_sentences(self) -> list[Sentence]:
        """Every `sentences` row, reconstituted as entities.Sentence (spec 2
        section 2 stores sentences; spec 1 section 1 owns the entity). Not a
        Protocol method (ports.py names no SentenceReader) -- load_syllabus
        (wiring.py) is this method's one caller, assembling a Syllabus from
        db-backed state alongside the curated files.
        """
        rows = self._con.execute(
            "select text, gloss, voice, source, origin, licence, acquired "
            "from sentences").fetchall()
        return [Sentence(text=text, gloss=gloss, voice=voice,
                         provenance=Provenance(source=source, origin=origin,
                                              licence=licence,
                                              acquired=date.fromisoformat(acquired)))
               for text, gloss, voice, source, origin, licence, acquired in rows]

    def has_media(self, sha: str) -> bool:
        row = self._con.execute("select 1 from media where sha=?", (sha,)).fetchone()
        return row is not None

    def media_provenance(self, sha: str) -> dict[str, Any] | None:
        """One `media` row, decoded (spec 2 section 2) -- compile.py's
        source for the file extension a resolved artifact sha was stored
        under, plus source/speaker for src-provenance tags and minimal_pair
        MemberKey's speaker component. `speaker` is the resolved Speaker
        for `speaker_id`, or None when `speaker_id` is absent. Not part of
        any Protocol (spec 1's MediaIndex is a narrower has/speakers-only
        read); this is compile()'s own dependency on SyllabusDb directly,
        same footing as the module docstring's other non-Protocol
        convenience methods.
        """
        row = self._con.execute(
            "select sha, kind, ext, source, origin, licence, acquired, "
            "speaker_id from media where sha=?", (sha,)).fetchone()
        if row is None:
            return None
        keys = ("sha", "kind", "ext", "source", "origin", "licence",
               "acquired", "speaker_id")
        result = dict(zip(keys, row))
        result["speaker"] = self.speaker(result["speaker_id"]) if result["speaker_id"] else None
        return result


IMAGE_MAX_LONG_EDGE = 800  # px (spec 4 section 3)

# Pillow format name -> file extension MediaStore stores it under.
_FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


@dataclass(frozen=True)
class ImageIngestResult:
    """MediaStore.add_image's return value: the sha (and extension) the
    normalized bytes were written under.
    """
    sha: str
    ext: str


def _normalize_image(data: bytes, ext: str) -> tuple[bytes, str]:
    """Bounded long edge (IMAGE_MAX_LONG_EDGE, aspect preserved), metadata
    stripped, re-encoded (spec 4 section 3). Building a brand-new Image
    from just the pixel data (`putdata`) is what strips metadata: EXIF/ICC/
    text chunks live on the source Image object and are never copied over,
    rather than being enumerated and deleted one by one.
    """
    import io

    with Image.open(io.BytesIO(data)) as src:
        src.load()
        fmt = src.format or ext.upper()
        mode = src.mode
        if mode not in ("RGB", "RGBA", "L"):
            src = src.convert("RGBA" if "A" in mode or mode == "P" else "RGB")
            mode = src.mode

        w, h = src.size
        long_edge = max(w, h)
        if long_edge > IMAGE_MAX_LONG_EDGE:
            scale = IMAGE_MAX_LONG_EDGE / long_edge
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            src = src.resize(new_size, Image.LANCZOS)

        save_fmt = fmt if fmt in _FORMAT_EXT else "PNG"
        clean_mode = "RGB" if save_fmt == "JPEG" and mode == "RGBA" else mode
        pixels = src.convert(clean_mode) if clean_mode != mode else src
        clean = Image.frombytes(clean_mode, pixels.size, pixels.tobytes())

        out = io.BytesIO()
        clean.save(out, format=save_fmt)
        return out.getvalue(), _FORMAT_EXT[save_fmt]


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

    def add_image(self, data: bytes, ext: str) -> ImageIngestResult:
        """Ingest normalization (spec 4 section 3): the stored, sha'd bytes
        are the normalized file -- what a judge or the card itself sees is
        identical.
        """
        try:
            normalized, out_ext = _normalize_image(data, ext)
        except Exception as exc:
            raise ValueError(f"cannot decode image: {exc}") from exc
        written_sha = self.write(normalized, out_ext)
        return ImageIngestResult(sha=written_sha, ext=out_ext)

    def has(self, sha: str, ext: str) -> bool:
        return self._object_path(sha, ext).exists()

    def path_for(self, sha: str, ext: str) -> Path:
        return self._object_path(sha, ext)
