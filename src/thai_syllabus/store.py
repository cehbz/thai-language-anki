"""SyllabusDb: sqlite-backed durable state (spec 2 section 2), plus
MediaStore, the content-addressed writer for media/objects/ (spec 2
section 1).

Five tables and nothing else; WAL mode; one transaction per append; a
cache row is never evicted -- a re-ask appends. `ts` is an integer count
of nanoseconds since the epoch, bumped monotonically per connection
(`_next_ts`), so the `cache` table's (key_sha, ts) primary key never
collides.

`append`/`latest`/`verdict` take a cachekeys.py CacheKey, and store
`key.encode()` under the `key` column (readable, for inspection) and its
sha256 under `key_sha`, the indexed column every lookup matches on.

AssessmentReader, RecordWriter, CacheReader and StudyReader are
implemented here as ports.py declares them; assessments_of, append_waiver,
append_study, add_sentence and add_media are further methods spec 2
section 3 and the migration surface need.
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

from .cachekeys import CacheKey, WaiverKey
from .entities import Clauses, Sentence, clauses_from_json, clauses_to_json
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
    clauses text,
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
    family text not null,
    anchor text not null,
    card_kind text not null,
    compile_id text not null,
    ts integer not null,
    grade integer not null,
    time_ms integer not null,
    member_index text,
    speaker_id text,
    primary key (family, anchor, card_kind, ts)
);
"""


def _key_text(key: "CacheKey") -> str:
    return key.encode()


def _key_sha(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _row_to_answer(row: tuple) -> Answer:
    port, backend, key, key_sha, subject, question, answer, cost, ts = row
    return Answer(port=port, backend=backend, key_sha=key_sha, key=key,
                 subject=subject, question=json.loads(question),
                 answer=json.loads(answer), cost=cost, ts=ts)


def _row_to_study_record(row: tuple) -> StudyRecord:
    family, anchor, card_kind, compile_id, ts, grade, time_ms, member_index, speaker_id = row
    return StudyRecord(family=family, anchor=anchor, card_kind=card_kind,
                       compile_id=compile_id, ts=ts, grade=grade, time_ms=time_ms,
                       member_index=member_index, speaker_id=speaker_id)


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
        self._add_clauses_column_if_missing()
        self._last_ts = 0

    def _add_clauses_column_if_missing(self) -> None:
        """A pre-r10 db's `sentences` table predates the `clauses` column
        (spec 2 section 4 migration): add it once, nullable, so existing
        rows read back with clauses=None until set_clauses backfills them.
        """
        columns = {row[1] for row in self._con.execute("pragma table_info(sentences)")}
        if "clauses" not in columns:
            self._con.execute("alter table sentences add column clauses text")

    def close(self) -> None:
        self._con.close()

    def _next_ts(self, requested: int | None = None) -> int:
        base = requested if requested is not None else time.time_ns()
        self._last_ts = base if base > self._last_ts else self._last_ts + 1
        return self._last_ts

    # --- RecordWriter --------------------------------------------------

    def append(self, port: str, backend: str, key: "CacheKey", subject: str,
               question: Any, answer: Any, cost: float = 0.0,
               ts: int | None = None) -> int:
        ts = self._next_ts(ts)
        key_text = _key_text(key)
        with self._con:
            self._con.execute(
                "insert into cache (port, backend, key, key_sha, subject, "
                "question, answer, cost, ts) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (port, backend, key_text, _key_sha(key_text), subject,
                 json.dumps(question, sort_keys=True),
                 json.dumps(answer, sort_keys=True), cost, ts))
        return ts

    # --- CacheReader (spec 3): the general cache-first read surface -------

    def latest(self, port: str, backend: str, key: "CacheKey") -> Answer | None:
        row = self._con.execute(
            "select port, backend, key, key_sha, subject, question, answer, "
            "cost, ts from cache where port=? and backend=? and key_sha=? "
            "order by ts desc limit 1", (port, backend, _key_sha(_key_text(key)))
        ).fetchone()
        if row is None:
            return None
        return _row_to_answer(row)

    # --- AssessmentReader ------------------------------------------------

    def verdict(self, backend: str, key: "CacheKey") -> Answer | None:
        return self.latest("assess", backend, key)

    def is_waived(self, finding: "Finding") -> bool:
        key = WaiverKey(rule_id=finding.rule, note_id=finding.note_id,
                        artifact_sha=finding.artifact_sha)
        row = self._con.execute(
            "select answer from cache where port='assess' and backend='learner' "
            "and key_sha=? order by ts desc limit 1",
            (_key_sha(key.encode()),)).fetchone()
        if row is None:
            return False
        return bool(json.loads(row[0])["waived"])

    def assessments_of(self, subject: str) -> list[Answer]:
        rows = self._con.execute(
            "select port, backend, key, key_sha, subject, question, answer, "
            "cost, ts from cache where subject=? order by ts asc",
            (subject,)).fetchall()
        return [_row_to_answer(r) for r in rows]

    def rows_since(self, port: str, backend: str, since_ts: int) -> list[Answer]:
        rows = self._con.execute(
            "select port, backend, key, key_sha, subject, question, answer, "
            "cost, ts from cache where port=? and backend=? and ts>=? order by ts asc",
            (port, backend, since_ts)).fetchall()
        return [_row_to_answer(r) for r in rows]

    # --- convenience writers ------------------------------------------------

    def append_waiver(self, *, rule_id: str, note_id: str,
                      artifact_sha: str | None, waived: bool,
                      reason: str = "") -> None:
        """port="assess", backend="learner", key=cachekeys.WaiverKey(...);
        subject=note_id, so assessments_of(note_id) sees the waiver.
        """
        key = WaiverKey(rule_id=rule_id, note_id=note_id, artifact_sha=artifact_sha)
        question = {"kind": "waiver", "rule": rule_id, "note_id": note_id,
                    "artifact_sha": artifact_sha}
        answer = {"waived": waived, "reason": reason}
        self.append(port="assess", backend="learner", key=key, subject=note_id,
                    question=question, answer=answer)

    # --- study / StudyReader ----------------------------------------------

    def append_study(self, record: StudyRecord) -> bool:
        """Insert-or-ignore on the table's primary key (family, anchor,
        card_kind, ts). Returns True if a new row was inserted, False if
        that exact key already had one. `record.ts` is stored verbatim
        (anki_import.py's revlog import passes the revlog row's own
        epoch-ms review id).
        """
        with self._con:
            cur = self._con.execute(
                "insert or ignore into study (family, anchor, card_kind, "
                "compile_id, ts, grade, time_ms, member_index, speaker_id) "
                "values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.family, record.anchor, record.card_kind, record.compile_id,
                 record.ts, record.grade, record.time_ms, record.member_index,
                 record.speaker_id))
            return cur.rowcount > 0

    def records(self, family: str, anchor: str, card_kind: str) -> list[StudyRecord]:
        rows = self._con.execute(
            "select family, anchor, card_kind, compile_id, ts, grade, time_ms, "
            "member_index, speaker_id from study where family=? and anchor=? "
            "and card_kind=? order by ts asc", (family, anchor, card_kind)).fetchall()
        return [_row_to_study_record(r) for r in rows]

    def study_rows(self) -> list[StudyRecord]:
        """Every `study` row, ordered by ts, for a caller that groups
        study history itself (the Syllabus aggregate does).
        """
        rows = self._con.execute(
            "select family, anchor, card_kind, compile_id, ts, grade, time_ms, "
            "member_index, speaker_id from study order by ts asc").fetchall()
        return [_row_to_study_record(r) for r in rows]

    # --- sentences ----------------------------------------------------

    def add_sentence(self, *, text_sha: str, text: str, clauses: Clauses, gloss: str,
                     voice: str, source: str, origin: str, licence: str,
                     acquired: date) -> bool:
        """Insert-or-ignore on text_sha. Returns True if a new row was
        inserted, False if text_sha already had one. `clauses` is stored as
        clauses_to_json's JSON shape (spec 2 section 2).
        """
        with self._con:
            cur = self._con.execute(
                "insert or ignore into sentences (text_sha, text, clauses, gloss, "
                "voice, source, origin, licence, acquired) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (text_sha, text, json.dumps(clauses_to_json(clauses)), gloss, voice,
                 source, origin, licence, acquired.isoformat()))
            return cur.rowcount > 0

    def set_clauses(self, text_sha: str, clauses: Clauses) -> None:
        """Backfill a sentence row's clauses column (spec 2 section 4
        migration's parse-ask step writes verified clauses this way).
        """
        with self._con:
            self._con.execute(
                "update sentences set clauses=? where text_sha=?",
                (json.dumps(clauses_to_json(clauses)), text_sha))

    def delete_sentence(self, text_sha: str) -> None:
        """Remove a sentences row (spec 2 section 4 migration deletes a
        row whose parse ask fails; the draft that produced it stays in the
        record's own cache rows).
        """
        with self._con:
            self._con.execute("delete from sentences where text_sha=?", (text_sha,))

    def sentences_without_clauses(self) -> list[tuple[str, str]]:
        """(text_sha, text) for every sentences row with no clauses yet --
        the migration's worklist for the parse ask.
        """
        rows = self._con.execute(
            "select text_sha, text from sentences where clauses is null").fetchall()
        return [(text_sha, text) for text_sha, text in rows]

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
        db-backed state alongside the curated files. Raises ValueError
        naming the row's text_sha when its clauses column is NULL (a
        pre-r10 row the migration has not backfilled yet) or does not parse
        (clauses_from_json's own ValueError).
        """
        rows = self._con.execute(
            "select text_sha, text, clauses, gloss, voice, source, origin, licence, "
            "acquired from sentences").fetchall()
        sentences = []
        for text_sha, text, clauses_json, gloss, voice, source, origin, licence, acquired in rows:
            if clauses_json is None:
                raise ValueError(f"sentence {text_sha!r} has no clauses (run migrate)")
            try:
                clauses = clauses_from_json(json.loads(clauses_json))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"sentence {text_sha!r} has malformed clauses: {exc}") from exc
            sentences.append(Sentence(
                clauses=clauses, text=text, gloss=gloss, voice=voice,
                provenance=Provenance(source=source, origin=origin, licence=licence,
                                      acquired=date.fromisoformat(acquired))))
        return sentences

    def has_media(self, sha: str) -> bool:
        row = self._con.execute("select 1 from media where sha=?", (sha,)).fetchone()
        return row is not None

    def media_provenance(self, sha: str) -> dict[str, Any] | None:
        """One `media` row, decoded (spec 2 section 2): the extension a
        sha was stored under, its source, and `speaker`, the resolved
        Speaker for `speaker_id` or None when there is none.
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
    stripped, re-encoded (spec 4 section 3). The saved image is built from
    the source's pixel data alone, so EXIF/ICC/text chunks never carry
    over.
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
    section 1): bytes in, sha out. Provenance is SyllabusDb.add_media's.
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
