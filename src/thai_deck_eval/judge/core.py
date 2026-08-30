import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel

class Verdict(BaseModel):
    rule: str
    passed: bool
    confidence: float
    rationale: str

class Verdicts(BaseModel):
    verdicts: list[Verdict]

@dataclass
class JudgeRequest:
    note_id: str
    rules: list[str]
    prompt: str
    image_path: str | None = None

class Judge(Protocol):
    def judge(self, req: JudgeRequest) -> list[Verdict]: ...

    # Optional: backends that answer a whole deck in one submission (see
    # BatchJudge) implement judge_many; CachedJudge falls back to judge().


class FakeJudge:
    def __init__(self, verdicts: dict[str, list[Verdict]]):
        self._v = verdicts
    def judge(self, req: JudgeRequest) -> list[Verdict]:
        if req.note_id in self._v:
            return self._v[req.note_id]
        return [Verdict(rule=r, passed=True, confidence=1.0, rationale="")
                for r in req.rules]

class CachedJudge:
    def __init__(self, inner: Judge, db_path: Path, model: str, prompt_version: str):
        self.inner, self.model, self.prompt_version = inner, model, prompt_version
        self.calls = 0
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, payload TEXT)")

    def _key(self, req: JudgeRequest) -> str:
        image_sha = None
        if req.image_path and Path(req.image_path).is_file():
            image_sha = hashlib.sha256(Path(req.image_path).read_bytes()).hexdigest()
        blob = json.dumps([sorted(req.rules), self.prompt_version, self.model,
                           req.prompt, image_sha], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "CachedJudge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def judge_many(self, reqs: list[JudgeRequest]) -> dict[str, list[Verdict]]:
        """Verdicts by note id, cached cards served from sqlite and only the
        rest handed to the backend -- in one call when it supports batching."""
        out: dict[str, list[Verdict]] = {}
        pending: list[JudgeRequest] = []
        for req in reqs:
            cached = self._cached(self._key(req))
            if cached is None:
                pending.append(req)
            else:
                out[req.note_id] = cached
        if not pending:
            return out

        inner_many = getattr(self.inner, "judge_many", None)
        if inner_many is None:
            for req in pending:
                out[req.note_id] = self.judge(req)
            return out

        fresh = inner_many(pending)
        keys = {req.note_id: self._key(req) for req in pending}
        for note_id, verdicts in fresh.items():
            self.calls += 1
            self._store(keys[note_id], verdicts)
            out[note_id] = verdicts
        self._db.commit()
        return out

    def _cached(self, key: str) -> list[Verdict] | None:
        row = self._db.execute("SELECT payload FROM verdicts WHERE key=?",
                               (key,)).fetchone()
        return Verdicts.model_validate_json(row[0]).verdicts if row else None

    def _store(self, key: str, verdicts: list[Verdict]) -> None:
        self._db.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?)",
                         (key, Verdicts(verdicts=verdicts).model_dump_json()))

    def judge(self, req: JudgeRequest) -> list[Verdict]:
        key = self._key(req)
        cached = self._cached(key)
        if cached is not None:
            return cached
        out = self.inner.judge(req)
        self.calls += 1
        self._store(key, out)
        self._db.commit()
        return out
