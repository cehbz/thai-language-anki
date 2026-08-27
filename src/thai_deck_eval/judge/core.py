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

    def judge(self, req: JudgeRequest) -> list[Verdict]:
        key = self._key(req)
        row = self._db.execute("SELECT payload FROM verdicts WHERE key=?",
                               (key,)).fetchone()
        if row:
            return Verdicts.model_validate_json(row[0]).verdicts
        out = self.inner.judge(req)
        self.calls += 1
        self._db.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?)",
                         (key, Verdicts(verdicts=out).model_dump_json()))
        self._db.commit()
        return out
