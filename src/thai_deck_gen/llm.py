import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Protocol


class LlmError(Exception):
    pass


def cli_failure_detail(stdout: str, stderr: str, limit: int = 500) -> str:
    """Readable reason for a failed `claude -p`: in --output-format json mode
    the error text is the `result` field on stdout, not stderr."""
    try:
        result = json.loads(stdout).get("result")
        if result:
            return str(result)[:limit]
    except (json.JSONDecodeError, AttributeError):
        pass
    return (stderr or stdout)[:limit]


class Llm(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class CliBackend:
    def __init__(self, runner=subprocess.run, model: str | None = None):
        self.runner, self.model = runner, model

    def complete(self, prompt: str) -> str:
        cmd = ["claude", "-p", prompt, "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        try:
            r = self.runner(cmd, capture_output=True, text=True, timeout=600)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise LlmError(str(exc)) from exc
        if r.returncode != 0:
            raise LlmError(cli_failure_detail(r.stdout, r.stderr))
        try:
            return json.loads(r.stdout)["result"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise LlmError(f"unparseable claude output: {exc}") from exc


class ApiBackend:
    """Drafting through the Anthropic API rather than the `claude` CLI.

    The CLI path spends subscription quota and drags its whole harness
    prompt into every call; this one sends the prompt and nothing else.
    """

    def __init__(self, model: str, api_key: str | None = None, client=None):
        self.model = model
        if client is None:
            import anthropic
            client = (anthropic.Anthropic(api_key=api_key) if api_key
                      else anthropic.Anthropic())
        self.client = client

    def complete(self, prompt: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model, max_tokens=8192,
                messages=[{"role": "user", "content": prompt}])
        except Exception as exc:
            raise LlmError(f"api backend failed: {exc}") from exc
        return "".join(b.text for b in message.content
                       if getattr(b, "type", None) == "text")


class CachedLlm:
    def __init__(self, inner: Llm, db_path: Path, model: str):
        self.inner, self.model, self.calls = inner, model, 0
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS completions (key TEXT PRIMARY KEY, payload TEXT)"
        )

    def complete(self, producer: str, prompt_version: str, prompt: str) -> str:
        blob = json.dumps([producer, prompt_version, self.model, prompt],
                          ensure_ascii=False)
        key = hashlib.sha256(blob.encode()).hexdigest()
        row = self._db.execute("SELECT payload FROM completions WHERE key=?",
                               (key,)).fetchone()
        if row:
            return row[0]
        out = self.inner.complete(prompt)
        self.calls += 1
        self._db.execute("INSERT OR REPLACE INTO completions VALUES (?,?)", (key, out))
        self._db.commit()
        return out

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
