"""LLM transports shared by provider.py's llm backend and assessor.py's
judge backend (spec 3 section 2: "one Assessor implementation, three
transports (cli/api/batch) selected by config"; the llm Provider backend
uses the same cli/api pair). `anthropic` is an optional dependency
(pyproject.toml's `llm` extra), imported lazily inside the method that
needs it. The api and batch transports send `thinking` on every request
(spec 3 §4).

Costs are in different currencies (spec 3 section 2): cli spends
subscription token quota, api/batch spend cash. A transport returns a
`Completion` (text + token usage); the backend prices it.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)


def strip_fences(text: str) -> str:
    """A markdown code fence around `text`, ```` ``` ```` or ```` ```json ````,
    removed -- the drafter's and the judge's parsers both accept an answer
    a model wraps this way before they call json.loads.
    """
    return re.sub(r"^```[a-z]*\n|\n```$", "", text.strip())


class TransportError(RuntimeError):
    """A transport failed to produce a completion -- the caller's ask()
    must NOT cache this (spec 3 section 7: transport errors are not
    answers).
    """


class FetchRefused(TransportError):
    """A fetcher refused a url: `reason` is the tool's typed kind (wire,
    http, content-type, too-large, format, io); `served` is whether a
    server answered (every kind but wire). `body` is the response body a
    content-type refusal carries (empty for every other reason) -- e.g.
    Forvo's own daily-limit statement served at an expired mp3 url (spec
    3 section 6a); attempts.py inspects it to tell that apart from any
    other content-type refusal."""

    def __init__(self, reason: str, detail: str, body: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.body = body

    @property
    def served(self) -> bool:
        return self.reason != "wire"


class SynthesisRefused(TransportError):
    """A synthesis service refused this text or voice (a 4xx other than
    429)."""


class QuotaExhausted(TransportError):
    """A source's own answer states its allowance for the day is spent
    (spec 3 section 6a's Quota state: Forvo's 400 body
    `["Limit/day reached."]`), recognized by the backend and raised
    typed -- never matched downstream by text. `source` names the
    backend that answered, carried for the log message."""

    def __init__(self, source: str):
        super().__init__(f"{source} quota exhausted")
        self.source = source


def _import_anthropic():
    try:
        import anthropic
    except ImportError as e:
        raise TransportError(
            "the anthropic package is not installed; install the 'llm' "
            "extra (pip install thai-deck-eval[llm]) to use the api/batch "
            "transports") from e
    return anthropic


@dataclass(frozen=True)
class Completion:
    """One transport answer: the text plus the token usage the wire reported
    (0 where the wire reports none, e.g. cli)."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


_MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}


def image_media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lstrip(".").lower(), "application/octet-stream")


def image_block(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image",
            "source": {"type": "base64", "media_type": image_media_type(path), "data": data}}


def _content(prompt: str, attachments: Sequence[Path]) -> list[dict] | str:
    if not attachments:
        return prompt
    return [image_block(Path(p)) for p in attachments] + [{"type": "text", "text": prompt}]


def _completion_of(message) -> Completion:
    text = ""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break
    usage = getattr(message, "usage", None)
    return Completion(text=text,
                      input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                      output_tokens=int(getattr(usage, "output_tokens", 0) or 0))


@dataclass
class ClaudeCliTransport:
    """`claude -p <prompt>` via subprocess. No dollar cost -- spends
    subscription token quota (tracked by the caller's Budget, not here).
    Attachments are linked (or copied) into a fresh temp dir passed as
    `--add-dir` with `--allowedTools Read`, named in the prompt, and the
    temp dir is removed afterwards.
    """
    binary: str = "claude"
    runner: Callable[..., Any] = field(default=subprocess.run)

    def complete(self, prompt: str, attachments: Sequence[Path] = ()) -> Completion:
        scope_dir = None
        cmd = [self.binary, "-p"]
        if attachments:
            scope_dir = tempfile.mkdtemp(prefix="thai-syllabus-judge-")
            names = []
            for i, src in enumerate(attachments):
                dst = Path(scope_dir) / f"{i}-{Path(src).name}"
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
                names.append(str(dst))
            prompt = prompt + "\nAttached files (read each with the Read tool): " + ", ".join(names)
        cmd.append(prompt)
        if scope_dir:
            cmd += ["--allowedTools", "Read", "--add-dir", scope_dir]
        try:
            proc = self.runner(cmd, capture_output=True, text=True)
        except OSError as e:
            raise TransportError(f"cannot run `{self.binary} -p`: {e}") from e
        finally:
            if scope_dir:
                shutil.rmtree(scope_dir, ignore_errors=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise TransportError(f"`{self.binary} -p` failed: {detail}")
        text = (proc.stdout or "").strip()
        if not text:
            raise TransportError(f"`{self.binary} -p` returned no output")
        return Completion(text=text)


@dataclass
class ClaudeApiTransport:
    """Single-message api transport. Guarded anthropic import; a
    `client_factory` may be injected (tests supply a fake client without
    anthropic installed at all).
    """
    api_key: str
    model: str
    max_tokens: int = 4096
    thinking: str = "disabled"
    client_factory: Callable[[], Any] | None = None

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        anthropic = _import_anthropic()
        return anthropic.Anthropic(api_key=self.api_key)

    def complete(self, prompt: str, attachments: Sequence[Path] = ()) -> Completion:
        try:
            client = self._client()
            response = client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                thinking={"type": self.thinking},
                messages=[{"role": "user", "content": _content(prompt, attachments)}])
        except Exception as e:  # noqa: BLE001 -- any SDK exception is a transport error
            raise TransportError(f"api transport failed: {e}") from e
        completion = _completion_of(response)
        if not completion.text:
            raise TransportError(
                "api transport returned no text block "
                f"(stop_reason={getattr(response, 'stop_reason', None)}, "
                f"output_tokens={completion.output_tokens})")
        return completion


@dataclass
class ClaudeBatchTransport:
    """Message Batches transport (anthropic SDK: client.messages.batches.*
    -- create/retrieve/results). Submission and polling are separate calls
    so callers can persist the batch id between them (spec 3: "Batch
    resume state is a cache row ... not a sidecar file").

    `api_key`, when set, is passed to the SDK client; empty, the SDK
    falls back to its own default resolution.
    """
    model: str
    api_key: str = ""
    max_tokens: int = 4096
    thinking: str = "disabled"
    client_factory: Callable[[], Any] | None = None

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        anthropic = _import_anthropic()
        if self.api_key:
            return anthropic.Anthropic(api_key=self.api_key)
        return anthropic.Anthropic()

    def submit(self, requests: Mapping[str, tuple[str, Sequence[Path]]]) -> str:
        """requests: custom_id -> (prompt, attachments). Returns the batch id."""
        try:
            client = self._client()
            batch = client.messages.batches.create(requests=[
                {"custom_id": custom_id,
                 "params": {"model": self.model, "max_tokens": self.max_tokens,
                           "thinking": {"type": self.thinking},
                           "messages": [{"role": "user",
                                         "content": _content(prompt, attachments)}]}}
                for custom_id, (prompt, attachments) in requests.items()])
        except Exception as e:  # noqa: BLE001
            raise TransportError(f"batch submit failed: {e}") from e
        return batch.id

    def status(self, batch_id: str) -> str:
        """"in_progress" | "canceling" | "ended" (anthropic's
        processing_status values)."""
        try:
            client = self._client()
            batch = client.messages.batches.retrieve(batch_id)
        except Exception as e:  # noqa: BLE001
            raise TransportError(f"batch status check failed: {e}") from e
        return batch.processing_status

    def results(self, batch_id: str) -> dict[str, Completion | None]:
        """custom_id -> Completion, or None for a non-succeeded result
        (errored/canceled/expired) -- callers decide how to treat those
        (typically: leave the subject queued, do not cache a miss).
        """
        try:
            client = self._client()
            out: dict[str, Completion | None] = {}
            for result in client.messages.batches.results(batch_id):
                completion = None
                if result.result.type == "succeeded":
                    candidate = _completion_of(result.result.message)
                    completion = candidate if candidate.text else None
                    if completion is None:
                        _log.warning("batch %s: %s succeeded with no text block (stop_reason=%s)",
                                     batch_id, result.custom_id,
                                     getattr(result.result.message, "stop_reason", None))
                out[result.custom_id] = completion
            return out
        except Exception as e:  # noqa: BLE001
            raise TransportError(f"batch results failed: {e}") from e
