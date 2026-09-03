"""LLM transports shared by provider.py's llm backend and assessor.py's
judge backend (spec 3 section 2: "one Assessor implementation, three
transports (cli/api/batch) selected by config"; the llm Provider backend
uses the same cli/api pair). `anthropic` is an optional dependency (see
pyproject.toml's `llm` extra) -- every class here imports it lazily,
inside the method that needs it, so importing this module never requires
it and the default test suite (no live network, no anthropic import)
stays clean.

Costs are in different currencies (spec 3 section 2): cli spends
subscription token quota (sunk monthly, ~35K harness tokens/call, no
dollar cost recorded here), api/batch spend cash. Callers (provider.py,
assessor.py) attach the currency-appropriate cost; these transports only
return text.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable


class TransportError(RuntimeError):
    """A transport failed to produce a completion -- the caller's ask()
    must NOT cache this (spec 3 section 7: transport errors are not
    answers).
    """


def _import_anthropic():
    try:
        import anthropic
    except ImportError as e:
        raise TransportError(
            "the anthropic package is not installed; install the 'llm' "
            "extra (pip install thai-deck-eval[llm]) to use the api/batch "
            "transports") from e
    return anthropic


@dataclass
class ClaudeCliTransport:
    """`claude -p <prompt>` via subprocess. No dollar cost -- spends
    subscription token quota (tracked by the caller's Budget, not here).
    """
    binary: str = "claude"
    runner: Callable[..., Any] = field(default=subprocess.run)

    def complete(self, prompt: str) -> str:
        try:
            proc = self.runner([self.binary, "-p", prompt],
                               capture_output=True, text=True)
        except OSError as e:
            raise TransportError(f"cannot run `{self.binary} -p`: {e}") from e
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise TransportError(f"`{self.binary} -p` failed: {detail}")
        text = (proc.stdout or "").strip()
        if not text:
            raise TransportError(f"`{self.binary} -p` returned no output")
        return text


@dataclass
class ClaudeApiTransport:
    """Single-message api transport. Guarded anthropic import; a
    `client_factory` may be injected (tests supply a fake client without
    anthropic installed at all).
    """
    api_key: str
    model: str
    max_tokens: int = 4096
    client_factory: Callable[[], Any] | None = None

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        anthropic = _import_anthropic()
        return anthropic.Anthropic(api_key=self.api_key)

    def complete(self, prompt: str) -> str:
        client = self._client()
        try:
            response = client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001 -- any SDK exception is a transport error
            raise TransportError(f"api transport failed: {e}") from e
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise TransportError("api transport returned no text block")


@dataclass
class ClaudeBatchTransport:
    """Message Batches transport (anthropic SDK: client.messages.batches.*
    -- create/retrieve/results). Submission and polling are separate calls
    so callers can persist the batch id between them (spec 3: "Batch
    resume state is a cache row ... not a sidecar file").
    """
    model: str
    max_tokens: int = 4096
    client_factory: Callable[[], Any] | None = None

    def _client(self) -> Any:
        if self.client_factory is not None:
            return self.client_factory()
        anthropic = _import_anthropic()
        return anthropic.Anthropic()

    def submit(self, requests: dict[str, str]) -> str:
        """requests: custom_id -> prompt. Returns the batch id."""
        client = self._client()
        try:
            batch = client.messages.batches.create(requests=[
                {"custom_id": custom_id,
                 "params": {"model": self.model, "max_tokens": self.max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}}
                for custom_id, prompt in requests.items()])
        except Exception as e:  # noqa: BLE001
            raise TransportError(f"batch submit failed: {e}") from e
        return batch.id

    def status(self, batch_id: str) -> str:
        """"in_progress" | "ended" (anthropic's processing_status values)."""
        client = self._client()
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except Exception as e:  # noqa: BLE001
            raise TransportError(f"batch status check failed: {e}") from e
        return batch.processing_status

    def results(self, batch_id: str) -> dict[str, str | None]:
        """custom_id -> completion text, or None for a non-succeeded result
        (errored/canceled/expired) -- callers decide how to treat those
        (typically: leave the subject queued, do not cache a miss).
        """
        client = self._client()
        try:
            out: dict[str, str | None] = {}
            for result in client.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    text = None
                    for block in result.result.message.content:
                        if getattr(block, "type", None) == "text":
                            text = block.text
                            break
                    out[result.custom_id] = text
                else:
                    out[result.custom_id] = None
            return out
        except Exception as e:  # noqa: BLE001
            raise TransportError(f"batch results failed: {e}") from e
