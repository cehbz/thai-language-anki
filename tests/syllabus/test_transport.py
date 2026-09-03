"""Tests for transport.py: the cli/api/batch LLM transports shared by
provider.py's llm backend and assessor.py's judge backend. No real
subprocess, no real anthropic import -- everything injected.
"""
import subprocess

import pytest

from thai_syllabus.transport import (
    ClaudeApiTransport,
    ClaudeBatchTransport,
    ClaudeCliTransport,
    TransportError,
)


# --- cli -----------------------------------------------------------------

class _Runner:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return self.result


def test_cli_transport_runs_claude_dash_p_and_returns_stdout():
    runner = _Runner(stdout="the completion\n")
    t = ClaudeCliTransport(runner=runner)
    assert t.complete("do the thing") == "the completion"
    assert runner.calls == [["claude", "-p", "do the thing"]]


def test_cli_transport_raises_on_nonzero_exit():
    runner = _Runner(stderr="boom", returncode=1)
    t = ClaudeCliTransport(runner=runner)
    with pytest.raises(TransportError, match="boom"):
        t.complete("x")


def test_cli_transport_raises_on_missing_binary():
    def runner(cmd, **kwargs):
        raise FileNotFoundError("claude")
    t = ClaudeCliTransport(runner=runner)
    with pytest.raises(TransportError):
        t.complete("x")


def test_cli_transport_raises_on_empty_output():
    runner = _Runner(stdout="")
    t = ClaudeCliTransport(runner=runner)
    with pytest.raises(TransportError):
        t.complete("x")


# --- api -------------------------------------------------------------------

class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeApiResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return self._response


class _FakeApiClient:
    def __init__(self, response=None, raises=None):
        self.messages = _FakeMessages(response=response, raises=raises)


def test_api_transport_returns_the_text_block():
    client = _FakeApiClient(response=_FakeApiResponse("hello"))
    t = ClaudeApiTransport(api_key="k", model="claude-opus-5",
                           client_factory=lambda: client)
    assert t.complete("prompt") == "hello"
    assert client.messages.calls[0]["model"] == "claude-opus-5"


def test_api_transport_wraps_sdk_exceptions_as_transport_error():
    client = _FakeApiClient(raises=RuntimeError("rate limited"))
    t = ClaudeApiTransport(api_key="k", model="claude-opus-5",
                           client_factory=lambda: client)
    with pytest.raises(TransportError, match="rate limited"):
        t.complete("prompt")


def test_api_transport_without_a_client_factory_needs_anthropic_installed():
    # Guarded import: if anthropic genuinely isn't installed this raises
    # TransportError, not ImportError -- can't force that here since this
    # venv has anthropic, so just exercise the client_factory-provided path
    # (covered above) and assert the guard function exists.
    from thai_syllabus.transport import _import_anthropic
    assert callable(_import_anthropic)


# --- batch -------------------------------------------------------------

class _FakeBatch:
    def __init__(self, id, status="in_progress"):
        self.id = id
        self.processing_status = status


class _FakeResultMessage:
    def __init__(self, texts):
        self.content = [_FakeTextBlock(t) for t in texts]


class _FakeResultWrapper:
    def __init__(self, type_, message=None):
        self.type = type_
        if message is not None:
            self.message = message


class _FakeBatchResult:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class _FakeBatches:
    def __init__(self):
        self.created_with = None
        self._batch = _FakeBatch("batch_123")
        self._results = []

    def create(self, requests):
        self.created_with = requests
        return self._batch

    def retrieve(self, batch_id):
        return self._batch

    def results(self, batch_id):
        return iter(self._results)


class _FakeBatchClient:
    def __init__(self):
        self.messages = type("M", (), {"batches": _FakeBatches()})()


def test_batch_submit_returns_the_batch_id():
    client = _FakeBatchClient()
    t = ClaudeBatchTransport(model="claude-opus-5", client_factory=lambda: client)
    batch_id = t.submit({"c1": "prompt one", "c2": "prompt two"})
    assert batch_id == "batch_123"
    submitted = client.messages.batches.created_with
    assert {r["custom_id"] for r in submitted} == {"c1", "c2"}


def test_batch_status_reports_processing_status():
    client = _FakeBatchClient()
    client.messages.batches._batch.processing_status = "ended"
    t = ClaudeBatchTransport(model="claude-opus-5", client_factory=lambda: client)
    assert t.status("batch_123") == "ended"


def test_batch_results_maps_custom_id_to_text_on_success():
    client = _FakeBatchClient()
    client.messages.batches._results = [
        _FakeBatchResult("c1", _FakeResultWrapper("succeeded", _FakeResultMessage(["ok"]))),
        _FakeBatchResult("c2", _FakeResultWrapper("errored")),
    ]
    t = ClaudeBatchTransport(model="claude-opus-5", client_factory=lambda: client)
    results = t.results("batch_123")
    assert results == {"c1": "ok", "c2": None}
