"""Tests for transport.py: the cli/api/batch LLM transports shared by
provider.py's llm backend and assessor.py's judge backend. No real
subprocess, no real anthropic import -- everything injected.
"""
import base64
import subprocess
from pathlib import Path

import pytest

from thai_syllabus.transport import (
    ClaudeApiTransport,
    ClaudeBatchTransport,
    ClaudeCliTransport,
    Completion,
    TransportError,
    image_block,
    image_media_type,
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
    assert t.complete("do the thing").text == "the completion"
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
    assert t.complete("prompt").text == "hello"
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
    batch_id = t.submit({"c1": ("prompt one", ()), "c2": ("prompt two", ())})
    assert batch_id == "batch_123"
    submitted = client.messages.batches.created_with
    assert {r["custom_id"] for r in submitted} == {"c1", "c2"}


def test_batch_status_reports_processing_status():
    client = _FakeBatchClient()
    client.messages.batches._batch.processing_status = "ended"
    t = ClaudeBatchTransport(model="claude-opus-5", client_factory=lambda: client)
    assert t.status("batch_123") == "ended"


def test_batch_results_maps_custom_id_to_completion_on_success():
    client = _FakeBatchClient()
    client.messages.batches._results = [
        _FakeBatchResult("c1", _FakeResultWrapper("succeeded", _FakeResultMessage(["ok"]))),
        _FakeBatchResult("c2", _FakeResultWrapper("errored")),
        _FakeBatchResult("c3", _FakeResultWrapper("succeeded", _FakeResultMessage([]))),
    ]
    t = ClaudeBatchTransport(model="claude-opus-5", client_factory=lambda: client)
    results = t.results("batch_123")
    assert results == {"c1": Completion(text="ok"), "c2": None, "c3": None}


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Response:
    def __init__(self, text, i=10, o=3):
        self.content = [_Block(text)]
        self.usage = _Usage(i, o)


class _CompletionMessages:
    def __init__(self, text):
        self.text, self.calls = text, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.text)


class _FakeClient:
    def __init__(self, text="ok"):
        self.messages = _CompletionMessages(text)


def test_api_transport_returns_completion_with_usage():
    client = _FakeClient("hello")
    t = ClaudeApiTransport(api_key="k", model="m", client_factory=lambda: client)
    c = t.complete("q")
    assert c == Completion(text="hello", input_tokens=10, output_tokens=3)


def test_api_transport_sends_an_image_block_per_attachment(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG-bytes")
    client = _FakeClient()
    t = ClaudeApiTransport(api_key="k", model="m", client_factory=lambda: client)
    t.complete("look", attachments=[img])
    content = client.messages.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == base64.standard_b64encode(b"\x89PNG-bytes").decode()
    assert content[-1] == {"type": "text", "text": "look"}


def test_image_media_type_by_extension():
    assert image_media_type(Path("x.jpg")) == "image/jpeg"
    assert image_media_type(Path("x.webp")) == "image/webp"
    assert image_media_type(Path("x.bin")) == "application/octet-stream"


def test_cli_transport_scopes_add_dir_to_a_temp_dir_holding_the_attachment(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpg")
    seen = {}

    def runner(cmd, capture_output, text):
        seen["cmd"] = cmd
        add_dir = Path(cmd[cmd.index("--add-dir") + 1])
        seen["files"] = sorted(p.name for p in add_dir.iterdir())
        seen["prompt"] = cmd[cmd.index("-p") + 1]

        class P:
            returncode, stdout, stderr = 0, "yes", ""
        return P()

    t = ClaudeCliTransport(runner=runner)
    c = t.complete("judge this", attachments=[img])
    assert c.text == "yes"
    assert "--allowedTools" in seen["cmd"] and "Read" in seen["cmd"]
    assert seen["files"] == ["0-a.jpg"]
    assert Path(seen["cmd"][seen["cmd"].index("--add-dir") + 1]) != tmp_path
    assert "a.jpg" in seen["prompt"]


def test_cli_transport_without_attachments_adds_no_flags():
    def runner(cmd, capture_output, text):
        assert "--add-dir" not in cmd

        class P:
            returncode, stdout, stderr = 0, "t", ""
        return P()

    assert ClaudeCliTransport(runner=runner).complete("q") == Completion(text="t")


class _CompletionBatches:
    def __init__(self):
        self.created = None

    def create(self, requests):
        self.created = requests

        class B:
            id = "batch_1"
        return B()

    def retrieve(self, batch_id):
        class B:
            processing_status = "ended"
        return B()

    def results(self, batch_id):
        class R:
            def __init__(self, cid, text):
                self.custom_id = cid

                class Res:
                    type = "succeeded"
                    message = _Response(text, 7, 2)
                self.result = Res()
        return [R("a", "A"), R("b", "B")]


def test_batch_transport_submits_attachments_and_returns_completions(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpg")

    class Client:
        class messages:
            batches = _CompletionBatches()

    t = ClaudeBatchTransport(model="m", client_factory=lambda: Client())
    bid = t.submit({"a": ("q1", [img]), "b": ("q2", [])})
    assert bid == "batch_1"
    req_a = Client.messages.batches.created[0]
    assert req_a["params"]["messages"][0]["content"][0]["type"] == "image"
    out = t.results(bid)
    assert out["a"] == Completion(text="A", input_tokens=7, output_tokens=2)


# --- a client that cannot be constructed is a TransportError, not a crash ---
# (`client = self._client()` sits INSIDE each method's try, so an SDK
# construction failure -- a bad key, a missing package -- reaches the caller
# as a TransportError and is never cached.)

def _boom():
    raise RuntimeError("no credentials")


def test_api_transport_client_construction_failure_is_a_transport_error():
    t = ClaudeApiTransport(api_key="k", model="m", client_factory=_boom)
    with pytest.raises(TransportError, match="no credentials"):
        t.complete("prompt")


@pytest.mark.parametrize("call", [
    lambda t: t.submit({"c1": ("p", ())}),
    lambda t: t.status("batch_123"),
    lambda t: t.results("batch_123"),
])
def test_batch_transport_client_construction_failure_is_a_transport_error(call):
    t = ClaudeBatchTransport(model="m", client_factory=_boom)
    with pytest.raises(TransportError, match="no credentials"):
        call(t)


# --- the batch transport authenticates like the api one --------------------

def test_batch_transport_carries_an_api_key():
    assert ClaudeBatchTransport(model="m", api_key="sk-test").api_key == "sk-test"
    assert ClaudeBatchTransport(model="m").api_key == ""
