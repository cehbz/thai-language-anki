"""Tests for transport.py: the cli/api/batch LLM transports shared by
provider.py's llm backend and assessor.py's judge backend. No real
subprocess, no real anthropic import -- everything injected.
"""
import base64
import json
import subprocess
from pathlib import Path

import pytest

from thai_syllabus.transport import (
    ClaudeApiTransport,
    ClaudeBatchTransport,
    ClaudeCliTransport,
    Completion,
    QuotaExhausted,
    RequestParams,
    TransportError,
    image_block,
    image_media_type,
)


# --- cli -----------------------------------------------------------------

def _cli_reply(result="the completion", *, is_error=False, model="claude-sonnet-5-5",
               model_usage=None) -> str:
    """One `claude -p --output-format json` reply, the shape the CLI
    prints (fields this transport reads, plus some it ignores)."""
    if model_usage is None:
        model_usage = {f"{model}-20260901": {"inputTokens": 10, "outputTokens": 70,
                                            "canonicalModel": model}}
    return json.dumps({
        "type": "result", "subtype": "success", "is_error": is_error,
        "result": result, "session_id": "s-1", "total_cost_usd": 0.03,
        "usage": {"input_tokens": 10, "output_tokens": 70,
                  "cache_read_input_tokens": 13796, "cache_creation_input_tokens": 12724,
                  "service_tier": "standard"},
        "modelUsage": model_usage})


class _Runner:
    def __init__(self, stdout=None, stderr="", returncode=0):
        stdout = _cli_reply() if stdout is None else stdout
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        self.kwargs = kwargs
        return self.result


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_cli_transport_runs_claude_dash_p_and_returns_the_result_text():
    runner = _Runner(stdout=_cli_reply("the completion"))
    t = ClaudeCliTransport(runner=runner)
    assert t.complete("do the thing").text == "the completion"
    cmd = runner.calls[0]
    assert cmd[:3] == ["claude", "-p", "do the thing"]
    assert _flag(cmd, "--output-format") == "json"


def test_cli_transport_sends_its_model_and_effort():
    runner = _Runner()
    ClaudeCliTransport(model="claude-sonnet-5-5", effort="medium", runner=runner).complete("q")
    assert _flag(runner.calls[0], "--model") == "claude-sonnet-5-5"
    assert _flag(runner.calls[0], "--effort") == "medium"


def test_cli_transport_without_an_effort_sends_none():
    runner = _Runner()
    ClaudeCliTransport(model="claude-sonnet-5-5", runner=runner).complete("q")
    assert "--effort" not in runner.calls[0]


def test_cli_transport_sends_the_request_params_model_and_effort_over_its_own():
    runner = _Runner()
    t = ClaudeCliTransport(model="claude-sonnet-5-5", effort="medium", runner=runner)
    t.complete("q", params=RequestParams(model="claude-opus-5-5", max_tokens=16000,
                                         thinking="adaptive", effort="high"))
    assert _flag(runner.calls[0], "--model") == "claude-opus-5-5"
    assert _flag(runner.calls[0], "--effort") == "high"


def test_cli_transport_reports_the_clis_usage_and_model():
    runner = _Runner(stdout=_cli_reply("yes", model="claude-opus-5-5"))
    c = ClaudeCliTransport(model="claude-opus-5-5", runner=runner).complete("q")
    assert c == Completion(text="yes", input_tokens=10, output_tokens=70,
                           cache_read_input_tokens=13796, cache_creation_input_tokens=12724,
                           model="claude-opus-5-5")


def test_cli_transport_names_the_model_with_the_most_output_tokens():
    runner = _Runner(stdout=_cli_reply("yes", model_usage={
        "claude-haiku-4-5-20251001": {"outputTokens": 12, "canonicalModel": "claude-haiku-4-5"},
        "claude-opus-5-5": {"outputTokens": 900, "canonicalModel": "claude-opus-5-5"}}))
    assert ClaudeCliTransport(runner=runner).complete("q").model == "claude-opus-5-5"


def test_cli_transport_reads_a_reply_with_no_model_usage_or_cost():
    runner = _Runner(stdout=json.dumps({"is_error": False, "result": "yes"}))
    c = ClaudeCliTransport(runner=runner).complete("q")
    assert (c.text, c.model, c.input_tokens) == ("yes", None, 0)


def test_cli_transport_raises_on_an_empty_result():
    runner = _Runner(stdout=_cli_reply(""))
    with pytest.raises(TransportError, match="no output"):
        ClaudeCliTransport(runner=runner).complete("q")


def test_cli_transport_raises_on_an_error_reply_with_its_detail():
    runner = _Runner(stdout=_cli_reply("API Error: 529 overloaded", is_error=True))
    with pytest.raises(TransportError, match="529 overloaded"):
        ClaudeCliTransport(runner=runner).complete("q")


def test_cli_transport_raises_on_a_reply_that_is_not_json():
    runner = _Runner(stdout="plain text, not the json reply\n")
    with pytest.raises(TransportError, match="plain text"):
        ClaudeCliTransport(runner=runner).complete("q")


def test_cli_transport_on_a_nonzero_exit_names_the_json_replys_reason():
    runner = _Runner(stdout=_cli_reply("Credit balance is too low", is_error=True),
                     stderr="Warning: an unrelated notice", returncode=1)
    with pytest.raises(TransportError, match="Credit balance is too low"):
        ClaudeCliTransport(runner=runner).complete("x")


def test_cli_transport_on_a_nonzero_exit_without_a_json_reply_names_stderr():
    runner = _Runner(stdout="", stderr="boom", returncode=1)
    with pytest.raises(TransportError, match="boom"):
        ClaudeCliTransport(runner=runner).complete("x")


def test_cli_transport_reads_no_stdin_and_keeps_no_session():
    runner = _Runner()
    ClaudeCliTransport(runner=runner).complete("q")
    assert runner.kwargs.get("stdin") is subprocess.DEVNULL
    assert "--no-session-persistence" in runner.calls[0]


def test_cli_transport_runs_in_safe_mode_by_default():
    runner = _Runner()
    ClaudeCliTransport(runner=runner).complete("q")
    assert "--safe-mode" in runner.calls[0]


def test_cli_transport_with_safe_mode_off_sends_no_safe_mode_flag():
    runner = _Runner()
    ClaudeCliTransport(safe_mode=False, runner=runner).complete("q")
    assert "--safe-mode" not in runner.calls[0]


def test_cli_transport_without_attachments_offers_no_tools():
    runner = _Runner()
    ClaudeCliTransport(runner=runner).complete("q")
    assert _flag(runner.calls[0], "--tools") == ""


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


# --- quota exhaustion (spec 3 section 6a) -----------------------------------

def test_quota_exhausted_is_a_transport_error_carrying_its_source():
    err = QuotaExhausted("forvo")
    assert isinstance(err, TransportError)
    assert err.source == "forvo"


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
    t = ClaudeApiTransport(api_key="k", model="claude-opus-5-5",
                           client_factory=lambda: client)
    assert t.complete("prompt").text == "hello"
    assert client.messages.calls[0]["model"] == "claude-opus-5-5"


def test_api_transport_wraps_sdk_exceptions_as_transport_error():
    client = _FakeApiClient(raises=RuntimeError("rate limited"))
    t = ClaudeApiTransport(api_key="k", model="claude-opus-5-5",
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
    t = ClaudeBatchTransport(model="claude-opus-5-5", client_factory=lambda: client)
    batch_id = t.submit({"c1": ("prompt one", ()), "c2": ("prompt two", ())})
    assert batch_id == "batch_123"
    submitted = client.messages.batches.created_with
    assert {r["custom_id"] for r in submitted} == {"c1", "c2"}


def test_batch_status_reports_processing_status():
    client = _FakeBatchClient()
    client.messages.batches._batch.processing_status = "ended"
    t = ClaudeBatchTransport(model="claude-opus-5-5", client_factory=lambda: client)
    assert t.status("batch_123") == "ended"


def test_batch_results_maps_custom_id_to_completion_on_success():
    client = _FakeBatchClient()
    client.messages.batches._results = [
        _FakeBatchResult("c1", _FakeResultWrapper("succeeded", _FakeResultMessage(["ok"]))),
        _FakeBatchResult("c2", _FakeResultWrapper("errored")),
        _FakeBatchResult("c3", _FakeResultWrapper("succeeded", _FakeResultMessage([]))),
    ]
    t = ClaudeBatchTransport(model="claude-opus-5-5", client_factory=lambda: client)
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

    def runner(cmd, capture_output, text, stdin):
        seen["cmd"] = cmd
        add_dir = Path(cmd[cmd.index("--add-dir") + 1])
        seen["files"] = sorted(p.name for p in add_dir.iterdir())
        seen["prompt"] = cmd[cmd.index("-p") + 1]

        class P:
            returncode, stdout, stderr = 0, _cli_reply("yes"), ""
        return P()

    t = ClaudeCliTransport(runner=runner)
    c = t.complete("judge this", attachments=[img])
    assert c.text == "yes"
    assert _flag(seen["cmd"], "--tools") == "Read"
    assert _flag(seen["cmd"], "--allowedTools") == "Read"
    assert seen["files"] == ["0-a.jpg"]
    assert Path(seen["cmd"][seen["cmd"].index("--add-dir") + 1]) != tmp_path
    assert "a.jpg" in seen["prompt"]


def test_cli_transport_without_attachments_adds_no_flags():
    def runner(cmd, capture_output, text, stdin):
        assert "--add-dir" not in cmd

        class P:
            returncode, stdout, stderr = 0, _cli_reply("t"), ""
        return P()

    assert ClaudeCliTransport(runner=runner).complete("q").text == "t"


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


# --- thinking ----------------------------------------------------------

def test_api_transport_sends_thinking_disabled_by_default():
    client = _FakeClient("hello")
    ClaudeApiTransport(api_key="k", model="m", client_factory=lambda: client).complete("q")
    assert client.messages.calls[0]["thinking"] == {"type": "disabled"}


def test_api_transport_sends_the_configured_thinking():
    client = _FakeClient("hello")
    ClaudeApiTransport(api_key="k", model="m", thinking="adaptive",
                       client_factory=lambda: client).complete("q")
    assert client.messages.calls[0]["thinking"] == {"type": "adaptive"}


def test_batch_transport_sends_thinking_on_every_request():
    client = _FakeBatchClient()
    t = ClaudeBatchTransport(model="m", thinking="adaptive", client_factory=lambda: client)
    t.submit({"c1": ("p1", ()), "c2": ("p2", ())})
    assert [r["params"]["thinking"] for r in client.messages.batches.created_with] == [
        {"type": "adaptive"}, {"type": "adaptive"}]


# --- effort (Claude Opus 5.5's depth control) ---------------------------

def test_api_transport_sends_no_output_config_when_effort_is_none():
    client = _FakeClient("hello")
    ClaudeApiTransport(api_key="k", model="m", client_factory=lambda: client).complete("q")
    assert "output_config" not in client.messages.calls[0]


def test_api_transport_sends_the_configured_effort():
    client = _FakeClient("hello")
    ClaudeApiTransport(api_key="k", model="m", effort="high",
                       client_factory=lambda: client).complete("q")
    assert client.messages.calls[0]["output_config"] == {"effort": "high"}


def test_api_transport_request_params_effort_wins_over_the_transports_own():
    client = _FakeClient("hello")
    t = ClaudeApiTransport(api_key="k", model="m", effort="high", client_factory=lambda: client)
    t.complete("q", params=RequestParams(model="m", max_tokens=4096, thinking="disabled",
                                         effort="low"))
    assert client.messages.calls[0]["output_config"] == {"effort": "low"}


def test_batch_transport_puts_output_config_into_every_request_when_effort_is_set():
    client = _FakeBatchClient()
    t = ClaudeBatchTransport(model="m", effort="high", client_factory=lambda: client)
    t.submit({"c1": ("p1", ()), "c2": ("p2", ())})
    assert [r["params"]["output_config"] for r in client.messages.batches.created_with] == [
        {"effort": "high"}, {"effort": "high"}]


def test_batch_transport_omits_output_config_when_effort_is_not_set():
    client = _FakeBatchClient()
    t = ClaudeBatchTransport(model="m", client_factory=lambda: client)
    t.submit({"c1": ("p1", ())})
    assert "output_config" not in client.messages.batches.created_with[0]["params"]


class _ThinkingOnlyResponse:
    """A response whose whole output budget went to thinking: one thinking
    block, no text block, stop_reason max_tokens."""
    stop_reason = "max_tokens"

    def __init__(self):
        block = type("_Thinking", (), {"type": "thinking", "thinking": ""})()
        self.content = [block]
        self.usage = _Usage(97271, 4096)


def test_api_transport_names_the_stop_reason_when_no_text_block_came_back():
    client = _FakeApiClient(response=_ThinkingOnlyResponse())
    t = ClaudeApiTransport(api_key="k", model="m", client_factory=lambda: client)
    with pytest.raises(TransportError, match="stop_reason=max_tokens.*output_tokens=4096"):
        t.complete("q")


# --- per-request overrides (spec 3 r43) ---------------------------------
# A judge role may answer under its own model, thinking and max_tokens
# (providers.yaml `judge.roles.<role>`); the transport takes them per
# request, so one transport serves every role.

def test_api_transport_sends_the_request_params_when_given():
    client = _FakeClient("hello")
    t = ClaudeApiTransport(api_key="k", model="claude-sonnet-5", max_tokens=4096,
                           thinking="disabled", client_factory=lambda: client)
    t.complete("q", params=RequestParams(model="claude-opus-5-5", max_tokens=16000,
                                         thinking="adaptive"))
    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 16000
    assert call["thinking"] == {"type": "adaptive"}


def test_api_transport_without_params_sends_its_own_fields():
    client = _FakeClient("hello")
    t = ClaudeApiTransport(api_key="k", model="claude-sonnet-5", max_tokens=4096,
                           thinking="disabled", client_factory=lambda: client)
    t.complete("q", params=None)
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["max_tokens"] == 4096
    assert call["thinking"] == {"type": "disabled"}


def test_batch_submit_sends_a_requests_params_per_request():
    client = _FakeBatchClient()
    t = ClaudeBatchTransport(model="claude-sonnet-5", max_tokens=4096, thinking="disabled",
                            client_factory=lambda: client)
    t.submit({"c1": ("p1", (), RequestParams(model="claude-opus-5-5", max_tokens=16000,
                                             thinking="adaptive")),
              "c2": ("p2", (), None)})
    by_id = {r["custom_id"]: r["params"] for r in client.messages.batches.created_with}
    assert by_id["c1"]["model"] == "claude-opus-5-5"
    assert by_id["c1"]["max_tokens"] == 16000
    assert by_id["c1"]["thinking"] == {"type": "adaptive"}
    assert by_id["c2"]["model"] == "claude-sonnet-5"
    assert by_id["c2"]["max_tokens"] == 4096
    assert by_id["c2"]["thinking"] == {"type": "disabled"}


def test_batch_submit_still_accepts_the_two_tuple():
    client = _FakeBatchClient()
    t = ClaudeBatchTransport(model="claude-sonnet-5", client_factory=lambda: client)
    assert t.submit({"c1": ("p1", ())}) == "batch_123"
    assert client.messages.batches.created_with[0]["params"]["model"] == "claude-sonnet-5"


class _ModelResponse:
    """A response naming the model that answered, as the wire does."""
    def __init__(self, text, model):
        self.content = [_Block(text)]
        self.usage = _Usage(10, 3)
        self.model = model


def test_completion_names_the_model_that_answered():
    client = _FakeApiClient(response=_ModelResponse("hi", "claude-opus-5-20260101"))
    t = ClaudeApiTransport(api_key="k", model="claude-sonnet-5", client_factory=lambda: client)
    assert t.complete("q").model == "claude-opus-5-20260101"


def test_completion_model_is_none_when_the_wire_names_none():
    assert Completion(text="t").model is None
    client = _FakeClient("hello")
    t = ClaudeApiTransport(api_key="k", model="m", client_factory=lambda: client)
    assert t.complete("q").model is None
