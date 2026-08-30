import json
import pytest
from thai_deck_gen.llm import CachedLlm, CliBackend, LlmError
from tests.gen.fakes import FakeLlm


def test_cached_llm_creates_missing_parent_dir(tmp_path):
    fake = FakeLlm(["one"])
    with CachedLlm(fake, tmp_path / "work" / "c.sqlite", model="m") as llm:
        assert llm.complete("p", "v1", "hello") == "one"


def test_cached_llm_caches_by_content(tmp_path):
    fake = FakeLlm(["one", "two"])
    with CachedLlm(fake, tmp_path / "c.sqlite", model="m") as llm:
        assert llm.complete("p", "v1", "hello") == "one"
        assert llm.complete("p", "v1", "hello") == "one"
        assert llm.calls == 1
        assert llm.complete("p", "v2", "hello") == "two"
        assert llm.calls == 2


def _runner_ok(cmd, **kw):
    class R:
        returncode = 0
        stdout = json.dumps({"result": "answer"})
        stderr = ""

    return R()


def _runner_fail(cmd, **kw):
    class R:
        returncode = 1
        stdout = ""
        stderr = "boom"

    return R()


def test_cli_backend_parses_result():
    assert CliBackend(runner=_runner_ok).complete("hi") == "answer"


def test_cli_backend_normalizes_failure():
    with pytest.raises(LlmError):
        CliBackend(runner=_runner_fail).complete("hi")


def _runner_fail_stdout(cmd, **kw):
    class R:
        returncode = 1
        stdout = '{"is_error":true,"result":"usage limit reached"}'
        stderr = ""

    return R()


def test_cli_backend_failure_reports_stdout_when_stderr_empty():
    with pytest.raises(LlmError, match="usage limit reached"):
        CliBackend(runner=_runner_fail_stdout).complete("hi")


def test_gen_config_defaults_to_opus_model():
    from thai_deck_gen.config import GenConfig
    assert GenConfig().model == "claude-opus-5"


def test_cli_llm_factory_passes_configured_model_to_backend(tmp_path):
    from thai_deck_gen.cli import _cli_llm
    from thai_deck_gen.config import GenConfig
    llm = _cli_llm(tmp_path, GenConfig(model="claude-sonnet-5"))
    assert llm.inner.model == "claude-sonnet-5"
    assert llm.model == "claude-sonnet-5"      # cache key namespace follows
    llm.close()


def _runner_fail_json(cmd, **kw):
    class R:
        returncode = 1
        stdout = ('{"type":"result","is_error":true,"duration_api_ms":0,"usage":{"input_tokens":0},'
                  '"result":"You have hit your usage limit. Resets at 8:10pm."}')
        stderr = ""

    return R()


def test_cli_backend_failure_reports_the_result_text_from_json_stdout():
    with pytest.raises(LlmError, match=r"^You have hit your usage limit"):
        CliBackend(runner=_runner_fail_json).complete("hi")


# --- API backend: keeps drafting off the subscription quota ---

class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Msg:
    def __init__(self, text): self.content = [_Block(text)]


class FakeAnthropic:
    def __init__(self, reply="ok"):
        self.calls = []
        outer = self
        class _Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                return _Msg(reply)
        self.messages = _Messages()


def test_api_backend_returns_text_and_passes_the_model():
    from thai_deck_gen.llm import ApiBackend
    client = FakeAnthropic("drafted")
    out = ApiBackend(model="claude-sonnet-5", client=client).complete("prompt")
    assert out == "drafted"
    assert client.calls[0]["model"] == "claude-sonnet-5"
    assert client.calls[0]["messages"][0]["content"] == "prompt"


def test_api_backend_surfaces_failures_as_llm_error():
    from thai_deck_gen.llm import ApiBackend, LlmError

    class Boom:
        class messages:
            @staticmethod
            def create(**kw):
                raise RuntimeError("no credit")

    with pytest.raises(LlmError):
        ApiBackend(model="claude-sonnet-5", client=Boom()).complete("p")
