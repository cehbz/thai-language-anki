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
