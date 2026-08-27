import json
import subprocess
import pytest
from thai_deck_eval.config import JudgeConfig
from thai_deck_eval.judge.cli_judge import CliJudge, JudgeError
from thai_deck_eval.judge.api_judge import ApiJudge
from thai_deck_eval.judge.core import JudgeRequest, Verdicts

GOOD = json.dumps({"result": json.dumps({"verdicts": [
    {"rule": "judge/x", "passed": True, "confidence": 0.9, "rationale": "ok"}]})})

class FakeRun:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.cmds = []
    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        class R:
            returncode = 0
            stdout = self.outputs.pop(0)
            stderr = ""
        return R()

def test_cli_judge_parses():
    runner = FakeRun([GOOD])
    j = CliJudge(JudgeConfig(), runner=runner)
    out = j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))
    assert out[0].passed is True
    assert runner.cmds[0][:2] == ["claude", "-p"]

def test_cli_judge_retries_then_raises():
    runner = FakeRun([json.dumps({"result": "not json"}),
                      json.dumps({"result": "still not"})])
    j = CliJudge(JudgeConfig(), runner=runner)
    with pytest.raises(JudgeError):
        j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))
    assert len(runner.cmds) == 2

def test_cli_judge_image_adds_read_tool():
    runner = FakeRun([GOOD])
    j = CliJudge(JudgeConfig(), runner=runner)
    j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p",
                         image_path="/tmp/x/img.png"))
    cmd = runner.cmds[0]
    assert "--allowedTools" in cmd and "Read" in cmd
    assert any("img.png" in part for part in cmd)

class FakeTimeoutRun:
    def __call__(self, cmd, **kw):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=600)

def test_cli_judge_timeout_raises_judge_error():
    j = CliJudge(JudgeConfig(), runner=FakeTimeoutRun())
    with pytest.raises(JudgeError):
        j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))


# --- ApiJudge -----------------------------------------------------------

class FakeParseResponse:
    def __init__(self, stop_reason="end_turn", parsed_output=None):
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output

class FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls = []
    def parse(self, **kw):
        self.calls.append(kw)
        return self._response

class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)

GOOD_VERDICTS = Verdicts(verdicts=[
    {"rule": "judge/x", "passed": True, "confidence": 0.9, "rationale": "ok"}])

def test_api_judge_parses():
    client = FakeClient(FakeParseResponse(parsed_output=GOOD_VERDICTS))
    j = ApiJudge(JudgeConfig(), client=client)
    out = j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))
    assert out[0].passed is True
    call = client.messages.calls[0]
    assert call["model"] == JudgeConfig().model
    assert call["output_config"] == {"effort": JudgeConfig().effort}
    assert call["output_format"] is Verdicts

def test_api_judge_raises_on_refusal():
    client = FakeClient(FakeParseResponse(stop_reason="refusal", parsed_output=None))
    j = ApiJudge(JudgeConfig(), client=client)
    with pytest.raises(JudgeError):
        j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))

def test_api_judge_raises_on_validation_failure():
    client = FakeClient(FakeParseResponse(parsed_output=None))
    j = ApiJudge(JudgeConfig(), client=client)
    with pytest.raises(JudgeError):
        j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))

def test_api_judge_image_adds_vision_block(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = FakeClient(FakeParseResponse(parsed_output=GOOD_VERDICTS))
    j = ApiJudge(JudgeConfig(), client=client)
    j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p",
                         image_path=str(img)))
    content = client.messages.calls[0]["messages"][0]["content"]
    assert any(b["type"] == "image" for b in content)
    image_block = next(b for b in content if b["type"] == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
