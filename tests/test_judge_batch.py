import json
from pathlib import Path

import pytest

from thai_deck_eval.config import JudgeConfig
from thai_deck_eval.judge.batch_judge import BatchJudge
from thai_deck_eval.judge.cli_judge import JudgeError
from thai_deck_eval.judge.core import JudgeRequest

VERDICT_JSON = json.dumps({"verdicts": [
    {"rule": "judge/unnatural-sentence", "passed": True,
     "confidence": 0.9, "rationale": "fine"}]})


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Message:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.stop_reason = "end_turn"


class _Result:
    def __init__(self, type_, message=None):
        self.type, self.message = type_, message


class _Entry:
    def __init__(self, custom_id, result):
        self.custom_id, self.result = custom_id, result


class _Batch:
    def __init__(self, id_, status="ended"):
        self.id, self.processing_status = id_, status


class FakeBatches:
    """Stands in for client.messages.batches."""

    def __init__(self, statuses=("ended",), body=VERDICT_JSON, failures=()):
        self.submitted = []
        self.statuses = list(statuses)
        self.body = body
        self.failures = set(failures)
        self.results_calls = 0

    def create(self, requests):
        self.submitted.append(requests)
        return _Batch(f"batch_{len(self.submitted)}")

    def retrieve(self, batch_id):
        status = self.statuses[0] if len(self.statuses) == 1 else self.statuses.pop(0)
        return _Batch(batch_id, status)

    def results(self, batch_id):
        self.results_calls += 1
        for req in self.submitted[-1]:
            cid = req["custom_id"]
            if cid in self.failures:
                yield _Entry(cid, _Result("errored"))
            else:
                yield _Entry(cid, _Result("succeeded", _Message(self.body)))


class FakeClient:
    def __init__(self, batches):
        self.messages = type("M", (), {"batches": batches})()


def _judge(tmp_path, batches, **kw):
    return BatchJudge(JudgeConfig(backend="batch", model="claude-sonnet-5"),
                      client=FakeClient(batches),
                      state_path=tmp_path / "judge_batch.json",
                      sleep=lambda s: None, **kw)


def _req(note_id="sn-1", image=None):
    return JudgeRequest(note_id=note_id, rules=["judge/unnatural-sentence"],
                        prompt="judge this", image_path=image)


def test_one_batch_carries_every_request(tmp_path):
    batches = FakeBatches()
    out = _judge(tmp_path, batches).judge_many([_req("sn-1"), _req("sn-2")])
    assert len(batches.submitted) == 1
    assert len(batches.submitted[0]) == 2
    assert set(out) == {"sn-1", "sn-2"}
    assert out["sn-1"][0].rule == "judge/unnatural-sentence"


def test_custom_ids_are_api_safe_for_thai_note_ids(tmp_path):
    """Note ids carry Thai; custom_id must stay ^[a-zA-Z0-9_-]{1,64}$."""
    import re
    batches = FakeBatches()
    out = _judge(tmp_path, batches).judge_many([_req("pw-u-กิน"), _req("pw-u-ข้าว")])
    for req in batches.submitted[0]:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", req["custom_id"])
    assert set(out) == {"pw-u-กิน", "pw-u-ข้าว"}


def test_image_is_sent_inline(tmp_path):
    img = tmp_path / "pw.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
    batches = FakeBatches()
    _judge(tmp_path, batches).judge_many([_req("pw-1", image=str(img))])
    content = batches.submitted[0][0]["params"]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert content[1]["type"] == "text"


def test_polls_until_the_batch_ends(tmp_path):
    batches = FakeBatches(statuses=("in_progress", "in_progress", "ended"))
    out = _judge(tmp_path, batches).judge_many([_req("sn-1")])
    assert out["sn-1"]
    assert batches.results_calls == 1


def test_failed_items_are_omitted_not_fabricated(tmp_path):
    batches = FakeBatches(failures={"n0"})
    out = _judge(tmp_path, batches).judge_many([_req("sn-1"), _req("sn-2")])
    assert set(out) == {"sn-2"}


def test_unparseable_body_omits_that_item(tmp_path):
    batches = FakeBatches(body="I could not comply.")
    out = _judge(tmp_path, batches).judge_many([_req("sn-1")])
    assert out == {}


def test_in_flight_batch_is_resumed_not_resubmitted(tmp_path):
    """A killed run must harvest the batch it already paid for."""
    state = tmp_path / "judge_batch.json"
    batches = FakeBatches()
    first = _judge(tmp_path, batches)
    first.judge_many([_req("sn-1")])          # populates and clears state
    assert not state.exists()

    # Simulate a run killed after submit: rewrite the state by hand.
    state.write_text(json.dumps({"batch_id": "batch_1", "ids": {"n0": "sn-1"}}),
                     encoding="utf-8")
    resumed = _judge(tmp_path, batches)
    out = resumed.judge_many([_req("sn-1")])
    assert len(batches.submitted) == 1        # nothing new submitted
    assert out["sn-1"]
    assert not state.exists()


def test_submit_failure_raises_judge_error(tmp_path):
    class Boom(FakeBatches):
        def create(self, requests):
            raise RuntimeError("network down")

    with pytest.raises(JudgeError):
        _judge(tmp_path, Boom()).judge_many([_req("sn-1")])
