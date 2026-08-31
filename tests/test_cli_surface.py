"""Every option of the evaluator CLI, and the configuration seam.

The evaluator's exit code is a gate other tools depend on, and its options
decide what runs and what it costs. Each test here fails if an option stops
being honoured.
"""
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import thai_deck_eval.cli as cli
from tests.helpers import DeckBuilder


@pytest.fixture
def deck_dir(tmp_path):
    return DeckBuilder(tmp_path).build()


def _run(deck_dir, *args):
    return CliRunner().invoke(cli.main, [str(deck_dir), *args])


def test_no_judge_skips_the_judge_stage(deck_dir, monkeypatch):
    monkeypatch.setattr(cli, "build_judge",
                        lambda cfg: pytest.fail("judge built despite --no-judge"))
    res = _run(deck_dir, "--no-judge", "--format", "json")
    assert res.exit_code in (0, 1)
    assert "judge" not in json.loads(res.output)["stages_run"]


def test_stages_option_selects_exactly_those_stages(deck_dir):
    res = _run(deck_dir, "--stages", "mechanical", "--format", "json")
    report = json.loads(res.output)
    assert report["stages_run"] == ["mechanical"]


def test_report_option_writes_the_json_to_that_path(deck_dir, tmp_path):
    out = tmp_path / "r.json"
    _run(deck_dir, "--no-judge", "--report", str(out))
    assert json.loads(out.read_text())["deck_name"]


def test_format_text_is_human_readable_and_json_is_parseable(deck_dir):
    text = _run(deck_dir, "--no-judge", "--format", "text").output
    assert "gate:" in text.lower()
    assert json.loads(_run(deck_dir, "--no-judge", "--format", "json").output)


def test_rulebook_option_changes_behaviour(deck_dir, tmp_path):
    """A rulebook that scores nothing proves the file was actually read."""
    rb = tmp_path / "rb.yaml"
    rb.write_text(yaml.safe_dump({"deductions": {"error": 0.0, "warn": 0.0,
                                                 "info": 0.0}}))
    res = _run(deck_dir, "--no-judge", "--rulebook", str(rb), "--format", "json")
    scores = json.loads(res.output)["scores"]
    assert scores["integrity"] == 100.0


def test_exit_code_one_on_an_error_finding(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["minimal_pairs"][0]["members"][0]["audio"]["source"] = "tts"
    res = _run(b.build(), "--no-judge")
    assert res.exit_code == 1


def test_exit_code_zero_when_the_gate_passes(deck_dir):
    assert _run(deck_dir, "--no-judge").exit_code == 0


def test_waivers_from_the_deck_are_applied(tmp_path):
    """A waived error must not fail the gate."""
    import hashlib
    b = DeckBuilder(tmp_path)
    b.data["minimal_pairs"][0]["members"][0]["audio"]["source"] = "tts"
    root = b.build()
    note_id = b.data["minimal_pairs"][0]["id"]
    (root / "waivers.yaml").write_text(yaml.safe_dump([{
        "note_id": note_id, "rule": "meth/tts-audio", "reason": "reviewed",
        "date": "2026-08-31"}]))
    res = _run(root, "--stages", "method", "--format", "json")
    report = json.loads(res.output)
    assert [f["rule"] for f in report["waived"]] == ["meth/tts-audio"]
    assert res.exit_code == 0


@pytest.mark.parametrize("backend,expected", [
    ("fake", "FakeJudge"), ("cli", "CliJudge"), ("batch", "BatchJudge"),
    ("api", "ApiJudge"),
])
def test_every_judge_backend_can_be_built(backend, expected, monkeypatch):
    """A rename in this factory breaks every real run while units stay green."""
    from thai_deck_eval.config import JudgeConfig, RulebookConfig
    monkeypatch.setattr(cli, "_api_client", lambda cfg: "CLIENT")
    judge = cli.build_judge(RulebookConfig(judge=JudgeConfig(backend=backend)))
    inner = getattr(judge, "inner", judge)
    assert type(inner).__name__ == expected


def test_batch_backend_gets_a_state_path_beside_its_cache(monkeypatch, tmp_path):
    """Without it a killed run resubmits a batch that was already paid for."""
    from thai_deck_eval.config import JudgeConfig, RulebookConfig
    monkeypatch.setattr(cli, "_api_client", lambda cfg: "CLIENT")
    cfg = RulebookConfig(judge=JudgeConfig(backend="batch",
                                           cache_path=str(tmp_path / "c.sqlite")))
    judge = cli.build_judge(cfg)
    assert judge.inner.state_path == Path(f"{tmp_path / 'c.sqlite'}.batch.json")


def test_api_key_reference_is_resolved_for_the_api_backends(monkeypatch, tmp_path):
    """The key is a reference in the rulebook, never a literal."""
    from thai_deck_eval.config import JudgeConfig, RulebookConfig
    key = tmp_path / "k.key"
    key.write_text("sk-test\n")
    key.chmod(0o600)
    captured = {}

    class FakeAnthropic:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key

    monkeypatch.setitem(__import__("sys").modules, "anthropic",
                        type("m", (), {"Anthropic": FakeAnthropic}))
    cli._api_client(RulebookConfig(judge=JudgeConfig(api_key=str(key))))
    assert captured["api_key"] == "sk-test"
