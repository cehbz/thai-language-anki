import subprocess

import pytest

from thai_deck_eval.secrets import SecretError, SecretStore, resolve_secret


def _key_file(tmp_path, text="k3y\n", mode=0o600):
    path = tmp_path / "forvo.key"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


class _Runner:
    """subprocess.run stand-in recording argv."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.result = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return self.result


def test_op_reference_resolves_via_op_read():
    runner = _Runner(stdout="s3cret")
    assert resolve_secret("op://Shared/Forvo/API Key", name="forvo",
                          runner=runner) == "s3cret"
    assert runner.calls == [["op", "read", "--no-newline",
                             "op://Shared/Forvo/API Key"]]


def test_op_failure_raises_with_key_name_and_stderr():
    runner = _Runner(stderr="item not found", returncode=1)
    with pytest.raises(SecretError) as err:
        resolve_secret("op://Shared/Nope/key", name="forvo", runner=runner)
    assert "secrets.forvo" in str(err.value)
    assert "item not found" in str(err.value)


def test_missing_op_binary_raises_secret_error():
    def runner(cmd, **kwargs):
        raise FileNotFoundError("op")

    with pytest.raises(SecretError):
        resolve_secret("op://Shared/Forvo/API Key", name="forvo", runner=runner)


def test_file_reference_reads_and_strips(tmp_path):
    path = _key_file(tmp_path)
    assert resolve_secret(str(path), name="forvo") == "k3y"


def test_file_reference_expands_tilde(tmp_path, monkeypatch):
    path = _key_file(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_secret("~/forvo.key", name="forvo") == "k3y"


def test_group_or_world_readable_file_is_refused(tmp_path):
    path = _key_file(tmp_path, mode=0o644)
    with pytest.raises(SecretError) as err:
        resolve_secret(str(path), name="forvo")
    assert "644" in str(err.value)


def test_empty_file_is_refused(tmp_path):
    path = _key_file(tmp_path, text="\n")
    with pytest.raises(SecretError):
        resolve_secret(str(path), name="forvo")


def test_inline_literal_is_refused(tmp_path):
    with pytest.raises(SecretError) as err:
        resolve_secret("abc123deadbeef", name="forvo")
    assert "op://" in str(err.value)


def test_store_caches_one_resolution():
    runner = _Runner(stdout="s3cret")
    store = SecretStore(specs={"forvo": "op://Shared/Forvo/API Key"}, runner=runner)
    assert store.configured("forvo")
    assert store.get("forvo") == "s3cret"
    assert store.get("forvo") == "s3cret"
    assert len(runner.calls) == 1


def test_store_unconfigured_is_none_and_never_runs():
    runner = _Runner(stdout="s3cret")
    store = SecretStore(specs={"forvo": None}, runner=runner)
    assert not store.configured("forvo")
    assert store.get("forvo") is None
    assert store.get("google_tts") is None
    assert runner.calls == []


def test_store_fixed_values_report_configured():
    store = SecretStore.fixed(forvo="KEY")
    assert store.configured("forvo")
    assert store.get("forvo") == "KEY"
    assert not store.configured("google_tts")
