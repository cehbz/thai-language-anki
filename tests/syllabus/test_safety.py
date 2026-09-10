"""Tests for safety.py's CuratedHistory (spec 2 section 6, "Deck safety"):
a git repository over curated/, committed by label, that a writing
command commits before and after touching any store.
"""
import subprocess

import pytest

from thai_syllabus.safety import CuratedHistory, HistoryError


def _git_log_subjects(root):
    out = subprocess.run(
        ["git", "-C", str(root), "log", "--format=%s"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.splitlines()


@pytest.fixture
def history(tmp_path):
    return CuratedHistory(root=tmp_path / "curated")


def test_ensure_creates_git_dir(history):
    history.ensure()
    assert (history.root / ".git").is_dir()


def test_ensure_twice_is_a_noop(history):
    history.ensure()
    marker = history.root / ".git" / "HEAD"
    before = marker.read_text(encoding="utf-8")
    history.ensure()
    assert marker.read_text(encoding="utf-8") == before


def test_commit_on_empty_fresh_repo_returns_none(history):
    history.ensure()
    assert history.commit("pre migrate 2026-09-10T00:00:00") is None


def test_commit_after_writing_a_file_returns_a_sha_with_the_label_subject(history):
    history.ensure()
    (history.root / "words.yaml").write_text("word: khaw5\n", encoding="utf-8")
    sha = history.commit("pre migrate 2026-09-10T00:00:00")
    assert sha is not None
    assert len(sha) == 40
    assert _git_log_subjects(history.root)[0] == "pre migrate 2026-09-10T00:00:00"


def test_second_commit_with_nothing_changed_returns_none(history):
    history.ensure()
    (history.root / "words.yaml").write_text("word: khaw5\n", encoding="utf-8")
    history.commit("pre migrate 2026-09-10T00:00:00")
    assert history.commit("post migrate 2026-09-10T00:00:01") is None


def test_last_pre_commit_is_the_newest_pre_subject(history):
    history.ensure()
    (history.root / "words.yaml").write_text("v1\n", encoding="utf-8")
    history.commit("pre migrate 2026-09-10T00:00:00")
    history.commit("post migrate 2026-09-10T00:00:01")
    (history.root / "words.yaml").write_text("v2\n", encoding="utf-8")
    second_pre = history.commit("pre migrate 2026-09-10T00:01:00")
    (history.root / "words.yaml").write_text("v3\n", encoding="utf-8")
    history.commit("post migrate 2026-09-10T00:01:01")
    assert history.last_pre_commit() == second_pre


def test_last_pre_commit_is_none_without_any_pre_commit(history):
    history.ensure()
    assert history.last_pre_commit() is None


def test_restore_tree_brings_back_a_changed_file_and_deletes_one_added_since(history):
    history.ensure()
    (history.root / "words.yaml").write_text("v1\n", encoding="utf-8")
    sha = history.commit("pre migrate 2026-09-10T00:00:00")
    (history.root / "words.yaml").write_text("v2\n", encoding="utf-8")
    (history.root / "targets.yaml").write_text("new\n", encoding="utf-8")

    history.restore_tree(sha)

    assert (history.root / "words.yaml").read_text(encoding="utf-8") == "v1\n"
    assert not (history.root / "targets.yaml").exists()


def test_restore_tree_with_an_unknown_sha_raises_history_error_with_stderr(history):
    history.ensure()
    with pytest.raises(HistoryError) as excinfo:
        history.restore_tree("deadbeef" * 5)
    assert str(excinfo.value).strip()


# -- Fix round 1 ----------------------------------------------------------

def test_last_pre_commit_on_a_root_never_ensured_raises_history_error_naming_it(history):
    with pytest.raises(HistoryError) as excinfo:
        history.last_pre_commit()
    assert str(history.root) in str(excinfo.value)


def test_last_pre_commit_on_an_ensured_zero_commit_repo_returns_none(history):
    history.ensure()
    assert history.last_pre_commit() is None


def test_commit_ignores_a_repo_local_hook_that_would_block_it(history):
    history.ensure()
    hooks_dir = history.root / ".hooks"
    hooks_dir.mkdir()
    hook = hooks_dir / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(history.root), "config", "core.hooksPath", str(hooks_dir)],
        check=True, capture_output=True, text=True,
    )
    (history.root / "words.yaml").write_text("v1\n", encoding="utf-8")

    sha = history.commit("pre migrate 2026-09-10T00:00:00")

    assert sha is not None


def test_commit_adds_a_file_a_gitignore_in_curated_ignores(history):
    history.ensure()
    (history.root / ".gitignore").write_text("*.yaml\n", encoding="utf-8")
    (history.root / "words.yaml").write_text("v1\n", encoding="utf-8")

    history.commit("pre migrate 2026-09-10T00:00:00")

    tracked = subprocess.run(
        ["git", "-C", str(history.root), "ls-files"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert "words.yaml" in tracked


def test_restore_tree_deletes_an_added_file_a_gitignore_in_curated_names(history):
    history.ensure()
    (history.root / ".gitignore").write_text("*.yaml\n", encoding="utf-8")
    (history.root / "words.yaml").write_text("v1\n", encoding="utf-8")
    sha = history.commit("pre migrate 2026-09-10T00:00:00")
    (history.root / "targets.yaml").write_text("new\n", encoding="utf-8")

    history.restore_tree(sha)

    assert not (history.root / "targets.yaml").exists()
