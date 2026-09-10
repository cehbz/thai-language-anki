"""Tests for safety.py's CuratedHistory (spec 2 section 6, "Deck safety"):
a git repository over curated/, committed by label, that a writing
command commits before and after touching any store.
"""
import sqlite3
import subprocess
from datetime import date

import pytest

from thai_syllabus.cachekeys import ProvideKey
from thai_syllabus.safety import Counts, CuratedHistory, HistoryError, check, deck_counts, snapshot
from thai_syllabus.store import SyllabusDb


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


# -- snapshot ---------------------------------------------------------------

def _append_row(db, query):
    db.append("provide", "openverse", ProvideKey(source="openverse", kind="", query=query),
              "rice", {"kind": "picture", "query": query}, {"items": []}, 0)


def test_snapshot_copies_a_db_with_one_row_to_the_backup_path(tmp_path):
    db_path = tmp_path / "deck" / "syllabus.db"
    backup_path = tmp_path / "deck" / "backup" / "syllabus.db"
    db = SyllabusDb(db_path)
    _append_row(db, "k1")

    snapshot(db_path, backup_path)

    con = sqlite3.connect(backup_path)
    assert con.execute("select count(*) from cache").fetchone()[0] == 1
    con.close()
    db.close()


def test_a_row_appended_after_the_snapshot_leaves_the_backup_at_one(tmp_path):
    db_path = tmp_path / "deck" / "syllabus.db"
    backup_path = tmp_path / "deck" / "backup" / "syllabus.db"
    db = SyllabusDb(db_path)
    _append_row(db, "k1")
    snapshot(db_path, backup_path)

    _append_row(db, "k2")

    con = sqlite3.connect(backup_path)
    assert con.execute("select count(*) from cache").fetchone()[0] == 1
    con.close()
    db.close()


def test_a_second_snapshot_replaces_the_first(tmp_path):
    db_path = tmp_path / "deck" / "syllabus.db"
    backup_path = tmp_path / "deck" / "backup" / "syllabus.db"
    db = SyllabusDb(db_path)
    _append_row(db, "k1")
    snapshot(db_path, backup_path)
    _append_row(db, "k2")

    snapshot(db_path, backup_path)

    con = sqlite3.connect(backup_path)
    assert con.execute("select count(*) from cache").fetchone()[0] == 2
    con.close()
    db.close()


def test_snapshot_failing_to_open_the_backup_path_still_releases_the_source(tmp_path):
    """backup_path pointing at an existing directory makes
    sqlite3.connect(backup_path) raise OperationalError; snapshot must
    still release the source connection it already opened, or a leaked
    connection could later block another connection from writing db_path
    (spec 2 section 6 -- snapshot never holds the live db open).
    """
    db_path = tmp_path / "deck" / "syllabus.db"
    backup_path = tmp_path / "deck" / "backup" / "syllabus.db"
    backup_path.mkdir(parents=True)  # a directory sits where snapshot wants a file
    db = SyllabusDb(db_path)
    _append_row(db, "k1")
    db.close()

    with pytest.raises(sqlite3.OperationalError):
        snapshot(db_path, backup_path)

    # a leaked src connection could leave the db locked; prove it did not
    # by opening and writing through a fresh connection.
    other = SyllabusDb(db_path)
    other.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="k2"),
                 "rice", {"kind": "picture", "query": "k2"}, {"items": []}, 0)
    other.close()


# -- deck_counts --------------------------------------------------------

def test_deck_counts_reads_words_targets_and_table_rows(tmp_path):
    deck = tmp_path / "deck"
    curated = deck / "curated"
    curated.mkdir(parents=True)
    (curated / "words.yaml").write_text("- rice\n- fish\n", encoding="utf-8")
    (curated / "targets.yaml").write_text("- rice_receptive\n", encoding="utf-8")
    db = SyllabusDb(deck / "syllabus.db")
    db.add_sentence(text_sha="s1", text="ข้าว", clauses=(("rice",),), gloss="rice",  # rice
                    voice="learner_voice", source="llm", origin="draft",
                    licence="n/a", acquired=date(2026, 1, 1))
    db.close()

    assert deck_counts(deck) == Counts(words=2, targets=1, sentences=1, cache=0, media=0)


def test_deck_counts_on_an_absent_deck_is_all_zero(tmp_path):
    assert deck_counts(tmp_path / "no-such-deck") == Counts(
        words=0, targets=0, sentences=0, cache=0, media=0)


def test_deck_counts_sees_a_committed_row_while_the_writer_is_still_open(tmp_path):
    """The read-only URI connection must see a row committed by a live
    WAL-mode writer that has not closed (and so has not checkpointed) --
    sqlite's WAL read path, not the checkpointed main db file, is what a
    read-only connection reads from (spec 2 section 6).
    """
    deck = tmp_path / "deck"
    db = SyllabusDb(deck / "syllabus.db")
    _append_row(db, "k1")

    counts = deck_counts(deck)

    assert counts.cache == 1
    db.close()


# -- check --------------------------------------------------------------

def test_check_names_a_sentences_shortfall():
    before = Counts(words=2, targets=1, sentences=3, cache=5, media=0)
    after = Counts(words=2, targets=1, sentences=2, cache=5, media=0)

    assert check(before, after, {}) == [
        "sentences: 2 rows after, 3 in the snapshot, 0 removal(s) reported"]


def test_check_accepts_a_shortfall_the_removals_cover():
    before = Counts(words=2, targets=1, sentences=3, cache=5, media=0)
    after = Counts(words=2, targets=1, sentences=2, cache=5, media=0)

    assert check(before, after, {"sentences": 1}) == []
