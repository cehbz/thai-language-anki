"""Deck safety (spec 2 section 6): curated/'s history as a git repository
that every writing command commits before and after, the syllabus.db
snapshot a writing command copies before it writes, the sanity check that
compares the deck's row counts to that snapshot, and writing_command --
the context manager every writing command (run, migrate, review, import)
runs under to get all three for free.
"""
from __future__ import annotations

import contextlib
import logging
import sqlite3
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

_log = logging.getLogger(__name__)


class HistoryError(RuntimeError):
    """A git call under curated/'s history exited non-zero (spec 2 section 6)."""


@dataclass(frozen=True)
class CuratedHistory:
    """curated/'s history: a git repository at root, machine-owned, never
    rewritten -- committed by label before and after a writing command
    (spec 2 section 6).
    """
    root: Path

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Every git call for this history goes through here: pinned to a
        fixed identity so no test depends on the machine's git identity,
        and with `-c core.hooksPath=/dev/null -c commit.gpgsign=false` so
        an automated commit can never block on an inherited hook or a
        signing agent (spec 2 section 6). By default a non-zero exit
        raises HistoryError carrying stderr; pass check=False to get the
        CompletedProcess back instead, for a caller that must tell one
        non-zero exit apart from another.
        """
        result = subprocess.run(
            ["git", "-C", str(self.root),
             "-c", "user.name=thai-syllabus",
             "-c", "user.email=thai-syllabus@localhost",
             "-c", "core.hooksPath=/dev/null",
             "-c", "commit.gpgsign=false",
             *args],
            capture_output=True, text=True,
        )
        if check and result.returncode != 0:
            raise HistoryError(result.stderr)
        return result

    def ensure(self) -> None:
        """Create root and `git init -q` it when root/.git is absent; no
        commit (spec 2 section 6).
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / ".git").exists():
            self._run("init", "-q")

    def commit(self, label: str) -> str | None:
        """`git add -A -f` (curated/ holds only curated files, so a
        `.gitignore` there never hides one from a commit) then commit as
        label; returns the new sha, or None when the tree already equals
        HEAD, or HEAD is absent and the tree is empty (spec 2 section 6).
        """
        self._run("add", "-A", "-f")
        if not self._run("status", "--porcelain").stdout.strip():
            return None
        self._run("commit", "-q", "-m", label)
        return self._run("rev-parse", "HEAD").stdout.strip()

    def last_pre_commit(self) -> str | None:
        """The newest commit whose subject starts with "pre " (git log
        --format=%H %s), else None when HEAD is absent (an empty
        history). Raises HistoryError naming root when root/.git is
        absent or the repo is otherwise unreadable: a missing or corrupt
        history is never silently treated as "no pre commit" (spec 2
        section 6).
        """
        if not (self.root / ".git").exists():
            raise HistoryError(f"no curated history at {self.root}: .git is absent")
        probe = self._run("rev-parse", "--verify", "-q", "HEAD", check=False)
        if probe.returncode == 1 and not probe.stderr.strip():
            return None  # unborn HEAD: no commits yet, not an error
        if probe.returncode != 0:
            raise HistoryError(probe.stderr)
        log = self._run("log", "--format=%H %s").stdout.splitlines()
        for line in log:
            sha, _, subject = line.partition(" ")
            if subject.startswith("pre "):
                return sha
        return None

    def restore_tree(self, sha: str) -> None:
        """Restore the working tree to sha, then `git clean -fdxq` (the
        -x also removes a file a `.gitignore` in curated/ names, since
        curated/ holds only curated files and ignore rules never apply
        there) so the working tree equals sha (spec 2 section 6).
        """
        self._run("restore", "--source", sha, "--worktree", "--staged", "--", ".")
        self._run("clean", "-fdxq")


def snapshot(db_path: Path, backup_path: Path) -> None:
    """Copy db_path through sqlite's online backup API to backup_path,
    replacing whatever backup_path already held: one generation, the
    state before the most recent writing command (spec 2 section 6). Safe
    against a live WAL-mode source -- the backup API reads a consistent
    snapshot without requiring the source be closed or checkpointed. Both
    connections are closed on every path, including when opening
    backup_path itself fails (e.g. a directory sits there), so a failed
    snapshot never leaves the source db open.
    """
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(db_path)) as src, \
            contextlib.closing(sqlite3.connect(backup_path)) as dst:
        src.backup(dst)


@dataclass(frozen=True)
class Counts:
    """A deck's row counts, as the sanity check compares them before and
    after a writing command: words and targets from curated/, sentences,
    cache and media from syllabus.db (spec 2 section 6).
    """
    words: int
    targets: int
    sentences: int
    cache: int
    media: int


def _yaml_list_len(path: Path) -> int:
    if not path.exists():
        return 0
    return len(yaml.safe_load(path.read_text(encoding="utf-8")) or [])


def deck_counts(deck: Path) -> Counts:
    """words and targets are curated/words.yaml and curated/targets.yaml's
    own list lengths (0 when a file is absent); sentences, cache and media
    are read from syllabus.db through a read-only connection, so a check
    never takes a write lock or creates a db that was not there. Every
    field is 0 when syllabus.db is absent (spec 2 section 6).
    """
    deck = Path(deck)
    words = _yaml_list_len(deck / "curated" / "words.yaml")
    targets = _yaml_list_len(deck / "curated" / "targets.yaml")
    db_path = deck / "syllabus.db"
    if not db_path.exists():
        return Counts(words=words, targets=targets, sentences=0, cache=0, media=0)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sentences = con.execute("select count(*) from sentences").fetchone()[0]
        cache = con.execute("select count(*) from cache").fetchone()[0]
        media = con.execute("select count(*) from media").fetchone()[0]
    finally:
        con.close()
    return Counts(words=words, targets=targets, sentences=sentences, cache=cache, media=media)


def check(before: Counts, after: Counts, removals: Mapping[str, int]) -> list[str]:
    """One message per field (words, targets, sentences, cache, media)
    where `after` counts fewer rows than `before` minus the removals
    reported for that field -- a writing command's sanity check (spec 2
    section 6).
    """
    problems = []
    for field in ("words", "targets", "sentences", "cache", "media"):
        b = getattr(before, field)
        a = getattr(after, field)
        n = removals.get(field, 0)
        if a < b - n:
            problems.append(
                f"{field}: {a} rows after, {b} in the snapshot, {n} removal(s) reported")
    return problems


class SafetyCheckFailed(RuntimeError):
    """A writing command's post-check (safety.check, spec 2 section 6)
    found a row-count shortfall that the command's own removals did not
    account for. `failures` carries check()'s messages, one per field.
    """

    def __init__(self, failures: list[str]) -> None:
        super().__init__("; ".join(failures))
        self.failures = failures


@dataclass
class Guard:
    """A writing command's own account of rows it deliberately removed
    (e.g. migrate's unmigratable sentences), keyed by the same field
    names as Counts/check -- passed to writing_command's post-check so a
    deliberate removal never reads as a safety failure (spec 2 section 6).
    """
    removals: dict[str, int] = field(default_factory=dict)

    def removed(self, store: str, ids: Iterable[str]) -> None:
        """Record len(ids) removals against store, adding to any already
        recorded for it this command (spec 2 section 6).
        """
        count = len(list(ids))
        self.removals[store] = self.removals.get(store, 0) + count


@contextmanager
def writing_command(deck: Path, name: str, *,
                    now: Callable[[], datetime] = datetime.now) -> Iterator[Guard]:
    """The context every writing command (run, migrate, review, import)
    runs its body under (spec 2 section 6): a "pre {name} {stamp}" commit
    of curated/'s history, a snapshot of syllabus.db when one exists, then
    the body runs with a Guard it can report deliberate removals to. A
    "post {name} {stamp}" commit always follows the body, whether it
    returned or raised -- the history covers what the command actually
    left behind either way. When the body raised, that exception
    propagates and no post-check runs; a HistoryError from that post
    commit itself is only logged (never lets a history-write failure mask
    the body's own exception) (spec 2 section 6). When the body returned,
    the before/after row counts are compared (safety.check) against the
    guard's reported removals; any unaccounted shortfall raises
    SafetyCheckFailed after the post commit is made.
    """
    stamp = now().astimezone().isoformat(timespec="seconds")
    history = CuratedHistory(deck / "curated")
    history.ensure()
    history.commit(f"pre {name} {stamp}")
    db_path = deck / "syllabus.db"
    if db_path.exists():
        snapshot(db_path, deck / "backup" / "syllabus.db")
    before = deck_counts(deck)
    guard = Guard()
    try:
        yield guard
    except BaseException:
        try:
            history.commit(f"post {name} {stamp}")
        except HistoryError as e:
            _log.warning("post commit failed after %s: %s", name, e)
        raise
    failures = check(before, deck_counts(deck), guard.removals)
    history.commit(f"post {name} {stamp}")
    if failures:
        raise SafetyCheckFailed(failures)
