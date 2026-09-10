"""Deck safety (spec 2 section 6): curated/'s history as a git repository
that every writing command commits before and after. Only CuratedHistory
lives here so far; a later task adds the snapshot, counts, the
writing_command context manager and restore to this module.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


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
