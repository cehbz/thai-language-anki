"""Guards for every test.

No test may reach a credential store: a fixture's providers.yaml may carry
an `op://` reference as a placeholder, and resolving it for real runs the
1Password CLI, which prompts the user for authentication (observed
2026-09-17). Every test gets a fake `op` first on PATH that refuses at
once, so a test that reaches a secret fails loudly instead of prompting.
No test may run the Claude CLI either: a cli transport a test leaves
unreplaced spawns the real `claude`, which spends subscription quota and
answers or fails by the live account (observed 2026-10-03). A fake
`claude` that refuses at once sits beside the fake `op`.
"""
from __future__ import annotations

import os
import stat

import pytest


@pytest.fixture(autouse=True)
def _no_real_secret_store(tmp_path_factory, monkeypatch):
    fake_bin = tmp_path_factory.mktemp("fake-op")
    op = fake_bin / "op"
    op.write_text("#!/bin/sh\necho 'tests must not resolve 1Password secrets' >&2\nexit 97\n")
    op.chmod(op.stat().st_mode | stat.S_IXUSR)
    claude = fake_bin / "claude"
    claude.write_text("#!/bin/sh\necho 'tests must not run the claude CLI' >&2\nexit 97\n")
    claude.chmod(claude.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
