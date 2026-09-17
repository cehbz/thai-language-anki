"""Test-wide guards.

No test may reach a credential store: a fixture's providers.yaml may carry
an `op://` reference as a placeholder, and resolving it for real runs the
1Password CLI, which prompts the user for authentication (observed
2026-09-17). Every test gets a fake `op` first on PATH that refuses at
once, so a test that reaches a secret fails loudly instead of prompting.

No unit test may load the real pronunciation engines either: constructing
`phonology.default_engines()` imports pythainlp and torch, which costs
seconds and hundreds of megabytes, and makes the test's result depend on
a model file. Engines are injected (`Sourcing.engines`, the seam every
caller reads first), so a test that reaches the default is a test with a
wiring hole -- fix the test, never the guard. The `integration` marker is
the one exemption: that is the suite that exists to exercise the real
thing (pyproject's default filter deselects it).
"""
from __future__ import annotations

import os
import stat
import sys

import pytest


@pytest.fixture(autouse=True)
def _no_real_secret_store(tmp_path_factory, monkeypatch):
    fake_bin = tmp_path_factory.mktemp("fake-op")
    op = fake_bin / "op"
    op.write_text("#!/bin/sh\necho 'tests must not resolve 1Password secrets' >&2\nexit 97\n")
    op.chmod(op.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")


@pytest.fixture(autouse=True)
def real_default_engines(request, monkeypatch):
    """Refuses `phonology.default_engines` for every test but an
    `integration` one, and hands back the real function so the one test
    that is *about* that function can still reach it.

    `attempts.py` and `run.py` bind the name at import (`from .phonology
    import default_engines`), so patching the definition alone would miss
    them: every module holding that exact function object is patched. A
    module imported after this fixture runs picks the refusal up through
    the same from-import.
    """
    from thai_syllabus import phonology
    real = phonology.default_engines
    if request.node.get_closest_marker("integration"):
        return real

    def _refuse(*_args, **_kwargs):
        raise AssertionError(
            "tests must inject Engines; the real thaig2p never loads in a test")

    for module in [m for name, m in list(sys.modules.items())
                   if name.startswith("thai_syllabus")
                   and getattr(m, "default_engines", None) is real]:
        monkeypatch.setattr(module, "default_engines", _refuse)
    return real
