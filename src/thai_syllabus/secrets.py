"""API-key resolution for curated/providers.yaml's secret references
(spec 3 section 5), ported out of thai_deck_eval/secrets.py -- the one
carry-over module the spec names explicitly. No cross-package import
(this package imports nothing out of thai_deck_eval/thai_deck_gen by
design, see __init__.py); this is a straight copy, kept in its own
module here rather than shared, so thai_syllabus stays self-contained.

A config file (curated/providers.yaml) holds a *reference* to each
secret, never the secret itself: either a 1Password secret reference
(`op://<vault>/<item>/<field>`) or a path to an owner-only (0600) file.
An inline literal is refused, so a deck directory never becomes a place
secrets accumulate.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

OP_PREFIX = "op://"


class SecretError(RuntimeError):
    """Raised when a configured secret cannot be resolved."""


def resolve_secret(spec: str, *, name: str, runner: Callable = subprocess.run) -> str:
    """Resolve one `secrets.<name>` reference to its value."""
    if spec.startswith(OP_PREFIX):
        return _read_op(spec, name=name, runner=runner)
    return _read_file(spec, name=name)


def _read_op(spec: str, *, name: str, runner: Callable) -> str:
    cmd = ["op", "read", "--no-newline", spec]
    try:
        proc = runner(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise SecretError(f"secrets.{name}: cannot run `op read`: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise SecretError(f"secrets.{name}: `op read {spec}` failed: {detail}")
    value = (proc.stdout or "").strip()
    if not value:
        raise SecretError(f"secrets.{name}: `op read {spec}` returned nothing")
    return value


def _read_file(spec: str, *, name: str) -> str:
    path = Path(spec).expanduser()
    if not path.is_file():
        raise SecretError(
            f"secrets.{name}: {spec!r} is neither an {OP_PREFIX} reference nor an "
            "existing file. Secrets are referenced from providers.yaml, never "
            "written into it.")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise SecretError(
            f"secrets.{name}: {path} is readable beyond its owner "
            f"(mode {mode:03o}); chmod 600 it")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise SecretError(f"secrets.{name}: {path} is empty")
    return value


@dataclass
class SecretStore:
    """Lazily resolves configured secrets, once each per process.

    Resolution is deferred so commands that touch no paid channel never
    reach for 1Password, and so a misconfigured reference fails at the
    start of the run that needs it rather than hours in.
    """
    specs: Mapping[str, str | None] = field(default_factory=dict)
    runner: Callable = subprocess.run
    _resolved: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, secrets_config, runner: Callable = subprocess.run) -> "SecretStore":
        return cls(specs=dict(secrets_config), runner=runner)

    @classmethod
    def fixed(cls, **values: str) -> "SecretStore":
        """Store of already-resolved values (tests, injected credentials)."""
        store = cls(specs={k: v for k, v in values.items()})
        store._resolved.update(values)
        return store

    def configured(self, name: str) -> bool:
        return bool(self.specs.get(name))

    def get(self, name: str) -> str | None:
        """Resolved value, or None when `secrets.<name>` is unset."""
        if name in self._resolved:
            return self._resolved[name]
        spec = self.specs.get(name)
        if not spec:
            return None
        value = resolve_secret(spec, name=name, runner=self.runner)
        self._resolved[name] = value
        return value
