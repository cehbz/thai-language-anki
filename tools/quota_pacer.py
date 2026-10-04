"""Quota pacer (spec 3 r64 section 8): releases judge calls each tick up to
the week's unused pace-line allowance, under a 5-hour-window ceiling and a
per-tick cap, and runs one cycle with that many.

    uv run python tools/quota_pacer.py --deck D [--dry-run] [--now ISO]
                                       [--usage-file PATH]

The plan's windows come from the usage-probe mod (tools/claude-mods/
usage-probe): one trivial `claude -p` turn with the mod loaded writes
`$.session.usage()`'s rateLimits to the file THAI_SYLLABUS_USAGE_OUT
names. `--usage-file` reads such a file instead of probing.

Allowance (weekly percentage points) = elapsed fraction e of the weekly
window x 100 - weekly percent used - reserve_percent x (1 - e): the
reserve is whole just after the reset and gone at it. Calls M =
floor(min(allowance / points_per_call,
          (session_ceiling_percent - 5h percent used) / session_points_per_call,
          max_calls_per_tick)).

Runs only for a deck whose judge is on the cli transport. `--require-measured`
refuses a deck whose `pacer` block does not set both per-call figures.

Exit codes: the cycle's own when it ran (0 done), 3 nothing to do
(M < min_calls), 1 error (config, probe, missing or stale usage file).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import yaml

from thai_syllabus.curated import (CuratedValidationError, PacerConfig, ProvidersConfig,
                                   load_providers_config)

REPO = Path(__file__).resolve().parent.parent
MOD_DIR = REPO / "tools" / "claude-mods" / "usage-probe"
USAGE_ENV = "THAI_SYLLABUS_USAGE_OUT"
WEEK = timedelta(days=7)
FIVE_HOURS = timedelta(hours=5)
MAX_AGE = timedelta(minutes=10)
MAX_AHEAD = timedelta(seconds=60)
NO_FILE_HINT = "mod not loaded? check `claude --plugin-dir` / plugin_errors in the JSON"
# A quotient within this of a whole number is that number (15 / 0.05).
_EPSILON = 1e-9


class PacerError(Exception):
    """A tick that cannot be decided: exit 1 with the message."""


@dataclass(frozen=True)
class Window:
    kind: str
    percent_used: float
    resets_at: datetime | None


@dataclass(frozen=True)
class Reading:
    read_at: datetime
    five_hour: Window
    seven_day: Window


@dataclass(frozen=True)
class Tick:
    weekly_used: float
    weekly_elapsed: float   # percent of the weekly window elapsed: the pace line
    reserve: float          # reserve_percent tapered by the weekly window's remaining fraction
    session_used: float
    allowance: float        # weekly percentage points
    calls: int              # M


@dataclass(frozen=True)
class ProbeResult:
    returncode: int
    stdout: str
    stderr: str


ProbeRunner = Callable[..., ProbeResult]
CycleRunner = Callable[..., int]


def _parse_time(text: str, what: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise PacerError(f"{what}: {text!r} is not an ISO 8601 time") from None
    return moment if moment.tzinfo else moment.astimezone()


def parse_reading(text: str) -> Reading:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PacerError(f"usage file is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise PacerError("usage file is not a JSON object")
    read_at = _parse_time(data.get("read_at"), "usage file read_at")
    limits = {limit.get("kind"): limit for limit in data.get("rateLimits") or []
              if isinstance(limit, dict)}
    return Reading(read_at, _window(limits, "five_hour"), _window(limits, "seven_day"))


def _window(limits: dict[Any, dict], kind: str) -> Window:
    if kind not in limits:
        raise PacerError(f"usage file has no {kind} window (rateLimits: "
                         f"{sorted(str(k) for k in limits)})")
    used = limits[kind].get("percentUsed")
    if isinstance(used, bool) or not isinstance(used, (int, float)):
        raise PacerError(f"usage file's {kind} percentUsed is {used!r}, not a number")
    resets_at = limits[kind].get("resetsAt")
    if not resets_at and kind == "seven_day":
        raise PacerError(f"usage file's {kind} window has no resetsAt; "
                         "the pace line cannot be placed")
    return Window(kind, float(used),
                  _parse_time(resets_at, f"{kind} resetsAt") if resets_at else None)


def load_reading(path: Path, now: datetime, max_age: timedelta = MAX_AGE) -> Reading:
    path = Path(path)
    if not path.is_file():
        raise PacerError(f"no usage file at {path}")
    reading = parse_reading(path.read_text(encoding="utf-8"))
    age = now - reading.read_at
    if -age > MAX_AHEAD:
        raise PacerError(f"usage file {path} was read in the future: "
                         f"{reading.read_at.isoformat()}, {-age} after {now.isoformat()}")
    if age > max_age:
        raise PacerError(f"usage file {path} is stale: read {reading.read_at.isoformat()}, "
                         f"{age} before {now.isoformat()}")
    return reading


def elapsed_fraction(resets_at: datetime, now: datetime, length: timedelta) -> float:
    """The fraction of a fixed-length window elapsed, counted back from its reset."""
    return min(1.0, max(0.0, 1 - (resets_at - now) / length))


def plan_tick(reading: Reading, cfg: PacerConfig, now: datetime) -> Tick:
    weekly = reading.seven_day
    elapsed = elapsed_fraction(weekly.resets_at, now, WEEK)
    reserve = cfg.reserve_percent * (1 - elapsed)
    allowance = 100 * elapsed - weekly.percent_used - reserve
    headroom = cfg.session_ceiling_percent - reading.five_hour.percent_used
    bound = min(allowance / cfg.points_per_call,
                headroom / cfg.session_points_per_call,
                cfg.max_calls_per_tick)
    calls = max(0, math.floor(bound + _EPSILON))
    return Tick(weekly.percent_used, 100 * elapsed, reserve, reading.five_hour.percent_used,
                allowance, calls)


def decide(calls: int, cfg: PacerConfig) -> str:
    return "run" if calls >= cfg.min_calls else "skip"


def probe_command(cfg: PacerConfig, mod_dir: Path) -> list[str]:
    return ["claude", "-p", "--model", cfg.probe_model, "--plugin-dir", str(mod_dir),
            "--no-session-persistence", "--output-format", "json", "ok"]


def run_command(deck: Path, calls: int) -> list[str]:
    return ["uv", "run", "thai-syllabus", "run", "--deck", str(deck), "--cycles", "1",
            "--judge-asks", str(calls)]


def log_line(now: datetime, tick: Tick, action: str) -> str:
    return (f"{now.isoformat(timespec='seconds')} weekly {tick.weekly_used:.1f}% used"
            f" / {tick.weekly_elapsed:.1f}% elapsed · reserve {tick.reserve:.2f}"
            f" · 5h {tick.session_used:.1f}% used"
            f" · allowance {tick.allowance:.2f} pts · M {tick.calls} · {action}")


def _subprocess_probe(cmd: list[str], *, env: dict, cwd: Path, timeout: float) -> ProbeResult:
    try:
        done = subprocess.run(cmd, env=env, cwd=cwd, timeout=timeout, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return ProbeResult(124, "", f"timed out after {timeout}s")
    return ProbeResult(done.returncode, done.stdout, done.stderr)


def _probe_failure(result: ProbeResult) -> str:
    """The headless JSON's `result` carries the failure reason; stderr the rest."""
    parts = [f"usage probe exited {result.returncode}"]
    try:
        reason = json.loads(result.stdout).get("result")
    except (json.JSONDecodeError, AttributeError):
        reason = result.stdout.strip()[-500:] or None
    if reason:
        parts.append(f"result: {reason}")
    if result.stderr.strip():
        parts.append(f"stderr: {result.stderr.strip()[-500:]}")
    return "; ".join(parts)


def _subprocess_cycle(cmd: list[str], *, cwd: Path) -> int:
    sys.stdout.flush()
    return subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL).returncode


def probe(cfg: PacerConfig, now: datetime, runner: ProbeRunner) -> Reading:
    """One trivial turn with the usage-probe mod loaded, in a scratch
    directory so no project settings or hooks join it."""
    with tempfile.TemporaryDirectory(prefix="quota-pacer-") as scratch:
        out = Path(scratch) / "usage.json"
        env = {**os.environ, USAGE_ENV: str(out)}
        command = probe_command(cfg, MOD_DIR)
        try:
            result = runner(command, env=env, cwd=Path(scratch),
                            timeout=cfg.probe_timeout_seconds)
        except OSError as exc:
            raise PacerError(f"usage probe could not start {command[0]}: {exc}") from None
        if result.returncode != 0:
            raise PacerError(_probe_failure(result))
        if not out.is_file():
            raise PacerError(f"no usage file at {out} after the probe exited 0 ({NO_FILE_HINT})")
        return load_reading(out, now)


def pacer_keys_set(providers_yaml: Path) -> frozenset[str]:
    """The keys the deck's `pacer` block sets, as written: a measured figure
    is one the curator wrote, whatever its value."""
    block = (yaml.safe_load(providers_yaml.read_text(encoding="utf-8")) or {}).get("pacer")
    return frozenset(block) if isinstance(block, dict) else frozenset()


def check_config(providers: ProvidersConfig, require_measured: bool,
                 keys_set: frozenset[str] = frozenset()) -> PacerConfig:
    """The deck's settings the pacer can run under, or PacerError."""
    if providers.judge.transport != "cli":
        raise PacerError(f"providers.judge.transport is {providers.judge.transport!r}; the "
                         "pacer spends subscription quota and runs only a 'cli' judge")
    cfg = providers.pacer
    if cfg.min_calls > cfg.max_calls_per_tick:
        raise PacerError(f"providers.pacer.min_calls ({cfg.min_calls}) exceeds "
                         f"max_calls_per_tick ({cfg.max_calls_per_tick}): no tick could run")
    for name in ("points_per_call", "session_points_per_call"):
        value = getattr(cfg, name)
        if value <= 0:
            raise PacerError(f"providers.pacer.{name}: {value!r} must be positive")
        if require_measured and name not in keys_set:
            raise PacerError(f"providers.pacer.{name} is unset (placeholder {value!r}); "
                             "set the measured figure (--require-measured)")
    return cfg


def main(argv: list[str] | None = None, *, probe_runner: ProbeRunner = _subprocess_probe,
         cycle_runner: CycleRunner = _subprocess_cycle) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deck", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now", help="ISO 8601 time to decide at (default: now)")
    parser.add_argument("--usage-file", type=Path, help="read this probe file instead of probing")
    parser.add_argument("--require-measured", action="store_true",
                        help="refuse per-call figures still at their placeholder defaults")
    args = parser.parse_args(argv)
    deck = args.deck.expanduser().resolve()

    try:
        now = _parse_time(args.now, "--now") if args.now else datetime.now().astimezone()
        providers_yaml = deck / "curated" / "providers.yaml"
        try:
            providers = load_providers_config(providers_yaml)
        except CuratedValidationError as exc:
            raise PacerError(str(exc)) from None
        cfg = check_config(providers, args.require_measured, pacer_keys_set(providers_yaml))
        reading = (load_reading(args.usage_file, now) if args.usage_file
                   else probe(cfg, now, probe_runner))
    except PacerError as exc:
        print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} error: {exc}",
              file=sys.stderr, flush=True)
        return 1

    tick = plan_tick(reading, cfg, now)
    action = decide(tick.calls, cfg)
    if action == "skip":
        print(log_line(now, tick, "skip"), flush=True)
        return 3
    command = run_command(deck, tick.calls)
    if args.dry_run:
        print(log_line(now, tick, "dry-run"), flush=True)
        print(" ".join(command), flush=True)
        return 0
    print(log_line(now, tick, "run"), flush=True)
    return cycle_runner(command, cwd=REPO)


if __name__ == "__main__":
    sys.exit(main())
