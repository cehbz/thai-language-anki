"""tools/quota_pacer.py: the pace-line arithmetic, the probe file, the
commands, the log line, the re-import, and main() with the probe, the
cycle, the compile and AnkiConnect injected. No test runs `claude` or
`thai-syllabus` or reaches Anki."""
import json
import sqlite3
import sys
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import quota_pacer as qp  # noqa: E402

from thai_syllabus.cachekeys import RunReportKey  # noqa: E402
from thai_syllabus.curated import PacerConfig  # noqa: E402
from thai_syllabus.store import SyllabusDb  # noqa: E402

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
CFG = PacerConfig()  # reserve 15, ceiling 50, max 40, min 5, 0.05 / 0.5 per call


def reading(*, weekly_used, weekly_resets_in, session_used=2.0,
            session_resets_in=timedelta(hours=3), read_at=NOW):
    return qp.Reading(
        read_at=read_at,
        five_hour=qp.Window("five_hour", session_used, read_at + session_resets_in),
        seven_day=qp.Window("seven_day", weekly_used, read_at + weekly_resets_in))


MISSING = object()


def probe_json(*, read_at="2026-10-03T11:58:00.000Z", weekly=52, session=2,
               kinds=("five_hour", "seven_day"), weekly_reset="2026-10-06T11:00:00.000Z",
               session_reset="2026-10-03T14:40:00.000Z"):
    limits = []
    if "five_hour" in kinds:
        window = {"kind": "five_hour"}
        if session is not MISSING:
            window["percentUsed"] = session
        if session_reset is not None:
            window["resetsAt"] = session_reset
        limits.append(window)
    if "seven_day" in kinds:
        window = {"kind": "seven_day"}
        if weekly is not MISSING:
            window["percentUsed"] = weekly
        if weekly_reset is not None:
            window["resetsAt"] = weekly_reset
        limits.append(window)
    return json.dumps({"read_at": read_at, "rateLimits": limits,
                       "context": {"tokens": 26619, "window": 200000, "percent": 13},
                       "cost": {"usd": 0.0275706}})


# --- the window's elapsed fraction ------------------------------------------

def test_weekly_elapsed_fraction_counts_back_from_the_reset():
    assert qp.elapsed_fraction(NOW + timedelta(days=3, hours=12), NOW,
                               qp.WEEK) == pytest.approx(0.5)


def test_five_hour_elapsed_fraction_likewise():
    assert qp.elapsed_fraction(NOW + timedelta(hours=1), NOW,
                               qp.FIVE_HOURS) == pytest.approx(0.8)


def test_elapsed_fraction_is_clamped_to_the_window():
    assert qp.elapsed_fraction(NOW - timedelta(minutes=5), NOW, qp.WEEK) == 1.0
    assert qp.elapsed_fraction(NOW + timedelta(days=8), NOW, qp.WEEK) == 0.0


# --- the tick's arithmetic ---------------------------------------------------
# allowance = elapsed% - used% - reserve x (1 - elapsed fraction): the reserve
# is whole just after the reset and gone at it, so the week ends spent.

def test_the_reserve_tapers_to_nothing_at_the_reset():
    mid = qp.plan_tick(reading(weekly_used=20, weekly_resets_in=timedelta(days=3.5)), CFG, NOW)
    assert mid.reserve == pytest.approx(7.5)
    start = qp.plan_tick(reading(weekly_used=0, weekly_resets_in=qp.WEEK), CFG, NOW)
    assert start.reserve == pytest.approx(15.0)
    end = qp.plan_tick(reading(weekly_used=0, weekly_resets_in=timedelta(0)), CFG, NOW)
    assert end.reserve == pytest.approx(0.0)


def test_under_the_pace_line_the_allowance_is_pace_less_used_less_tapered_reserve():
    tick = qp.plan_tick(reading(weekly_used=20, weekly_resets_in=timedelta(days=3.5)),
                        CFG, NOW)
    assert tick.weekly_elapsed == pytest.approx(50.0)
    assert tick.allowance == pytest.approx(22.5)   # 50 - 20 - 15 x 0.5


def test_over_the_pace_line_nothing_is_released():
    tick = qp.plan_tick(reading(weekly_used=45, weekly_resets_in=timedelta(days=3.5)),
                        CFG, NOW)
    assert tick.allowance == pytest.approx(-2.5)
    assert tick.calls == 0


def test_an_hour_after_the_reset_an_unused_week_waits_behind_the_reserve():
    r = reading(weekly_used=0, weekly_resets_in=qp.WEEK - timedelta(hours=1))
    tick = qp.plan_tick(r, CFG, NOW)
    assert tick.allowance == pytest.approx(100 / 168 - 15 * (1 - 1 / 168))
    assert qp.decide(tick.calls, CFG) == "skip"


def test_half_an_hour_before_the_reset_the_last_points_are_spent():
    r = reading(weekly_used=84.5, weekly_resets_in=timedelta(minutes=30))
    tick = qp.plan_tick(r, CFG, NOW)
    elapsed = 1 - 0.5 / 168
    assert tick.allowance == pytest.approx(100 * elapsed - 84.5 - 15 * (1 - elapsed))
    assert tick.calls == 40
    assert qp.decide(tick.calls, CFG) == "run"


def test_the_reserve_keeps_a_week_just_under_the_line_idle():
    r = reading(weekly_used=42.4, weekly_resets_in=timedelta(days=3.5))
    assert qp.plan_tick(r, CFG, NOW).calls == 2          # 0.1 pts / 0.05
    no_reserve = PacerConfig(reserve_percent=0)
    assert qp.plan_tick(r, no_reserve, NOW).calls == 40  # the per-tick cap


def test_near_the_reset_unspent_allowance_carries_over():
    tick = qp.plan_tick(reading(weekly_used=52, weekly_resets_in=timedelta(hours=2)),
                        CFG, NOW)
    elapsed = 1 - 2 / 168
    assert tick.allowance == pytest.approx(100 * elapsed - 52 - 15 * (1 - elapsed))
    assert tick.allowance > 40
    assert tick.calls == 40


def test_the_five_hour_ceiling_binds():
    r = reading(weekly_used=20, weekly_resets_in=timedelta(days=3.5), session_used=45)
    assert qp.plan_tick(r, CFG, NOW).calls == 10         # (50 - 45) / 0.5


def test_the_five_hour_window_at_its_ceiling_releases_nothing():
    r = reading(weekly_used=20, weekly_resets_in=timedelta(days=3.5), session_used=55)
    assert qp.plan_tick(r, CFG, NOW).calls == 0


def test_the_per_tick_cap_binds():
    r = reading(weekly_used=0, weekly_resets_in=timedelta(days=3.5), session_used=0)
    assert qp.plan_tick(r, PacerConfig(max_calls_per_tick=7), NOW).calls == 7


def test_allowance_converts_by_points_per_call():
    r = reading(weekly_used=27.5, weekly_resets_in=timedelta(days=3.5), session_used=0)
    cfg = PacerConfig(points_per_call=1.0, max_calls_per_tick=1000)
    assert qp.plan_tick(r, cfg, NOW).calls == 15          # 15 pts / 1.0


def test_a_whole_quotient_is_not_lost_to_float_rounding():
    # 15 / 0.05 is 299.99999999999994 in binary floating point
    r = reading(weekly_used=27.5, weekly_resets_in=timedelta(days=3.5), session_used=0)
    cfg = PacerConfig(max_calls_per_tick=1000, session_points_per_call=0.01)
    assert qp.plan_tick(r, cfg, NOW).calls == 300


def test_min_calls_floor_decides_the_action():
    assert qp.decide(4, CFG) == "skip"
    assert qp.decide(5, CFG) == "run"


# --- the probe's file ---------------------------------------------------------

def test_the_probe_file_parses_both_windows():
    r = qp.parse_reading(probe_json())
    assert r.read_at == datetime(2026, 10, 3, 11, 58, tzinfo=timezone.utc)
    assert r.seven_day == qp.Window("seven_day", 52.0,
                                    datetime(2026, 10, 6, 11, 0, tzinfo=timezone.utc))
    assert r.five_hour.percent_used == 2.0


@pytest.mark.parametrize("kinds,missing", [(("five_hour",), "seven_day"),
                                           (("seven_day",), "five_hour"),
                                           ((), "five_hour")])
def test_a_missing_window_is_refused(kinds, missing):
    with pytest.raises(qp.PacerError, match=missing):
        qp.parse_reading(probe_json(kinds=kinds))


def test_a_weekly_window_without_a_reset_is_refused():
    with pytest.raises(qp.PacerError, match="seven_day.*resetsAt"):
        qp.parse_reading(probe_json(weekly_reset=None))


def test_a_five_hour_window_without_a_reset_is_read():
    assert qp.parse_reading(probe_json(session_reset=None)).five_hour.resets_at is None


@pytest.mark.parametrize("value", [MISSING, None, "52", True])
@pytest.mark.parametrize("window", ["seven_day", "five_hour"])
def test_a_percent_used_that_is_not_a_number_is_refused(window, value):
    key = "weekly" if window == "seven_day" else "session"
    with pytest.raises(qp.PacerError, match=f"{window}.*percentUsed"):
        qp.parse_reading(probe_json(**{key: value}))


def test_a_file_that_is_not_json_is_refused():
    with pytest.raises(qp.PacerError, match="JSON"):
        qp.parse_reading("not json")


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(qp.PacerError, match="no usage file"):
        qp.load_reading(tmp_path / "absent.json", NOW)


def test_a_stale_file_is_refused(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(probe_json(read_at="2026-10-03T11:49:00Z"))
    with pytest.raises(qp.PacerError, match="stale"):
        qp.load_reading(path, NOW)


def test_a_file_read_in_the_future_is_refused(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(probe_json(read_at="2026-10-03T12:01:01Z"))
    with pytest.raises(qp.PacerError, match="future"):
        qp.load_reading(path, NOW)


def test_a_file_read_within_a_minute_ahead_is_read(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(probe_json(read_at="2026-10-03T12:00:59Z"))
    assert qp.load_reading(path, NOW).seven_day.percent_used == 52.0


def test_a_fresh_file_is_read(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text(probe_json(read_at="2026-10-03T11:51:00Z"))
    assert qp.load_reading(path, NOW).seven_day.percent_used == 52.0


# --- the commands -------------------------------------------------------------

def test_the_probe_command_loads_the_mod_for_one_haiku_turn():
    assert qp.probe_command(CFG, Path("/r/tools/claude-mods/usage-probe")) == [
        "claude", "-p", "--model", "haiku",
        "--plugin-dir", "/r/tools/claude-mods/usage-probe",
        "--no-session-persistence", "--output-format", "json", "ok"]


def test_the_run_command_caps_the_judge_asks_at_m():
    assert qp.run_command(Path("/decks/thai"), 17) == [
        "uv", "run", "thai-syllabus", "run", "--deck", "/decks/thai",
        "--cycles", "1", "--judge-asks", "17"]


# --- the log line -------------------------------------------------------------

def test_the_log_line_carries_the_tick_numbers():
    tick = qp.plan_tick(reading(weekly_used=20, weekly_resets_in=timedelta(days=3.5),
                                session_used=2.5), CFG, NOW)
    assert qp.log_line(NOW, tick, "run") == (
        "2026-10-03T12:00:00+00:00 weekly 20.0% used / 50.0% elapsed"
        " · reserve 7.50 · 5h 2.5% used · allowance 22.50 pts · M 40 · run")


# --- main, with the probe and the cycle injected ----------------------------

ANKI_OK = {"version": {"result": 6, "error": None},
           "importPackage": {"result": True, "error": None}}
ANKI_CLOSED = {"version": ConnectionRefusedError(61, "Connection refused")}
OPEN = ["probe", "anki:version", "harvest"]      # a tick's start with Anki open


class Runner:
    """Stands in for subprocess and AnkiConnect: the probe writes `usage`
    to the file the environment names; the cycle appends `cycle_rows`
    record rows and returns `cycle_code`; the compile returns
    `compile_code` with `compile_stdout`; the harvest appends
    `harvest_reviews` study rows and returns `harvest_code` with
    `harvest_stderr`; AnkiConnect answers each action from `anki` (a dict
    body, or an exception to raise)."""

    def __init__(self, usage=None, probe_code=0, cycle_code=0, stdout="", stderr="",
                 raises=None, cycle_rows=0, compile_code=0, compile_stdout="", anki=None,
                 harvest_code=0, harvest_stderr="", harvest_reviews=0):
        self.usage, self.probe_code, self.cycle_code = usage, probe_code, cycle_code
        self.stdout, self.stderr, self.raises = stdout, stderr, raises
        self.cycle_rows = cycle_rows
        self.compile_code, self.compile_stdout = compile_code, compile_stdout
        self.anki_answers = {**ANKI_OK, **(anki or {})}
        self.harvest_code, self.harvest_stderr = harvest_code, harvest_stderr
        self.harvest_reviews = harvest_reviews
        self.calls = []
        self.usage_path = None

    def probe(self, cmd, *, env, cwd, timeout):
        self.calls.append(("probe", cmd, timeout))
        if self.raises:
            raise self.raises
        self.usage_path = Path(env[qp.USAGE_ENV])
        if self.usage is not None:
            self.usage_path.write_text(self.usage)
        return qp.ProbeResult(self.probe_code, self.stdout, self.stderr)

    def cycle(self, cmd, *, cwd):
        self.calls.append(("cycle", cmd, cwd))
        deck = Path(cmd[cmd.index("--deck") + 1])
        for _ in range(self.cycle_rows):
            add_row(deck)
        return self.cycle_code

    def harvest(self, cmd, *, cwd):
        self.calls.append(("harvest", cmd, cwd))
        deck = Path(cmd[cmd.index("--deck") + 1])
        for _ in range(self.harvest_reviews):
            add_review(deck)
        return qp.ProbeResult(self.harvest_code, "", self.harvest_stderr)

    def compile(self, cmd, *, cwd):
        self.calls.append(("compile", cmd, cwd))
        return qp.ProbeResult(self.compile_code, self.compile_stdout, "")

    def anki(self, url, body, timeout):
        request = json.loads(body)
        self.calls.append(("anki", url, request, timeout))
        answer = self.anki_answers[request["action"]]
        if isinstance(answer, BaseException):
            raise answer
        return json.dumps(answer).encode()

    def kinds(self):
        return [c[0] if c[0] != "anki" else f"anki:{c[2]['action']}" for c in self.calls]


def add_review(deck, ts=None):
    """One harvested study row; `ts` defaults past every row already there."""
    db = SyllabusDb(deck / "syllabus.db")
    db.close()
    con = sqlite3.connect(deck / "syllabus.db")
    try:
        with con:
            if ts is None:
                ts = (con.execute("select max(ts) from study").fetchone()[0] or 0) + 1
            con.execute("insert into study (family, anchor, card_kind, compile_id, ts, grade, "
                        "time_ms) values ('word', 'w1', 'listening', 'c', ?, 3, 1000)", (ts,))
    finally:
        con.close()
    return ts


def add_row(deck, port="assess", ts=None):
    """One record row, as run would append it; the db is created on first use."""
    db = SyllabusDb(deck / "syllabus.db")
    try:
        return db.append(port=port, backend="judge", key=RunReportKey(), subject="s",
                         question={}, answer={}, ts=ts)
    finally:
        db.close()


def write_deck(root, *, judge=None, pacer=None, **extra):
    (root / "curated").mkdir(exist_ok=True)
    (root / "curated" / "providers.yaml").write_text(yaml.safe_dump({
        "imgfetch_path": "/opt/bin/imgfetch", "audiofetch_path": "/opt/bin/audiofetch",
        "judge": judge or {"transport": "cli", "model": "m"},
        "pacer": pacer or {"probe_timeout_seconds": 90}, **extra}))
    return root


@pytest.fixture
def deck(tmp_path):
    return write_deck(tmp_path)


def run_main(deck, runner, *extra):
    return qp.main(["--deck", str(deck), "--now", "2026-10-03T12:00:00+00:00", *extra],
                   probe_runner=runner.probe, cycle_runner=runner.cycle,
                   compile_runner=runner.compile, harvest_runner=runner.harvest,
                   http=runner.anki)


def test_main_probes_then_runs_one_capped_cycle(deck, capsys):
    runner = Runner(usage=probe_json(weekly=20, weekly_reset="2026-10-07T00:00:00Z"),
                    cycle_code=0)
    assert run_main(deck, runner) == 0
    (_, probe_cmd, timeout), (_, cycle_cmd, cwd) = [c for c in runner.calls
                                                    if c[0] in ("probe", "cycle")]
    assert probe_cmd[:4] == ["claude", "-p", "--model", "haiku"]
    assert timeout == 90
    assert cycle_cmd == qp.run_command(deck, 40)
    assert cwd == qp.REPO
    assert capsys.readouterr().out.splitlines()[0].endswith("· M 40 · run")


def test_main_exits_with_the_cycles_code(deck):
    runner = Runner(usage=probe_json(weekly=20, weekly_reset="2026-10-07T00:00:00Z"),
                    cycle_code=1)
    assert run_main(deck, runner) == 1


def test_main_with_nothing_to_do_logs_and_exits_3(deck, capsys):
    runner = Runner(usage=probe_json(weekly=90))
    assert run_main(deck, runner) == 3
    assert [c[0] for c in runner.calls] == ["probe", "anki", "harvest"]
    assert capsys.readouterr().out.splitlines()[0].endswith("· M 0 · skip")


def error_line(capsys):
    captured = capsys.readouterr()
    assert captured.out == ""
    (line,) = captured.err.splitlines()
    assert qp.datetime.fromisoformat(line.split(" ", 1)[0])
    assert line.split(" ", 2)[1] == "error:"
    return line


def test_main_refuses_when_the_probe_writes_nothing(deck, capsys):
    runner = Runner(usage=None)
    assert run_main(deck, runner) == 1
    line = error_line(capsys)
    assert "no usage file" in line and "mod not loaded?" in line and "plugin_errors" in line
    assert [c[0] for c in runner.calls] == ["probe"]
    assert not runner.usage_path.parent.exists()   # the temp directory is gone


def test_main_refuses_when_the_probe_fails_quoting_result_and_stderr(deck, capsys):
    stdout = json.dumps({"type": "result", "is_error": True, "result": "Not logged in"})
    runner = Runner(usage=probe_json(), probe_code=1, stdout=stdout, stderr="boom")
    assert run_main(deck, runner) == 1
    line = error_line(capsys)
    assert "exited 1" in line and "Not logged in" in line and "boom" in line
    assert [c[0] for c in runner.calls] == ["probe"]


def test_main_refuses_when_claude_cannot_start(deck, capsys):
    runner = Runner(raises=FileNotFoundError(2, "No such file or directory", "claude"))
    assert run_main(deck, runner) == 1
    assert "claude" in error_line(capsys)


def test_the_real_probe_runner_reports_claude_missing_from_path(deck, tmp_path, monkeypatch,
                                                              capsys):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert qp.main(["--deck", str(deck)], cycle_runner=Runner().cycle) == 1
    assert "claude" in error_line(capsys)


def test_the_real_probe_runner_turns_a_timeout_into_an_exit_code(tmp_path):
    result = qp._subprocess_probe(["sleep", "5"], env={"PATH": "/bin:/usr/bin"},
                                  cwd=tmp_path, timeout=0.2)
    assert result.returncode != 0 and "timed out" in result.stderr


def test_main_refuses_a_judge_not_on_the_cli_transport(tmp_path, capsys):
    deck = write_deck(tmp_path, judge={"transport": "batch", "model": "m",
                                       "price_per_mtok": {"input": 1, "output": 5}},
                      secrets={"anthropic": "op://vault/anthropic/key"})
    runner = Runner(usage=probe_json(weekly=20, weekly_reset="2026-10-07T00:00:00Z"))
    assert run_main(deck, runner) == 1
    assert "judge.transport" in error_line(capsys)
    assert runner.calls == []


def test_main_refuses_min_calls_above_the_per_tick_cap(tmp_path, capsys):
    deck = write_deck(tmp_path, pacer={"min_calls": 50, "max_calls_per_tick": 40})
    assert run_main(deck, Runner(usage=probe_json())) == 1
    assert "min_calls" in error_line(capsys)


@pytest.mark.parametrize("pacer,named", [
    ({}, "points_per_call"),
    ({"points_per_call": 0.03}, "session_points_per_call"),
    ({"session_points_per_call": 0.4}, "points_per_call")])
def test_require_measured_refuses_an_unset_per_call_figure(tmp_path, capsys, pacer, named):
    deck = write_deck(tmp_path, pacer=pacer or {"min_calls": 5})
    runner = Runner(usage=probe_json(weekly=20, weekly_reset="2026-10-07T00:00:00Z"))
    assert run_main(deck, runner, "--require-measured") == 1
    assert named in error_line(capsys)
    assert runner.calls == []


def test_require_measured_accepts_a_measured_figure_equal_to_the_placeholder(tmp_path):
    deck = write_deck(tmp_path, pacer={"points_per_call": 0.05,
                                       "session_points_per_call": 0.5})
    runner = Runner(usage=probe_json(weekly=20, weekly_reset="2026-10-07T00:00:00Z"))
    assert run_main(deck, runner, "--require-measured") == 0
    assert [c[0] for c in runner.calls] == ["probe", "anki", "harvest", "cycle"]


def test_require_measured_passes_measured_figures(tmp_path):
    deck = write_deck(tmp_path, pacer={"points_per_call": 0.03,
                                       "session_points_per_call": 0.4})
    runner = Runner(usage=probe_json(weekly=20, weekly_reset="2026-10-07T00:00:00Z"))
    assert run_main(deck, runner, "--require-measured") == 0
    assert [c[0] for c in runner.calls] == ["probe", "anki", "harvest", "cycle"]


def test_dry_run_with_a_usage_file_neither_probes_nor_runs(deck, tmp_path, capsys):
    usage = tmp_path / "usage.json"
    usage.write_text(probe_json(weekly=20, weekly_reset="2026-10-07T00:00:00Z"))
    runner = Runner()
    assert run_main(deck, runner, "--dry-run", "--usage-file", str(usage)) == 0
    assert runner.calls == []
    out = capsys.readouterr().out
    assert "· M 40 · dry-run" in out
    assert "--judge-asks 40" in out


def test_a_stale_usage_file_is_refused_by_main(deck, tmp_path, capsys):
    usage = tmp_path / "usage.json"
    usage.write_text(probe_json(read_at="2026-10-03T11:00:00Z"))
    assert run_main(deck, Runner(), "--usage-file", str(usage)) == 1
    assert "stale" in capsys.readouterr().err


# --- the harvest and the re-import (spec 3 section 8 `pacer`) ---------------
# Each tick harvests the collection's reviews before its cycle. "Changed" is
# the record's newest cache row (run's own report rows aside) or its newest
# study row being newer than the ones the last successful import saw; both
# are kept in a state file beside the log, so a failed import stays pending
# for later ticks, run or not.

RUN_USAGE = dict(weekly=20, weekly_reset="2026-10-07T00:00:00Z")
GATE_CLOSED = ("compile refused: gate is closed (12 finding(s)); pass --force to compile "
               "anyway\n  sentence/recording-required: no recording (note n1)\n")


def test_the_record_marks_are_the_newest_cache_and_study_rows(tmp_path):
    assert qp.record_marks(tmp_path) == qp.Marks(-1, -1)     # no syllabus.db yet
    add_row(tmp_path, ts=100)
    add_row(tmp_path, port="attempt", ts=150)               # run's adoption rows count
    add_row(tmp_path, port="run", ts=200)                   # run's own report does not
    add_review(tmp_path, ts=70)
    assert qp.record_marks(tmp_path) == qp.Marks(150, 70)


def test_marks_are_newer_when_either_table_moved():
    assert qp.Marks(2, 1).newer_than(qp.Marks(1, 1))
    assert qp.Marks(1, 2).newer_than(qp.Marks(1, 1))
    assert not qp.Marks(1, 1).newer_than(qp.Marks(1, 1))


def test_the_state_file_round_trips_and_is_absent_as_never_imported(tmp_path):
    path = tmp_path / "work" / "pacer-import.json"
    assert qp.read_imported(path) == qp.Marks(-1, -1)
    qp.write_imported(path, qp.Marks(42, 7))
    assert qp.read_imported(path) == qp.Marks(42, 7)
    assert json.loads(path.read_text()) == {"cache_ts": 42, "study_ts": 7}


def test_the_harvest_command_reads_the_collection_file_or_the_open_anki():
    deck = Path("/decks/thai-ff")
    assert qp.harvest_command(deck, collection=Path("/A/User 1/collection.anki2")) == [
        "uv", "run", "thai-syllabus", "import", "--deck", "/decks/thai-ff",
        "--collection", "/A/User 1/collection.anki2"]
    assert qp.harvest_command(deck, anki_connect="http://127.0.0.1:8765") == [
        "uv", "run", "thai-syllabus", "import", "--deck", "/decks/thai-ff",
        "--anki-connect", "http://127.0.0.1:8765"]


def test_the_state_file_sits_beside_the_log():
    assert qp.state_path(Path("/decks/thai-ff")) == Path("/decks/thai-ff/work/pacer-import.json")


def test_the_compile_command_writes_the_decks_package_and_never_forces():
    deck = Path("/decks/thai-ff")
    assert qp.package_path(deck) == Path("/decks/thai-ff/thai-ff.apkg")
    assert qp.compile_command(deck) == [
        "uv", "run", "thai-syllabus", "compile", "--deck", "/decks/thai-ff",
        "--out", "/decks/thai-ff/thai-ff.apkg"]


def test_the_import_request_names_the_absolute_package():
    assert qp.import_request(Path("/decks/thai-ff/thai-ff.apkg")) == {
        "action": "importPackage", "version": 6,
        "params": {"path": "/decks/thai-ff/thai-ff.apkg"}}


def test_a_run_that_changed_the_deck_compiles_and_imports(deck, capsys):
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1)
    assert run_main(deck, runner) == 0
    assert runner.kinds() == [*OPEN, "cycle", "anki:version", "compile",
                              "anki:importPackage"]
    _, compile_cmd, cwd = runner.calls[runner.kinds().index("compile")]
    assert compile_cmd == qp.compile_command(deck) and "--force" not in compile_cmd
    assert cwd == qp.REPO
    _, url, request, timeout = runner.calls[runner.kinds().index("anki:importPackage")]
    assert url == "http://127.0.0.1:8765"
    assert request == {"action": "importPackage", "version": 6,
                       "params": {"path": str(deck / f"{deck.name}.apkg")}}
    assert Path(request["params"]["path"]).is_absolute()
    assert timeout == 120
    assert qp.read_imported(qp.state_path(deck)) == qp.record_marks(deck)
    assert "imported" in capsys.readouterr().out.splitlines()[-1]


def test_a_run_that_changed_nothing_imports_nothing(deck, capsys):
    add_row(deck)
    qp.write_imported(qp.state_path(deck), qp.record_marks(deck))
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=0)
    assert run_main(deck, runner) == 0
    assert runner.kinds() == [*OPEN, "cycle"]
    assert capsys.readouterr().out.splitlines()[-1].endswith(
        "deck unchanged since the last import; not imported")


def test_import_off_neither_harvests_nor_compiles_nor_calls_anki(tmp_path):
    deck = write_deck(tmp_path, pacer={"import": False})
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1)
    assert run_main(deck, runner) == 0
    assert runner.kinds() == ["probe", "cycle"]


def test_a_closed_gate_is_logged_and_not_imported(deck, capsys):
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1, compile_code=1,
                    compile_stdout=GATE_CLOSED)
    assert run_main(deck, runner) == 0
    assert runner.kinds() == [*OPEN, "cycle", "anki:version", "compile"]
    assert capsys.readouterr().out.splitlines()[-1].endswith(
        " gate closed: 12 finding(s); not imported")
    assert not qp.state_path(deck).exists()


def test_a_compile_that_fails_otherwise_is_logged_and_not_imported(deck, capsys):
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1, compile_code=2,
                    compile_stdout="Traceback ...\nKeyError: 'x'\n")
    assert run_main(deck, runner) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert "compile exited 2" in line and "KeyError" in line and line.endswith("not imported")
    assert not qp.state_path(deck).exists()


@pytest.mark.parametrize("refusal", [
    urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
    ConnectionRefusedError(61, "Connection refused")])
def test_anki_not_running_is_logged_and_nothing_is_compiled(deck, capsys, refusal):
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1, anki={"version": refusal})
    assert run_main(deck, runner) == 0
    assert runner.kinds() == ["probe", "anki:version", "harvest", "cycle", "anki:version"]
    assert capsys.readouterr().out.splitlines()[-1].endswith(
        " Anki not running; not imported")
    assert not qp.state_path(deck).exists()


def test_anki_closing_during_the_compile_is_anki_not_running(deck, capsys):
    refusal = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1,
                    anki={"importPackage": refusal})
    assert run_main(deck, runner) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith(
        " Anki not running; not imported")
    assert not qp.state_path(deck).exists()


@pytest.mark.parametrize("answer,shown", [
    ({"result": None, "error": "collection is not available"}, "collection is not available"),
    ({"result": False, "error": None}, "returned false"),
])
def test_an_anki_connect_error_is_logged_and_not_imported(deck, capsys, answer, shown):
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1,
                    anki={"importPackage": answer})
    assert run_main(deck, runner) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert "importPackage" in line and shown in line and line.endswith("not imported")
    assert not qp.state_path(deck).exists()


def test_a_failed_import_is_retried_by_the_next_tick_without_a_run(deck, capsys):
    refused = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1, anki=ANKI_CLOSED)
    assert run_main(deck, refused) == 0
    idle = Runner(usage=probe_json(weekly=90))        # M 0: the tick skips its cycle
    assert run_main(deck, idle) == 3
    assert idle.kinds() == [*OPEN, "anki:version", "compile", "anki:importPackage"]
    assert qp.read_imported(qp.state_path(deck)) == qp.record_marks(deck)
    again = Runner(usage=probe_json(weekly=90))
    assert run_main(deck, again) == 3
    assert again.kinds() == OPEN


def test_a_failed_cycle_attempts_no_import(deck):
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1, cycle_code=1)
    assert run_main(deck, runner) == 1
    assert runner.kinds() == [*OPEN, "cycle"]


def test_dry_run_prints_the_import_it_would_make_without_making_it(deck, tmp_path, capsys):
    add_row(deck, ts=7)
    usage = tmp_path / "usage.json"
    usage.write_text(probe_json(**RUN_USAGE))
    runner = Runner()
    assert run_main(deck, runner, "--dry-run", "--usage-file", str(usage)) == 0
    assert runner.calls == []
    out = capsys.readouterr().out
    assert "harvest (Anki open): " + " ".join(
        qp.harvest_command(deck, anki_connect="http://127.0.0.1:8765")) in out
    assert "harvest (Anki closed): " + " ".join(
        qp.harvest_command(deck, collection=PacerConfig().collection_path)) in out
    assert "import pending: record ts 7, study ts -1; last imported -1, -1" in out
    assert " ".join(qp.compile_command(deck)) in out
    assert f"POST http://127.0.0.1:8765 importPackage {deck / f'{deck.name}.apkg'}" in out
    assert not qp.state_path(deck).exists()


def test_dry_run_with_import_off_says_so(tmp_path, capsys):
    deck = write_deck(tmp_path, pacer={"import": False})
    usage = tmp_path / "usage.json"
    usage.write_text(probe_json(**RUN_USAGE))
    assert run_main(deck, Runner(), "--dry-run", "--usage-file", str(usage)) == 0
    assert "import off (pacer.import: false)" in capsys.readouterr().out


def harvest_call(runner):
    return runner.calls[runner.kinds().index("harvest")]


def test_with_anki_open_the_tick_harvests_through_anki_connect_before_its_cycle(deck):
    runner = Runner(usage=probe_json(**RUN_USAGE))
    assert run_main(deck, runner) == 0
    _, harvest_cmd, cwd = harvest_call(runner)
    assert harvest_cmd == qp.harvest_command(deck, anki_connect="http://127.0.0.1:8765")
    assert cwd == qp.REPO
    assert runner.kinds()[:4] == [*OPEN, "cycle"]


def test_with_anki_closed_the_tick_harvests_the_collection_file_and_imports_nothing(tmp_path,
                                                                                   capsys):
    deck = write_deck(tmp_path, pacer={"collection_path": "~/A/collection.anki2"})
    runner = Runner(usage=probe_json(**RUN_USAGE), cycle_rows=1, anki=ANKI_CLOSED)
    assert run_main(deck, runner) == 0
    _, harvest_cmd, _cwd = harvest_call(runner)
    assert harvest_cmd == qp.harvest_command(
        deck, collection=Path.home() / "A" / "collection.anki2")
    assert runner.kinds() == ["probe", "anki:version", "harvest", "cycle", "anki:version"]
    assert capsys.readouterr().out.splitlines()[-1].endswith(" Anki not running; not imported")
    assert not qp.state_path(deck).exists()


LOCKED = ("Traceback (most recent call last):\n  ...\n"
          "sqlite3.OperationalError: database is locked\n")


def test_a_locked_collection_is_logged_and_the_tick_still_runs(deck, capsys):
    # Anki open with AnkiConnect not answering: the file is locked
    runner = Runner(usage=probe_json(**RUN_USAGE), harvest_code=1, harvest_stderr=LOCKED,
                    cycle_rows=1, anki=ANKI_CLOSED)
    assert run_main(deck, runner) == 0
    assert runner.kinds()[:4] == ["probe", "anki:version", "harvest", "cycle"]
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].endswith(" collection locked (Anki open); reviews not harvested")


def test_a_harvest_that_fails_otherwise_is_logged_and_the_tick_continues(deck, capsys):
    runner = Runner(usage=probe_json(**RUN_USAGE), harvest_code=1,
                    harvest_stderr="FileNotFoundError: no collection\n")
    assert run_main(deck, runner) == 0
    assert runner.kinds()[:4] == [*OPEN, "cycle"]
    line = capsys.readouterr().out.splitlines()[1]
    assert "harvest exited 1" in line and "FileNotFoundError" in line
    assert line.endswith("reviews not harvested")


def test_harvested_reviews_alone_make_the_deck_changed(deck):
    add_row(deck)
    qp.write_imported(qp.state_path(deck), qp.record_marks(deck))
    runner = Runner(usage=probe_json(weekly=90), harvest_reviews=1)   # a skip tick
    assert run_main(deck, runner) == 3
    assert runner.kinds() == [*OPEN, "anki:version", "compile", "anki:importPackage"]
    assert qp.read_imported(qp.state_path(deck)).study_ts == qp.record_marks(deck).study_ts
