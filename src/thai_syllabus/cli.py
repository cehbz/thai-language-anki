"""thai-syllabus: one entry point over the pipeline.

    thai-syllabus migrate  --old-deck DIR --old-data DIR --new-root DIR
    thai-syllabus review   --deck DIR [--port 8877]
    thai-syllabus import   --deck DIR --collection PATH
    thai-syllabus compile  --deck DIR --out PATH [--force]
    thai-syllabus run      --deck DIR [--backend-cap NAME=N ...]
                          [--cycles N] [--spend-cap USD] [--poll-seconds S]
    thai-syllabus restore  --deck DIR

Each command wires itself through wiring.py: load_syllabus() for the
Syllabus, build_sourcing() for run()'s Sourcing ctx, both from the deck's
own curated/providers.yaml and rulebook.yaml.

Exit codes: 0 done, 1 refused/incomplete (`compile` hit a closed gate;
`run` could not reach the judge), 2 no subcommand matched.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from . import anki_import, migrate as migrate_mod, record, reviewserver
from .assessor import JudgeUnreachable
from .compile import GateRefusal, compile_syllabus
from .curated import load_providers_config
from .run import Budget, RunReport
from .run import run as run_pipeline
from .safety import SafetyCheckFailed, restore, writing_command
from .wiring import build_sourcing, default_budgets, load_derivations


def _providers_config_path(deck: Path) -> Path:
    return deck / "curated" / "providers.yaml"


def _cmd_compile(args: argparse.Namespace) -> int:
    derivations = load_derivations(args.deck)
    try:
        result = compile_syllabus(derivations.syllabus, derivations.db, derivations.media_store,
                                  args.out, force=args.force,
                                  current_rubric=derivations.current_rubric,
                                  prior=derivations.prior,
                                  provenance_source=derivations.provenance_source)
    except GateRefusal as e:
        print(f"compile refused: gate is closed ({e.blocking} finding(s)); "
             f"pass --force to compile anyway")
        for f in e.report.findings:
            print(f"  {f.rule}: {f.evidence} (note {f.note_id})")
        return 1

    report = result.report
    print(f"compile_id={report.compile_id} out_path={report.out_path}")
    print(f"notes_written={report.notes_written} cards_written={report.cards_written} "
         f"dropped={len(report.dropped)}")
    if report.forced:
        print(f"forced past a closed gate ({len(report.warnings)} warning(s)):")
        for w in report.warnings:
            print(f"  {w}")
    if report.dropped:
        print("dropped cards:")
        for d in report.dropped:
            print(f"  {d.family}/{d.kind} {d.subject}: {d.reason}")
    return 0


def _parse_backend_cap(raw: str) -> tuple[str, int]:
    name, sep, value = raw.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"--backend-cap expects NAME=N, got {raw!r}")
    try:
        return name, int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--backend-cap NAME=N: N must be an integer, got {value!r}") from None


def _min_one(flag: str) -> Callable[[str], int]:
    """An argparse `type=` for an int flag that refuses anything below 1
    (e.g. --cycles, --poll-seconds): argparse turns an ArgumentTypeError
    from a type function into a normal usage error (exit 2), the same as
    an unparseable value.
    """
    def parse(raw: str) -> int:
        value = int(raw)
        if value < 1:
            raise argparse.ArgumentTypeError(f"--{flag} must be at least 1, got {value}")
        return value
    return parse


def _print_run_report(cycle: int, report: RunReport) -> None:
    """One run's report block (spec 3 section 7), prefixed with the
    cycle it belongs to -- `cycle=1` for a single-cycle (default) run.
    """
    print(f"cycle={cycle} attempted={report.attempted} improved={report.improved} "
         f"exhausted={report.exhausted} available={report.available} "
         f"pending={report.pending} sentences_adopted={report.sentences_adopted} "
         f"drafted={report.drafted} excluded={report.excluded} "
         f"unserved={report.unserved} budgeted={report.budgeted} "
         f"deferred={report.deferred} preferences={report.preferences} "
         f"unreachable={report.unreachable} "
         f"batch_id={report.batch_id}")
    for name, spend in sorted(report.spend.items()):
        print(f"  {name}: asks={spend.asks} cost={spend.cost:.4f}")
    for name, count in sorted(report.source_failures.items()):
        print(f"  source_failures: {name}={count}")


def _cmd_run(args: argparse.Namespace, *,
            sleep: Callable[[float], None] = time.sleep) -> int:
    """Wires a Sourcing ctx through wiring.build_sourcing, then layers
    --backend-cap overrides onto default_budgets before running.

    Loops run_pipeline up to --cycles times (spec 3 section 7: the
    two-run cycle -- at most one batch outstanding at a time). Between
    cycles this waits for the batch the cycle just submitted to end, so
    the next cycle's own resolve can see it; it never waits after the
    last cycle requested. A cycle stops the loop early when the judge
    was unreachable -- whether run_pipeline said so, or the wait between
    cycles found the batch transport itself unreachable (exit 1, same as
    a single run today) -- when the report says there is nothing left to
    do (no batch out, nothing adopted or improved), or when --spend-cap
    is set and the judge's and tts's own cost on record has reached it.
    """
    with writing_command(args.deck, "run"):
        cfg = load_providers_config(_providers_config_path(args.deck))
        ctx = build_sourcing(args.deck, cfg)
        budgets = dict(default_budgets(cfg))
        for raw in args.backend_cap:
            name, max_asks = _parse_backend_cap(raw)
            budgets[name] = Budget(max_asks=max_asks)

        for cycle in range(1, args.cycles + 1):
            report = run_pipeline(ctx, budgets)
            _print_run_report(cycle, report)
            # A run that could not reach the judge exits non-zero, so a
            # script or a cron job sees the difference from "nothing left
            # to do".
            if report.unreachable:
                print("run: the judge is unreachable; stopped early", file=sys.stderr)
                return 1
            if (report.batch_id is None and report.sentences_adopted == 0
                    and report.improved == 0):
                return 0  # nothing left to do
            if args.spend_cap is not None:
                spent = (record.cost_since(ctx.db, "assess", "judge", 0)
                        + record.cost_since(ctx.db, "provide", "tts", 0))
                if spent >= args.spend_cap:
                    print("spend cap reached")
                    return 0
            if cycle < args.cycles and report.batch_id is not None:
                try:
                    while ctx.assessor.batch_status(report.batch_id) != "ended":
                        sleep(args.poll_seconds)
                except JudgeUnreachable:
                    print("run: the judge is unreachable; stopped early", file=sys.stderr)
                    return 1
        return 0


def main(argv: list[str] | None = None, *,
        sleep: Callable[[float], None] = time.sleep) -> int:
    # One logging configuration, at the process entry point: attempts.py
    # and assessor.py report a dead judge, an unusable candidate and a
    # dropped question at WARNING, and with nothing configured those go
    # nowhere. Failing undetectably is a bug.
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)

    parser = argparse.ArgumentParser(prog="thai-syllabus")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("migrate", help="one-shot migration from the old deck layout")
    p.add_argument("--old-deck", type=Path, required=True)
    p.add_argument("--old-data", type=Path, required=True)
    p.add_argument("--new-root", type=Path, required=True)

    p = sub.add_parser("review", help="feedback screen (proof gallery + question session)")
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--port", type=int, default=8877)

    p = sub.add_parser("import", help="revlog, flags, and ReviewNote harvest from Anki")
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)

    p = sub.add_parser("compile", help="translate a Syllabus into an Anki .apkg (spec 4)")
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true",
                   help="compile past a closed gate anyway (spec 4 section 2)")

    p = sub.add_parser("run", help="the batch sourcing run (spec 3 section 4)")
    p.add_argument("--deck", type=Path, required=True)
    p.add_argument("--backend-cap", action="append", default=[], metavar="NAME=N",
                   help="cap NAME's asks at N per day, measured from the record "
                        "(spend since local midnight plus this run's own), "
                        "e.g. --backend-cap forvo=100 (repeatable)")
    p.add_argument("--cycles", type=_min_one("cycles"), default=1,
                   help="repeat resolve/attempt/submit this many times, waiting "
                        "between cycles for the batch just submitted to end "
                        "(spec 3 section 7); default 1, today's single-run behavior")
    p.add_argument("--spend-cap", type=float, default=None, metavar="USD",
                   help="stop cycling once the judge's own cost (port assess) "
                        "plus tts's own cost (port provide) on record reaches this")
    p.add_argument("--poll-seconds", type=_min_one("poll-seconds"), default=300,
                   help="how long to sleep between polls of an outstanding "
                        "batch's status")

    p = sub.add_parser(
        "restore",
        help="put the last snapshot and the pre-command curated state back "
             "(spec 2 section 6)")
    p.add_argument("--deck", type=Path, required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "migrate":
            with writing_command(args.new_root, "migrate") as guard:
                report = migrate_mod.migrate(args.old_deck, args.old_data, args.new_root)
                guard.removed(
                    "sentences",
                    [u.identity for u in report.unmigratable if u.source == "sentences"])
            print(report.summary() if hasattr(report, "summary") else report)
            return 0
        if args.command == "review":
            with writing_command(args.deck, "review"):
                return reviewserver.main(["--deck", str(args.deck), "--port", str(args.port)])
        if args.command == "import":
            with writing_command(args.deck, "import"):
                derivations = load_derivations(args.deck)
                report = anki_import.import_collection(
                    args.collection, derivations.db, current_rubric=derivations.current_rubric,
                    prior=derivations.prior, provenance_source=derivations.provenance_source)
                print(report)
                return 0
        if args.command == "compile":
            return _cmd_compile(args)
        if args.command == "run":
            return _cmd_run(args, sleep=sleep)
        if args.command == "restore":
            try:
                report = restore(args.deck)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 1
            backup = args.deck / "backup" / "syllabus.db"
            print(f"restored syllabus.db from {backup} ({report.snapshot_mtime}); "
                 f"curated/ at {report.commit}; replaced db parked at {report.parked}")
            return 0
        return 2
    except SafetyCheckFailed as e:
        print("safety check failed:", file=sys.stderr)
        for failure in e.failures:
            print(f"  {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
