"""thai-syllabus: one entry point over the pipeline.

    thai-syllabus migrate  --old-deck DIR --old-data DIR --new-root DIR
    thai-syllabus review   --deck DIR [--port 8877]
    thai-syllabus import   --deck DIR --collection PATH
    thai-syllabus compile  --deck DIR --out PATH [--force]
    thai-syllabus run      --deck DIR [--backend-cap NAME=N ...]
                          [--cycles N] [--spend-cap USD] [--poll-seconds S]
                          [--poll-max-seconds S] [--max-wait-seconds S]
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
from .attempts import Sourcing
from .compile import GateRefusal, compile_syllabus
from .curated import load_providers_config
from .run import Budget, RunReport
from .run import run as run_pipeline
from .safety import HistoryError, SafetyCheckFailed, restore, writing_command
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


def _spend_so_far(ctx: Sourcing, start_ns: int) -> float:
    """This invocation's own cash cost so far: the judge's own cost (port
    assess) plus tts's and the illustrator's own cost (port provide, spec
    3 r34, `price_per_image`), recorded since THIS invocation started --
    the same sum --spend-cap checks (an earlier invocation's spend never
    counts). Computed unconditionally (not just when --spend-cap is set)
    so both the cycle line and the waiting line can show it.
    """
    return (record.cost_since(ctx.db, "assess", "judge", start_ns)
            + record.cost_since(ctx.db, "provide", "tts", start_ns)
            + record.cost_since(ctx.db, "provide", "illustrator", start_ns))


def _print_run_report(cycle: int, report: RunReport, spent: float) -> None:
    """One run's report block (spec 3 section 7), prefixed with the
    cycle it belongs to -- `cycle=1` for a single-cycle (default) run --
    and `spent`, this invocation's own cash cost so far (the same sum
    --spend-cap checks), visible whether or not a cap is set. Flushed:
    under `nohup` a long invocation's stdout is block-buffered, and these
    lines are the only way to watch it progress from the outside.
    """
    print(f"cycle={cycle} spent={spent:.4f} attempted={report.attempted} "
         f"improved={report.improved} "
         f"exhausted={report.exhausted} available={report.available} "
         f"pending={report.pending} sentences_adopted={report.sentences_adopted} "
         f"adjudicated={report.adjudicated} "
         f"stayed_disputed={report.stayed_disputed} "
         f"drafted={report.drafted} retired={report.retired} "
         f"covered_new={report.covered_new} "
         f"requeried={report.requeried} "
         f"adopted_graphemes={report.adopted_graphemes} "
         f"adopted_words={report.adopted_words} "
         f"adoption_skipped={report.adoption_skipped} "
         f"comments_read={report.comments_read} "
         f"comment_actions={report.comment_actions} "
         f"comment_unactionable={report.comment_unactionable} "
         f"excluded={report.excluded} "
         f"unserved={report.unserved} budgeted={report.budgeted} "
         f"deferred={report.deferred} preferences={report.preferences} "
         f"unreachable={report.unreachable} "
         f"batch_id={report.batch_id}", flush=True)
    for name, spend in sorted(report.spend.items()):
        print(f"  {name}: asks={spend.asks} cost={spend.cost:.4f}", flush=True)
    for name, count in sorted(report.source_failures.items()):
        print(f"  source_failures: {name}={count}", flush=True)


def _cmd_run(args: argparse.Namespace, *,
            sleep: Callable[[float], None] = time.sleep) -> int:
    """Wires a Sourcing ctx through wiring.build_sourcing, then layers
    --backend-cap overrides onto default_budgets before running.

    Repeats run_pipeline to quiescence (spec 3 r39): a picture need
    advances one source per pass, with verdicts only landing on a later
    pass's resolve, so one invocation loops resolve/attempt/submit rather
    than stopping after a single pass. Default --cycles is None
    (unbounded); an explicit N caps the passes. Between passes that
    submitted a batch this waits for it to end (status polled at growing
    intervals: --poll-seconds doubling each poll up to --poll-max-seconds),
    so the next pass's own resolve can see it; it never waits after the
    last cycle requested. The wait is bounded by --max-wait-seconds: once
    the sleep accumulated for that one batch reaches it, the run gives up
    (exit 1) rather than polling forever against a batch that never ends.
    The loop stops early when the judge was unreachable -- whether
    run_pipeline said so, or the wait itself found the batch transport
    unreachable (exit 1, same as a single run today) -- when a pass
    raises no batch and appends nothing to the record but its own report
    (spec 3 r39 correction: `ctx.db.newest_ts(excluding_port="run")` is
    unchanged across the pass -- nothing left to do, exit 0; a tally such
    as `attempted` measures effort, not progress, since the sentence
    attempt counts its open Targets every pass even when its cached
    answer yields only refused drafts), or when --spend-cap is set and
    the cash cost recorded since this invocation started -- the judge,
    tts and the illustrator's drawings
    (r34) -- has reached it, checked before any wait (spec 3 section 7
    governs a source's own daily budget; --spend-cap is this invocation's
    own budget, so spend from an earlier invocation never counts against
    it).
    """
    start_ns = time.time_ns()
    with writing_command(args.deck, "run") as guard:
        cfg = load_providers_config(_providers_config_path(args.deck))
        ctx = build_sourcing(args.deck, cfg)
        ctx.guard = guard
        budgets = dict(default_budgets(cfg))
        for raw in args.backend_cap:
            name, max_asks = _parse_backend_cap(raw)
            budgets[name] = Budget(max_asks=max_asks)

        cycle = 1
        while args.cycles is None or cycle <= args.cycles:
            mark = ctx.db.newest_ts(excluding_port="run")
            report = run_pipeline(ctx, budgets)
            spent = _spend_so_far(ctx, start_ns)
            _print_run_report(cycle, report, spent)
            # A run that could not reach the judge exits non-zero, so a
            # script or a cron job sees the difference from "nothing left
            # to do".
            if report.unreachable:
                print("run: the judge is unreachable; stopped early",
                     file=sys.stderr, flush=True)
                return 1
            if report.batch_id is None and ctx.db.newest_ts(excluding_port="run") == mark:
                print("nothing left to do: the pass appended nothing to the record",
                     flush=True)
                return 0
            if args.spend_cap is not None and spent >= args.spend_cap:
                print(f"spend cap reached: {spent:.4f} of {args.spend_cap:.4f} "
                     f"USD this invocation", flush=True)
                return 0
            more_cycles = args.cycles is None or cycle < args.cycles
            if more_cycles and report.batch_id is not None:
                interval = args.poll_seconds
                elapsed = 0
                try:
                    while ctx.assessor.batch_status(report.batch_id) != "ended":
                        print(f"waiting on batch {report.batch_id}; "
                             f"spent {spent:.4f} USD this invocation; "
                             f"next poll in {interval} s", flush=True)
                        sleep(interval)
                        elapsed += interval
                        if elapsed >= args.max_wait_seconds:
                            print(f"run: batch {report.batch_id} did not end within "
                                 f"{args.max_wait_seconds} s; stopped",
                                 file=sys.stderr, flush=True)
                            return 1
                        interval = min(interval * 2, args.poll_max_seconds)
                except JudgeUnreachable:
                    print("run: the judge is unreachable; stopped early",
                         file=sys.stderr, flush=True)
                    return 1
            cycle += 1
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
    p.add_argument("--cycles", type=_min_one("cycles"), default=None,
                   help="cap the number of resolve/attempt/submit passes; default "
                        "unbounded -- the invocation repeats until a pass raises "
                        "no batch and appends nothing to the record but its own "
                        "report (spec 3 r39, r39 correction)")
    p.add_argument("--spend-cap", type=float, default=None, metavar="USD",
                   help="stop cycling once the judge's own cost (port assess) "
                        "plus tts's and the illustrator's own cost (port provide) "
                        "recorded since THIS invocation started reaches this -- an "
                        "earlier invocation's spend never counts against it; a "
                        "drafter on the api transport is not counted")
    p.add_argument("--poll-seconds", type=_min_one("poll-seconds"), default=300,
                   help="how long to sleep before the first poll of an "
                        "outstanding batch's status; each later poll doubles "
                        "this, up to --poll-max-seconds")
    p.add_argument("--poll-max-seconds", type=_min_one("poll-max-seconds"), default=900,
                   help="the cap the doubling --poll-seconds backoff grows to "
                        "between polls of an outstanding batch's status; must "
                        "be at least --poll-seconds; default 900 (15 minutes)")
    p.add_argument("--max-wait-seconds", type=_min_one("max-wait-seconds"), default=21600,
                   help="give up on one batch after this much waiting (exit 1; the "
                        "batch keeps processing and the next invocation resolves "
                        "it); default 21600 (6 hours) -- the longest of eleven "
                        "measured batches took 188 minutes")

    p = sub.add_parser(
        "restore",
        help="put the last snapshot and the pre-command curated state back "
             "(spec 2 section 6)")
    p.add_argument("--deck", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "run" and args.poll_max_seconds < args.poll_seconds:
        parser.error(
            f"--poll-max-seconds ({args.poll_max_seconds}) must be at least "
            f"--poll-seconds ({args.poll_seconds})")

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
            # Not wrapped in writing_command: review appends learner rows
            # only and writes no curated file, so it takes no snapshot,
            # makes no commit, and runs alongside a writing command --
            # an append never shrinks a count (spec 2 section 6 r13).
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
    except (HistoryError, OSError) as e:
        # A writing command's git call under curated/'s history (or the db
        # snapshot/restore filesystem work around it) can fail for reasons
        # outside the pipeline's control -- git missing, a corrupt repo, a
        # permissions error. Report it in one line rather than let it
        # surface as an uncaught traceback (spec 2 section 6).
        print(f"deck safety unavailable: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
