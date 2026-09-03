"""thai-syllabus: one entry point over the redesigned pipeline.

    thai-syllabus migrate  --old-deck DIR --old-data DIR --new-root DIR
    thai-syllabus review   --deck DIR [--port 8877]
    thai-syllabus import   --deck DIR --collection PATH
    thai-syllabus compile  --deck DIR --out PATH [--force]
    thai-syllabus run      --deck DIR [--backend-cap NAME=N ...]

compile and run were library-level only until their configs settled
(compile_syllabus needed a wired Syllabus loader; run() needed
providers.yaml-driven backend construction) -- wiring.py is that
settling: load_syllabus() wires the Syllabus loader; build_provider/
build_assessor/build_levers/default_budgets wire run()'s backend rosters
and budgets from curated/providers.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import anki_import, migrate as migrate_mod, reviewserver
from .compile import GateRefusal, compile_syllabus
from .curated import load_providers_config
from .run import Budget
from .run import run as run_pipeline
from .store import MediaStore, SyllabusDb
from .wiring import build_assessor, build_levers, build_provider, default_budgets, load_syllabus


def _providers_config_path(deck: Path) -> Path:
    return deck / "curated" / "providers.yaml"


def _cmd_compile(args: argparse.Namespace) -> int:
    syllabus = load_syllabus(args.deck)
    db = SyllabusDb(args.deck / "syllabus.db")
    media_store = MediaStore(args.deck / "media")
    try:
        result = compile_syllabus(syllabus, db, media_store, args.out, force=args.force)
    except GateRefusal as e:
        print(f"compile refused: gate is closed ({len(e.report.findings)} finding(s)); "
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


def _cmd_run(args: argparse.Namespace) -> int:
    syllabus = load_syllabus(args.deck)
    db = SyllabusDb(args.deck / "syllabus.db")
    media_store = MediaStore(args.deck / "media")
    cfg = load_providers_config(_providers_config_path(args.deck))

    provider = build_provider(cfg, db, media_store)
    assessor = build_assessor(cfg, db)
    budgets = dict(default_budgets(cfg))
    for raw in args.backend_cap:
        name, max_asks = _parse_backend_cap(raw)
        budgets[name] = Budget(max_asks=max_asks)
    levers = build_levers(syllabus, provider, cfg)

    report = run_pipeline(syllabus, db, budgets, levers)
    print(f"attempted={report.attempted} improved={report.improved} "
         f"exhausted={report.exhausted} available={report.available}")
    for name, spend in sorted(report.spend.items()):
        print(f"  {name}: asks={spend.asks} cost={spend.cost:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
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
                   help="override a backend's max_asks budget for this run, "
                        "e.g. --backend-cap forvo=100 (repeatable)")

    args = parser.parse_args(argv)

    if args.command == "migrate":
        report = migrate_mod.migrate(args.old_deck, args.old_data, args.new_root)
        print(report.summary() if hasattr(report, "summary") else report)
        return 0
    if args.command == "review":
        return reviewserver.main(["--deck", str(args.deck), "--port", str(args.port)])
    if args.command == "import":
        db = SyllabusDb(args.deck / "syllabus.db")
        report = anki_import.import_collection(args.collection, db)
        print(report)
        return 0
    if args.command == "compile":
        return _cmd_compile(args)
    if args.command == "run":
        return _cmd_run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
