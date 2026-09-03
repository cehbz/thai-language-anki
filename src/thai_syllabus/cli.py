"""thai-syllabus: one entry point over the redesigned pipeline.

    thai-syllabus migrate --old-deck DIR --old-data DIR --new-root DIR
    thai-syllabus review  --deck DIR [--port 8877]
    thai-syllabus import  --deck DIR --collection PATH

compile and the sourcing run stay library-level until their configs
settle (compile_syllabus needs a wired Syllabus loader; run() needs
providers.yaml-driven backend construction) -- both exist and are
tested; the subcommands land with that wiring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import anki_import, migrate as migrate_mod, reviewserver
from .store import SyllabusDb


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
    return 2


if __name__ == "__main__":
    sys.exit(main())
