import argparse
from pathlib import Path

from thai_deck_gen.llm import CliBackend
from thai_deck_gen.wordlist import draft_word_list

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thai-deck-gen")
    sub = p.add_subparsers(dest="command", required=True)
    wl = sub.add_parser("wordlist")
    wl.add_argument("--out", type=Path, default=Path("data/word_list_th.yaml"))
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "wordlist":
        warnings = []
        count = draft_word_list(CliBackend(), Path("data/categories.yaml"),
                                Path("data/frequency_th.txt"), args.out,
                                warnings=warnings)
        print(f"wrote {count} entries to {args.out}")
        for w in warnings:
            print(f"warning: {w}")
    return 0
