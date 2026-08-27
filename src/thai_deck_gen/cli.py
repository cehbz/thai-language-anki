import argparse
from pathlib import Path

from thai_deck_gen.llm import CliBackend
from thai_deck_gen.producers.sentences import fetch_exemplars
from thai_deck_gen.wordlist import draft_word_list

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thai-deck-gen")
    sub = p.add_subparsers(dest="command", required=True)
    wl = sub.add_parser("wordlist")
    wl.add_argument("--out", type=Path, default=Path("data/word_list_th.yaml"))

    sn = sub.add_parser("sentences")
    sn_sub = sn.add_subparsers(dest="sentences_command", required=True)
    fx = sn_sub.add_parser("fetch-exemplars")
    fx.add_argument("--deck", type=Path, required=True,
                    help="deck root; exemplars are written to <deck>/work/exemplars.txt")
    fx.add_argument("--out", type=Path, default=None,
                    help="override output path (default: <deck>/work/exemplars.txt)")
    fx.add_argument("--sample-size", type=int, default=500)
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
    elif args.command == "sentences" and args.sentences_command == "fetch-exemplars":
        out = args.out or (args.deck / "work" / "exemplars.txt")
        count = fetch_exemplars(out, sample_size=args.sample_size)
        print(f"wrote {count} exemplar sentences to {out}")
    return 0
