import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thai-deck-gen")
    p.add_subparsers(dest="command", required=True)
    return p

def main(argv=None) -> int:
    build_parser().parse_args(argv)
    return 0
