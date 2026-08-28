import argparse
import datetime
import json
import os
from pathlib import Path

from thai_deck_eval.model.deck import load_deck
from thai_deck_gen.compiler.build import compile_deck
from thai_deck_gen.config import load_config
from thai_deck_gen.context import build_context
from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.llm import CachedLlm, CliBackend
from thai_deck_gen.media.commission import import_commission, write_commission_batch
from thai_deck_gen.media.forvo import ForvoClient, fetch_forvo
from thai_deck_gen.media.images import fill_images, flagged_image_note_ids
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import NATIVE_TIER_FAMILIES, pending_audio, pending_images
from thai_deck_gen.media.thai1000 import audio_index, import_thai1000
from thai_deck_gen.media.tts import GoogleTts, fill_tts
from thai_deck_gen.orchestrator import EvalError, generate, run_eval
from thai_deck_gen.producers.pairs import fill_pairs
from thai_deck_gen.producers.sentences import fetch_exemplars, fill_sentences
from thai_deck_gen.producers.spelling import fill_spelling
from thai_deck_gen.producers.words import fill_words
from thai_deck_gen.report import parse_report
from thai_deck_gen.wordlist import draft_word_list

DEFAULT_DATA_DIR = Path("data")


# --- THAI_DECK_GEN_FAKE test seam: avoids pythainlp/claude CLI in generate's
# CLI path so it can be exercised without the "nlp" and "llm" extras. ---

class _FakeG2P:
    def syllables(self, word):
        return None


class _FakeTokenizer:
    def tokens(self, text):
        return text.split()


class _FakeFreq:
    def rank(self, word):
        return None


class _FakeGenLlm:
    def complete(self, producer, prompt_version, prompt):
        return ""


def _today() -> str:
    return datetime.date.today().isoformat()


def _build_ctx(deck_dir: Path, data_dir: Path):
    cfg = load_config(deck_dir)
    if os.environ.get("THAI_DECK_GEN_FAKE") == "1":
        return build_context(deck_dir, data_dir, _FakeGenLlm(), nlp=False,
                             g2p=_FakeG2P(), tokenizer=_FakeTokenizer(), freq=_FakeFreq(),
                             config=cfg)
    llm = CachedLlm(CliBackend(), deck_dir / "work" / "llm_cache.sqlite3", model=cfg.model)
    return build_context(deck_dir, data_dir, llm, nlp=True, config=cfg)


def _gaps_for(deck_dir: Path, data_dir: Path):
    last = deck_dir / ".last-report.json"
    report = (json.loads(last.read_text(encoding="utf-8")) if last.exists()
             else run_eval(deck_dir))
    return parse_report(report, data_dir / "contrasts.yaml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thai-deck-gen")
    sub = p.add_subparsers(dest="command", required=True)

    wl = sub.add_parser("wordlist")
    wl.add_argument("--out", type=Path, default=Path("data/word_list_th.yaml"))

    init = sub.add_parser("init")
    init.add_argument("dir", type=Path)
    init.add_argument("--name", required=True)
    init.add_argument("--phases", required=True,
                      help="comma-separated phases, e.g. sounds,words,sentences")

    gen = sub.add_parser("generate")
    gen.add_argument("dir", type=Path)
    gen.add_argument("--max-iterations", type=int, default=None)
    gen.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    for name in ("pairs", "spelling", "words"):
        sp = sub.add_parser(name)
        sp.add_argument("--deck", type=Path, required=True)
        sp.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    sn = sub.add_parser("sentences")
    sn_sub = sn.add_subparsers(dest="sentences_command", required=True)
    fx = sn_sub.add_parser("fetch-exemplars")
    fx.add_argument("--deck", type=Path, required=True,
                    help="deck root; exemplars are written to <deck>/work/exemplars.txt")
    fx.add_argument("--out", type=Path, default=None,
                    help="override output path (default: <deck>/work/exemplars.txt)")
    fx.add_argument("--sample-size", type=int, default=500)
    fl = sn_sub.add_parser("fill")
    fl.add_argument("--deck", type=Path, required=True)
    fl.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    au = sub.add_parser("audio")
    au_sub = au.add_subparsers(dest="audio_command", required=True)

    ff = au_sub.add_parser("fetch-forvo")
    ff.add_argument("--deck", type=Path, required=True)
    ff.add_argument("--max-speakers", type=int, default=3)

    it = au_sub.add_parser("import-thai1000")
    it.add_argument("--deck", type=Path, required=True)
    it.add_argument("--apkg", type=Path, required=True)

    ic = au_sub.add_parser("import-commission")
    ic.add_argument("--deck", type=Path, required=True)
    ic.add_argument("--recordings", type=Path, required=True)
    ic.add_argument("--batch", type=Path, required=True)
    ic.add_argument("--speaker", required=True)

    tt = au_sub.add_parser("tts")
    tt.add_argument("--deck", type=Path, required=True)

    cm = au_sub.add_parser("commission")
    cm.add_argument("--deck", type=Path, required=True)

    im = sub.add_parser("images")
    im.add_argument("--deck", type=Path, required=True)
    im.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    cp = sub.add_parser("compile")
    cp.add_argument("--deck", type=Path, required=True)
    cp.add_argument("--out", type=Path, required=True)
    cp.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    cp.add_argument("--base", type=int, default=None,
                    help="defaults to gen.yaml's sentence_base")

    return p


def _report_result(name: str, res) -> None:
    print(f"{name}: +{res.added} changed={res.changed} blocked={len(res.blocked)}")


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

    elif args.command == "init":
        deck = new_deck(args.dir, args.name, args.phases.split(","))
        write_deck(deck)
        print(f"initialized deck {args.name!r} at {args.dir}")

    elif args.command == "generate":
        deck = load_deck(args.dir)
        ctx = _build_ctx(args.dir, args.data_dir)
        if args.max_iterations is not None:
            ctx.config.max_iterations = args.max_iterations
        summaries = generate(deck, ctx)
        print(f"ran {len(summaries)} iteration(s)")

    elif args.command == "pairs":
        deck = load_deck(args.deck)
        ctx = _build_ctx(args.deck, args.data_dir)
        gaps = _gaps_for(args.deck, args.data_dir)
        res = fill_pairs(gaps, deck, ctx)
        write_deck(deck)
        _report_result("pairs", res)

    elif args.command == "spelling":
        deck = load_deck(args.deck)
        ctx = _build_ctx(args.deck, args.data_dir)
        gaps = _gaps_for(args.deck, args.data_dir)
        res = fill_spelling(gaps, deck, ctx)
        write_deck(deck)
        _report_result("spelling", res)

    elif args.command == "words":
        deck = load_deck(args.deck)
        ctx = _build_ctx(args.deck, args.data_dir)
        gaps = _gaps_for(args.deck, args.data_dir)
        res = fill_words(gaps, deck, ctx)
        write_deck(deck)
        _report_result("words", res)

    elif args.command == "sentences" and args.sentences_command == "fetch-exemplars":
        out = args.out or (args.deck / "work" / "exemplars.txt")
        count = fetch_exemplars(out, sample_size=args.sample_size)
        print(f"wrote {count} exemplar sentences to {out}")

    elif args.command == "sentences" and args.sentences_command == "fill":
        deck = load_deck(args.deck)
        ctx = _build_ctx(args.deck, args.data_dir)
        gaps = _gaps_for(args.deck, args.data_dir)
        res = fill_sentences(gaps, deck, ctx)
        write_deck(deck)
        _report_result("sentences", res)

    elif args.command == "audio" and args.audio_command == "fetch-forvo":
        deck = load_deck(args.deck)
        manifest = Manifest.load(deck.root)
        client = ForvoClient(os.environ["FORVO_API_KEY"])
        needs = [n for n in pending_audio(deck) if n.family in NATIVE_TIER_FAMILIES]
        res = fetch_forvo(needs, deck, manifest, client, _today(),
                          max_speakers=args.max_speakers)
        write_deck(deck)
        manifest.save(deck.root)
        _report_result("forvo", res)

    elif args.command == "audio" and args.audio_command == "import-thai1000":
        deck = load_deck(args.deck)
        manifest = Manifest.load(deck.root)
        index = audio_index(args.apkg)
        res = import_thai1000(pending_audio(deck), deck, manifest, index, _today())
        write_deck(deck)
        manifest.save(deck.root)
        _report_result("thai1000", res)

    elif args.command == "audio" and args.audio_command == "import-commission":
        deck = load_deck(args.deck)
        manifest = Manifest.load(deck.root)
        res = import_commission(args.recordings, args.batch, deck, manifest,
                                args.speaker, _today())
        write_deck(deck)
        manifest.save(deck.root)
        _report_result("commission", res)

    elif args.command == "audio" and args.audio_command == "tts":
        deck = load_deck(args.deck)
        manifest = Manifest.load(deck.root)
        tts = GoogleTts(os.environ["GOOGLE_TTS_API_KEY"])
        res = fill_tts(pending_audio(deck), deck, manifest, tts, _today())
        write_deck(deck)
        manifest.save(deck.root)
        _report_result("tts", res)

    elif args.command == "audio" and args.audio_command == "commission":
        deck = load_deck(args.deck)
        needs = [n for n in pending_audio(deck) if n.family in NATIVE_TIER_FAMILIES]
        path = write_commission_batch(needs, deck.root)
        if path is None:
            print("no pending native-tier audio needs; no batch written")
        else:
            print(f"wrote commission batch to {path} ({len(needs)} item(s))")

    elif args.command == "images":
        deck = load_deck(args.deck)
        ctx = _build_ctx(args.deck, args.data_dir)
        gaps = _gaps_for(args.deck, args.data_dir)
        manifest = Manifest.load(deck.root)
        flagged = flagged_image_note_ids(gaps)
        res = fill_images(pending_images(deck, flagged=flagged), gaps, deck, manifest, ctx,
                          _today())
        write_deck(deck)
        manifest.save(deck.root)
        _report_result("images", res)

    elif args.command == "compile":
        from thai_deck_eval.data_io import FileFrequencyList
        deck = load_deck(args.deck)
        manifest = Manifest.load(deck.root)
        gaps = _gaps_for(args.deck, args.data_dir)
        freq = FileFrequencyList(args.data_dir / "frequency_th.txt")
        base = args.base if args.base is not None else load_config(args.deck).sentence_base
        compile_deck(deck, manifest, args.out, freq, gaps.pair_by_note, base=base)
        print(f"compiled {args.out}")

    return 0
