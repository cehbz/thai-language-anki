import argparse
from collections import Counter
import datetime
import json
import os
import sys

import yaml
from pathlib import Path

from thai_deck_eval.model.deck import load_deck
from thai_deck_eval.waivers import Waiver, image_sha, load_waivers, save_waivers
from thai_deck_gen.compiler.build import compile_deck
from thai_deck_gen.config import GenConfig, load_config
from thai_deck_gen.context import (build_context, image_judge_for, imagegen_for,
                                   load_image_query_hints)
from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.llm import ApiBackend, CachedLlm, CliBackend
from thai_deck_gen.media.commission import import_commission, write_commission_batch
from thai_deck_gen.media.forvo import ForvoClient, fetch_forvo
from thai_deck_gen.media.forvo_memo import ForvoMemo
from thai_deck_gen.media.images import (audit_picturable, fill_images,
                                        flagged_image_note_ids,
                                        search_reachable)
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
from thai_deck_eval.secrets import SecretStore
from thai_deck_gen.emphasis import load_emphasis
from thai_deck_gen.wordlist import (apply_query_proposals, draft_image_queries,
                                    draft_word_list, extend_word_list,
                                     assign_ids)

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
    return build_context(deck_dir, data_dir, _cli_llm(deck_dir, cfg), nlp=True, config=cfg)


def _drafting_backend(deck_dir: Path, cfg):
    """The `claude` CLI unless gen.yaml asks for the API.

    Subscription tokens are a flat monthly cost that mostly goes unspent;
    API tokens are incremental cash. So the CLI is the default even though
    it drags a 35K-token harness prompt into every call -- that overhead is
    free. `llm_backend: api` is for work the CLI genuinely cannot do.
    """
    store = SecretStore.from_config(cfg.secrets)
    if cfg.llm_backend == "api" and store.configured("anthropic"):
        return ApiBackend(model=cfg.model, api_key=store.get("anthropic"))
    return CliBackend(model=cfg.model)


def _cli_llm(deck_dir: Path, cfg) -> CachedLlm:
    return CachedLlm(_drafting_backend(deck_dir, cfg),
                     deck_dir / "work" / "llm_cache.sqlite3", model=cfg.model)


def _require_secret(deck_dir: Path, name: str) -> str:
    """Resolve secrets.<name> from the deck's gen.yaml, or explain what's missing."""
    store = SecretStore.from_config(load_config(deck_dir).secrets)
    value = store.get(name)
    if value is None:
        raise SystemExit(
            f"{deck_dir}/gen.yaml has no `secrets.{name}`; set it to an "
            "op://<vault>/<item>/<field> reference or a path to a 0600 key file")
    return value


def _gaps_for(deck_dir: Path, data_dir: Path):
    last = deck_dir / ".last-report.json"
    report = (json.loads(last.read_text(encoding="utf-8")) if last.exists()
             else run_eval(deck_dir))
    return parse_report(report, data_dir / "contrasts.yaml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="thai-deck-gen")
    sub = p.add_subparsers(dest="command", required=True)

    wl = sub.add_parser("wordlist")
    wl.add_argument("--extend", action="store_true",
                    help="add theme-relevant entries per data/emphasis.yaml on top of the base list")
    wl.add_argument("--out", type=Path, default=Path("data/word_list_th.yaml"))
    wl.add_argument("--assign-ids", action="store_true",
                    help="seed an id on every row lacking one, remove exact "
                         "duplicate rows, and report collisions")

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
    ff.add_argument("--limit", type=int, default=None,
                    help="cap API lookups this run (default: gen.yaml forvo_request_limit)")

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

    ap = sub.add_parser("approve",
                        help="record that a finding was reviewed and accepted")
    ap.add_argument("--deck", type=Path, required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--reason", required=True)

    st = sub.add_parser("search-terms")
    st.add_argument("--deck", type=Path, required=True)
    st.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    st.add_argument("--apply-proposals", action="store_true",
                    help="adopt the phrases the judge proposed for words whose "
                         "images it rejected (work/image_query_proposals.yaml)")
    st.add_argument("--audit-picturable", action="store_true",
                    help="search for the words marked picturable: false and "
                         "report the ones a query still reaches")

    im = sub.add_parser("images")
    im.add_argument("--deck", type=Path, required=True)
    im.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    im.add_argument("--limit", type=int, default=None,
                    help="attempt only the first N words (smoke runs)")
    im.add_argument("--no-verify", action="store_true",
                    help="accept the first search hit instead of judging "
                         "several candidates (gen.yaml `rulebook` supplies the judge)")

    cp = sub.add_parser("compile")
    cp.add_argument("--deck", type=Path, required=True)
    cp.add_argument("--out", type=Path, required=True)
    cp.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    cp.add_argument("--base", type=int, default=None,
                    help="defaults to gen.yaml's sentence_base")
    cp.add_argument("--skip-incomplete", action="store_true",
                    help="drop notes whose essential media is missing instead "
                         "of failing; optional media compiles as an empty field")

    return p


def _report_result(name: str, res) -> None:
    print(f"{name}: +{res.added} changed={res.changed} blocked={len(res.blocked)}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "wordlist" and args.assign_ids:
        rows = yaml.safe_load(args.out.read_text(encoding="utf-8")) or []
        rows, notes = assign_ids(rows)
        args.out.write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")
        for note in notes:
            print(note)
        missing = [r["gloss"] for r in rows if not r.get("id")]
        print(f"{len(rows)} row(s); {len(missing)} without an id")
        return 1 if missing else 0

    if args.command == "wordlist":
        warnings = []
        backend = CliBackend(model=GenConfig().model)
        if args.extend:
            emphasis = load_emphasis(Path("data/emphasis.yaml"))
            if emphasis is None:
                print("error: data/emphasis.yaml not found", file=sys.stderr)
                return 2
            count = extend_word_list(backend, Path("data/categories.yaml"),
                                     Path("data/frequency_th.txt"), args.out,
                                     emphasis, warnings=warnings)
            print(f"{count} emphasis entries in {args.out}")
        else:
            count = draft_word_list(backend, Path("data/categories.yaml"),
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
        res = fill_sentences(gaps, deck, ctx, checkpoint=lambda: write_deck(deck))
        write_deck(deck)
        _report_result("sentences", res)

    elif args.command == "audio" and args.audio_command == "fetch-forvo":
        deck = load_deck(args.deck)
        manifest = Manifest.load(deck.root)
        client = ForvoClient(_require_secret(args.deck, "forvo"))
        needs = [n for n in pending_audio(deck) if n.family in NATIVE_TIER_FAMILIES]
        limit = (args.limit if args.limit is not None
                 else load_config(args.deck).forvo_request_limit)
        res = fetch_forvo(needs, deck, manifest, client, _today(),
                          max_speakers=args.max_speakers, limit=limit,
                          checkpoint=lambda: write_deck(deck),
                          memo=ForvoMemo.load(deck.root))
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
        tts = GoogleTts(_require_secret(args.deck, "google_tts"))
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

    elif args.command == "approve":
        deck = load_deck(args.deck)
        waivers = [w for w in load_waivers(deck.root)
                   if not (w.note_id == args.note and w.rule == args.rule)]
        waivers.append(Waiver(note_id=args.note, rule=args.rule,
                              reason=args.reason, date=_today(),
                              sha=image_sha(deck, args.note)))
        save_waivers(deck.root, waivers)
        print(f"waived {args.rule} for {args.note}")

    elif args.command == "search-terms" and args.audit_picturable:
        deck = load_deck(args.deck)
        ctx = _build_ctx(args.deck, args.data_dir)
        unreachable = search_reachable(ctx.http_get) if ctx.http_get else None
        if unreachable:
            print(f"error: {unreachable}", file=sys.stderr)
            return 2
        judge = image_judge_for(args.deck, ctx.config)
        if judge is None:
            print("error: no judge configured (gen.yaml rulebook:); a result "
                  "count cannot tell a picture from a calendar", file=sys.stderr)
            return 2
        found = audit_picturable(ctx.word_list, deck, ctx, judge,
                                 getattr(ctx, "image_query_hints", None) or {})
        for thai, picture in found.items():
            print(f"{thai}: {picture.source} {picture.url}")
        print(f"{len(found)} word(s) written off that a picture can serve")

    elif args.command == "search-terms" and args.apply_proposals:
        n = apply_query_proposals(args.data_dir / "word_list_th.yaml",
                                  args.deck / "work" / "image_query_proposals.yaml")
        print(f"adopted {n} judge-proposed image phrase(s)")

    elif args.command == "search-terms":
        cfg = load_config(args.deck)
        warnings: list[str] = []
        n = draft_image_queries(_drafting_backend(args.deck, cfg),
                                args.data_dir / "word_list_th.yaml", warnings,
                                hints=load_image_query_hints(args.data_dir))
        for w in warnings:
            print(f"warning: {w}")
        print(f"drafted {n} image query phrase(s)")

    elif args.command == "images":
        deck = load_deck(args.deck)
        ctx = _build_ctx(args.deck, args.data_dir)
        ctx.imagegen = imagegen_for(ctx)
        gaps = _gaps_for(args.deck, args.data_dir)
        manifest = Manifest.load(deck.root)
        flagged = flagged_image_note_ids(gaps)
        glosses = {e.thai: e.gloss for e in ctx.word_list}
        queries = {e.thai: e.image_query for e in ctx.word_list if e.image_query}
        unreachable = search_reachable(ctx.http_get) if ctx.http_get else None
        if unreachable:
            print(f"error: {unreachable}", file=sys.stderr)
            print("       (search_proxy in gen.yaml needs its ssh tunnel up)",
                  file=sys.stderr)
            return 2
        judge = None if args.no_verify else image_judge_for(args.deck, ctx.config)
        # With a judge the run scores the deck's own pictures, so it needs all
        # of them; without one it can only act on what the last report flagged.
        res = fill_images(
            pending_images(deck, flagged=flagged, glosses=glosses,
                           image_queries=queries,
                           include_present=judge is not None),
            gaps, deck, manifest, ctx, _today(), judge=judge, limit=args.limit)
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
        wl_rows = yaml.safe_load((args.data_dir / "word_list_th.yaml").read_text()) or []
        gloss_of = {}
        for row in wl_rows:
            gloss_of.setdefault(row.get("thai", ""), row.get("gloss", ""))
        dropped = compile_deck(deck, manifest, args.out, freq, gaps.pair_by_note,
                               base=base, gloss_of=gloss_of,
                               emphasis=load_emphasis(args.data_dir / "emphasis.yaml"),
                               skip_incomplete=args.skip_incomplete)
        if dropped:
            by_family = Counter(family for family, _ in dropped)
            print(f"skipped {len(dropped)} note(s) missing essential media: "
                  + ", ".join(f"{fam} {n}" for fam, n in sorted(by_family.items())))
        print(f"compiled {args.out}")

    return 0
