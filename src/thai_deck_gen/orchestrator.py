import datetime
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from thai_deck_eval.model.deck import Deck
from thai_deck_gen.deckio import write_deck
from thai_deck_gen.media.forvo import ForvoClient, fetch_forvo
from thai_deck_gen.media.images import fill_images
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import pending_audio, pending_images
from thai_deck_gen.media.thai1000 import audio_index, import_thai1000
from thai_deck_gen.media.tts import GoogleTts, fill_tts
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.producers.pairs import fill_pairs
from thai_deck_gen.producers.sentences import fill_sentences
from thai_deck_gen.producers.spelling import fill_spelling
from thai_deck_gen.producers.words import fill_words
from thai_deck_gen.report import Gaps, fingerprint, parse_report


class EvalError(Exception):
    pass


@dataclass
class IterationSummary:
    gaps_fingerprint: str
    results: dict[str, ProducerResult] = field(default_factory=dict)


def run_eval(deck_dir: Path, runner=subprocess.run) -> dict:
    """Run the evaluator on `deck_dir` and return the parsed JSON report.

    Exit code 2 (or a stdout that isn't parseable JSON) means the evaluator
    itself failed -> EvalError. Exit codes 0 and 1 both mean "ran fine,
    report may show gate: pass or fail" -- stdout is the JSON report either
    way. The raw report is also stashed at <deck_dir>/.last-report.json.
    """
    deck_dir = Path(deck_dir)
    cmd = ["uv", "run", "thai-deck-eval", str(deck_dir), "--no-judge", "--format", "json"]
    result = runner(cmd, capture_output=True, text=True)
    if result.returncode == 2:
        raise EvalError(f"thai-deck-eval failed (exit 2): {result.stderr[:500]}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise EvalError(f"unparseable evaluator output: {exc}") from exc
    (deck_dir / ".last-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _fillable(gaps: Gaps) -> bool:
    return bool(gaps.missing_contrasts or gaps.missing_categories or gaps.findings)


def _dispatch_content(gaps: Gaps, deck: Deck, ctx) -> dict[str, ProducerResult]:
    order = [("pairs", fill_pairs), ("spelling", fill_spelling),
             ("words", fill_words), ("sentences", fill_sentences)]
    results: dict[str, ProducerResult] = {}
    for name, producer in order:
        res = producer(gaps, deck, ctx)
        results[name] = res
        print(f"  {name}: +{res.added} changed={res.changed} blocked={len(res.blocked)}")
    return results


def _today() -> str:
    return datetime.date.today().isoformat()


def _dispatch_media(gaps: Gaps, deck: Deck, ctx) -> dict[str, ProducerResult]:
    """Media fillers, each only run when configured on ctx. Unconfigured
    channels are skipped; their pending needs get rolled into the final
    blocked summary in `generate`."""
    results: dict[str, ProducerResult] = {}
    manifest = Manifest.load(deck.root)
    today = _today()

    if ctx.thai1000_apkg is not None and Path(ctx.thai1000_apkg).exists():
        index = audio_index(ctx.thai1000_apkg)
        res = import_thai1000(pending_audio(deck), deck, manifest, index, today)
        results["thai1000"] = res
        print(f"  thai1000: changed={res.changed} blocked={len(res.blocked)}")

    if ctx.forvo_api_key:
        client = ForvoClient(ctx.forvo_api_key)
        res = fetch_forvo(pending_audio(deck), deck, manifest, client, today)
        results["forvo"] = res
        print(f"  forvo: changed={res.changed} blocked={len(res.blocked)}")

    if ctx.tts_api_key:
        tts = GoogleTts(ctx.tts_api_key)
        res = fill_tts(pending_audio(deck), deck, manifest, tts, today)
        results["tts"] = res
        print(f"  tts: changed={res.changed} blocked={len(res.blocked)}")

    if ctx.http_get is not None:
        res = fill_images(pending_images(deck), gaps, deck, manifest, ctx, today)
        results["images"] = res
        print(f"  images: changed={res.changed} blocked={len(res.blocked)}")

    manifest.save(deck.root)
    return results


def _blocked_summary(deck: Deck, ctx) -> str:
    audio_needs = pending_audio(deck)
    image_needs = pending_images(deck)
    unconfigured = [name for name, configured in (
        ("thai1000", ctx.thai1000_apkg is not None),
        ("forvo", bool(ctx.forvo_api_key)),
        ("tts", bool(ctx.tts_api_key)),
        ("images", ctx.http_get is not None),
    ) if not configured]
    return (f"blocked: {len(audio_needs)} audio need(s), {len(image_needs)} image need(s) "
           f"pending; unconfigured channels: {', '.join(unconfigured) or 'none'}")


def generate(deck: Deck, ctx, evaluate=run_eval) -> list[IterationSummary]:
    """Evaluate -> parse_report -> dispatch producers -> write_deck, looping
    until: a report has no fillable gaps, the gaps fingerprint repeats
    (no progress), or ctx.config.max_iterations is reached."""
    summaries: list[IterationSummary] = []
    prev_fingerprint: str | None = None

    for i in range(ctx.config.max_iterations):
        report = evaluate(deck.root)
        gaps = parse_report(report, ctx.data_dir / "contrasts.yaml")
        fp = fingerprint(gaps)

        if not _fillable(gaps):
            print(f"iteration {i + 1}: no fillable gaps, stopping")
            break
        if fp == prev_fingerprint:
            print(f"iteration {i + 1}: no progress (gaps unchanged), stopping")
            break

        print(f"iteration {i + 1}:")
        results = _dispatch_content(gaps, deck, ctx)
        results.update(_dispatch_media(gaps, deck, ctx))
        write_deck(deck)
        summaries.append(IterationSummary(gaps_fingerprint=fp, results=results))
        prev_fingerprint = fp

    print(_blocked_summary(deck, ctx))
    return summaries
