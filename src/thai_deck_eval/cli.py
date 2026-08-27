import sys
from pathlib import Path
import click
# import stage modules for rule registration side effects
from .stages import judge_rules, linguistic, mechanical, method  # noqa: F401
from .config import load_rulebook
from .core.context import EvalContext
from .core.findings import Stage
from .core.pipeline import evaluate_path
from .judge.core import CachedJudge, FakeJudge
from .report.model import build_report
from .report.render import render_text
from .report.scoring import compute_scores

def _build_language_ports(vocab: set[str]):
    """Return (g2p, g2p_second, tokenizer, freq); None entries disable checks.

    `vocab` seeds the tokenizer's custom dictionary with this deck's own
    vocabulary (picture words + sentence targets) so deck-specific words
    pythainlp's dictionary doesn't already know still segment as single
    tokens.
    """
    g2p = second = tok = freq = None
    try:
        from .lang.pythainlp_adapter import (PyThaiNLPG2P, PyThaiNLPTokenizer,
                                             TltkG2P)
        g2p, second, tok = (PyThaiNLPG2P(), TltkG2P(),
                            PyThaiNLPTokenizer(extra_words=vocab))
    except ImportError:
        click.echo("warning: pythainlp not installed; linguistic checks skipped",
                   err=True)
    try:
        from .data_io import FileFrequencyList
        freq = FileFrequencyList()
    except OSError:
        click.echo("warning: frequency list missing", err=True)
    return g2p, second, tok, freq

def _build_judge(cfg):
    if cfg.judge.backend == "fake":
        return FakeJudge({})
    if cfg.judge.backend == "api":
        from .judge.api_judge import ApiJudge
        inner = ApiJudge(cfg.judge)
    else:
        from .judge.cli_judge import CliJudge
        inner = CliJudge(cfg.judge)
    return CachedJudge(inner, Path(cfg.judge.cache_path),
                       cfg.judge.model, cfg.judge.prompt_version)

@click.command()
@click.argument("deck_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--report", "report_path", type=click.Path(path_type=Path),
             help="Write the full JSON report to this file, in addition to "
                  "printing it to stdout.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text",
             help="Output format for the report printed to stdout.")
@click.option("--no-judge", is_flag=True,
             help="Skip the judge (LLM) stage; run only mechanical, "
                  "linguistic, and method-fidelity checks.")
@click.option("--stages", "stages_opt",
             help="Comma-separated list of stages to run "
                  "(mechanical,linguistic,method,judge), overriding --no-judge.")
@click.option("--rulebook", type=click.Path(path_type=Path),
             help="Path to a rulebook YAML config; defaults to built-in defaults.")
def main(deck_dir, report_path, fmt, no_judge, stages_opt, rulebook):
    try:
        cfg = load_rulebook(rulebook)
        stages = None
        if stages_opt:
            stages = [Stage(s.strip()) for s in stages_opt.split(",")]
        elif no_judge:
            stages = [Stage.MECHANICAL, Stage.LINGUISTIC, Stage.METHOD]

        def ctx_factory(deck):
            vocab = ({w.thai for w in deck.picture_words}
                    | {s.target for s in deck.sentences})
            g2p, second, tok, freq = _build_language_ports(vocab)
            judge = None if no_judge else _build_judge(cfg)
            return EvalContext(deck=deck, config=cfg, g2p=g2p, g2p_second=second,
                               tokenizer=tok, freq=freq, judge=judge)

        result = evaluate_path(deck_dir, ctx_factory, stages=stages)
        scores = compute_scores(result, cfg)
        name, version = "?", "?"
        try:
            from .model.deck import load_deck
            meta = load_deck(deck_dir).meta
            name, version = meta.name, meta.version
        except Exception:
            pass
        rep = build_report(name, version, result, scores, cfg)
        out = rep.model_dump_json(indent=2) if fmt == "json" else render_text(rep)
        click.echo(out, nl=False)
        if report_path:
            report_path.write_text(rep.model_dump_json(indent=2))
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(2)

    sys.exit(1 if rep.gate == "fail" else 0)
