import functools
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests
import yaml

from thai_deck_gen.config import GenConfig, load_config
from thai_deck_gen.emphasis import Emphasis, load_emphasis
from thai_deck_gen.media.imgfetch import ImgFetch
from thai_deck_gen.producers.sentences import load_exemplars
from thai_deck_eval.secrets import SecretStore
from thai_deck_gen.wordlist import WordEntry, load_word_list


@dataclass
class GenContext:
    """Bundles everything producers consume."""
    g2p: object
    tokenizer: object
    freq: object
    llm: object
    word_list: list[WordEntry]
    lexicon_words: list[str]
    exceptions: dict[str, str]
    pair_seeds: dict
    grammar_points: list[dict]
    exemplars: list[str]
    config: GenConfig
    data_dir: Path
    adjudication_queue: Path
    targets_path: Path                        # spelling_targets.yaml, for fill_spelling
    emphasis: Emphasis | None = None          # data/emphasis.yaml, if present
    thai1000_apkg: Path | None = None
    secrets: SecretStore = field(default_factory=SecretStore)
    imagegen: object | None = None
    imgfetch: ImgFetch | None = None          # image downloads (whitelistable binary)
    http_get: Callable | None = field(default=requests.get)


def _have_pysocks() -> bool:
    return importlib.util.find_spec("socks") is not None


def proxied_get(proxy: str | None, getter: Callable = requests.get) -> Callable:
    """`getter` with every request routed through `proxy`.

    Image search only: api.openverse.org answers this network's egress with
    a Cloudflare challenge, so searches go out through a SOCKS tunnel to a
    host with a clean address. Downloads keep going direct via imgfetch.
    """
    if not proxy:
        return getter
    if proxy.startswith("socks") and not _have_pysocks():
        raise RuntimeError(
            f"search_proxy {proxy!r} needs PySocks: install the `gen` extra "
            "(requests[socks]), or drop search_proxy from gen.yaml")
    return functools.partial(getter, proxies={"http": proxy, "https": proxy})


def _load_frequency_words(path: Path, top_n: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    words = [w for w in lines if w and not w.startswith("#")]
    return words[:top_n]


def _load_word_list(deck_root: Path, data_dir: Path) -> list[WordEntry]:
    path = data_dir / "word_list_th.yaml"
    if not path.exists():
        print(f"warning: {path} not found; starting with an empty word list")
        return []
    return load_word_list(path, data_dir / "categories.yaml")


def _resolve_thai1000_apkg(deck_root: Path, config: GenConfig) -> Path | None:
    if not config.thai1000_apkg:
        return None
    path = Path(config.thai1000_apkg)
    return path if path.is_absolute() else deck_root / path


def build_context(deck_root: Path, data_dir: Path, llm, nlp: bool,
                  g2p=None, tokenizer=None, freq=None,
                  config: GenConfig | None = None) -> GenContext:
    """Build a GenContext.

    nlp=True wires real pythainlp adapters (imported here, not at module
    scope, so importing this module stays default-suite-clean).
    nlp=False requires the caller to inject g2p/tokenizer/freq.

    Media channel config is derived here so it's consistent across every
    caller: API keys from gen.yaml's `secrets:` references (resolved lazily,
    on first use), thai1000_apkg from gen.yaml's `thai1000_apkg` key
    (deck-root-relative unless absolute), and http_get wired only when
    gen.yaml's `images` key is true (default) and THAI_DECK_GEN_FAKE isn't
    set (keeps the fake/test seam network-free).
    """
    deck_root, data_dir = Path(deck_root), Path(data_dir)
    if nlp:
        from thai_deck_eval.data_io import FileFrequencyList
        from thai_deck_eval.lang.pythainlp_adapter import (PyThaiNLPG2P,
                                                            PyThaiNLPTokenizer)
        g2p = g2p or PyThaiNLPG2P()
        tokenizer = tokenizer or PyThaiNLPTokenizer()
        freq = freq or FileFrequencyList(data_dir / "frequency_th.txt")
    elif g2p is None or tokenizer is None or freq is None:
        raise ValueError("nlp=False requires caller-injected g2p, tokenizer, freq")

    config = config or load_config(deck_root)
    word_list = _load_word_list(deck_root, data_dir)

    freq_words = _load_frequency_words(data_dir / "frequency_th.txt", config.lexicon_top_n)
    lexicon_words = sorted({w.thai for w in word_list} | set(freq_words))

    exceptions_path = data_dir / "g2p_exceptions.yaml"
    exceptions = (yaml.safe_load(exceptions_path.read_text(encoding="utf-8")) or {}
                 if exceptions_path.exists() else {})

    pair_seeds_path = data_dir / "pair_seeds.yaml"
    pair_seeds = (yaml.safe_load(pair_seeds_path.read_text(encoding="utf-8")) or {}
                 if pair_seeds_path.exists() else {})

    grammar_points_path = data_dir / "grammar_points.yaml"
    grammar_points = (yaml.safe_load(grammar_points_path.read_text(encoding="utf-8")) or []
                      if grammar_points_path.exists() else [])

    exemplars_path = deck_root / "work" / "exemplars.txt"
    exemplars = load_exemplars(exemplars_path) if exemplars_path.exists() else []

    fake = os.environ.get("THAI_DECK_GEN_FAKE") == "1"
    http_get = proxied_get(config.search_proxy) if (config.images and not fake) else None

    return GenContext(
        g2p=g2p, tokenizer=tokenizer, freq=freq, llm=llm,
        word_list=word_list, lexicon_words=lexicon_words, exceptions=exceptions,
        pair_seeds=pair_seeds, grammar_points=grammar_points, exemplars=exemplars,
        config=config, data_dir=data_dir,
        emphasis=load_emphasis(data_dir / "emphasis.yaml"),
        adjudication_queue=deck_root / "work" / "ipa_adjudication.yaml",
        targets_path=data_dir / "spelling_targets.yaml",
        thai1000_apkg=_resolve_thai1000_apkg(deck_root, config),
        secrets=SecretStore.from_config(config.secrets),
        http_get=http_get,
        imgfetch=ImgFetch(config.imgfetch),
    )


def imagegen_for(ctx: GenContext):
    """AI image generator when `secrets.openai` is configured, else None.

    Built at the image entry points rather than in build_context so that
    commands which generate no images never resolve the key.
    """
    if not ctx.secrets.configured("openai"):
        return None
    from thai_deck_gen.media.images import OpenAiImageGen
    return OpenAiImageGen(ctx.secrets.get("openai"))
