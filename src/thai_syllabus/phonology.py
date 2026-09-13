"""Pronunciation engines and the corroboration rule (spec 3 r28 section 5;
the 2026-09-01 domain-language ruling: engines are cheap oracles, the
judge the better-read oracle when they disagree, its verdict plus one
engine is corroboration). Engines are callables so the run injects the
real ones and tests inject fakes; the engines themselves live in
engines.py, and thaig2p (torch) loads only inside default_engines()."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from .entities import Syllable, Tone


@dataclass(frozen=True)
class Engines:
    g2p: Callable[[str], tuple[Syllable, ...] | None]
    tone: Callable[[str], Tone | None]


def syllables_from_verdict(value: Mapping) -> tuple[Syllable, ...]:
    return tuple(Syllable(segments=tuple(s["segments"]), vowel_length=s["vowel_length"],
                          tone=s["tone"]) for s in value["syllables"])


def _same_segments(a: Syllable, b: Syllable) -> bool:
    return a.segments == b.segments and a.vowel_length == b.vowel_length


def corroborates(judge: tuple[Syllable, ...], thai: str, engines: Engines) -> bool:
    g2p = engines.g2p(thai)
    if g2p is None or len(g2p) != len(judge):
        return False
    if not all(_same_segments(a, b) for a, b in zip(judge, g2p)):
        return False
    if all(a.tone == b.tone for a, b in zip(judge, g2p)):
        return True
    if len(judge) != 1:
        return False
    return engines.tone(thai) == judge[0].tone


def default_engines() -> Engines:
    """The real engines (spec 3 r28 section 5): pythainlp's thaig2p for
    segments/length/tone, and the deterministic tone-rule engine as the
    second opinion on a monosyllabic tone disagreement. Constructing the
    thaig2p engine pulls in pythainlp/torch, so it happens inside this
    function -- unit tests inject fake Engines and never reach here.
    """
    from .engines import Thaig2p, rule_tone
    return Engines(g2p=Thaig2p(), tone=rule_tone)
