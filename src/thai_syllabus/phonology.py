"""Pronunciation engines and the corroboration rule (spec 3 r28 section 5;
the 2026-09-01 domain-language ruling: engines are cheap oracles, the
judge the better-read oracle when they disagree, its verdict plus one
engine is corroboration). Engines are callables so the run injects the
real ones and tests inject fakes; thaig2p (torch) loads only inside
default_engines()."""
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
    second opinion on a monosyllabic tone disagreement. Both modules pull
    in pythainlp/torch, so the imports stay inside this function -- unit
    tests inject fake Engines and never reach here.
    """
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPG2P
    from thai_deck_eval.lang.tone import analyze_syllable
    g2p_engine = PyThaiNLPG2P()

    def g2p(thai: str) -> tuple[Syllable, ...] | None:
        syls = g2p_engine.syllables(thai)
        if syls is None:
            return None
        return tuple(Syllable(segments=(s.onset, s.vowel, s.coda or ""),
                              vowel_length="long" if s.long else "short",
                              tone=str(s.tone.value)) for s in syls)

    def tone(thai: str) -> Tone | None:
        analysis = analyze_syllable(thai)
        return str(analysis.tone.value) if analysis is not None else None

    return Engines(g2p=g2p, tone=tone)
