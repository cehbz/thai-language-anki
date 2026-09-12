"""Repo inventories the sound stage is built from (design 2026-09-12 §1):
the 44 consonants and the vowel signs, tone marks and diacritics, each
with the recited name Thai speakers spell with. Fixed knowledge;
validated on load, never proposed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

_CONSONANTS = {chr(c) for c in range(0x0E01, 0x0E2F)} - {"ฤ", "ฦ"}   # 44 letters


@dataclass(frozen=True)
class ConsonantRow:
    symbol: str
    consonant_class: Literal["mid", "high", "low"]
    sound: str
    name_thai: str
    keyword_thai: str
    keyword_gloss: str


@dataclass(frozen=True)
class VowelRow:
    symbol: str
    kind: Literal["vowel_sign", "tone_mark", "diacritic"]
    sound: str
    name_thai: str


class InventoryError(ValueError):
    pass


def load_consonants(path: str | Path) -> tuple[ConsonantRow, ...]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    out = []
    for i, r in enumerate(rows):
        try:
            row = ConsonantRow(symbol=r["symbol"], consonant_class=r["class"], sound=r["sound"],
                               name_thai=r["name"], keyword_thai=r["keyword"], keyword_gloss=r["gloss"])
        except (KeyError, TypeError) as e:
            raise InventoryError(f"consonants[{i}]: malformed row ({e})") from e
        if row.consonant_class not in ("mid", "high", "low"):
            raise InventoryError(f"consonants[{i}] ({row.symbol}): class {row.consonant_class!r}")
        if not row.name_thai.startswith(row.symbol):
            raise InventoryError(f"consonants[{i}] ({row.symbol}): name {row.name_thai!r} does not start with the symbol")
        out.append(row)
    symbols = [r.symbol for r in out]
    if set(symbols) != _CONSONANTS or len(symbols) != 44:
        missing = sorted(_CONSONANTS - set(symbols)); extra = sorted(set(symbols) - _CONSONANTS)
        raise InventoryError(f"consonants: expected the 44 letters once each; missing {missing}, extra {extra}")
    return tuple(out)


def load_vowels(path: str | Path) -> tuple[VowelRow, ...]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    out = []
    for i, r in enumerate(rows):
        try:
            row = VowelRow(symbol=r["symbol"], kind=r["kind"], sound=str(r["sound"]), name_thai=r["name"])
        except (KeyError, TypeError) as e:
            raise InventoryError(f"vowels[{i}]: malformed row ({e})") from e
        if row.kind not in ("vowel_sign", "tone_mark", "diacritic"):
            raise InventoryError(f"vowels[{i}] ({row.symbol}): kind {row.kind!r}")
        out.append(row)
    if len({r.symbol for r in out}) != len(out):
        raise InventoryError("vowels: duplicate symbol")
    return tuple(out)
