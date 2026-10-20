"""Recorded engines: a local engine's answer for a form is a record row,
so a fold can settle pronunciation from the record alone without running
(or loading) the engine. The pattern is `dictionary.Wiktionary`'s: the row
(`EngineKey`: engine, version, form) is read first; on a miss the engine
runs once and its answer is appended -- "no reading" included -- so one
form costs one computation per engine version ever. The engines stay
injectable callables; these wrap them.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .cachekeys import EngineKey
from .dictionary import DictionaryStore
from .entities import Syllable, Tone
from .phonology import read_form, syllables_from_verdict

PORT = "provide"

Reading = tuple[Syllable, ...]


class RecordedEngine:
    """`compute(form)`, a JSON-able answer, recorded under
    `EngineKey(name, version, form)`: the row's answer is
    `{"value": <answer>}`, so a None answer is a row too."""

    def __init__(self, cache: DictionaryStore, name: str, version: str,
                 compute: Callable[[str], Any]) -> None:
        self._cache = cache
        self._name = name
        self._version = version
        self._compute = compute

    def __call__(self, form: str) -> Any:
        key = EngineKey(engine=self._name, version=self._version, form=form)
        row = self._cache.latest(PORT, self._name, key)
        if row is not None:
            return row.answer["value"]
        value = self._compute(form)
        self._cache.append(port=PORT, backend=self._name, key=key,
                           subject=f"engine:{form}", question={"form": form},
                           answer={"value": value}, cost=0.0)
        return value


def recorded_reading(cache: DictionaryStore, name: str, version: str,
                     engine: Callable[[str], Reading | None]
                     ) -> Callable[[str], Reading | None]:
    """A g2p engine's reading of a form, recorded as
    `phonology.read_form` presents it (phrases token by token, ʔ-coda
    convention, degenerate readings dropped), so the record yields what the
    live engine would through `Engines.readings`."""
    ask = RecordedEngine(cache, name, version, lambda form: _to_json(read_form(engine, form)))

    def reading(form: str) -> Reading | None:
        syllables = ask(form)
        return None if syllables is None else syllables_from_verdict({"syllables": syllables})

    return reading


def recorded_tone(cache: DictionaryStore, name: str, version: str,
                  engine: Callable[[str], Tone | None]) -> Callable[[str], Tone | None]:
    """The rule tone engine's tone for a form, or None, recorded."""
    return RecordedEngine(cache, name, version, engine)


def _to_json(reading: Reading | None) -> list[dict] | None:
    if reading is None:
        return None
    return [{"segments": list(s.segments), "vowel_length": s.vowel_length,
             "tone": s.tone} for s in reading]
