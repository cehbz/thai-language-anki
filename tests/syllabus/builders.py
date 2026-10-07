"""Terse constructors for tests -- not part of the domain, just less
boilerplate around the frozen dataclasses' full field lists.
"""
import functools
import hashlib
from collections.abc import Callable
from datetime import date

from thai_syllabus.entities import Clauses, Pronunciation, Sentence, Syllable, Target, Word, render
from thai_syllabus.ids import TargetId, WordId
from thai_syllabus.media import Provenance

PROV = Provenance(source="test", origin="fixture", licence="cc0",
                   acquired=date(2026, 1, 1))


def syl(onset="m", vowel="a", coda="", length="short", tone="mid") -> Syllable:
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def pron(*syllables: Syllable, corroboration="engines_agree") -> Pronunciation:
    if not syllables:
        syllables = (syl(),)
    return Pronunciation(syllables=tuple(syllables), corroboration=corroboration)


def word(id: str, thai: str, meaning: str = "", classifier: str | None = None,
         syllables: tuple[Syllable, ...] | None = None,
         corroboration: str = "engines_agree",
         speaker: str | None = None) -> Word:
    p = pron(*syllables, corroboration=corroboration) if syllables else pron(corroboration=corroboration)
    return Word(id=WordId(id), thai=thai, pron=p, meaning=meaning or id,
               classifier=WordId(classifier) if classifier else None, speaker=speaker)


def target(id: str, word_id: str, skill: str = "receptive",
           introduction: str = "picture_card", sentences: int = 1) -> Target:
    return Target(id=TargetId(id), word=WordId(word_id), skill=skill,
                 introduction=introduction, sentences=sentences)


def thai_of(*words: Word) -> Callable[[WordId], str]:
    """Word id -> thai text, from a fixture's own registered Words --
    the lookup a sentence() call renders its text through.
    """
    index = {w.id: w.thai for w in words}
    return index.__getitem__


def sentence(clauses: Clauses, thai_of: Callable[[WordId], str], *,
             gloss: str = "", voice: str = "learner_voice") -> Sentence:
    return Sentence(clauses=clauses, text=render(clauses, thai_of), gloss=gloss,
                    voice=voice, provenance=PROV)


def clip(seconds: float = 0.4, freq: int = 440, fmt: str = "mp3") -> bytes:
    """A real audio clip (a sine tone, ffmpeg lavfi), for paths that
    condition recordings (spec 1 r33); distinct `freq` gives distinct bytes."""
    import subprocess
    return subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                           f"sine=f={freq}:r=24000:d={seconds}", "-f", fmt, "pipe:1"],
                          capture_output=True, check=True).stdout


@functools.lru_cache(maxsize=None)
def clip_for(text: str) -> bytes:
    """A real clip standing for `text` (a fake TTS's or fetcher's answer):
    equal text gives equal bytes, distinct text a distinct tone."""
    return clip(freq=200 + int(hashlib.sha256(text.encode()).hexdigest()[:6], 16) % 3000)
