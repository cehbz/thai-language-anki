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


def pron(*syllables: Syllable, posterior: float = 1.0,
         spoken_dissent: bool = False) -> Pronunciation:
    if not syllables:
        syllables = (syl(),)
    return Pronunciation(syllables=tuple(syllables), posterior=posterior,
                         spoken_dissent=spoken_dissent)


def word(id: str, thai: str, meaning: str = "", classifier: str | None = None,
         syllables: tuple[Syllable, ...] | None = None,
         posterior: float = 1.0,
         speaker: str | None = None) -> Word:
    p = pron(*syllables, posterior=posterior) if syllables else pron(posterior=posterior)
    return Word(id=WordId(id), thai=thai, pron=p, meaning=meaning or id,
               classifier=WordId(classifier) if classifier else None, speaker=speaker)


def on_record(db, form: str, *readings: tuple[Syllable, ...]) -> None:
    """`form`'s readings on the record as the local engines' rows (spec 3
    section 3), what the settlement folds (spec 3 section 6): one reading,
    read by thaig2p and tltk alike, settles at posterior 1; two, one per
    engine (two error families), leave the form unsettled."""
    from thai_syllabus.cachekeys import EngineKey
    from thai_syllabus.settlement import ENGINE_VERSIONS
    first, second = (readings[0], readings[0]) if len(readings) == 1 else readings
    for engine, reading in (("thaig2p", first), ("tltk", second)):
        db.append(port="provide", backend=engine,
                  key=EngineKey(engine, ENGINE_VERSIONS[engine], form),
                  subject=f"engine:{form}", question={"form": form},
                  answer={"value": [{"segments": list(s.segments),
                                     "vowel_length": s.vowel_length, "tone": s.tone}
                                    for s in reading]})


def offline(ctx):
    """A wired Sourcing's evidence pass kept local (spec 3 r68): engines
    that read nothing (conftest refuses the real ones) and no dictionary,
    whose real backends would reach the network."""
    from thai_syllabus.phonology import Engines
    ctx.engines = Engines(g2p={"thaig2p": lambda thai: None, "tltk": lambda thai: None},
                          tone=lambda thai: None)
    ctx.dictionary = ctx.thai_language_com = None
    return ctx


def record_pronunciations(db, words) -> None:
    """Each Word's own `pron` put on the record as its form's evidence
    (`on_record`; the first Word of a form decides), so a deck loaded
    from files settles it again: a settled one read alike by both
    engines, an unsettled one split between them on tone; None, nothing."""
    seen: set[str] = set()
    for w in words:
        if w.pron is None or w.thai in seen:
            continue
        seen.add(w.thai)
        if w.pron.posterior >= 0.99:
            on_record(db, w.thai, w.pron.syllables)
        else:
            other = tuple(Syllable(segments=s.segments, vowel_length=s.vowel_length,
                                   tone="rising" if s.tone != "rising" else "mid")
                          for s in w.pron.syllables)
            on_record(db, w.thai, w.pron.syllables, other)


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
