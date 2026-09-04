"""Terse constructors for tests -- not part of the domain, just less
boilerplate around the frozen dataclasses' full field lists.
"""
from datetime import date

from thai_syllabus.entities import Pronunciation, Sentence, Syllable, Target, Word
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
         corroboration: str = "engines_agree") -> Word:
    p = pron(*syllables, corroboration=corroboration) if syllables else pron(corroboration=corroboration)
    return Word(id=WordId(id), thai=thai, pron=p, meaning=meaning or id,
               classifier=WordId(classifier) if classifier else None)


def target(id: str, word_id: str, skill: str = "receptive",
           introduction: str = "picture_card") -> Target:
    return Target(id=TargetId(id), word=WordId(word_id), skill=skill,
                 introduction=introduction)


def sentence(text: str, voice: str = "learner_voice", gloss: str = "") -> Sentence:
    return Sentence(text=text, gloss=gloss, voice=voice, provenance=PROV)
