from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Contrast = Literal["tone", "vowel_length", "aspiration", "vowel_quality", "consonant", "final"]

class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Audio(_Model):
    file: str
    source: Literal["native", "tts"]
    speaker: str

class PairMember(_Model):
    thai: str
    ipa: str
    audio: Audio
    gloss: str | None = None

class MinimalPairNote(_Model):
    id: str
    contrast: Contrast
    members: list[PairMember] = Field(min_length=2, max_length=3)

class SpellingSoundNote(_Model):
    id: str
    pattern: str
    pattern_kind: Literal["consonant", "vowel", "tone_mark"]
    consonant_class: Literal["mid", "high", "low"] | None = None
    example_word: str
    audio: Audio
    image: str

class PictureWordNote(_Model):
    id: str
    thai: str
    image: str
    audio: Audio
    frequency_rank: int
    category: str
    part_of_speech: Literal["noun", "verb", "adjective", "other"] = "other"
    classifier: str | None = None
    ipa: str | None = None
    test_spelling: bool = False
    personal_connection: str | None = None
    gloss: str | None = None

class SentenceNote(_Model):
    id: str
    kind: Literal["new_word", "word_form", "word_order"]
    thai: str
    target: str
    audio: Audio
    image: str | None = None
    definition: str | None = None
    gloss: str | None = None
    grammar_note: str | None = None

class StagePlan(_Model):
    phases: list[Literal["sounds", "words", "sentences"]]

class DeckMeta(_Model):
    name: str
    version: str
    stage_plan: StagePlan
