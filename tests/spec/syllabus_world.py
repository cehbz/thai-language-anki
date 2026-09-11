"""Spec-level world for the compiled-deck properties (spec 4 sections 1-3):
a Syllabus over the NEW thai_syllabus package, with a real SyllabusDb and
MediaStore under tmp_path, so a property test asserts on the actual .apkg
compile_syllabus writes rather than on a return value alone.

This module and tests/spec/test_compiled_deck.py import only from the new
thai_syllabus package: no old generator/evaluator package, and no helper
from that old stack's own test tree. `read_apkg` below is a fresh,
stdlib-only "zip -> collection.anki2 sqlite" reader -- the same shape a
sibling helper for the old compiler's tests already uses, reimplemented
here (not imported) so this module's own import list stays clear of that
older test tree.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from thai_syllabus.authority import ROLE_FOR_KIND
from thai_syllabus.cachekeys import JudgeKey, MechanicalKey, ProvideKey, rendition_identity
from thai_syllabus.entities import (
    Grapheme, MinimalPair, Pronunciation, Sentence, SoundConfusion, Syllable,
    Target, Word, render,
)
from thai_syllabus.ids import ConfusionId, PairId, TargetId, WordId
from thai_syllabus.media import Provenance, Speaker
from thai_syllabus.profile import Profile
from thai_syllabus.rulebook import RULES, sentence_note_id
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.wiring import _DbMediaIndex

PROV = Provenance(source="test", origin="fixture", licence="cc0",
                  acquired=date(2026, 1, 1))

# Completeness ERROR rules that would close the gate on a fixture built for
# compile.py's own mechanics (fields, guids, due, dropped-card counting)
# rather than for full curated coverage -- the same exclusion
# tests/syllabus/test_compile.py uses, for the identical reason.
# card/unique-front is excluded from the default set too; only the test
# that exercises it turns it back on.
_COMPLETENESS_ERROR_IDS = {
    "target/picture-required", "target/recording-required", "target/sentence-required",
    "pair/rendition-required", "grapheme/keyword-picture-required",
    "sentence/recording-required", "card/unique-front",
}
RULES_WITHOUT_COMPLETENESS = tuple(r for r in RULES if r.id not in _COMPLETENESS_ERROR_IDS)
UNIQUE_FRONT_RULE = next(r for r in RULES if r.id == "card/unique-front")


# --- reading a real .apkg back ---------------------------------------------

def read_apkg(path: Path) -> dict:
    """Notes/cards/models/media out of a real .apkg: unzip, read
    collection.anki2 (sqlite) and the media manifest, decode each note's
    \\x1f-joined fields.
    """
    with zipfile.ZipFile(path) as zf:
        media_map: dict[str, str] = json.loads(zf.read("media").decode())
        media = {name: zf.read(idx) for idx, name in media_map.items()}
        db_bytes = zf.read("collection.anki2")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "collection.anki2"
        db_path.write_bytes(db_bytes)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            notes = [dict(r) for r in conn.execute("select * from notes")]
            cards = [dict(r) for r in conn.execute("select * from cards")]
            (models_json,) = conn.execute("select models from col").fetchone()
        finally:
            conn.close()

    for n in notes:
        n["flds"] = n["flds"].split("\x1f")

    return {"notes": notes, "cards": cards,
           "models": json.loads(models_json), "media": media}


def _syl(onset: str = "m", vowel: str = "a", coda: str = "",
        length: str = "short", tone: str = "mid") -> Syllable:
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def _pron(*syllables: Syllable) -> Pronunciation:
    return Pronunciation(syllables=tuple(syllables) or (_syl(),),
                         corroboration="engines_agree")


def _word(id_: str, thai: str, meaning: str, tone: str = "mid") -> Word:
    return Word(id=WordId(id_), thai=thai, pron=_pron(_syl(tone=tone)), meaning=meaning)


def _sentence(words: tuple[Word, ...], clauses, *, gloss: str,
             voice: str = "learner_voice") -> Sentence:
    """Sentence.text as entities.render over `clauses`, resolving each
    element's word id against `words`' own thai forms.
    """
    thai_of = {w.id: w.thai for w in words}.__getitem__
    return Sentence(clauses=clauses, text=render(clauses, thai_of), gloss=gloss,
                    voice=voice, provenance=PROV)


# --- the real db + media store, and what compile.py needs seeded ----------

@dataclass
class SyllabusWorld:
    """A real SyllabusDb + MediaStore under tmp_path. Its seed_* methods
    write exactly the rows compile_syllabus's media resolution
    (derivations.current_best) needs to treat a subject's artifact as
    current-best: a provide row plus a passing judge verdict (words,
    graphemes), or a passing rendition verdict (pairs).
    """
    db: SyllabusDb
    media: MediaStore
    out_path: Path

    @classmethod
    def create(cls, tmp_path: Path) -> "SyllabusWorld":
        return cls(db=SyllabusDb(tmp_path / "syllabus.db"),
                   media=MediaStore(tmp_path / "media"),
                   out_path=tmp_path / "out" / "deck.apkg")

    def _pass_judge(self, subject: str, kind: str, sha: str) -> None:
        role = ROLE_FOR_KIND.get(kind, kind)
        self.db.append(port="assess", backend="judge",
                       key=JudgeKey.for_rule("seed", sha, subject, role), subject=subject,
                       question={"role": role,
                                "artifact_sha": sha, "rubric": "seed", "kind": kind},
                       answer={"value": True})

    def seed_recording(self, subject: str, text: str, speaker: str = "somchai") -> str:
        sha = self.media.write(f"audio:{subject}:{text}".encode(), ext="mp3")
        self.db.add_speaker(Speaker(id=speaker, kind="native"))
        self.db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                          origin="https://forvo.com/x", licence="cc-by",
                          acquired=date(2026, 1, 1), speaker_id=speaker)
        self.db.append(port="provide", backend="forvo",
                       key=ProvideKey(source="forvo", kind="", query=subject),
                       subject=subject, question={"provides": "recording", "kind": "recording"},
                       answer={"items": [{"sha": sha}]})
        self._pass_judge(subject, "recording", sha)
        return sha

    def seed_picture(self, subject: str, text: str) -> str:
        sha = self.media.write(f"image:{subject}:{text}".encode(), ext="jpg")
        self.db.add_media(sha=sha, kind="picture", ext="jpg", source="openverse",
                          origin="https://example.com/x.jpg", licence="cc0",
                          acquired=date(2026, 1, 1))
        self.db.append(port="provide", backend="openverse",
                       key=ProvideKey(source="openverse", kind="", query=subject),
                       subject=subject, question={"provides": "picture", "kind": "picture"},
                       answer={"items": [{"sha": sha}]})
        self._pass_judge(subject, "picture", sha)
        return sha

    def seed_rendition(self, pair: MinimalPair, texts: dict[str, str],
                       speaker: str = "somchai") -> dict[str, str]:
        """Writes the passing "rendition" mechanical verdict (spec 3
        section 5) that makes `pair.id` resolve a current-best rendition:
        one recording per member, all under the same speaker. Returns
        member id -> sha.
        """
        self.db.add_speaker(Speaker(id=speaker, kind="native"))
        shas: dict[str, str] = {}
        for member in pair.members:
            sha = self.media.write(f"rendition:{pair.id}:{member}:{texts[member]}".encode(),
                                   ext="mp3")
            self.db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                              origin="https://forvo.com/x", licence="cc-by",
                              acquired=date(2026, 1, 1), speaker_id=speaker)
            shas[member] = sha
        self.db.append(port="assess", backend="rendition",
                       key=MechanicalKey(check="rendition", params="v1", subject=str(pair.id),
                                        artifact_sha=rendition_identity(shas)),
                       subject=pair.id,
                       question={"role": "rendition-for-pair",
                                "artifact_sha": rendition_identity(shas),
                                "rubric": None, "kind": "rendition", "subject_kind": "pair",
                                "params": {"members": shas}},
                       answer={"value": True})
        return shas


# --- fixture: a word target, a pair, a grapheme, and a sentence that fills
# several targets at once (spec 4 sections 1-2) ----------------------------

def full_syllabus() -> Syllabus:
    """pom "ผม" (I, male speaker), gin "กิน" (to eat), rice "ข้าว" (cooked
    rice -- carries both a receptive and a productive Target), chicken
    "ไก่" (chicken, a grapheme keyword), the letter name "กอ ไก่" ("gɔɔ
    gài", the grapheme ก's own recited name), a near/far tone pair "ใกล้"/
    "ไกล" (near/far), the grapheme ก itself, and the sentence "ผมกินข้าว"
    (I eat rice -- names pom, gin and rice as its elements, so it fills
    all three of their Targets, spec 1 section 3's fills()).
    """
    pom = _word("pom", "ผม", "I (male speaker)")
    gin = _word("gin", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    chicken = _word("chicken", "ไก่", "chicken")
    ko_name = _word("letter-name:ko", "กอ ไก่", "the letter ก (recited name)")

    near = _word("near", "ใกล้", "near", tone="mid")
    far = _word("far", "ไกล", "far", tone="low")
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = MinimalPair.create(id=PairId("tone:mid-low/klai"), confusion=confusion,
                              members=(near, far))

    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",  # ก: the letter k
                               consonant_class="mid", keyword_word=chicken,
                               name_word=ko_name)

    targets = (
        Target(id=TargetId("pom/receptive"), word=pom.id, skill="receptive"),
        Target(id=TargetId("gin/receptive"), word=gin.id, skill="receptive"),
        Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive"),
        Target(id=TargetId("rice/productive"), word=rice.id, skill="productive"),
    )
    sentence = _sentence((pom, gin, rice), ((pom.id, gin.id, rice.id),),
                        gloss="I eat rice")  # I eat rice

    return Syllabus(
        words=(pom, gin, rice, chicken, ko_name, near, far), targets=targets,
        pairs=(pair,), graphemes=(grapheme,), sentences=(sentence,),
        confusions=(confusion,), profile=Profile(register="male_colloquial"),
        rules=RULES_WITHOUT_COMPLETENESS)


def seed_full(world: SyllabusWorld, syllabus: Syllabus) -> None:
    """Every current-best artifact `full_syllabus` needs for a clean
    (gate-open) compile: rice's picture and recording (its productive
    Target needs both), pom's and gin's recordings, chicken's picture (the
    grapheme's keyword picture), the letter name's own recording (the
    grapheme's Audio -- no substitute allowed, spec 4 section 1), near's
    and far's own word recordings (unused by compile, which plays the
    PAIR's rendition instead -- a separate seed_rendition call), and the
    fixture sentence's own recording.
    """
    world.seed_picture("rice", "cooked rice")
    world.seed_recording("rice", "cooked rice")
    world.seed_recording("pom", "I")
    world.seed_recording("gin", "eat")
    world.seed_picture("chicken", "chicken")
    world.seed_recording("letter-name:ko", "gɔɔ")
    world.seed_recording("near", "near")
    world.seed_recording("far", "far")
    world.seed_recording(sentence_note_id(syllabus.sentences[0]), "ผมกินข้าว")  # I eat rice


def fully_seeded_syllabus(world: SyllabusWorld) -> Syllabus:
    """`full_syllabus` with every artifact it needs already seeded in
    `world`, its own rendition seeded so the pair compiles too, and its
    `media` port wired to `world.db` (compile.py's pair-rendition lookup
    goes through Syllabus.media, not through the word/picture Resolver
    the rest of this fixture's artifacts resolve through).
    """
    syllabus = full_syllabus()
    seed_full(world, syllabus)
    world.seed_rendition(syllabus.pairs[0], {"near": "near", "far": "far"})
    return dataclasses.replace(
        syllabus, media=_DbMediaIndex(db=world.db, pairs=syllabus.pairs))


# --- fixture: two words sharing one spelling, for card/unique-front -------

def duplicate_front_syllabus() -> Syllabus:
    """Two distinct words both spelled "ข้าว" (rice) -- their Reading
    template fronts ("{{Thai}}" alone) render identically, tripping
    card/unique-front (A3: no two cards share a front).
    """
    rice_a = _word("rice-a", "ข้าว", "cooked rice (a)")
    rice_b = _word("rice-b", "ข้าว", "cooked rice (b)")
    targets = (Target(id=TargetId("rice-a/receptive"), word=rice_a.id, skill="receptive"),
              Target(id=TargetId("rice-b/receptive"), word=rice_b.id, skill="receptive"))
    rules = RULES_WITHOUT_COMPLETENESS + (UNIQUE_FRONT_RULE,)
    return Syllabus(words=(rice_a, rice_b), targets=targets, rules=rules)


# --- fixture: one receptive-only Target filled by its own sentence --------

def receptive_only_sentence_syllabus() -> Syllabus:
    """gin "กิน" (to eat), one receptive Target, one sentence using only
    that word -- so its last used word (Syllabus.last_used_word) carries
    no productive Target and the sentence note gets no Cloze card (spec 4
    section 1: Productive gates the Cloze card).
    """
    gin = _word("gin", "กิน", "to eat")
    target = Target(id=TargetId("gin/receptive"), word=gin.id, skill="receptive")
    eat = _sentence((gin,), ((gin.id,),), gloss="to eat")  # to eat
    return Syllabus(words=(gin,), targets=(target,), sentences=(eat,),
                    profile=Profile(register="male_colloquial"), rules=RULES_WITHOUT_COMPLETENESS)


def seed_receptive_only_sentence(world: SyllabusWorld, syllabus: Syllabus) -> None:
    world.seed_recording("gin", "eat")
    world.seed_recording(sentence_note_id(syllabus.sentences[0]), "กิน")


# --- fixture: one minimal pair with no rendition seeded -------------------

def pair_only_syllabus() -> tuple[Syllabus, MinimalPair]:
    """A near/far tone pair ("ใกล้"/"ไกล") with no word Targets at all --
    the syllabus a "no rendition seeded" test wires its own `media` port
    onto (see test_compiled_deck.py), so compile must drop both member
    notes and count exactly one DroppedCard (spec 4 section 3: "a pair
    with no current-best rendition compiles no notes for either member").
    """
    near = _word("near", "ใกล้", "near", tone="mid")
    far = _word("far", "ไกล", "far", tone="low")
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = MinimalPair.create(id=PairId("p1"), confusion=confusion, members=(near, far))
    syllabus = Syllabus(words=(near, far), pairs=(pair,), confusions=(confusion,),
                        profile=Profile(register="male_colloquial"),
                        rules=RULES_WITHOUT_COMPLETENESS)
    return syllabus, pair
