"""A deck on disk, in the shape `thai_syllabus` reads it: curated/*.yaml,
a frequency corpus, syllabus.db and media/objects.

The doctrine tests beside this module (test_deck_doctrine.py) drive the
public entry points -- `wiring.load_syllabus` -> `Syllabus.report()`, and
`compile.compile_syllabus` -- against a deck this builder writes, exactly
as the CLI does. `build()` writes a complete deck: every error-severity
rule is satisfied, so the gate is open and the findings list is empty;
each doctrine test breaks one thing and asks what the deck became.

The builder is the new-format counterpart of the old evaluator's
tests/helpers.DeckBuilder. It is not a migration of one: `migrate.py`
carries a word list, its targets, pictures and learner rows, and writes
`graphemes=(), confusions=(), pairs=()` with no sentences and no audio
(spec 2 section 4), so a deck built through it can carry no minimal
pair, no grapheme and no sentence -- none of the doctrines below would
survive the trip.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from thai_syllabus.authority import ROLE_FOR_KIND
from thai_syllabus.cachekeys import JudgeKey, MechanicalKey, ProvideKey, rendition_identity
from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
from thai_syllabus.entities import (
    Category, Grapheme, MinimalPair, Pronunciation, Sentence, SoundConfusion, Syllable,
    Target, Word, render,
)
from thai_syllabus.ids import ConfusionId, PairId, TargetId, WordId
from thai_syllabus.media import Provenance, Speaker
from thai_syllabus.profile import Profile
from thai_syllabus.rulebook import RULES, rubrics_for, sentence_note_id
from thai_syllabus.store import MediaStore, SyllabusDb

PROV = Provenance(source="test", origin="fixture", licence="cc0",
                  acquired=date(2026, 1, 1))
ACQUIRED = date(2026, 1, 1)

# A judged artifact ranks current-best only under the rubric text the live
# rulebook carries for its role (spec 3 section 6's staleness check), so
# the seeds below judge under exactly that text.
RUBRICS = rubrics_for(RULES)


def syllable(onset: str = "m", vowel: str = "a", coda: str = "",
             length: str = "short", tone: str = "mid") -> Syllable:
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def word(id: str, thai: str, meaning: str, *, tone: str = "mid", onset: str = "m",
         corroboration: str = "engines_agree", classifier: str | None = None) -> Word:
    return Word(id=WordId(id), thai=thai, meaning=meaning,
                pron=Pronunciation(syllables=(syllable(onset=onset, tone=tone),),
                                   corroboration=corroboration),
                classifier=WordId(classifier) if classifier else None)


def sentence(clauses, words, *, gloss: str, voice: str = "learner_voice") -> Sentence:
    thai_of = {w.id: w.thai for w in words}.__getitem__
    return Sentence(clauses=clauses, text=render(clauses, thai_of), gloss=gloss,
                    voice=voice, provenance=PROV)


# --- the golden deck ------------------------------------------------------
#
# rice "ข้าว" and eat "กิน", each a picture-introduced receptive Target (rice
# a productive one too), both exercised by the sentence "กินข้าว" ("eat
# rice"); a near/far tone pair
# "ใกล้"/"ไกล" over the mid/low confusion; and the grapheme ก with the
# keyword "ไก่" ("gài", chicken).

RICE = word("rice", "ข้าว", "cooked rice")
EAT = word("eat", "กิน", "to eat")
NEAR = word("near", "ใกล้", "near", tone="mid")
FAR = word("far", "ไกล", "far", tone="low")
CHICKEN = word("chicken", "ไก่", "chicken")

TONE_CONFUSION = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                                sounds=("mid", "low"))


def _golden_pair() -> MinimalPair:
    return MinimalPair.create(id=PairId("tone:mid-low/klai"), confusion=TONE_CONFUSION,
                              members=(NEAR, FAR))


def _golden_sentence() -> Sentence:
    return sentence(((EAT.id, RICE.id),), (EAT, RICE), gloss="eat rice")


@dataclass
class DeckBuilder:
    """A deck directory whose parts are plain attributes: mutate one, then
    `build()`. Defaults are the golden deck -- complete, gate open.
    """
    tmp_path: Path
    words: list[Word] = field(default_factory=lambda: [RICE, EAT, NEAR, FAR, CHICKEN])
    targets: list[Target] = field(default_factory=lambda: [
        Target(id=TargetId("eat/receptive"), word=EAT.id, skill="receptive",
               introduction="picture_card"),
        Target(id=TargetId("rice/receptive"), word=RICE.id, skill="receptive",
               introduction="picture_card"),
        # rice is also productive: the sentence's last used word is rice,
        # so it fills this Target -- and the deck compiles rice's
        # Production card, the picture-prompted card the F3 doctrine is
        # about.
        Target(id=TargetId("rice/productive"), word=RICE.id, skill="productive",
               introduction="picture_card")])
    confusions: list[SoundConfusion] = field(default_factory=lambda: [TONE_CONFUSION])
    pairs: list[MinimalPair] = field(default_factory=lambda: [_golden_pair()])
    graphemes: list[Grapheme] = field(default_factory=lambda: [
        Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=CHICKEN)])
    categories: list[Category] = field(default_factory=lambda: [
        Category(name="Food", members=frozenset({"rice"})),
        Category(name="Verbs", members=frozenset({"eat"}))])
    sentences: list[Sentence] = field(default_factory=lambda: [_golden_sentence()])
    severities: dict[str, str] = field(default_factory=dict)
    # subject -> whether it is seeded; a doctrine test drops an entry to
    # make that artifact missing.
    pictures: list[str] = field(default_factory=lambda: ["rice", "eat", "chicken"])
    recordings: list[str] = field(default_factory=lambda: ["rice", "eat"])
    # pair ids whose rendition is seeded, and the speaker kind that voices
    # them ("synthetic" is TTS).
    rendition_speaker_kind: str = "native"
    # curated file name -> literal text, written after save_curated: the
    # way a malformed curated file gets onto disk.
    raw_curated: dict[str, str] = field(default_factory=dict)
    # shas whose media object is deleted from the store after seeding.
    orphan_media: bool = False

    @property
    def root(self) -> Path:
        return self.tmp_path / "deck"

    def target(self, id: str) -> Target:
        return next(t for t in self.targets if t.id == id)

    def build(self) -> Path:
        root = self.root
        save_curated(root / "curated", CuratedBundle(
            words=tuple(self.words), targets=tuple(self.targets),
            graphemes=tuple(self.graphemes), confusions=tuple(self.confusions),
            pairs=tuple(self.pairs), profile=Profile(register="male_colloquial"),
            rulebook=RulebookConfig(severities=dict(self.severities)),
            categories=tuple(self.categories)))
        # No word appears in the corpus, so none is ranked and no
        # productive Target is derived (spec 1 r9): the deck's Targets are
        # exactly the curated ones.
        (root / "curated" / "frequency_th.txt").write_text("", encoding="utf-8")
        # What load_providers_config requires of any deck the CLI reads.
        # No doctrine test reaches a provider: every one of them stops at
        # report() or at compile, both of which source nothing.
        (root / "curated" / "providers.yaml").write_text(
            "imgfetch_path: /opt/bin/imgfetch\n"
            "audiofetch_path: /opt/bin/audiofetch\n"
            "secrets: {anthropic: op://Shared/Anthropic/API Key}\n"
            "judge: {transport: api, model: m, "
            "price_per_mtok: {input: 2.0, output: 10.0}}\n", encoding="utf-8")
        for name, text in self.raw_curated.items():
            (root / "curated" / name).write_text(text, encoding="utf-8")

        db = SyllabusDb(root / "syllabus.db")
        media = MediaStore(root / "media")
        shas: list[tuple[str, str]] = []
        for s in self.sentences:
            db.add_sentence(text_sha=s.text_sha, text=s.text, clauses=s.clauses,
                            gloss=s.gloss, voice=s.voice, source="test", origin="fixture",
                            licence="cc0", acquired=ACQUIRED)
        for subject in self.pictures:
            shas.append((self._seed_picture(db, media, subject), "jpg"))
        for subject in self.recordings:
            shas.append((self._seed_recording(db, media, subject), "mp3"))
        for s in self.sentences:
            shas.append((self._seed_recording(db, media, sentence_note_id(s)), "mp3"))
        for pair in self.pairs:
            shas.extend((sha, "mp3") for sha in
                        self._seed_rendition(db, media, pair).values())
        db.close()

        if self.orphan_media:
            for sha, ext in shas:
                media.path_for(sha, ext).unlink(missing_ok=True)
        return root

    # --- seeds: what makes an artifact current-best (spec 3 section 6) ---

    @staticmethod
    def _pass_judge(db: SyllabusDb, subject: str, kind: str, sha: str) -> None:
        role = ROLE_FOR_KIND.get(kind, kind)
        rubric = RUBRICS.get(role, "seed")
        db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(rubric, sha, subject, role), subject=subject,
                  question={"role": role, "artifact_sha": sha, "rubric": rubric,
                            "kind": kind},
                  answer={"value": True})

    def _seed_picture(self, db: SyllabusDb, media: MediaStore, subject: str) -> str:
        sha = media.write(f"image:{subject}".encode(), ext="jpg")
        db.add_media(sha=sha, kind="picture", ext="jpg", source="openverse",
                     origin="https://example.invalid/x.jpg", licence="cc0",
                     acquired=ACQUIRED)
        db.append(port="provide", backend="openverse",
                  key=ProvideKey(source="openverse", kind="", query=subject),
                  subject=subject,
                  question={"provides": "picture", "kind": "picture"},
                  answer={"items": [{"sha": sha}]})
        self._pass_judge(db, subject, "picture", sha)
        return sha

    def _seed_recording(self, db: SyllabusDb, media: MediaStore, subject: str,
                        speaker: str = "somchai", kind: str = "native") -> str:
        sha = media.write(f"audio:{subject}:{speaker}".encode(), ext="mp3")
        db.add_speaker(Speaker(id=speaker, kind=kind, sex="male", age_band="adult",
                               region="central"))
        db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                     origin="https://forvo.invalid/x", licence="cc-by",
                     acquired=ACQUIRED, speaker_id=speaker)
        db.append(port="provide", backend="forvo",
                  key=ProvideKey(source="forvo", kind="", query=subject),
                  subject=subject,
                  question={"provides": "recording", "kind": "recording"},
                  answer={"items": [{"sha": sha}]})
        self._pass_judge(db, subject, "recording", sha)
        return sha

    def _seed_rendition(self, db: SyllabusDb, media: MediaStore,
                        pair: MinimalPair) -> dict[str, str]:
        """The passing "rendition" mechanical verdict that makes `pair`
        resolve a current-best rendition: one recording per member, all in
        one speaker's voice.
        """
        speaker = "tts-th" if self.rendition_speaker_kind == "synthetic" else "somchai"
        shas = {m: self._seed_recording(db, media, f"{pair.id}:{m}", speaker=speaker,
                                        kind=self.rendition_speaker_kind)
                for m in pair.members}
        db.append(port="assess", backend="rendition",
                  key=MechanicalKey(check="rendition", params="v1", subject=str(pair.id),
                                    artifact_sha=rendition_identity(shas)),
                  subject=pair.id,
                  question={"role": "rendition-for-pair",
                            "artifact_sha": rendition_identity(shas), "rubric": None,
                            "kind": "rendition", "subject_kind": "pair",
                            "params": {"members": shas}},
                  answer={"value": True})
        return shas


def copy_of(builder: DeckBuilder) -> DeckBuilder:
    return copy.deepcopy(builder)
