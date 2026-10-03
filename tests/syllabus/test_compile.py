"""compile.compile_syllabus (spec 4): models, fields,
guids, tags, due, gate refusal, and dropped-card counting, against a small
synthetic Syllabus compiled through a real SyllabusDb + MediaStore (in a
tmp_path) end to end -- reading the produced .apkg back with the same
"read a real collection.anki2" pattern scripts/proof_gallery.py and
.helpers_apkg.py use.

The `ยา`/`โรงพยาบาล` (medicine/hospital) substring-corruption case is
table-tested directly against `thai_cloze`, without a full compile.
"""
import dataclasses
from datetime import date

import pytest

from thai_syllabus.authority import ROLE_FOR_KIND
from thai_syllabus.cachekeys import JudgeKey, MechanicalKey, ProvideKey, rendition_identity
from thai_syllabus.compile import (
    CLOZE_SLOTS, GateRefusal, SENTENCE_MODEL, STRIDE, WORD_MODEL, _TEMPLATE_DROP_CAUSES,
    compile_syllabus, thai_cloze,
)
from thai_syllabus.entities import (
    Grapheme, MinimalPair, Pronunciation, REPEAT_MARK, Sentence, SoundConfusion, Syllable,
    Target, Word, render,
)
from thai_syllabus.ids import ConfusionId, PairId, TargetId, WordId, sentence_cloze_key
from thai_syllabus.media import Provenance, Speaker
from thai_syllabus.profile import Profile
from thai_syllabus.rulebook import RULES, sentence_note_id
from thai_syllabus.rules import DroppedCard
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.wiring import _DbMediaIndex

from tests.spec.syllabus_world import read_apkg

PROV = Provenance(source="test", origin="fixture", licence="cc0",
                  acquired=date(2026, 1, 1))

# This module's fixtures build a Syllabus whose `media` port is the default
# NullMediaIndex -- artifact resolution for the actual compiled cards goes
# straight through fx.db/fx.media (Fixture.seed_picture/seed_recording), not
# through Syllabus.media, so these completeness ERROR rules (spec 4) would
# always fire here regardless of what's seeded, closing the gate on every
# fixture that has targets or sentences. This module's subject is
# compile.py's own field/guid/due/dropped-card mechanics, not target
# completeness, so these are dropped from the default rules; everything
# else (closure, exact-confusion, ...) still runs.
#
# card/unique-front is dropped too: `_small_syllabus`'s one sentence fills
# three targets, so its "Listening" cards ({{Audio}} alone as the front)
# legitimately share one front across those targets -- a real content
# question this module's mechanics-only fixtures aren't meant to raise;
# the dedicated card/unique-front tests build their own syllabus with it
# enabled.
_COMPLETENESS_ERROR_IDS = {
    "target/picture-required", "target/recording-required", "target/sentence-required",
    "pair/rendition-required", "grapheme/keyword-picture-required",
    "sentence/recording-required", "card/unique-front",
}
_RULES_WITHOUT_COMPLETENESS = tuple(r for r in RULES if r.id not in _COMPLETENESS_ERROR_IDS)


def _syl(onset="m", vowel="a", coda="", length="short", tone="mid") -> Syllable:
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def _pron(*syllables, corroboration="engines_agree") -> Pronunciation:
    return Pronunciation(syllables=tuple(syllables) or (_syl(),), corroboration=corroboration)


def _word(id_, thai, meaning, tone="mid") -> Word:
    return Word(id=WordId(id_), thai=thai, pron=_pron(_syl(tone=tone)), meaning=meaning)


def _sentence(words: tuple[Word, ...], clauses, *, gloss: str,
             voice: str = "learner_voice") -> Sentence:
    """Sentence.text as entities.render over `clauses`, resolving each
    element's word id against `words`' own thai forms.
    """
    thai_of = {w.id: w.thai for w in words}.__getitem__
    return Sentence(clauses=clauses, text=render(clauses, thai_of), gloss=gloss,
                    voice=voice, provenance=PROV)


# --- fixture: a small synthetic Syllabus, seeded db + media store --------

class Fixture:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.db = SyllabusDb(tmp_path / "syllabus.db")
        self.media = MediaStore(tmp_path / "media")
        self.out_path = tmp_path / "out" / "deck.apkg"

    def _pass_judge(self, subject: str, kind: str, sha: str) -> None:
        # derivations.current_best only promotes a candidate that has EITHER
        # a learner rating or a passing judge verdict (spec 3 section 3) --
        # a bare provide row alone is just an untried candidate. Seed a
        # trivial passing judge verdict so compile.py's current_best lookups
        # resolve these fixture artifacts.
        role = ROLE_FOR_KIND.get(kind, kind)
        self.db.append(port="assess", backend="judge",
                       key=JudgeKey.for_rule("seed", sha, subject, role), subject=subject,
                       question={"role": role, "artifact_sha": sha,
                                "rubric": "seed", "kind": kind},
                       answer={"value": True})

    def seed_recording(self, subject: str, text: str, speaker="somchai") -> str:
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

    def seed_picture(self, subject: str, text: str, content: bytes | None = None) -> str:
        sha = self.media.write(content or f"image:{subject}:{text}".encode(), ext="jpg")
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
                       speaker="somchai") -> dict[str, str]:
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


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


def _small_syllabus(extra_targets=()) -> Syllabus:
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

    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                               consonant_class="mid", keyword_word=chicken,
                               name_word=ko_name)

    targets = [
        Target(id=TargetId("pom/receptive"), word=pom.id, skill="receptive"),
        Target(id=TargetId("gin/receptive"), word=gin.id, skill="receptive"),
        Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive"),
        Target(id=TargetId("rice/productive"), word=rice.id, skill="productive"),
        *extra_targets,
    ]

    sentence = _sentence((pom, gin, rice), ((pom.id, gin.id, rice.id),),
                        gloss="I eat rice")  # I eat rice

    return Syllabus(
        words=(pom, gin, rice, chicken, ko_name, near, far),
        targets=tuple(targets),
        pairs=(pair,),
        graphemes=(grapheme,),
        sentences=(sentence,),
        confusions=(confusion,),
        profile=Profile(register="male_colloquial"),
        rules=_RULES_WITHOUT_COMPLETENESS,
    )


def _fully_seeded(fx) -> Syllabus:
    syllabus = _small_syllabus()

    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("letter-name:ko", "gɔɔ")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")
    text_sha = sentence_note_id(syllabus.sentences[0])
    fx.seed_recording(text_sha, "ผมกินข้าว")
    fx.seed_picture(text_sha, "a man eating rice")
    return syllabus


# --- thai_cloze: renders over clauses, blanking by element word identity --

def test_thai_cloze_blanks_the_target_word_only():
    pom = _word("pom", "ผม", "I")
    gin = _word("gin", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    words = (pom, gin, rice)
    clauses = ((pom.id, gin.id, rice.id),)
    thai_of = {w.id: w.thai for w in words}.__getitem__
    sentence = Sentence(clauses=clauses, text=render(clauses, thai_of),
                        gloss="I eat rice", voice="learner_voice", provenance=PROV)
    assert thai_cloze(sentence, rice.id, thai_of) == "ผมกิน___"  # I eat ___


def test_thai_cloze_blanks_every_occurrence_of_the_target_word():
    med = _word("med", "ยา", "medicine")
    pom = _word("pom", "ผม", "I")
    words = (pom, med)
    clauses = ((pom.id, med.id), (med.id,))  # I take medicine, medicine
    thai_of = {w.id: w.thai for w in words}.__getitem__
    sentence = Sentence(clauses=clauses, text=render(clauses, thai_of),
                        gloss="I take medicine, medicine", voice="learner_voice",
                        provenance=PROV)
    assert thai_cloze(sentence, med.id, thai_of) == "ผม___ ___"  # I take ___, ___


def test_thai_cloze_keeps_the_repeat_mark_outside_the_blank():
    run = _word("run", "วิ่ง", "run")
    fast = _word("fast", "เร็ว", "fast")
    words = (run, fast)
    clauses = ((run.id, (fast.id, REPEAT_MARK)),)  # run fast-fast (reduplicated)
    thai_of = {w.id: w.thai for w in words}.__getitem__
    sentence = Sentence(clauses=clauses, text=render(clauses, thai_of),
                        gloss="run fast", voice="learner_voice", provenance=PROV)
    assert thai_cloze(sentence, fast.id, thai_of) == "วิ่ง___ๆ"  # run ___-___ (reduplicated)


def test_thai_cloze_leaves_a_word_whose_form_is_a_substring_of_another_registered_words_form_untouched():
    """"โรง" (building) is a substring of "โรงพยาบาล" (hospital), a
    distinct registered Word -- the corruption class thai_cloze cannot
    produce by construction: an element names its own word, never a
    substring of another word's rendered form (spec 4 section 1).
    """
    building = _word("building", "โรง", "building")
    pom = _word("pom", "ผม", "I")
    go = _word("go", "ไป", "go")
    hospital = _word("hospital", "โรงพยาบาล", "hospital")
    words = (building, pom, go, hospital)
    clauses = ((pom.id, go.id, hospital.id),)  # I go to the hospital
    thai_of = {w.id: w.thai for w in words}.__getitem__
    sentence = Sentence(clauses=clauses, text=render(clauses, thai_of),
                        gloss="I go to the hospital", voice="learner_voice", provenance=PROV)
    assert thai_cloze(sentence, building.id, thai_of) == sentence.text


# --- compile agrees with the rulebook about a stale rubric (F1 defect 4) --

def test_resolver_agrees_with_the_rulebook_about_a_stale_rubric_verdict(fx):
    """compile._Resolver must resolve current-best under the same
    current_rubric derivations.current_best does: a verdict under a
    rubric the deck has since moved off must not be current-best for
    compile when it is not for the rulebook either, on the very same db.
    """
    from thai_syllabus.compile import _Resolver
    from thai_syllabus.derivations import current_best

    fx.seed_picture("rice", "cooked rice")  # judged under rubric "seed"
    current_rubric = {"picture-for-word": "a rubric the deck has since moved on to"}

    rulebook_best = current_best(fx.db, "rice", "picture", current_rubric=current_rubric,
                                 prior=(), provenance_source=lambda s: None)
    resolver = _Resolver(db=fx.db, media_store=fx.media, current_rubric=current_rubric,
                         prior=(), provenance_source=lambda s: None)
    compile_best = resolver.artifact("rice", "picture")

    assert rulebook_best.artifact_sha is None   # stale -- current_best refuses it
    assert compile_best is None                 # compile agrees: nothing to stage


# --- compile: gate refusal ------------------------------------------------

def test_compile_refuses_when_the_gate_is_closed(fx):
    from thai_syllabus.rules import Rule, Finding

    def always_fails(s):
        return [Finding(rule="test/always-fails", note_id="x", evidence="bad")]

    rule = Rule(id="test/always-fails", principle="F1", severity="error",
               shape="check", check=always_fails)
    syllabus = _small_syllabus()
    syllabus = syllabus_with_rules(syllabus, (rule,))
    with pytest.raises(GateRefusal):
        compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert not fx.out_path.exists()


def test_gate_refusal_counts_unwaived_errors_only(fx):
    from thai_syllabus.rules import Rule, Finding

    def one_error(s):
        return [Finding(rule="test/one-error", note_id="x", evidence="bad")]

    def one_warn(s):
        return [Finding(rule="test/one-warn", note_id="y", evidence="minor")]

    rules = (
        Rule(id="test/one-error", principle="F1", severity="error",
            shape="check", check=one_error),
        Rule(id="test/one-warn", principle="F1", severity="warn",
            shape="check", check=one_warn),
    )
    syllabus = syllabus_with_rules(_small_syllabus(), rules)
    with pytest.raises(GateRefusal) as excinfo:
        compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert excinfo.value.blocking == 1


# _RULES_WITHOUT_COMPLETENESS plus card/unique-front (dropped from it
# above) -- every other completeness check still excluded, so a fixture
# with no picture, no recording index, and no sentence stays gate-clean
# apart from the duplicate front itself.
_RULES_FOR_UNIQUE_FRONT = _RULES_WITHOUT_COMPLETENESS + (
    next(r for r in RULES if r.id == "card/unique-front"),)


def _duplicate_front_syllabus() -> Syllabus:
    # Two distinct words sharing one Thai spelling -- their Reading
    # template fronts ("{{Thai}}" alone) render identically.
    rice_a = _word("rice-a", "ข้าว", "cooked rice (a)")
    rice_b = _word("rice-b", "ข้าว", "cooked rice (b)")
    targets = (Target(id=TargetId("rice-a/receptive"), word=rice_a.id, skill="receptive"),
              Target(id=TargetId("rice-b/receptive"), word=rice_b.id, skill="receptive"))
    return Syllabus(words=(rice_a, rice_b), targets=targets,
                    rules=_RULES_FOR_UNIQUE_FRONT)


def test_compile_refuses_on_duplicate_card_fronts(fx):
    syllabus = _duplicate_front_syllabus()
    fx.seed_recording("rice-a", "recording a")
    fx.seed_recording("rice-b", "recording b")
    with pytest.raises(GateRefusal) as excinfo:
        compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert not fx.out_path.exists()
    assert any(f.rule == "card/unique-front" for f in excinfo.value.report.findings)


def test_compile_forced_past_duplicate_fronts_reports_the_finding_and_writes(fx):
    syllabus = _duplicate_front_syllabus()
    fx.seed_recording("rice-a", "recording a")
    fx.seed_recording("rice-b", "recording b")
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path, force=True,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert fx.out_path.exists()
    assert compiled.report.gate is False
    assert any(f.rule == "card/unique-front" for f in compiled.report.findings)
    assert any("card/unique-front" in w for w in compiled.report.warnings)


def test_compile_with_distinct_fronts_reports_no_unique_front_finding(fx):
    # Same setup as _duplicate_front_syllabus but with distinct Thai
    # spellings -- no two Reading fronts collide.
    rice = _word("rice", "ข้าว", "cooked rice")
    dog = _word("dog", "หมา", "dog")
    targets = (Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive"),
              Target(id=TargetId("dog/receptive"), word=dog.id, skill="receptive"))
    syllabus = Syllabus(words=(rice, dog), targets=targets,
                        rules=_RULES_FOR_UNIQUE_FRONT)
    fx.seed_recording("rice", "recording rice")
    fx.seed_recording("dog", "recording dog")
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert compiled.report.gate is True
    assert compiled.report.findings == ()


def test_compile_forced_past_a_closed_gate_records_declared_warnings(fx):
    from thai_syllabus.rules import Rule, Finding

    def always_fails(s):
        return [Finding(rule="test/always-fails", note_id="x", evidence="bad thing")]

    rule = Rule(id="test/always-fails", principle="F1", severity="error",
               shape="check", check=always_fails)
    syllabus = _fully_seeded(fx)
    syllabus = syllabus_with_rules(syllabus, (rule,))
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path, force=True,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert compiled.report.forced is True
    assert compiled.report.gate is False
    assert any("bad thing" in w for w in compiled.report.warnings)
    assert fx.out_path.exists()


def syllabus_with_rules(syllabus: Syllabus, rules) -> Syllabus:
    import dataclasses
    return dataclasses.replace(syllabus, rules=tuple(rules))


# --- compile: full pipeline over a fully-seeded fixture -------------------

def test_compile_writes_an_apkg_and_stamps_compile_id_everywhere(fx):
    syllabus = _fully_seeded(fx)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert fx.out_path.exists()
    assert compiled.report.gate is True
    assert compiled.report.forced is False

    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    field_index = {}
    for mid, model in models.items():
        names = [f["name"] for f in model["flds"]]
        field_index[model["name"]] = {n: i for i, n in enumerate(names)}

    for note in pkg["notes"]:
        model = models[str(note["mid"])]
        idx = field_index[model["name"]]["CompileId"]
        assert note["flds"][idx] == compiled.compile_id


def test_word_note_has_expected_fields_guid_and_tags(fx):
    syllabus = _fully_seeded(fx)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]

    word_model = next(m for m in models.values() if m["name"] == "word")
    field_names = [f["name"] for f in word_model["flds"]]
    rice_note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                     and dict(zip(field_names, n["flds"]))["Thai"] == "ข้าว")
    fields = dict(zip(field_names, rice_note["flds"]))

    assert fields["Meaning"] == "cooked rice"
    assert fields["Picture"].startswith("<img")
    assert fields["Audio"].startswith("[sound:")
    assert fields["ProductiveTarget"] == "1"  # rice has a productive Target

    import genanki
    assert rice_note["guid"] == genanki.guid_for("word", "rice")

    tags = rice_note["tags"].split(" ")
    assert "family::word" in tags
    assert "word::rice" in tags
    assert not any(t.startswith("target::") for t in tags)
    assert f"compile::{compiled.compile_id}" in tags


def test_word_note_production_card_is_dropped_without_a_productive_target(fx):
    # pom/gin only have RECEPTIVE targets -- Production must not appear,
    # even though both have a recording (word cards don't need a picture
    # to generate Listening/Reading).
    syllabus = _fully_seeded(fx)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    word_model = next(m for m in models.values() if m["name"] == "word")
    field_names = [f["name"] for f in word_model["flds"]]
    pom_note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                    and dict(zip(field_names, n["flds"]))["Thai"] == "ผม")

    tmpl_names = [t["name"] for t in word_model["tmpls"]]
    pom_cards = [c for c in pkg["cards"] if c["nid"] == pom_note["id"]]
    generated = {tmpl_names[c["ord"]] for c in pom_cards}
    assert "Production" not in generated

    dropped_kinds = {(d.family, d.kind, d.subject) for d in compiled.report.dropped}
    assert ("word", "Production", "pom") in dropped_kinds


def test_a_sentence_introduced_word_compiles_no_word_note(fx):
    # A word note is compiled only for a word with a picture-introduced
    # Target (spec 4 section 1) -- eat's Target is introduction="sentence"
    # (a sentence-introduced word), so eat compiles no word note; it is
    # carried by its sentence note instead. rice's Target is the default
    # picture_card, so rice DOES compile a word note.
    rice = _word("rice", "ข้าว", "cooked rice")
    eat = _word("eat", "กิน", "to eat")
    rice_target = Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive")
    eat_target = Target(id=TargetId("eat/receptive"), word=eat.id, skill="receptive",
                        introduction="sentence")
    kin_khaao = _sentence((eat, rice), ((eat.id, rice.id),), gloss="eat rice")  # eat rice
    syllabus = Syllabus(words=(rice, eat), targets=(rice_target, eat_target),
                        sentences=(kin_khaao,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("eat", "to eat")
    fx.seed_recording(sentence_note_id(kin_khaao), "กินข้าว")

    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    word_model = next(m for m in models.values() if m["name"] == "word")
    field_names = [f["name"] for f in word_model["flds"]]
    word_notes = [n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]]
    thai_values = {dict(zip(field_names, n["flds"]))["Thai"] for n in word_notes}
    assert "ข้าว" in thai_values   # rice: picture-introduced, compiles a word note
    assert "กิน" not in thai_values  # eat: sentence-introduced, no word note

    s_model = next(m for m in models.values() if m["name"] == "sentence")
    s_notes = [n for n in pkg["notes"] if str(n["mid"]) == s_model["id"]]
    assert len(s_notes) == 1
    tags = s_notes[0]["tags"].split(" ")
    assert "target::eat/receptive" in tags  # still carried by the sentence note


def test_dropped_reason_distinguishes_gate_from_missing(fx):
    # pom/gin are gated out (no productive Target, so no productive test);
    # rice has a productive Target and every artifact seeded -- nothing
    # drops it. The two pom reasons name the gate, not a missing artifact.
    syllabus = _fully_seeded(fx)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    reasons = {(d.subject, d.kind): d.reason for d in compiled.report.dropped}
    assert reasons[("pom", "Production")] == "gated: no productive Target"
    assert reasons[("pom", "Spelling")] == "gated: spelling not tested"
    assert ("rice", "Production") not in reasons
    assert ("rice", "Spelling") not in reasons


def test_word_note_listening_dropped_and_counted_when_audio_is_missing(fx):
    syllabus = _small_syllabus()
    # Deliberately do NOT seed pom's recording.
    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("letter-name:ko", "gɔɔ")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    word_model = next(m for m in models.values() if m["name"] == "word")
    field_names = [f["name"] for f in word_model["flds"]]
    pom_note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                    and dict(zip(field_names, n["flds"]))["Thai"] == "ผม")
    tmpl_names = [t["name"] for t in word_model["tmpls"]]
    pom_cards = [c for c in pkg["cards"] if c["nid"] == pom_note["id"]]
    generated = {tmpl_names[c["ord"]] for c in pom_cards}
    assert "Listening" not in generated
    assert "Reading" in generated  # text-only front, never media-gated

    dropped_kinds = {(d.family, d.kind, d.subject) for d in compiled.report.dropped}
    assert ("word", "Listening", "pom") in dropped_kinds

    reasons = {(d.subject, d.kind): d.reason for d in compiled.report.dropped}
    assert reasons[("pom", "Listening")] == "no current-best recording"
    assert reasons[("pom", "Production")] == "gated: no productive Target"
    assert reasons[("pom", "Spelling")] == "gated: spelling not tested"


def test_word_spelling_dropped_for_missing_recording_when_productive(fx):
    # rice has a productive Target (TestSpelling is truthy) but no
    # recording seeded -- Listening and Spelling both drop for the
    # missing artifact, not the gate; Production (gated only by
    # ProductiveTarget, never by Audio) still generates.
    syllabus = _small_syllabus()
    fx.seed_picture("rice", "cooked rice")
    # Deliberately do NOT seed rice's recording.
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("letter-name:ko", "gɔɔ")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    word_model = next(m for m in models.values() if m["name"] == "word")
    field_names = [f["name"] for f in word_model["flds"]]
    rice_note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                     and dict(zip(field_names, n["flds"]))["Thai"] == "ข้าว")
    tmpl_names = [t["name"] for t in word_model["tmpls"]]
    rice_cards = [c for c in pkg["cards"] if c["nid"] == rice_note["id"]]
    generated = {tmpl_names[c["ord"]] for c in rice_cards}
    assert "Listening" not in generated
    assert "Spelling" not in generated
    assert "Production" in generated

    reasons = {(d.subject, d.kind): d.reason for d in compiled.report.dropped}
    assert reasons[("rice", "Listening")] == "no current-best recording"
    assert reasons[("rice", "Spelling")] == "no current-best recording"


def test_word_production_card_is_dropped_without_a_picture(fx):
    # rice has a productive Target and a recording, but no current-best
    # picture -- the Production front is {{Picture}} once
    # {{#ProductiveTarget}} gates it open, so genanki's own required-field
    # computation must not settle for ProductiveTarget alone: a missing
    # current-best artifact drops the dependent card, never an empty
    # front (spec 4 section 3; section 1: "productive Target and a
    # current-best picture; no picture, no card"). Listening/Reading are
    # unaffected -- their fronts don't reference Picture.
    syllabus = _small_syllabus()
    # Deliberately do NOT seed rice's picture.
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("letter-name:ko", "gɔɔ")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    word_model = next(m for m in models.values() if m["name"] == "word")
    field_names = [f["name"] for f in word_model["flds"]]
    rice_note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                     and dict(zip(field_names, n["flds"]))["Thai"] == "ข้าว")
    fields = dict(zip(field_names, rice_note["flds"]))
    tmpl_names = [t["name"] for t in word_model["tmpls"]]
    rice_cards = [c for c in pkg["cards"] if c["nid"] == rice_note["id"]]
    generated = {tmpl_names[c["ord"]] for c in rice_cards}

    # The bug this pins: no compiled Production card ever has an empty
    # Picture front (23 such cards in the live deck, e.g. นี่, อะไร).
    assert not ("Production" in generated and fields["Picture"] == "")
    assert "Production" not in generated
    assert "Listening" in generated
    assert "Reading" in generated

    dropped_kinds = {(d.family, d.kind, d.subject) for d in compiled.report.dropped}
    assert ("word", "Production", "rice") in dropped_kinds
    reasons = {(d.subject, d.kind): d.reason for d in compiled.report.dropped}
    assert reasons[("rice", "Production")] == "no current-best picture"


def test_every_word_and_sentence_template_has_a_registered_drop_cause():
    # A renamed or added template with no entry must fail loudly (a
    # KeyError from _template_drop_reason), not silently report a
    # generic reason -- this pins that every template genanki can
    # actually build for these two models is covered.
    for model in (WORD_MODEL, SENTENCE_MODEL):
        for tpl in model.templates:
            assert (model.name, tpl["name"]) in _TEMPLATE_DROP_CAUSES


def test_grapheme_note_name_thai_is_the_name_words_own_text(fx):
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    g_model = next(m for m in models.values() if m["name"] == "grapheme")
    field_names = [f["name"] for f in g_model["flds"]]
    note = next(n for n in pkg["notes"] if str(n["mid"]) == g_model["id"])
    fields = dict(zip(field_names, note["flds"]))
    assert fields["NameThai"] == "กอ ไก่"  # "gɔɔ gài" -- the name word's own text
    assert fields["KeywordThai"] == "ไก่"  # chicken
    assert fields["Audio"].startswith("[sound:")

    import genanki
    assert note["guid"] == genanki.guid_for("grapheme", "ก")


def test_grapheme_without_a_name_recording_is_dropped_not_substituted(fx):
    chicken = _word("chicken", "ไก่", "chicken")
    ko_name = _word("letter-name:ko", "กอ ไก่", "the letter ก (recited name)")
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                               consonant_class="mid", keyword_word=chicken,
                               name_word=ko_name)
    syllabus = Syllabus(words=(chicken, ko_name), graphemes=(grapheme,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("chicken", "chicken")
    # NOT seeding letter-name:ko's recording -- the keyword's recording
    # must never substitute for it.

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert compiled.report.dropped == (
        DroppedCard(family="grapheme", kind="Reading", subject="ก",
                   reason="no name recording"),)

    pkg = read_apkg(fx.out_path)
    assert not any("family::grapheme" in n["tags"] for n in pkg["notes"])


def test_grapheme_without_a_name_word_is_dropped_not_fabricated(fx):
    chicken = _word("chicken", "ไก่", "chicken")
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                               consonant_class="mid", keyword_word=chicken)
    rice = _word("rice", "ข้าว", "cooked rice")
    pom = _word("pom", "ผม", "I")
    gin = _word("gin", "กิน", "to eat")
    targets = [
        Target(id=TargetId("pom/receptive"), word=pom.id, skill="receptive"),
        Target(id=TargetId("gin/receptive"), word=gin.id, skill="receptive"),
        Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive"),
    ]
    syllabus = Syllabus(words=(pom, gin, rice, chicken), targets=tuple(targets),
                        graphemes=(grapheme,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_recording("chicken", "chicken")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    dropped_kinds = {(d.family, d.kind, d.subject): d.reason for d in compiled.report.dropped}
    assert dropped_kinds[("grapheme", "Reading", "ก")] == "no name word"

    pkg = read_apkg(fx.out_path)
    assert not any("family::grapheme" in n["tags"] for n in pkg["notes"])


# --- compile: minimal_pair notes play the pair's rendition ---------------

def _pair_only_syllabus() -> tuple[Syllabus, MinimalPair]:
    near = _word("near", "ใกล้", "near", tone="mid")
    far = _word("far", "ไกล", "far", tone="low")
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = MinimalPair.create(id=PairId("p1"), confusion=confusion, members=(near, far))
    syllabus = Syllabus(words=(near, far), pairs=(pair,), confusions=(confusion,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    return syllabus, pair


def _compile_pair(fx, *, with_rendition: bool):
    """Compiles a Syllabus holding one pair (p1: near/far), its `media`
    port a real _DbMediaIndex over `fx.db`; seeds a passing rendition
    first unless `with_rendition` is False. Returns (compiled, shas, notes)
    where `shas` is member id -> the seeded rendition's sha (empty when
    none was seeded) and each of `notes` is {fields, tags, due}.
    """
    syllabus, pair = _pair_only_syllabus()
    shas = fx.seed_rendition(pair, {"near": "near", "far": "far"}) if with_rendition else {}
    syllabus = dataclasses.replace(syllabus, media=_DbMediaIndex(db=fx.db, pairs=(pair,)))
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    pair_model = next((m for m in pkg["models"].values() if m["name"] == "minimal_pair"), None)
    if pair_model is None:
        return compiled, shas, []
    field_names = [f["name"] for f in pair_model["flds"]]
    raw_notes = [n for n in pkg["notes"] if str(n["mid"]) == pair_model["id"]]
    notes = [{"fields": dict(zip(field_names, n["flds"])),
             "tags": n["tags"].split(" "),
             "due": min(c["due"] for c in pkg["cards"] if c["nid"] == n["id"])}
            for n in raw_notes]
    return compiled, shas, notes


def test_minimal_pair_notes_one_per_member_with_playable_audio_both_sides(fx):
    compiled, shas, notes = _compile_pair(fx, with_rendition=True)
    assert len(notes) == 2  # one per member

    by_stimulus = {n["fields"]["Stimulus"]: n for n in notes}
    near = by_stimulus["ใกล้"]  # near
    assert near["fields"]["MemberKey"].startswith("p1:")
    assert near["fields"]["Audio"] == f"[sound:{shas['near']}.mp3]"
    assert near["fields"]["OtherAudio"] == f"[sound:{shas['far']}.mp3]"

    for n in notes:
        assert n["fields"]["CompileId"] == compiled.compile_id
        assert "family::minimal_pair" in n["tags"]
        assert "confusion::tone:mid-low" in n["tags"]
        assert "pair::p1" in n["tags"]


def test_pair_notes_use_the_rendition_not_member_recordings(fx):
    _compiled, shas, notes = _compile_pair(fx, with_rendition=True)
    audio = {n["fields"]["Audio"] for n in notes}
    assert audio == {f"[sound:{shas['near']}.mp3]", f"[sound:{shas['far']}.mp3]"}
    assert len({n["fields"]["Speaker"] for n in notes}) == 1


def test_pair_without_a_rendition_is_dropped_and_counted(fx):
    compiled, _shas, notes = _compile_pair(fx, with_rendition=False)
    assert notes == []
    assert compiled.report.dropped == (
        DroppedCard(family="minimal_pair", kind="Recognition", subject="p1",
                   reason="no rendition"),)


def test_pair_choices_are_in_member_order_on_both_notes(fx):
    _compiled, _shas, notes = _compile_pair(fx, with_rendition=True)
    choices = {n["fields"]["Choices"] for n in notes}
    assert choices == {"ใกล้ / ไกล"}  # near / far, in member order on every note


def test_pair_member_notes_are_not_adjacent(fx):
    _compiled, _shas, notes = _compile_pair(fx, with_rendition=True)
    dues = sorted(n["due"] for n in notes)
    assert dues[1] - dues[0] >= STRIDE


def test_two_pair_blocks_and_a_following_word_target_never_overlap(fx):
    # order() places pairs (sorted by id) before word targets: pA, then
    # pB, then chicken/receptive -- pA's and pB's due blocks must each be
    # wide enough for their own members (width = member count) before the
    # next entry's block starts, or pB's/chicken's dues would collide with
    # pA's/pB's own member dues (the fix round 1 regression).
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    near = _word("near", "ใกล้", "near", tone="mid")
    far = _word("far", "ไกล", "far", tone="low")
    pair_a = MinimalPair.create(id=PairId("pA"), confusion=confusion, members=(near, far))
    dog = _word("dog", "หมา", "dog", tone="mid")        # dog
    horse = _word("horse", "ม้า", "horse", tone="low")  # horse
    pair_b = MinimalPair.create(id=PairId("pB"), confusion=confusion, members=(dog, horse))
    chicken = _word("chicken", "ไก่", "chicken")
    target = Target(id=TargetId("chicken/receptive"), word=chicken.id, skill="receptive")

    syllabus = Syllabus(words=(near, far, dog, horse, chicken), targets=(target,),
                        pairs=(pair_a, pair_b), confusions=(confusion,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_rendition(pair_a, {"near": "near", "far": "far"})
    fx.seed_rendition(pair_b, {"dog": "dog", "horse": "horse"})
    syllabus = dataclasses.replace(
        syllabus, media=_DbMediaIndex(db=fx.db, pairs=(pair_a, pair_b)))

    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)

    pair_model = next(m for m in pkg["models"].values() if m["name"] == "minimal_pair")
    word_model = next(m for m in pkg["models"].values() if m["name"] == "word")
    word_field_names = [f["name"] for f in word_model["flds"]]

    def dues_of(note_id: str) -> list[int]:
        return [c["due"] for c in pkg["cards"] if c["nid"] == note_id]

    def pair_member_dues(pair_id: str) -> list[int]:
        notes = [n for n in pkg["notes"] if str(n["mid"]) == pair_model["id"]
                and n["flds"][0].split(":")[0] == pair_id]
        return [d for n in notes for d in dues_of(n["id"])]

    a_dues = pair_member_dues("pA")
    b_dues = pair_member_dues("pB")
    chicken_note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                        and dict(zip(word_field_names, n["flds"]))["Thai"] == "ไก่")  # chicken
    chicken_dues = dues_of(chicken_note["id"])

    assert len(a_dues) == len(b_dues) == 2  # one due per member, both pairs
    assert chicken_dues

    all_dues = a_dues + b_dues + chicken_dues
    assert len(all_dues) == len(set(all_dues))  # every due, across every block, is distinct
    assert max(a_dues) - min(a_dues) < STRIDE * len(pair_a.members)  # inside pA's own block
    assert max(b_dues) - min(b_dues) < STRIDE * len(pair_b.members)  # inside pB's own block
    assert max(a_dues) < min(b_dues)        # pA's block ends before pB's starts
    assert max(b_dues) < min(chicken_dues)  # pB's block ends before chicken's starts


def _sentence_notes(pkg, model_name: str) -> tuple[dict, list[str], list[dict]]:
    """(model, field names, notes) for the compiled model named `model_name`;
    an empty note list when the package has no such model.
    """
    model = next((m for m in pkg["models"].values() if m["name"] == model_name), None)
    if model is None:
        return {}, [], []
    field_names = [f["name"] for f in model["flds"]]
    return model, field_names, [n for n in pkg["notes"] if str(n["mid"]) == model["id"]]


def _templates_generated(pkg, model, note) -> set[str]:
    tmpl_names = [t["name"] for t in model["tmpls"]]
    return {tmpl_names[c["ord"]] for c in pkg["cards"] if c["nid"] == note["id"]}


def _target_tags(note) -> set[str]:
    return {t.split("::", 1)[1] for t in note["tags"].split(" ") if t.startswith("target::")}


def _note_of(pkg, sentence: Sentence) -> tuple[dict, list[str], dict]:
    """(sentence model, its field names, the sentence note tagged
    sentence::SHA for `sentence`).
    """
    model, field_names, notes = _sentence_notes(pkg, "sentence")
    tag = f"sentence::{sentence_note_id(sentence)}"
    (note,) = [n for n in notes if tag in n["tags"].split(" ")]
    return model, field_names, note


def _cards_by_ord(pkg, note) -> dict[int, dict]:
    return {c["ord"]: c for c in pkg["cards"] if c["nid"] == note["id"]}


def test_sentence_note_carries_its_listening_card_and_its_cloze_card(fx):
    # "ผมกินข้าว" (I eat rice) fills pom/receptive, gin/receptive,
    # rice/receptive and rice/productive: one sentence note (spec 4 r11)
    # tagged with every one of them, carrying the Listening card and the
    # Cloze card of rice/productive in rice's slot, the sentence's third
    # distinct word.
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, field_names, s_notes = _sentence_notes(pkg, "sentence")
    assert len(s_notes) == 1
    note = s_notes[0]
    assert _target_tags(note) == {"pom/receptive", "gin/receptive",
                                  "rice/receptive", "rice/productive"}
    assert {"kind::listening", "kind::cloze"} <= set(note["tags"].split(" "))

    fields = dict(zip(field_names, note["flds"]))
    assert fields["Thai"] == "ผมกินข้าว"
    assert fields["TargetWord"] == "ข้าว"   # the sentence's target words: rice, its one productive word
    assert fields["Audio"].startswith("[sound:")
    assert fields["ScenePicture"].startswith("<img ")
    assert fields["Gloss"] == "I eat rice"
    assert (fields["Cloze3"], fields["ClozeWord3"], fields["ClozeTarget3"]) == \
        ("ผมกิน___", "ข้าว", "rice/productive")
    assert [fields[f"Cloze{k}"] for k in (1, 2)] == ["", ""]
    assert _templates_generated(pkg, s_model, note) == {"Listening", "Cloze 3"}
    assert _sentence_notes(pkg, "sentence_cloze")[2] == []


def _two_productive_words(fx, *, rice_productive: bool = True):
    """"กินข้าว" (eat rice): eat and rice each carry a receptive and a
    productive Target (rice's productive one unless `rice_productive` is
    False), so the sentence fills productive Targets on both its words
    (spec 1 r25). Seeds the sentence's recording and scene picture.
    """
    eat = _word("eat", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    skills = {eat: ("receptive", "productive"),
              rice: ("receptive", "productive") if rice_productive else ("receptive",)}
    targets = tuple(Target(id=TargetId(f"{w.id}/{skill}"), word=w.id, skill=skill)
                    for w, ws in skills.items() for skill in ws)
    kin_khaao = _sentence((eat, rice), ((eat.id, rice.id),), gloss="eat rice")  # eat rice
    syllabus = Syllabus(words=(eat, rice), targets=targets, sentences=(kin_khaao,),
                        frequency={eat.id: 1, rice.id: 2},
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_recording("eat", "to eat")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording(sentence_note_id(kin_khaao), "กินข้าว")
    fx.seed_picture(sentence_note_id(kin_khaao), "a man eating rice")
    return syllabus, kin_khaao


def test_a_sentence_filling_two_productive_targets_compiles_one_note_with_a_cloze_card_in_each_words_slot(fx):
    # Spec 4 r11: one sentence note carries the Listening card (ord 0)
    # and a Cloze card per productive Target the sentence fills, in the
    # slot of its word's position among the sentence's distinct words:
    # eat first (Cloze 1, ord 1), rice second (Cloze 2, ord 2).
    syllabus, kin_khaao = _two_productive_words(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)

    s_model, field_names, s_notes = _sentence_notes(pkg, "sentence")
    assert len(s_notes) == 1
    note = s_notes[0]
    tmpl_names = [t["name"] for t in s_model["tmpls"]]
    assert tmpl_names[:3] == ["Listening", "Cloze 1", "Cloze 2"]
    assert sorted(_cards_by_ord(pkg, note)) == [0, 1, 2]
    assert _sentence_notes(pkg, "sentence_cloze")[2] == []

    fields = dict(zip(field_names, note["flds"]))
    assert (fields["Cloze1"], fields["ClozeWord1"], fields["ClozeTarget1"]) == \
        ("___ข้าว", "กิน", "eat/productive")
    assert (fields["Cloze2"], fields["ClozeWord2"], fields["ClozeTarget2"]) == \
        ("กิน___", "ข้าว", "rice/productive")
    tags = set(note["tags"].split(" "))
    assert {"family::sentence", f"sentence::{sentence_note_id(kin_khaao)}",
            "kind::listening", "kind::cloze"} <= tags
    assert _target_tags(note) == {"eat/receptive", "eat/productive",
                                  "rice/receptive", "rice/productive"}


def _capped_eat(fx, cap: int):
    """"กิน" (eat) then "กินข้าว" (eat rice), placed in that order; eat
    and rice each carry a receptive and a productive Target and the
    production cap is `cap` sentences per word, so at 1 the second
    sentence loses eat/productive to the first and keeps rice/productive.
    """
    eat = _word("eat", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    targets = tuple(Target(id=TargetId(f"{w.id}/{skill}"), word=w.id, skill=skill)
                    for w in (eat, rice) for skill in ("receptive", "productive"))
    kin = _sentence((eat, rice), ((eat.id,),), gloss="eat")  # eat
    kin_khaao = _sentence((eat, rice), ((eat.id, rice.id),), gloss="eat rice")  # eat rice
    for w in (eat, rice):
        fx.seed_recording(w.id, w.meaning)
    for s in (kin, kin_khaao):
        fx.seed_recording(sentence_note_id(s), s.text)
        fx.seed_picture(sentence_note_id(s), s.gloss)
    syllabus = Syllabus(words=(eat, rice), targets=targets, sentences=(kin, kin_khaao),
                        frequency={eat.id: 1, rice.id: 2},
                        profile=Profile(register="male_colloquial",
                                        production_sentences_per_word=cap),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    return syllabus, kin_khaao


def test_a_cloze_card_keeps_its_ord_guid_and_due_when_another_fill_of_its_sentence_goes(fx):
    # Spec 4 r11: a (sentence, Target) pair's card sits in its word's
    # slot, so the cap taking eat/productive from "กินข้าว" leaves rice's
    # Cloze card at ord 2 with the same note GUID and due.
    both, kin_khaao = _capped_eat(fx, cap=2)
    assert [t.id for t in both.productive_fills(kin_khaao)] == ["eat/productive", "rice/productive"]
    with_both = _dues_by_card(fx, both)
    guid = _note_of(read_apkg(fx.out_path), kin_khaao)[2]["guid"]

    (fx.tmp_path / "capped").mkdir()
    fx_capped = Fixture(fx.tmp_path / "capped")
    capped, _ = _capped_eat(fx_capped, cap=1)
    assert [t.id for t in capped.productive_fills(kin_khaao)] == ["rice/productive"]
    with_rice = _dues_by_card(fx_capped, capped)
    pkg = read_apkg(fx_capped.out_path)
    _m, _f, note = _note_of(pkg, kin_khaao)

    assert note["guid"] == guid
    assert sorted(_cards_by_ord(pkg, note)) == [0, 2]
    assert with_rice[(guid, 2)] == with_both[(guid, 2)]
    assert with_rice[(guid, 0)] == with_both[(guid, 0)]
    assert (guid, 1) in with_both and (guid, 1) not in with_rice


def test_an_unfilled_slot_names_its_words_productive_target_and_builds_no_card(fx):
    # Spec 4 r11: the cap takes eat/productive from "กินข้าว", so slot 1
    # has no cloze and no card, and is no drop; its ClozeTarget still
    # names eat/productive, so a card Anki kept there maps to its pair.
    capped, kin_khaao = _capped_eat(fx, cap=1)
    compiled = compile_syllabus(capped, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    _m, field_names, note = _note_of(pkg, kin_khaao)
    fields = dict(zip(field_names, note["flds"]))
    assert (fields["Cloze1"], fields["ClozeWord1"], fields["ClozeTarget1"]) == \
        ("", "", "eat/productive")
    assert fields["ClozeTarget2"] == "rice/productive"
    assert [fields[f"ClozeTarget{k}"] for k in range(3, CLOZE_SLOTS + 1)] == [""] * (CLOZE_SLOTS - 2)
    assert sorted(_cards_by_ord(pkg, note)) == [0, 2]
    assert not [d for d in compiled.report.dropped if d.family == "sentence"]


def _slot_of_rice(fx, words, clauses) -> tuple[set[str], dict[str, str]]:
    """Compiles one sentence over `clauses` in which rice alone is
    productive: (its templates generated, its fields)."""
    rice = next(w for w in words if w.id == "rice")
    targets = (*(Target(id=TargetId(f"{w.id}/receptive"), word=w.id, skill="receptive")
                 for w in words if w is not rice),
               Target(id=TargetId("rice/productive"), word=rice.id, skill="productive"))
    sentence = _sentence(words, clauses, gloss="a sentence")
    syllabus = Syllabus(words=words, targets=targets, sentences=(sentence,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    for w in words:
        fx.seed_recording(w.id, w.meaning)
    fx.seed_recording(sentence_note_id(sentence), sentence.text)
    fx.seed_picture(sentence_note_id(sentence), sentence.gloss)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, field_names, note = _note_of(pkg, sentence)
    return _templates_generated(pkg, s_model, note), dict(zip(field_names, note["flds"]))


def test_a_cloze_slot_counts_a_repeated_word_once(fx):
    # "ผมกินยา ยาข้าว" (I take medicine, medicine rice): rice is the 4th
    # element but the 3rd distinct word, so its card is Cloze 3.
    pom, med, rice = (_word("pom", "ผม", "I"), _word("med", "ยา", "medicine"),
                      _word("rice", "ข้าว", "cooked rice"))
    generated, fields = _slot_of_rice(fx, (pom, med, rice), ((pom.id, med.id), (med.id, rice.id)))
    assert generated == {"Listening", "Cloze 3"}
    assert fields["ClozeTarget3"] == "rice/productive"


def test_a_cloze_slot_counts_a_word_under_the_repetition_mark_once(fx):
    # "วิ่งเร็วๆ เร็วข้าว" (run fast-fast, fast rice): the ๆ element and
    # the later plain เร็ว are one distinct word, so rice is Cloze 3 (the
    # 4th element, the 5th rendered unit counting ๆ).
    run, fast, rice = (_word("run", "วิ่ง", "run"), _word("fast", "เร็ว", "fast"),
                       _word("rice", "ข้าว", "cooked rice"))
    generated, fields = _slot_of_rice(fx, (run, fast, rice),
                                      ((run.id, (fast.id, REPEAT_MARK)), (fast.id, rice.id)))
    assert generated == {"Listening", "Cloze 3"}
    assert fields["Cloze3"].endswith("___")


def _four_rice_sentences(fx):
    """rice (ข้าว) carries a receptive and a productive Target; four
    adopted learner-voice sentences use it, placed ข้าว (rice), ข้าวไก่
    (rice, chicken), ข้าวหมู (rice, pork), ข้าวปลา (rice, fish). Seeds
    every recording and every sentence's scene picture.
    """
    rice = _word("rice", "ข้าว", "cooked rice")
    others = (_word("chicken", "ไก่", "chicken"), _word("pork", "หมู", "pork"),
              _word("fish", "ปลา", "fish"))
    words = (rice, *others)
    targets = (Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive"),
               Target(id=TargetId("rice/productive"), word=rice.id, skill="productive"),
               *(Target(id=TargetId(f"{w.id}/receptive"), word=w.id, skill="receptive")
                 for w in others))
    sentences = (_sentence(words, ((rice.id,),), gloss="rice"),
                 *(_sentence(words, ((rice.id, w.id),), gloss=f"rice, {w.meaning}")
                   for w in others))
    for w in words:
        fx.seed_recording(w.id, w.meaning)
    for s in sentences:
        fx.seed_recording(sentence_note_id(s), s.text)
        fx.seed_picture(sentence_note_id(s), s.gloss)
    return Syllabus(words=words, targets=targets, sentences=sentences,
                    frequency={w.id: n for n, w in enumerate(words, start=1)},
                    profile=Profile(register="male_colloquial"),
                    rules=_RULES_WITHOUT_COMPLETENESS)


def test_a_word_gets_at_most_three_cloze_cards(fx):
    """Spec 1 r26: four sentences use rice, the first three in placement
    order fill rice/productive, so compile emits three Cloze cards for it,
    each in rice's slot (the first) of its sentence's note."""
    syllabus = _four_rice_sentences(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    cards = {s.text: sorted(_cards_by_ord(pkg, _note_of(pkg, s)[2])) for s in syllabus.sentences}
    assert cards == {s.text: [0, 1] for s in syllabus.sentences[:3]} | {
        syllabus.sentences[3].text: [0]}


def test_a_sentence_notes_cards_fit_its_due_block():
    """A sentence's block holds its Listening card and every Cloze slot;
    a word's its four templates."""
    assert len(SENTENCE_MODEL.templates) == 1 + CLOZE_SLOTS
    assert len(SENTENCE_MODEL.templates) <= STRIDE
    assert len(WORD_MODEL.templates) <= STRIDE


def test_a_productive_fill_beyond_the_last_cloze_slot_is_dropped_and_counted(fx):
    # A sentence of CLOZE_SLOTS + 1 distinct words whose last word alone
    # is productive: that word has no slot, so its Cloze card is dropped
    # with its own reason and the Listening card still compiles.
    thai = ["ผม", "กิน", "ข้าว", "ไก่", "หมู", "ปลา", "น้ำ", "ชา", "นม", "ไข่", "ผัก", "แกง",
            "ส้ม", "กุ้ง", "เกลือ", "พริก"]
    assert len(thai) > CLOZE_SLOTS
    words = tuple(_word(f"w{i}", t, f"word {i}") for i, t in enumerate(thai[:CLOZE_SLOTS + 1]))
    last = words[-1]
    targets = (*(Target(id=TargetId(f"{w.id}/receptive"), word=w.id, skill="receptive")
                 for w in words),
               Target(id=TargetId(f"{last.id}/productive"), word=last.id, skill="productive"))
    long = _sentence(words, (tuple(w.id for w in words),), gloss="a long sentence")
    syllabus = Syllabus(words=words, targets=targets, sentences=(long,),
                        frequency={w.id: n for n, w in enumerate(words, start=1)},
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    assert [t.id for t in syllabus.productive_fills(long)] == [f"{last.id}/productive"]
    for w in words:
        fx.seed_recording(w.id, w.meaning)
    fx.seed_recording(sentence_note_id(long), long.text)
    fx.seed_picture(sentence_note_id(long), long.gloss)

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, _f, note = _note_of(pkg, long)
    assert _templates_generated(pkg, s_model, note) == {"Listening"}
    assert [d for d in compiled.report.dropped if d.family == "sentence"] == [
        DroppedCard(family="sentence", kind="Cloze",
                    subject=sentence_cloze_key(sentence_note_id(long), f"{last.id}/productive"),
                    reason=f"beyond the last Cloze slot ({CLOZE_SLOTS})")]


def test_a_second_productive_target_on_a_slotted_word_is_dropped_and_counted(fx):
    # rice carries two productive Targets (a targets.yaml exception each):
    # rice's one slot holds the first in target-id order; the second is
    # dropped with the slot's holder named, never written over it.
    eat = _word("eat", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    targets = (Target(id=TargetId("eat/receptive"), word=eat.id, skill="receptive"),
               Target(id=TargetId("rice/productive"), word=rice.id, skill="productive"),
               Target(id=TargetId("rice/productive-meal"), word=rice.id, skill="productive"))
    kin_khaao = _sentence((eat, rice), ((eat.id, rice.id),), gloss="eat rice")  # eat rice
    syllabus = Syllabus(words=(eat, rice), targets=targets, sentences=(kin_khaao,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    assert len(syllabus.productive_fills(kin_khaao)) == 2
    for w in (eat, rice):
        fx.seed_recording(w.id, w.meaning)
    fx.seed_recording(sentence_note_id(kin_khaao), kin_khaao.text)
    fx.seed_picture(sentence_note_id(kin_khaao), kin_khaao.gloss)

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, s_fields, note = _note_of(pkg, kin_khaao)
    assert dict(zip(s_fields, note["flds"]))["ClozeTarget2"] == "rice/productive"
    assert [d for d in compiled.report.dropped if d.family == "sentence"] == [
        DroppedCard(family="sentence", kind="Cloze",
                    subject=sentence_cloze_key(sentence_note_id(kin_khaao), "rice/productive-meal"),
                    reason="Cloze slot 2 held by rice/productive")]


def test_a_sentences_cards_are_siblings_due_in_its_block_listening_first(fx):
    syllabus, kin_khaao = _two_productive_words(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    _m, _f, note = _note_of(pkg, kin_khaao)
    dues = {ord_: c["due"] for ord_, c in _cards_by_ord(pkg, note).items()}
    assert dues[0] % STRIDE == 0   # the sentence's own order() block
    assert dues == {0: dues[0], 1: dues[0] + 1, 2: dues[0] + 2}

    word_dues = [c["due"] for c in pkg["cards"] if c["nid"] != note["id"]]
    assert not set(word_dues) & set(dues.values())


def test_a_sentence_filling_no_productive_target_gets_no_cloze_card(fx):
    # gin's one Target is receptive: the sentence note carries its
    # Listening card, no Cloze card, and no Cloze card is counted as
    # dropped (an unfilled slot is no card to drop).
    gin = _word("gin", "กิน", "to eat")
    target = Target(id=TargetId("gin/receptive"), word=gin.id, skill="receptive")
    sentence = _sentence((gin,), ((gin.id,),), gloss="to eat")  # to eat
    syllabus = Syllabus(words=(gin,), targets=(target,), sentences=(sentence,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_recording("gin", "eat")
    fx.seed_recording(sentence_note_id(sentence), "กิน")
    fx.seed_picture(sentence_note_id(sentence), "eating")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, field_names, s_notes = _sentence_notes(pkg, "sentence")
    assert len(s_notes) == 1
    fields = dict(zip(field_names, s_notes[0]["flds"]))
    assert fields["TargetWord"] == "กิน"  # gin: the word of every Target it fills
    assert _templates_generated(pkg, s_model, s_notes[0]) == {"Listening"}
    assert not [d for d in compiled.report.dropped if d.family == "sentence"]


def test_an_other_voice_sentence_gets_a_listening_card_and_no_cloze_card(fx):
    # Spec 1 r28: "กินค่ะ" (eat, female polite) is other_voice -- ค่ะ marks
    # a female speaker the male learner's profile does not admit -- so
    # though gin carries a productive Target the sentence fills none, and
    # its note carries the Listening card alone.
    gin = _word("gin", "กิน", "to eat")
    kha = dataclasses.replace(_word("kha", "ค่ะ", "polite particle (female)"), speaker="female")
    targets = (Target(id=TargetId("gin/receptive"), word=gin.id, skill="receptive"),
               Target(id=TargetId("gin/productive"), word=gin.id, skill="productive"),
               Target(id=TargetId("kha/receptive"), word=kha.id, skill="receptive",
                      introduction="sentence"))
    sentence = _sentence((gin, kha), ((gin.id, kha.id),), gloss="eat", voice="other_voice")
    syllabus = Syllabus(words=(gin, kha), targets=targets, sentences=(sentence,),
                        frequency={gin.id: 1, kha.id: 2},
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_recording("gin", "eat")
    fx.seed_recording(sentence_note_id(sentence), "กินค่ะ")
    fx.seed_picture(sentence_note_id(sentence), "eating")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, _field_names, s_notes = _sentence_notes(pkg, "sentence")
    (note,) = s_notes
    assert _target_tags(note) == {"gin/receptive", "kha/receptive"}
    assert _templates_generated(pkg, s_model, note) == {"Listening"}
    assert not [d for d in compiled.report.dropped if d.family == "sentence"]


def test_a_productive_target_off_the_last_used_word_gets_its_cloze_card(fx):
    # eat's Target is BOTH receptive and productive; rice's is receptive
    # only, and rice is the sentence's last used word (frequency puts eat
    # first). Under spec 1 r25 eat/productive is filled; its Cloze card
    # blanks eat, in eat's slot.
    eat = _word("eat", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    eat_receptive = Target(id=TargetId("eat/receptive"), word=eat.id, skill="receptive")
    eat_productive = Target(id=TargetId("eat/productive"), word=eat.id, skill="productive")
    rice_receptive = Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive")
    kin_khaao = _sentence((eat, rice), ((eat.id, rice.id),), gloss="eat rice")  # eat rice
    syllabus = Syllabus(words=(eat, rice),
                        targets=(eat_receptive, eat_productive, rice_receptive),
                        sentences=(kin_khaao,),
                        frequency={eat.id: 1, rice.id: 2},
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    assert syllabus.last_used_word(kin_khaao) == rice.id  # pins the fixture's own premise

    fx.seed_recording("eat", "to eat")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording(sentence_note_id(kin_khaao), "กินข้าว")
    fx.seed_picture(sentence_note_id(kin_khaao), "a man eating rice")

    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, s_fields, s_notes = _sentence_notes(pkg, "sentence")
    assert len(s_notes) == 1
    assert _target_tags(s_notes[0]) == {"eat/productive", "eat/receptive", "rice/receptive"}
    assert _templates_generated(pkg, s_model, s_notes[0]) == {"Listening", "Cloze 1"}
    fields = dict(zip(s_fields, s_notes[0]["flds"]))
    assert fields["TargetWord"] == "กิน"
    assert fields["Cloze1"] == "___ข้าว"


def _rice_sentence(fx, *, picture: bool, recording: bool = True):
    """"กินข้าว" (eat rice) filling rice/productive, with its recording
    when `recording` and its scene picture when `picture`.
    """
    eat = _word("eat", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    targets = (Target(id=TargetId("eat/receptive"), word=eat.id, skill="receptive"),
               Target(id=TargetId("rice/productive"), word=rice.id, skill="productive"))
    kin_khaao = _sentence((eat, rice), ((eat.id, rice.id),), gloss="eat rice")  # eat rice
    syllabus = Syllabus(words=(eat, rice), targets=targets, sentences=(kin_khaao,),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_recording("eat", "to eat")
    fx.seed_recording("rice", "cooked rice")
    if recording:
        fx.seed_recording(sentence_note_id(kin_khaao), "กินข้าว")
    if picture:
        fx.seed_picture(sentence_note_id(kin_khaao), "a man eating rice")
    return syllabus, kin_khaao


def test_a_cloze_card_without_its_sentences_scene_picture_is_dropped_and_counted(fx):
    # Spec 4 r10: the Cloze front is the blanked sentence plus the scene
    # picture; with no current-best picture the Cloze card is not built
    # and is counted, while the sentence's Listening card still compiles.
    syllabus, kin_khaao = _rice_sentence(fx, picture=False)
    text_sha = sentence_note_id(kin_khaao)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, _f, note = _note_of(pkg, kin_khaao)
    assert _templates_generated(pkg, s_model, note) == {"Listening"}
    assert [d for d in compiled.report.dropped if d.family == "sentence"] == [
        DroppedCard(family="sentence", kind="Cloze",
                    subject=sentence_cloze_key(text_sha, "rice/productive"),
                    reason="no current-best picture")]


def test_a_cloze_card_without_its_sentences_recording_is_dropped_and_counted(fx):
    # Spec 4 r11: the Cloze back plays the sentence; with no current-best
    # recording the Cloze card is dropped and counted, beside the
    # Listening card's own drop.
    syllabus, kin_khaao = _rice_sentence(fx, picture=True, recording=False)
    text_sha = sentence_note_id(kin_khaao)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    assert _sentence_notes(pkg, "sentence")[2] == []
    assert sorted([d for d in compiled.report.dropped if d.family == "sentence"],
                  key=lambda d: d.kind) == [
        DroppedCard(family="sentence", kind="Cloze",
                    subject=sentence_cloze_key(text_sha, "rice/productive"),
                    reason="no current-best recording"),
        DroppedCard(family="sentence", kind="Listening", subject=text_sha,
                    reason="no current-best recording")]


def test_a_cloze_card_without_its_sentences_recording_or_picture_names_both(fx):
    syllabus, kin_khaao = _rice_sentence(fx, picture=False, recording=False)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert [d.reason for d in compiled.report.dropped if d.kind == "Cloze"] == [
        "no current-best picture and recording"]


def test_a_cloze_card_with_its_sentences_scene_picture_compiles(fx):
    syllabus, kin_khaao = _rice_sentence(fx, picture=True)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, s_fields, note = _note_of(pkg, kin_khaao)
    assert _templates_generated(pkg, s_model, note) == {"Listening", "Cloze 2"}
    assert dict(zip(s_fields, note["flds"]))["ScenePicture"].startswith("<img ")
    assert not [d for d in compiled.report.dropped if d.family == "sentence"]


def test_a_cloze_card_whose_scene_picture_is_rejected_is_dropped_and_counted(fx):
    # The picture is on record but the learner rejected it: no
    # current-best picture, so the Cloze card drops as if none existed.
    from thai_syllabus.cachekeys import LearnerKey
    from thai_syllabus.derivations import current_best, role_of

    syllabus, kin_khaao = _rice_sentence(fx, picture=True)
    text_sha = sentence_note_id(kin_khaao)
    scene = current_best(fx.db, text_sha, "picture", current_rubric={}, prior=(),
                         provenance_source=lambda s: None)
    role = role_of(fx.db, text_sha, "picture")
    fx.db.append(port="assess", backend="learner",
                 key=LearnerKey(artifact_sha=scene.artifact_sha, role=role), subject=text_sha,
                 question={"role": role, "artifact_sha": scene.artifact_sha, "kind": "rating"},
                 answer={"value": "unacceptable-none"})

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, _f, note = _note_of(pkg, kin_khaao)
    assert _templates_generated(pkg, s_model, note) == {"Listening"}
    assert [d for d in compiled.report.dropped if d.family == "sentence"] == [
        DroppedCard(family="sentence", kind="Cloze",
                    subject=sentence_cloze_key(text_sha, "rice/productive"),
                    reason="no current-best picture")]


def _two_sentences(fx, *, first_picture: bool):
    """"กินข้าว" (eat rice), filling eat/productive and rice/productive,
    beside "ไก่" (chicken), filling chicken/productive. Seeds every
    recording, the second sentence's scene picture, and the first's when
    `first_picture`.
    """
    eat = _word("eat", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    chicken = _word("chicken", "ไก่", "chicken")
    words = (eat, rice, chicken)
    targets = tuple(Target(id=TargetId(f"{w.id}/productive"), word=w.id, skill="productive")
                    for w in words)
    first = _sentence(words, ((eat.id, rice.id),), gloss="eat rice")
    second = _sentence(words, ((chicken.id,),), gloss="chicken")
    for w in words:
        fx.seed_recording(w.id, w.meaning)
    for s in (first, second):
        fx.seed_recording(sentence_note_id(s), s.text)
    fx.seed_picture(sentence_note_id(second), "a chicken")
    if first_picture:
        fx.seed_picture(sentence_note_id(first), "a man eating rice")
    return Syllabus(words=words, targets=targets, sentences=(first, second),
                    frequency={w.id: n for n, w in enumerate(words, start=1)},
                    profile=Profile(register="male_colloquial"),
                    rules=_RULES_WITHOUT_COMPLETENESS)


def _dues_by_card(fx, syllabus) -> dict[tuple[str, int], int]:
    """(note guid, card ord) -> due over one compile of `syllabus`."""
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                     current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    guid_by_nid = {n["id"]: n["guid"] for n in pkg["notes"]}
    return {(guid_by_nid[c["nid"]], c["ord"]): c["due"] for c in pkg["cards"]}


def test_dropping_a_cloze_card_moves_no_other_cards_due_and_it_returns_unchanged(fx):
    # The first sentence's two Cloze cards drop without its scene picture;
    # every remaining card keeps the due it has with the picture, and once
    # the picture exists the Cloze cards return with their GUIDs and dues.
    with_picture = _dues_by_card(fx, _two_sentences(fx, first_picture=True))

    (fx.tmp_path / "pictureless").mkdir()
    fx_without = Fixture(fx.tmp_path / "pictureless")
    syllabus = _two_sentences(fx_without, first_picture=False)
    without_picture = _dues_by_card(fx_without, syllabus)

    assert len(with_picture) - len(without_picture) == 2
    assert without_picture == {k: with_picture[k] for k in without_picture}

    fx_without.seed_picture(sentence_note_id(syllabus.sentences[0]), "a man eating rice")
    assert _dues_by_card(fx_without, syllabus) == with_picture


def test_pictureless_sentences_differing_only_in_the_blanked_word_share_no_front(fx):
    # "กินข้าว" (eat rice) and "กินไก่" (eat chicken) blank to the same
    # "กิน___"; with neither scene picture neither Cloze card is built, so
    # card/unique-front finds nothing to refuse.
    eat = _word("eat", "กิน", "to eat")
    rice = _word("rice", "ข้าว", "cooked rice")
    chicken = _word("chicken", "ไก่", "chicken")
    words = (eat, rice, chicken)
    targets = (Target(id=TargetId("eat/receptive"), word=eat.id, skill="receptive"),
               Target(id=TargetId("rice/productive"), word=rice.id, skill="productive"),
               Target(id=TargetId("chicken/productive"), word=chicken.id, skill="productive"))
    sentences = (_sentence(words, ((eat.id, rice.id),), gloss="eat rice"),
                 _sentence(words, ((eat.id, chicken.id),), gloss="eat chicken"))
    syllabus = Syllabus(words=words, targets=targets, sentences=sentences,
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_FOR_UNIQUE_FRONT)
    for w in words:
        fx.seed_recording(w.id, w.meaning)
    for s in sentences:
        fx.seed_recording(sentence_note_id(s), s.text)

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path, force=True,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert [f.evidence for f in compiled.report.findings if f.rule == "card/unique-front"] == []
    assert compiled.report.gate is True
    assert sorted(d.subject for d in compiled.report.dropped if d.kind == "Cloze") == sorted(
        sentence_cloze_key(sentence_note_id(s), f"{w}/productive")
        for s, w in zip(sentences, ("rice", "chicken")))


def test_cloze_cards_of_two_sentences_in_different_slots_are_compared_for_one_front(fx):
    # "กินข้าว" (eat rice) blanks rice in its slot 2; "กิ" + "น" + "ไก่"
    # ("กินไก่", eat chicken, spelled in three words) blanks chicken in
    # its slot 3. Given one scene picture, both Cloze fronts are "กิน___"
    # plus that picture: card/unique-front compares every Cloze card
    # whatever its slot, naming each card by its (sentence, Target).
    eat = _word("eat", "กิน", "to eat")
    ki = _word("ki", "กิ", "ki (a syllable)")
    n = _word("n", "น", "n (a letter)")
    rice = _word("rice", "ข้าว", "cooked rice")
    chicken = _word("chicken", "ไก่", "chicken")
    words = (eat, ki, n, rice, chicken)
    targets = (*(Target(id=TargetId(f"{w.id}/receptive"), word=w.id, skill="receptive")
                 for w in (eat, ki, n)),
               *(Target(id=TargetId(f"{w.id}/productive"), word=w.id, skill="productive")
                 for w in (rice, chicken)))
    two = _sentence(words, ((eat.id, rice.id),), gloss="eat rice")
    three = _sentence(words, ((ki.id, n.id, chicken.id),), gloss="eat chicken")
    syllabus = Syllabus(words=words, targets=targets, sentences=(two, three),
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_FOR_UNIQUE_FRONT)
    for w in words:
        fx.seed_recording(w.id, w.meaning)
    for s in (two, three):
        fx.seed_recording(sentence_note_id(s), s.gloss)
        fx.seed_picture(sentence_note_id(s), "a man eating", content=b"one scene")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path, force=True,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    s_model, _f, two_note = _note_of(pkg, two)
    _m, _f, three_note = _note_of(pkg, three)
    assert _templates_generated(pkg, s_model, two_note) == {"Listening", "Cloze 2"}
    assert _templates_generated(pkg, s_model, three_note) == {"Listening", "Cloze 3"}
    assert {f.note_id for f in compiled.report.findings if f.rule == "card/unique-front"} == {
        sentence_cloze_key(sentence_note_id(two), "rice/productive"),
        sentence_cloze_key(sentence_note_id(three), "chicken/productive")}


# --- due / bury-siblings ---------------------------------------------------

def test_sibling_cards_get_distinct_due_values(fx):
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    word_model = next(m for m in models.values() if m["name"] == "word")
    field_names = [f["name"] for f in word_model["flds"]]
    rice_note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                     and dict(zip(field_names, n["flds"]))["Thai"] == "ข้าว")
    rice_cards = [c for c in pkg["cards"] if c["nid"] == rice_note["id"]]
    assert len(rice_cards) > 1
    dues = [c["due"] for c in rice_cards]
    assert len(set(dues)) == len(dues)  # every sibling gets its own due


def test_graphemes_are_due_before_any_word(fx):
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    g_model = next(m for m in models.values() if m["name"] == "grapheme")
    word_model = next(m for m in models.values() if m["name"] == "word")
    g_note = next(n for n in pkg["notes"] if str(n["mid"]) == g_model["id"])
    g_due = min(c["due"] for c in pkg["cards"] if c["nid"] == g_note["id"])
    word_dues = [c["due"] for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                for c in pkg["cards"] if c["nid"] == n["id"]]
    assert all(g_due < d for d in word_dues)


def test_sentence_cards_are_due_after_every_word_target_they_use(fx):
    # r24: a sentence note's due comes straight from its own order() entry
    # position -- directly after its last used word's last Target, and
    # before the next word's first Target -- not after every word target
    # in the deck. "ผมกินข้าว" (I eat rice) uses pom, gin and rice; rice is
    # its last used word (all three tie on frequency, so word id order
    # gin < pom < rice decides). An extra word "zzz", sorting after rice,
    # gives this fixture a "next word" to bound the sentence against.
    syllabus = _small_syllabus(extra_targets=(
        Target(id=TargetId("zzz/receptive"), word=WordId("zzz"), skill="receptive"),))
    zzz = _word("zzz", "อื่น", "other word, sorts after rice")
    syllabus = dataclasses.replace(syllabus, words=(*syllabus.words, zzz))

    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("letter-name:ko", "gɔɔ")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")
    fx.seed_recording("zzz", "other")
    text_sha = sentence_note_id(syllabus.sentences[0])
    fx.seed_recording(text_sha, "ผมกินข้าว")

    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    s_model = next(m for m in models.values() if m["name"] == "sentence")
    word_model = next(m for m in models.values() if m["name"] == "word")
    word_field_names = [f["name"] for f in word_model["flds"]]

    def word_due(thai: str) -> int:
        note = next(n for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                   and dict(zip(word_field_names, n["flds"]))["Thai"] == thai)
        return min(c["due"] for c in pkg["cards"] if c["nid"] == note["id"])

    used_dues = [word_due(thai) for thai in ("ผม", "กิน", "ข้าว")]  # pom, gin, rice
    next_word_due = word_due("อื่น")  # zzz: sorts right after rice

    s_notes = [n for n in pkg["notes"] if str(n["mid"]) == s_model["id"]]
    sentence_dues = [c["due"] for n in s_notes for c in pkg["cards"] if c["nid"] == n["id"]]
    assert sentence_dues
    assert all(used < s for used in used_dues for s in sentence_dues)
    assert all(s < next_word_due for s in sentence_dues)


def test_two_targets_filled_by_one_sentence_share_its_due_position(fx):
    # "กินข้าว" (eat rice) fills BOTH eat/receptive and rice/receptive;
    # "หมาวิ่ง" (dog runs) fills both dog/receptive and run/receptive.
    # One sentence note per adopted Sentence (spec 4 r5, r11): each
    # compiles to ONE note carrying BOTH its filled targets' target:: tags
    # and one Listening card at the sentence's due position (neither fills
    # a productive Target, so no Cloze cards), and the two sentence
    # notes land in order() position order: "หมาวิ่ง" (r24: dealt directly
    # after run, the later of dog/run) due before "กินข้าว" (dealt directly
    # after rice, the later of eat/rice) -- two STRIDEs apart, not one,
    # because rice/receptive's own word_target block sits between them
    # (frequency order is eat, dog, run, rice).
    eat = _word("eat", "กิน", "to eat")
    dog = _word("dog", "หมา", "dog")
    run = _word("run", "วิ่ง", "to run")
    rice = _word("rice", "ข้าว", "cooked rice")
    targets = (
        Target(id=TargetId("eat/receptive"), word=eat.id, skill="receptive"),
        Target(id=TargetId("dog/receptive"), word=dog.id, skill="receptive"),
        Target(id=TargetId("run/receptive"), word=run.id, skill="receptive"),
        Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive"),
    )
    eat_rice = _sentence((eat, rice), ((eat.id, rice.id),), gloss="eat rice")  # eat rice
    dog_runs = _sentence((dog, run), ((dog.id, run.id),), gloss="dog runs")  # dog runs
    syllabus = Syllabus(words=(eat, dog, run, rice), targets=targets,
                        sentences=(eat_rice, dog_runs),
                        frequency={eat.id: 1, dog.id: 2, run.id: 3, rice.id: 4},
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    for w in ("eat", "dog", "run", "rice"):
        fx.seed_recording(w, w)
    fx.seed_recording(sentence_note_id(eat_rice), "กินข้าว")
    fx.seed_recording(sentence_note_id(dog_runs), "หมาวิ่ง")

    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    s_model = next(m for m in models.values() if m["name"] == "sentence")
    field_names = [f["name"] for f in s_model["flds"]]
    s_notes = [n for n in pkg["notes"] if str(n["mid"]) == s_model["id"]]
    assert len(s_notes) == 2

    def note_for(sentence_sha: str):
        return next(n for n in s_notes
                   if f"sentence::{sentence_sha}" in n["tags"].split(" "))

    def due_for(sentence_sha: str) -> int:
        note = note_for(sentence_sha)
        return min(c["due"] for c in pkg["cards"] if c["nid"] == note["id"])

    eat_rice_note = note_for(sentence_note_id(eat_rice))
    eat_rice_tags = {t.split("::", 1)[1] for t in eat_rice_note["tags"].split(" ")
                     if t.startswith("target::")}
    assert eat_rice_tags == {"eat/receptive", "rice/receptive"}
    eat_rice_fields = dict(zip(field_names, eat_rice_note["flds"]))
    assert eat_rice_fields["TargetWord"] == "กิน, ข้าว"  # receptive only: every filled Target's word

    dog_runs_note = note_for(sentence_note_id(dog_runs))
    dog_runs_tags = {t.split("::", 1)[1] for t in dog_runs_note["tags"].split(" ")
                     if t.startswith("target::")}
    assert dog_runs_tags == {"dog/receptive", "run/receptive"}
    dog_runs_fields = dict(zip(field_names, dog_runs_note["flds"]))
    assert dog_runs_fields["TargetWord"] == "หมา, วิ่ง"

    eat_due = due_for(sentence_note_id(eat_rice))
    dog_due = due_for(sentence_note_id(dog_runs))
    assert dog_due < eat_due
    assert eat_due - dog_due == 2 * STRIDE


def test_the_packages_own_options_group_buries_new_and_review_siblings(fx):
    # genanki's default dconf in the package's own collection sets
    # new.bury and rev.bury. An Anki import does not apply a package's
    # deck options, so this says nothing about the learner's preset
    # (spec 4 section 2); anki_import warns about that one.
    import json
    import sqlite3
    import tempfile
    import zipfile
    from pathlib import Path

    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(fx.out_path) as zf:
            zf.extractall(tmp)
        conn = sqlite3.connect(str(Path(tmp) / "collection.anki2"))
        (dconf_json,) = conn.execute("select dconf from col").fetchone()
    dconf = json.loads(dconf_json)
    default = dconf["1"]
    assert default["new"]["bury"] is True
    assert default["rev"]["bury"] is True


# --- atomic write ----------------------------------------------------------

def test_compile_leaves_no_leftover_tmp_file(fx):
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path,
                                current_rubric={}, prior=(), provenance_source=lambda sha: None)
    assert list(fx.out_path.parent.glob("*.tmp")) == []


# --- card meanings (spec 5 r9) ----------------------------------------------

from thai_syllabus.compile import (CARD_MEANINGS, GRAPHEME_MODEL, MINIMAL_PAIR_MODEL,
                                   card_kind_of, card_meaning)


def test_every_compiled_card_type_has_a_one_line_meaning():
    """Spec 5 r9 (design ruling 4): every card the compile can emit has a
    one-line meaning, keyed by the family and kind /api/cards reports."""
    expected = set()
    for family, model in (("word", WORD_MODEL), ("minimal_pair", MINIMAL_PAIR_MODEL),
                          ("grapheme", GRAPHEME_MODEL), ("sentence", SENTENCE_MODEL)):
        for template in model.templates:
            expected.add((family, card_kind_of(template["name"])))
    assert set(CARD_MEANINGS) == expected
    for meaning in CARD_MEANINGS.values():
        assert meaning and "\n" not in meaning and meaning.endswith(".")
    assert card_meaning("word", "listening") == CARD_MEANINGS[("word", "listening")]
    assert card_meaning("word", "no-such-kind") is None


def test_sentence_listening_back_labels_the_target_words():
    """Design ruling 4: the Listening back read as sentence plus a stray
    word; the target line now says what it is -- the sentence's target
    words (spec 3 r54), so plural."""
    listening = next(t for t in SENTENCE_MODEL.templates if t["name"] == "Listening")
    assert '<div class="target"><span class="label">target words</span> {{TargetWord}}</div>' in listening["afmt"]
    for cloze in SENTENCE_MODEL.templates[1:]:
        assert '<span class="label">' not in cloze["afmt"]


# --- positions over the interleaved name Targets (spec 1 r16) ------------

def test_a_name_words_targets_take_one_block_each_between_grapheme_and_word():
    """R6: compile's due blocks follow order() exactly -- the grapheme, then
    its name word's two Targets one block apiece, then the ordinary word
    targets. Nothing overlaps and nothing is skipped.

    Characterization: `_positions` gives every non-pair order entry a
    width-1 block (`_order_entry_width`) and a name word's Targets arrive
    as ordinary `word_target` entries, so this holds by construction --
    the test pins it against a later change.
    """
    from thai_syllabus.compile import _positions

    chicken = _word("chicken", "ไก่", "chicken")
    name = _word("name-chicken", "กอ ไก่", "the letter ก (recited name)")
    rice = _word("rice", "ข้าว", "cooked rice")
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    syllabus = Syllabus(
        words=(chicken, name, rice), graphemes=(g,),
        targets=(Target(id=TargetId("rice/receptive"), word=rice.id, skill="receptive"),
                 Target(id=TargetId("name-chicken/receptive"), word=name.id,
                        skill="receptive"),
                 Target(id=TargetId("name-chicken/productive"), word=name.id,
                        skill="productive")))

    positions = _positions(syllabus)

    assert positions.entry_index["ก"] == 0
    assert positions.target_index["name-chicken/receptive"] == 1
    assert positions.target_index["name-chicken/productive"] == 2
    assert positions.target_index["rice/receptive"] == 3
    assert positions.word_index["name-chicken"] == 1
    assert positions.order_length == 4
