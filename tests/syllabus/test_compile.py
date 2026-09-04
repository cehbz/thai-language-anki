"""Syllabus.compile() / compile.compile_syllabus (spec 4): models, fields,
guids, tags, due, gate refusal, and dropped-card counting, against a small
synthetic Syllabus compiled through a real SyllabusDb + MediaStore (in a
tmp_path) end to end -- reading the produced .apkg back with the same
"read a real collection.anki2" pattern scripts/proof_gallery.py and
tests/gen/helpers_apkg.py already use.

The `ยา`/`โรงพยาบาล` (medicine/hospital) substring-corruption case is
table-tested directly against `thai_cloze`, without a full compile.
"""
from datetime import date

import pytest

from thai_syllabus.assessor import ROLE_FOR_KIND
from thai_syllabus.compile import GateRefusal, compile_syllabus, thai_cloze
from thai_syllabus.entities import (
    Grapheme, MinimalPair, Pronunciation, Sentence, SoundConfusion, Syllable,
    Target, Word,
)
from thai_syllabus.ids import ConfusionId, PairId, TargetId, WordId
from thai_syllabus.media import Provenance, Speaker
from thai_syllabus.profile import Profile
from thai_syllabus.rulebook import RULES
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus

from tests.gen.helpers_apkg import read_apkg

PROV = Provenance(source="test", origin="fixture", licence="cc0",
                  acquired=date(2026, 1, 1))

# This module's fixtures build a Syllabus whose `media` port is the default
# NullMediaIndex -- artifact resolution for the actual compiled cards goes
# straight through fx.db/fx.media (Fixture.seed_picture/seed_recording), not
# through Syllabus.media, so these five completeness ERROR rules (spec 4)
# would always fire here regardless of what's seeded, closing the gate on
# every fixture that has targets. This module's subject is compile.py's own
# field/guid/due/dropped-card mechanics, not target completeness, so those
# five are dropped from the default rules; everything else (closure,
# exact-confusion, ...) still runs.
_COMPLETENESS_ERROR_IDS = {
    "target/picture-required", "target/recording-required", "target/sentence-required",
    "pair/rendition-required", "grapheme/keyword-picture-required",
}
_RULES_WITHOUT_COMPLETENESS = tuple(r for r in RULES if r.id not in _COMPLETENESS_ERROR_IDS)


def _syl(onset="m", vowel="a", coda="", length="short", tone="mid") -> Syllable:
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


def _pron(*syllables, corroboration="engines_agree") -> Pronunciation:
    return Pronunciation(syllables=tuple(syllables) or (_syl(),), corroboration=corroboration)


def _word(id_, thai, meaning, tone="mid") -> Word:
    return Word(id=WordId(id_), thai=thai, pron=_pron(_syl(tone=tone)), meaning=meaning)


class _SplitTokenizer:
    """A real (if crude) whole-string tokenizer good enough for the
    fixture's short sentences: splits on nothing (Thai has no spaces) --
    tests instead supply pre-tokenized text via a lookup, matching
    tests/syllabus/fakes.py's FakeTokenizer pattern but reusable here
    without importing a test-only fake into a fixture module other tests
    also import from.
    """
    def __init__(self, tokens_by_text: dict[str, list[str]]):
        self._map = tokens_by_text

    def tokens(self, text: str) -> list[str]:
        return self._map.get(text, [text])


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
        self.db.append(port="assess", backend="judge",
                       key=f"judge:seed:{sha}:{kind}", subject=subject,
                       question={"role": ROLE_FOR_KIND.get(kind, kind), "artifact_sha": sha,
                                "rubric": "seed"},
                       answer={"value": True})

    def seed_recording(self, subject: str, text: str, speaker="somchai") -> str:
        sha = self.media.write(f"audio:{subject}:{text}".encode(), ext="mp3")
        self.db.add_speaker(Speaker(id=speaker, kind="native"))
        self.db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                          origin="https://forvo.com/x", licence="cc-by",
                          acquired=date(2026, 1, 1), speaker_id=speaker)
        self.db.append(port="provide", backend="forvo", key=f"forvo:{subject}",
                       subject=subject, question={"provides": "recording"},
                       answer={"items": [{"sha": sha}]})
        self._pass_judge(subject, "recording", sha)
        return sha

    def seed_picture(self, subject: str, text: str) -> str:
        sha = self.media.write(f"image:{subject}:{text}".encode(), ext="jpg")
        self.db.add_media(sha=sha, kind="picture", ext="jpg", source="openverse",
                          origin="https://example.com/x.jpg", licence="cc0",
                          acquired=date(2026, 1, 1))
        self.db.append(port="provide", backend="openverse", key=f"openverse:{subject}",
                       subject=subject, question={"provides": "picture"},
                       answer={"items": [{"sha": sha}]})
        self._pass_judge(subject, "picture", sha)
        return sha


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


def _small_syllabus(tokenizer, extra_targets=()) -> Syllabus:
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

    sentence = Sentence(text="ผมกินข้าว", voice="learner_voice", provenance=PROV)

    return Syllabus(
        words=(pom, gin, rice, chicken, ko_name, near, far),
        targets=tuple(targets),
        pairs=(pair,),
        graphemes=(grapheme,),
        sentences=(sentence,),
        confusions=(confusion,),
        profile=Profile(register="male_colloquial"),
        tokenizer=tokenizer,
        rules=_RULES_WITHOUT_COMPLETENESS,
    )


def _fully_seeded(fx) -> Syllabus:
    tokenizer = _SplitTokenizer({"ผมกินข้าว": ["ผม", "กิน", "ข้าว"]})
    syllabus = _small_syllabus(tokenizer)

    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("letter-name:ko", "gɔɔ")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")
    text_sha = __import__("thai_syllabus.rulebook", fromlist=["sentence_note_id"]).sentence_note_id(
        syllabus.sentences[0])
    fx.seed_recording(text_sha, "ผมกินข้าว")
    return syllabus


# --- thai_cloze: the corruption table ------------------------------------

@pytest.mark.parametrize("tokens,target,expected", [
    (["ผม", "กิน", "ยา"], "ยา", "ผมกิน___"),  # I take medicine
    # the corruption class: "ยา" must NOT be blanked inside "โรงพยาบาล"
    # (hospital) -- a real tokenizer never splits it out as its own token,
    # and boundary matching alone (never str.replace) is what protects it.
    (["ผม", "ไป", "โรงพยาบาล"], "ยา", "ผมไปโรงพยาบาล"),
    (["หมา", "วิ่ง"], "หมา", "___วิ่ง"),               # dog runs -- exact token
    (["ยา", "ยา"], "ยา", "______"),                    # every matching token blanked
    (["น้ำยา"], "ยา", "___"),                          # compound: suffix match blanks whole token
])
def test_thai_cloze_blanks_only_boundary_matching_tokens(tokens, target, expected):
    assert thai_cloze(tokens, target) == expected


def test_thai_cloze_never_touches_a_non_matching_token():
    tokens = ["โรงพยาบาล", "ใหญ่"]  # hospital (big)
    assert thai_cloze(tokens, "ยา") == "".join(tokens)


# --- compile: gate refusal ------------------------------------------------

def test_compile_refuses_when_the_gate_is_closed(fx):
    from thai_syllabus.rules import Rule, Finding

    def always_fails(s):
        return [Finding(rule="test/always-fails", note_id="x", evidence="bad")]

    rule = Rule(id="test/always-fails", principle="F1", severity="error",
               shape="check", check=always_fails)
    syllabus = _small_syllabus(_SplitTokenizer({}))
    syllabus = syllabus_with_rules(syllabus, (rule,))
    with pytest.raises(GateRefusal):
        compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    assert not fx.out_path.exists()


def test_compile_forced_past_a_closed_gate_records_declared_warnings(fx):
    from thai_syllabus.rules import Rule, Finding

    def always_fails(s):
        return [Finding(rule="test/always-fails", note_id="x", evidence="bad thing")]

    rule = Rule(id="test/always-fails", principle="F1", severity="error",
               shape="check", check=always_fails)
    syllabus = _fully_seeded(fx)
    syllabus = syllabus_with_rules(syllabus, (rule,))
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path, force=True)
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
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
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
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
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
    assert "target::rice" in tags
    assert f"compile::{compiled.compile_id}" in tags


def test_word_note_production_card_is_dropped_without_a_productive_target(fx):
    # pom/gin only have RECEPTIVE targets -- Production must not appear,
    # even though both have a recording (word cards don't need a picture
    # to generate Listening/Reading).
    syllabus = _fully_seeded(fx)
    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
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


def test_word_note_listening_dropped_and_counted_when_audio_is_missing(fx):
    tokenizer = _SplitTokenizer({"ผมกินข้าว": ["ผม", "กิน", "ข้าว"]})
    syllabus = _small_syllabus(tokenizer)
    # Deliberately do NOT seed pom's recording.
    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("letter-name:ko", "gɔɔ")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")

    compiled = compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
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


def test_grapheme_note_name_thai_and_audio_fallback(fx):
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    g_model = next(m for m in models.values() if m["name"] == "grapheme")
    field_names = [f["name"] for f in g_model["flds"]]
    note = next(n for n in pkg["notes"] if str(n["mid"]) == g_model["id"])
    fields = dict(zip(field_names, note["flds"]))
    assert fields["NameThai"] == "กอ ไก่"  # "gɔɔ gài" -- letter name + keyword
    assert fields["KeywordThai"] == "ไก่"  # chicken
    assert fields["Audio"].startswith("[sound:")

    import genanki
    assert note["guid"] == genanki.guid_for("grapheme", "ก")


def test_grapheme_audio_falls_back_to_keyword_when_name_word_recording_absent(fx):
    tokenizer = _SplitTokenizer({"ผมกินข้าว": ["ผม", "กิน", "ข้าว"]})
    syllabus = _small_syllabus(tokenizer)
    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_picture("chicken", "chicken")
    fx.seed_recording("near", "near")
    fx.seed_recording("far", "far")
    # NOT seeding letter-name:ko's recording -- Audio must fall back to
    # the keyword's (chicken's) recording instead.
    fx.seed_recording("chicken", "chicken")

    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    g_model = next(m for m in models.values() if m["name"] == "grapheme")
    field_names = [f["name"] for f in g_model["flds"]]
    note = next(n for n in pkg["notes"] if str(n["mid"]) == g_model["id"])
    fields = dict(zip(field_names, note["flds"]))
    assert fields["Audio"].startswith("[sound:")


def test_grapheme_name_thai_degrades_gracefully_without_a_name_word(fx):
    tokenizer = _SplitTokenizer({"ผมกินข้าว": ["ผม", "กิน", "ข้าว"]})
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
                        graphemes=(grapheme,), tokenizer=tokenizer,
                        profile=Profile(register="male_colloquial"),
                        rules=_RULES_WITHOUT_COMPLETENESS)
    fx.seed_picture("rice", "cooked rice")
    fx.seed_recording("rice", "cooked rice")
    fx.seed_recording("pom", "I")
    fx.seed_recording("gin", "eat")
    fx.seed_recording("chicken", "chicken")

    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    g_model = next(m for m in models.values() if m["name"] == "grapheme")
    field_names = [f["name"] for f in g_model["flds"]]
    note = next(n for n in pkg["notes"] if str(n["mid"]) == g_model["id"])
    fields = dict(zip(field_names, note["flds"]))
    assert fields["NameThai"] == "ก ไก่"  # symbol + keyword, no name_word


def test_minimal_pair_notes_one_per_member_with_playable_audio_both_sides(fx):
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    pair_model = next(m for m in models.values() if m["name"] == "minimal_pair")
    field_names = [f["name"] for f in pair_model["flds"]]
    pair_notes = [n for n in pkg["notes"] if str(n["mid"]) == pair_model["id"]]
    assert len(pair_notes) == 2  # one per member

    by_thai = {dict(zip(field_names, n["flds"]))["Thai"]: n for n in pair_notes}
    near_fields = dict(zip(field_names, by_thai["ใกล้"]["flds"]))
    assert near_fields["MemberKey"].startswith("tone:mid-low/klai:")
    assert near_fields["Audio"].startswith("[sound:")
    assert near_fields["OtherAudio"].startswith("[sound:")
    assert near_fields["OtherThai"] == "ไกล"

    for note in pair_notes:
        tags = note["tags"].split(" ")
        assert "family::minimal_pair" in tags
        assert "confusion::tone:mid-low" in tags
        assert "pair::tone:mid-low/klai" in tags


def test_sentence_note_cloze_and_listening(fx):
    # The fixture sentence "ผมกินข้าว" (I eat rice) mentions ALL of pom,
    # gin and rice at token boundaries, and every one of those words
    # already has a Target -- so Syllabus.fills() (spec 1 section 3,
    # clause 3: "any used word with a target anywhere is met by entry")
    # is True for every target whose word it mentions, not just rice's:
    # one sentence note per (Target, Sentence) it fills (spec 4 section
    # 1), so 4 notes here (pom/receptive, gin/receptive, rice/receptive,
    # rice/productive), each with its OWN target correctly blanked.
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    s_model = next(m for m in models.values() if m["name"] == "sentence")
    field_names = [f["name"] for f in s_model["flds"]]
    s_notes = [n for n in pkg["notes"] if str(n["mid"]) == s_model["id"]]
    assert len(s_notes) == 4

    by_target = {}
    for n in s_notes:
        tags = n["tags"].split(" ")
        target_tag = next(t for t in tags if t.startswith("target::"))
        by_target[target_tag.split("::", 1)[1]] = n

    assert set(by_target) == {"pom/receptive", "gin/receptive",
                              "rice/receptive", "rice/productive"}

    rice_note = by_target["rice/productive"]
    fields = dict(zip(field_names, rice_note["flds"]))
    assert fields["ThaiCloze"] == "ผมกิน___"
    assert fields["Thai"] == "ผมกินข้าว"
    assert fields["TargetWord"] == "ข้าว"
    assert fields["Audio"].startswith("[sound:")

    tmpl_names = [t["name"] for t in s_model["tmpls"]]
    cards = [c for c in pkg["cards"] if c["nid"] == rice_note["id"]]
    generated = {tmpl_names[c["ord"]] for c in cards}
    assert generated == {"Cloze", "Listening"}

    pom_note = by_target["pom/receptive"]
    pom_fields = dict(zip(field_names, pom_note["flds"]))
    assert pom_fields["ThaiCloze"] == "___กินข้าว"


# --- due / bury-siblings ---------------------------------------------------

def test_sibling_cards_get_distinct_due_values(fx):
    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
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
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    pkg = read_apkg(fx.out_path)
    models = pkg["models"]
    g_model = next(m for m in models.values() if m["name"] == "grapheme")
    word_model = next(m for m in models.values() if m["name"] == "word")
    g_note = next(n for n in pkg["notes"] if str(n["mid"]) == g_model["id"])
    g_due = min(c["due"] for c in pkg["cards"] if c["nid"] == g_note["id"])
    word_dues = [c["due"] for n in pkg["notes"] if str(n["mid"]) == word_model["id"]
                for c in pkg["cards"] if c["nid"] == n["id"]]
    assert all(g_due < d for d in word_dues)


def test_shipped_deck_options_group_already_buries_siblings(fx):
    # Not this module's own logic -- genanki's default dconf ships
    # new.bury=true / rev.bury=true already (spec 4 section 2's "both",
    # A5); this asserts that fact rather than re-implementing it.
    import json
    import sqlite3
    import tempfile
    import zipfile
    from pathlib import Path

    syllabus = _fully_seeded(fx)
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
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
    compile_syllabus(syllabus, fx.db, fx.media, fx.out_path)
    assert list(fx.out_path.parent.glob("*.tmp")) == []


def test_syllabus_compile_method_delegates(fx):
    syllabus = _fully_seeded(fx)
    compiled = syllabus.compile(fx.db, fx.media, fx.out_path)
    assert fx.out_path.exists()
    assert compiled.syllabus_state_id == syllabus.state_id()
