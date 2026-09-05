"""What the compiled .apkg must guarantee (spec 4 sections 1-3; principles
A2-A8). Stated as properties of the artifact a learner imports, verified by
reading the package back with tests/spec/syllabus_world.py's read_apkg --
nothing here knows how compile_syllabus works internally.
"""
import dataclasses

import pytest

from thai_syllabus.compile import GateRefusal, STRIDE, compile_syllabus
from thai_syllabus.rules import DroppedCard, Finding, Rule
from thai_syllabus.wiring import _DbMediaIndex

from tests.spec.syllabus_world import (
    SplitTokenizer, SyllabusWorld, duplicate_front_syllabus, fully_seeded_syllabus,
    pair_only_syllabus, read_apkg,
)


@pytest.fixture
def world(tmp_path):
    return SyllabusWorld.create(tmp_path)


def _models_by_name(pkg: dict) -> dict[str, dict]:
    return {m["name"]: m for m in pkg["models"].values()}


def _notes_of(pkg: dict, model_name: str) -> list[dict]:
    model = _models_by_name(pkg)[model_name]
    return [n for n in pkg["notes"] if str(n["mid"]) == model["id"]]


def _field_index(pkg: dict, model_name: str) -> dict[str, int]:
    model = _models_by_name(pkg)[model_name]
    return {f["name"]: i for i, f in enumerate(model["flds"])}


def _cards_of(pkg: dict, note_id) -> list[dict]:
    return [c for c in pkg["cards"] if c["nid"] == note_id]


# --- A2: recompiling updates the same notes rather than duplicating them --

def test_recompiling_a_changed_syllabus_updates_notes_in_place(world):
    syllabus = fully_seeded_syllabus(world)
    first = compile_syllabus(syllabus, world.db, world.media, world.out_path)
    pkg1 = read_apkg(world.out_path)

    # rice's meaning changes; its identity (word id) does not -- A2: card
    # identity survives a content edit, so Anki merges on guid instead of
    # duplicating.
    changed_words = tuple(
        dataclasses.replace(w, meaning="cooked rice (revised)") if w.id == "rice" else w
        for w in syllabus.words)
    changed = dataclasses.replace(syllabus, words=changed_words)
    second = compile_syllabus(changed, world.db, world.media, world.out_path)
    pkg2 = read_apkg(world.out_path)

    assert {n["guid"] for n in pkg1["notes"]} == {n["guid"] for n in pkg2["notes"]}
    assert {n["mid"] for n in pkg1["notes"]} == {n["mid"] for n in pkg2["notes"]}
    assert first.report.notes_written == second.report.notes_written == len(pkg2["notes"])


# --- A3: no two cards share a front ----------------------------------------

def test_two_notes_sharing_a_front_refuse_the_compile_with_a_finding(world):
    syllabus = duplicate_front_syllabus(SplitTokenizer({}))
    world.seed_recording("rice-a", "recording a")
    world.seed_recording("rice-b", "recording b")
    with pytest.raises(GateRefusal) as excinfo:
        compile_syllabus(syllabus, world.db, world.media, world.out_path)
    assert not world.out_path.exists()
    assert any(f.rule == "card/unique-front" for f in excinfo.value.report.findings)


# --- A4: every media reference resolves to a file in the package ----------

def test_every_media_reference_resolves_to_a_file_in_the_package(world):
    syllabus = fully_seeded_syllabus(world)
    compile_syllabus(syllabus, world.db, world.media, world.out_path)
    pkg = read_apkg(world.out_path)

    referenced = set()
    for note in pkg["notes"]:
        for field_value in note["flds"]:
            if "[sound:" in field_value:
                referenced.add(field_value.split("[sound:")[1].split("]")[0])
            if "<img src=" in field_value:
                referenced.add(field_value.split('<img src="')[1].split('"')[0])

    assert referenced, "the fully-seeded fixture should reference at least one media file"
    assert referenced <= set(pkg["media"])


# --- A5: due order follows order() positions, with sibling and pair
# separation ---------------------------------------------------------------

def test_due_order_separates_siblings_and_pair_members_by_a_stride(world):
    syllabus = fully_seeded_syllabus(world)
    compile_syllabus(syllabus, world.db, world.media, world.out_path)
    pkg = read_apkg(world.out_path)

    def due_of(model_name: str, thai_field: str, thai_value: str) -> list[int]:
        idx = _field_index(pkg, model_name)[thai_field]
        note = next(n for n in _notes_of(pkg, model_name) if n["flds"][idx] == thai_value)
        return [c["due"] for c in _cards_of(pkg, note["id"])]

    # order/sounds-first (F8): graphemes and pairs precede every word target.
    grapheme_due = due_of("grapheme", "Symbol", "ก")
    rice_due = due_of("word", "Thai", "ข้าว")  # rice
    assert max(grapheme_due) < min(rice_due)

    # sibling cards of one note (rice: Listening/Production/Reading/Spelling)
    # each land on their own due value -- never collapsed onto one.
    assert len(set(rice_due)) == len(rice_due)

    # A5: the two member notes of a pair are never adjacent -- placed a
    # STRIDE apart, not interleaved onto the same due value.
    pair_model = _models_by_name(pkg)["minimal_pair"]
    pair_notes = [n for n in pkg["notes"] if str(n["mid"]) == pair_model["id"]]
    member_dues = sorted(min(c["due"] for c in _cards_of(pkg, n["id"])) for n in pair_notes)
    assert len(member_dues) == 2
    assert member_dues[1] - member_dues[0] == STRIDE

    # order/sentence-after-words (F8): the sentence "ผมกินข้าว" (I eat rice)
    # uses pom, gin and rice -- its cards are due after every one of them.
    pom_due = due_of("word", "Thai", "ผม")     # I
    gin_due = due_of("word", "Thai", "กิน")    # eat
    s_model = _models_by_name(pkg)["sentence"]
    sentence_dues = [c["due"] for n in pkg["notes"] if str(n["mid"]) == s_model["id"]
                    for c in _cards_of(pkg, n["id"])]
    assert sentence_dues
    assert all(used < s for used in (*pom_due, *gin_due, *rice_due) for s in sentence_dues)


# --- CompileId stamped on every note ----------------------------------------

def test_compile_id_is_stamped_on_every_note(world):
    syllabus = fully_seeded_syllabus(world)
    compiled = compile_syllabus(syllabus, world.db, world.media, world.out_path)
    pkg = read_apkg(world.out_path)
    models = pkg["models"]

    for note in pkg["notes"]:
        model = models[str(note["mid"])]
        idx = {f["name"]: i for i, f in enumerate(model["flds"])}["CompileId"]
        assert note["flds"][idx] == compiled.compile_id


# --- A7: a gated compile refuses, counting only unwaived errors -----------

def test_a_gated_compile_refuses_and_counts_only_unwaived_errors(world):
    syllabus = fully_seeded_syllabus(world)

    def always_fails(s):
        return [Finding(rule="test/always-fails", note_id="x", evidence="bad thing")]

    rule = Rule(id="test/always-fails", principle="A7", severity="error",
               shape="check", check=always_fails)
    gated = dataclasses.replace(syllabus, rules=(*syllabus.rules, rule))

    with pytest.raises(GateRefusal) as excinfo:
        compile_syllabus(gated, world.db, world.media, world.out_path)
    assert not world.out_path.exists()
    assert excinfo.value.blocking == 1


def test_forcing_past_a_closed_gate_writes_the_package_with_declared_warnings(world):
    syllabus = fully_seeded_syllabus(world)

    def always_fails(s):
        return [Finding(rule="test/always-fails", note_id="x", evidence="bad thing")]

    rule = Rule(id="test/always-fails", principle="A7", severity="error",
               shape="check", check=always_fails)
    gated = dataclasses.replace(syllabus, rules=(*syllabus.rules, rule))

    compiled = compile_syllabus(gated, world.db, world.media, world.out_path, force=True)
    assert compiled.report.forced is True
    assert compiled.report.gate is False
    assert any("bad thing" in w for w in compiled.report.warnings)

    # The gate closing must not stop the package itself from being a real,
    # readable compile -- forcing writes the notes, not an empty shell.
    pkg = read_apkg(world.out_path)
    assert len(pkg["notes"]) == compiled.report.notes_written > 0


# --- a receptive-only sentence note yields only the Listening card --------

def test_a_receptive_only_sentence_note_yields_only_the_listening_card(world):
    syllabus = fully_seeded_syllabus(world)
    compile_syllabus(syllabus, world.db, world.media, world.out_path)
    pkg = read_apkg(world.out_path)

    s_model = _models_by_name(pkg)["sentence"]
    target_idx = None
    tmpl_names = [t["name"] for t in s_model["tmpls"]]
    s_notes = [n for n in pkg["notes"] if str(n["mid"]) == s_model["id"]]

    # pom/receptive is the only target whose skill is receptive -- its
    # tag names it (spec 4 section 2: tags carry target::TARGET).
    pom_note = next(n for n in s_notes if "target::pom/receptive" in n["tags"].split(" "))
    generated = {tmpl_names[c["ord"]] for c in _cards_of(pkg, pom_note["id"])}
    assert generated == {"Listening"}


# --- a pair with no rendition is dropped and counted -----------------------

def test_a_pair_with_no_rendition_is_dropped_and_counted(world):
    syllabus, pair = pair_only_syllabus(SplitTokenizer({}))
    syllabus = dataclasses.replace(syllabus, media=_DbMediaIndex(db=world.db, pairs=(pair,)))
    # Deliberately no seed_rendition call.

    compiled = compile_syllabus(syllabus, world.db, world.media, world.out_path)
    pkg = read_apkg(world.out_path)

    assert "minimal_pair" not in _models_by_name(pkg)
    assert compiled.report.dropped == (
        DroppedCard(family="minimal_pair", kind="Recognition", subject="p1",
                   reason="no rendition"),)
