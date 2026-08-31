"""What the compiled .apkg must guarantee.

Stated as properties of the artifact a learner imports, verified by reading
the package back. Nothing here knows how the compiler works.
"""
import pytest

from thai_deck_gen.compiler.build import CompileError, compile_deck
from tests.gen.helpers_apkg import read_apkg
from tests.gen.test_compiler import (FREQ, PAIR_BY_NOTE, _deck, _manifest,
                                     _write_media)


def _compile(tmp_path, deck=None, **kwargs):
    deck = deck or _deck(tmp_path)
    _write_media(deck, skip=kwargs.pop("skip", None))
    out = tmp_path / "out.apkg"
    dropped = compile_deck(deck, _manifest(), out, FREQ, PAIR_BY_NOTE,
                           base=kwargs.pop("base", 0), **kwargs)
    return read_apkg(out), dropped, deck


def _fields(data, model_name):
    models = {int(m["id"]): m["name"] for m in data["models"].values()}
    return [n for n in data["notes"] if models[n["mid"]] == model_name]


def test_recompiling_updates_the_same_notes_rather_than_duplicating_them(tmp_path):
    """Anki merges on guid: a changed deck must not reset the learner's
    scheduling or double their cards."""
    first, _, deck = _compile(tmp_path)
    deck.picture_words[0].ipa = "changed"
    second, _, _ = _compile(tmp_path, deck=deck)
    assert {n["guid"] for n in first["notes"]} == {n["guid"] for n in second["notes"]}


def test_a_note_keeps_its_guid_when_its_content_changes(tmp_path):
    first, _, deck = _compile(tmp_path)
    before = {n["guid"] for n in _fields(first, "picture_word")}
    deck.picture_words[0].thai = "totally different"
    second, _, _ = _compile(tmp_path, deck=deck)
    assert before == {n["guid"] for n in _fields(second, "picture_word")}


def test_cards_are_ordered_for_study_not_by_accident(tmp_path):
    """Introduction order is the method: sounds and frequent words first."""
    data, _, _ = _compile(tmp_path)
    dues = [c["due"] for c in data["cards"]]
    assert len(set(dues)) > 1
    assert min(dues) == 0


def test_a_minimal_pair_becomes_one_card_per_member(tmp_path):
    """Each side must be asked on its own, or the contrast is never tested."""
    data, _, deck = _compile(tmp_path)
    members = len(deck.minimal_pairs[0].members)
    assert len(_fields(data, "minimal_pair")) == members


def test_every_referenced_media_file_is_inside_the_package(tmp_path):
    data, _, _ = _compile(tmp_path)
    referenced = set()
    for note in data["notes"]:
        for field in note["flds"]:
            if "[sound:" in field:
                referenced.add(field.split("[sound:")[1].split("]")[0])
            if "<img src=" in field:
                referenced.add(field.split('<img src="')[1].split('"')[0])
    assert referenced <= set(data["media"])


def test_missing_media_stops_the_compile_by_default(tmp_path):
    """A deck that references what it does not have is not shippable."""
    with pytest.raises(CompileError):
        _compile(tmp_path, skip="images/pw-1.jpg")


def test_skip_incomplete_drops_only_what_it_must(tmp_path):
    """A half-recorded deck should still be studiable: drop the note whose
    essential medium is missing, keep the rest."""
    data, dropped, _ = _compile(tmp_path, skip="images/pw-1.jpg",
                                skip_incomplete=True)
    assert ("picture_word", "pw-1") in dropped
    assert len(data["notes"]) > 0
    assert all("pw-1" not in "".join(n["flds"]) for n in data["notes"])


def test_an_optional_medium_degrades_instead_of_dropping_the_note(tmp_path):
    """A picture word without its recording still teaches the word."""
    data, dropped, _ = _compile(tmp_path, skip="audio/picture_words/pw-1.mp3",
                                skip_incomplete=True)
    assert dropped == []
    word = next(n for n in _fields(data, "picture_word") if n["flds"][0] == "one")
    assert word["flds"][2] == ""


def test_provenance_travels_with_the_cards(tmp_path):
    """Where a picture or a voice came from must survive into the deck."""
    data, _, _ = _compile(tmp_path)
    tags = " ".join(n["tags"] for n in data["notes"])
    assert "audio-src::forvo" in tags


def test_cards_carry_their_family_and_stage(tmp_path):
    data, _, _ = _compile(tmp_path)
    tags = " ".join(n["tags"] for n in data["notes"])
    for family in ("minimal_pair", "picture_word", "sentence", "spelling_sound"):
        assert f"family::{family}" in tags


def test_a_contrast_is_recorded_on_the_pair_it_teaches(tmp_path):
    """Without this the learner cannot review by confusion, which is the
    whole point of a pair deck."""
    data, _, _ = _compile(tmp_path)
    pair_tags = " ".join(n["tags"] for n in _fields(data, "minimal_pair"))
    assert "contrast::tone:mid-falling" in pair_tags
