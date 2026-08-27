from pathlib import Path

import pytest
from thai_deck_eval.model.notes import (Audio, MinimalPairNote, PairMember,
                                        PictureWordNote, SentenceNote,
                                        SpellingSoundNote)
from thai_deck_gen.compiler.build import MODELS, CompileError, compile_deck
from thai_deck_gen.deckio import new_deck
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from tests.gen.helpers_apkg import read_apkg
from tests.gen.test_words import FakeFreq

FREQ = FakeFreq({"a1": 1, "a2": 2, "d": 5})

def _audio(path):
    return Audio(file=path, source="native", speaker="pending")

def _deck(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds", "words", "sentences"])
    deck.minimal_pairs = [MinimalPairNote(
        id="mp-1", contrast="tone",
        members=[PairMember(thai="a1", ipa="a1-ipa",
                            audio=_audio("audio/minimal_pairs/mp-1_0.mp3")),
                PairMember(thai="a2", ipa="a2-ipa",
                          audio=_audio("audio/minimal_pairs/mp-1_1.mp3"))])]
    deck.spelling_sound = [SpellingSoundNote(
        id="sp-1", pattern="-ะ", pattern_kind="vowel", example_word="d",
        audio=_audio("audio/spelling_sound/sp-1.mp3"), image="images/sp-1.jpg")]
    deck.picture_words = [
        PictureWordNote(id="pw-1", thai="one", image="images/pw-1.jpg",
                        audio=_audio("audio/picture_words/pw-1.mp3"),
                        frequency_rank=1, category="Numbers", test_spelling=False),
        PictureWordNote(id="pw-2", thai="two", image="images/pw-2.jpg",
                        audio=_audio("audio/picture_words/pw-2.mp3"),
                        frequency_rank=2, category="Numbers", test_spelling=True),
    ]
    deck.sentences = [SentenceNote(
        id="sn-1", kind="new_word", thai="foo one bar", target="one",
        audio=_audio("audio/sentences/sn-1.mp3"))]
    return deck

def _write_media(deck, skip=None):
    for family, note in deck.all_notes():
        if family == "minimal_pair":
            for m in note.members:
                _write(deck, m.audio.file, skip)
        else:
            _write(deck, note.audio.file, skip)
            image = getattr(note, "image", None)
            if image:
                _write(deck, image, skip)

def _write(deck, ref, skip):
    if ref == skip:
        return
    path = deck.root / "media" / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

def _manifest():
    m = Manifest()
    m.record(MediaEntry(file="media/audio/minimal_pairs/mp-1_0.mp3", channel="forvo",
                        origin="https://forvo.example/a1", speaker="forvo:joe",
                        fetched="2026-08-27"))
    return m

PAIR_BY_NOTE = {"mp-1": "tone:mid-falling"}

def test_compile_deck_models_notes_and_cards(tmp_path):
    deck = _deck(tmp_path)
    _write_media(deck)
    out = tmp_path / "out.apkg"
    compile_deck(deck, _manifest(), out, FREQ, PAIR_BY_NOTE, base=0)

    data = read_apkg(out)
    assert len(data["models"]) == 4
    assert set(m["name"] for m in data["models"].values()) == set(MODELS)

    # 2 pair members + spelling + 2 words + 1 sentence = 6 genanki notes
    assert len(data["notes"]) == 6

    cards_by_nid: dict[int, list] = {}
    for c in data["cards"]:
        cards_by_nid.setdefault(c["nid"], []).append(c)
    notes_by_id = {n["id"]: n for n in data["notes"]}
    model_name_by_id = {int(m["id"]): m["name"] for m in data["models"].values()}

    def note_family(note):
        return model_name_by_id[note["mid"]]

    counts_by_family: dict[str, list[int]] = {}
    for nid, cards in cards_by_nid.items():
        fam = note_family(notes_by_id[nid])
        counts_by_family.setdefault(fam, []).append(len(cards))

    assert sorted(counts_by_family["minimal_pair"]) == [1, 1]
    assert sorted(counts_by_family["spelling_sound"]) == [2]
    assert sorted(counts_by_family["picture_word"]) == [2, 3]   # False -> 2, True -> 3
    assert sorted(counts_by_family["sentence"]) == [2]          # Audio present -> listening card

def test_compile_deck_due_matches_intro_order(tmp_path):
    deck = _deck(tmp_path)
    _write_media(deck)
    out = tmp_path / "out.apkg"
    compile_deck(deck, _manifest(), out, FREQ, PAIR_BY_NOTE, base=0)

    data = read_apkg(out)
    model_name_by_id = {int(m["id"]): m["name"] for m in data["models"].values()}
    notes_by_id = {n["id"]: n for n in data["notes"]}

    def identity(note):
        fam = model_name_by_id[note["mid"]]
        if fam == "minimal_pair":
            return (fam, note["flds"][1])          # Thai
        if fam == "spelling_sound":
            return (fam, note["flds"][0])          # Pattern
        if fam == "picture_word":
            return (fam, note["flds"][0])          # Thai
        return (fam, note["flds"][1])              # Target

    due_by_nid: dict[int, set] = {}
    for c in data["cards"]:
        due_by_nid.setdefault(c["nid"], set()).add(c["due"])

    for nid, dues in due_by_nid.items():
        assert len(dues) == 1, "all cards of one note share the same due"

    ordered = sorted(due_by_nid, key=lambda nid: next(iter(due_by_nid[nid])))
    sequence = [identity(notes_by_id[nid]) for nid in ordered]

    assert sequence == [
        ("minimal_pair", "a1"), ("minimal_pair", "a2"),
        ("spelling_sound", "-ะ"),
        ("picture_word", "one"), ("sentence", "one"),
        ("picture_word", "two"),
    ]

def test_compile_deck_guids_stable_across_recompiles(tmp_path):
    deck = _deck(tmp_path)
    _write_media(deck)
    out1 = tmp_path / "out1.apkg"
    out2 = tmp_path / "out2.apkg"
    compile_deck(deck, _manifest(), out1, FREQ, PAIR_BY_NOTE, base=0)
    compile_deck(deck, _manifest(), out2, FREQ, PAIR_BY_NOTE, base=0)

    guids1 = {n["guid"] for n in read_apkg(out1)["notes"]}
    guids2 = {n["guid"] for n in read_apkg(out2)["notes"]}
    assert guids1 == guids2
    assert len(guids1) == 6

def test_compile_deck_tags_contrast_and_audio_src(tmp_path):
    deck = _deck(tmp_path)
    _write_media(deck)
    out = tmp_path / "out.apkg"
    compile_deck(deck, _manifest(), out, FREQ, PAIR_BY_NOTE, base=0)

    data = read_apkg(out)
    model_name_by_id = {int(m["id"]): m["name"] for m in data["models"].values()}
    pair_notes = [n for n in data["notes"]
                 if model_name_by_id[n["mid"]] == "minimal_pair"]
    tag_sets = [set(n["tags"].split()) for n in pair_notes]

    assert all("contrast::tone:mid-falling" in t for t in tag_sets)
    assert any("audio-src::forvo" in t for t in tag_sets)          # member 0 only
    assert any("audio-src::forvo" not in t for t in tag_sets)      # member 1 unknown channel
    assert all("family::minimal_pair" in t and "stage::sounds" in t for t in tag_sets)

def test_compile_deck_missing_media_raises_and_writes_nothing(tmp_path):
    deck = _deck(tmp_path)
    missing_ref = "audio/picture_words/pw-1.mp3"
    _write_media(deck, skip=missing_ref)
    out = tmp_path / "out.apkg"

    with pytest.raises(CompileError) as exc:
        compile_deck(deck, _manifest(), out, FREQ, PAIR_BY_NOTE, base=0)

    assert missing_ref in exc.value.files
    assert not out.exists()
