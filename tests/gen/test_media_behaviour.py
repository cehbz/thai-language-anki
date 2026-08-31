"""Media plumbing: what gets fetched, what gets rejected, what gets scanned.

Behaviour of the media layer stated in terms a run cares about -- which
needs are found, which clips are kept, which failures stop a run and which
merely skip an item.
"""
import json
import subprocess

import pytest

from thai_deck_eval.model.notes import (Audio, MinimalPairNote, PairMember,
                                        PictureWordNote, SentenceNote,
                                        SpellingSoundNote)
from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.media.ffmpeg import AudioError, duration_ok, normalize_audio
from thai_deck_gen.media.scan import (NATIVE_TIER_FAMILIES, pending_audio,
                                      pending_images)


def _probe(duration=None, returncode=0, stdout=None):
    def runner(cmd, **kwargs):
        payload = stdout if stdout is not None else json.dumps(
            {"format": {"duration": str(duration)}} if duration is not None else {})
        return subprocess.CompletedProcess(cmd, returncode, payload, "")
    return runner


@pytest.mark.parametrize("duration,ok", [
    (1.2, True), (0.2, True), (5.0, True), (0.05, False), (9.0, False),
])
def test_duration_bounds_are_inclusive(tmp_path, duration, ok):
    """Forvo clips are user-uploaded: a two-second word is fine, a nine
    second one is somebody's sentence."""
    assert duration_ok(tmp_path / "a.mp3", runner=_probe(duration)) is ok


def test_unreadable_audio_is_rejected_rather_than_assumed_good(tmp_path):
    assert duration_ok(tmp_path / "a.mp3", runner=_probe(returncode=1)) is False
    assert duration_ok(tmp_path / "a.mp3", runner=_probe(stdout="not json")) is False
    assert duration_ok(tmp_path / "a.mp3", runner=_probe(stdout="{}")) is False


def test_normalize_reports_what_ffmpeg_said(tmp_path):
    def runner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, b"", b"codec not found")
    with pytest.raises(AudioError) as err:
        normalize_audio(b"raw", tmp_path / "o.mp3", runner=runner)
    assert "codec not found" in str(err.value)


def test_normalize_asks_for_mono_44k_loudnorm(tmp_path):
    seen = {}
    def runner(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    normalize_audio(b"raw", tmp_path / "o.mp3", runner=runner)
    assert seen["cmd"][seen["cmd"].index("-ac") + 1] == "1"
    assert seen["cmd"][seen["cmd"].index("-ar") + 1] == "44100"
    assert "loudnorm" in seen["cmd"]


# --- scanning: which needs a deck reports ---

def _audio(path, source="native", speaker="pending"):
    return Audio(file=path, source=source, speaker=speaker)


def _full_deck(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds", "words", "sentences"])
    deck.minimal_pairs = [MinimalPairNote(
        id="mp-1", contrast="tone",
        members=[PairMember(thai="a", ipa="a", audio=_audio("audio/mp/1_0.mp3")),
                 PairMember(thai="b", ipa="b", audio=_audio("audio/mp/1_1.mp3"))])]
    deck.spelling_sound = [SpellingSoundNote(
        id="sp-1", pattern="-ะ", pattern_kind="vowel", example_word="d",
        audio=_audio("audio/sp/1.mp3"), image="images/sp-1.jpg")]
    deck.picture_words = [PictureWordNote(
        id="pw-1", thai="w", image="images/pw-1.jpg",
        audio=_audio("audio/pw/1.mp3"), frequency_rank=1, category="Food")]
    deck.sentences = [SentenceNote(
        id="sn-1", kind="new_word", thai="s", target="w",
        audio=_audio("audio/sn/1.mp3", source="tts"), image="images/sn-1.jpg")]
    write_deck(deck)
    return deck


def test_every_family_with_audio_is_scanned(tmp_path):
    families = {n.family for n in pending_audio(_full_deck(tmp_path))}
    assert families == {"minimal_pair", "spelling_sound", "picture_word", "sentence"}


def test_native_tier_excludes_sentences(tmp_path):
    """Sentences are TTS tier: commissioning a speaker for 700 of them is
    the mistake this boundary prevents."""
    needs = pending_audio(_full_deck(tmp_path))
    native = {n.family for n in needs if n.family in NATIVE_TIER_FAMILIES}
    assert "sentence" not in native
    assert native == {"minimal_pair", "spelling_sound", "picture_word"}


def test_pair_members_are_scanned_individually(tmp_path):
    needs = [n for n in pending_audio(_full_deck(tmp_path))
             if n.family == "minimal_pair"]
    assert sorted(n.member_index for n in needs) == [0, 1]
    assert all(n.native_required for n in needs), "a pair card is its audio"


def test_a_present_file_with_a_real_speaker_is_not_a_need(tmp_path):
    deck = _full_deck(tmp_path)
    path = deck.root / "media" / deck.picture_words[0].audio.file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"mp3")
    deck.picture_words[0].audio.speaker = "forvo:someone"
    assert not [n for n in pending_audio(deck) if n.note_id == "pw-1"]


def test_every_family_with_an_image_is_scanned(tmp_path):
    families = {n.family for n in pending_images(_full_deck(tmp_path))}
    assert families == {"spelling_sound", "picture_word", "sentence"}


def test_flagged_notes_are_needs_even_with_a_file_present(tmp_path):
    deck = _full_deck(tmp_path)
    for ref in ("images/pw-1.jpg", "images/sp-1.jpg", "images/sn-1.jpg"):
        p = deck.root / "media" / ref
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpg")
    assert not pending_images(deck)
    assert {n.note_id for n in pending_images(deck, flagged={"pw-1", "sn-1"})} == \
        {"pw-1", "sn-1"}
