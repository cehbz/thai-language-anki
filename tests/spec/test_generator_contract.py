"""What a generation run promises.

Stated as properties of the run and the deck it leaves behind. The world
outside the process is faked; the deck directory is real, and every
assertion reads the artifact or what the run consumed.
"""
import pytest

from thai_deck_gen.media.forvo import fetch_forvo
from thai_deck_gen.media.forvo_memo import ForvoMemo
from thai_deck_gen.media.images import fill_images
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import (NATIVE_TIER_FAMILIES, pending_audio,
                                      pending_images)
from thai_deck_gen.media.tts import fill_tts
from thai_deck_gen.orchestrator import generate
from thai_deck_gen.producers.words import fill_words
from thai_deck_gen.report import parse_report
from tests.spec.world import World, report


def _gaps(rep=None):
    from pathlib import Path
    return parse_report(rep or report(), Path("data") / "contrasts.yaml")


# --- the loop ---

def test_a_run_closes_the_gaps_its_report_names(tmp_path):
    world = World(tmp_path)
    generate(world.deck(), world.context(),
             evaluate=lambda root: report(missing_categories=["Food", "Animals"]))
    assert {n.thai for n in world.deck().picture_words} == {w.thai for w in world.words}


def test_running_twice_adds_nothing(tmp_path):
    """Idempotence: the deck is the state, and a repeat run is a no-op."""
    world = World(tmp_path)
    ctx = world.context()
    gapped = lambda root: report(missing_categories=["Food", "Animals"])
    generate(world.deck(), ctx, evaluate=gapped)
    after_first = len(world.deck().picture_words)
    generate(world.deck(), ctx, evaluate=gapped)
    assert len(world.deck().picture_words) == after_first


def test_a_run_stops_when_the_gaps_stop_changing(tmp_path):
    """Without this the loop bills forever against a gap it cannot close."""
    world = World(tmp_path, max_iterations=10)
    seen = []

    def evaluate(root):
        seen.append(1)
        return report(missing_contrasts=["tone:mid-low"])

    generate(world.deck(), world.context(), evaluate=evaluate)
    assert len(seen) < 10, "the loop did not detect that it had stalled"


def test_a_run_stops_when_there_is_nothing_left_to_fill(tmp_path):
    world = World(tmp_path, max_iterations=10)
    calls = []
    clean = report()
    clean["metrics"][2]["value"] = 1.0

    def evaluate(root):
        calls.append(1)
        return {**clean, "gate": "pass", "findings": []}

    generate(world.deck(), world.context(), evaluate=evaluate)
    assert len(calls) == 1


def test_max_iterations_bounds_the_run(tmp_path):
    world = World(tmp_path, max_iterations=2)
    calls = []

    def evaluate(root):
        calls.append(len(calls))
        return report(missing_categories=[f"cat{len(calls)}"])

    generate(world.deck(), world.context(), evaluate=evaluate)
    assert len(calls) <= 2


# --- never invent ---

def test_a_word_whose_pronunciation_cannot_be_verified_is_queued_not_guessed(tmp_path):
    """Authoring IPA the engines cannot confirm would teach a wrong sound."""
    world = World(tmp_path)

    class NoG2P:
        def syllables(self, w): return None

    ctx = world.context(g2p=NoG2P())
    fill_words(_gaps(), world.deck(), ctx)
    queued = world.work_file("ipa_adjudication.yaml")
    assert set(queued) == {w.thai for w in world.words}


def test_an_image_is_never_accepted_without_a_verdict(tmp_path):
    """A judge configured means no picture enters the deck unjudged."""
    world = World(tmp_path, images_pass=False)
    deck = world.deck()
    fill_words(_gaps(), deck, world.context())
    res = fill_images(pending_images(deck), _gaps(), deck,
                      Manifest.load(deck.root), world.context(), "2026-08-31",
                      judge=world)
    assert res.changed == 0
    assert world.spend.judgments > 0
    assert not [n for n in deck.picture_words
                if (deck.root / "media" / n.image).exists()]


# --- spend is never repeated ---

def test_a_word_forvo_lacks_is_looked_up_once_ever(tmp_path):
    """A daily quota makes the answer worth more than the audio."""
    world = World(tmp_path, forvo_has=())
    deck = world.deck()
    fill_words(_gaps(), deck, world.context())
    needs = [n for n in pending_audio(deck) if n.family in NATIVE_TIER_FAMILIES]

    fetch_forvo(needs, deck, Manifest.load(deck.root), world._Forvo(world),
                "2026-08-31", memo=ForvoMemo.load(deck.root))
    first = list(world.spend.forvo_lookups)
    fetch_forvo(needs, deck, Manifest.load(deck.root), world._Forvo(world),
                "2026-09-01", memo=ForvoMemo.load(deck.root))
    assert world.spend.forvo_lookups == first, "a known miss was re-queried"
    assert first, "nothing was looked up at all"


def test_an_unchanged_image_is_never_re_judged(tmp_path):
    world = World(tmp_path, images_pass=False)
    deck = world.deck()
    fill_words(_gaps(), deck, world.context())
    fill_images(pending_images(deck), _gaps(), deck, Manifest.load(deck.root),
                world.context(), "2026-08-31", judge=world)
    spent = world.spend.judgments
    fill_images(pending_images(deck), _gaps(), deck, Manifest.load(deck.root),
                world.context(), "2026-09-01", judge=world)
    assert world.spend.judgments == spent, "a settled word was judged again"


# --- media tiering ---

def test_sentences_are_never_sent_for_native_recording(tmp_path):
    """Commissioning a speaker for hundreds of sentences is the mistake this
    boundary exists to prevent."""
    world = World(tmp_path)
    deck = world.deck()
    fill_words(_gaps(), deck, world.context())
    native = [n for n in pending_audio(deck) if n.family in NATIVE_TIER_FAMILIES]
    assert all(n.family != "sentence" for n in native)


def test_synthetic_audio_is_only_used_for_sentences(tmp_path):
    world = World(tmp_path)
    deck = world.deck()
    fill_words(_gaps(), deck, world.context())
    fill_tts(pending_audio(deck), deck, Manifest.load(deck.root),
             world._Tts(world), "2026-08-31")
    assert world.spend.tts_calls == [], "a picture word was given a synthetic voice"


# --- provenance ---

def test_every_file_a_run_writes_is_recorded_with_where_it_came_from(tmp_path):
    world = World(tmp_path, forvo_has={"ข้าว", "หมา"})
    deck = world.deck()
    fill_words(_gaps(), deck, world.context())
    manifest = Manifest.load(deck.root)
    import thai_deck_gen.media.forvo as forvo_mod
    forvo_mod.normalize_audio = lambda raw, dst, runner=None: dst.write_bytes(raw)
    forvo_mod.duration_ok = lambda path: True
    fetch_forvo([n for n in pending_audio(deck) if n.family in NATIVE_TIER_FAMILIES],
                deck, manifest, world._Forvo(world), "2026-08-31",
                memo=ForvoMemo.load(deck.root))

    entries = world.manifest()
    assert entries, "nothing was recorded"
    for entry in entries.values():
        assert entry["channel"] and entry["origin"] and entry["fetched"]


# --- fault tolerance ---

def test_one_item_failing_does_not_stop_the_others(tmp_path):
    world = World(tmp_path, forvo_has={"หมา"})
    deck = world.deck()
    fill_words(_gaps(), deck, world.context())
    import thai_deck_gen.media.forvo as forvo_mod
    forvo_mod.normalize_audio = lambda raw, dst, runner=None: dst.write_bytes(raw)
    forvo_mod.duration_ok = lambda path: True

    res = fetch_forvo([n for n in pending_audio(deck)
                       if n.family in NATIVE_TIER_FAMILIES],
                      deck, Manifest.load(deck.root), world._Forvo(world),
                      "2026-08-31", memo=ForvoMemo.load(deck.root))
    assert res.changed >= 1 and res.blocked, "a miss should block only itself"


def test_every_note_gets_its_own_identity(tmp_path):
    """Ids name media paths and Anki guids: two notes sharing one id would
    overwrite each other's picture and merge into one card."""
    world = World(tmp_path)

    class TiedRanks:
        def rank(self, word): return 1        # a frequency list with ties

    deck = world.deck()
    fill_words(_gaps(), deck, world.context(freq=TiedRanks()))
    ids = [n.id for n in deck.picture_words]
    assert len(ids) == len(set(ids)), f"duplicate note ids: {ids}"
    images = [n.image for n in deck.picture_words]
    assert len(images) == len(set(images)), "two notes share one image path"
