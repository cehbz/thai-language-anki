"""What a Fluent Forever Thai deck must be, stated as requirements.

Derived from the design spec and the doctrine decisions, not from the
code. Every test drives a public entry point -- `load_derivations` ->
`Syllabus.report()`, `compile_syllabus` -- and asserts a property of the
product. If the implementation is rewritten entirely, these should still
be the tests.

Each name is the requirement it enforces.

This file is the port of the same file written against the old
`thai_deck_eval` evaluator (git history, through daea575). Every doctrine
below was first run on the old implementation over an old-format deck and
its verdict recorded; the assertions here hold the new implementation to
that recorded verdict. Three doctrines the old gate closed on, the new
one does not; each is marked DIVERGENCE and asserts what the new
implementation actually does, so that closing the gap makes the test fail
and points at the doctrine. The full old-vs-new table is in
.superpowers/sdd/2026-09-18-second-engine/doctrine-port-report.md.
"""
import pytest

from thai_syllabus.compile import GateRefusal, build_deck, compile_syllabus, render_card
from thai_syllabus.curated import CuratedValidationError
from thai_syllabus.entities import Category, MinimalPair, Target
from thai_syllabus.ids import PairId, TargetId
from thai_syllabus.wiring import load_derivations

from .doctrine_deck import DeckBuilder, sentence, word


def deck(builder):
    """The deck the CLI sees: one built directory, read back through the
    same wiring `thai-syllabus compile` reads it through.
    """
    return load_derivations(builder.build())


def report(builder):
    """The evaluator's own output: the only thing a caller sees."""
    return deck(builder).syllabus.report()


def rules_fired(builder):
    return {f.rule for f in report(builder).findings}


def cards(d, kind):
    """(subject, front, back) for every compiled card of template `kind`."""
    built = build_deck(d.syllabus, d.db, d.media_store, current_rubric=d.current_rubric,
                       prior=d.prior, provenance_source=d.provenance_source)
    out = []
    for item in built.built:
        for i, template in enumerate(item.model.templates):
            if template["name"] == kind:
                out.append((item.subject, *render_card(item.model, item.note, i)))
    return out


def compile_to(d, tmp_path, *, force=False):
    return compile_syllabus(d.syllabus, d.db, d.media_store, tmp_path / "deck.apkg",
                            current_rubric=d.current_rubric, prior=d.prior,
                            provenance_source=d.provenance_source, force=force)


# --- Doctrine: no translation on picture cards ---

def test_a_picture_card_front_carries_no_l1_gloss(tmp_path):
    """The image is the meaning; a translation on the card defeats the
    card. The old evaluator reported a gloss field on a picture word;
    here the production card's front cannot carry one -- it is the
    picture alone.
    """
    d = deck(DeckBuilder(tmp_path))
    fronts = {subject: front for subject, front, _ in cards(d, "Production")}
    assert "<img" in fronts["rice"]
    assert "cooked rice" not in fronts["rice"]


def test_a_picture_card_back_may_carry_a_gloss(tmp_path):
    """The gloss fixes the sense once the learner has answered."""
    d = deck(DeckBuilder(tmp_path))
    backs = {subject: back for subject, _, back in cards(d, "Listening")}
    assert "cooked rice" in backs["rice"]


def test_a_sentence_may_carry_a_gloss(tmp_path):
    """The community correction to the book: abstract material needs one.
    A sentence's gloss is required here, not merely permitted -- an
    unglossed draft is not adoptable -- and it never closes the gate.
    """
    b = DeckBuilder(tmp_path)
    assert b.sentences[0].gloss
    assert report(b).gate
    backs = [back for _, _, back in cards(deck(b), "Listening")]
    assert any("eat rice" in back for back in backs)


# --- Doctrine: minimal pairs teach one contrast, in a native voice ---

def test_a_minimal_pair_in_a_synthetic_voice_is_reported(tmp_path):
    """Tone-bearing cards must be a human voice: TTS teaches TTS."""
    b = DeckBuilder(tmp_path)
    b.rendition_speaker_kind = "synthetic"
    assert "rendition/synthetic" in rules_fired(b)


def test_a_minimal_pair_in_a_synthetic_voice_closes_the_gate_only_on_request(tmp_path):
    """DIVERGENCE. The old evaluator raised TTS on a pair member to an
    error and failed the gate on it; `rendition/synthetic` is a warning,
    so by default the deck ships. The doctrine is reachable only through
    rulebook.yaml's own severity override -- spec 1 section 4's one
    relaxation path, used here in reverse.
    """
    b = DeckBuilder(tmp_path)
    b.rendition_speaker_kind = "synthetic"
    assert report(b).gate is True          # the old implementation: fail

    strict = DeckBuilder(tmp_path / "strict")
    strict.rendition_speaker_kind = "synthetic"
    strict.severities = {"rendition/synthetic": "error"}
    assert report(strict).gate is False


def test_a_pair_differing_in_more_than_the_declared_contrast_is_refused(tmp_path):
    """Two differences teach neither. The old evaluator scored the deck
    and reported the pair; here the deck does not load at all.
    """
    b = DeckBuilder(tmp_path)
    fried = word("fried", "ผัด", "fried", tone="low", onset="p")  # ผัด: differs in onset too
    b.words = [w for w in b.words if w.id != "far"] + [fried]
    b.pairs = [MinimalPair(id=PairId("tone:mid-low/klai"), confusion="tone:mid-low",
                           members=("near", "fried"))]
    with pytest.raises(CuratedValidationError) as refusal:
        deck(b)
    assert "not exactly {'tone'}" in str(refusal.value)


# --- Doctrine: the deck is a source directory, not an .apkg ---

def test_a_deck_missing_a_media_file_it_references_is_reported_at_compile(tmp_path):
    """DIVERGENCE. The old evaluator made a dangling media reference an
    error and failed the gate. The record here is content-addressed, so
    report() asks the record, never the disk: an object deleted from the
    store closes nothing and is caught only when compile goes to stage
    it, as a warning on a deck that still ships.
    """
    b = DeckBuilder(tmp_path)
    b.orphan_media = True
    d = deck(b)
    assert d.syllabus.report().gate is True          # the old implementation: fail

    result = compile_to(d, tmp_path, force=True)
    missing = [w for w in result.report.warnings if "media object missing on disk" in w]
    assert missing


def test_a_schema_violation_stops_evaluation_rather_than_scoring_it(tmp_path):
    """A deck that does not describe itself is refused, not scored. The
    old evaluator emitted a single schema/invalid finding and stopped;
    here the curated loader refuses, naming every unreadable row.
    """
    b = DeckBuilder(tmp_path)
    b.raw_curated = {"words.yaml": "- not: a valid note\n"}
    with pytest.raises(CuratedValidationError) as refusal:
        deck(b)
    assert "words[0]" in str(refusal.value)


# --- Doctrine: staging, sound system first ---

def test_the_report_measures_contrast_coverage(tmp_path):
    """Staging is the method's spine: the deck must know what it has yet
    to teach.
    """
    measured = {m.rule for m in report(DeckBuilder(tmp_path)).metrics}
    assert {"coverage/confusions", "coverage/categories", "coverage/pictures",
            "coverage/exercise-depth"} <= measured


def test_contrast_coverage_names_each_trained_confusion(tmp_path):
    """A coverage number the deck cannot break down is not a plan."""
    confusions = next(m for m in report(DeckBuilder(tmp_path)).metrics
                      if m.rule == "coverage/confusions")
    assert confusions.detail["tone:mid-low"]["pairs"] >= 1


def test_speaker_diversity_is_measured_for_pairs(tmp_path):
    """One voice teaches one voice's vowels."""
    speakers = next(m for m in report(DeckBuilder(tmp_path)).metrics
                    if m.rule == "coverage/speakers")
    assert speakers.detail["rendition"]["speakers"] >= 1


# --- Doctrine: a sentence introduces one new thing ---

def _two_new_things(tmp_path, introductions):
    """The golden deck plus one sentence "ดื่มน้ำ" ("drink water") over two
    words, each carrying a Target introduced as `introductions` says.
    Both introduced by the sentence, it carries two new things at once.
    """
    b = DeckBuilder(tmp_path)
    drink = word("drink", "ดื่ม", "to drink")    # ดื่ม: to drink
    water = word("water", "น้ำ", "water")        # น้ำ: water
    b.words = b.words + [drink, water]
    b.categories = b.categories + [
        Category(name="Beverages", members=frozenset({"drink", "water"}))]
    b.targets = b.targets + [
        Target(id=TargetId("drink/new"), word="drink", skill="receptive",
               introduction=introductions[0]),
        Target(id=TargetId("water/new"), word="water", skill="receptive",
               introduction=introductions[1])]
    b.sentences = b.sentences + [
        sentence(((drink.id, water.id),), (drink, water), gloss="drink water")]
    b.recordings = b.recordings + ["drink", "water"]
    b.pictures = b.pictures + ["drink", "water"]
    # Every other completeness error relaxed, so what closes the gate
    # below is the novelty finding and nothing standing behind it.
    b.severities = {"target/sentence-required": "warn"}
    return b


def test_a_sentence_introducing_one_new_thing_is_accepted(tmp_path):
    """One introduction per sentence is the whole point of a sentence."""
    b = _two_new_things(tmp_path, ("sentence", "picture_card"))
    rep = report(b)
    assert "sentence/fills-novelty" not in {f.rule for f in rep.findings}
    assert rep.gate is True


def test_a_sentence_introducing_more_than_one_new_thing_is_an_error(tmp_path):
    """A sentence carrying two unmet introductions teaches neither: it
    fills nothing, and that alone closes the gate.
    """
    b = _two_new_things(tmp_path, ("sentence", "sentence"))
    rep = report(b)
    assert {f.rule for f in rep.findings if f.rule != "target/sentence-required"} \
        == {"sentence/fills-novelty"}
    assert rep.gate is False


# --- Output contract ---

def test_the_gate_fails_on_any_error_and_passes_otherwise(tmp_path):
    assert report(DeckBuilder(tmp_path / "clean")).gate is True

    broken = DeckBuilder(tmp_path / "broken")
    broken.recordings = ["eat"]              # rice is targeted and has no voice
    rep = report(broken)
    assert "target/recording-required" in {f.rule for f in rep.findings}
    assert rep.gate is False


def test_a_closed_gate_refuses_to_compile(tmp_path):
    """The gate is not advice: it is what stands between a broken deck
    and the learner's collection.
    """
    broken = DeckBuilder(tmp_path / "broken")
    broken.recordings = ["eat"]
    with pytest.raises(GateRefusal) as refusal:
        compile_to(deck(broken), tmp_path, force=False)
    assert refusal.value.blocking >= 1

    forced = compile_to(deck(DeckBuilder(tmp_path / "forced")), tmp_path, force=True)
    assert forced.report.gate is True


def test_the_report_says_what_it_judged_and_what_judged_it(tmp_path):
    """The old evaluator scored four dimensions and printed the same run
    two ways. This one reports no score at all: a report is findings,
    measures and a gate, stamped with the deck state and the rulebook
    that read it, so a stale verdict is recognisable as one.
    """
    d = deck(DeckBuilder(tmp_path))
    rep = d.syllabus.report()
    assert rep.syllabus_state_id == d.syllabus.state_id()
    assert rep.rulebook_id == d.syllabus.rulebook_id()
    assert isinstance(rep.gate, bool)
