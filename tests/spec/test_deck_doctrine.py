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
that recorded verdict. Two doctrines the old gate closed on, the new one
does not; each is marked DIVERGENCE and asserts what the new
implementation actually does, so that closing the gap makes the test fail
and points at the doctrine. The full old-vs-new table is in
.superpowers/sdd/2026-09-18-second-engine/doctrine-port-report.md.

The sections from "pronunciation is computed on" down were added after
the port, from docs/principles.md directly: each states one principle at
the product level that only a unit test had asserted before. A DIVERGENCE
there is measured against the principle's own text, not the old
evaluator, and likewise asserts what the product does today.
"""
import pytest

from thai_syllabus.compile import GateRefusal, build_deck, compile_syllabus, render_card
from thai_syllabus.curated import CuratedValidationError
from thai_syllabus.entities import Category, MinimalPair, Target
from thai_syllabus.ids import PairId, TargetId
from thai_syllabus.media import Speaker
from thai_syllabus.wiring import load_derivations

from .doctrine_deck import DeckBuilder, PictureCandidate, sentence, word
from .syllabus_world import read_apkg


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


# --- Doctrine: pronunciation is computed on only for pair membership (E4) ---

def _with_disputed(b, word_id):
    b.words = [word(w.id, w.thai, w.meaning, tone=w.pron.syllables[0].tone,
                    onset=w.pron.syllables[0].segments[0], corroboration="disputed")
               if w.id == word_id else w for w in b.words]
    return b


def test_a_disputed_pronunciation_off_a_pair_blocks_no_card(tmp_path):
    """The learner drills pronunciation from audio; the IPA is reference.
    A transcription nobody computes on may be wrong the way a spelling
    may be non-phonetic, and the word's cards ship.
    """
    b = _with_disputed(DeckBuilder(tmp_path), "rice")
    rep = report(b)
    assert "pair/pronunciation-corroborated" not in {f.rule for f in rep.findings}
    assert rep.gate is True
    assert "rice" in {subject for subject, _, _ in cards(deck(b), "Production")}


def test_a_disputed_pronunciation_on_a_pair_member_closes_the_gate(tmp_path):
    """Pair validity is computed from the stored pronunciations, so there
    and only there corroboration is load-bearing.
    """
    b = _with_disputed(DeckBuilder(tmp_path), "near")
    rep = report(b)
    assert "pair/pronunciation-corroborated" in {f.rule for f in rep.findings}
    assert rep.gate is False


# --- Doctrine: a learner's answer is final (F9) ---

def test_a_learner_veto_takes_a_judge_passed_picture_off_the_card(tmp_path):
    """The judge passed it; the learner said no. The picture is gone and
    the word is a gap again, not a card carrying an overruled artifact.
    """
    b = DeckBuilder(tmp_path)
    b.picture_ratings = {"rice": "unacceptable-none"}
    d = deck(b)
    assert "target/picture-required" in {f.rule for f in d.syllabus.report().findings}
    fronts = {subject: front for subject, front, _ in cards(d, "Production")}
    assert b.seeded[("rice", 0)] not in fronts.get("rice", "")


def test_after_a_learner_veto_only_the_learner_can_seat_a_picture(tmp_path):
    """The learner vetoed the picture; the machine found another and the
    judge passed it. It stays a candidate: the learner is asked, not
    overruled, so the card stays empty until the learner rates. (The live
    deck's 42 alphabet keywords sat in exactly this state after their
    corpus photos were vetoed and the illustrator's replacements passed.)
    """
    b = DeckBuilder(tmp_path)
    b.picture_ratings = {"rice": "unacceptable-none"}
    b.picture_candidates = {"rice": [PictureCandidate(judge=True)]}
    d = deck(b)
    assert "target/picture-required" in {f.rule for f in d.syllabus.report().findings}
    fronts = {subject: front for subject, front, _ in cards(d, "Production")}
    assert b.seeded[("rice", 1)] not in fronts.get("rice", "")

    rated = DeckBuilder(tmp_path / "rated")
    rated.picture_ratings = {"rice": "unacceptable-none"}
    rated.picture_candidates = {"rice": [PictureCandidate(judge=True, learner="good")]}
    d = deck(rated)
    fronts = {subject: front for subject, front, _ in cards(d, "Production")}
    assert rated.seeded[("rice", 1)] in fronts["rice"]


def test_a_learner_choice_outranks_a_judge_that_failed_it(tmp_path):
    """The judge failed the second picture; the learner rated it good.
    The learner's picture is on the card, the judge's is not.
    """
    b = DeckBuilder(tmp_path)
    b.picture_candidates = {"rice": [PictureCandidate(judge=False, learner="good")]}
    d = deck(b)
    fronts = {subject: front for subject, front, _ in cards(d, "Production")}
    assert b.seeded[("rice", 1)] in fronts["rice"]
    assert b.seeded[("rice", 0)] not in fronts["rice"]


# --- Doctrine: no unjudged artifact on a card (F11) ---

def test_a_found_but_unjudged_picture_never_reaches_a_card(tmp_path):
    """A picture on record with no verdict is a candidate, not a picture:
    the word is reported as lacking one and no card shows it.
    """
    b = DeckBuilder(tmp_path)
    b.unjudged_pictures = ["rice"]
    d = deck(b)
    rep = d.syllabus.report()
    assert "target/picture-required" in {f.rule for f in rep.findings}
    assert rep.gate is False
    shown = "".join(front + back for _, front, back in
                    cards(d, "Production") + cards(d, "Listening"))
    assert b.seeded[("rice", 0)] not in shown


# --- Doctrine: native audio on what the learner produces (F7) ---

def test_tts_on_a_sentence_the_learner_produces_is_reported(tmp_path):
    """DIVERGENCE from F7's text ("a Sentence filling a productive Target
    carries native audio"): the deck reports it and ships it, the same
    warn-not-error stance the synthetic pair takes above.
    """
    b = DeckBuilder(tmp_path)
    b.sentence_speaker_kind = "synthetic"
    rep = report(b)
    assert "sentence/synthetic-productive" in {f.rule for f in rep.findings}
    assert rep.gate is True                  # F7 as written: fail


def test_tts_on_a_receptive_only_sentence_is_not_reported(tmp_path):
    """Receptive-only Sentences may be TTS."""
    b = DeckBuilder(tmp_path)
    b.sentence_speaker_kind = "synthetic"
    b.targets = [t for t in b.targets if t.id != "rice/productive"]
    assert "sentence/synthetic-productive" not in rules_fired(b)


# --- Doctrine: card identity survives regeneration (A2) ---

def _built(d):
    return build_deck(d.syllabus, d.db, d.media_store, current_rubric=d.current_rubric,
                      prior=d.prior, provenance_source=d.provenance_source).built


def test_a_card_keeps_its_identity_across_compiles_and_a_new_text_is_a_new_card(tmp_path):
    """Identity derives from what the card teaches -- a word card from its
    Word, a sentence card from its text -- so scheduling survives a
    rebuild, and a replaced sentence is a new card.
    """
    first = {(i.family, i.subject): i.note.guid for i in _built(deck(DeckBuilder(tmp_path / "a")))}

    reglossed = DeckBuilder(tmp_path / "b")
    eat, rice = reglossed.words[1], reglossed.words[0]
    reglossed.sentences = [sentence(((eat.id, rice.id),), (eat, rice), gloss="have a meal")]
    second = {(i.family, i.subject): i.note.guid for i in _built(deck(reglossed))}
    assert second == first                   # same words, same text: same cards

    retexted = DeckBuilder(tmp_path / "c")
    retexted.sentences = [sentence(((rice.id, eat.id),), (rice, eat), gloss="rice, eat")]
    third = {(i.family, i.subject): i.note.guid for i in _built(deck(retexted))}
    assert {k: v for k, v in third.items() if k[0] != "sentence"} == \
        {k: v for k, v in first.items() if k[0] != "sentence"}
    assert set(v for k, v in third.items() if k[0] == "sentence").isdisjoint(
        v for k, v in first.items() if k[0] == "sentence")


# --- Doctrine: every review maps back to what it taught (A6) ---

def test_every_note_names_what_it_teaches_and_the_compile_that_made_it(tmp_path):
    d = deck(DeckBuilder(tmp_path))
    compile_to(d, tmp_path)
    subject_prefixes = ("word::", "pair::", "grapheme::", "sentence::", "target::")
    for note in read_apkg(tmp_path / "deck.apkg")["notes"]:
        tags = note["tags"].split()
        assert any(t.startswith("family::") for t in tags), tags
        assert any(t.startswith("kind::") for t in tags), tags
        assert any(t.startswith("compile::") for t in tags), tags
        assert any(t.startswith(subject_prefixes) for t in tags), tags


# --- Doctrine: one picture per word, everywhere (F6a) ---

def test_a_grapheme_card_shows_its_keywords_own_picture(tmp_path):
    """The grapheme borrows the keyword Word's picture; it never has one
    of its own to drift from it.
    """
    b = DeckBuilder(tmp_path)
    d = deck(b)
    backs = [back for subject, _, back in cards(d, "Reading") if subject == "ก"]
    assert backs and b.seeded[("chicken", 0)] in backs[0]


def test_a_chart_cell_drawn_from_a_superseded_keyword_picture_is_not_the_name_words_picture(
        tmp_path):
    """One picture per word, everywhere: the name word's chart cell is
    composed from its keyword's picture, so a cell drawn from a picture
    the keyword no longer has is nobody's picture once a newer draw
    exists -- judge pass or not -- and the name word is a gap until the
    redrawn cell is judged.
    """
    b = DeckBuilder(tmp_path)
    b.stale_cell_from = "0" * 64
    d = deck(b)
    rep = d.syllabus.report()
    assert any(f.rule == "target/picture-required" and f.note_id == "name-chicken"
               for f in rep.findings)
    shown = "".join(front + back for _, front, back in cards(d, "Production"))
    assert b.seeded[("name-chicken", 0)] not in shown
    assert b.seeded[("name-chicken", 1)] not in shown


# --- Doctrine: sounds, then words, then their sentences; receptive before
# productive (E1, F8) ---

def _dues_by_family(tmp_path, d):
    """{family: [(subject template name, due)]} out of the written apkg."""
    compile_to(d, tmp_path)
    pkg = read_apkg(tmp_path / "deck.apkg")
    by_mid = {str(m["id"]): m for m in pkg["models"].values()}
    by_nid = {n["id"]: n for n in pkg["notes"]}
    out: dict[str, list[tuple[str, int]]] = {}
    for card in pkg["cards"]:
        model = by_mid[str(by_nid[card["nid"]]["mid"])]
        template = model["tmpls"][card["ord"]]["name"]
        out.setdefault(model["name"], []).append((template, card["due"]))
    return out


def test_every_grapheme_is_due_before_any_word_and_a_sentence_after_its_words(tmp_path):
    dues = _dues_by_family(tmp_path, deck(DeckBuilder(tmp_path)))
    assert max(due for _, due in dues["grapheme"]) < min(due for _, due in dues["word"])
    assert max(due for _, due in dues["word"]) < min(due for _, due in dues["sentence"])


def test_a_words_receptive_cards_are_due_before_its_production_card(tmp_path):
    """Per word: the listening card of "rice" precedes its production
    card. (Compile stamps one note per word at its earliest Target's block
    and separates the siblings by ord; order() places a word's productive
    Target directly after its receptive one, so nothing sits between.)
    """
    d = deck(DeckBuilder(tmp_path))
    compile_to(d, tmp_path)
    pkg = read_apkg(tmp_path / "deck.apkg")
    by_mid = {str(m["id"]): m for m in pkg["models"].values()}
    rice = next(n for n in pkg["notes"] if n["flds"][0] == "ข้าว")
    model = by_mid[str(rice["mid"])]
    due = {model["tmpls"][c["ord"]]["name"]: c["due"]
           for c in pkg["cards"] if c["nid"] == rice["id"]}
    assert due["Listening"] < due["Production"]


# --- Doctrine: an unknown speaker attribute never counts (E7) ---

def test_an_unknown_speaker_attribute_never_counts_as_coverage(tmp_path):
    b = DeckBuilder(tmp_path)
    anon = Speaker(id="anon", kind="native", sex="unknown", age_band="unknown",
                   region="unknown")
    b.speakers = {subject: anon for subject in b.recordings}
    speakers = next(m for m in report(b).metrics if m.rule == "coverage/speakers")
    assert speakers.detail["recording"]["speakers"] == 1
    assert speakers.detail["recording"]["sex"] == {}
    assert speakers.detail["recording"]["region"] == {}


# --- Doctrine: a word that marks its speaker's sex is voiced by that sex (E3) ---

def test_a_female_marked_word_in_a_male_voice_ships(tmp_path):
    """DIVERGENCE from E3's text ("voiced by a speaker of that sex on
    every card"): the marking is enforced where recordings are sourced
    (the voice constraint on the need) and by the run's veto path, not at
    the deck. A recording supplied directly in the wrong voice is neither
    reported nor dropped.
    """
    b = DeckBuilder(tmp_path)
    dichan = word("i-female", "ดิฉัน", "I (female speaker)", speaker="female")
    b.words = b.words + [dichan]
    b.targets = b.targets + [Target(id=TargetId("i-female/receptive"), word="i-female",
                                    skill="receptive", introduction="picture_card")]
    b.categories = b.categories + [Category(name="Pronouns", members=frozenset({"i-female"}))]
    b.pictures = b.pictures + ["i-female"]
    b.recordings = b.recordings + ["i-female"]      # SOMCHAI, male
    b.severities = {"target/sentence-required": "warn"}
    d = deck(b)
    rep = d.syllabus.report()
    assert rep.gate is True                  # E3 as written: a finding
    backs = {subject: back for subject, _, back in cards(d, "Listening")}
    assert "[sound:" in backs["i-female"]


# --- Doctrine: productive new-card rate capped (F12) ---

def test_production_cards_share_the_one_deck_limit_with_every_other_card(tmp_path):
    """DIVERGENCE from F12's text ("productive new-card rate capped"):
    every card kind lands in one deck under one options group, so the
    only cap is Anki's own new-cards-per-day for the whole deck.
    """
    import json
    import sqlite3
    import tempfile
    import zipfile
    d = deck(DeckBuilder(tmp_path))
    compile_to(d, tmp_path)
    pkg = read_apkg(tmp_path / "deck.apkg")
    assert len({card["did"] for card in pkg["cards"]}) == 1
    with zipfile.ZipFile(tmp_path / "deck.apkg") as zf:
        db_bytes = zf.read("collection.anki2")
    with tempfile.TemporaryDirectory() as tmp:
        db_path = tmp_path / "col.anki2"
        db_path.write_bytes(db_bytes)
        conn = sqlite3.connect(str(db_path))
        (dconf_json,) = conn.execute("select dconf from col").fetchone()
        conn.close()
    assert len(json.loads(dconf_json)) == 1
