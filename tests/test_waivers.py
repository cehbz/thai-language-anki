"""Human overrides of findings the deck owner has reviewed and accepted."""
import hashlib

import yaml

from thai_deck_eval.core import pipeline as pipeline_mod
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Dimension, Finding, Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.core.registry import RuleDef
from thai_deck_eval.model.deck import load_deck
from thai_deck_eval.waivers import Waiver, load_waivers, partition
from tests.helpers import DeckBuilder


def _deck(tmp_path):
    return load_deck(DeckBuilder(tmp_path).build())


def _image_sha(deck, note):
    return hashlib.sha256((deck.root / "media" / note.image).read_bytes()).hexdigest()


def _finding(rule, note_id, severity=Severity.WARN):
    return Finding(rule=rule, severity=severity, dimension=Dimension.CONTENT,
                   message="flagged", note_id=note_id)


def test_load_waivers_reads_the_deck_file(tmp_path):
    (tmp_path / "waivers.yaml").write_text(yaml.safe_dump([{
        "note_id": "pw-1", "rule": "judge/image-embedded-text",
        "sha": "abc", "reason": "the expiry date is the point",
        "date": "2026-08-30"}]), encoding="utf-8")
    waivers = load_waivers(tmp_path)
    assert waivers[0].note_id == "pw-1"
    assert waivers[0].reason == "the expiry date is the point"


def test_load_waivers_is_empty_without_a_file(tmp_path):
    assert load_waivers(tmp_path) == []


def test_matching_waiver_waives_the_finding(tmp_path):
    deck = _deck(tmp_path)
    note = deck.picture_words[0]
    f = _finding("judge/image-embedded-text", note.id)
    kept, waived, stale = partition([f], deck, [Waiver(
        note_id=note.id, rule="judge/image-embedded-text",
        sha=_image_sha(deck, note), reason="reviewed", date="2026-08-30")])
    assert kept == [] and waived == [f] and stale == []


def test_waiver_for_another_rule_does_not_apply(tmp_path):
    deck = _deck(tmp_path)
    note = deck.picture_words[0]
    f = _finding("judge/image-irrelevant", note.id)
    kept, waived, _ = partition([f], deck, [Waiver(
        note_id=note.id, rule="judge/image-embedded-text", sha=None,
        reason="reviewed", date="2026-08-30")])
    assert kept == [f] and waived == []


def test_a_waiver_whose_image_changed_stops_applying(tmp_path):
    """Approval is of one image, not of the note forever."""
    deck = _deck(tmp_path)
    note = deck.picture_words[0]
    f = _finding("judge/image-embedded-text", note.id)
    waiver = Waiver(note_id=note.id, rule="judge/image-embedded-text",
                    sha="the-image-it-approved", reason="reviewed",
                    date="2026-08-30")
    kept, waived, stale = partition([f], deck, [waiver])
    assert kept == [f]                    # the finding stands against a new image
    assert waived == [] and stale == [waiver]


def test_waived_error_gates_neither_the_run_nor_dependent_stages(tmp_path, monkeypatch):
    deck = _deck(tmp_path)
    note = deck.picture_words[0]
    boom = RuleDef("mech/fake-error", Stage.MECHANICAL, Dimension.INTEGRITY,
                   Severity.ERROR,
                   lambda ctx: [_finding("mech/fake-error", note.id, Severity.ERROR)])
    monkeypatch.setattr(pipeline_mod, "rules_for",
                        lambda stage: [boom] if stage == Stage.MECHANICAL else [])

    ctx = EvalContext(deck=deck, judge=object(),
                      config={"depends_on": {"judge": ["schema", "mechanical"]}},
                      waivers=[Waiver(note_id=note.id, rule="mech/fake-error",
                                      sha=_image_sha(deck, note),
                                      reason="reviewed", date="2026-08-30")])
    res = run_pipeline(ctx, stages=[Stage.MECHANICAL, Stage.JUDGE])

    assert not res.has_errors
    assert [f.rule for f in res.waived] == ["mech/fake-error"]
    assert Stage.JUDGE in res.stages_run


def test_stale_waiver_is_reported_through_the_pipeline(tmp_path, monkeypatch):
    deck = _deck(tmp_path)
    note = deck.picture_words[0]
    flag = RuleDef("judge/fake", Stage.JUDGE, Dimension.CONTENT, Severity.WARN,
                   lambda ctx: [_finding("judge/fake", note.id)])
    monkeypatch.setattr(pipeline_mod, "rules_for",
                        lambda stage: [flag] if stage == Stage.JUDGE else [])

    ctx = EvalContext(deck=deck, judge=object(),
                      config={"depends_on": {"judge": ["schema"]}},
                      waivers=[Waiver(note_id=note.id, rule="judge/fake",
                                      sha="an-image-that-is-gone",
                                      reason="reviewed", date="2026-08-30")])
    res = run_pipeline(ctx, stages=[Stage.JUDGE])
    assert {f.rule for f in res.findings} == {"judge/fake", "waiver/stale"}
