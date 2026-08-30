import thai_deck_eval.stages.method  # noqa: F401
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.model.deck import load_deck
from tests.fakes import FakeTokenizer
from tests.helpers import DeckBuilder
from tests.test_linguistic import FREQ, G2P, TOK

def _run(root, config=None, **ctx_kw):
    kw = dict(g2p=G2P, tokenizer=TOK, freq=FREQ)
    kw.update(ctx_kw)
    return run_pipeline(EvalContext(deck=load_deck(root),
                                    config=config or {"sentence_base": 2}, **kw),
                        stages=[Stage.METHOD])

def _metric(res, name):
    return next(m for m in res.metrics if m.name == name)

def test_pair_coverage_metric(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    m = _metric(res, "coverage/minimal_pairs")
    assert 0 < m.value < 1
    assert "tone:low-rising" in m.detail["covered"]      # ขาว(rising)/ข่าว(low)
    assert "aspiration:velar" in m.detail["covered"]     # ไก่/ไข่
    assert "tone:mid-low" in m.detail["missing"]

def test_spelling_coverage_metric(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    m = _metric(res, "coverage/spelling")
    assert 0 < m.value < 0.1  # 1 of ~69 targets

def test_classifier_missing(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["classifier"] = None
    res = _run(b.build())
    assert any(f.rule == "meth/classifier-missing" for f in res.findings)

def test_tts_on_pair_is_error(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["minimal_pairs"][0]["members"][0]["audio"]["source"] = "tts"
    res = _run(b.build())
    f = next(f for f in res.findings if f.rule == "meth/tts-audio")
    assert f.severity == Severity.ERROR

def test_tts_on_picture_word_is_warn(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["audio"]["source"] = "tts"
    res = _run(b.build())
    f = next(f for f in res.findings if f.rule == "meth/tts-audio")
    assert f.severity == Severity.WARN

def test_spelling_taper(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][2]["test_spelling"] = True   # rank 90 → ok
    b.data["picture_words"][0]["frequency_rank"] = 400
    b.data["picture_words"][0]["test_spelling"] = True   # rank 400 → info
    res = _run(b.build())
    hits = [f for f in res.findings if f.rule == "meth/spelling-taper"]
    assert [f.note_id for f in hits] == ["w-dog"]

def test_premature_sentences(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())   # sentence_base=2, 3 words → ok
    assert not any(f.rule == "meth/premature-sentences" for f in res.findings)
    b = DeckBuilder(tmp_path / "b")
    b.data["picture_words"] = b.data["picture_words"][:1]
    res = _run(b.build())
    assert any(f.rule == "meth/premature-sentences" for f in res.findings)

def test_new_elements(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["thai"] = "หมาวิ่งกิน"
    b.data["sentences"][0]["target"] = "กิน"
    tok = FakeTokenizer({"หมาวิ่งกิน": ["หมา", "วิ่ง", "กิน"]})
    res = _run(b.build(), tokenizer=tok)
    f = next(f for f in res.findings if f.rule == "meth/new-elements")
    assert f.evidence["unknown"] == ["วิ่ง"]

def test_new_elements_boundary_aligned_known(tmp_path):
    # token กินข้าว is not itself known, but ends with the known word ข้าว
    # (a picture word in the golden deck) -> boundary-aligned known, not
    # reported as an unknown element.
    b = DeckBuilder(tmp_path)
    tok = FakeTokenizer({"หมามากินข้าว": ["หมา", "มา", "กินข้าว"]})
    res = _run(b.build(), tokenizer=tok)
    assert not any(f.rule == "meth/new-elements" for f in res.findings)

def test_speaker_diversity(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    assert _metric(res, "speakers/minimal_pairs").value == 1.0  # s1,s2,s3 / 3

def test_category_coverage_metric(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    m = _metric(res, "coverage/categories")
    # golden picture words: Animals, Verbs, Food -> 3 of 27 categories
    assert m.value == 3 / 27
    assert set(m.detail["covered"]) == {"Animals", "Verbs", "Food"}
    assert "Body" in m.detail["missing"]

def test_unknown_category_warns(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["category"] = "Bogus"
    res = _run(b.build())
    f = next(f for f in res.findings if f.rule == "meth/unknown-category")
    assert f.severity == Severity.WARN
    assert f.note_id == "w-dog"

def test_no_personal_connection_is_info(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())  # golden has none filled
    hits = [f for f in res.findings if f.rule == "meth/no-personal-connection"]
    assert len(hits) == 3 and all(f.severity == Severity.INFO for f in hits)

def test_pair_coverage_by_note_attribution(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    m = _metric(res, "coverage/minimal_pairs")
    assert m.detail["by_note"] == {"mp-tone-1": "tone:low-rising",
                                   "mp-asp-1": "aspiration:velar"}

def test_contrast_id_for_public_api(tmp_path):
    from thai_deck_eval.core.context import EvalContext
    from thai_deck_eval.model.deck import load_deck
    from thai_deck_eval.stages.method import contrast_id_for
    deck = load_deck(DeckBuilder(tmp_path).build())
    ctx = EvalContext(deck=deck, g2p=G2P)
    assert contrast_id_for(deck.minimal_pairs[0], ctx) == "tone:low-rising"
    assert contrast_id_for(deck.minimal_pairs[0],
                           EvalContext(deck=deck, g2p=None)) is None


# The golden sentence is หมามากินข้าว -> [หมา, มา, กิน, ข้าว], target กิน.

def test_new_elements_measures_vocabulary_known_at_the_sentence_position(tmp_path):
    """A sentence must use what the learner has met by the time it appears,
    not everything the finished deck will eventually contain."""
    b = DeckBuilder(tmp_path)
    ranks = {"มา": 1, "หมา": 2, "ข้าว": 900}
    for w in b.data["picture_words"]:
        w["frequency_rank"] = ranks[w["thai"]]
    # position comes from the base: only the first two words are known there
    res = _run(b.build(), config={"sentence_base": 2})
    finding = next(f for f in res.findings if f.rule == "meth/new-elements")
    assert finding.evidence["unknown"] == ["ข้าว"]      # rank 900, not yet met
    assert finding.evidence["position"] == 2


def test_new_elements_allows_words_introduced_earlier(tmp_path):
    b = DeckBuilder(tmp_path)
    for w in b.data["picture_words"]:
        w["frequency_rank"] = {"มา": 1, "หมา": 2, "ข้าว": 3}[w["thai"]]
    res = _run(b.build(), config={"sentence_base": 3})
    assert not [f for f in res.findings if f.rule == "meth/new-elements"]


def test_frame_diversity_metric_counts_distinct_openings(tmp_path):
    """43% of the first corpus opened with one of two frames and nothing
    reported it: per-card rules cannot see a corpus-level defect."""
    b = DeckBuilder(tmp_path)
    sent = b.data["sentences"][0]
    b.data["sentences"] = [dict(sent, id=f"sn-{i}", thai="หมามากินข้าว")
                           for i in range(4)]
    res = _run(b.build())
    m = _metric(res, "diversity/sentence_frames")
    assert m.value == 0.25                      # one frame across four sentences
    assert m.detail["most_common"][0][1] == 4
