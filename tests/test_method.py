import thai_deck_eval.stages.method  # noqa: F401
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.model.deck import load_deck
from tests.fakes import FakeFreq, FakeG2P, FakeTokenizer
from tests.helpers import DeckBuilder
from tests.test_linguistic import FREQ, G2P, TOK

def _run(root, **ctx_kw):
    kw = dict(g2p=G2P, tokenizer=TOK, freq=FREQ)
    kw.update(ctx_kw)
    return run_pipeline(EvalContext(deck=load_deck(root), config={"sentence_base": 2},
                                    **kw), stages=[Stage.METHOD])

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

def test_speaker_diversity(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    assert _metric(res, "speakers/minimal_pairs").value == 1.0  # s1,s2,s3 / 3

def test_no_personal_connection_is_info(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())  # golden has none filled
    hits = [f for f in res.findings if f.rule == "meth/no-personal-connection"]
    assert len(hits) == 3 and all(f.severity == Severity.INFO for f in hits)
