import thai_deck_eval.stages.linguistic  # noqa: F401
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.model.deck import load_deck
from tests.fakes import FakeFreq, FakeG2P, FakeTokenizer
from tests.helpers import DeckBuilder

G2P = FakeG2P({
    "ขาว": "kʰaːw˨˩˦", "ข่าว": "kʰaːw˨˩", "ข้าว": "kʰaːw˥˩",
    "ไก่": "kaj˨˩", "ไข่": "kʰaj˨˩", "หมา": "maː˨˩˦", "มา": "maː˧",
    "กิน": "kin˧", "ใหม่": "maj˨˩", "ไม้": "maj˦˥",
})
TOK = FakeTokenizer({"หมามากินข้าว": ["หมา", "มา", "กิน", "ข้าว"]})
FREQ = FakeFreq({"หมา": 120, "มา": 15, "ข้าว": 90})

def _run(root, g2p=G2P, second=None, tokenizer=TOK):
    ctx = EvalContext(deck=load_deck(root), g2p=g2p, g2p_second=second,
                      tokenizer=tokenizer, freq=FREQ)
    return run_pipeline(ctx, stages=[Stage.LINGUISTIC])

def _rules(res):
    rules = []
    for f in res.findings:
        if f.evidence and "rule_override" in f.evidence:
            rules.append(f.evidence["rule_override"])
        else:
            rules.append(f.rule)
    return sorted(rules)

def test_golden_clean(tmp_path):
    assert _rules(_run(DeckBuilder(tmp_path).build())) == []

def test_two_feature_pair_rejected(tmp_path):
    b = DeckBuilder(tmp_path)
    # ใหม่/ไม้ differ in tone AND (per fake) nothing else here — craft a real
    # two-feature case: ข้าว (long) vs a short-vowel fake entry
    b.data["minimal_pairs"][0]["members"][1] = {
        "thai": "ไม้", "ipa": "maj˦˥",
        "audio": {"file": "audio/khao-l.mp3", "source": "native", "speaker": "s2"}}
    res = _run(b.build())
    assert "lang/pair-not-minimal" in _rules(res)

def test_unknown_word_is_unverifiable(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["minimal_pairs"][0]["members"][0]["thai"] = "เรือ"
    res = _run(b.build())
    assert "lang/pair-unverifiable" in _rules(res)
    assert "lang/pair-not-minimal" not in _rules(res)

def test_ipa_mismatch(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["ipa"] = "maː˧"  # หมา is rising, not mid
    res = _run(b.build())
    assert "lang/ipa-mismatch" in _rules(res)

def test_ipa_mismatch_demoted_when_engines_disagree(tmp_path):
    from thai_deck_eval.core.findings import Severity
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["ipa"] = "maː˧"
    second = FakeG2P({"หมา": "maː˧"})  # second engine agrees with the author
    res = _run(b.build(), second=second)
    f = next(f for f in res.findings if f.rule == "lang/ipa-mismatch")
    assert f.severity == Severity.WARN

def test_ipa_mismatch_warn_when_second_is_none(tmp_path):
    # no second engine configured at all (ctx.g2p_second is None)
    from thai_deck_eval.core.findings import Severity
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["ipa"] = "maː˧"
    res = _run(b.build(), second=None)
    f = next(f for f in res.findings if f.rule == "lang/ipa-mismatch")
    assert f.severity == Severity.WARN

def test_ipa_mismatch_warn_when_second_returns_none(tmp_path):
    # second engine configured but doesn't know the word (returns None)
    from thai_deck_eval.core.findings import Severity
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["ipa"] = "maː˧"
    second = FakeG2P({})  # unknown to second engine -> syllables() is None
    res = _run(b.build(), second=second)
    f = next(f for f in res.findings if f.rule == "lang/ipa-mismatch")
    assert f.severity == Severity.WARN

def test_ipa_mismatch_error_when_second_corroborates(tmp_path):
    # second engine agrees with the primary engine, both disagree with author
    from thai_deck_eval.core.findings import Severity
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["ipa"] = "maː˧"
    second = FakeG2P({"หมา": "maː˨˩˦"})  # agrees with primary (rising)
    res = _run(b.build(), second=second)
    f = next(f for f in res.findings if f.rule == "lang/ipa-mismatch")
    assert f.severity == Severity.ERROR

def test_tone_mismatch_via_tone_engine(tmp_path):
    b = DeckBuilder(tmp_path)
    g2p = FakeG2P({**G2P.table, "หมา": "maː˧"})  # g2p wrong; tone engine says rising
    b.data["picture_words"][0]["ipa"] = "maː˧"
    res = _run(b.build(), g2p=g2p)
    assert "lang/tone-mismatch" in _rules(res)

def test_target_not_token(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["target"] = "มาก"  # substring of sentence, not a token
    b.data["sentences"][0]["thai"] = "หมามากินข้าว"
    res = _run(b.build())
    assert "lang/target-not-token" in _rules(res)

def test_target_not_token_boundary_aligned_compound(tmp_path):
    # target กิน is not itself a token, but it is the leading substring of
    # the dictionary-compound token กินข้าว -> boundary-aligned, no warning.
    b = DeckBuilder(tmp_path)
    tok = FakeTokenizer({"หมามากินข้าว": ["หมา", "มา", "กินข้าว"]})
    res = _run(b.build(), tokenizer=tok)
    assert "lang/target-not-token" not in _rules(res)

def test_target_not_token_mid_token_still_warns(tmp_path):
    # target นข is embedded strictly mid-token in กินข้าว (not a prefix or
    # suffix) -> still a warning.
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["target"] = "นข"
    tok = FakeTokenizer({"หมามากินข้าว": ["หมา", "มา", "กินข้าว"]})
    res = _run(b.build(), tokenizer=tok)
    assert "lang/target-not-token" in _rules(res)

def test_aspiration_triplet_same_place_passes(tmp_path):
    b = DeckBuilder(tmp_path)
    g2p = FakeG2P({**G2P.table, "บา": "baː˧", "ปา": "paː˧", "พา": "pʰaː˧"})
    b.data["minimal_pairs"].append({
        "id": "mp-asp-triplet", "contrast": "aspiration", "members": [
            {"thai": "บา", "ipa": "baː˧",
             "audio": {"file": "audio/baa.mp3", "source": "native", "speaker": "s1"}},
            {"thai": "ปา", "ipa": "paː˧",
             "audio": {"file": "audio/paa.mp3", "source": "native", "speaker": "s2"}},
            {"thai": "พา", "ipa": "pʰaː˧",
             "audio": {"file": "audio/phaa.mp3", "source": "native", "speaker": "s3"}},
        ]})
    res = _run(b.build(), g2p=g2p)
    assert "lang/pair-not-minimal" not in _rules(res)

def test_frequency_rank_wrong(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["frequency_rank"] = 4000
    res = _run(b.build())
    assert "lang/frequency-rank-wrong" in _rules(res)
