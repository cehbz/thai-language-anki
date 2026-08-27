import pytest

from thai_deck_eval.lang.tone import Tone

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def g2p():
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPG2P
    return PyThaiNLPG2P()


@pytest.mark.parametrize("word,tone", [
    ("ข่าว", Tone.LOW), ("ข้าว", Tone.FALLING), ("ขาว", Tone.RISING),
    ("ไก่", Tone.LOW), ("มา", Tone.MID),
])
def test_g2p_tones(g2p, word, tone):
    syls = g2p.syllables(word)
    assert syls is not None and len(syls) == 1
    assert syls[0].tone == tone


def test_g2p_vowel_length(g2p):
    assert g2p.syllables("ขาว")[0].long is True


def test_g2p_unknown_returns_none_or_parses(g2p):
    g2p.syllables("ฟหกด")  # must not raise


def test_g2p_multisyllable_word(g2p):
    # "สามารถ" (able to) is two syllables: rising + falling.
    syls = g2p.syllables("สามารถ")
    assert syls is not None and len(syls) == 2
    assert syls[0].tone == Tone.RISING
    assert syls[1].tone == Tone.FALLING


def test_tltk_g2p_does_not_raise():
    # tltk is a native/optional dependency that may be broken or unavailable
    # in a given environment (e.g. missing its own transitive deps); the
    # port contract is "never raise", not "always succeed".
    from thai_deck_eval.lang.pythainlp_adapter import TltkG2P
    TltkG2P().syllables("มา")  # must not raise


def test_tokenizer():
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPTokenizer
    # NOTE: the brief's original example sentence "หมามากินข้าว" does not
    # exercise this assertion under pythainlp's real default tokenizer:
    # "กินข้าว" ("to eat/have a meal") is itself a dictionary entry
    # (confirmed via `"กินข้าว" in pythainlp.corpus.thai_words()` -> True),
    # so newmm's longest-match segmentation deterministically keeps it as
    # one token: word_tokenize("หมามากินข้าว") == ['หมา', 'มา', 'กินข้าว'].
    # This sentence avoids that compound and still exercises the same
    # capability (splitting a sentence into words including "กิน").
    toks = PyThaiNLPTokenizer().tokens("แมวกินปลา")
    assert "กิน" in toks
