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


def test_tokenizer_without_extra_words_keeps_dictionary_compound():
    # กินข้าว is itself a pythainlp dictionary entry; without deck vocabulary
    # seeded in, newmm's longest-match segmentation keeps it as one token
    # even though กิน is also independently a word (this is exactly the
    # false-positive lang/target-not-token case boundary-aligned matching
    # in the linguistic/method stages is meant to absorb).
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPTokenizer
    toks = PyThaiNLPTokenizer().tokens("หมามากินข้าว")
    assert "กินข้าว" in toks
    assert "กิน" not in toks


def test_tokenizer_extra_words_enables_novel_word_as_single_token():
    # extra_words' real payoff: deck vocabulary pythainlp's dictionary
    # doesn't already know (e.g. a target coined for this deck) segments as
    # one token instead of being split up.
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPTokenizer
    novel = "ปูมกวย"  # not in pythainlp.corpus.thai_words()
    without = PyThaiNLPTokenizer().tokens(f"หมา{novel}มา")
    assert novel not in without
    with_extra = PyThaiNLPTokenizer(extra_words={novel}).tokens(f"หมา{novel}มา")
    assert novel in with_extra


def test_tokenizer_caches_trie_on_the_instance():
    from pythainlp.util import Trie
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPTokenizer
    tok = PyThaiNLPTokenizer(extra_words={"ปูมกวย"})
    assert isinstance(tok._trie, Trie)
    trie_before = tok._trie
    tok.tokens("หมามากินข้าว")
    tok.tokens("แมวกินปลา")
    assert tok._trie is trie_before
