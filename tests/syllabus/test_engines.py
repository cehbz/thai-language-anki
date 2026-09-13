"""Tests for engines.py: the two pronunciation engines thai_syllabus owns
outright (the thaig2p output converter and the deterministic tone-rule
engine), ported out of thai_deck_eval.lang so that thai_syllabus imports
nothing from the legacy packages.

Three layers:
  * the pure converter/tone cases ported from tests/test_pythainlp_convert.py
    and tests/test_tone.py, restated in the `Syllable` shape;
  * a parity test that runs every one of those fixtures through both the new
    and the legacy engine and demands equality -- this is what makes the port
    behaviour-preserving rather than merely plausible (importing
    thai_deck_eval is fine *here*: the boundary rule covers src/thai_syllabus,
    not its tests);
  * an integration-marked parity sweep over the live deck's whole word list.

No pythainlp/torch import is triggered by the unit tests: engines.py keeps
those imports inside Thaig2p.syllables, so the pure functions are importable
and callable without the "nlp" extra installed.
"""
from pathlib import Path

import pytest

from thai_syllabus.engines import (Thaig2p, _convert, rule_tone, tone_of)
from thai_syllabus.entities import Syllable


def S(onset, vowel, coda, length, tone):
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


# --- the converter: thaig2p-shaped raw strings ---------------------------
# (token-per-phone, Chao tone letters; see engines.py's module docstring)

def test_convert_simple_syllable():
    assert _convert("m aː ˧") == (S("m", "a", "", "long", "mid"),)


def test_convert_short_vowel_with_coda():
    assert _convert("k a j ˨˩") == (S("k", "a", "j", "short", "low"),)


def test_convert_rising_long_vowel_coda():
    assert _convert("kʰ aː w ˩˩˦") == (S("kʰ", "a", "w", "long", "rising"),)


def test_convert_strips_no_release_diacritic_on_coda():
    # สามารถ (able to), second syllable: the stop coda carries a
    # no-audible-release mark that has no field on Syllable and is stripped.
    syl = _convert("m aː t̚ ˥˩")[0]
    assert syl.coda == "t" and syl.tone == "falling"


def test_convert_multisyllable_word():
    syls = _convert("s aː ˩˩˦ . m aː t̚ ˥˩")
    assert len(syls) == 2
    assert syls[0].tone == "rising"
    assert syls[1].tone == "falling"


def test_convert_merges_diphthong_glide():
    # เมีย (wife) -> "ia" vowel: head "i" + non-syllabic "a" (combining
    # inverted breve below) merge into a single "ia" phone.
    syl = _convert("m i a̯ ˧")[0]
    assert syl.vowel == "ia" and syl.coda == ""


def test_convert_strips_affricate_tie_bar():
    # ช้าง (elephant) -> aspirated affricate onset, tie bar stripped to match
    # the deck's "tɕʰ" spelling.
    syl = _convert("t͡ɕʰ aː ŋ ˦˥")[0]
    assert syl.onset == "tɕʰ" and syl.coda == "ŋ" and syl.tone == "high"


def test_convert_glottal_onset_and_coda():
    # จะ (will) -> unaspirated affricate onset, explicit glottal-stop coda
    syl = _convert("t͡ɕ a ʔ ˨˩")[0]
    assert syl.onset == "tɕ" and syl.coda == "ʔ" and syl.tone == "low"


def test_convert_vowel_initial_syllable():
    # อา (paternal aunt/uncle) -> explicit glottal onset
    syl = _convert("ʔ aː ˧")[0]
    assert syl.onset == "ʔ" and syl.vowel_length == "long"


def test_convert_unparseable_cluster_returns_none():
    # "uə" isn't a recognised vowel token (thaig2p's real offglide shape is
    # "u a̯", merged elsewhere) -- this stays unparseable even with cluster
    # onsets supported, since the phone after "l" isn't vowel-shaped.
    assert _convert("k l uə ˧") is None


def test_convert_empty_returns_none():
    assert _convert("") is None


# --- initial consonant clusters (stop/fricative + r/l/w) -----------------
# Thai clusters: kr kl kw, kʰr kʰl kʰw, pr pl, pʰr pʰl, tr, and loanword
# br bl dr fr fl. The second consonant arrives as its own phone token;
# it's only a cluster when a vowel follows it (otherwise a w/j coda or the
# next syllable's initial would be swallowed).

def test_convert_cluster_onset_pl():
    # ปลา (fish)
    assert _convert("p l aː ˧") == (S("pl", "a", "", "long", "mid"),)


def test_convert_cluster_onset_kl_with_nasal_coda():
    # กลอง (drum)
    assert _convert("k l ɔː ŋ ˧") == (S("kl", "ɔ", "ŋ", "long", "mid"),)


def test_convert_cluster_onset_aspirated_kw_with_glide_coda():
    # ควาย (water buffalo)
    assert _convert("kʰ w aː j ˧") == (S("kʰw", "a", "j", "long", "mid"),)


def test_convert_w_coda_survives_before_a_following_syllable():
    # ขาว-shaped first syllable (w coda) followed by a second syllable: the
    # w coda must stay a coda -- it isn't immediately after the onset, so
    # it must not be mistaken for a cluster semivowel.
    syls = _convert("kʰ aː w ˩˩˦ . m aː ˧")
    assert syls == (
        S("kʰ", "a", "w", "long", "rising"),
        S("m", "a", "", "long", "mid"),
    )


def test_convert_bare_cluster_consonants_without_vowel_returns_none():
    # A stop + l with nothing after it: no vowel follows "l", so it must
    # not be merged into a phantom onset -- still declines to convert.
    assert _convert("p l ˧") is None


# --- the tone-rule engine ------------------------------------------------

@pytest.mark.parametrize("cls,live,long_v,mark,expected", [
    ("mid", True, True, None, "mid"),          # กา (crow)
    ("high", True, True, None, "rising"),      # ขา (leg)
    ("low", True, True, None, "mid"),          # คา (to be stuck)
    ("mid", False, True, None, "low"),         # บาท (baht)
    ("high", False, False, None, "low"),       # ขับ (to drive)
    ("low", False, False, None, "high"),       # คับ (tight)
    ("low", False, True, None, "falling"),     # มาก (much)
    ("mid", True, True, "่", "low"),
    ("high", True, True, "่", "low"),
    ("low", True, True, "่", "falling"),
    ("mid", True, True, "้", "falling"),
    ("high", True, True, "้", "falling"),
    ("low", True, True, "้", "high"),
    ("mid", True, True, "๊", "high"),
    ("mid", True, True, "๋", "rising"),
])
def test_tone_table(cls, live, long_v, mark, expected):
    assert tone_of(cls, live, long_v, mark) == expected


@pytest.mark.parametrize("word,tone", [
    ("มา", "mid"),        # มา: to come
    ("หมา", "rising"),    # หมา: dog -- ห นำ (leading silent ห)
    ("ไม่", "falling"),   # ไม่: not
    ("ไม้", "high"),      # ไม้: wood
    ("ใหม่", "low"),      # ใหม่: new
    ("ไหม", "rising"),    # ไหม: silk / question particle
    ("ขาว", "rising"),    # ขาว: white
    ("ข่าว", "low"),      # ข่าว: news
    ("ข้าว", "falling"),  # ข้าว: rice
    ("ไก่", "low"),       # ไก่: chicken
    ("ไข่", "low"),       # ไข่: egg
    ("มาก", "falling"),   # มาก: much
    ("อยู่", "low"),      # อยู่: to be (located) -- อ นำ
    ("กิน", "mid"),       # กิน: to eat
])
def test_rule_tone_of_known_words(word, tone):
    assert rule_tone(word) == tone


@pytest.mark.parametrize("word", ["โรงเรียน", "ธนา", "คนา", "ชนา"])
def test_rule_tone_returns_none_when_unparseable(word):
    # โรงเรียน (school): multi-syllable. ธนา/คนา/ชนา: CCV two-syllable words
    # whose middle consonant is the *second* syllable's initial, not a final --
    # must not be mis-analyzed as a single syllable with a fabricated tone.
    assert rule_tone(word) is None


# --- parity with the legacy engines --------------------------------------
# Every fixture in tests/test_pythainlp_convert.py (thaig2p shape) and every
# word in tests/test_tone.py, run through both implementations.

RAW_FIXTURES = [
    "m aː ˧",
    "k a j ˨˩",
    "kʰ aː w ˩˩˦",
    "m aː t̚ ˥˩",
    "s aː ˩˩˦ . m aː t̚ ˥˩",
    "m i a̯ ˧",
    "t͡ɕʰ aː ŋ ˦˥",
    "t͡ɕ a ʔ ˨˩",
    "ʔ aː ˧",
    "k l uə ˧",
    "",
    # a merged diphthong followed by a real coda -- the glide-merge and the
    # coda slot exercised together, which none of the legacy fixtures do
    "m i a̯ ŋ ˧",    # เมี่ยง-shaped: "ia" + ŋ
    "h ɯ a̯ j ˥˩",   # ห้วย-shaped: "ɯa" + j
]

TONE_WORDS = [
    "มา", "หมา", "ไม่", "ไม้", "ใหม่", "ไหม", "ขาว", "ข่าว", "ข้าว",
    "ไก่", "ไข่", "มาก", "อยู่", "กิน",
    "โรงเรียน", "ธนา", "คนา", "ชนา",
    "โน้ต",    # โน้ต: (musical) note -- tone-marked dead syllable, the mark
               # branch of tone_of taking precedence over live/dead
    "เกลือ",   # เกลือ: salt -- cluster onset + complex เ-ือ vowel, an
               # out-of-scope form both engines must decline the same way
]


def _legacy_shape(syls):
    """The legacy IpaSyllable list mapped into the Syllable shape -- exactly
    the mapping phonology.default_engines() used to do inline."""
    if syls is None:
        return None
    return tuple(Syllable(segments=(s.onset, s.vowel, s.coda or ""),
                          vowel_length="long" if s.long else "short",
                          tone=str(s.tone.value)) for s in syls)


@pytest.mark.parametrize("raw", RAW_FIXTURES)
def test_converter_matches_the_legacy_converter(raw):
    from thai_deck_eval.lang.pythainlp_adapter import _convert as legacy_convert
    assert _convert(raw) == _legacy_shape(legacy_convert(raw))


# The legacy converter has no cluster-onset support: these fixtures are
# exactly where the port's behaviour diverges from the legacy one on
# purpose (spec: the cluster-onset gap fix), so they get their own test
# instead of being folded into the parity fixture list above.
CLUSTER_FIXTURES = [
    "p l aː ˧",     # ปลา (fish)
    "k l ɔː ŋ ˧",   # กลอง (drum)
    "kʰ w aː j ˧",  # ควาย (water buffalo)
]


@pytest.mark.parametrize("raw", CLUSTER_FIXTURES)
def test_cluster_onsets_convert_where_the_legacy_declines(raw):
    from thai_deck_eval.lang.pythainlp_adapter import _convert as legacy_convert
    assert legacy_convert(raw) is None
    assert _convert(raw) is not None


@pytest.mark.parametrize("word", TONE_WORDS)
def test_tone_engine_matches_the_legacy_tone_engine(word):
    from thai_deck_eval.lang.tone import analyze_syllable
    legacy = analyze_syllable(word)
    assert rule_tone(word) == (str(legacy.tone.value) if legacy is not None else None)


WORDS_YAML = Path.home() / "decks" / "thai-ff" / "curated" / "words.yaml"


@pytest.mark.integration
def test_g2p_matches_the_legacy_g2p_over_the_live_deck():
    """Behaviour preservation plus the cluster-onset fix, over every Thai
    form in the live deck: where the legacy converter returns a result, the
    ported converter must equal it (the fix must not touch anything the
    legacy engine already handles). Where the legacy declines (None), the
    ported converter is now allowed -- and expected -- to convert more,
    since it supports two-phone initial clusters (pl/kl/kʰw/...) the legacy
    engine never did; ปลา (fish), กลอง (drum), ควาย (water buffalo) must be
    among the newly-converted words. Reads the deck read-only; skips when it
    -- or pythainlp/torch -- isn't there.
    """
    import yaml
    if not WORDS_YAML.exists():  # pragma: no cover - deck absent in CI
        pytest.skip(f"no deck word list at {WORDS_YAML}")
    words = [w["thai"] for w in yaml.safe_load(WORDS_YAML.read_text()) or []
             if w.get("thai")]
    assert words, "the deck word list should not be empty"
    try:
        from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPG2P
        legacy = PyThaiNLPG2P()
        ported = Thaig2p()
        ported.syllables(words[0])
    except Exception as e:  # pragma: no cover - model/deps absent in CI
        pytest.skip(f"thaig2p unavailable: {e}")

    results = [(w, ported.syllables(w), _legacy_shape(legacy.syllables(w)))
               for w in words]

    both = [(w, p, l) for w, p, l in results if l is not None]
    mismatches = [(w, p, l) for w, p, l in both if p != l]
    assert not mismatches, mismatches[:5]

    legacy_none = [(w, p) for w, p, l in results if l is None]
    newly_converted = [w for w, p in legacy_none if p is not None]
    still_none = [w for w, p in legacy_none if p is None]

    assert "ปลา" in newly_converted   # fish
    assert "กลอง" in newly_converted  # drum
    assert "ควาย" in newly_converted  # water buffalo

    print(f"\nconverted by both (legacy result matched): {len(both)}")
    print(f"newly converted by the ported engine (legacy None): "
          f"{len(newly_converted)}")
    print(f"still None on both engines: {len(still_none)}")
    print(f"still-None Thai forms ({len(still_none)}): {still_none}")
