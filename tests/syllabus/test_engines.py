"""Tests for engines.py: the two pronunciation engines thai_syllabus owns
outright (the thaig2p output converter and the deterministic tone-rule
engine), ported out of thai_deck_eval.lang so that thai_syllabus imports
nothing from the legacy packages.

Three layers:
  * the pure converter/tone cases ported from tests/test_pythainlp_convert.py
    and tests/test_tone.py, restated in the `Syllable` shape;
  * a parity test that runs every one of those fixtures through both the new
    engine and a *frozen recording* of the legacy engine's output, and
    demands equality -- this is what makes the port behaviour-preserving
    rather than merely plausible. The legacy package (thai_deck_eval) was
    deleted 2026-09-18 once the migration off it was complete; its outputs
    over every fixture these tests use were captured beforehand into
    tests/syllabus/fixtures/legacy_convert_parity.json (see that file's
    header for how/when), so the parity check survives the package's
    removal;
  * an integration-marked parity sweep over the live deck's whole word list,
    likewise checked against the frozen recording rather than a live import.

No pythainlp/torch import is triggered by the unit tests: engines.py keeps
those imports inside Thaig2p.syllables, so the pure functions are importable
and callable without the "nlp" extra installed.
"""
import json
from pathlib import Path

import pytest

from thai_syllabus.engines import (Thaig2p, Tltk, _convert, _convert_tltk,
                                    convert_wiktionary, extract_standard_ipa,
                                    rule_tone, tone_of)
from thai_syllabus.entities import Syllable, without_glottal_coda

_LEGACY_PARITY_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_convert_parity.json").read_text())


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
    # inverted breve below) merge into a single "ia" phone. A centering
    # diphthong is long (design 2026-09-20 §1): thaig2p writes no ː on it.
    syl = _convert("m i a̯ ˧")[0]
    assert syl.vowel == "ia" and syl.coda == ""
    assert syl.vowel_length == "long"


def test_every_converter_marks_a_diphthong_long():
    for raw in ("k l u a̯ j ˥˩", "h ɯ a̯ j ˥˩", "s ɯ a̯ ˩˩˦"):
        assert _convert(raw)[0].vowel_length == "long", raw
    for raw in ("kluːaj3", "hɯːaj3", "sɯːa5"):
        assert _convert_tltk(raw)[0].vowel_length == "long", raw


def test_convert_strips_affricate_tie_bar():
    # ช้าง (elephant) -> aspirated affricate onset, tie bar stripped to match
    # the deck's "tɕʰ" spelling.
    syl = _convert("t͡ɕʰ aː ŋ ˦˥")[0]
    assert syl.onset == "tɕʰ" and syl.coda == "ŋ" and syl.tone == "high"


def test_convert_glottal_coda_is_normalized_away():
    # จะ (will) -> unaspirated affricate onset; thaig2p's explicit
    # glottal-stop coda is normalized away (design 2026-09-18 §5,
    # without_glottal_coda) so a dead open syllable matches tltk, which
    # never writes one.
    assert _convert("t͡ɕ a ʔ ˨˩")[0] == S("tɕ", "a", "", "short", "low")


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


# --- new vowel-form coverage (2026-09-18 second-engine coverage arc) ------
# Six forms rule_tone previously refused (all currently None on main):
# -อ (ɔː), เ-อ (ɤː), -ือ (ɯː), -ำ (am), เ-า (aw), and the implicit /o/ vowel
# of a bare CC monosyllable. Each word below is checked against its known,
# uncontested tone (deck-independent -- these are common words, not probes
# tied to the deck fixture).

@pytest.mark.parametrize("word,tone", [
    ("คอ", "mid"),       # คอ: neck -- mid class, live (-อ long), no mark
    ("ขอ", "rising"),    # ขอ: to ask for -- high class, live
    ("พ่อ", "falling"),  # พ่อ: father -- low class, mai ek -> falling
    ("จอ", "mid"),       # จอ: screen -- mid class, live
])
def test_rule_tone_of_final_o_vowel(word, tone):
    assert rule_tone(word) == tone


@pytest.mark.parametrize("word,tone", [
    ("เธอ", "mid"),   # เธอ: you/she -- low class, live (เ-อ), no mark
    ("เจอ", "mid"),   # เจอ: to meet/find -- mid class, live
])
def test_rule_tone_of_pre_vowel_oe(word, tone):
    assert rule_tone(word) == tone


@pytest.mark.parametrize("word,tone", [
    ("มือ", "mid"),      # มือ: hand -- low class, live (-ือ long)
    ("ถือ", "rising"),   # ถือ: to hold -- high class, live
])
def test_rule_tone_of_ue_vowel(word, tone):
    assert rule_tone(word) == tone


@pytest.mark.parametrize("word,tone", [
    ("น้ำ", "high"),  # น้ำ: water -- low class, mai tho -> high
    ("ดำ", "mid"),    # ดำ: black -- mid class, live (-ำ is live)
    ("ทำ", "mid"),    # ทำ: to do -- low class, live
])
def test_rule_tone_of_am_vowel(word, tone):
    assert rule_tone(word) == tone


@pytest.mark.parametrize("word,tone", [
    ("เขา", "rising"),  # เขา: he/she/they -- high class, live (เ-า diphthong)
    ("เรา", "mid"),     # เรา: we -- low class, live
    ("เมา", "mid"),     # เมา: drunk -- low class, live
])
def test_rule_tone_of_pre_vowel_aw(word, tone):
    assert rule_tone(word) == tone


@pytest.mark.parametrize("word,tone", [
    ("คน", "mid"),       # คน: person -- low class, live final (implicit /o/)
    ("ลม", "mid"),       # ลม: wind -- low class, live final
    ("นก", "high"),      # นก: bird -- low class, dead final (stop)
    ("กบ", "low"),       # กบ: frog -- mid class, dead final
    ("จบ", "low"),       # จบ: to finish -- mid class, dead final
    ("ผม", "rising"),    # ผม: I/hair -- high class, live final
])
def test_rule_tone_of_implicit_vowel(word, tone):
    assert rule_tone(word) == tone


@pytest.mark.parametrize("word", [
    "เกาะ",   # เ-าะ: short o+glottal, out of scope -- must not be confused
              # with the เ-า (aw) diphthong now handled
    "เกลือ",  # เ-ือ: complex diphthong, out of scope
    "เสีย",   # เ-ีย: complex diphthong, out of scope
    "กลัว",   # -ัว vowel behind a cluster onset: out of scope (must not
              # gain coverage as a side effect of any of the above)
])
def test_rule_tone_still_declines_out_of_scope_complex_vowels(word):
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


def _frozen_shape(recorded):
    """A frozen fixture record (list of {segments, vowel_length, tone} dicts,
    or None) into the same Syllable-tuple shape _legacy_shape produces."""
    if recorded is None:
        return None
    return tuple(Syllable(segments=tuple(r["segments"]),
                          vowel_length=r["vowel_length"],
                          tone=r["tone"]) for r in recorded)


@pytest.mark.parametrize("raw", RAW_FIXTURES)
def test_converter_matches_the_legacy_converter(raw):
    legacy = _frozen_shape(_LEGACY_PARITY_FIXTURE["raw_fixtures"][raw])
    assert _convert(raw) == legacy


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
    assert _LEGACY_PARITY_FIXTURE["cluster_fixtures"][raw] is None
    assert _convert(raw) is not None


# The legacy converter still writes a ʔ coda on a dead open syllable
# (จะ); the ported converter normalizes it away (design 2026-09-18 §5,
# without_glottal_coda) so thaig2p's reading agrees with tltk's, which
# never writes one. Also a deliberate divergence, not a regression, so it
# gets its own test rather than the parity fixture list above.
GLOTTAL_CODA_FIXTURES = [
    "t͡ɕ a ʔ ˨˩",   # จะ (will)
]


@pytest.mark.parametrize("raw", GLOTTAL_CODA_FIXTURES)
def test_glottal_coda_diverges_from_the_legacy_converter_on_purpose(raw):
    """Full parity, modulo exactly the one stated divergence: the ported
    converter must equal the legacy converter's reading with the glottal
    coda normalized away, not merely agree that a coda was dropped -- a
    regression in จะ's vowel, length or tone must still be caught here."""
    legacy = _frozen_shape(_LEGACY_PARITY_FIXTURE["glottal_coda_fixtures"][raw])
    assert legacy[0].coda == "ʔ"
    assert _convert(raw) == without_glottal_coda(legacy)


@pytest.mark.parametrize("word", TONE_WORDS)
def test_tone_engine_matches_the_legacy_tone_engine(word):
    assert rule_tone(word) == _LEGACY_PARITY_FIXTURE["tone_words"][word]


@pytest.mark.integration
def test_g2p_matches_the_legacy_g2p_over_the_live_deck():
    """Behaviour preservation plus the cluster-onset fix, over every Thai
    form in the deck as it stood when tests/syllabus/fixtures/
    legacy_convert_parity.json was recorded: where the legacy converter
    returned a result, the ported converter must equal it (the fix must not
    touch anything the legacy engine already handled). Where the legacy
    declined (None), the ported converter is allowed -- and expected -- to
    convert more, since it supports two-phone initial clusters (pl/kl/kʰw/
    ...) the legacy engine never did; ปลา (fish), กลอง (drum), ควาย (water
    buffalo) must be among the newly-converted words. Compares against the
    frozen recording rather than a live legacy import (thai_deck_eval no
    longer exists); skips when the recording has none (fixture captured
    without deck access) or thaig2p itself is unavailable.
    """
    deck_sweep = _LEGACY_PARITY_FIXTURE["deck_sweep"]
    if deck_sweep is None:  # pragma: no cover - fixture captured without deck access
        pytest.skip("frozen fixture has no deck_sweep recording")
    words = list(deck_sweep.keys())
    assert words, "the frozen deck word list should not be empty"
    try:
        ported = Thaig2p()
        ported.syllables(words[0])
    except Exception as e:  # pragma: no cover - model/deps absent in CI
        pytest.skip(f"thaig2p unavailable: {e}")

    results = [(w, ported.syllables(w), _frozen_shape(deck_sweep[w]))
               for w in words]

    both = [(w, p, l) for w, p, l in results if l is not None]
    mismatches = [(w, p, l) for w, p, l in both if p != l]
    # The glottal-coda convention (design 2026-09-18 §5): the legacy
    # converter still writes a ʔ coda on a dead open syllable; the ported
    # converter normalizes it away (without_glottal_coda) so thaig2p
    # agrees with tltk. A word whose only difference from the legacy
    # reading is that normalization is a deliberate divergence, not a
    # regression -- the live deck has exactly 19 such syllables.
    glottal_coda_only = [(w, p, l) for w, p, l in mismatches
                         if p == without_glottal_coda(l)]
    real_mismatches = [(w, p, l) for w, p, l in mismatches
                       if p != without_glottal_coda(l)]
    assert not real_mismatches, real_mismatches[:5]
    # Pins the count so this bucket can't silently drift or collapse to
    # zero (e.g. if without_glottal_coda were dropped from _convert, every
    # one of these 19 would become a real mismatch instead and the
    # assertion above would catch it -- but only if this one also fails
    # when the bucket empties out from underneath it).
    assert len(glottal_coda_only) == 19, (
        f"this counts live-deck forms where thaig2p writes a ʔ coda "
        f"without_glottal_coda then strips; expected 19, got "
        f"{len(glottal_coda_only)}: {[w for w, _, _ in glottal_coda_only]}. "
        f"A CHANGE in this number is expected as the deck grows (this arc's "
        f"adoption pass adds words) -- update the expected count. A DROP TO "
        f"ZERO is not growth: it means without_glottal_coda's normalization "
        f"has been removed from _convert, and every one of these 19 would "
        f"become a real mismatch -- a real bug, not a number to update away.")

    legacy_none = [(w, p) for w, p, l in results if l is None]
    newly_converted = [w for w, p in legacy_none if p is not None]
    still_none = [w for w, p in legacy_none if p is None]

    assert "ปลา" in newly_converted   # fish
    assert "กลอง" in newly_converted  # drum
    assert "ควาย" in newly_converted  # water buffalo

    print(f"\nconverted by both (legacy result matched): {len(both)}")
    print(f"glottal-coda-only divergences (expected): {len(glottal_coda_only)}")
    print(f"newly converted by the ported engine (legacy None): "
          f"{len(newly_converted)}")
    print(f"still None on both engines: {len(still_none)}")
    print(f"still-None Thai forms ({len(still_none)}): {still_none}")


# --- tltk's raw output -> Syllable (design 2026-09-18 §4) ------------------
# Observed from tltk.nlp.th2ipa as installed 2026-09-18. Digit tones were
# verified against nine words whose tone is not in doubt (มา ไก่ ข้าว ช้าง
# ขาว จะ นา น้ำ หมา).

def test_a_monosyllable_with_its_digit_tone():
    assert _convert_tltk("maː1 <s/>") == (
        Syllable(segments=("m", "a", ""), vowel_length="long", tone="mid"),)


def test_every_digit_maps_to_its_tone():
    got = {}
    for raw, thai in (("maː1", "มา"), ("kaj2", "ไก่"), ("kʰaːw3", "ข้าว"),
                      ("cʰaːŋ4", "ช้าง"), ("kʰaːw5", "ขาว")):
        got[thai] = _convert_tltk(raw)[0].tone
    assert got == {"มา": "mid", "ไก่": "low", "ข้าว": "falling",
                   "ช้าง": "high", "ขาว": "rising"}


def test_syllables_split_on_dot_space_tilde_and_bar():
    for raw in ("pʰaːp3.waːt3", "pʰaːp3 waːt3", "pʰaːp3~waːt3", "pʰaːp3|waːt3"):
        assert _convert_tltk(raw) == (
            Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),
            Syllable(segments=("w", "a", "t"), vowel_length="long", tone="falling"))


def test_tltks_open_o_and_affricates_become_the_decks_spelling():
    """tltk writes ᴐ (U+1D10) for the deck's ɔ, and c/cʰ for tɕ/tɕʰ."""
    assert _convert_tltk("ʔᴐːk2")[0].segments == ("ʔ", "ɔ", "k")
    assert _convert_tltk("cʰaː4")[0].segments == ("tɕʰ", "a", "")
    assert _convert_tltk("ca2")[0].segments == ("tɕ", "a", "")


def test_a_diphthong_is_written_with_the_length_mark_inside():
    """เมีย is 'miːa1': the deck spells that vowel 'ia', long (design
    2026-09-20 §1) -- tltk's own length mark now stands."""
    assert _convert_tltk("miːa1") == (
        Syllable(segments=("m", "ia", ""), vowel_length="long", tone="mid"),)
    assert _convert_tltk("sɯːa5")[0].segments == ("s", "ɯa", "")
    assert _convert_tltk("wuːa1")[0].segments == ("w", "ua", "")


def test_a_consonant_cluster_is_one_onset():
    """Without this merge 87 of the live deck's 890 words fail to convert:
    the onset parse takes 'p' and then reads 'l' as the vowel."""
    assert _convert_tltk("plaː1")[0].segments == ("pl", "a", "")
    assert _convert_tltk("kwaːj1")[0].segments == ("kw", "a", "j")
    assert _convert_tltk("kra2")[0].segments == ("kr", "a", "")


def test_a_glide_coda_is_a_coda_not_a_cluster():
    """The merge must not fire when no vowel follows the r/l/w."""
    assert _convert_tltk("kʰaːw5")[0].segments == ("kʰ", "a", "w")


def test_unmappable_input_is_no_reading_rather_than_a_raise():
    assert _convert_tltk("") is None
    assert _convert_tltk("<s/>") is None
    assert _convert_tltk("maː") is None          # no tone digit
    assert _convert_tltk("zzz9") is None         # unknown everything


# --- an empty syllable slot is a truncated reading, refused (FIX 1) -------
# tltk marks a part it could not read as an empty syllable slot -- a "."
# with nothing on one side of it where a real syllable belongs. Verified on
# the real engine 2026-09-18: exactly 2 of the live deck's 875 distinct
# forms produce one, both genuine truncations, so this detector is exact on
# real data.

def test_a_trailing_dot_with_nothing_after_it_is_a_truncated_reading():
    # สิบเอ็ด (sìp.ʔèt, 2 syllables): tltk emits only "si2" and marks the
    # lost second syllable with a trailing dot -- refused, not returned as
    # one syllable.
    assert _convert_tltk("si2. <s/>") is None


def test_a_double_dot_mid_string_is_an_empty_slot_between_two_readings():
    # ชานมไข่มุก (5 syllables): tltk reads only the first and last, marking
    # the three lost middle syllables with a double dot. The naive
    # `_TLTK_SYLLABLE_SEP` regex collapses ".." into one separator and would
    # otherwise silently turn this into a 2-syllable reading.
    assert _convert_tltk("cʰaːn1..muk4 <s/>") is None


def test_a_sound_reading_with_no_dot_at_all_is_unaffected():
    # ไก่: one syllable, no separator of any kind -- must not be mistaken
    # for a truncation.
    assert _convert_tltk("kaj2 <s/>") == (
        Syllable(segments=("k", "a", "j"), vowel_length="short", tone="low"),)


def test_a_sound_multisyllable_reading_with_one_ordinary_dot_is_unaffected():
    # ภาพวาด: two real syllables joined by exactly one dot -- the ordinary
    # separator case, not a slot.
    assert _convert_tltk("pʰaːp3.waːt3 <s/>") == (
        Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),
        Syllable(segments=("w", "a", "t"), vowel_length="long", tone="falling"))


def test_ordinary_whitespace_between_two_readings_is_not_mistaken_for_a_slot():
    # ข้างๆ: space-separated, no dot anywhere -- this is the negative case
    # that would break if plain whitespace were treated as a slot marker.
    assert _convert_tltk("kʰaːŋ3 kʰaːŋ3 <s/>") == (
        Syllable(segments=("kʰ", "a", "ŋ"), vowel_length="long", tone="falling"),
        Syllable(segments=("kʰ", "a", "ŋ"), vowel_length="long", tone="falling"))


# --- the Tltk engine callable ---------------------------------------------

@pytest.mark.integration
def test_tltk_reads_the_compounds_thaig2p_loops_on():
    """The whole point of the second engine: ภาพวาด is four consonant
    letters and thaig2p reads it as eleven syllables."""
    tltk = Tltk()
    assert tltk("ภาพวาด") == (
        Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),
        Syllable(segments=("w", "a", "t"), vowel_length="long", tone="falling"))
    assert len(tltk("โรงพยาบาล")) == 4
    assert len(tltk("ออกกำลังกาย")) == 4


@pytest.mark.integration
def test_tltk_never_raises_on_a_form_it_cannot_read():
    """tltk.nlp.th2ipa raises ValueError on some two-token phrases (ธอ ธง
    and ฝอ ฝา on the live deck); the engine returns None instead of letting
    that exception escape. A version of Tltk that didn't catch it would
    raise here and fail this test."""
    tltk = Tltk()
    for word in ("ธอ ธง", "ฝอ ฝา"):
        result = tltk(word)
        assert result is None or (
            isinstance(result, tuple)
            and all(isinstance(s, Syllable) for s in result))
    assert tltk("") is None


def test_thaig2p_and_tltk_agree_once_the_glottal_coda_is_normalized():
    """thaig2p emits the coda, tltk does not; both must land on the same
    Syllable or no dead open syllable could ever be corroborated."""
    assert _convert("t͡ɕ a ʔ ˨˩") == _convert_tltk("ca2")


# --- Wiktionary's rendered IPA -> Syllable (design 2026-09-20 §2) ----------
# Observed 2026-09-20 on en.wiktionary.org (Template:th-pron output).

def test_wiktionary_chao_letters_and_length():
    assert convert_wiktionary("/ma˦˥.ka˨˩.raː˧/") == (
        Syllable(segments=("m", "a", ""), vowel_length="short", tone="high"),
        Syllable(segments=("k", "a", ""), vowel_length="short", tone="low"),
        Syllable(segments=("r", "a", ""), vowel_length="long", tone="mid"))


def test_wiktionary_rising_falling_and_unreleased_stops():
    assert convert_wiktionary("/sip̚˨˩.ʔet̚˨˩/") == (
        Syllable(segments=("s", "i", "p"), vowel_length="short", tone="low"),
        Syllable(segments=("ʔ", "e", "t"), vowel_length="short", tone="low"))
    assert convert_wiktionary("/huŋ˩˩˦/")[0].tone == "rising"
    assert convert_wiktionary("/t͡ɕaːw˥˩/") == (
        Syllable(segments=("tɕ", "a", "w"), vowel_length="long", tone="falling"),)


def test_wiktionary_diphthong_is_long_and_the_glide_mark_is_dropped():
    assert convert_wiktionary("/klua̯j˥˩/") == (
        Syllable(segments=("kl", "ua", "j"), vowel_length="long", tone="falling"),)
    assert convert_wiktionary("/sɯa̯˩˩˦/")[0].segments == ("s", "ɯa", "")


def test_wiktionary_glottal_coda_is_normalized_away():
    assert convert_wiktionary("/t͡ɕaʔ˨˩/") == (
        Syllable(segments=("tɕ", "a", ""), vowel_length="short", tone="low"),)


def test_wiktionary_affricate_aspirated_and_cluster():
    assert convert_wiktionary("/t͡ɕʰaːŋ˦˥/")[0].segments == ("tɕʰ", "a", "ŋ")
    assert convert_wiktionary("/pʰon˩˩˦.la˦˥.maːj˦˥/")[1].segments == ("l", "a", "")


def test_wiktionary_unmappable_is_no_reading():
    assert convert_wiktionary("") is None
    assert convert_wiktionary("/ma/") is None          # no tone letters
    assert convert_wiktionary("/zz˧/") is None         # unknown onset


_ROW = ('<tr><th colspan="2">(<i><a href="x">standard</a></i>) <a href="y">IPA</a>'
        '<sup>(<a href="z">key</a>)</sup></th>'
        '<td><span class="IPA">/ma˦˥.ka˨˩.raː˧/</span><sup>(R)</sup></td>'
        '<td><span class="IPA">/mok̚˦˥.ka˨˩.raː˧/</span></td></tr>')


def test_extract_standard_ipa_reads_every_reading_in_the_standard_row():
    html = '<table><tr><th>Royal Institute</th><td><span class="tr">ma-ka-ra</span></td></tr>' + _ROW + '</table>'
    assert extract_standard_ipa(html) == ["ma˦˥.ka˨˩.raː˧", "mok̚˦˥.ka˨˩.raː˧"]


def test_extract_standard_ipa_is_empty_without_the_row():
    assert extract_standard_ipa('<table><tr><th>Paiboon</th><td>má-gà-raa</td></tr></table>') == []
    assert extract_standard_ipa("") == []


def test_wiktionary_an_empty_syllable_group_is_no_reading():
    """Wiktionary marks a compound-linking variant with a TRAILING "."
    (observed live 2026-09-21: น้ำ lists /naːm˦˥/, /naːm˦˥./ and
    /nam˦˥./; นรก lists /na˦˥.rok̚˦˥./). The dot means "this form links
    onward", not "a syllable follows", so the reading is incomplete as
    written -- the same refusal `_tltk_has_empty_slot` makes for tltk's
    own empty slot. Dropping the empty group instead would record a
    short reading as if it were the whole word."""
    assert convert_wiktionary("/nam˦˥./") is None
    assert convert_wiktionary("/na˦˥.rok̚˦˥.ka˨˩./") is None
    assert convert_wiktionary("/ma..ka/") is None
    assert convert_wiktionary("/.naːm˦˥/") is None
    assert convert_wiktionary("/naːm˦˥/") == (
        Syllable(segments=("n", "a", "m"), vowel_length="long", tone="high"),)


_ISAN_ROW = ('<tr><th colspan="2">(<i><a href="x">standard</a></i>) <a href="y">IPA</a>'
             '<sup>(<a href="z">key</a>)</sup></th>'
             '<td><span class="IPA">/nam˦˥/</span></td></tr>')


def test_extract_standard_ipa_reads_the_thai_section_only():
    """A Wiktionary page is one page per spelling, not per language: น้ำ
    carries Northern Thai, Nyaw and Isan sections beside the Thai one,
    each with its own "(standard) IPA" row. Only the Thai section's
    readings are this deck's."""
    html = ('<div class="mw-heading mw-heading2"><h2 id="Isan">Isan</h2></div>'
            '<table>' + _ISAN_ROW + '</table>'
            '<div class="mw-heading mw-heading2"><h2 id="Thai">Thai</h2></div>'
            '<table>' + _ROW + '</table>')
    assert extract_standard_ipa(html) == ["ma˦˥.ka˨˩.raː˧", "mok̚˦˥.ka˨˩.raː˧"]


def _wiktionary_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / "wiktionary"
            / f"{name}.html").read_text(encoding="utf-8")


def test_the_captured_makara_page_yields_both_readings():
    """The real markup, captured from en.wiktionary.org 2026-09-21 (the
    Thai section's pronunciation table, unedited)."""
    ipa = extract_standard_ipa(_wiktionary_fixture("มกรา"))
    assert ipa == ["ma˦˥.ka˨˩.raː˧", "mok̚˦˥.ka˨˩.raː˧"]
    assert [convert_wiktionary(one) for one in ipa] == [
        (S("m", "a", "", "short", "high"), S("k", "a", "", "short", "low"),
         S("r", "a", "", "long", "mid")),
        (S("m", "o", "k", "short", "high"), S("k", "a", "", "short", "low"),
         S("r", "a", "", "long", "mid"))]


def test_the_captured_nam_page_keeps_only_the_reading_that_is_whole():
    """น้ำ's row lists the compound-linking variants beside the plain
    reading; only the plain one converts (B1), so the entry contributes
    one reading, not three."""
    ipa = extract_standard_ipa(_wiktionary_fixture("น้ำ"))
    assert ipa == ["naːm˦˥", "naːm˦˥.", "nam˦˥."]
    assert [convert_wiktionary(one) for one in ipa] == [
        (S("n", "a", "m", "long", "high"),), None, None]
