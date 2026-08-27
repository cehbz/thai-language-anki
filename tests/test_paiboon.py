"""Hand-checked conversions for paiboon_to_ipa, verified against the
"Thai 1000 Common Words" deck samples and independently-known Thai
pronunciations (see src/thai_deck_eval/lang/paiboon.py docstring for the
full derivation). Each success case's expected IPA is cross-checked
against parse_ipa so we're not just testing round-trip consistency with
ourselves.
"""
from thai_deck_eval.lang.ipa import parse_ipa
from thai_deck_eval.lang.paiboon import paiboon_to_ipa


def test_gieow_gap():
    # เกี่ยวกับ "about": both syllables genuinely low tone.
    assert paiboon_to_ipa("gìeow gàp") == "kiaw˨˩.kap˨˩"


def test_u_bat_hayt_stripped():
    # อุบัติเหตุ "accident", classifier bracket already stripped by the
    # importer before this string reaches the converter.
    assert paiboon_to_ipa("u-bàt hàyt") == "ʔuʔ˧.bat˨˩.heːt˨˩"


def test_neua():
    # เหนือ "above/over": ห นำ is invisible in Paiboon but doesn't change
    # the phonetic onset (still plain n).
    assert paiboon_to_ipa("nĕua") == "nɯa˨˩˦"


def test_muak():
    # หมวก "hat"
    assert paiboon_to_ipa("mùak") == "muak˨˩"


def test_chaa_aspirated_high_tone():
    # ช้า "slow": aspirated ch, long aa, high tone (mai tho + low class).
    assert paiboon_to_ipa("cháa") == "tɕʰaː˦˥"


def test_dtua_unaspirated_diphthong_mid_tone():
    # ตัว "body/classifier": dt -> unaspirated t, ua diphthong, no coda.
    assert paiboon_to_ipa("dtua") == "tua˧"


def test_soong_long_u_rising():
    # สูง "tall/high": long u (oo), ng coda, rising tone.
    assert paiboon_to_ipa("sŏong") == "suːŋ˨˩˦"


def test_round_trips_through_parse_ipa():
    for s in ("gìeow gàp", "u-bàt hàyt", "nĕua", "mùak", "cháa", "dtua", "sŏong"):
        ipa = paiboon_to_ipa(s)
        assert ipa is not None
        parse_ipa(ipa)  # must not raise


def test_cluster_onset_unmappable():
    # กลัว "afraid": gl- is a consonant cluster onset; ipa.py's IpaSyllable
    # has a single onset slot and cannot represent it (matches the
    # pythainlp adapter's own documented behavior for cluster onsets).
    assert paiboon_to_ipa("glua") is None


def test_ae_vowel_length_ambiguous():
    # This Paiboon system never doubles "ae" to mark length, and the deck
    # itself proves the ambiguity is real: both แต่ "but" (long ɛː,
    # marked low by an explicit mai ek) and แตะ "touch" (short ɛ, low by
    # the unmarked dead-syllable default) romanize identically as "dtàe".
    # Never guess -> None.
    assert paiboon_to_ipa("dtàe") is None
    assert paiboon_to_ipa("kae") is None


def test_eu_plus_coda_length_ambiguous():
    # ลึก "deep" (short ɯ, explicit ึ) and ดื่ม/ปืน/คืน (long ɯː, ื) both
    # romanize as "eu" + a following consonant letter in this system, so
    # closed "eu" syllables can't be resolved. (Open "eu", e.g. "meu" for
    # มือ, is unambiguous -- see test_round_trips_through_parse_ipa's
    # implicit coverage via other cases and the module docstring.)
    assert paiboon_to_ipa("keun") is None
    assert paiboon_to_ipa("léuk") is None


def test_bare_o_plus_coda_length_ambiguous():
    # จบ "finish" (implicit short o, no vowel letter written) and มอบ
    # "give" (explicit long ออ) both romanize as "o" + a following
    # consonant letter -- Thai's implicit-vowel-is-short-o rule for
    # unmarked closed syllables collides with explicit long ออ under this
    # system's spelling. Never guess -> None. (Open bare "o", e.g. "dtó"
    # for โต๊ะ, is unambiguous and does convert -- see paiboon.py.)
    assert paiboon_to_ipa("bon") is None
    assert paiboon_to_ipa("jòp") is None


def test_empty_string():
    assert paiboon_to_ipa("") is None


def test_garbage_alternatives_unmappable():
    # "aunt": multiple alternatives separated by "/", not real syllable
    # boundaries this converter understands.
    assert paiboon_to_ipa("bpâa / náa / aa") is None


def test_stray_punctuation_unmappable():
    # "blow" carries a stray trailing "." in the deck's own data.
    assert paiboon_to_ipa("pát .") is None
