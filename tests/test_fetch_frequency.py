"""Unit tests for scripts/fetch_frequency.py's pure filter/blend logic.
No network access and no pythainlp import: `_fetch_subtitle_words` /
`_fetch_tnc_words` / `_thai_dictionary` (which do real network/pythainlp
work) are exercised only by actually running the script, not by these
tests -- see the follow-up brief's report for that run's sanity check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_frequency as ff  # noqa: E402


def test_is_thai_token_accepts_thai_script_only():
    assert ff._is_thai_token("หมา")
    assert not ff._is_thai_token("dog")
    assert not ff._is_thai_token("หมา1")
    assert not ff._is_thai_token("หมา dog")
    assert not ff._is_thai_token("")


def test_filter_thai_tokens_drops_non_thai_preserving_order():
    words = ["หมา", "cat", "มา", "ก1ข", "ข้าว"]
    assert ff._filter_thai_tokens(words) == ["หมา", "มา", "ข้าว"]


def test_filter_dictionary_words_drops_non_dict_and_non_thai():
    dictionary = {"หมา", "มา"}
    words = ["หมา", "xyz", "มา", "ปูมกวย", "มา"]
    assert ff._filter_dictionary_words(words, dictionary) == ["หมา", "มา", "มา"]


def test_ranks_are_1_based_by_position():
    assert ff._ranks(["a", "b", "c"]) == {"a": 1, "b": 2, "c": 3}


def test_blend_word_ranked_well_in_both_sources_beats_one_ranked_well_in_only_one():
    subtitle = ["a", "b", "c", "d"]
    tnc = ["a", "c", "b", "d"]
    ranked = ff._blend(subtitle, tnc)
    assert ranked[0] == "a"  # rank 1 in both sources


def test_blend_penalizes_word_missing_from_a_source():
    # a: subtitle rank 1/4, absent from tnc -> 0.7*0.25 + 0.3*1.0 = 0.475
    # b: subtitle rank 2/4, tnc rank 1/3   -> 0.7*0.5 + 0.3*0.333 = 0.45
    # despite b's worse subtitle rank, corroboration from a second source
    # (and a's missing-source 1.0 penalty) puts b ahead.
    subtitle = ["a", "b", "c", "d"]
    tnc = ["b", "c", "d"]  # "a" absent from tnc
    ranked = ff._blend(subtitle, tnc)
    assert ranked.index("b") < ranked.index("a")


def test_blend_weights_subtitle_source_more_than_tnc():
    # a: #1 in subtitle, absent from tnc -> 0.7*(1/3) + 0.3*1.0 = 0.5333
    # b: #1 in tnc, absent from subtitle -> 0.7*1.0 + 0.3*(1/3) = 0.8
    subtitle = ["a", "x", "y"]
    tnc = ["b", "p", "q"]
    ranked = ff._blend(subtitle, tnc, w_subtitle=0.7, w_tnc=0.3)
    assert ranked.index("a") < ranked.index("b")


def test_blend_returns_union_of_both_sources():
    subtitle = ["a", "b"]
    tnc = ["c", "d"]
    assert set(ff._blend(subtitle, tnc)) == {"a", "b", "c", "d"}
