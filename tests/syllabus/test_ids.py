"""ids.slug_id: the id a newly adopted closure Word takes from its gloss
(design 2026-09-12 §1; the live convention is `delicious-2`)."""
import pytest

from thai_syllabus.ids import slug_id


def test_a_gloss_becomes_a_lowercase_hyphenated_slug():
    assert slug_id("water buffalo") == "water-buffalo"
    assert slug_id("Chicken") == "chicken"


def test_punctuation_and_runs_collapse_to_single_hyphens():
    assert slug_id("Montho (a character)") == "montho-a-character"
    assert slug_id("  small   cymbals  ") == "small-cymbals"


def test_a_taken_id_is_suffixed_from_two_upwards():
    assert slug_id("chicken", ("chicken",)) == "chicken-2"
    assert slug_id("chicken", ("chicken", "chicken-2")) == "chicken-3"


def test_a_gloss_with_no_usable_characters_is_refused_naming_it():
    with pytest.raises(ValueError, match="'!!!'"):
        slug_id("!!!")
