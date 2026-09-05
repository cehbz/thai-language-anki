"""query.py: the picture search query -- head term of the gloss, the
category qualifier, and a drafted phrase's precedence."""
import pytest

from thai_syllabus.query import QUERY_HINTS, head_term, load_query_hints, picture_query

from .builders import word


def test_head_term_reduces_a_learner_definition():
    assert head_term("I (female speaker, or casual general)") == "I"
    assert head_term("orange (the fruit)") == "orange"


def test_head_term_takes_the_first_alternative():
    assert head_term("soil / earth / dirt") == "soil"
    assert head_term("to go, to leave") == "to go"
    assert head_term("chicken or hen") == "chicken"
    assert head_term("rice; grain") == "rice"


def test_head_term_keeps_a_whole_compound_that_has_no_separator():
    assert head_term("navy blue") == "navy blue"


def test_head_term_drops_a_leading_note_rather_than_cutting_at_it():
    assert head_term("(for) a long time") == "a long time"


def test_picture_query_adds_the_category_qualifier():
    # ส้ม: orange
    assert picture_query(word("orange", "ส้ม", meaning="orange (the colour)"),
                         "Colors", None, {"Colors": "color swatch"}) == "orange color swatch"


def test_picture_query_is_the_head_term_alone_where_the_category_has_no_qualifier():
    # ส้ม: orange
    assert picture_query(word("orange", "ส้ม", meaning="orange (the fruit)"),
                         "Closure", None, {"Colors": "color swatch"}) == "orange"
    assert picture_query(word("orange", "ส้ม", meaning="orange (the fruit)"),
                         None, None, {"Colors": "color swatch"}) == "orange"


def test_picture_query_uses_a_drafted_phrase_verbatim():
    # ส้ม: orange
    assert picture_query(word("orange", "ส้ม", meaning="orange (the fruit)"),
                         "Colors", "a single ripe tangerine on a white plate",
                         {"Colors": "color swatch"}) == "a single ripe tangerine on a white plate"


def test_picture_query_refuses_a_word_whose_gloss_describes_nothing():
    # ส้ม: orange
    with pytest.raises(ValueError, match="orange"):
        picture_query(word("orange", "ส้ม", meaning="(informal)"), "Colors", None, {})


def test_load_query_hints_reads_the_category_qualifier_file(tmp_path):
    path = tmp_path / "image_query_hints.yaml"
    path.write_text("Colors: color\nFood: food\n", encoding="utf-8")
    assert load_query_hints(path) == {"Colors": "color", "Food": "food"}


def test_load_query_hints_refuses_an_absent_file_by_name(tmp_path):
    missing = tmp_path / "nothing.yaml"
    with pytest.raises(FileNotFoundError, match="nothing.yaml"):
        load_query_hints(missing)


def test_the_repo_hints_carry_a_qualifier_for_a_curated_category():
    assert QUERY_HINTS["Colors"] == "color"
