"""The word list's identity contract: one readable immutable id per row."""
import pytest
import yaml
from pathlib import Path

from thai_deck_gen.wordlist import (WordEntry, assign_ids, head_term,
                                    load_word_list, slug)

DATA = Path(__file__).parents[2] / "data"


def _write(tmp_path, rows):
    p = tmp_path / "wl.yaml"
    p.write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


def test_a_row_carries_its_id_through_load():
    e = WordEntry(id="hair", thai="ผม", gloss="hair (on the head)",
                  category="Body", part_of_speech="noun", classifier="เส้น")
    assert e.id == "hair"


def test_image_query_source_survives_load(tmp_path):
    """148 values in the file were being dropped silently: the field was an
    untyped key pydantic ignored, so a judge-written phrase looked
    hand-written to every consumer."""
    p = _write(tmp_path, [{"id": "hair", "thai": "ผม", "gloss": "hair",
                           "category": "Body", "part_of_speech": "noun",
                           "classifier": "เส้น", "image_query": "x",
                           "image_query_source": "judge"}])
    [e] = load_word_list(p, DATA / "categories.yaml")
    assert e.image_query_source == "judge"


def test_a_row_without_an_id_is_refused(tmp_path):
    """Identity is stored, never derived: a row with no id has no identity,
    and inventing one at load time is the derive-at-read the design forbids."""
    p = _write(tmp_path, [{"thai": "ผม", "gloss": "hair", "category": "Body",
                           "part_of_speech": "noun", "classifier": "เส้น"}])
    with pytest.raises(ValueError, match="id"):
        load_word_list(p, DATA / "categories.yaml")


def test_two_rows_with_one_id_are_refused(tmp_path):
    p = _write(tmp_path, [
        {"id": "week", "thai": "สัปดาห์", "gloss": "week (formal)",
         "category": "Time", "part_of_speech": "noun", "classifier": "สัปดาห์"},
        {"id": "week", "thai": "อาทิตย์", "gloss": "week (colloquial)",
         "category": "Time", "part_of_speech": "noun", "classifier": "อาทิตย์"},
    ])
    with pytest.raises(ValueError, match="week"):
        load_word_list(p, DATA / "categories.yaml")


def test_two_rows_with_one_thai_and_two_ids_both_load(tmp_path):
    """A homograph is two words. The string is not the identity."""
    p = _write(tmp_path, [
        {"id": "hair", "thai": "ผม", "gloss": "hair (on the head)",
         "category": "Body", "part_of_speech": "noun", "classifier": "เส้น"},
        {"id": "i-male", "thai": "ผม", "gloss": "I (male speaker)",
         "category": "Pronouns", "part_of_speech": "other"},
    ])
    assert [e.id for e in load_word_list(p, DATA / "categories.yaml")] == ["hair", "i-male"]


# --- seeding ids from the gloss ---

def _row(thai, gloss, category="Time", **kw):
    return {"thai": thai, "gloss": gloss, "category": category,
            "part_of_speech": "noun", "classifier": "x", **kw}


def test_slug_is_the_head_term_lowercased_and_hyphenated():
    assert slug("Navy blue") == "navy-blue"
    assert slug("I (female speaker, or casual general)") == "i"
    assert slug("soil / earth / dirt") == "soil"
    assert slug("cheap / correct") == "cheap"


def test_unique_glosses_get_their_slug():
    rows, _ = assign_ids([_row("ผม", "hair (on the head)"),
                          _row("ผม", "I (male speaker)")])
    assert [r["id"] for r in rows] == ["hair", "i"]


def test_synonyms_are_told_apart_by_the_parenthetical():
    """Different words, same head gloss: the gloss already says which is
    which, so the id says so too rather than numbering them."""
    rows, _ = assign_ids([_row("สัปดาห์", "week (formal)"),
                          _row("อาทิตย์", "week (colloquial)")])
    assert [r["id"] for r in rows] == ["week-formal", "week-colloquial"]


def test_a_collision_with_no_parenthetical_falls_back_to_a_number():
    rows, notes = assign_ids([_row("ก", "thing"), _row("ข", "thing")])
    assert [r["id"] for r in rows] == ["thing", "thing-2"]
    assert any("thing" in n for n in notes)


def test_exact_duplicates_are_removed_before_assignment():
    """Rows that repeat another row's thai and gloss carry nothing the
    surviving row lacks."""
    rows, notes = assign_ids([_row("หมา", "dog", "Animals"),
                              _row("หมา", "dog", "Animals"),
                              _row("แมว", "cat", "Animals")])
    assert [r["id"] for r in rows] == ["dog", "cat"]
    assert any("duplicate" in n for n in notes)


def test_an_existing_id_is_never_rewritten():
    """Stored, not derived: editing the gloss later must not move the id."""
    rows, _ = assign_ids([_row("ผม", "hair, now reworded", id="hair")])
    assert rows[0]["id"] == "hair"


def test_a_row_whose_gloss_yields_no_slug_is_reported_not_guessed():
    rows, notes = assign_ids([_row("ก", "???")])
    assert "id" not in rows[0]
    assert any("???" in n for n in notes)


def test_drafted_ids_avoid_the_ids_already_in_the_list():
    """New rows join an existing list; a fresh 'dog' must not shadow the
    'dog' that is already there."""
    rows, _ = assign_ids([_row("สุนัข", "dog (formal)", "Animals")], taken={"dog"})
    assert rows[0]["id"] == "dog-formal"


def test_a_leading_parenthetical_is_a_prefix_note_not_a_sense_note():
    """"(for) a long time" cut at the first paren is nothing. The image
    query built from it was empty too."""
    assert head_term("(for) a long time") == "a long time"
    assert slug("(for) a long time") == "a-long-time"


def test_collisions_join_more_of_the_parenthetical_before_numbering():
    """Two rows whose parentheticals share a first phrase: the second phrase
    tells them apart, and a number would throw that away."""
    rows, _ = assign_ids([_row("ฉัน", "I (female speaker)"),
                          _row("ดิฉัน", "I (female speaker, polite)")])
    assert [r["id"] for r in rows] == ["i-female-speaker", "i-female-speaker-polite"]
