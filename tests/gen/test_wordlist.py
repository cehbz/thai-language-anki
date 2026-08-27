import pytest, yaml
from pathlib import Path
from thai_deck_gen.wordlist import WordEntry, draft_word_list, load_word_list
from tests.gen.fakes import FakeLlm

DATA = Path(__file__).parents[2] / "data"

def _write_list(tmp_path, entries):
    p = tmp_path / "wl.yaml"
    p.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")
    return p

def test_load_valid_word_list(tmp_path):
    p = _write_list(tmp_path, [
        {"thai": "น้ำ", "gloss": "water", "category": "Beverages",
         "part_of_speech": "noun", "classifier": "แก้ว"}])
    entries = load_word_list(p, DATA / "categories.yaml")
    assert entries[0].thai == "น้ำ"

def test_load_rejects_unknown_category(tmp_path):
    p = _write_list(tmp_path, [
        {"thai": "น้ำ", "gloss": "water", "category": "Nope",
         "part_of_speech": "noun", "classifier": "แก้ว"}])
    with pytest.raises(ValueError, match="category"):
        load_word_list(p, DATA / "categories.yaml")

def test_load_rejects_noun_without_classifier(tmp_path):
    p = _write_list(tmp_path, [
        {"thai": "น้ำ", "gloss": "water", "category": "Beverages",
         "part_of_speech": "noun"}])
    with pytest.raises(ValueError, match="classifier"):
        load_word_list(p, DATA / "categories.yaml")

def test_draft_word_list_writes_entries(tmp_path):
    per_cat = yaml.safe_dump([
        {"thai": "น้ำ", "gloss": "water", "category": "CAT",
         "part_of_speech": "noun", "classifier": "แก้ว"}], allow_unicode=True)
    n_categories = len([c for c in yaml.safe_load(
        (DATA / "categories.yaml").read_text()) ])
    fake = FakeLlm([per_cat] * n_categories)
    out = tmp_path / "wl.yaml"
    count = draft_word_list(fake, DATA / "categories.yaml",
                            DATA / "frequency_th.txt", out)
    assert count == n_categories          # one valid entry per category
    assert out.exists()
    assert len(fake.prompts) == n_categories
    assert "YAML" in fake.prompts[0]

def test_draft_word_list_reports_dropped_entries(tmp_path):
    categories = yaml.safe_load((DATA / "categories.yaml").read_text())
    good = yaml.safe_dump([
        {"thai": "น้ำ", "gloss": "water", "category": "CAT",
         "part_of_speech": "noun", "classifier": "แก้ว"}], allow_unicode=True)
    bad = yaml.safe_dump([
        {"thai": "น้ำ", "gloss": "water", "category": "CAT",
         "part_of_speech": "noun"}], allow_unicode=True)  # missing classifier
    fake = FakeLlm([bad] + [good] * (len(categories) - 1))
    out = tmp_path / "wl.yaml"
    warnings = []
    count = draft_word_list(fake, DATA / "categories.yaml",
                            DATA / "frequency_th.txt", out, warnings=warnings)
    assert count == len(categories) - 1
    assert len(warnings) == 1
    assert categories[0] in warnings[0]
    assert "classifier" in warnings[0]
