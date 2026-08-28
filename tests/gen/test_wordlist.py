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

class _FailAfter:
    """Yields canned responses, then raises on the next call."""
    def __init__(self, responses, exc):
        self.responses = list(responses)
        self.exc = exc
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise self.exc
        return self.responses.pop(0)

def _entry(thai):
    return yaml.safe_dump([
        {"thai": thai, "gloss": "water", "category": "CAT",
         "part_of_speech": "noun", "classifier": "แก้ว"}], allow_unicode=True)

def test_draft_word_list_persists_completed_categories_on_failure(tmp_path):
    from thai_deck_gen.llm import LlmError
    fake = _FailAfter([_entry("น้ำ"), _entry("นม")], LlmError("limit"))
    out = tmp_path / "wl.yaml"
    with pytest.raises(LlmError):
        draft_word_list(fake, DATA / "categories.yaml",
                        DATA / "frequency_th.txt", out)
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert len(saved) == 2                # the two completed categories
    categories = yaml.safe_load((DATA / "categories.yaml").read_text())
    assert {e["category"] for e in saved} == set(categories[:2])

def test_draft_word_list_resumes_skipping_completed_categories(tmp_path):
    categories = yaml.safe_load((DATA / "categories.yaml").read_text())
    out = tmp_path / "wl.yaml"
    out.write_text(yaml.safe_dump([
        {"thai": "น้ำ", "gloss": "water", "category": categories[0],
         "part_of_speech": "noun", "classifier": "แก้ว"}], allow_unicode=True),
        encoding="utf-8")
    fake = FakeLlm([_entry("นม")] * (len(categories) - 1))
    count = draft_word_list(fake, DATA / "categories.yaml",
                            DATA / "frequency_th.txt", out)
    assert len(fake.prompts) == len(categories) - 1   # first category skipped
    assert count == len(categories)                    # existing entry kept
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert {e["category"] for e in saved} == set(categories)
    assert any(e["thai"] == "น้ำ" for e in saved)

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
