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
        {"id": "water", "thai": "น้ำ", "gloss": "water", "category": "Beverages",
         "part_of_speech": "noun", "classifier": "แก้ว"}])
    entries = load_word_list(p, DATA / "categories.yaml")
    assert entries[0].thai == "น้ำ"

def test_load_rejects_unknown_category(tmp_path):
    p = _write_list(tmp_path, [
        {"id": "water", "thai": "น้ำ", "gloss": "water", "category": "Nope",
         "part_of_speech": "noun", "classifier": "แก้ว"}])
    with pytest.raises(ValueError, match="category"):
        load_word_list(p, DATA / "categories.yaml")

def test_load_rejects_noun_without_classifier(tmp_path):
    p = _write_list(tmp_path, [
        {"id": "water", "thai": "น้ำ", "gloss": "water", "category": "Beverages",
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
        {"id": "water", "thai": "น้ำ", "gloss": "water", "category": categories[0],
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

# --- extension pass -------------------------------------------------------

from thai_deck_gen.emphasis import Emphasis
from thai_deck_gen.wordlist import extend_word_list

def _base_file(tmp_path, categories):
    out = tmp_path / "wl.yaml"
    out.write_text(yaml.safe_dump([
        {"id": "rice", "thai": "ข้าว", "gloss": "rice", "category": "Food",
         "part_of_speech": "noun", "classifier": "จาน"},
        {"id": "dog", "thai": "หมา", "gloss": "dog", "category": "Animals",
         "part_of_speech": "noun", "classifier": "ตัว"}], allow_unicode=True),
        encoding="utf-8")
    return out

def test_load_word_list_keeps_emphasis_tag(tmp_path):
    p = _write_list(tmp_path, [
        {"id": "water", "thai": "น้ำ", "gloss": "water", "category": "Beverages",
         "part_of_speech": "noun", "classifier": "แก้ว", "emphasis": True}])
    entries = load_word_list(p, DATA / "categories.yaml")
    assert entries[0].emphasis is True

def test_extend_word_list_adds_tagged_entries_only_for_weighted_categories(tmp_path):
    categories = yaml.safe_load((DATA / "categories.yaml").read_text())
    out = _base_file(tmp_path, categories)
    fake = FakeLlm([_entry("ผัด")])              # one call expected: Food only
    emphasis = Emphasis(theme="food and cooking",
                        category_weights={"Food": 2})   # others weight 1 -> 0 extra
    count = extend_word_list(fake, DATA / "categories.yaml",
                             DATA / "frequency_th.txt", out, emphasis)
    assert len(fake.prompts) == 1
    assert "food and cooking" in fake.prompts[0]
    assert "ข้าว" in fake.prompts[0]              # existing Food entries are exclusions
    assert "Already listed" in fake.prompts[0] or "already" in fake.prompts[0].lower()
    saved = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert count == 1
    added = [e for e in saved if e.get("emphasis")]
    assert [e["thai"] for e in added] == ["ผัด"]
    assert len(saved) == 3                        # base entries untouched

def test_extend_word_list_requests_weight_scaled_count(tmp_path):
    categories = yaml.safe_load((DATA / "categories.yaml").read_text())
    out = _base_file(tmp_path, categories)
    fake = FakeLlm([_entry("ผัด")])
    per_category = -(-625 // len(categories))     # 24
    extend_word_list(fake, DATA / "categories.yaml", DATA / "frequency_th.txt",
                     out, Emphasis(theme="t", category_weights={"Food": 3}))
    assert f"Target: {round(per_category * 2)} " in fake.prompts[0]   # 3x -> 48 extra

def test_extend_word_list_resumes_skipping_extended_categories(tmp_path):
    categories = yaml.safe_load((DATA / "categories.yaml").read_text())
    out = _base_file(tmp_path, categories)
    emphasis = Emphasis(theme="t", category_weights={"Food": 2})
    extend_word_list(FakeLlm([_entry("ผัด")]), DATA / "categories.yaml",
                     DATA / "frequency_th.txt", out, emphasis)
    again = FakeLlm([])
    count = extend_word_list(again, DATA / "categories.yaml",
                             DATA / "frequency_th.txt", out, emphasis)
    assert again.prompts == []                    # Food already extended
    assert count == 1

def test_parse_entries_strips_markdown_code_fences():
    from thai_deck_gen.wordlist import _parse_entries
    fenced = "```yaml\n" + _entry("ผัด") + "```\n"
    warnings = []
    parsed = _parse_entries(fenced, "Food", warnings)
    assert [e.thai for e in parsed] == ["ผัด"]
    assert warnings == []
