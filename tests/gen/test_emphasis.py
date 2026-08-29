import yaml
from thai_deck_gen.emphasis import Emphasis, load_emphasis


def test_load_emphasis_reads_theme_and_weights(tmp_path):
    p = tmp_path / "emphasis.yaml"
    p.write_text(yaml.safe_dump({"theme": "food, cooking, and eating out",
                                 "category_weights": {"default": 1.2, "Food": 3}}))
    e = load_emphasis(p)
    assert e.theme == "food, cooking, and eating out"
    assert e.weight("Food") == 3
    assert e.weight("Verbs") == 1.2          # falls back to default


def test_load_emphasis_missing_file_is_none(tmp_path):
    assert load_emphasis(tmp_path / "nope.yaml") is None


def test_emphasis_weight_defaults_to_one_without_default_key():
    assert Emphasis(theme="x", category_weights={"Food": 2}).weight("Verbs") == 1.0


def test_emphasized_means_weighted_above_the_default():
    e = Emphasis(theme="x", category_weights={"default": 1.2, "Food": 3, "Colors": 1.2})
    assert e.emphasized("Food") is True
    assert e.emphasized("Verbs") is False        # default weight only
    assert e.emphasized("Colors") is False       # explicitly at the default


def test_emphasized_without_default_key_means_any_weight_above_one():
    assert Emphasis(theme="x", category_weights={"Food": 2}).emphasized("Food") is True
    assert Emphasis(theme="x", category_weights={"Food": 2}).emphasized("Verbs") is False
