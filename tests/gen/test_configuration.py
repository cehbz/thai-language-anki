"""What a deck's configuration actually enables.

Config is code that nothing type-checks: a missing key or an unreferenced
secret disables a whole channel silently, and every function below it keeps
passing its own tests. These assert what a run built from a given gen.yaml
can and cannot do.
"""
from pathlib import Path

import pytest
import yaml

from thai_deck_gen.context import build_context, image_judge_for
from thai_deck_gen.media.images import usable_corpora


class _Fake:
    def syllables(self, w): return None
    def tokens(self, t): return t.split()
    def rank(self, w): return None


def _deck(tmp_path, gen_yaml: dict | None = None, name="deck"):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    if gen_yaml is not None:
        (root / "gen.yaml").write_text(yaml.safe_dump(gen_yaml, allow_unicode=True))
    return root


def _ctx(root, data_dir=Path("data")):
    return build_context(root, data_dir, llm=None, nlp=False,
                         g2p=_Fake(), tokenizer=_Fake(), freq=_Fake())


def _keyfile(tmp_path, name, value="secret-value"):
    p = tmp_path / name
    p.write_text(value + "\n")
    p.chmod(0o600)
    return p


def test_a_secret_reference_enables_its_channel(tmp_path):
    key = _keyfile(tmp_path, "pexels.key")
    ctx = _ctx(_deck(tmp_path, {"secrets": {"pexels": str(key)}}))
    assert ctx.pexels_key == "secret-value"
    assert "pexels" in usable_corpora(ctx)


def test_an_unreferenced_secret_leaves_its_channel_disabled(tmp_path):
    """The Pexels key existed on disk for three runs while gen.yaml never
    mentioned it, and every one silently searched without it."""
    _keyfile(tmp_path, "pexels.key")
    ctx = _ctx(_deck(tmp_path, {}))
    assert ctx.pexels_key is None
    assert "pexels" not in usable_corpora(ctx)


def test_a_world_readable_key_is_refused(tmp_path):
    key = tmp_path / "loose.key"
    key.write_text("value\n")
    key.chmod(0o644)
    from thai_deck_eval.secrets import SecretError
    with pytest.raises(SecretError):
        _ctx(_deck(tmp_path, {"secrets": {"pexels": str(key)}})).pexels_key


def test_images_flag_controls_whether_search_is_wired(tmp_path):
    assert _ctx(_deck(tmp_path, {"images": True}, "on")).http_get is not None
    assert _ctx(_deck(tmp_path, {"images": False}, "off")).http_get is None


def test_search_proxy_is_applied_to_search_requests(tmp_path):
    ctx = _ctx(_deck(tmp_path, {"search_proxy": "http://proxy:8888"}))
    assert ctx.http_get.keywords["proxies"]["https"] == "http://proxy:8888"


def test_rulebook_reference_supplies_the_image_judge(tmp_path):
    rb = tmp_path / "rb.yaml"
    rb.write_text(yaml.safe_dump({"judge": {"backend": "fake"}}))
    root = _deck(tmp_path, {"rulebook": str(rb)})
    assert image_judge_for(root, _ctx(root).config) is not None


def test_without_a_rulebook_there_is_no_image_judge(tmp_path):
    root = _deck(tmp_path, {})
    assert image_judge_for(root, _ctx(root).config) is None


def test_llm_backend_defaults_to_the_subscription_cli(tmp_path):
    """Subscription tokens are already paid for; API spend is not."""
    import thai_deck_gen.cli as gcli
    from thai_deck_gen.config import load_config
    key = _keyfile(tmp_path, "anthropic.key")
    root = _deck(tmp_path, {"secrets": {"anthropic": str(key)}})
    assert type(gcli._drafting_backend(root, load_config(root))).__name__ == "CliBackend"


def test_llm_backend_api_is_opt_in(tmp_path):
    import thai_deck_gen.cli as gcli
    from thai_deck_gen.config import load_config
    key = _keyfile(tmp_path, "anthropic.key")
    root = _deck(tmp_path, {"llm_backend": "api",
                            "secrets": {"anthropic": str(key)}})
    assert type(gcli._drafting_backend(root, load_config(root))).__name__ == "ApiBackend"


def test_defaults_apply_when_gen_yaml_is_absent(tmp_path):
    ctx = _ctx(_deck(tmp_path))
    assert ctx.config.image_candidates == 5
    assert ctx.config.sentence_base == 300
    assert ctx.pexels_key is None


def test_the_real_deck_config_enables_what_we_think_it_does():
    """The live deck, not a fixture: this is the check that was missing."""
    root = Path.home() / "decks" / "thai-ff"
    if not (root / "gen.yaml").exists():
        pytest.skip("live deck not present")
    ctx = _ctx(root)
    assert ctx.pexels_key, "pexels is unreferenced in the live gen.yaml"
    assert ctx.http_get is not None, "image search is disabled"
    assert image_judge_for(root, ctx.config) is not None, "no image judge configured"
    assert {e.thai for e in ctx.word_list if e.image_query}, "no search phrases"
