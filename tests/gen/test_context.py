import pytest
from pathlib import Path

from thai_deck_gen.context import build_context

DATA = Path(__file__).parents[2] / "data"


class _Fake:
    def syllables(self, word):
        return None

    def tokens(self, text):
        return text.split()

    def rank(self, word):
        return None


def _build(tmp_path):
    return build_context(tmp_path, DATA, llm=None, nlp=False,
                         g2p=_Fake(), tokenizer=_Fake(), freq=_Fake())


def test_media_config_defaults_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("FORVO_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_TTS_API_KEY", raising=False)
    monkeypatch.delenv("THAI_DECK_GEN_FAKE", raising=False)
    ctx = _build(tmp_path)
    assert ctx.forvo_api_key is None
    assert ctx.tts_api_key is None
    assert ctx.thai1000_apkg is None
    assert ctx.http_get is not None                # images default on, no network fired


def test_forvo_api_key_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORVO_API_KEY", "secret-forvo")
    ctx = _build(tmp_path)
    assert ctx.forvo_api_key == "secret-forvo"


def test_tts_api_key_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "secret-tts")
    ctx = _build(tmp_path)
    assert ctx.tts_api_key == "secret-tts"


def test_thai1000_apkg_from_gen_yaml_relative(tmp_path, monkeypatch):
    monkeypatch.delenv("THAI_DECK_GEN_FAKE", raising=False)
    (tmp_path / "gen.yaml").write_text("thai1000_apkg: audio/thai1000.apkg\n")
    ctx = _build(tmp_path)
    assert ctx.thai1000_apkg == tmp_path / "audio" / "thai1000.apkg"


def test_thai1000_apkg_from_gen_yaml_absolute(tmp_path, monkeypatch):
    abs_apkg = tmp_path / "elsewhere" / "thai1000.apkg"
    (tmp_path / "gen.yaml").write_text(f"thai1000_apkg: {abs_apkg}\n")
    ctx = _build(tmp_path)
    assert ctx.thai1000_apkg == abs_apkg


def test_images_false_disables_http_get(tmp_path, monkeypatch):
    (tmp_path / "gen.yaml").write_text("images: false\n")
    ctx = _build(tmp_path)
    assert ctx.http_get is None


def test_fake_env_disables_http_get_even_when_images_true(tmp_path, monkeypatch):
    monkeypatch.setenv("THAI_DECK_GEN_FAKE", "1")
    ctx = _build(tmp_path)
    assert ctx.http_get is None


def test_context_loads_emphasis_profile(tmp_path):
    ctx = _build(tmp_path)
    assert ctx.emphasis is not None
    assert ctx.emphasis.weight("Food") > 1


def test_context_builds_imgfetch_from_config(tmp_path):
    from thai_deck_gen.config import GenConfig
    from thai_deck_gen.media.imgfetch import ImgFetch
    ctx = build_context(tmp_path, DATA, llm=None, nlp=False,
                        g2p=_Fake(), tokenizer=_Fake(), freq=_Fake(),
                        config=GenConfig(imgfetch="/opt/bin/imgfetch"))
    assert isinstance(ctx.imgfetch, ImgFetch)
    assert ctx.imgfetch.binary == "/opt/bin/imgfetch"


def test_gen_config_imgfetch_defaults_to_path_lookup():
    from thai_deck_gen.config import GenConfig
    assert GenConfig().imgfetch == "imgfetch"


def test_gen_config_search_proxy_defaults_to_none():
    from thai_deck_gen.config import GenConfig
    assert GenConfig().search_proxy is None


def test_proxied_get_returns_the_plain_getter_without_a_proxy():
    from thai_deck_gen.context import proxied_get
    def getter(url, **kw): return url
    assert proxied_get(None, getter) is getter


def test_proxied_get_sends_both_schemes_through_the_proxy():
    from thai_deck_gen.context import proxied_get
    seen = {}
    def getter(url, **kw):
        seen.update(kw)
        return "ok"
    assert proxied_get("socks5h://127.0.0.1:1080", getter)("https://x", timeout=5) == "ok"
    assert seen["proxies"] == {"http": "socks5h://127.0.0.1:1080",
                               "https": "socks5h://127.0.0.1:1080"}
    assert seen["timeout"] == 5


def test_proxied_get_rejects_a_socks_proxy_without_pysocks(monkeypatch):
    import thai_deck_gen.context as context
    from thai_deck_gen.context import proxied_get
    monkeypatch.setattr(context, "_have_pysocks", lambda: False)
    with pytest.raises(RuntimeError, match="PySocks"):
        proxied_get("socks5h://127.0.0.1:1080", lambda url, **kw: None)


def test_context_routes_image_search_through_the_configured_proxy(tmp_path):
    from thai_deck_gen.config import GenConfig
    ctx = build_context(tmp_path, DATA, llm=None, nlp=False,
                        g2p=_Fake(), tokenizer=_Fake(), freq=_Fake(),
                        config=GenConfig(search_proxy="socks5h://127.0.0.1:1080"))
    assert ctx.http_get.keywords["proxies"]["https"] == "socks5h://127.0.0.1:1080"
