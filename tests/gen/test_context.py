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
