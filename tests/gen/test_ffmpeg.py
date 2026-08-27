import pytest
from thai_deck_gen.media.ffmpeg import AudioError, normalize_audio

def test_normalize_invokes_ffmpeg(tmp_path):
    calls = []
    def runner(cmd, **kw):
        calls.append(cmd)
        class R: returncode = 0; stderr = b""
        return R()
    normalize_audio(b"raw", tmp_path / "out.mp3", runner=runner)
    cmd = calls[0]
    assert cmd[0] == "ffmpeg" and "loudnorm" in " ".join(cmd)
    assert str(tmp_path / "out.mp3") in cmd

def test_normalize_raises_on_failure(tmp_path):
    def runner(cmd, **kw):
        class R: returncode = 1; stderr = b"bad"
        return R()
    with pytest.raises(AudioError):
        normalize_audio(b"raw", tmp_path / "out.mp3", runner=runner)
