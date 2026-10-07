"""Recording conditioning (spec 1 r33): ends trimmed relative to the
clip's own peak with 50 ms kept, inner silence untouched, loudness
normalized to -20 LUFS by linear gain with true peak held below -1.5 dBTP.
Clips are generated with ffmpeg's lavfi sources into tmp_path.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from thai_syllabus import audio
from thai_syllabus.audio import (TARGET_LUFS, TRIM_PAD_S,
                                 TRUE_PEAK_CEILING_DBTP, NoSpeech, condition_recording,
                                 holds_speech)

RATE = 24000


def _render(tmp_path: Path, name: str, graph: str, codec: str = "libmp3lame") -> bytes:
    """`graph`: a filter_complex whose output pad is [out]."""
    out = tmp_path / name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-filter_complex", graph, "-map", "[out]",
                    "-c:a", codec, str(out)], check=True)
    return out.read_bytes()


def _tone(d: float, db: float = -20.0) -> str:
    """A 440 Hz sine of `d` seconds peaking near `db` dBFS."""
    return f"sine=f=440:r={RATE}:d={d},volume={db + 18:g}dB"   # sine's own peak is -18 dBFS


def _silence(d: float) -> str:
    return f"anullsrc=r={RATE}:cl=mono:d={d}"


def _sequence(*parts: str) -> str:
    pads = "".join(f"{p}[p{i}];" for i, p in enumerate(parts))
    ins = "".join(f"[p{i}]" for i in range(len(parts)))
    return f"{pads}{ins}concat=n={len(parts)}:v=0:a=1[out]"


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration,format_name:stream=sample_rate,channels",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _duration(path: Path) -> float:
    return float(_probe(path)["format"]["duration"])


def _conditioned_loudness(path: Path) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) as conditioning measures them:
    looping an mp3 would put its encoder padding between the plays."""
    return audio.loudness(path, audio._duration(path, subprocess.run))


def _loudness(path: Path, loops: int = 0) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) by ebur128; `loops` extra plays
    for a clip shorter than its 400 ms gating block."""
    err = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats",
                          "-stream_loop", str(loops), "-i", str(path),
                          "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
                         capture_output=True, text=True, check=True).stderr
    summary = err[err.rindex("Summary:"):]
    i = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))
    tp = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1))
    return i, tp


def test_leading_and_trailing_silence_is_trimmed_to_the_pad(tmp_path):
    raw = _render(tmp_path, "in.mp3", _sequence(_silence(0.5), _tone(0.5), _silence(0.5)))
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "mp3"))
    assert _duration(out) == pytest.approx(0.5 + 2 * TRIM_PAD_S, abs=0.04)


def test_a_gapped_clips_inner_break_survives(tmp_path):
    """Spec 3 r65's gapped recording: a 600 ms break between the halves
    is inner silence, untouched."""
    raw = _render(tmp_path, "in.mp3", _sequence(_silence(0.3), _tone(0.4), _silence(0.6),
                                                _tone(0.4), _silence(0.3)))
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "mp3"))
    assert _duration(out) == pytest.approx(1.4 + 2 * TRIM_PAD_S, abs=0.04)


def test_the_trim_threshold_is_relative_to_the_clips_own_peak(tmp_path):
    """A quiet clip (peak -35 dBFS) keeps a tail at -60 dBFS (25 dB below
    its peak) and loses one at -80 dBFS (45 dB below)."""
    raw = _render(tmp_path, "in.wav", _sequence(_tone(0.4, db=-35), _tone(0.3, db=-60),
                                                _silence(0.2)), codec="pcm_s16le")
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "wav"))
    assert _duration(out) == pytest.approx(0.7 + TRIM_PAD_S, abs=0.04)
    raw = _render(tmp_path, "in2.wav", _sequence(_tone(0.4, db=-35), _tone(0.3, db=-80),
                                                 _silence(0.2)), codec="pcm_s16le")
    out = _write(tmp_path, "out2.mp3", condition_recording(raw, "wav"))
    assert _duration(out) == pytest.approx(0.4 + TRIM_PAD_S, abs=0.04)


def test_a_quiet_clip_is_normalized_to_the_target_loudness(tmp_path):
    raw = _render(tmp_path, "in.mp3", _sequence(_tone(1.0, db=-35)))
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "mp3"))
    i, tp = _loudness(out)
    assert i == pytest.approx(TARGET_LUFS, abs=0.5)


def test_a_clip_shorter_than_the_gating_block_is_normalized_too(tmp_path):
    raw = _render(tmp_path, "in.mp3", _sequence(_tone(0.2, db=-35)))
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "mp3"))
    i, _ = _conditioned_loudness(out)
    # the mp3's encoder padding is silence the trimmed wav did not have;
    # under 400 ms it is a measurable share of the looped block
    assert i == pytest.approx(TARGET_LUFS, abs=1.0)


def test_the_gain_is_capped_by_the_true_peak_ceiling(tmp_path):
    """Sparse clicks: reaching -20 LUFS would push their peaks past 0 dBFS,
    so the gain stops at the true-peak ceiling and the clip stays quieter
    than the target."""
    clicks = (f"aevalsrc=exprs='if(lt(mod(t\\,0.25)\\,0.004)\\,0.25*sin(2*PI*1000*t)\\,0)'"
              f":s={RATE}:d=1.0")
    raw = _render(tmp_path, "in.wav", f"{clicks}[out]", codec="pcm_s16le")
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "wav"))
    i, tp = _loudness(out)
    assert tp <= TRUE_PEAK_CEILING_DBTP + 0.5   # mp3 encoding may overshoot slightly
    assert i < TARGET_LUFS - 1


def test_the_conditioned_clip_is_mono_mp3_at_the_source_sample_rate(tmp_path):
    raw = _render(tmp_path, "in.wav",
                  "sine=f=440:r=44100:d=0.5,aformat=channel_layouts=stereo[out]",
                  codec="pcm_s16le")
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "wav"))
    info = _probe(out)
    assert info["format"]["format_name"] == "mp3"
    assert info["streams"][0]["sample_rate"] == "44100"
    assert info["streams"][0]["channels"] == 1


def test_conditioning_is_deterministic(tmp_path):
    raw = _render(tmp_path, "in.mp3", _sequence(_silence(0.2), _tone(0.5), _silence(0.2)))
    assert condition_recording(raw, "mp3") == condition_recording(raw, "mp3")


# --- the speech rule (Silero VAD). `holds_speech` here is this module's own
# import of the real function; conftest's `clips_hold_speech` patches only
# the module attribute that condition_recording reads.

SPEECH = Path(__file__).parent / "fixtures" / "audio" / "will-erinome.mp3"   # จะ "will", Chirp3-HD-Erinome


def _peak(path: Path) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                          "astats=measure_overall=Peak_level:measure_perchannel=none",
                          "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    return float(re.search(r"Peak level dB:\s*(-?[\d.]+)", err).group(1))


def _speech_at(tmp_path: Path, peak_db: float) -> bytes:
    """The speech fixture scaled to peak at `peak_db` dBFS."""
    out = tmp_path / f"speech{peak_db:g}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(SPEECH), "-af",
                    f"volume={peak_db - _peak(SPEECH):.2f}dB", "-c:a", "pcm_s16le", str(out)],
                   check=True)
    return out.read_bytes()


def test_the_speech_probability_floor_is_a_quarter():
    """Spec 1 r33: the two silent Chirp3-HD clips scored 0.046/0.059, every
    real clip on the deck 0.31 or more."""
    assert audio.SPEECH_PROBABILITY_FLOOR == 0.25


def test_a_tts_word_holds_speech():
    assert holds_speech(SPEECH.read_bytes(), "mp3") is True


def test_a_speech_clip_peaking_at_minus_33_dbfs_still_holds_speech(tmp_path):
    """The quietest real clip on the deck peaks at -33.3 dBFS."""
    assert holds_speech(_speech_at(tmp_path, -33.0), "wav") is True


def test_digital_silence_holds_no_speech(tmp_path):
    raw = _render(tmp_path, "in.wav", _sequence(_silence(0.5)), codec="pcm_s16le")
    assert holds_speech(raw, "wav") is False


def test_a_low_sine_holds_no_speech(tmp_path):
    """A steady tone; a tone gated on and off from silence is not used,
    since its edge clicks score up to about 0.3."""
    raw = _render(tmp_path, "in.mp3", _sequence(_tone(0.5, db=-45)))
    assert holds_speech(raw, "mp3") is False


def test_white_noise_holds_no_speech(tmp_path):
    raw = _render(tmp_path, "in.wav", "anoisesrc=d=1:c=white:a=0.3:r=24000:seed=1[out]",
                  codec="pcm_s16le")
    assert holds_speech(raw, "wav") is False


def test_a_clip_holding_no_speech_is_refused(tmp_path, no_speech):
    raw = _render(tmp_path, "in.mp3", _sequence(_tone(0.5)))
    with pytest.raises(NoSpeech, match="no speech"):
        condition_recording(raw, "mp3")


def test_a_quiet_speech_clip_is_conditioned_under_the_real_rule(tmp_path, monkeypatch):
    """Turned down to a -33 dBFS peak, the fixture still holds speech and
    is brought to the target: its crest (-20.2 LUFS at -2.1 dBTP) leaves
    -20 LUFS inside the true-peak ceiling."""
    monkeypatch.setattr(audio, "holds_speech", holds_speech)
    raw = _speech_at(tmp_path, -33.0)
    out = _write(tmp_path, "out.mp3", condition_recording(raw, "wav"))
    i, tp = _conditioned_loudness(out)
    assert i == pytest.approx(TARGET_LUFS, abs=0.5)
    assert tp <= TRUE_PEAK_CEILING_DBTP + 0.5   # mp3 encoding may overshoot slightly


def test_undecodable_bytes_are_a_value_error():
    with pytest.raises(ValueError, match="cannot decode recording"):
        condition_recording(b"not audio at all", "mp3")
