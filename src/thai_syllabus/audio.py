"""Recording conditioning (spec 1 r33): the bytes a Recording is hashed
over. A clip in which voice activity detection finds no speech
(`holds_speech`: no Silero VAD window's speech probability reaches
SPEECH_PROBABILITY_FLOOR) is refused (NoSpeech). Otherwise
leading and trailing audio more than TRIM_BELOW_PEAK_DB below
the clip's own peak is trimmed, TRIM_PAD_S kept at each end; inner
silence is untouched. The trimmed clip is brought to TARGET_LUFS
integrated loudness by one linear gain, capped so its true peak stays at
TRUE_PEAK_CEILING_DBTP, and encoded as mono VBR mp3 at the source's
sample rate.

After `holds_speech`, five ffmpeg/ffprobe runs over a temp dir: the peak
(astats, on the mono downmix); the trim (silenceremove with peak detection, the trailing end
trimmed by running it again between two areverse); the trimmed duration;
the loudness and true peak (ebur128), a clip shorter than ebur128's 400 ms
gating block measured looped to at least that length; the gain and
encode, bitexact and without an ID3 tag, so equal input gives equal bytes.
"""
from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from pysilero_vad import SileroVoiceActivityDetector

SPEECH_PROBABILITY_FLOOR = 0.25   # Silero VAD, per 32 ms window
TRIM_BELOW_PEAK_DB = 40.0
TRIM_PAD_S = 0.05
TARGET_LUFS = -20.0
TRUE_PEAK_CEILING_DBTP = -1.5
MP3_VBR_QUALITY = 4          # libmp3lame -q:a
LOUDNESS_BLOCK_S = 0.4       # ebur128's gating block

_MONO = "aformat=channel_layouts=mono"
_FFMPEG = ("ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-y")


class NoSpeech(ValueError):
    """A clip `holds_speech` rejects: refused at ingest."""


def _trim_end(threshold_db: float) -> str:
    return (f"silenceremove=start_periods=1:start_threshold={threshold_db:.2f}dB"
            f":start_silence={TRIM_PAD_S}:detection=peak")


def trim_filter(peak_db: float) -> str:
    """The filter chain that trims both ends of a clip peaking at
    `peak_db` dBFS."""
    end = _trim_end(peak_db - TRIM_BELOW_PEAK_DB)
    return f"{_MONO},{end},areverse,{end},areverse"


def _run(cmd: list[str], runner: Callable[..., Any]) -> str:
    """stderr of `cmd`; ValueError when it fails."""
    proc = runner(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"cannot decode recording: {(proc.stderr or '').strip()[-300:]}")
    return proc.stderr or ""


def _number(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"cannot decode recording: no match for {pattern!r}")
    return float(m.group(1))


def peak_db(path: Path, runner: Callable[..., Any] = subprocess.run) -> float:
    """The clip's sample peak in dBFS on its mono downmix; -inf for
    digital silence."""
    err = _run([*_FFMPEG, "-i", str(path), "-af",
                f"{_MONO},astats=measure_overall=Peak_level:measure_perchannel=none",
                "-f", "null", "-"], runner)
    return _number(r"Peak level dB:\s*(-?inf|-?[\d.]+)", err)


def _duration(path: Path, runner: Callable[..., Any]) -> float:
    """ffprobe's duration of the trimmed clip."""
    proc = runner(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float((proc.stdout or "").strip())
    except ValueError as e:
        raise ValueError(f"cannot decode recording: ffprobe read no duration: {e}") from e


def loudness(path: Path, duration: float,
             runner: Callable[..., Any] = subprocess.run) -> tuple[float, float]:
    """(integrated loudness LUFS, true peak dBTP) of the clip, looped to
    LOUDNESS_BLOCK_S when shorter."""
    loops = max(0, math.ceil(LOUDNESS_BLOCK_S / duration) - 1)
    err = _run([*_FFMPEG, "-stream_loop", str(loops), "-i", str(path),
                "-af", "ebur128=peak=true", "-f", "null", "-"], runner)
    summary = err[err.rfind("Summary:"):]
    return (_number(r"I:\s+(-?inf|-?[\d.]+) LUFS", summary),
            _number(r"Peak:\s+(-?inf|-?[\d.]+) dBFS", summary))


def speech_probability(pcm: bytes) -> float:
    """Silero VAD's highest speech probability over the windows of `pcm`
    (16 kHz mono s16le), the last window zero-padded."""
    vad = SileroVoiceActivityDetector()
    size = vad.chunk_bytes()
    return max((vad(pcm[i:i + size].ljust(size, b"\0")) for i in range(0, len(pcm), size)),
               default=0.0)


def holds_speech(data: bytes, ext: str,
                 runner: Callable[..., Any] = subprocess.run) -> bool:
    """Whether the raw clip `data` (format `ext`) holds speech: some
    window's speech probability reaches SPEECH_PROBABILITY_FLOOR.
    ValueError when ffmpeg cannot decode it."""
    with tempfile.TemporaryDirectory(prefix="speech-") as tmp:
        src = Path(tmp) / f"src.{ext}"
        src.write_bytes(data)
        proc = runner([*_FFMPEG, "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "s16le",
                       "pipe:1"], capture_output=True)
    if proc.returncode != 0:
        raise ValueError("cannot decode recording: "
                         f"{(proc.stderr or b'').decode(errors='replace').strip()[-300:]}")
    return speech_probability(proc.stdout) >= SPEECH_PROBABILITY_FLOOR


def condition_recording(data: bytes, ext: str,
                        runner: Callable[..., Any] = subprocess.run) -> bytes:
    """The conditioned mp3 bytes of `data` (a clip in format `ext`);
    NoSpeech when it holds no speech, ValueError when ffmpeg cannot
    decode it."""
    if not holds_speech(data, ext, runner):
        raise NoSpeech("no speech: the clip holds no speech")
    with tempfile.TemporaryDirectory(prefix="condition-") as tmp:
        src = Path(tmp) / f"src.{ext}"
        src.write_bytes(data)
        peak = peak_db(src, runner)
        trimmed = Path(tmp) / "trimmed.wav"
        _run([*_FFMPEG, "-i", str(src), "-af", trim_filter(peak),
              "-c:a", "pcm_f32le", str(trimmed)], runner)
        duration = _duration(trimmed, runner)
        integrated, true_peak = loudness(trimmed, duration, runner)
        gain = min(TARGET_LUFS - integrated, TRUE_PEAK_CEILING_DBTP - true_peak)
        out = Path(tmp) / "out.mp3"
        _run([*_FFMPEG, "-i", str(trimmed), "-af", f"volume={gain:.2f}dB",
              "-c:a", "libmp3lame", "-q:a", str(MP3_VBR_QUALITY),
              "-map_metadata", "-1", "-id3v2_version", "0",
              "-fflags", "+bitexact", "-flags:a", "+bitexact", str(out)], runner)
        return out.read_bytes()
