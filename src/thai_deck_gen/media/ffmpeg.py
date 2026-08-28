import subprocess
from pathlib import Path
import json

class AudioError(Exception):
    """Raised when audio processing fails"""
    pass

def normalize_audio(raw: bytes, dst: Path, runner=subprocess.run) -> None:
    """
    Normalize audio using ffmpeg.
    - Input: raw audio bytes via pipe:0
    - Output: mono, 44100 Hz, loudnorm filter, mp3 format
    - Raises: AudioError on ffmpeg failure
    """
    cmd = [
        "ffmpeg",
        "-y",                 # overwrite destination without prompting
        "-i", "pipe:0",
        "-ac", "1",           # mono
        "-ar", "44100",       # sample rate
        "-af", "loudnorm",    # loudness normalization filter
        "-c:a", "libmp3lame", # mp3 codec
        str(dst)
    ]

    result = runner(cmd, input=raw, capture_output=True)

    if result.returncode != 0:
        stderr_msg = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
        raise AudioError(f"ffmpeg failed with code {result.returncode}: {stderr_msg}")

def duration_ok(path: Path, lo=0.2, hi=5.0, runner=subprocess.run) -> bool:
    """
    Check if audio duration is within acceptable range using ffprobe.
    Returns True if duration is between lo and hi seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path)
    ]

    result = runner(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return False

    try:
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        return lo <= duration <= hi
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
