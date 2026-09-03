"""Image downloads go through the standalone `imgfetch` binary
(tools/mediafetch, cmd/imgfetch) so the firewall can whitelist that one
executable for arbitrary image hosts without opening the generator itself."""
import subprocess
import tempfile
from pathlib import Path


class ImgFetchUnavailable(Exception):
    """The imgfetch binary could not be executed (not installed / wrong path)."""


class ImgFetch:
    def __init__(self, binary: str, runner=subprocess.run, timeout: int = 60):
        self.binary, self.runner, self.timeout = binary, runner, timeout

    def fetch(self, url: str) -> bytes | None:
        """Bytes of the validated image, or None when imgfetch refused it."""
        with tempfile.TemporaryDirectory(prefix="imgfetch-") as tmp:
            out = Path(tmp) / "image"
            cmd = [self.binary, url, str(out)]
            try:
                r = self.runner(cmd, capture_output=True, text=True, timeout=self.timeout)
            except OSError as exc:
                raise ImgFetchUnavailable(f"imgfetch not found at {self.binary}: {exc}") from exc
            except subprocess.TimeoutExpired:
                return None
            if r.returncode != 0 or not out.is_file():
                return None
            return out.read_bytes()
