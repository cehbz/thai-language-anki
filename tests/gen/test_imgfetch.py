from pathlib import Path

import pytest

from thai_deck_gen.media.imgfetch import ImgFetch, ImgFetchUnavailable


class _Run:
    """Fake subprocess.run: writes `payload` to the out path on success."""
    def __init__(self, returncode=0, payload=b"PNGBYTES", raise_os=False):
        self.returncode, self.payload, self.raise_os = returncode, payload, raise_os
        self.cmds = []

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        if self.raise_os:
            raise FileNotFoundError(cmd[0])
        if self.returncode == 0:
            Path(cmd[-1]).write_bytes(self.payload)
        class R:
            returncode = self.returncode
            stdout = '{"format":"png","width":2,"height":2,"bytes":8}\n'
            stderr = "" if self.returncode == 0 else "imgfetch: refused: not an image"
        return R()


def test_imgfetch_returns_bytes_and_cleans_up():
    run = _Run()
    data = ImgFetch("/usr/local/bin/imgfetch", runner=run).fetch("https://x/y.png")
    assert data == b"PNGBYTES"
    cmd = run.cmds[0]
    assert cmd[0] == "/usr/local/bin/imgfetch" and cmd[-2] == "https://x/y.png"
    assert not Path(cmd[-1]).exists()          # temp output removed


def test_imgfetch_returns_none_when_refused():
    run = _Run(returncode=1)
    assert ImgFetch("imgfetch", runner=run).fetch("https://x/y") is None


def test_imgfetch_missing_binary_raises_unavailable():
    with pytest.raises(ImgFetchUnavailable, match="imgfetch"):
        ImgFetch("imgfetch", runner=_Run(raise_os=True)).fetch("https://x/y")
