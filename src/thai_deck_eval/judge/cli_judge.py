import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from .core import JudgeRequest, Verdict, Verdicts
from ..config import JudgeConfig

class JudgeError(Exception):
    pass

class CliJudge:
    def __init__(self, config: JudgeConfig, runner=subprocess.run):
        self.config = config
        self.runner = runner

    def judge(self, req: JudgeRequest) -> list[Verdict]:
        prompt = req.prompt
        image_dir = None
        try:
            if req.image_path:
                # Scope --add-dir to a fresh temp dir holding ONLY a copy of
                # this note's image, never the deck's whole media directory
                # (which would let the judge read every other note's image).
                image_dir = tempfile.mkdtemp(prefix="thai-deck-eval-judge-")
                src = Path(req.image_path)
                dst = Path(image_dir) / src.name
                shutil.copy(src, dst)
                prompt += f"\nImage file to inspect: {dst}"
            for attempt in range(2):
                cmd = ["claude", "-p", prompt, "--output-format", "json"]
                if image_dir:
                    cmd += ["--allowedTools", "Read", "--add-dir", image_dir]
                try:
                    r = self.runner(cmd, capture_output=True, text=True, timeout=600)
                except (subprocess.TimeoutExpired, OSError) as exc:
                    raise JudgeError(
                        f"claude -p failed for note {req.note_id}: {exc}") from exc
                if r.returncode != 0:
                    raise JudgeError(f"claude -p failed: {r.stderr[:500]}")
                try:
                    text = json.loads(r.stdout)["result"]
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if not m:
                        raise ValueError("no JSON object in result")
                    return Verdicts.model_validate_json(m.group(0)).verdicts
                except Exception:
                    prompt += "\nReturn ONLY the JSON object."
            raise JudgeError(f"unparseable judge output for note {req.note_id}")
        finally:
            if image_dir is not None:
                shutil.rmtree(image_dir, ignore_errors=True)
