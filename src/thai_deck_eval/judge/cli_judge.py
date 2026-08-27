import json
import re
import subprocess
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
        if req.image_path:
            prompt += f"\nImage file to inspect: {req.image_path}"
        for attempt in range(2):
            cmd = ["claude", "-p", prompt, "--output-format", "json"]
            if req.image_path:
                from pathlib import Path
                cmd += ["--allowedTools", "Read",
                        "--add-dir", str(Path(req.image_path).parent)]
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
