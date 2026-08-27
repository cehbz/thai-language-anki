import base64
import mimetypes
from pathlib import Path
from .core import JudgeRequest, Verdict, Verdicts
from .cli_judge import JudgeError
from ..config import JudgeConfig

class ApiJudge:
    """Judge backend that calls the Anthropic API directly.

    `client` supports constructor injection for tests (a fake with a
    `.messages.parse(**kwargs)` method); when omitted, a real
    `anthropic.Anthropic()` client is constructed lazily so importing this
    module (and the default test suite) never requires the `anthropic`
    package to be installed.
    """

    def __init__(self, config: JudgeConfig, client=None):
        self.config = config
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client

    def _content(self, req: JudgeRequest) -> list[dict]:
        content: list[dict] = []
        if req.image_path:
            path = Path(req.image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        content.append({"type": "text", "text": req.prompt})
        return content

    def judge(self, req: JudgeRequest) -> list[Verdict]:
        response = self.client.messages.parse(
            model=self.config.model,
            max_tokens=4096,
            output_config={"effort": self.config.effort},
            messages=[{"role": "user", "content": self._content(req)}],
            output_format=Verdicts,
        )
        if response.stop_reason == "refusal":
            raise JudgeError(f"API judge refused for note {req.note_id}")
        parsed = response.parsed_output
        if not isinstance(parsed, Verdicts):
            raise JudgeError(f"invalid API judge output for note {req.note_id}")
        return parsed.verdicts
