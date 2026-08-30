"""Judge backend over the Message Batches API.

Every card is an independent yes/no judgment, which is exactly the shape
the Batches API is priced for: one submission carries the whole deck at
half the per-token rate, and nothing needs a verdict back interactively.

A submitted batch is money already spent, so the batch id and its
custom-id map are written to disk before the first poll: a run killed
while waiting resumes that batch instead of paying for a second one.
"""

import base64
import json
import mimetypes
import re
import time
from pathlib import Path

from ..config import JudgeConfig
from .cli_judge import JudgeError
from .core import JudgeRequest, Verdict, Verdicts

TERMINAL = {"ended", "canceled", "expired"}


class BatchJudge:
    def __init__(self, config: JudgeConfig, client=None, state_path: Path | None = None,
                 poll_seconds: int = 30, sleep=time.sleep):
        self.config = config
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.state_path = Path(state_path) if state_path else None
        self.poll_seconds = poll_seconds
        self.sleep = sleep

    # --- Judge protocol ---------------------------------------------------

    def judge(self, req: JudgeRequest) -> list[Verdict]:
        return self.judge_many([req]).get(req.note_id, [])

    def judge_many(self, reqs: list[JudgeRequest]) -> dict[str, list[Verdict]]:
        """Verdicts by note id. Items the API failed on are omitted, never
        fabricated, so an unjudged card stays uncached and retries."""
        out: dict[str, list[Verdict]] = {}
        if not reqs:
            return out

        resumed = self._resume()
        if resumed is not None:
            batch_id, ids = resumed
            print(f"  judge: resuming in-flight batch {batch_id}")
            out.update(self._harvest(batch_id, ids))
            reqs = [r for r in reqs if r.note_id not in out]
            if not reqs:
                return out

        ids = {f"n{i}": r.note_id for i, r in enumerate(reqs)}
        payload = [self._request(cid, r) for cid, r in zip(ids, reqs)]
        try:
            batch = self.client.messages.batches.create(requests=payload)
        except Exception as exc:
            raise JudgeError(f"batch submit failed: {exc}") from exc

        self._save(batch.id, ids)
        print(f"  judge: submitted batch {batch.id} ({len(payload)} card(s))")
        out.update(self._harvest(batch.id, ids))
        return out

    # --- batch plumbing ---------------------------------------------------

    def _request(self, custom_id: str, req: JudgeRequest) -> dict:
        content: list[dict] = []
        if req.image_path and Path(req.image_path).is_file():
            path = Path(req.image_path)
            media_type = mimetypes.guess_type(path.name)[0] or "image/png"
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type,
                           "data": base64.standard_b64encode(
                               path.read_bytes()).decode("utf-8")},
            })
        content.append({"type": "text", "text": req.prompt})
        return {
            "custom_id": custom_id,
            "params": {
                "model": self.config.model,
                "max_tokens": 4096,
                "output_config": {"effort": self.config.effort},
                "messages": [{"role": "user", "content": content}],
            },
        }

    def _harvest(self, batch_id: str, ids: dict[str, str]) -> dict[str, list[Verdict]]:
        self._await(batch_id)
        out: dict[str, list[Verdict]] = {}
        failed = 0
        try:
            entries = self.client.messages.batches.results(batch_id)
        except Exception as exc:
            raise JudgeError(f"batch {batch_id} results failed: {exc}") from exc
        for entry in entries:
            note_id = ids.get(entry.custom_id)
            if note_id is None:
                continue
            verdicts = self._verdicts(entry)
            if verdicts is None:
                failed += 1
                continue
            out[note_id] = verdicts
        if failed:
            print(f"  judge: {failed} card(s) returned no usable verdict; "
                  "they stay uncached and retry on the next run")
        self._clear()
        return out

    def _await(self, batch_id: str) -> None:
        while True:
            try:
                status = self.client.messages.batches.retrieve(batch_id).processing_status
            except Exception as exc:
                raise JudgeError(f"batch {batch_id} poll failed: {exc}") from exc
            if status in TERMINAL:
                return
            self.sleep(self.poll_seconds)

    @staticmethod
    def _verdicts(entry) -> list[Verdict] | None:
        result = entry.result
        if getattr(result, "type", None) != "succeeded":
            return None
        message = result.message
        if getattr(message, "stop_reason", None) == "refusal":
            return None
        text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return Verdicts.model_validate_json(match.group(0)).verdicts
        except Exception:
            return None

    # --- resume state -----------------------------------------------------

    def _resume(self) -> tuple[str, dict[str, str]] | None:
        if not self.state_path or not self.state_path.exists():
            return None
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return state["batch_id"], state["ids"]
        except (json.JSONDecodeError, KeyError):
            return None

    def _save(self, batch_id: str, ids: dict[str, str]) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"batch_id": batch_id, "ids": ids}, ensure_ascii=False),
            encoding="utf-8")

    def _clear(self) -> None:
        if self.state_path and self.state_path.exists():
            self.state_path.unlink()
