"""The AnkiConnect client (add-on 2055492159, API version 6): one POST of
{"action", "version", "params"} per call, answered {"result", "error"}.
AnkiConnect serves only while Anki is open; with no Origin header a local
caller needs no permission."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

API_VERSION = 6
DEFAULT_URL = "http://127.0.0.1:8765"

HttpPost = Callable[[str, bytes, float], bytes]


class AnkiDown(Exception):
    """AnkiConnect refused the connection: Anki is not running."""


class AnkiFailed(Exception):
    """AnkiConnect answered with an error, or did not answer."""


def urllib_post(url: str, body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


@dataclass(frozen=True)
class AnkiConnect:
    url: str = DEFAULT_URL
    post: HttpPost = urllib_post
    timeout: float = 120

    def call(self, action: str, **params: Any) -> Any:
        return self.call_within(self.timeout, action, **params)

    def call_within(self, timeout: float, action: str, **params: Any) -> Any:
        """One action's `result`, or AnkiDown / AnkiFailed."""
        request = {"action": action, "version": API_VERSION, "params": params}
        try:
            body = self.post(self.url, json.dumps(request).encode(), timeout)
        except OSError as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if isinstance(reason, ConnectionRefusedError):
                raise AnkiDown() from None
            raise AnkiFailed(f"AnkiConnect {action} at {self.url} did not answer: "
                             f"{reason}") from None
        try:
            answer = json.loads(body)
        except ValueError:
            raise AnkiFailed(f"AnkiConnect {action} answered non-JSON: {body[:200]!r}") from None
        if not isinstance(answer, dict):
            raise AnkiFailed(f"AnkiConnect {action} answered {answer!r}")
        if answer.get("error") is not None:
            raise AnkiFailed(f"AnkiConnect {action} error: {answer['error']}")
        return answer.get("result")
