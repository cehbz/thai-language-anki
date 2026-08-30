"""Memoized Forvo word lookups.

The Forvo tier is a daily request quota, so a lookup is worth more than the
audio it returns: a word that Forvo doesn't have costs a request to learn,
and re-learning it tomorrow costs the same request again. Every lookup is
recorded here, hit or miss, and replayed on later runs.

Append-only JSONL, one record per lookup, last entry wins. Appending keeps
a killed run's answers (rewriting the whole file per lookup is quadratic,
the lesson the media manifest already learned).
"""

import json
from pathlib import Path

MEMO_PATH = Path("work") / "forvo_lookups.jsonl"


class ForvoMemo:
    """Word to pronunciation-list memo, persisted under the deck root."""

    def __init__(self, root: Path, entries: dict[str, dict] | None = None):
        self.root = Path(root)
        self.entries = entries or {}

    @classmethod
    def load(cls, root: Path) -> "ForvoMemo":
        path = Path(root) / MEMO_PATH
        entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a run killed mid-write leaves one torn line
                if "word" in rec:
                    entries[rec["word"]] = rec
        return cls(root, entries)

    def seen(self, word: str) -> bool:
        return word in self.entries

    def items(self, word: str) -> list[dict]:
        """Memoized pronunciations; empty for a memoized miss."""
        return self.entries.get(word, {}).get("items", [])

    def record(self, word: str, items: list[dict], fetched: str) -> None:
        rec = {"word": word, "items": items, "fetched": fetched}
        self.entries[word] = rec
        path = self.root / MEMO_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
