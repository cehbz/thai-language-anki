from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict

Channel = Literal["thai1000", "forvo", "commissioned", "tts",
                  "openverse", "wikimedia", "ai", "manual"]

class MediaEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str            # deck-relative, e.g. media/audio/picture_words/pw-1.mp3
    channel: Channel
    origin: str          # URL, order id, or source path
    speaker: str | None = None
    license: str | None = None
    fetched: str         # ISO date, passed in by caller (no clock reads in lib code)

class Manifest:
    def __init__(self):
        self.entries: dict[str, MediaEntry] = {}

    @classmethod
    def load(cls, deck_root: Path) -> "Manifest":
        """Load manifest from deck_root/media_manifest.yaml. Missing file = empty."""
        manifest = cls()
        manifest_path = Path(deck_root) / "media_manifest.yaml"
        if not manifest_path.exists():
            return manifest

        data = yaml.safe_load(manifest_path.read_text())
        if data is None:
            return manifest

        for entry_data in data.get("entries", []):
            entry = MediaEntry.model_validate(entry_data)
            manifest.entries[entry.file] = entry

        return manifest

    def save(self, deck_root: Path) -> None:
        """Save manifest to deck_root/media_manifest.yaml"""
        manifest_path = Path(deck_root) / "media_manifest.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entries": [entry.model_dump(exclude_none=True) for entry in self.entries.values()]
        }

        with open(manifest_path, "w") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    def record(self, entry: MediaEntry) -> None:
        """Record a media entry in the manifest."""
        self.entries[entry.file] = entry

    def channel_of(self, file: str) -> Channel | None:
        """Get the channel for a file, or None if not recorded."""
        entry = self.entries.get(file)
        return entry.channel if entry else None
