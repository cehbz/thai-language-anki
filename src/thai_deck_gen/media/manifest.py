from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict

Channel = Literal["thai1000", "forvo", "commissioned", "tts",
                  "openverse", "pexels", "wikimedia", "ai", "manual"]

class MediaEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str            # deck-relative, e.g. media/audio/picture_words/pw-1.mp3
    channel: Channel
    origin: str          # URL, order id, or source path
    speaker: str | None = None
    license: str | None = None
    fetched: str         # ISO date, passed in by caller (no clock reads in lib code)

class Manifest:
    def __init__(self, deck_root: Path | None = None):
        self.entries: dict[str, MediaEntry] = {}
        # set by load(): every record() then writes through, so a media file
        # on disk always has its provenance on disk too, even if the process
        # dies inside a filler
        self.deck_root = Path(deck_root) if deck_root is not None else None

    @classmethod
    def load(cls, deck_root: Path) -> "Manifest":
        """Load manifest from deck_root/media_manifest.yaml. Missing file = empty."""
        manifest = cls(deck_root)
        manifest_path = Path(deck_root) / "media_manifest.yaml"
        if not manifest_path.exists():
            return manifest

        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
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

        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    def record(self, entry: MediaEntry) -> None:
        """Record a media entry, persisting it when the deck root is known."""
        self.entries[entry.file] = entry
        if self.deck_root is not None:
            self._append(entry, self.deck_root)

    def _append(self, entry: MediaEntry, deck_root: Path) -> None:
        """Append one entry to the file. load() is last-wins, so an appended
        entry supersedes an earlier one for the same file and save() compacts."""
        path = Path(deck_root) / "media_manifest.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        item = yaml.safe_dump([entry.model_dump(exclude_none=True)],
                              allow_unicode=True, sort_keys=False)
        with path.open("a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("entries:\n")
            f.write(item)

    def channel_of(self, file: str) -> Channel | None:
        """Get the channel for a file, or None if not recorded."""
        entry = self.entries.get(file)
        return entry.channel if entry else None
