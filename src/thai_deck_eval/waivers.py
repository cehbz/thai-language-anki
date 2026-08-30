"""Human overrides of findings the deck owner has reviewed and accepted.

Some findings are correct about the rule and wrong about the deck: a photo
of an expiry date really is a photo of text, and that is the point of the
card. A waiver records that decision against the exact artifact it was made
about -- replace the image and the waiver stops applying, so an approval
can never silently transfer to something nobody looked at.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

WAIVERS_FILE = "waivers.yaml"


@dataclass
class Waiver:
    note_id: str
    rule: str
    reason: str
    date: str
    sha: str | None = None      # of the image the decision was made about


def load_waivers(deck_root: Path) -> list[Waiver]:
    path = Path(deck_root) / WAIVERS_FILE
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Waiver(**item) for item in raw]


def save_waivers(deck_root: Path, waivers: list[Waiver]) -> None:
    path = Path(deck_root) / WAIVERS_FILE
    path.write_text(
        yaml.safe_dump([w.__dict__ for w in waivers], allow_unicode=True,
                       sort_keys=False),
        encoding="utf-8")


def image_sha(deck, note_id: str) -> str | None:
    """sha256 of a note's image, or None when it has none on disk."""
    for _family, note in deck.all_notes():
        if note.id != note_id:
            continue
        ref = getattr(note, "image", None)
        if not ref:
            return None
        path = Path(deck.root) / "media" / ref
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return None


def partition(findings, deck, waivers: list[Waiver]):
    """Split findings into (kept, waived, stale_waivers).

    A waiver with a `sha` applies only while the note's image still hashes to
    it; one that no longer matches is reported rather than ignored, so a
    waiver left behind by a re-fetch is visible instead of silently dead.
    """
    if not waivers:
        return list(findings), [], []

    by_key: dict[tuple[str, str], Waiver] = {
        (w.note_id, w.rule): w for w in waivers}
    kept, waived, stale = [], [], []
    sha_cache: dict[str, str | None] = {}
    for finding in findings:
        waiver = by_key.get((finding.note_id or "", finding.rule))
        if waiver is None:
            kept.append(finding)
            continue
        if waiver.sha is not None:
            if waiver.note_id not in sha_cache:
                sha_cache[waiver.note_id] = image_sha(deck, waiver.note_id)
            if sha_cache[waiver.note_id] != waiver.sha:
                kept.append(finding)
                stale.append(waiver)
                continue
        waived.append(finding)
    return kept, waived, stale
