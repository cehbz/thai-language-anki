import yaml
from pathlib import Path
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.ffmpeg import AudioError, normalize_audio
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import AudioNeed
from thai_deck_gen.producers import ProducerResult

_INSTRUCTIONS = (
    "Record in a quiet room, natural citation form, one word per file. "
    "Save as mp3 or m4a, named exactly <item id>.mp3 (or .m4a)."
)

def _item_id(need: AudioNeed) -> str:
    if need.member_index is None:
        return need.note_id
    return f"{need.note_id}_{need.member_index}"

def write_commission_batch(needs: list[AudioNeed], deck_root: Path) -> Path | None:
    if not needs:
        return None

    work_dir = Path(deck_root) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    nums = [int(p.stem.rsplit("_", 1)[1])
            for p in work_dir.glob("commission_batch_*.yaml")]
    n = max(nums, default=0) + 1
    path = work_dir / f"commission_batch_{n:03d}.yaml"

    data = {
        "instructions": _INSTRUCTIONS,
        "naming": "<item id>.mp3",
        "items": [{"id": _item_id(need), "thai": need.text,
                   "note_id": need.note_id, "family": need.family}
                  for need in needs],
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return path

def _target(deck: Deck, item: dict):
    for family, note in deck.all_notes():
        if family != item["family"] or note.id != item["note_id"]:
            continue
        if family == "minimal_pair":
            for member in note.members:
                if member.thai == item["thai"]:
                    return member
            return None
        return note
    return None

def import_commission(recordings_dir: Path, batch_file: Path, deck: Deck,
                      manifest: Manifest, speaker: str, today: str) -> ProducerResult:
    result = ProducerResult()
    batch = yaml.safe_load(Path(batch_file).read_text())
    recordings = {p.stem: p for p in Path(recordings_dir).iterdir() if p.is_file()}

    for item in batch["items"]:
        rec = recordings.get(item["id"])
        target = _target(deck, item)
        if rec is None or target is None:
            result.blocked.append(item["id"])
            continue

        try:
            dst = deck.root / "media" / target.audio.file
            dst.parent.mkdir(parents=True, exist_ok=True)
            normalize_audio(rec.read_bytes(), dst)

            manifest.record(MediaEntry(
                file=f"media/{target.audio.file}", channel="commissioned",
                origin=str(Path(batch_file).name),
                speaker=f"commissioned:{speaker}", fetched=today))
            target.audio.speaker = f"commissioned:{speaker}"
            target.audio.source = "native"
            result.changed += 1
        except AudioError:
            result.blocked.append(item["id"])

    return result
