import json
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.media.ffmpeg import normalize_audio
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import AudioNeed
from thai_deck_gen.producers import ProducerResult

_SOUND_TAG = re.compile(r"\[sound:(?P<name>[^\]]+)\]")

def audio_index(apkg: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(apkg) as zf:
        media_map: dict[str, str] = json.loads(zf.read("media").decode())
        db_bytes = zf.read("collection.anki2")
        reverse = {v: k for k, v in media_map.items()}

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "collection.anki2"
            db_path.write_bytes(db_bytes)
            conn = sqlite3.connect(str(db_path))
            try:
                rows = conn.execute("select flds from notes").fetchall()
            finally:
                conn.close()

        index: dict[str, bytes] = {}
        for (flds,) in rows:
            fields = flds.split("\x1f")
            if len(fields) < 4:
                continue
            word_tha, audio_field = fields[1], fields[3]
            m = _SOUND_TAG.search(audio_field)
            if not m:
                continue
            key = reverse.get(m.group("name"))
            if key is None:
                continue
            index[word_tha] = zf.read(key)
        return index

def _find_note(deck: Deck, need: AudioNeed):
    for family, note in deck.all_notes():
        if family == need.family and note.id == need.note_id:
            return note

def import_thai1000(needs: list[AudioNeed], deck: Deck, manifest: Manifest,
                    index: dict[str, bytes], today: str) -> ProducerResult:
    result = ProducerResult()
    for need in needs:
        if need.family == "minimal_pair":
            continue
        raw = index.get(need.text)
        if raw is None:
            continue

        note = _find_note(deck, need)
        dst = deck.root / "media" / need.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        normalize_audio(raw, dst)

        manifest.record(MediaEntry(
            file=f"media/{need.path}", channel="thai1000",
            origin=f"apkg:{need.text}", speaker="thai1000:main", fetched=today))
        note.audio.speaker = "thai1000:main"
        note.audio.source = "native"
        result.changed += 1

    return result
