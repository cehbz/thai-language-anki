"""Programmatic fixture decks. build() writes a valid golden mini-deck."""
from pathlib import Path
import yaml

def _aud(name, speaker="s1", source="native"):
    return {"file": f"audio/{name}", "source": source, "speaker": speaker}

GOLDEN = {
    "deck": {"name": "golden", "version": "0.1",
             "stage_plan": {"phases": ["sounds", "words", "sentences"]}},
    "minimal_pairs": [
        {"id": "mp-tone-1", "contrast": "tone", "members": [
            {"thai": "ขาว", "ipa": "kʰaːw˨˩˦", "audio": _aud("khao-r.mp3", "s1")},
            {"thai": "ข่าว", "ipa": "kʰaːw˨˩", "audio": _aud("khao-l.mp3", "s2")}]},
        {"id": "mp-asp-1", "contrast": "aspiration", "members": [
            {"thai": "ไก่", "ipa": "kaj˨˩", "audio": _aud("kai.mp3", "s1")},
            {"thai": "ไข่", "ipa": "kʰaj˨˩", "audio": _aud("khai.mp3", "s3")}]},
    ],
    "spelling_sound": [
        {"id": "ss-1", "pattern": "ข", "pattern_kind": "consonant",
         "consonant_class": "high", "example_word": "ไข่",
         "audio": _aud("khai.mp3"), "image": "images/egg.png"},
    ],
    "picture_words": [
        {"id": "w-dog", "thai": "หมา", "image": "images/dog.png",
         "audio": _aud("maa.mp3"), "frequency_rank": 120, "category": "Animals",
         "part_of_speech": "noun", "classifier": "ตัว", "ipa": "maː˨˩˦"},
        {"id": "w-come", "thai": "มา", "image": "images/come.png",
         "audio": _aud("maa2.mp3"), "frequency_rank": 15, "category": "Verbs",
         "part_of_speech": "verb", "ipa": "maː˧"},
        {"id": "w-rice", "thai": "ข้าว", "image": "images/rice.png",
         "audio": _aud("khao-f.mp3"), "frequency_rank": 90, "category": "Food",
         "part_of_speech": "noun", "classifier": "จาน", "ipa": "kʰaːw˥˩"},
    ],
    "sentences": [
        {"id": "s-1", "kind": "new_word", "thai": "หมามากินข้าว",
         "target": "กิน", "audio": _aud("s1.mp3"),
         "image": "images/eat.png", "definition": "เอาอาหารเข้าปาก"},
    ],
}

class DeckBuilder:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "deck"
        import copy
        self.data = copy.deepcopy(GOLDEN)

    def build(self) -> Path:
        notes = self.root / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        (self.root / "deck.yaml").write_text(
            yaml.safe_dump(self.data["deck"], allow_unicode=True))
        for fam in ("minimal_pairs", "spelling_sound", "picture_words", "sentences"):
            (notes / f"{fam}.yaml").write_text(
                yaml.safe_dump(self.data[fam], allow_unicode=True))
        self._write_media()
        return self.root

    def _write_media(self):
        for sub in ("audio", "images"):
            (self.root / "media" / sub).mkdir(parents=True, exist_ok=True)
        for ref in self._media_refs():
            p = self.root / "media" / ref
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00stub")

    def _media_refs(self):
        refs = []
        def walk(o):
            if isinstance(o, dict):
                if "file" in o and "source" in o:
                    refs.append(o["file"])
                for k, v in o.items():
                    if k == "image" and isinstance(v, str):
                        refs.append(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(self.data)
        return refs
