from dataclasses import dataclass, field
from pathlib import Path
import yaml
from pydantic import ValidationError
from .notes import (DeckMeta, MinimalPairNote, PictureWordNote,
                    SentenceNote, SpellingSoundNote)

_FAMILIES = [
    ("minimal_pairs", "minimal_pair", MinimalPairNote),
    ("spelling_sound", "spelling_sound", SpellingSoundNote),
    ("picture_words", "picture_word", PictureWordNote),
    ("sentences", "sentence", SentenceNote),
]

class DeckSchemaError(Exception):
    def __init__(self, issues: list[str]):
        super().__init__(f"{len(issues)} schema issue(s)")
        self.issues = issues

@dataclass
class Deck:
    meta: DeckMeta
    root: Path
    minimal_pairs: list[MinimalPairNote] = field(default_factory=list)
    spelling_sound: list[SpellingSoundNote] = field(default_factory=list)
    picture_words: list[PictureWordNote] = field(default_factory=list)
    sentences: list[SentenceNote] = field(default_factory=list)

    def all_notes(self) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        for attr, fam, _ in _FAMILIES:
            out += [(fam, n) for n in getattr(self, attr)]
        return out

def load_deck(path: Path) -> Deck:
    issues: list[str] = []
    path = Path(path)
    try:
        meta = DeckMeta.model_validate(
            yaml.safe_load((path / "deck.yaml").read_text()))
    except (OSError, ValidationError, yaml.YAMLError) as e:
        raise DeckSchemaError([f"deck.yaml: {e}"])
    deck = Deck(meta=meta, root=path)
    for attr, _fam, model in _FAMILIES:
        fpath = path / "notes" / f"{attr}.yaml"
        try:
            raw = yaml.safe_load(fpath.read_text()) or []
        except (OSError, yaml.YAMLError) as e:
            issues.append(f"notes/{attr}.yaml: {e}")
            continue
        for entry in raw:
            note_id = entry.get("id", "?") if isinstance(entry, dict) else "?"
            try:
                getattr(deck, attr).append(model.model_validate(entry))
            except ValidationError as e:
                issues.append(f"notes/{attr}.yaml [{note_id}]: {e}")
    if issues:
        raise DeckSchemaError(issues)
    return deck
