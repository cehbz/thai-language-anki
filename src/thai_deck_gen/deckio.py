import os, tempfile, yaml
from pathlib import Path
from thai_deck_eval.model.deck import _FAMILIES, Deck
from thai_deck_eval.model.notes import DeckMeta, StagePlan

def new_deck(root: Path, name: str, phases: list[str]) -> Deck:
    meta = DeckMeta(name=name, version="0.1",
                    stage_plan=StagePlan(phases=phases))
    return Deck(meta=meta, root=Path(root))

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)

def _dump(data) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

def write_deck(deck: Deck) -> None:
    _atomic_write(deck.root / "deck.yaml",
                  _dump(deck.meta.model_dump(exclude_none=True)))
    for attr, _fam, _model in _FAMILIES:
        notes = [n.model_dump(exclude_none=True)
                 for n in getattr(deck, attr)]
        _atomic_write(deck.root / "notes" / f"{attr}.yaml", _dump(notes))
    (deck.root / "media" / "audio").mkdir(parents=True, exist_ok=True)
    (deck.root / "media" / "images").mkdir(parents=True, exist_ok=True)
