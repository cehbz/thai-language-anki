"""Synthetic .apkg builder/reader for tests.

Mirrors the on-disk shape scripts/import_apkg.py depends on: a zip
containing collection.anki2 (sqlite: col/notes/cards), a `media` JSON
member mapping zip-member index (as a string) to the real media filename,
and the raw numbered media members themselves.
"""
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path

MODEL_ID = 1
DECK_ID = 1

FIELD_NAMES = ["word_eng", "word_tha", "word_phonetic", "audio", "form",
              "sentence_tha", "sentence_phonetic", "sentence_eng"]

def build_apkg(path: Path, notes: list[list[str]], media: dict[str, bytes],
              field_names: list[str] = FIELD_NAMES, crt: int = 1700000000) -> None:
    """Write a synthetic .apkg zip at `path`.

    notes: list of field-value lists, one per note, in field_names order.
    media: real filename -> bytes; referenced from a note's audio field
           via "[sound:<filename>]".
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "collection.anki2"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript("""
                create table col (
                    id integer primary key, crt integer, mod integer,
                    scm integer, ver integer, dty integer, usn integer,
                    ls integer, conf text, models text, decks text,
                    dconf text, tags text);
                create table notes (
                    id integer primary key, guid text, mid integer,
                    mod integer, usn integer, tags text, flds text,
                    sfld text, csum integer, flags integer, data text);
                create table cards (
                    id integer primary key, nid integer, did integer,
                    ord integer, mod integer, usn integer, type integer,
                    queue integer, due integer, ivl integer, factor integer,
                    reps integer, lapses integer, left integer,
                    odue integer, odid integer, flags integer, data text);
            """)
            model = {str(MODEL_ID): {
                "id": MODEL_ID, "name": "synthetic",
                "flds": [{"name": n, "ord": i} for i, n in enumerate(field_names)],
                "tmpls": [{"name": "Card 1", "ord": 0}],
                "did": DECK_ID,
            }}
            deck = {str(DECK_ID): {"id": DECK_ID, "name": "synthetic"}}
            conn.execute(
                "insert into col values (1,?,?,?,11,0,0,0,'{}',?,?,'{}','{}')",
                (crt, crt, crt, json.dumps(model), json.dumps(deck)))

            for i, flds in enumerate(notes, start=1):
                joined = "\x1f".join(flds)
                conn.execute(
                    "insert into notes values (?,?,?,?,0,'',?,?,0,0,'')",
                    (i, f"guid{i}", MODEL_ID, crt, joined, flds[0] if flds else ""))
                conn.execute(
                    "insert into cards values "
                    "(?,?,?,0,?,0,0,0,?,0,0,0,0,0,0,0,0,'')",
                    (i, i, DECK_ID, crt, i))
            conn.commit()
        finally:
            conn.close()

        names = list(media)
        with zipfile.ZipFile(path, "w") as zf:
            zf.write(db_path, "collection.anki2")
            zf.writestr("media", json.dumps({str(i): n for i, n in enumerate(names)}))
            for i, name in enumerate(names):
                zf.writestr(str(i), media[name])

def read_apkg(path: Path) -> dict:
    """Read notes/cards/models/media back out of a (real or synthetic) .apkg."""
    with zipfile.ZipFile(path) as zf:
        media_map: dict[str, str] = json.loads(zf.read("media").decode())
        media = {name: zf.read(idx) for idx, name in media_map.items()}
        db_bytes = zf.read("collection.anki2")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "collection.anki2"
        db_path.write_bytes(db_bytes)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            notes = [dict(r) for r in conn.execute("select * from notes")]
            cards = [dict(r) for r in conn.execute("select * from cards")]
            (models_json,) = conn.execute("select models from col").fetchone()
        finally:
            conn.close()

    for n in notes:
        n["flds"] = n["flds"].split("\x1f")

    return {"notes": notes, "cards": cards,
            "models": json.loads(models_json), "media": media}
