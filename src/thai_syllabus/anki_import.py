"""The return path (spec 4 section 4): revlog import, flag import, and
ReviewNote harvest -- one command, one report, all reading a real
collection.anki2 directly and read-only (the proven pattern:
scripts/proof_gallery.py's sqlite reads, generalized here to the
notes/cards/revlog shape). `import_collection`'s `collection_path`
parameter is always the caller's to supply -- never hardcoded to
~/Library/.../collection.anki2 (the actual location on a real machine),
so tests and any future caller point it at whatever collection they mean.

Card identity -> (family, card_key, compile_id) is read from compile.py's
own tags/CompileId convention (see compile.py's module docstring for the
tag shapes this depends on: family::, target::/pair::/grapheme::/
sentence::, CompileId as a note field). `card_key`'s KIND component
(Listening/Production/.../Cloze) is read from the card's own template
name via the collection's `col.models` JSON and the card's `ord` --
NOT parsed out of a tag -- because Anki tags are a NOTE-level property
shared by every sibling card, so a single `kind::` tag cannot
disambiguate which of several sibling cards a given review or flag
belongs to; the template name is unambiguous and already present in the
collection compile.py wrote. (The `kind::` tags compile.py DOES emit are
for the Anki browser's own tag-based search/filtering, not for this
module's identity reconstruction.)

Revlog idempotence (spec 4 section 4, "idempotent by (card_key, ts)"):
`ts` is the revlog row's OWN id (Anki's epoch-ms review timestamp,
already unique per review) -- store.py's `append_study(ts=...)` stores it
verbatim rather than through the cache table's collision-avoiding
`_next_ts` bump (see its docstring). Before each insert this module
checks `db.records(card_key)` for an existing row at that exact ts and
skips if found, since the `study` table itself carries no uniqueness
constraint of its own (spec 2 keeps it a plain append-only table); the
check is done here, at the application layer, rather than by adding a
schema constraint spec 2 doesn't otherwise call for.

Flag import (spec 4 section 4, "role from the card kind"): the card-kind
-> Assessor role mapping is this module's own resolution -- spec 3's
AUTHORITY_ORDER (assessor.py) names roles by WHAT is being judged
(picture-for-word, recording-for-word, ...), not by card-template name,
and nothing in specs 1-4 gives an exhaustive table from one to the other.
Only the two word templates whose FRONT is unambiguously one specific
artifact map to that artifact's role (Listening's front is the
recording -> "recording-for-word"; Production's front is the picture ->
"picture-for-word"); every other flagged card kind (Reading, Spelling,
Recognition, Cloze, sentence Listening) maps to the generic "card-flag"
role, which AUTHORITY_ORDER already lists as learner-authoritative. A
flag's COLOR carries no defined meaning anywhere in specs 1-4 (Anki
flags are just seven colors with project-specific meaning, undefined
here), so any non-zero flag is read as one undifferentiated "the learner
marked this" signal.
"recording-for-word" is spec 3's tone-correctness-adjacent role (its
AUTHORITY_ORDER row: `("mechanical", "judge")`, with the module comment
"the learner ... unqualified on tone correctness" -- exactly spec 4
section 4's "a flag on a tone-correctness role"): a flag there is
therefore NOT written as a normal learner rating (which
derivations.current_best always treats as authoritative outright,
regardless of AUTHORITY_ORDER -- writing one would let an unqualified
flag silently override a mechanically-verified recording). Instead it
lands as a `{"kind": "reverify-request", ...}` row: no `"value"` key
recognized by `derivations.LEARNER_RANK`, so it is invisible to
current_best's fold and exists purely as a signal a future
judge/mechanical run can query for and act on -- "queues machine
re-verification instead of overriding" (spec 4 section 4), verbatim.
Every other role writes a normal learner rating,
`{"value": "unacceptable-none"}` (spec 3's
current_best/LEARNER_RANK vocabulary) -- the conservative "something's
wrong, no known-good replacement yet" reading of an undifferentiated
flag, on the role/subject the learner IS authoritative for.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cachekeys import sha
from .store import SyllabusDb

__all__ = ["ImportReport", "import_collection"]

# card kind (template name, lowercased) -> Assessor role, for the two
# templates whose front is unambiguously one specific artifact; every
# other kind falls back to the generic "card-flag" role.
_ARTIFACT_ROLE_BY_KIND: dict[str, tuple[str, str]] = {
    # kind_slug -> (role, provider `kind` for current_best lookup)
    "listening": ("recording-for-word", "recording"),
    "production": ("picture-for-word", "picture"),
}
TONE_CORRECTNESS_ROLES = frozenset({"recording-for-word"})


@dataclass(frozen=True)
class ImportReport:
    revlog_imported: int = 0
    revlog_skipped: int = 0
    flags_imported: int = 0
    flags_skipped: int = 0
    notes_harvested: int = 0
    notes_skipped: int = 0
    skips: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    # (kind, identity, reason) -- kind in {"revlog", "flag", "review_note"}


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).resolve()
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def _tag_value(tags: list[str], prefix: str) -> str | None:
    needle = prefix + "::"
    for t in tags:
        if t.startswith(needle):
            return t[len(needle):]
    return None


def _anchor_for(family: str, tags: list[str]) -> str | None:
    if family == "word":
        # target:: on a word note holds the WORD id, not a Target id (a
        # word note aggregates every Target the word has into one note --
        # see compile.py's module docstring, "card_key's word/pair
        # anchor").
        return _tag_value(tags, "target")
    if family == "minimal_pair":
        return _tag_value(tags, "pair")
    if family == "grapheme":
        return _tag_value(tags, "grapheme")
    if family == "sentence":
        target = _tag_value(tags, "target")
        text_sha = _tag_value(tags, "sentence")
        if target is None or text_sha is None:
            return None
        return f"{target}:{text_sha}"
    return None


@dataclass(frozen=True)
class _Collection:
    models: dict[str, Any]
    notes: dict[int, dict[str, Any]]     # note id -> {mid, flds, tags}
    cards: dict[int, dict[str, Any]]     # card id -> {nid, ord, flags}


def _load_collection(conn: sqlite3.Connection) -> _Collection:
    (models_json,) = conn.execute("select models from col").fetchone()
    models = json.loads(models_json)
    notes: dict[int, dict[str, Any]] = {}
    for nid, mid, flds, tags in conn.execute("select id, mid, flds, tags from notes"):
        notes[nid] = {"mid": mid, "flds": flds.split("\x1f"),
                     "tags": [t for t in tags.split(" ") if t]}
    cards: dict[int, dict[str, Any]] = {}
    for cid, nid, ord_, flags_ in conn.execute("select id, nid, ord, flags from cards"):
        cards[cid] = {"nid": nid, "ord": ord_, "flags": flags_}
    return _Collection(models=models, notes=notes, cards=cards)


def _field_index(model: dict, name: str) -> int | None:
    for f in model["flds"]:
        if f["name"] == name:
            return f["ord"]
    return None


@dataclass(frozen=True)
class _CardIdentity:
    family: str
    anchor: str
    kind_slug: str
    card_key: str
    compile_id: str
    note_id: int


def _identify_card(col: _Collection, card_id: int) -> _CardIdentity | None:
    card = col.cards.get(card_id)
    if card is None:
        return None
    note = col.notes.get(card["nid"])
    if note is None:
        return None
    model = col.models.get(str(note["mid"]))
    if model is None:
        return None
    tags = note["tags"]
    family = _tag_value(tags, "family")
    if family is None:
        return None
    anchor = _anchor_for(family, tags)
    if anchor is None:
        return None
    tmpls = model["tmpls"]
    ord_ = card["ord"]
    if not (0 <= ord_ < len(tmpls)):
        return None
    kind_slug = tmpls[ord_]["name"].lower()
    card_key = f"{anchor}::{kind_slug}"
    compile_idx = _field_index(model, "CompileId")
    compile_id = note["flds"][compile_idx] if compile_idx is not None else ""
    return _CardIdentity(family=family, anchor=anchor, kind_slug=kind_slug,
                         card_key=card_key, compile_id=compile_id,
                         note_id=card["nid"])


# --- revlog import -------------------------------------------------------

def _import_revlog(conn: sqlite3.Connection, col: _Collection, db: SyllabusDb,
                   skips: list[tuple[str, str, str]]) -> tuple[int, int]:
    imported = 0
    skipped = 0
    rows = conn.execute("select id, cid, ease, time from revlog order by id").fetchall()
    for rev_id, card_id, ease, time_ms in rows:
        identity = _identify_card(col, card_id)
        if identity is None:
            skipped += 1
            skips.append(("revlog", str(card_id),
                          "card not recognized (no family:: tag, or model/template unknown)"))
            continue
        existing = db.records(identity.card_key)
        if any(r.ts == rev_id for r in existing):
            skipped += 1
            skips.append(("revlog", f"{identity.card_key}@{rev_id}", "already imported"))
            continue
        db.append_study(card_key=identity.card_key, compile_id=identity.compile_id,
                        grade=int(ease), time_ms=int(time_ms), ts=int(rev_id))
        imported += 1
    return imported, skipped


# --- flag import -----------------------------------------------------------

def _flag_role(kind_slug: str) -> tuple[str, str | None]:
    """-> (role, provider-kind-for-current_best-lookup-or-None)."""
    return _ARTIFACT_ROLE_BY_KIND.get(kind_slug, ("card-flag", None))


def _import_flags(col: _Collection, db: SyllabusDb,
                  skips: list[tuple[str, str, str]]) -> tuple[int, int]:
    imported = 0
    skipped = 0
    for card_id, card in col.cards.items():
        if not card["flags"]:
            continue
        identity = _identify_card(col, card_id)
        if identity is None:
            skipped += 1
            skips.append(("flag", str(card_id), "card not recognized"))
            continue
        role, provide_kind = _flag_role(identity.kind_slug)
        artifact_sha = None
        if provide_kind is not None:
            from .derivations import current_best
            artifact_sha = current_best(db, identity.anchor, provide_kind).artifact_sha

        if role in TONE_CORRECTNESS_ROLES:
            key = f"learner:reverify:{artifact_sha or identity.anchor}:{role}"
            existing_key = f"flag-import:{card_id}:{card['flags']}"
        else:
            key = f"learner:{artifact_sha or identity.anchor}:{role}"
            existing_key = f"flag-import:{card_id}:{card['flags']}"

        already = db.latest("assess", "learner", existing_key)
        if already is not None:
            skipped += 1
            skips.append(("flag", f"card {card_id}", "already imported (same flags value)"))
            continue

        if role in TONE_CORRECTNESS_ROLES:
            question = {"role": role, "artifact_sha": artifact_sha,
                       "kind": "reverify-request", "flag_import_key": existing_key}
            answer = {"flagged": True, "flag": card["flags"]}
        else:
            question = {"role": role, "artifact_sha": artifact_sha,
                       "flag_import_key": existing_key}
            answer = {"value": "unacceptable-none", "flag": card["flags"]}
        db.append(port="assess", backend="learner", key=key, subject=identity.anchor,
                 question=question, answer=answer)
        # A second row under `existing_key` records "this exact flags
        # value on this card has been imported", the idempotence marker
        # `_import_flags` checks above -- kept distinct from the rating/
        # reverify row itself (whose key must stay the readable
        # learner:ARTIFACT:ROLE shape derivations.py folds over).
        db.append(port="assess", backend="learner", key=existing_key, subject=identity.anchor,
                 question={"kind": "flag-import-marker", "card_id": card_id},
                 answer={"flags": card["flags"]})
        imported += 1
    return imported, skipped


# --- ReviewNote harvest ----------------------------------------------------

def _import_review_notes(col: _Collection, db: SyllabusDb,
                         skips: list[tuple[str, str, str]]) -> tuple[int, int]:
    imported = 0
    skipped = 0
    for note_id, note in col.notes.items():
        model = col.models.get(str(note["mid"]))
        if model is None:
            continue
        idx = _field_index(model, "ReviewNote")
        if idx is None:
            continue
        text = note["flds"][idx].strip()
        if not text:
            continue  # cleared/empty: appends nothing, retracts nothing
        key = f"learner-note:{note_id}:{sha(text)}"
        already = db.latest("assess", "learner-note", key)
        if already is not None:
            skipped += 1
            skips.append(("review_note", f"note {note_id}", "already harvested (unchanged text)"))
            continue
        db.append(port="assess", backend="learner-note", key=key, subject=str(note_id),
                 question={"note_id": note_id, "text_sha": sha(text)},
                 answer={"text": text})
        imported += 1
    return imported, skipped


# --- the one command -------------------------------------------------------

def import_collection(collection_path: str | Path, db: SyllabusDb) -> ImportReport:
    """Read `collection_path` (an Anki collection.anki2, or an extracted
    .apkg's own copy) read-only and import revlog rows, card flags, and
    ReviewNote text into `db`. One pass, one report.
    """
    conn = _connect_readonly(collection_path)
    try:
        col = _load_collection(conn)
        skips: list[tuple[str, str, str]] = []
        revlog_imported, revlog_skipped = _import_revlog(conn, col, db, skips)
        flags_imported, flags_skipped = _import_flags(col, db, skips)
        notes_harvested, notes_skipped = _import_review_notes(col, db, skips)
    finally:
        conn.close()
    return ImportReport(
        revlog_imported=revlog_imported, revlog_skipped=revlog_skipped,
        flags_imported=flags_imported, flags_skipped=flags_skipped,
        notes_harvested=notes_harvested, notes_skipped=notes_skipped,
        skips=tuple(skips))
