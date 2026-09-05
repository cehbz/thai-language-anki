"""The return path (spec 4 section 4): revlog import, flag import, and
ReviewNote harvest -- one command, one report, all reading a real
collection.anki2 directly and read-only (the proven pattern:
scripts/proof_gallery.py's sqlite reads, generalized here to the
notes/cards/revlog shape). `import_collection`'s `collection_path`
parameter is always the caller's to supply -- never hardcoded to
~/Library/.../collection.anki2 (the actual location on a real machine),
so tests and any future caller point it at whatever collection they mean.

Card identity -> (family, anchor, card_key, compile_id) is read from
compile.py's own tags/CompileId convention (see compile.py's module
docstring for the tag shapes this depends on: family::, word::/pair::/
grapheme::/target::/sentence::/member::/speaker::, CompileId as a note
field). Every tag is atomic (one part per tag); an anchor spanning
several tags (a pair member's MemberKey, a sentence's target+sha) is
built by composing those tags' values, never by parsing one tag's value
into parts, and never by reading MemberKey or any other note field back.
`card_key`'s KIND component (Listening/Production/.../Cloze) is read
from the card's own template name via the collection's `col.models`
JSON and the card's `ord` -- NOT parsed out of a tag -- because Anki
tags are a NOTE-level property shared by every sibling card, so a
single `kind::` tag cannot disambiguate which of several sibling cards
a given review or flag belongs to; the template name is unambiguous and
already present in the collection compile.py wrote. (The `kind::` tags
compile.py DOES emit are for the Anki browser's own tag-based search/
filtering, not for this module's identity reconstruction.)

Revlog idempotence (spec 4 section 4, "idempotent by (card_key, ts)"):
`ts` is the revlog row's OWN id (Anki's epoch-ms review timestamp,
already unique per review) -- store.py's `append_study(ts=...)` stores it
verbatim rather than through the cache table's collision-avoiding
`_next_ts` bump (see its docstring). The `study` table's primary key is
(card_key, ts); `append_study` is insert-or-ignore against that key and
reports whether it inserted, so a reimport's duplicate rows are detected
by the store, not by a read-then-write check here.

Flag import (spec 4 section 4, "role from the card kind"): the card-kind
-> Assessor role mapping is this module's own resolution -- spec 3's
AUTHORITY_ORDER (authority.py) names roles by WHAT is being judged
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
lands as a `{"kind": "reverify", ...}` row: no `"value"` key
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

from .cachekeys import LearnerKey, LearnerNoteKey, ReverifyKey, sha
from .store import SyllabusDb

__all__ = ["ImportReport", "card_identities", "import_collection"]

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


def _word_anchor(tags: list[str]) -> tuple[str, dict[str, str]] | None:
    word_id = _tag_value(tags, "word")
    if word_id is None:
        return None
    return word_id, {"word_id": word_id}


def _pair_anchor(tags: list[str]) -> tuple[str, dict[str, str]] | None:
    # The anchor is the member's MemberKey, composed from three atomic
    # tags -- never read back from the note's own MemberKey field.
    pair_id = _tag_value(tags, "pair")
    speaker_id = _tag_value(tags, "speaker")
    member_index = _tag_value(tags, "member")
    if pair_id is None or speaker_id is None or member_index is None:
        return None
    anchor = f"{pair_id}:{speaker_id}:{member_index}"
    return anchor, {"pair_id": pair_id, "speaker_id": speaker_id,
                    "member_index": member_index}


def _grapheme_anchor(tags: list[str]) -> tuple[str, dict[str, str]] | None:
    symbol = _tag_value(tags, "grapheme")
    if symbol is None:
        return None
    return symbol, {"grapheme_symbol": symbol}


def _sentence_anchor(tags: list[str]) -> tuple[str, dict[str, str]] | None:
    target_id = _tag_value(tags, "target")
    sentence_sha = _tag_value(tags, "sentence")
    if target_id is None or sentence_sha is None:
        return None
    anchor = f"{target_id}:{sentence_sha}"
    return anchor, {"target_id": target_id, "sentence_sha": sentence_sha}


_ANCHOR_BUILDERS: dict[str, Any] = {
    "word": _word_anchor,
    "minimal_pair": _pair_anchor,
    "grapheme": _grapheme_anchor,
    "sentence": _sentence_anchor,
}


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
    # Anchor parts, kept alongside the composed `anchor` string so a
    # future column-writing importer never re-parses it; populated per
    # family (see _word_anchor/_pair_anchor/_grapheme_anchor/
    # _sentence_anchor).
    word_id: str | None = None
    pair_id: str | None = None
    speaker_id: str | None = None
    member_index: str | None = None
    grapheme_symbol: str | None = None
    target_id: str | None = None
    sentence_sha: str | None = None


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
    builder = _ANCHOR_BUILDERS.get(family)
    built = builder(tags) if builder is not None else None
    if built is None:
        return None
    anchor, parts = built
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
                         note_id=card["nid"], **parts)


def card_identities(collection_path: str | Path) -> list[_CardIdentity]:
    """Every card's identity in `collection_path`, read read-only."""
    conn = _connect_readonly(collection_path)
    try:
        col = _load_collection(conn)
    finally:
        conn.close()
    return [identity for identity in
           (_identify_card(col, card_id) for card_id in col.cards)
           if identity is not None]


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
        inserted = db.append_study(card_key=identity.card_key, compile_id=identity.compile_id,
                                   grade=int(ease), time_ms=int(time_ms), ts=int(rev_id))
        if not inserted:
            skipped += 1
            skips.append(("revlog", f"{identity.card_key}@{rev_id}", "skipped: already present"))
            continue
        imported += 1
    return imported, skipped


# --- flag import -----------------------------------------------------------

def _flag_role(kind_slug: str) -> tuple[str, str | None]:
    """-> (role, provider-kind-for-current_best-lookup-or-None)."""
    return _ARTIFACT_ROLE_BY_KIND.get(kind_slug, ("card-flag", None))


# family -> the _CardIdentity field naming that family's own ENTITY
# subject: a word's id, a pair's id, a grapheme's symbol, a sentence's
# text_sha. This is what rendition rows, current_best, and compile's own
# audio/picture resolution are all keyed on -- never a per-card anchor
# (a pair member's MemberKey names one card, not the pair the learner's
# flag is about).
_ENTITY_SUBJECT_FIELD: dict[str, str] = {
    "word": "word_id",
    "minimal_pair": "pair_id",
    "grapheme": "grapheme_symbol",
    "sentence": "sentence_sha",
}


def _entity_subject(identity: _CardIdentity) -> str | None:
    field_name = _ENTITY_SUBJECT_FIELD.get(identity.family)
    return getattr(identity, field_name) if field_name is not None else None


def _import_flags(col: _Collection, db: SyllabusDb,
                  skips: list[tuple[str, str, str]]) -> tuple[int, int]:
    imported = 0
    skipped = 0
    for card_id, card in col.cards.items():
        if not card["flags"]:
            continue
        identity = _identify_card(col, card_id)
        subject = _entity_subject(identity) if identity is not None else None
        if subject is None:
            skipped += 1
            skips.append(("flag", str(card_id), "card not recognized"))
            continue
        role, provide_kind = _flag_role(identity.kind_slug)
        artifact_sha = None
        if provide_kind is not None:
            from .derivations import current_best
            artifact_sha = current_best(db, subject, provide_kind,
                                        current_rubric={}, prior=(),
                                        provenance_source=lambda s: None).artifact_sha

        existing_key = f"flag-import:{card_id}:{card['flags']}"
        if role in TONE_CORRECTNESS_ROLES:
            key = ReverifyKey(artifact_sha=artifact_sha, anchor=subject, role=role)
        else:
            key = LearnerKey(artifact_sha=artifact_sha, role=role)

        already = db.latest("assess", "learner", existing_key)
        if already is not None:
            skipped += 1
            skips.append(("flag", f"card {card_id}", "already imported (same flags value)"))
            continue

        if role in TONE_CORRECTNESS_ROLES:
            question = {"role": role, "artifact_sha": artifact_sha,
                       "kind": "reverify", "flag_import_key": existing_key}
            answer = {"flagged": True, "flag": card["flags"]}
        else:
            question = {"role": role, "artifact_sha": artifact_sha,
                       "kind": "rating", "flag_import_key": existing_key}
            answer = {"value": "unacceptable-none", "flag": card["flags"]}
        db.append(port="assess", backend="learner", key=key, subject=subject,
                 question=question, answer=answer)
        # A second row under `existing_key` records "this exact flags
        # value on this card has been imported", the idempotence marker
        # `_import_flags` checks above -- kept distinct from the rating/
        # reverify row itself (whose key must stay the readable
        # learner:ARTIFACT:ROLE shape derivations.py folds over).
        db.append(port="assess", backend="learner", key=existing_key, subject=subject,
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
        text_sha = sha(text)
        key = LearnerNoteKey(anchor=str(note_id), text_sha=text_sha)
        already = db.latest("assess", "learner-note", key)
        if already is not None:
            skipped += 1
            skips.append(("review_note", f"note {note_id}", "already harvested (unchanged text)"))
            continue
        db.append(port="assess", backend="learner-note", key=key, subject=str(note_id),
                 question={"note_id": note_id, "text_sha": text_sha},
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
