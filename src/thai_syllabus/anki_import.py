"""The return path (spec 4 section 4): revlog import, flag import and
ReviewNote harvest over one caller-supplied collection.anki2, read-only
-- one command, one report.

Card identity -> (family, anchor, card kind, compile_id) comes from
compile.py's tag/CompileId convention: family::, word::/pair::/grapheme::/
target::/sentence::/member::/speaker:: tags, CompileId as a note field.
Every tag is atomic; a pair member's anchor (MemberKey) composes its three
tags' values, and a sentence note's target:: tags (one per filled target,
plural) come along as target_ids beside its own text_sha anchor -- the
sentence anchor itself is the sentence:: tag's value alone, matching the
note's guid. The card kind (Listening/Production/.../Cloze, lowered) is
the card's own template name, via `col.models` and the card's `ord`,
which names one sibling where a note-level `kind::` tag names them all.

Revlog import appends one `study` row per revlog entry, keyed (spec 2
section 2) by (family, anchor, card_kind, ts); `anchor` is the family's
entity id (word id, grapheme symbol, sentence text_sha), or a pair id for
family "minimal_pair" (the member's speaker/index go in the row's own
columns). `ts` is the revlog row's own id, stored verbatim, and
`append_study` is insert-or-ignore on that primary key.

Flag import: (family, card kind) resolves to a role through the two
tables below. A rating or card-flag row's idempotence key is a FlagKey
over (family, anchor, card_kind, flags), the card-and-flags fact itself.
A sentence card's flag anchor is its text_sha; a flag keyed under the
old target:sha shape re-imports once under the text_sha anchor.

ReviewNote harvest: each non-empty ReviewNote field appends a
learner-note row on the note's own entity subject, keyed by
LearnerNoteKey(anchor, sha(text)) -- re-harvesting unchanged text is a
key hit, edited text is a new key, a cleared field appends nothing.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .authority import role_for
from .cachekeys import FlagKey, LearnerNoteKey, ReverifyKey, sha
from .compile import card_kind_of
from .ports import StudyRecord
from .store import SyllabusDb

__all__ = ["ImportReport", "card_identities", "import_collection"]

# (family, card kind slug) -> the tone-correctness role its flag queues
# machine re-verification for (spec 4 section 4), as a {"kind":
# "reverify"} row under a ReverifyKey; current_best is left alone.
_TONE_ROLE: dict[tuple[str, str], str] = {
    ("word", "listening"): role_for("recording", "word"),
    ("sentence", "listening"): role_for("recording", "sentence"),
}

# (family, card kind slug) -> (role, provide kind) for a card whose front
# carries an artifact the learner is authoritative over: with a
# current-best artifact of that kind, the flag rates it under `role`
# ({"value": "unacceptable-none"}); with none, the flag is a card-flag.
_RATED_ROLE: dict[tuple[str, str], tuple[str, str]] = {
    ("word", "production"): (role_for("picture", "word"), "picture"),
    ("sentence", "cloze"): (role_for("picture", "sentence"), "picture"),
}


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


def _tag_values(tags: list[str], prefix: str) -> tuple[str, ...]:
    """Every tag's value for `prefix` (atomic, one part per tag) -- a
    sentence note carries one target:: tag per filled target.
    """
    needle = prefix + "::"
    return tuple(t[len(needle):] for t in tags if t.startswith(needle))


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


def _sentence_anchor(tags: list[str]) -> tuple[str, dict[str, Any]] | None:
    # One note per adopted Sentence (spec 4 section 1): the anchor is its
    # own text_sha, matching the note's guid; target_ids carries every
    # target:: tag the note fills (target-id order preserved from the
    # note's own tags).
    sentence_sha = _tag_value(tags, "sentence")
    target_ids = _tag_values(tags, "target")
    if sentence_sha is None or not target_ids:
        return None
    return sentence_sha, {"sentence_sha": sentence_sha, "target_ids": target_ids}


_ANCHOR_BUILDERS: dict[str, Any] = {
    "word": _word_anchor,
    "minimal_pair": _pair_anchor,
    "grapheme": _grapheme_anchor,
    "sentence": _sentence_anchor,
}

# family -> the _CardIdentity field naming that family's own ENTITY
# subject: a word's id, a pair's id, a grapheme's symbol, a sentence's
# text_sha. This is what rendition rows, current_best, compile's own
# audio/picture resolution, and study rows are all keyed on -- never a
# per-card anchor (a pair member's MemberKey names one card, not the pair
# a learner's flag or a study row's confusion grouping is about).
_ENTITY_SUBJECT_FIELD: dict[str, str] = {
    "word": "word_id",
    "minimal_pair": "pair_id",
    "grapheme": "grapheme_symbol",
    "sentence": "sentence_sha",
}


def _family_anchor_parts(tags: list[str]) -> tuple[str, str, dict[str, Any]] | None:
    family = _tag_value(tags, "family")
    if family is None:
        return None
    builder = _ANCHOR_BUILDERS.get(family)
    built = builder(tags) if builder is not None else None
    if built is None:
        return None
    anchor, parts = built
    return family, anchor, parts


def _note_subject(tags: list[str]) -> str | None:
    """The entity subject a note's own family/anchor tags name -- no
    card-level (template/ord) information needed.
    """
    resolved = _family_anchor_parts(tags)
    if resolved is None:
        return None
    family, _anchor, parts = resolved
    field_name = _ENTITY_SUBJECT_FIELD.get(family)
    return parts.get(field_name) if field_name is not None else None


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
    compile_id: str
    note_id: int
    # Anchor parts, kept alongside the composed `anchor` string so nothing
    # ever re-parses it (see _word_anchor/_pair_anchor/_grapheme_anchor/
    # _sentence_anchor).
    word_id: str | None = None
    pair_id: str | None = None
    speaker_id: str | None = None
    member_index: str | None = None
    grapheme_symbol: str | None = None
    target_ids: tuple[str, ...] = ()
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
    resolved = _family_anchor_parts(note["tags"])
    if resolved is None:
        return None
    family, anchor, parts = resolved
    tmpls = model["tmpls"]
    ord_ = card["ord"]
    if not (0 <= ord_ < len(tmpls)):
        return None
    kind_slug = card_kind_of(tmpls[ord_]["name"])
    compile_idx = _field_index(model, "CompileId")
    compile_id = note["flds"][compile_idx] if compile_idx is not None else ""
    return _CardIdentity(family=family, anchor=anchor, kind_slug=kind_slug,
                         compile_id=compile_id, note_id=card["nid"], **parts)


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


def _entity_subject(identity: _CardIdentity) -> str | None:
    field_name = _ENTITY_SUBJECT_FIELD.get(identity.family)
    return getattr(identity, field_name) if field_name is not None else None


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
        anchor = _entity_subject(identity)
        record = StudyRecord(family=identity.family, anchor=anchor,
                             card_kind=identity.kind_slug, compile_id=identity.compile_id,
                             ts=int(rev_id), grade=int(ease), time_ms=int(time_ms),
                             member_index=identity.member_index, speaker_id=identity.speaker_id)
        if not db.append_study(record):
            skipped += 1
            skips.append(("revlog", f"{identity.family}:{anchor}:{identity.kind_slug}@{rev_id}",
                          "skipped: already present"))
            continue
        imported += 1
    return imported, skipped


# --- flag import -----------------------------------------------------------

def _current_best_sha(db: SyllabusDb, subject: str, provide_kind: str, *,
                      current_rubric: Mapping[str, str], prior: Sequence[str],
                      provenance_source: Callable[[str], str | None]) -> str | None:
    from .derivations import current_best
    return current_best(db, subject, provide_kind, current_rubric=current_rubric, prior=prior,
                        provenance_source=provenance_source).artifact_sha


def _import_flags(col: _Collection, db: SyllabusDb, skips: list[tuple[str, str, str]], *,
                  current_rubric: Mapping[str, str], prior: Sequence[str],
                  provenance_source: Callable[[str], str | None]) -> tuple[int, int]:
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

        flags = card["flags"]
        combo = (identity.family, identity.kind_slug)
        tone_role = _TONE_ROLE.get(combo)
        rated = _RATED_ROLE.get(combo)
        artifact_sha = None

        if tone_role is not None:
            artifact_sha = _current_best_sha(db, subject, "recording",
                                             current_rubric=current_rubric, prior=prior,
                                             provenance_source=provenance_source)
            key = ReverifyKey(artifact_sha=artifact_sha, anchor=subject, role=tone_role)
            role, row_kind = tone_role, "reverify"
            answer = {"flagged": True, "flag": flags}
        else:
            role = "card-flag"
            if rated is not None:
                rated_role, provide_kind = rated
                artifact_sha = _current_best_sha(db, subject, provide_kind,
                                                 current_rubric=current_rubric, prior=prior,
                                                 provenance_source=provenance_source)
                if artifact_sha is not None:
                    role = rated_role
            key = FlagKey(family=identity.family, anchor=identity.anchor,
                          card_kind=identity.kind_slug, flags=flags)
            row_kind = "card-flag" if role == "card-flag" else "rating"
            answer = ({"flagged": True, "flag": flags} if row_kind == "card-flag"
                     else {"value": "unacceptable-none", "flag": flags})

        already = db.latest("assess", "learner", key)
        if already is not None:
            skipped += 1
            skips.append(("flag", f"card {card_id}", "already imported (same flags value)"))
            continue

        question = {"kind": row_kind, "role": role, "family": identity.family,
                   "anchor": identity.anchor, "card_kind": identity.kind_slug,
                   "flags": flags}
        if row_kind == "rating":
            question["artifact_sha"] = artifact_sha
        db.append(port="assess", backend="learner", key=key, subject=subject,
                 question=question, answer=answer)
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
        subject = _note_subject(note["tags"])
        if subject is None:
            skipped += 1
            skips.append(("review_note", str(note_id), "note not recognized"))
            continue
        text_sha = sha(text)
        key = LearnerNoteKey(anchor=subject, text_sha=text_sha)
        already = db.latest("assess", "learner-note", key)
        if already is not None:
            skipped += 1
            skips.append(("review_note", f"note {note_id}", "already harvested (unchanged text)"))
            continue
        db.append(port="assess", backend="learner-note", key=key, subject=subject,
                 question={"note_id": note_id, "text_sha": text_sha},
                 answer={"text": text})
        imported += 1
    return imported, skipped


# --- the one command -------------------------------------------------------

def import_collection(collection_path: str | Path, db: SyllabusDb, *,
                      current_rubric: Mapping[str, str], prior: Sequence[str],
                      provenance_source: Callable[[str], str | None]
                      ) -> ImportReport:
    """Read `collection_path` (an Anki collection.anki2, or an extracted
    .apkg's own copy) read-only and import revlog rows, card flags, and
    ReviewNote text into `db`. One pass, one report. `current_rubric`/
    `prior`/`provenance_source` are the flag import's own current_best
    parameters (wiring.Derivations carries the deck's real ones).
    """
    conn = _connect_readonly(collection_path)
    try:
        col = _load_collection(conn)
        skips: list[tuple[str, str, str]] = []
        revlog_imported, revlog_skipped = _import_revlog(conn, col, db, skips)
        flags_imported, flags_skipped = _import_flags(
            col, db, skips, current_rubric=current_rubric, prior=prior,
            provenance_source=provenance_source)
        notes_harvested, notes_skipped = _import_review_notes(col, db, skips)
    finally:
        conn.close()
    return ImportReport(
        revlog_imported=revlog_imported, revlog_skipped=revlog_skipped,
        flags_imported=flags_imported, flags_skipped=flags_skipped,
        notes_harvested=notes_harvested, notes_skipped=notes_skipped,
        skips=tuple(skips))
