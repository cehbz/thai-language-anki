"""The one-shot migration (spec 2 section 4): old thai-ff deck + old
word-list data -> the new <deck>/ layout (curated/*.yaml, media/objects/,
syllabus.db). Idempotent: every cache append checks db.latest(port,
backend, key) first, so a second run appends no new cache row and counts
the skip in MigrationReport.already_present; media writes are already
idempotent (content-addressed, insert-or-ignore provenance).

Implements items 1-4 and 6 of spec 2 section 4:
  1. word list -> curated/words.yaml + curated/targets.yaml, refusing any
     row with no category
  2. judged images -> media CAS + provenance (through ingest_picture,
     which normalizes at ingest per spec 4 section 3) + judge-backend
     cache rows under LEGACY_PICTURE_RUBRIC (a candidates.yaml verdict
     never says which rubric version judged, so it is carried as
     evidence that never ranks under the current rubric); no marker of
     the old deck's chosen picture is written
  3. Forvo answers -> provide/forvo cache rows, hit and miss alike
  4. proof-gallery notes + waivers.yaml -> learner assessment rows
  6. StudyRecords: nothing is written to the `study` table
Item 5 (judge_cache.sqlite) is retired: this module never opens
work/judge_cache.sqlite.

Old word ids are gloss slugs (e.g. "slow"); old picture-word note ids are
pw-NNN and never coincide with a word id. An old picture note joins a
word-list row by (thai, category) (join_key): a key matching more than
one row is reported in MigrationReport.ambiguous and left unjoined; a
note whose key matches no row is reported unmatched. Everything the old
deck keyed by note id (candidates.yaml verdicts, proof-note learner rows)
is re-keyed here under the word id the join finds. Spelling-sound notes
never migrate (no graphemes migrate this cutover) and always drop.

Every row this migration cannot place is reported, never silently
dropped -- see MigrationReport.unmigratable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import genanki
import yaml

from .cachekeys import JudgeKey, LearnerNoteKey, WaiverKey
from .cachekeys import sha as _key_component_sha
from .curated import build_categories, save_curated, CuratedBundle, RulebookConfig
from .entities import Pronunciation, Syllable, Target, Word
from .ids import TargetId, WordId
from .media import Provenance
from .profile import Profile
from .store import MediaStore, SyllabusDb

# A candidates.yaml verdict's rubric is unknown (the old record never said
# which rubric version judged); carried under this id, it can never equal
# a live rubric text's sha, so derivations.current_best's staleness check
# always excludes it from ranking (spec 3 section 6).
LEGACY_PICTURE_RUBRIC = "legacy-picture-rules"

# The old deck's one note model for picture-word notes -- proof_notes.jsonl
# rows carry an Anki guid computed from this model and the note id.
_PICTURE_WORD_MODEL = "picture_word"

# --- IPA -> Pronunciation ----------------------------------------------
#
# Ported (not imported -- src/thai_syllabus stays stdlib+PyYAML only, and
# this package imports nothing out of thai_deck_eval/thai_deck_gen by
# design, see __init__.py) out of thai_deck_eval/lang/ipa.py's parse_ipa,
# whose onset/vowel/coda token lists and 5-Chao-tone-letter table were
# independently confirmed against this deck's actual note data (see the
# implementation report: exactly {˧, ˨˩, ˥˩, ˦˥, ˨˩˦} appear, matching
# mid/low/falling/high/rising one-to-one).

_TONES = {"˧": "mid", "˨˩˦": "rising", "˨˩": "low", "˥˩": "falling", "˦˥": "high"}
_ONSETS = ["tɕʰ", "tɕ", "pʰ", "tʰ", "kʰ", "b", "d", "p", "t", "k", "ʔ",
           "m", "n", "ŋ", "f", "s", "h", "w", "l", "j", "r"]
_VOWELS = ["ɯa", "ia", "ua", "ɯ", "ɤ", "ɛ", "ɔ", "i", "e", "a", "o", "u"]
_CODAS = ["p", "t", "k", "ʔ", "m", "n", "ŋ", "j", "w"]


class IpaParseError(ValueError):
    pass


def _take(s: str, options: list[str]) -> tuple[str | None, str]:
    for o in options:
        if s.startswith(o):
            return o, s[len(o):]
    return None, s


def _parse_ipa_syllable(s: str) -> Syllable:
    tone = None
    for mark in sorted(_TONES, key=len, reverse=True):
        if s.endswith(mark):
            tone, s = _TONES[mark], s[: -len(mark)]
            break
    if tone is None:
        raise IpaParseError(f"no tone letters in {s!r}")
    onset, s = _take(s, _ONSETS)
    if onset is None:
        raise IpaParseError(f"unknown onset in {s!r}")
    vowel, s = _take(s, _VOWELS)
    if vowel is None:
        raise IpaParseError(f"unknown vowel in {s!r}")
    long = s.startswith("ː")
    s = s[1:] if long else s
    coda, s = _take(s, _CODAS)
    if s:
        raise IpaParseError(f"trailing {s!r}")
    return Syllable(segments=(onset, vowel, coda or ""),
                    vowel_length="long" if long else "short", tone=tone)


def _parse_ipa(ipa: str) -> tuple[Syllable, ...]:
    parts = [p for p in ipa.strip().replace(" ", ".").split(".") if p]
    if not parts:
        raise IpaParseError("empty ipa string")
    return tuple(_parse_ipa_syllable(p) for p in parts)


_PLACEHOLDER_PRON = Pronunciation(
    syllables=(Syllable(segments=("", "", ""), vowel_length="short", tone="mid"),),
    corroboration="disputed")


def _pron_from_ipa(ipa: str) -> Pronunciation:
    return Pronunciation(syllables=_parse_ipa(ipa), corroboration="curated_exception")


# --- report ----------------------------------------------------------------

@dataclass(frozen=True)
class UnmigratableItem:
    source: str      # which step/file this came from
    identity: str     # the row/entry's own identity (id, word, guid, ...)
    reason: str


@dataclass
class MigrationReport:
    curated: dict[str, int] = field(default_factory=dict)
    media: dict[str, int] = field(default_factory=dict)
    cache: dict[str, int] = field(default_factory=dict)
    study: dict[str, int] = field(default_factory=lambda: {"records": 0})
    unmigratable: list[UnmigratableItem] = field(default_factory=list)
    ambiguous: list[tuple[str, str]] = field(default_factory=list)
    already_present: dict[str, int] = field(default_factory=dict)
    audio_skipped: int = 0

    def drop(self, source: str, identity: str, reason: str) -> None:
        self.unmigratable.append(UnmigratableItem(source, identity, reason))

    def bump(self, bucket: dict[str, int], key: str, n: int = 1) -> None:
        bucket[key] = bucket.get(key, 0) + n


# --- helpers ---------------------------------------------------------------

def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[tuple[int, dict | None, str | None]]:
    """Returns (line_no, parsed_dict_or_None, error_or_None) for every
    line, so callers can report malformed lines instead of silently
    skipping them.
    """
    out: list[tuple[int, dict | None, str | None]] = []
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append((i, json.loads(line), None))
        except json.JSONDecodeError as e:
            out.append((i, None, str(e)))
    return out


def _date_of(value: Any, fallback: date) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return fallback
    return fallback


def join_key(thai: str, category: str) -> tuple[str, str]:
    """The (thai, category) pair an old picture note and a word-list row
    join on (spec 2 section 4 item 2).
    """
    return (thai, category)


def ingest_picture(media_store: MediaStore, db: SyllabusDb, data: bytes, ext: str,
                   provenance: Provenance) -> tuple[str, bool]:
    """Normalizes `data` through MediaStore.add_image (spec 4 section 3;
    `ext` is only add_image's fallback hint for an undetectable format)
    and records `provenance` as a media row if the resulting sha is new.
    Returns (sha, is_new) -- is_new is db.add_media's own insert-or-ignore
    result, so a caller can count a re-run's skips. Idempotent: a repeat
    call with the same bytes writes the object and the provenance row at
    most once. Raises ValueError if `data` cannot be decoded as an image.
    """
    result = media_store.add_image(data, ext=ext)
    is_new = db.add_media(sha=result.sha, kind="picture", ext=result.ext,
                          source=provenance.source, origin=provenance.origin,
                          licence=provenance.licence, acquired=provenance.acquired)
    return result.sha, is_new


def _ingest_and_count(media_store: MediaStore, db: SyllabusDb, report: MigrationReport,
                      data: bytes, ext: str, provenance: Provenance) -> str:
    """ingest_picture, counted: a new object bumps report.media
    ["objects_written"], a re-run hit bumps report.already_present["media"].
    """
    sha, is_new = ingest_picture(media_store, db, data, ext, provenance)
    if is_new:
        report.bump(report.media, "objects_written")
    else:
        report.bump(report.already_present, "media")
    return sha


def _record_once(db: SyllabusDb, report: MigrationReport, bucket: str, *,
                 port: str, backend: str, key: Any, write) -> None:
    """Runs `write()` -- a zero-argument call that appends one cache row
    under (port, backend, key) -- only if no row already exists there;
    otherwise counts the skip in report.already_present[bucket].
    """
    if db.latest(port, backend, key) is not None:
        report.bump(report.already_present, bucket)
        return
    write()
    report.bump(report.cache, bucket)


# --- item 1: word list -----------------------------------------------------

def _migrate_word_list(old_data: Path, old_deck: Path, db: SyllabusDb,
                       report: MigrationReport
                       ) -> tuple[list[tuple[Word, str | None]], list[Target],
                                 dict[tuple[str, str], str], set[tuple[str, str]]]:
    rows = _load_yaml(old_data / "word_list_th.yaml") or []
    picture_words = _load_yaml(old_deck / "notes" / "picture_words.yaml") or []
    ipa_by_thai: dict[str, str] = {}
    for note in picture_words:
        thai, ipa = note.get("thai"), note.get("ipa")
        if thai and ipa and thai not in ipa_by_thai:
            ipa_by_thai[thai] = ipa

    word_rows: list[tuple[Word, str | None]] = []
    targets: list[Target] = []
    seen_ids: set[str] = set()
    word_id_by_thai: dict[str, str] = {}          # first-seen, classifier lookups only
    word_ids_by_key: dict[tuple[str, str], list[str]] = {}
    classifier_words_by_thai: dict[str, Word] = {}

    def pron_for(thai: str, source: str, identity: str) -> Pronunciation:
        ipa = ipa_by_thai.get(thai)
        if ipa is None:
            report.bump(report.curated, "words_without_pronunciation")
            return _PLACEHOLDER_PRON
        try:
            return _pron_from_ipa(ipa)
        except IpaParseError as e:
            report.drop(source, identity, f"unparseable ipa {ipa!r}: {e}")
            report.bump(report.curated, "words_without_pronunciation")
            return _PLACEHOLDER_PRON

    good_rows = []
    for i, row in enumerate(rows):
        identity = row.get("id") if isinstance(row, dict) else None
        identity = identity or f"<row {i}>"
        if not isinstance(row, dict) or not row.get("id") or not row.get("thai") \
                or not row.get("gloss"):
            report.drop("data/word_list_th.yaml", str(identity),
                        "missing required field(s) among id/thai/gloss")
            continue
        if row["id"] in seen_ids:
            report.drop("data/word_list_th.yaml", row["id"], "duplicate id")
            continue
        seen_ids.add(row["id"])
        word_id_by_thai.setdefault(row["thai"], row["id"])
        good_rows.append(row)

    for row in good_rows:
        word_id = row["id"]
        category = row.get("category")
        if not category:
            report.drop("data/word_list_th.yaml", word_id, "missing category")
            continue
        classifier_id: str | None = None
        classifier_thai = row.get("classifier")
        if classifier_thai:
            if classifier_thai in word_id_by_thai:
                classifier_id = word_id_by_thai[classifier_thai]
            else:
                classifier_id = f"classifier:{classifier_thai}"
                if classifier_thai not in classifier_words_by_thai:
                    classifier_words_by_thai[classifier_thai] = Word(
                        id=WordId(classifier_id), thai=classifier_thai,
                        pron=pron_for(classifier_thai, "data/word_list_th.yaml",
                                     f"classifier {classifier_thai}"),
                        meaning="(classifier -- no gloss migrated)", classifier=None)

        word_rows.append((Word(id=WordId(word_id), thai=row["thai"],
                              pron=pron_for(row["thai"], "data/word_list_th.yaml", word_id),
                              meaning=row["gloss"],
                              classifier=WordId(classifier_id) if classifier_id else None),
                         category))
        targets.append(Target(id=TargetId(f"{word_id}/receptive"), word=WordId(word_id),
                              skill="receptive", introduction="picture_card"))
        word_ids_by_key.setdefault(join_key(row["thai"], category), []).append(word_id)

        if row.get("image_query") and row.get("image_query_source") == "human":
            # Not one of spec 3's roster rows (a direction carries no
            # artifact_sha yet, so the learner Assessor's own
            # learner:sha(ARTIFACT):ROLE template doesn't fit) -- kept
            # readable and namespaced under the learner backend anyway.
            key = f"learner:direction:image_query:{word_id}"
            _record_once(db, report, "direction", port="assess", backend="learner", key=key,
                        write=lambda k=key, s=word_id, v=row["image_query"]: db.append(
                            port="assess", backend="learner", key=k, subject=s,
                            question={"kind": "direction", "of": "image_query"},
                            answer={"direction": v}))
        # Dropped, per spec 2 section 4 item 1: picturable, emphasis,
        # image_query (non-human source), split_of, part_of_speech.

    ambiguous_keys = {k for k, ids in word_ids_by_key.items() if len(ids) > 1}
    word_id_by_key = {k: ids[0] for k, ids in word_ids_by_key.items() if len(ids) == 1}

    # Classifier words are synthesized, untargeted closure words (spec 1:
    # closure words are in no category).
    word_rows.extend((w, None) for w in classifier_words_by_thai.values())
    report.bump(report.curated, "classifier_words_synthesized",
               len(classifier_words_by_thai))
    report.bump(report.curated, "words", len(word_rows))
    report.bump(report.curated, "targets", len(targets))
    return word_rows, targets, word_id_by_key, ambiguous_keys


# --- note subjects: old note id -> word id, via the (thai, category) join --
#
# Word ids are gloss slugs (e.g. "slow"); old picture-word note ids are
# pw-NNN -- they never coincide, so the row and the note join on
# (thai, category) instead. Everything keyed by note id in the old deck
# (candidates.yaml verdicts, proof-note learner rows) must land under the
# word id this join finds, not the note id.

def _note_subjects(old_deck: Path, word_id_by_key: dict[tuple[str, str], str],
                   ambiguous_keys: set[tuple[str, str]],
                   report: MigrationReport) -> dict[str, str]:
    subjects: dict[str, str] = {}
    picture_words = _load_yaml(old_deck / "notes" / "picture_words.yaml") or []
    for note in picture_words:
        note_id = note.get("id", "<unknown>")
        key = join_key(note.get("thai", ""), note.get("category", ""))
        if key in ambiguous_keys:
            report.drop("notes/picture_words.yaml", note_id,
                        f"(thai, category) {key} matches more than one word row")
            continue
        word_id = word_id_by_key.get(key)
        if word_id is None:
            report.drop("notes/picture_words.yaml", note_id,
                        "no word with this Thai form and category")
            continue
        subjects[note_id] = word_id

    # Spelling-sound notes (grapheme images) never migrate -- no graphemes
    # migrate in this cutover -- so every one drops, unconditionally.
    spelling_sound = _load_yaml(old_deck / "notes" / "spelling_sound.yaml") or []
    for note in spelling_sound:
        report.drop("notes/spelling_sound.yaml", note.get("id", "<unknown>"),
                    "grapheme images are not migrated (no graphemes migrated)")
    return subjects


# --- item 2: judged images ---------------------------------------------

def _migrate_media_manifest(old_deck: Path, media_store: MediaStore, db: SyllabusDb,
                            report: MigrationReport) -> None:
    manifest = _load_yaml(old_deck / "media_manifest.yaml") or {}
    for i, entry in enumerate(manifest.get("entries", [])):
        rel = entry.get("file", "")
        identity = rel or f"<entry {i}>"
        if not rel.startswith("media/images/"):
            report.audio_skipped += 1  # audio regenerates; not migrated
            continue
        src_path = old_deck / rel
        if not src_path.exists():
            report.drop("media_manifest.yaml", identity, "file missing on disk")
            continue
        ext = src_path.suffix.lstrip(".") or "bin"
        try:
            _ingest_and_count(media_store, db, report, src_path.read_bytes(), ext,
                              Provenance(source=entry.get("channel", "unknown"),
                                        origin=entry.get("origin", ""),
                                        licence=entry.get("license", "unknown"),
                                        acquired=_date_of(entry.get("fetched"), date(1970, 1, 1))))
        except ValueError as exc:
            report.drop("media_manifest.yaml", identity, str(exc))


def _migrate_current_deck_images(old_deck: Path, media_store: MediaStore, db: SyllabusDb,
                                 report: MigrationReport) -> None:
    # Picture words only -- spelling-sound notes never migrate (see
    # _note_subjects) so their images are left to _migrate_media_manifest.
    # No cache row is written for the deck's current picture (no marker of
    # the old deck's choice); a legacy verdict on the same sha still lands
    # through _migrate_candidates when a candidates.yaml entry names it.
    notes = _load_yaml(old_deck / "notes" / "picture_words.yaml") or []
    for note in notes:
        image = note.get("image")
        if not image:
            continue
        note_id = note.get("id", "<unknown>")
        path = old_deck / "media" / image
        if not path.exists():
            report.drop("notes/picture_words.yaml", note_id,
                        f"image file missing on disk: {image}")
            continue
        ext = path.suffix.lstrip(".") or "bin"
        try:
            _ingest_and_count(media_store, db, report, path.read_bytes(), ext,
                              Provenance(source="legacy-current", origin=image,
                                        licence="unknown", acquired=date(1970, 1, 1)))
        except ValueError as exc:
            report.drop("notes/picture_words.yaml", note_id, str(exc))


def _migrate_candidates(old_deck: Path, media_store: MediaStore, db: SyllabusDb,
                        note_subjects: dict[str, str], report: MigrationReport) -> None:
    for cand_file in sorted(old_deck.glob("work/candidates/*/candidates.yaml")):
        note_id = cand_file.parent.name
        word_id = note_subjects.get(note_id)
        if word_id is None:
            # A candidates/ working dir with no corresponding (mapped) note
            # -- CAS media below still lands, but there is no word id to
            # rank a verdict under.
            report.drop("work/candidates", note_id,
                        "no word mapped for this picture note")
        data = _load_yaml(cand_file) or {}
        # Two on-disk shapes seen in the wild: {"corpora": [...], "candidates":
        # [...]} (current) and a bare top-level list (measured 2026-09-04:
        # 67 of 580 real candidates.yaml files use the bare shape). Handle
        # both.
        candidates = data.get("candidates", []) if isinstance(data, dict) else data
        for i, cand in enumerate(candidates):
            identity = f"{note_id}/{cand.get('file', f'<item {i}>')}"
            if not isinstance(cand, dict) or not cand.get("file"):
                report.drop("work/candidates", identity, "malformed candidate entry")
                continue
            img_path = cand_file.parent / cand["file"]
            if not img_path.exists():
                report.drop("work/candidates", identity, "candidate image missing on disk")
                continue
            ext = img_path.suffix.lstrip(".") or "bin"
            try:
                sha = _ingest_and_count(
                    media_store, db, report, img_path.read_bytes(), ext,
                    Provenance(source=cand.get("source", "unknown"),
                              origin=cand.get("url", ""),
                              licence=cand.get("license", "unknown"),
                              acquired=date.fromtimestamp(img_path.stat().st_mtime)))
            except ValueError as exc:
                report.drop("work/candidates", identity, str(exc))
                continue

            if word_id is None:
                continue

            failed_rules = cand.get("failed_rules") or []
            judge_key = JudgeKey.for_rule(LEGACY_PICTURE_RUBRIC, sha, word_id,
                                          "picture-for-word")
            question = {"role": "picture-for-word", "artifact_sha": sha,
                       "rubric": LEGACY_PICTURE_RUBRIC, "kind": "picture"}
            if cand.get("passed"):
                _record_once(
                    db, report, "judge_pass", port="assess", backend="judge", key=judge_key,
                    write=lambda k=judge_key, s=word_id, q=question: db.append_judge_verdict(
                        key=k, subject=s, question=q,
                        answer={"value": True, "evidence": "migrated: passed every picture rule"}))
            elif failed_rules:
                _record_once(
                    db, report, "judge_fail", port="assess", backend="judge", key=judge_key,
                    write=lambda k=judge_key, s=word_id, q=question, fr=failed_rules:
                        db.append_judge_verdict(
                            key=k, subject=s, question=q,
                            answer={"value": False,
                                   "evidence": "migrated: failed " + ", ".join(fr)}))
            else:
                report.bump(report.cache, "judge_unrecorded")


# --- item 3: forvo ---------------------------------------------------------

def _migrate_forvo(old_deck: Path, db: SyllabusDb, report: MigrationReport) -> None:
    path = old_deck / "work" / "forvo_lookups.jsonl"
    for line_no, entry, error in _load_jsonl(path):
        if error is not None:
            report.drop("work/forvo_lookups.jsonl", f"line {line_no}",
                        f"malformed json: {error}")
            continue
        word = entry.get("word")
        if not word:
            report.drop("work/forvo_lookups.jsonl", f"line {line_no}",
                        "missing 'word'")
            continue
        # spec 3 roster: forvo's key is "forvo:WORD" (never re-asked). No
        # explicit ts: an idempotence check keyed on (port, backend, key)
        # alone must not race a deterministic ts into the cache table's
        # (key_sha, ts) primary key on a second run.
        key = f"forvo:{word}"
        items = entry.get("items", [])
        _record_once(db, report, "forvo", port="provide", backend="forvo", key=key,
                    write=lambda k=key, w=word, it=items: db.append(
                        port="provide", backend="forvo", key=k, subject=w,
                        question={"word": w, "kind": "recording"}, answer={"items": it}))


# --- item 4: proof-gallery notes + waivers ---------------------------------

def _migrate_proof_notes(old_deck: Path, note_subjects: dict[str, str], db: SyllabusDb,
                         report: MigrationReport) -> None:
    # proof_notes.jsonl carries an Anki guid, not the old note id -- the
    # guid -> note id map is built once, from every picture-word note's own
    # (fixed) model.
    picture_words = _load_yaml(old_deck / "notes" / "picture_words.yaml") or []
    guid_to_note_id = {genanki.guid_for(_PICTURE_WORD_MODEL, n["id"]): n["id"]
                      for n in picture_words if n.get("id")}
    path = old_deck / "work" / "proof_notes.jsonl"
    for line_no, entry, error in _load_jsonl(path):
        if error is not None:
            report.drop("work/proof_notes.jsonl", f"line {line_no}",
                        f"malformed json: {error}")
            continue
        guid = entry.get("guid")
        if not guid:
            report.drop("work/proof_notes.jsonl", f"line {line_no}", "missing 'guid'")
            continue
        note_id = guid_to_note_id.get(guid)
        if note_id is None:
            report.drop("work/proof_notes.jsonl", guid,
                        "guid matches no picture-word note")
            continue
        word_id = note_subjects.get(note_id)
        if word_id is None:
            continue  # already reported by _note_subjects (unmapped Thai/category)
        try:
            ts = int(datetime.fromisoformat(entry["ts"]).timestamp() * 1_000_000_000)
        except (KeyError, ValueError):
            ts = None
        # These are free-text reviewer comments (no explicit accept/reject
        # enum in the source) -- migrated as a "rating" kind assessment
        # carrying a note but no verdict, per spec 2 section 4 item 4's
        # "kind per content: rating, direction, waiver" (this content is
        # closest to "rating": a learner reaction to a specific card).
        # Keyed by (word id, text sha) so an edited text is a new row and
        # a repeated text is an exact-key hit, not by guid (spec 4
        # section 4's ReviewNote-harvest convention, reused here).
        text = entry.get("text", "")
        key = LearnerNoteKey(anchor=word_id, text_sha=_key_component_sha(text))
        question = {"kind": "note", "guid": guid, "note_id": entry.get("note_id"),
                   "model": entry.get("model", ""), "tags": entry.get("tags", [])}
        answer = {"kind": "rating", "rating": None, "note": text}
        _record_once(db, report, "learner_rating", port="assess", backend="learner", key=key,
                    write=lambda k=key, s=word_id, q=question, a=answer, t=ts: db.append(
                        port="assess", backend="learner", key=k, subject=s,
                        question=q, answer=a, ts=t))


def _migrate_waivers(old_deck: Path, db: SyllabusDb, report: MigrationReport) -> None:
    path = old_deck / "waivers.yaml"
    rows = _load_yaml(path)
    if not rows:
        return
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "rule" not in row or "note_id" not in row:
            report.drop("waivers.yaml", f"<row {i}>",
                        "missing required field(s) among rule/note_id")
            continue
        key = WaiverKey(rule_id=row["rule"], note_id=row["note_id"],
                        artifact_sha=row.get("artifact_sha"))
        _record_once(db, report, "waiver", port="assess", backend="learner", key=key,
                    write=lambda r=row: db.append_waiver(
                        rule_id=r["rule"], note_id=r["note_id"],
                        artifact_sha=r.get("artifact_sha"), waived=True,
                        reason=r.get("reason", "")))


# --- entry point -------------------------------------------------------

def migrate(old_deck: Path, old_data: Path, new_root: Path) -> MigrationReport:
    old_deck, old_data, new_root = Path(old_deck), Path(old_data), Path(new_root)
    new_root.mkdir(parents=True, exist_ok=True)
    curated_dir = new_root / "curated"
    media_store = MediaStore(new_root / "media")
    db = SyllabusDb(new_root / "syllabus.db")
    report = MigrationReport()

    word_rows, targets, word_id_by_key, ambiguous_keys = \
        _migrate_word_list(old_data, old_deck, db, report)
    report.ambiguous = sorted(ambiguous_keys)
    save_curated(curated_dir, CuratedBundle(
        words=tuple(w for w, _ in word_rows), targets=tuple(targets), graphemes=(),
        confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(), categories=build_categories(word_rows)))

    note_subjects = _note_subjects(old_deck, word_id_by_key, ambiguous_keys, report)

    _migrate_media_manifest(old_deck, media_store, db, report)
    _migrate_current_deck_images(old_deck, media_store, db, report)
    _migrate_candidates(old_deck, media_store, db, note_subjects, report)
    _migrate_forvo(old_deck, db, report)
    _migrate_proof_notes(old_deck, note_subjects, db, report)
    _migrate_waivers(old_deck, db, report)

    db.close()
    return report
