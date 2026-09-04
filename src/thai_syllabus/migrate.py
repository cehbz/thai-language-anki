"""The one-shot migration (spec 2 section 4): old thai-ff deck + old
word-list data -> the new <deck>/ layout (curated/*.yaml, media/objects/,
syllabus.db). Idempotent (media/cache writes are content-addressed or
naturally re-askable; running it twice just re-derives the same rows --
cache is append-only so a second run does add duplicate cache rows, which
is the documented append-is-checkpoint behaviour, not a bug).

Only items 1-4 and 6 of spec 2 section 4 are implemented, exactly:
  1. word list -> curated/words.yaml + curated/targets.yaml (also builds
     the thai -> word id join key everything below is keyed through)
  2. judged images -> media CAS + provenance + judge-backend cache rows
     under role="picture-for-word"/PICTURE_FIT_RUBRIC (candidates.yaml's
     recorded passed/failed_rules become judge PASS/FAIL rows so
     derivations.current_best can rank them) + a machine-chosen marker on
     the deck's currently-selected images
  3. Forvo answers -> provide/forvo cache rows, hit and miss alike
  4. proof-gallery notes + waivers.yaml -> learner assessment rows
  6. StudyRecords: nothing is written to the `study` table -- item 6 says
     none migrate, so migrate() never calls SyllabusDb.append_study.
Item 5 (judge_cache.sqlite) is explicitly retired: this module never
opens work/judge_cache.sqlite.

Old word ids are gloss slugs (e.g. "slow"); old picture-word note ids are
pw-NNN and never coincide with a word id. Everything the old deck keyed by
note id (machine-chosen markers, candidates.yaml verdicts, proof-note
learner rows) is re-keyed here under the word id found by joining the
note's Thai form against the word list (_note_subjects). Spelling-sound
notes never migrate (no graphemes migrate this cutover) and always drop.

Every row this migration cannot place is reported, never silently
dropped -- see MigrationReport.unmigratable and the ambiguity notes below
each step for the conventions this implementation had to invent where the
spec fixes only the destination shape, not the source-to-destination
mapping (spec 3, which owns per-backend cache-key functions, is out of
scope for spec 2 and wasn't available to this implementation).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import genanki
import yaml

from .curated import build_categories, save_curated, CuratedBundle, RulebookConfig
from .entities import Pronunciation, Syllable, Target, Word
from .ids import TargetId, WordId
from .profile import Profile
from .rulebook import PICTURE_FIT_RUBRIC
from .store import MediaStore, SyllabusDb

# --- IPA -> Pronunciation ----------------------------------------------
#
# Ported (not imported -- src/thai_syllabus stays stdlib+PyYAML only, and
# this package imports nothing from thai_deck_eval/thai_deck_gen by
# design, see __init__.py) from thai_deck_eval/lang/ipa.py's parse_ipa,
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


def _midnight_utc_ns(d: date) -> int:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _write_media(media_store: MediaStore, data: bytes, ext: str) -> tuple[str, bool]:
    """Like MediaStore.write, but also reports whether the object was
    newly written (for accurate report counts) -- computed by hashing
    before the write since MediaStore.write itself only returns the sha.
    """
    sha = hashlib.sha256(data).hexdigest()
    is_new = not media_store.has(sha, ext)
    media_store.write(data, ext=ext)
    return sha, is_new


# --- item 1: word list -----------------------------------------------------

def _migrate_word_list(old_data: Path, old_deck: Path, db: SyllabusDb,
                       report: MigrationReport
                       ) -> tuple[list[tuple[Word, str | None]], list[Target], dict[str, str]]:
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
    thai_to_word_id: dict[str, str] = {}
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

    # pass 1: real word-list rows (also seeds thai_to_word_id for classifier
    # resolution below).
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
        if row["thai"] in thai_to_word_id:
            # First row with this Thai form wins the join key; later rows
            # sharing it (homographs in the old data) still become their
            # own Word/Target below, they just can't be the join target
            # for note-keyed rows sharing that Thai form. Reported item by
            # item as well as counted -- "no silent drops" means naming the
            # row and the word its picture note went to instead, not just
            # how many such rows there were.
            report.bump(report.curated, "homograph_rows")
            report.drop("data/word_list_th.yaml", row["id"],
                        f"homograph of {thai_to_word_id[row['thai']]}: "
                        "the picture note joins to that word")
        else:
            thai_to_word_id[row["thai"]] = row["id"]
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
            if classifier_thai in thai_to_word_id:
                classifier_id = thai_to_word_id[classifier_thai]
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

        if row.get("image_query") and row.get("image_query_source") == "human":
            # Not one of spec 3's roster rows (a direction carries no
            # artifact_sha yet, so the learner Assessor's own
            # learner:sha(ARTIFACT):ROLE template doesn't fit) -- kept
            # readable and namespaced under the learner backend anyway.
            key = f"learner:direction:image_query:{word_id}"
            db.append(port="assess", backend="learner", key=key, subject=word_id,
                     question={"kind": "direction", "of": "image_query"},
                     answer={"direction": row["image_query"]})
            report.bump(report.cache, "direction")
        # Dropped, per spec 2 section 4 item 1: picturable, emphasis,
        # image_query (non-human source), split_of, part_of_speech.

    # Classifier words are synthesized, untargeted closure words (spec 1:
    # closure words are in no category).
    word_rows.extend((w, None) for w in classifier_words_by_thai.values())
    report.bump(report.curated, "classifier_words_synthesized",
               len(classifier_words_by_thai))
    report.bump(report.curated, "words", len(word_rows))
    report.bump(report.curated, "targets", len(targets))
    return word_rows, targets, thai_to_word_id


# --- note subjects: old note id -> word id, via the Thai-form join ---------
#
# Word ids are gloss slugs (e.g. "slow"); old picture-word note ids are
# pw-NNN -- they never coincide, so the row and the note share only the
# Thai form. Everything keyed by note id in the old deck (marker rows,
# candidates.yaml verdicts, proof-note learner rows) must land under the
# word id this join finds, not the note id.

def _note_subjects(old_deck: Path, thai_to_word_id: dict[str, str],
                   report: MigrationReport) -> dict[str, str]:
    subjects: dict[str, str] = {}
    picture_words = _load_yaml(old_deck / "notes" / "picture_words.yaml") or []
    for note in picture_words:
        note_id = note.get("id", "<unknown>")
        word_id = thai_to_word_id.get(note.get("thai"))
        if word_id is None:
            report.drop("notes/picture_words.yaml", note_id,
                        "no word with this Thai form")
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
        if not rel.startswith("media/images/"):
            continue  # audio is explicitly out of scope for migration
        identity = rel or f"<entry {i}>"
        src_path = old_deck / rel
        if not src_path.exists():
            report.drop("media_manifest.yaml", identity, "file missing on disk")
            continue
        data = src_path.read_bytes()
        ext = src_path.suffix.lstrip(".") or "bin"
        sha, is_new_object = _write_media(media_store, data, ext=ext)
        inserted = db.add_media(sha=sha, kind="picture", ext=ext,
                                source=entry.get("channel", "unknown"),
                                origin=entry.get("origin", ""),
                                licence=entry.get("license", "unknown"),
                                acquired=_date_of(entry.get("fetched"), date(1970, 1, 1)))
        if is_new_object:
            report.bump(report.media, "objects_written")
        if inserted:
            report.bump(report.media, "provenance_rows")


def _resolve_note_image_sha(old_deck: Path, note_image: str, media_store: MediaStore,
                            db: SyllabusDb, report: MigrationReport,
                            source: str, identity: str) -> str | None:
    path = old_deck / "media" / note_image
    if not path.exists():
        report.drop(source, identity, f"image file missing on disk: {note_image}")
        return None
    data = path.read_bytes()
    ext = path.suffix.lstrip(".") or "bin"
    sha, is_new_object = _write_media(media_store, data, ext=ext)
    if is_new_object:
        report.bump(report.media, "objects_written")
    if not db.has_media(sha):
        # Not in media_manifest.yaml (unexpected but not fatal) -- record
        # minimal provenance so the media table stays the single source of
        # truth for every object under media/objects/.
        db.add_media(sha=sha, kind="picture", ext=ext, source="legacy-current",
                    origin=note_image, licence="unknown",
                    acquired=date(1970, 1, 1))
        report.bump(report.media, "provenance_rows")
    return sha


def _migrate_current_deck_images(old_deck: Path, media_store: MediaStore, db: SyllabusDb,
                                 note_subjects: dict[str, str],
                                 report: MigrationReport) -> None:
    # Picture words only -- spelling-sound notes never migrate (see
    # _note_subjects) so their images are left to _migrate_media_manifest
    # (CAS/provenance still land, just with no marker row on top).
    notes = _load_yaml(old_deck / "notes" / "picture_words.yaml") or []
    for note in notes:
        image = note.get("image")
        if not image:
            continue
        note_id = note.get("id", "<unknown>")
        word_id = note_subjects.get(note_id)
        if word_id is None:
            continue  # already reported by _note_subjects
        sha = _resolve_note_image_sha(old_deck, image, media_store, db, report,
                                      "notes/picture_words.yaml", note_id)
        if sha is None:
            continue
        key = f"machine-chosen:{word_id}:{sha}"
        db.append(port="assess", backend="machine-chosen", key=key, subject=word_id,
                 question={"note_id": note_id, "word": note.get("thai")},
                 answer={"marker": "machine-chosen", "sha": sha})
        report.bump(report.cache, "machine_chosen")


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
        # [...]} (current) and a bare top-level list (an older layout some
        # candidates.yaml files were never migrated off of). Handle both.
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
            data_bytes = img_path.read_bytes()
            ext = img_path.suffix.lstrip(".") or "bin"
            sha, is_new_object = _write_media(media_store, data_bytes, ext=ext)
            if is_new_object:
                report.bump(report.media, "objects_written")
            if not db.has_media(sha):
                acquired = date.fromtimestamp(img_path.stat().st_mtime)
                db.add_media(sha=sha, kind="picture", ext=ext,
                            source=cand.get("source", "unknown"),
                            origin=cand.get("url", ""),
                            licence=cand.get("license", "unknown"), acquired=acquired)
                report.bump(report.media, "provenance_rows")

            if word_id is None:
                continue

            failed_rules = cand.get("failed_rules") or []
            if cand.get("passed"):
                db.append_judge_verdict(rule_id="picture-for-word", note_id=word_id,
                                        artifact_sha=sha, verdict=True,
                                        rubric=PICTURE_FIT_RUBRIC,
                                        evidence="migrated: passed every picture rule")
                report.bump(report.cache, "judge_pass")
            elif failed_rules:
                db.append_judge_verdict(rule_id="picture-for-word", note_id=word_id,
                                        artifact_sha=sha, verdict=False,
                                        rubric=PICTURE_FIT_RUBRIC,
                                        evidence="migrated: failed " + ", ".join(failed_rules))
                report.bump(report.cache, "judge_fail")
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
        fetched = _date_of(entry.get("fetched"), date(1970, 1, 1))
        # spec 3 roster: forvo's key is "forvo:WORD" (never re-asked; the
        # answer outlives the quota).
        db.append(port="provide", backend="forvo", key=f"forvo:{word}", subject=word,
                 question={"word": word}, answer={"items": entry.get("items", [])},
                 ts=_midnight_utc_ns(fetched))
        report.bump(report.cache, "forvo")


# --- item 4: proof-gallery notes + waivers ---------------------------------

def _migrate_proof_notes(old_deck: Path, note_subjects: dict[str, str], db: SyllabusDb,
                         report: MigrationReport) -> None:
    # proof_notes.jsonl carries an Anki guid, not the old note id -- invert
    # genanki.guid_for(entry["model"], note_id) over every old picture-word
    # note id to recover which note (and therefore which word) each row
    # belongs to.
    picture_word_ids = [n.get("id") for n in
                        (_load_yaml(old_deck / "notes" / "picture_words.yaml") or [])
                        if n.get("id")]
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
        model = entry.get("model", "")
        note_id = next((nid for nid in picture_word_ids
                        if genanki.guid_for(model, nid) == guid), None)
        if note_id is None:
            report.drop("work/proof_notes.jsonl", guid,
                        "guid matches no picture-word note")
            continue
        word_id = note_subjects.get(note_id)
        if word_id is None:
            continue  # already reported by _note_subjects (unmapped Thai)
        try:
            ts = int(datetime.fromisoformat(entry["ts"]).timestamp() * 1_000_000_000)
        except (KeyError, ValueError):
            ts = None
        # These are free-text reviewer comments (no explicit accept/reject
        # enum in the source) -- migrated as a "rating" kind assessment
        # carrying a note but no verdict, per spec 2 section 4 item 4's
        # "kind per content: rating, direction, waiver" (this content is
        # closest to "rating": a learner reaction to a specific card).
        db.append(port="assess", backend="learner",
                 key=f"learner:note:{word_id}:{guid}", subject=word_id,
                 question={"kind": "note", "guid": guid,
                          "note_id": entry.get("note_id"), "model": model,
                          "tags": entry.get("tags", [])},
                 answer={"kind": "rating", "rating": None, "note": entry.get("text", "")},
                 ts=ts)
        report.bump(report.cache, "learner_rating")


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
        db.append_waiver(rule_id=row["rule"], note_id=row["note_id"],
                         artifact_sha=row.get("artifact_sha"), waived=True,
                         reason=row.get("reason", ""))
        report.bump(report.cache, "waiver")


# --- entry point -------------------------------------------------------

def migrate(old_deck: Path, old_data: Path, new_root: Path) -> MigrationReport:
    old_deck, old_data, new_root = Path(old_deck), Path(old_data), Path(new_root)
    new_root.mkdir(parents=True, exist_ok=True)
    curated_dir = new_root / "curated"
    media_store = MediaStore(new_root / "media")
    db = SyllabusDb(new_root / "syllabus.db")
    report = MigrationReport()

    word_rows, targets, thai_to_word_id = _migrate_word_list(old_data, old_deck, db, report)
    save_curated(curated_dir, CuratedBundle(
        words=tuple(w for w, _ in word_rows), targets=tuple(targets), graphemes=(),
        confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(), categories=build_categories(word_rows)))

    note_subjects = _note_subjects(old_deck, thai_to_word_id, report)

    _migrate_media_manifest(old_deck, media_store, db, report)
    _migrate_current_deck_images(old_deck, media_store, db, note_subjects, report)
    _migrate_candidates(old_deck, media_store, db, note_subjects, report)
    _migrate_forvo(old_deck, db, report)
    _migrate_proof_notes(old_deck, note_subjects, db, report)
    _migrate_waivers(old_deck, db, report)

    db.close()
    return report
