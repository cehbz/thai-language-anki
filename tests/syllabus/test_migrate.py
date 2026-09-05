"""Tests for migrate.py (spec 2 section 4): the one-shot migration, run
against a small synthetic old-deck fixture built under tmp_path. Never
touches ~/decks or the real data/ directory.
"""
import io
import json
import sqlite3

import genanki
import pytest
import yaml
from PIL import Image

from thai_syllabus import curated
from thai_syllabus.derivations import current_best
from thai_syllabus.migrate import LEGACY_PICTURE_RUBRIC, MigrationReport, migrate
from thai_syllabus.rulebook import PICTURE_FIT_RUBRIC
from thai_syllabus.store import IMAGE_MAX_LONG_EDGE, SyllabusDb

PW1_GUID = genanki.guid_for("picture_word", "pw-1")


def _write_yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in rows:
        lines.append(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _img_bytes(size, color, fmt="JPEG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format=fmt)
    return buf.getvalue()


def _row_count(db_path, table):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(f"select count(*) from {table}").fetchone()[0]
    finally:
        con.close()


def _append_word_rows(old_data, *rows):
    word_list = old_data / "word_list_th.yaml"
    existing = yaml.safe_load(word_list.read_text(encoding="utf-8"))
    existing.extend(rows)
    word_list.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")


@pytest.fixture
def old_deck(tmp_path):
    d = tmp_path / "old_deck"
    (d / "media" / "images").mkdir(parents=True)
    # pw-1's own image is oversized (1000x600): exercises normalization at
    # ingest (spec 4 section 3) end to end.
    (d / "media" / "images" / "pw-1.jpg").write_bytes(_img_bytes((1000, 600), "red"))
    (d / "media" / "images" / "pw-2.jpg").write_bytes(_img_bytes((4, 4), "blue"))
    (d / "media" / "images" / "sp-k.jpg").write_bytes(_img_bytes((4, 4), "green"))

    _write_yaml(d / "notes" / "picture_words.yaml", [
        {"id": "pw-1", "thai": "ไก่", "category": "Animals",  # ไก่ = chicken
         "ipa": "kaj˨˩", "image": "images/pw-1.jpg"},
        {"id": "pw-2", "thai": "ช้า", "category": "Adjectives",  # ช้า = slow
         "image": "images/pw-2.jpg"},
    ])
    _write_yaml(d / "notes" / "spelling_sound.yaml", [
        {"id": "sp-1", "pattern": "ก", "image": "images/sp-k.jpg"},
    ])
    _write_yaml(d / "media_manifest.yaml", {"entries": [
        {"file": "media/images/pw-1.jpg", "channel": "openverse",
         "origin": "https://example.com/pw-1.jpg", "license": "cc0",
         "fetched": "2026-08-29"},
        {"file": "media/images/sp-k.jpg", "channel": "pexels",
         "origin": "https://example.com/sp-k.jpg", "license": "pexels",
         "fetched": "2026-08-30"},
        {"file": "media/images/missing.jpg", "channel": "pexels",
         "origin": "https://example.com/missing.jpg", "license": "pexels",
         "fetched": "2026-08-30"},
        {"file": "media/audio/pw-1.mp3", "channel": "tts",
         "origin": "https://example.com/pw-1.mp3", "license": "cc0",
         "fetched": "2026-08-29"},  # audio is out of scope; regenerates
    ]})

    cand_dir = d / "work" / "candidates" / "pw-1"
    cand_dir.mkdir(parents=True)
    (cand_dir / "0.jpg").write_bytes(_img_bytes((4, 4), "yellow"))
    (cand_dir / "1.jpg").write_bytes(_img_bytes((4, 4), "orange"))
    _write_yaml(cand_dir / "candidates.yaml", {
        "corpora": ["openverse"],
        "candidates": [
            {"file": "0.jpg", "url": "https://example.com/0.jpg", "source": "openverse",
             "license": "cc0", "passed": False,
             "failed_rules": ["judge/image-irrelevant"], "accepted": False},
            {"file": "1.jpg", "url": "https://example.com/1.jpg", "source": "openverse",
             "license": "cc0", "passed": True, "failed_rules": [], "accepted": True},
            {"file": "missing.jpg", "url": "https://example.com/m.jpg",
             "source": "openverse", "license": "cc0", "passed": False,
             "failed_rules": ["judge/image-irrelevant"], "accepted": False},
        ]})

    cand_dir2 = d / "work" / "candidates" / "pw-2"
    cand_dir2.mkdir(parents=True)
    (cand_dir2 / "0.jpg").write_bytes(_img_bytes((4, 4), "purple"))
    (cand_dir2 / "1.jpg").write_bytes(_img_bytes((4, 4), "cyan"))
    _write_yaml(cand_dir2 / "candidates.yaml", {
        "corpora": ["openverse"],
        "candidates": [
            {"file": "0.jpg", "url": "https://example.com/slow-0.jpg", "source": "openverse",
             "license": "cc0", "passed": True, "failed_rules": [], "accepted": True},
            {"file": "1.jpg", "url": "https://example.com/slow-1.jpg", "source": "openverse",
             "license": "cc0", "passed": False,
             "failed_rules": ["judge/image-irrelevant"], "accepted": False},
        ]})

    _write_jsonl(d / "work" / "forvo_lookups.jsonl", [
        {"word": "ไก่", "items": [{"id": 1}], "fetched": "2026-08-29"},  # ไก่ = chicken
        {"word": "หมา", "items": [], "fetched": "2026-08-30"},  # หมา = dog
        "{not valid json",  # deliberately malformed
    ])

    _write_jsonl(d / "work" / "proof_notes.jsonl", [
        {"index": 1, "note_id": 111, "guid": PW1_GUID, "model": "picture_word",
         "tags": ["stage::words"], "kind": "note", "text": "looks good",
         "ts": "2026-09-02T15:58:34.647168+00:00"},
        {"note_id": 111, "guid": PW1_GUID, "model": "picture_word", "tags": [],
         "kind": "note", "text": "actually reconsider",
         "ts": "2026-09-02T15:59:00.000000+00:00"},
    ])

    _write_yaml(d / "waivers.yaml", [
        {"rule": "pair/exact-confusion", "note_id": "mp-1", "reason": "known issue"},
    ])

    return d


@pytest.fixture
def old_data(tmp_path):
    d = tmp_path / "old_data"
    d.mkdir()
    _write_yaml(d / "word_list_th.yaml", [
        {"id": "chicken", "thai": "ไก่", "gloss": "chicken", "category": "Animals",  # ไก่ = chicken
         "classifier": "ตัว", "picturable": True, "part_of_speech": "noun"},
        {"id": "dog", "thai": "หมา", "gloss": "dog", "category": "Animals",  # หมา = dog
         "picturable": True},
        {"id": "human-directed", "thai": "ผู้ชาย", "gloss": "man", "category": "People",  # ผู้ชาย = man
         "image_query": "a man standing", "image_query_source": "human"},
        {"id": "slow", "thai": "ช้า", "gloss": "slow", "category": "Adjectives"},  # ช้า = slow
        # deliberately malformed: missing 'thai'
        {"id": "broken-row", "gloss": "nothing here"},
    ])
    return d


def test_migration_report_counts_and_no_silent_drops(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    report = migrate(old_deck, old_data, new_root)

    assert isinstance(report, MigrationReport)
    # word list: 5 rows in, 1 malformed -> 4 words + 1 synthesized classifier
    assert report.curated["words"] == 5  # chicken, dog, human-directed, slow, classifier:ตัว
    assert report.curated["targets"] == 4
    assert report.curated["classifier_words_synthesized"] == 1

    # every dropped item has a reason, and the malformed row was NOT silently skipped
    assert report.unmigratable, "expected at least one unmigratable item"
    reasons_by_source = {(u.source, u.identity): u.reason for u in report.unmigratable}
    assert ("data/word_list_th.yaml", "broken-row") in reasons_by_source
    assert "missing" in reasons_by_source[("data/word_list_th.yaml", "broken-row")]

    # spelling-sound notes never migrate
    assert ("notes/spelling_sound.yaml", "sp-1") in reasons_by_source
    assert "graphemes" in reasons_by_source[("notes/spelling_sound.yaml", "sp-1")]

    # media: pw-1.jpg + sp-k.jpg from manifest, missing.jpg dropped w/ reason
    assert report.media["objects_written"] >= 2
    assert any(u.source == "media_manifest.yaml" and "missing.jpg" in u.identity
              for u in report.unmigratable)
    # the one audio entry in the manifest is counted, not silently skipped
    assert report.audio_skipped == 1

    # candidates: pw-1 (1 fail, 1 pass) + pw-2 (1 pass, 1 fail); pw-1's
    # missing.jpg is dropped (candidate image missing on disk)
    assert report.cache["judge_pass"] == 2
    assert report.cache["judge_fail"] == 2
    assert any(u.source == "work/candidates" and "missing.jpg" in u.identity
              for u in report.unmigratable)

    db = SyllabusDb(new_root / "syllabus.db")
    chicken_verdicts = [a for a in db.assessments_of("chicken") if a.backend == "judge"]
    assert chicken_verdicts and all(a.question["kind"] == "picture" for a in chicken_verdicts)

    # forvo: 2 good lines, 1 malformed json dropped
    assert report.cache["forvo"] == 2
    assert any(u.source == "work/forvo_lookups.jsonl" for u in report.unmigratable)

    # proof notes: 2 rows, same guid -> newest-wins on read
    assert report.cache["learner_rating"] == 2

    # waivers
    assert report.cache["waiver"] == 1

    # StudyRecords: none migrate
    assert report.study["records"] == 0

    # no ambiguous (thai, category) forms in the base fixture
    assert report.ambiguous == []


def test_migrated_curated_data_loads_cleanly(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    bundle = curated.load_curated(new_root / "curated")
    ids = {w.id for w in bundle.words}
    assert "chicken" in ids and "dog" in ids and "human-directed" in ids
    assert "classifier:ตัว" in ids
    chicken = next(w for w in bundle.words if w.id == "chicken")
    assert chicken.classifier == "classifier:ตัว"
    assert chicken.pron.corroboration == "curated_exception"  # ipa was found
    dog = next(w for w in bundle.words if w.id == "dog")
    assert dog.pron.corroboration == "disputed"  # no ipa found for หมา


def test_migrated_media_bytes_are_content_addressed(old_deck, old_data, tmp_path):
    import hashlib
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    objects = list((new_root / "media" / "objects").glob("*"))
    assert objects
    for obj in objects:
        assert hashlib.sha256(obj.read_bytes()).hexdigest() == obj.stem


def test_migrate_normalizes_images(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    objects = list((new_root / "media" / "objects").glob("*"))
    assert objects
    saw_bounded = False
    for obj in objects:
        with Image.open(obj) as im:
            assert max(im.size) <= IMAGE_MAX_LONG_EDGE
            if max(im.size) == IMAGE_MAX_LONG_EDGE:
                saw_bounded = True
    assert saw_bounded  # pw-1.jpg (1000x600) was bounded down to the limit


def test_migrate_writes_no_marker(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    db = SyllabusDb(new_root / "syllabus.db")
    assert not any(a.backend == "machine-chosen" for a in db.assessments_of("chicken"))
    assert db.assessments_of("pw-1") == []  # old note id is never a subject


def test_direction_row_migrated_only_for_human_source(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    db = SyllabusDb(new_root / "syllabus.db")
    answers = db.assessments_of("human-directed")
    directions = [a for a in answers if a.answer.get("direction")]
    assert len(directions) == 1
    assert directions[0].answer["direction"] == "a man standing"


def test_waiver_row_is_readable_via_is_waived(old_deck, old_data, tmp_path):
    from thai_syllabus.rules import Finding
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    db = SyllabusDb(new_root / "syllabus.db")
    finding = Finding(rule="pair/exact-confusion", note_id="mp-1", evidence="x")
    assert db.is_waived(finding) is True


def test_proof_notes_newest_wins_via_verdict_history(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    db = SyllabusDb(new_root / "syllabus.db")
    answers = [a for a in db.assessments_of("chicken")
              if a.backend == "learner" and a.question.get("kind") == "note"]
    assert len(answers) == 2
    assert answers[-1].answer["note"] == "actually reconsider"


def test_migration_is_idempotent_wrt_media_cas(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    objects_after_first = sorted((new_root / "media" / "objects").glob("*"))
    migrate(old_deck, old_data, new_root)
    objects_after_second = sorted((new_root / "media" / "objects").glob("*"))
    assert objects_after_first == objects_after_second


def test_migrate_twice_appends_nothing(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    n_cache = _row_count(new_root / "syllabus.db", "cache")
    n_media = _row_count(new_root / "syllabus.db", "media")
    report = migrate(old_deck, old_data, new_root)
    assert _row_count(new_root / "syllabus.db", "cache") == n_cache
    assert _row_count(new_root / "syllabus.db", "media") == n_media
    assert sum(report.already_present.values()) > 0
    assert report.already_present["media"] > 0
    assert report.media.get("objects_written", 0) == 0


def test_does_not_touch_judge_cache_sqlite(old_deck, old_data, tmp_path):
    # judge_cache.sqlite is retired, not migrated (item 5) -- if present it
    # must never be opened/read.
    poison = old_deck / "work" / "judge_cache.sqlite"
    poison.write_bytes(b"not a real sqlite file")
    new_root = tmp_path / "new_root"
    report = migrate(old_deck, old_data, new_root)  # must not raise
    assert isinstance(report, MigrationReport)


def test_candidates_yaml_bare_list_shape_is_handled(old_deck, old_data, tmp_path):
    # An older on-disk layout: candidates.yaml is a bare top-level list,
    # not {"corpora": [...], "candidates": [...]}. "pw-bare" has no
    # corresponding note at all, so its candidate contributes media only
    # (no word id to rank a verdict under) and is itself reported.
    cand_dir = old_deck / "work" / "candidates" / "pw-bare"
    cand_dir.mkdir(parents=True)
    (cand_dir / "0.jpg").write_bytes(_img_bytes((4, 4), "magenta"))
    _write_yaml(cand_dir / "candidates.yaml", [
        {"file": "0.jpg", "url": "https://example.com/0.jpg", "source": "openverse",
         "license": "cc0", "passed": False, "failed_rules": ["judge/image-irrelevant"],
         "accepted": False},
    ])
    new_root = tmp_path / "new_root"
    report = migrate(old_deck, old_data, new_root)
    # pw-1's failed candidate + pw-2's failed candidate = 2; pw-bare's
    # candidate has no word mapping and contributes no verdict.
    assert report.cache["judge_fail"] == 2
    assert any(u.source == "work/candidates" and u.identity == "pw-bare"
              for u in report.unmigratable)


def test_forvo_rows_use_the_spec_3_readable_key(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    db = SyllabusDb(new_root / "syllabus.db")
    answers = [a for a in db.assessments_of("ไก่") if a.backend == "forvo"]  # chicken
    assert len(answers) == 1
    assert answers[0].key == "forvo:ไก่"  # chicken
    assert answers[0].question["kind"] == "recording"  # record.rows_for reads this back


def test_missing_waivers_yaml_is_not_an_error(old_deck, old_data, tmp_path):
    (old_deck / "waivers.yaml").unlink()
    new_root = tmp_path / "new_root"
    report = migrate(old_deck, old_data, new_root)
    assert report.cache.get("waiver", 0) == 0


# --- new for this task: (thai, category) joins, carried verdicts never
# rank, no marker, and idempotence -----------------------------------------

def test_migrate_joins_pictures_by_thai_and_category(old_deck, old_data, tmp_path):
    # ร้อน (hot) covers both an Adjectives sense ("hot") and a Seasons
    # sense ("hot (weather)"); the picture note is filed under Seasons, so
    # it must join "hot-weather", never "hot".
    _append_word_rows(
        old_data,
        {"id": "hot", "thai": "ร้อน", "gloss": "hot", "category": "Adjectives"},
        {"id": "hot-weather", "thai": "ร้อน", "gloss": "hot (weather)", "category": "Seasons"},
    )
    (old_deck / "media" / "images" / "pw-hot.jpg").write_bytes(_img_bytes((4, 4), "brown"))
    notes = old_deck / "notes" / "picture_words.yaml"
    notes.write_text(notes.read_text(encoding="utf-8")
                     + "- id: pw-hot\n  thai: ร้อน\n  category: Seasons\n"
                       "  image: images/pw-hot.jpg\n", encoding="utf-8")
    cand_dir = old_deck / "work" / "candidates" / "pw-hot"
    cand_dir.mkdir(parents=True)
    (cand_dir / "0.jpg").write_bytes(_img_bytes((4, 4), "brown"))
    _write_yaml(cand_dir / "candidates.yaml", {
        "corpora": ["openverse"],
        "candidates": [
            {"file": "0.jpg", "url": "https://example.com/hot-0.jpg", "source": "openverse",
             "license": "cc0", "passed": True, "failed_rules": [], "accepted": True},
        ]})

    report = migrate(old_deck, old_data, tmp_path / "new")
    db = SyllabusDb(tmp_path / "new" / "syllabus.db")
    hot_weather_verdicts = [a for a in db.assessments_of("hot-weather")
                            if a.question.get("rubric") == LEGACY_PICTURE_RUBRIC]
    hot_verdicts = [a for a in db.assessments_of("hot")
                   if a.question.get("rubric") == LEGACY_PICTURE_RUBRIC]
    assert hot_weather_verdicts and not hot_verdicts
    assert report.ambiguous == []


def test_migrate_reports_a_form_ambiguous_under_the_key(old_deck, old_data, tmp_path):
    # เย็น covers both "cool" and "cold" -- two Adjectives rows share the
    # exact same (thai, category) key, so the key joins nothing.
    _append_word_rows(
        old_data,
        {"id": "cool", "thai": "เย็น", "gloss": "cool", "category": "Adjectives"},
        {"id": "cold", "thai": "เย็น", "gloss": "cold", "category": "Adjectives"},
    )
    report = migrate(old_deck, old_data, tmp_path / "new")
    assert report.ambiguous == [("เย็น", "Adjectives")]


def test_carried_verdicts_do_not_rank(old_deck, old_data, tmp_path):
    migrate(old_deck, old_data, tmp_path / "new")
    db = SyllabusDb(tmp_path / "new" / "syllabus.db")
    rows = [a for a in db.assessments_of("slow")
           if a.question.get("rubric") == LEGACY_PICTURE_RUBRIC]
    assert rows
    best = current_best(db, "slow", "picture",
                        current_rubric={"picture-for-word": PICTURE_FIT_RUBRIC})
    assert best.artifact_sha is None


def test_unmapped_note_thai_is_reported(old_deck, old_data, tmp_path):
    notes = old_deck / "notes" / "picture_words.yaml"
    notes.write_text(notes.read_text(encoding="utf-8")
                     + "- id: pw-9\n  thai: ไม่มี\n  category: Animals\n"  # ไม่มี = "none"
                       "  image: images/pw-1.jpg\n", encoding="utf-8")
    report = migrate(old_deck, old_data, tmp_path / "new")
    assert any(u.identity == "pw-9" and "no word" in u.reason for u in report.unmigratable)


def test_proof_note_lands_under_the_word_id(old_deck, old_data, tmp_path):
    guid = genanki.guid_for("picture_word", "pw-2")
    (old_deck / "work" / "proof_notes.jsonl").write_text(
        '{"guid": "%s", "model": "picture_word", "note_id": 1, "tags": [], "text": "looks fine", '
        '"ts": "2026-09-02T15:58:34+00:00"}\n' % guid, encoding="utf-8")
    migrate(old_deck, old_data, tmp_path / "new")
    db = SyllabusDb(tmp_path / "new" / "syllabus.db")
    rows = [r for r in db.assessments_of("slow") if r.backend == "learner" and r.question.get("kind") == "note"]
    assert rows and rows[0].answer["note"] == "looks fine"


def test_guid_matching_no_picture_word_note_is_reported(old_deck, old_data, tmp_path):
    (old_deck / "work" / "proof_notes.jsonl").write_text(
        '{"guid": "not-a-real-guid", "model": "picture_word", "note_id": 1, "tags": [], '
        '"text": "orphan", "ts": "2026-09-02T15:58:34+00:00"}\n', encoding="utf-8")
    report = migrate(old_deck, old_data, tmp_path / "new")
    assert any(u.source == "work/proof_notes.jsonl" and u.identity == "not-a-real-guid"
              and "guid matches no picture-word note" in u.reason
              for u in report.unmigratable)


def test_word_list_row_without_a_category_is_reported_and_refused(old_deck, old_data, tmp_path):
    _append_word_rows(old_data, {"id": "cat-less", "thai": "แมว", "gloss": "cat"})  # แมว = cat
    report = migrate(old_deck, old_data, tmp_path / "new")

    dropped = [u for u in report.unmigratable
              if u.source == "data/word_list_th.yaml" and u.identity == "cat-less"]
    assert len(dropped) == 1
    assert "category" in dropped[0].reason

    bundle = curated.load_curated(tmp_path / "new" / "curated")
    assert "cat-less" not in {w.id for w in bundle.words}
