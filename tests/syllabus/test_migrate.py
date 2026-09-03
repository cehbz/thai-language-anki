"""Tests for migrate.py (spec 2 section 4): the one-shot migration, run
against a small synthetic old-deck fixture built under tmp_path. Never
touches ~/decks or the real data/ directory.
"""
import json

import pytest
import yaml

from thai_syllabus import curated
from thai_syllabus.migrate import MigrationReport, migrate
from thai_syllabus.store import SyllabusDb


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


@pytest.fixture
def old_deck(tmp_path):
    d = tmp_path / "old_deck"
    (d / "media" / "images").mkdir(parents=True)
    (d / "media" / "images" / "pw-1.jpg").write_bytes(b"picture-bytes-1")
    (d / "media" / "images" / "sp-k.jpg").write_bytes(b"picture-bytes-sp-k")

    _write_yaml(d / "notes" / "picture_words.yaml", [
        {"id": "pw-1", "thai": "ไก่", "ipa": "kaj˨˩", "image": "images/pw-1.jpg"},
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
    ]})

    cand_dir = d / "work" / "candidates" / "pw-1"
    cand_dir.mkdir(parents=True)
    (cand_dir / "0.jpg").write_bytes(b"candidate-0")
    (cand_dir / "1.jpg").write_bytes(b"candidate-1-passed")
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

    _write_jsonl(d / "work" / "forvo_lookups.jsonl", [
        {"word": "ไก่", "items": [{"id": 1}], "fetched": "2026-08-29"},
        {"word": "หมา", "items": [], "fetched": "2026-08-30"},
        "{not valid json",  # deliberately malformed
    ])

    _write_jsonl(d / "work" / "proof_notes.jsonl", [
        {"index": 1, "note_id": 111, "guid": "abc123", "model": "picture_word",
         "tags": ["stage::words"], "kind": "note", "text": "looks good",
         "ts": "2026-09-02T15:58:34.647168+00:00"},
        {"note_id": 111, "guid": "abc123", "model": "picture_word", "tags": [],
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
        {"id": "chicken", "thai": "ไก่", "gloss": "chicken", "category": "Animals",
         "classifier": "ตัว", "picturable": True, "part_of_speech": "noun"},
        {"id": "dog", "thai": "หมา", "gloss": "dog", "category": "Animals",
         "picturable": True},
        {"id": "human-directed", "thai": "ผู้ชาย", "gloss": "man",
         "image_query": "a man standing", "image_query_source": "human"},
        # deliberately malformed: missing 'thai'
        {"id": "broken-row", "gloss": "nothing here"},
    ])
    return d


def test_migration_report_counts_and_no_silent_drops(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    report = migrate(old_deck, old_data, new_root)

    assert isinstance(report, MigrationReport)
    # word list: 4 rows in, 1 malformed -> 3 words + 1 synthesized classifier
    assert report.curated["words"] == 4  # chicken, dog, human-directed, classifier:ตัว
    assert report.curated["targets"] == 3
    assert report.curated["classifier_words_synthesized"] == 1

    # every dropped item has a reason, and the malformed row was NOT silently skipped
    assert report.unmigratable, "expected at least one unmigratable item"
    reasons_by_source = {(u.source, u.identity): u.reason for u in report.unmigratable}
    assert ("data/word_list_th.yaml", "broken-row") in reasons_by_source
    assert "missing" in reasons_by_source[("data/word_list_th.yaml", "broken-row")]

    # media: pw-1.jpg + sp-k.jpg from manifest, missing.jpg dropped w/ reason
    assert report.media["objects_written"] >= 2
    assert any(u.source == "media_manifest.yaml" and "missing.jpg" in u.identity
              for u in report.unmigratable)

    # candidates: one failed (judge row), one passed (no rubric to key under),
    # one missing on disk (dropped)
    assert report.cache["judge"] == 1
    assert report.media.get("candidates_passed_no_recorded_rubric") == 1
    assert any(u.source == "work/candidates" and "missing.jpg" in u.identity
              for u in report.unmigratable)

    # forvo: 2 good lines, 1 malformed json dropped
    assert report.cache["forvo"] == 2
    assert any(u.source == "work/forvo_lookups.jsonl" for u in report.unmigratable)

    # proof notes: 2 rows, same guid -> newest-wins on read
    assert report.cache["learner_rating"] == 2

    # waivers
    assert report.cache["waiver"] == 1

    # StudyRecords: none migrate
    assert report.study["records"] == 0


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
    sha = hashlib.sha256(b"picture-bytes-1").hexdigest()
    assert (new_root / "media" / "objects" / f"{sha}.jpg").read_bytes() == b"picture-bytes-1"


def test_machine_chosen_marker_on_current_deck_images(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    db = SyllabusDb(new_root / "syllabus.db")
    answers = db.assessments_of("pw-1")
    kinds = [a.answer.get("marker") for a in answers if a.backend == "machine-chosen"]
    assert "machine-chosen" in kinds


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
    answers = db.assessments_of("abc123")
    assert len(answers) == 2
    assert answers[-1].answer["note"] == "actually reconsider"


def test_migration_is_idempotent_wrt_media_cas(old_deck, old_data, tmp_path):
    new_root = tmp_path / "new_root"
    migrate(old_deck, old_data, new_root)
    objects_after_first = sorted((new_root / "media" / "objects").glob("*"))
    migrate(old_deck, old_data, new_root)
    objects_after_second = sorted((new_root / "media" / "objects").glob("*"))
    assert objects_after_first == objects_after_second


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
    # not {"corpora": [...], "candidates": [...]}.
    cand_dir = old_deck / "work" / "candidates" / "pw-bare"
    cand_dir.mkdir(parents=True)
    (cand_dir / "0.jpg").write_bytes(b"bare-candidate-0")
    _write_yaml(cand_dir / "candidates.yaml", [
        {"file": "0.jpg", "url": "https://example.com/0.jpg", "source": "openverse",
         "license": "cc0", "passed": False, "failed_rules": ["judge/image-irrelevant"],
         "accepted": False},
    ])
    new_root = tmp_path / "new_root"
    report = migrate(old_deck, old_data, new_root)
    assert report.cache["judge"] == 2  # 1 from pw-1's candidates + 1 from pw-bare


def test_missing_waivers_yaml_is_not_an_error(old_deck, old_data, tmp_path):
    (old_deck / "waivers.yaml").unlink()
    new_root = tmp_path / "new_root"
    report = migrate(old_deck, old_data, new_root)
    assert report.cache.get("waiver", 0) == 0
