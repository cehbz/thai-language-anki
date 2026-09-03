"""Tests for store.py (spec 2 section 2/3): SyllabusDb over sqlite (WAL,
one transaction per append), the four tables, the RecordWriter/
AssessmentReader/StudyReader read/write surface, and MediaStore's
content-addressed writes.
"""
import sqlite3
from datetime import date

import pytest

from thai_syllabus.ports import Answer, StudyRecord
from thai_syllabus.rules import Finding
from thai_syllabus.store import MediaStore, SyllabusDb


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


# --- schema / WAL -----------------------------------------------------

def test_creates_the_four_tables(db):
    con = sqlite3.connect(db.path)
    names = {row[0] for row in
             con.execute("select name from sqlite_master where type='table'")}
    assert {"sentences", "media", "cache", "study"} <= names


def test_journal_mode_is_wal(db):
    con = sqlite3.connect(db.path)
    mode = con.execute("pragma journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_reopening_an_existing_db_does_not_lose_data(tmp_path):
    path = tmp_path / "syllabus.db"
    db1 = SyllabusDb(path)
    db1.append(port="assess", backend="judge", key="k1", subject="s1",
               question={"q": 1}, answer={"a": 1})
    db2 = SyllabusDb(path)
    assert len(db2.assessments_of("s1")) == 1


# --- cache / RecordWriter / AssessmentReader ---------------------------

def test_append_is_readable_via_assessments_of(db):
    db.append(port="assess", backend="judge", key="k1", subject="subj-1",
              question={"rule": "r"}, answer={"verdict": True}, cost=0.5)
    answers = db.assessments_of("subj-1")
    assert len(answers) == 1
    a = answers[0]
    assert isinstance(a, Answer)
    assert a.port == "assess" and a.backend == "judge"
    assert a.question == {"rule": "r"}
    assert a.answer == {"verdict": True}
    assert a.cost == 0.5


def test_cache_row_keeps_the_readable_key_alongside_its_hash(db):
    db.append(port="provide", backend="forvo", key="forvo:ไก่", subject="ไก่",
              question={"word": "ไก่"}, answer={"items": []})  # forvo:chicken
    answer = db.assessments_of("ไก่")[0]
    assert answer.key == "forvo:ไก่"  # chicken
    import hashlib
    assert answer.key_sha == hashlib.sha256("forvo:ไก่".encode()).hexdigest()


def test_append_returns_the_ts_the_row_was_written_under(db):
    ts = db.append(port="provide", backend="forvo", key="forvo:x", subject="x",
                   question={}, answer={"items": []})
    answer = db.assessments_of("x")[0]
    assert answer.ts == ts


# --- CacheReader.latest(): the cache-first hit lookup ------------------

def test_latest_is_none_on_a_cache_miss(db):
    assert db.latest("provide", "forvo", "forvo:missing") is None


def test_latest_returns_the_newest_row_for_an_exact_key(db):
    db.append(port="provide", backend="forvo", key="forvo:ไก่", subject="ไก่",
              question={}, answer={"items": [1]})  # chicken
    db.append(port="provide", backend="forvo", key="forvo:ไก่", subject="ไก่",
              question={}, answer={"items": [2]})
    hit = db.latest("provide", "forvo", "forvo:ไก่")
    assert hit.answer == {"items": [2]}


def test_latest_does_not_cross_backends_on_the_same_key_text(db):
    # same literal key string, different backend -- must not collide.
    db.append(port="provide", backend="forvo", key="same-key", subject="s",
              question={}, answer={"items": ["forvo"]})
    db.append(port="provide", backend="tts", key="same-key", subject="s",
              question={}, answer={"items": ["tts"]})
    assert db.latest("provide", "forvo", "same-key").answer == {"items": ["forvo"]}
    assert db.latest("provide", "tts", "same-key").answer == {"items": ["tts"]}


def test_satisfies_the_cache_reader_protocol(db):
    from thai_syllabus.ports import CacheReader
    assert isinstance(db, CacheReader)


def test_reask_appends_a_new_row_never_updates(db):
    db.append(port="assess", backend="learner", key="k1", subject="subj-1",
              question={"q": 1}, answer={"rating": "good"})
    db.append(port="assess", backend="learner", key="k1", subject="subj-1",
              question={"q": 1}, answer={"rating": "bad"})
    con = sqlite3.connect(db.path)
    n = con.execute("select count(*) from cache").fetchone()[0]
    assert n == 2


def test_assessments_of_orders_newest_last(db):
    db.append(port="assess", backend="learner", key="k1", subject="subj-1",
              question={}, answer={"n": 1})
    db.append(port="assess", backend="learner", key="k1", subject="subj-1",
              question={}, answer={"n": 2})
    answers = db.assessments_of("subj-1")
    assert [a.answer["n"] for a in answers] == [1, 2]
    assert answers[0].ts <= answers[-1].ts


def test_verdict_is_exact_key_newest_row(db):
    # two different (rule, note_id) pairs must not collide
    db.append_judge_verdict(rule_id="pair/exact-confusion", note_id="mp-1",
                            verdict=False)
    db.append_judge_verdict(rule_id="pair/exact-confusion", note_id="mp-1",
                            verdict=True)  # re-judged, newest wins
    db.append_judge_verdict(rule_id="pair/exact-confusion", note_id="mp-2",
                            verdict=False)
    assert db.verdict("pair/exact-confusion", "mp-1") is True
    assert db.verdict("pair/exact-confusion", "mp-2") is False
    assert db.verdict("pair/exact-confusion", "unknown") is None


def test_verdict_keys_on_artifact_sha_too(db):
    db.append_judge_verdict(rule_id="media/picture-fit", note_id="w-1",
                            artifact_sha="aaa", verdict=True)
    db.append_judge_verdict(rule_id="media/picture-fit", note_id="w-1",
                            artifact_sha="bbb", verdict=False)
    assert db.verdict("media/picture-fit", "w-1", "aaa") is True
    assert db.verdict("media/picture-fit", "w-1", "bbb") is False
    assert db.verdict("media/picture-fit", "w-1") is None


def test_is_waived_reads_learner_waiver_rows(db):
    finding = Finding(rule="pair/exact-confusion", note_id="mp-1",
                      evidence="bad pair")
    assert db.is_waived(finding) is False
    db.append_waiver(rule_id="pair/exact-confusion", note_id="mp-1",
                     artifact_sha=None, waived=True, reason="known issue")
    assert db.is_waived(finding) is True


def test_is_waived_newest_wins(db):
    finding = Finding(rule="pair/exact-confusion", note_id="mp-1",
                      evidence="bad pair")
    db.append_waiver(rule_id="pair/exact-confusion", note_id="mp-1",
                     artifact_sha=None, waived=True, reason="waived")
    db.append_waiver(rule_id="pair/exact-confusion", note_id="mp-1",
                     artifact_sha=None, waived=False, reason="reopened")
    assert db.is_waived(finding) is False


def test_satisfies_the_assessment_reader_protocol(db):
    from thai_syllabus.ports import AssessmentReader
    assert isinstance(db, AssessmentReader)


def test_satisfies_record_writer_protocol(db):
    from thai_syllabus.ports import RecordWriter
    assert isinstance(db, RecordWriter)


def test_satisfies_study_reader_protocol(db):
    from thai_syllabus.ports import StudyReader
    assert isinstance(db, StudyReader)


# --- study ---------------------------------------------------------------

def test_append_study_and_read_back_by_card_key(db):
    db.append_study(card_key="target:cheap:picture_card", compile_id="c1",
                    grade=3, time_ms=1200)
    records = db.records("target:cheap:picture_card")
    assert len(records) == 1
    assert isinstance(records[0], StudyRecord)
    assert records[0].grade == 3
    assert records[0].compile_id == "c1"


def test_study_is_append_only(db):
    db.append_study(card_key="k1", compile_id="c1", grade=1, time_ms=100)
    db.append_study(card_key="k1", compile_id="c1", grade=3, time_ms=200)
    assert len(db.records("k1")) == 2
    con = sqlite3.connect(db.path)
    assert con.execute("select count(*) from study").fetchone()[0] == 2


def test_records_by_confusion_aggregates_pair_card_keys(db):
    db.set_pair_confusions({"mp-1": "tone:mid-low", "mp-2": "tone:mid-low",
                            "mp-3": "aspiration:labial"})
    db.append_study(card_key="mp-1::pair_card", compile_id="c1", grade=2,
                    time_ms=500)
    db.append_study(card_key="mp-2::pair_card", compile_id="c1", grade=4,
                    time_ms=700)
    db.append_study(card_key="mp-3::pair_card", compile_id="c1", grade=1,
                    time_ms=300)
    records = db.records("tone:mid-low")
    assert {r.card_key for r in records} == {"mp-1::pair_card", "mp-2::pair_card"}


# --- sentences / media provenance --------------------------------------

def test_add_sentence_and_read_back(db):
    db.add_sentence(text_sha="abc123", text="text", voice="learner_voice",
                    source="llm", origin="draft", licence="n/a",
                    acquired=date(2026, 1, 1))
    con = sqlite3.connect(db.path)
    row = con.execute("select text_sha, text, voice from sentences").fetchone()
    assert row == ("abc123", "text", "learner_voice")


def test_add_media_provenance_idempotent(db):
    db.add_media(sha="deadbeef", kind="picture", ext="jpg", source="openverse",
                origin="https://example.com/x.jpg", licence="cc0",
                acquired=date(2026, 1, 1))
    db.add_media(sha="deadbeef", kind="picture", ext="jpg", source="openverse",
                origin="https://example.com/x.jpg", licence="cc0",
                acquired=date(2026, 1, 1))
    con = sqlite3.connect(db.path)
    assert con.execute("select count(*) from media").fetchone()[0] == 1


# --- MediaStore (CAS) -----------------------------------------------------

def test_media_store_writes_content_addressed(tmp_path):
    store = MediaStore(tmp_path)
    sha = store.write(b"hello world", ext="jpg")
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert sha == expected
    assert (tmp_path / "objects" / f"{expected}.jpg").read_bytes() == b"hello world"


def test_media_store_write_is_idempotent(tmp_path):
    store = MediaStore(tmp_path)
    sha1 = store.write(b"same bytes", ext="png")
    sha2 = store.write(b"same bytes", ext="png")
    assert sha1 == sha2
    files = list((tmp_path / "objects").glob(f"{sha1}.*"))
    assert len(files) == 1


def test_media_store_has(tmp_path):
    store = MediaStore(tmp_path)
    assert not store.has("nonexistent", ext="jpg")
    sha = store.write(b"data", ext="jpg")
    assert store.has(sha, ext="jpg")
