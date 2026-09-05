"""Tests for store.py (spec 2 section 2/3): SyllabusDb over sqlite (WAL,
one transaction per append), the five tables, the RecordWriter/
AssessmentReader/StudyReader read/write surface, and MediaStore's
content-addressed writes.
"""
import sqlite3
from datetime import date

import pytest

from thai_syllabus.cachekeys import JudgeKey, WaiverKey, sha
from thai_syllabus.media import Speaker
from thai_syllabus.ports import Answer, StudyRecord
from thai_syllabus.rules import Finding
from thai_syllabus.store import MediaStore, SyllabusDb


def _append_judge_verdict(db, *, rule_id, note_id, verdict, artifact_sha=None,
                          rubric=None, evidence=None, cost=0.0):
    """Test helper: builds the JudgeKey the way Syllabus.report() does
    (JudgeKey.for_rule) and calls the new key-taking append_judge_verdict.
    """
    key = JudgeKey.for_rule(rubric, artifact_sha, note_id, rule_id)
    answer = {"value": verdict}
    if evidence is not None:
        answer["evidence"] = evidence
    db.append_judge_verdict(key=key, subject=note_id,
                            question={"role": rule_id, "artifact_sha": artifact_sha,
                                     "rubric": rubric},
                            answer=answer, cost=cost)


def _judge_key(rule_id, note_id, artifact_sha=None, rubric=None):
    return JudgeKey.for_rule(rubric, artifact_sha, note_id, rule_id)


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


def test_store_does_not_build_keys():
    import thai_syllabus.store as s
    assert not hasattr(s, "_judge_verdict_key")
    assert not hasattr(s, "_finding_key")
    assert not hasattr(s, "_finding_subject")


def test_waiver_is_visible_under_the_note_subject(db):
    key = WaiverKey(rule_id="picture/fit", note_id="rice", artifact_sha="a" * 64)
    db.append("assess", "learner", key, "rice", {"kind": "waiver"}, {"waived": True}, 0.0)
    assert any(a.question.get("kind") == "waiver" for a in db.assessments_of("rice"))


def test_judge_backend_key_equals_cachekeys(db):
    from thai_syllabus.assessor import AssessQuestion, JudgeBackend
    q = AssessQuestion(subject="rice", role="picture-for-word", artifact_sha="a" * 64, rubric="R")
    backend = JudgeBackend(model="m", transport="cli", complete=lambda p: "true")
    assert backend.cache_key(q) == JudgeKey(rubric_sha=sha("R"), identity="a" * 64,
                                            role="picture-for-word")


# --- schema / WAL -----------------------------------------------------

def test_creates_the_five_tables(db):
    con = sqlite3.connect(db.path)
    names = {row[0] for row in
             con.execute("select name from sqlite_master where type='table'")}
    assert {"sentences", "media", "speakers", "cache", "study"} <= names


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


def test_rows_since_returns_that_port_and_backend_from_the_window_onward(db):
    db.append(port="provide", backend="forvo", key="k1", subject="rice",
              question={"kind": "recording"}, answer={"items": []}, cost=1.0, ts=100)
    db.append(port="provide", backend="forvo", key="k2", subject="fish",
              question={"kind": "recording"}, answer={"items": []}, cost=2.0, ts=300)
    db.append(port="provide", backend="tts", key="k3", subject="rice",
              question={"kind": "recording"}, answer={"items": []}, cost=0.0, ts=300)
    db.append(port="assess", backend="forvo", key="k4", subject="rice",
              question={"kind": "recording"}, answer={"value": True}, cost=0.0, ts=300)
    rows = db.rows_since("provide", "forvo", 200)
    assert [(r.subject, r.cost) for r in rows] == [("fish", 2.0)]
    assert [r.subject for r in db.rows_since("provide", "forvo", 0)] == ["rice", "fish"]


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
    _append_judge_verdict(db, rule_id="pair/exact-confusion", note_id="mp-1", verdict=False)
    _append_judge_verdict(db, rule_id="pair/exact-confusion", note_id="mp-1",
                          verdict=True)  # re-judged, newest wins
    _append_judge_verdict(db, rule_id="pair/exact-confusion", note_id="mp-2", verdict=False)
    assert db.verdict("judge", _judge_key("pair/exact-confusion", "mp-1")).answer["value"] is True
    assert db.verdict("judge", _judge_key("pair/exact-confusion", "mp-2")).answer["value"] is False
    assert db.verdict("judge", _judge_key("pair/exact-confusion", "unknown")) is None


def test_verdict_distinguishes_notes_with_no_artifact_sha(db):
    # Two different notes judged under the same rule and no artifact_sha
    # must not collide -- the merged spec-3 key convention falls back to
    # note_id (see store.py's module docstring), not a shared placeholder.
    _append_judge_verdict(db, rule_id="sentence/register-natural", note_id="s-1", verdict=True)
    _append_judge_verdict(db, rule_id="sentence/register-natural", note_id="s-2", verdict=False)
    assert db.verdict("judge", _judge_key("sentence/register-natural", "s-1")).answer["value"] is True
    assert db.verdict("judge", _judge_key("sentence/register-natural", "s-2")).answer["value"] is False


def test_verdict_key_matches_the_spec_3_judge_backend_convention(db):
    # Same key shape as assessor.JudgeBackend.cache_key: role=rule_id,
    # rubric folded in -- so a judged Rule's verdict and a direct
    # Assessor.ask("judge", ...) call under the same (rubric, role,
    # artifact_sha-or-subject) land on the SAME cache row (one convention).
    from thai_syllabus.assessor import AssessQuestion, JudgeBackend
    backend = JudgeBackend(model="m", transport="cli", complete=lambda p: "true")
    q = AssessQuestion(subject="mp-1", role="pair/exact-confusion",
                       artifact_sha="aaa", rubric="is this pair exact?")
    expected_key = backend.cache_key(q)
    _append_judge_verdict(db, rule_id="pair/exact-confusion", note_id="mp-1",
                          artifact_sha="aaa", verdict=True,
                          rubric="is this pair exact?")
    row = db._con.execute(  # white-box: confirm it landed under the shared key
        "select key from cache where port='assess' and backend='judge'").fetchone()
    assert row[0] == expected_key.encode()
    answer = db.verdict("judge", _judge_key("pair/exact-confusion", "mp-1", "aaa",
                                            rubric="is this pair exact?"))
    assert answer.answer["value"] is True


def test_verdict_keys_on_artifact_sha_too(db):
    _append_judge_verdict(db, rule_id="media/picture-fit", note_id="w-1",
                          artifact_sha="aaa", verdict=True)
    _append_judge_verdict(db, rule_id="media/picture-fit", note_id="w-1",
                          artifact_sha="bbb", verdict=False)
    assert db.verdict("judge", _judge_key("media/picture-fit", "w-1", "aaa")).answer["value"] is True
    assert db.verdict("judge", _judge_key("media/picture-fit", "w-1", "bbb")).answer["value"] is False
    assert db.verdict("judge", _judge_key("media/picture-fit", "w-1")) is None


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


def test_append_study_with_an_explicit_ts_stores_it_verbatim(db):
    # anki_import.py's revlog import needs the STORED ts to be exactly the
    # revlog row's own id (an epoch-ms review timestamp) for "idempotent
    # by (card_key, ts)" (spec 4 section 4) to mean anything on reimport
    # -- _next_ts's monotonic-bump-for-collision-avoidance behavior (built
    # for the cache table's own ts source, epoch nanoseconds) must NOT
    # silently renumber an explicitly supplied ts.
    db.append_study(card_key="k1", compile_id="c1", grade=3, time_ms=100,
                    ts=1_700_000_000_000)
    records = db.records("k1")
    assert records[0].ts == 1_700_000_000_000


def test_append_study_with_an_explicit_ts_does_not_disturb_auto_generated_ones(db):
    db.append_study(card_key="k1", compile_id="c1", grade=1, time_ms=100)
    auto_ts = db.records("k1")[0].ts
    db.append_study(card_key="k1", compile_id="c1", grade=2, time_ms=100,
                    ts=1)  # far smaller than the nanosecond auto ts above
    records = sorted(db.records("k1"), key=lambda r: r.ts)
    assert records[0].ts == 1
    assert records[1].ts == auto_ts


def test_study_is_append_only(db):
    db.append_study(card_key="k1", compile_id="c1", grade=1, time_ms=100)
    db.append_study(card_key="k1", compile_id="c1", grade=3, time_ms=200)
    assert len(db.records("k1")) == 2
    con = sqlite3.connect(db.path)
    assert con.execute("select count(*) from study").fetchone()[0] == 2


def test_append_study_ignores_a_duplicate(db):
    assert db.append_study(card_key="rice::listening", compile_id="c1", ts=5,
                           grade=3, time_ms=900)
    assert not db.append_study(card_key="rice::listening", compile_id="c1", ts=5,
                               grade=3, time_ms=900)
    assert len(db.records("rice::listening")) == 1


def test_append_study_with_a_new_ts_for_a_known_card_key_is_not_a_duplicate(db):
    assert db.append_study(card_key="rice::listening", compile_id="c1", ts=5,
                           grade=3, time_ms=900)
    assert db.append_study(card_key="rice::listening", compile_id="c1", ts=6,
                           grade=4, time_ms=800)
    assert len(db.records("rice::listening")) == 2


def test_study_rows_returns_every_row_ordered_by_ts(db):
    db.append_study(card_key="k2", compile_id="c1", ts=2, grade=1, time_ms=100)
    db.append_study(card_key="k1", compile_id="c1", ts=1, grade=2, time_ms=100)
    rows = db.study_rows()
    assert [r.card_key for r in rows] == ["k1", "k2"]
    assert [r.ts for r in rows] == [1, 2]


# --- sentences / media provenance --------------------------------------

def test_add_sentence_and_read_back(db):
    db.add_sentence(text_sha="abc123", text="text", gloss="a gloss",
                    voice="learner_voice", source="llm", origin="draft",
                    licence="n/a", acquired=date(2026, 1, 1))
    con = sqlite3.connect(db.path)
    row = con.execute("select text_sha, text, gloss, voice from sentences").fetchone()
    assert row == ("abc123", "text", "a gloss", "learner_voice")


def test_add_sentence_is_idempotent_and_returns_whether_inserted(db):
    assert db.add_sentence(text_sha="dup", text="text", gloss="a gloss",
                           voice="learner_voice", source="llm", origin="draft",
                           licence="n/a", acquired=date(2026, 1, 1)) is True
    assert db.add_sentence(text_sha="dup", text="text", gloss="a gloss",
                           voice="learner_voice", source="llm", origin="draft",
                           licence="n/a", acquired=date(2026, 1, 1)) is False
    con = sqlite3.connect(db.path)
    assert con.execute("select count(*) from sentences").fetchone()[0] == 1


def test_sentences_table_round_trips_gloss(db):
    db.add_sentence(text_sha="x" * 64, text="ผมกินข้าว", gloss="I eat rice",  # I eat rice
                    voice="learner_voice", source="llm", origin="o", licence="cc",
                    acquired=date(2026, 9, 4))
    assert db.all_sentences()[0].gloss == "I eat rice"


def test_all_sentences_reads_back_as_entities(db):
    from thai_syllabus.entities import Sentence
    from thai_syllabus.media import Provenance

    db.add_sentence(text_sha="s1", text="ผมกินข้าว", gloss="I eat rice",  # I eat rice
                    voice="learner_voice", source="llm", origin="draft", licence="n/a",
                    acquired=date(2026, 1, 1))
    db.add_sentence(text_sha="s2", text="เขากินข้าว", gloss="(s)he eats rice",  # (s)he eats rice
                    voice="other_voice", source="forvo", origin="https://forvo.com/x",
                    licence="cc-by", acquired=date(2026, 2, 2))
    sentences = db.all_sentences()
    assert len(sentences) == 2
    assert all(isinstance(s, Sentence) for s in sentences)
    by_text = {s.text: s for s in sentences}
    assert by_text["ผมกินข้าว"].voice == "learner_voice"  # I eat rice
    assert by_text["ผมกินข้าว"].gloss == "I eat rice"  # I eat rice
    assert by_text["ผมกินข้าว"].provenance == Provenance(
        source="llm", origin="draft", licence="n/a", acquired=date(2026, 1, 1))
    assert by_text["เขากินข้าว"].provenance.origin == "https://forvo.com/x"  # (s)he eats rice


def test_all_sentences_on_empty_store_is_empty(db):
    assert db.all_sentences() == []


# --- speakers -----------------------------------------------------------

def test_add_media_requires_a_known_speaker(db):
    with pytest.raises(ValueError, match="speaker"):
        db.add_media(sha="a" * 64, kind="recording", ext="mp3", source="forvo",
                    origin="u", licence="cc", acquired=date(2026, 9, 4),
                    speaker_id="forvo:somchai")
    db.add_speaker(Speaker(id="forvo:somchai", kind="native", sex="male", region="TH"))
    assert db.add_media(sha="a" * 64, kind="recording", ext="mp3", source="forvo",
                        origin="u", licence="cc", acquired=date(2026, 9, 4),
                        speaker_id="forvo:somchai")
    assert db.speaker("forvo:somchai").sex == "male"


def test_add_speaker_is_insert_or_ignore(db):
    db.add_speaker(Speaker(id="forvo:somchai", kind="native", sex="male"))
    db.add_speaker(Speaker(id="forvo:somchai", kind="native", sex="female"))
    assert db.speaker("forvo:somchai").sex == "male"


def test_speaker_returns_none_for_an_unknown_id(db):
    assert db.speaker("nope") is None


def test_add_media_provenance_idempotent(db):
    db.add_media(sha="deadbeef", kind="picture", ext="jpg", source="openverse",
                origin="https://example.com/x.jpg", licence="cc0",
                acquired=date(2026, 1, 1))
    db.add_media(sha="deadbeef", kind="picture", ext="jpg", source="openverse",
                origin="https://example.com/x.jpg", licence="cc0",
                acquired=date(2026, 1, 1))
    con = sqlite3.connect(db.path)
    assert con.execute("select count(*) from media").fetchone()[0] == 1


def test_media_provenance_returns_none_for_an_unknown_sha(db):
    assert db.media_provenance("nope") is None


def test_media_provenance_returns_the_stored_row(db):
    db.add_speaker(Speaker(id="somchai", kind="native"))
    db.add_media(sha="deadbeef", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id="somchai")
    prov = db.media_provenance("deadbeef")
    assert prov["ext"] == "mp3"
    assert prov["source"] == "forvo"
    assert prov["speaker_id"] == "somchai"
    assert prov["speaker"] == Speaker(id="somchai", kind="native")


def test_media_provenance_speaker_is_none_when_no_speaker_id_is_on_file(db):
    db.add_media(sha="deadbeef", kind="picture", ext="jpg", source="openverse",
                origin="https://example.com/x.jpg", licence="cc0",
                acquired=date(2026, 1, 1))
    assert db.media_provenance("deadbeef")["speaker"] is None


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


# --- MediaStore.add_image: ingest normalization (spec 4 section 3) --------
# Pillow is a hard dependency; undecodable bytes refuse rather than being
# stored raw. The real-Pillow normalization behavior is covered, gated on
# Pillow being installed, in test_media_normalization.py.

def test_add_image_refuses_undecodable_bytes(tmp_path):
    media_store = MediaStore(tmp_path)
    with pytest.raises(ValueError, match="decode"):
        media_store.add_image(b"not really an image", ext="jpg")
