"""Tests for assessor.py (spec 3 sections 1-2): Assessor.ask()'s cache-first
shape, judge's three transports behind one implementation (incl. batch
persistence/resume), the mechanical backend, and the learner/listener
non-implementations. Real SyllabusDb for cache-first behavior; fakes for
transports -- no network, no subprocess, no anthropic import.
"""
import pytest

from thai_syllabus.assessor import (
    AUTHORITY_ORDER,
    AssessQuestion,
    Assessor,
    BatchPending,
    JudgeBackend,
    LearnerAskNotSupported,
    RawVerdict,
    Verdict,
    duration_mechanical_backend,
    format_mechanical_backend,
)
from thai_syllabus.store import SyllabusDb
from thai_syllabus.transport import TransportError


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _FakeBackend:
    def __init__(self, key="k", raises=None, value=True, evidence=None,
                suggestion=None, cost=0.0):
        self.key = key
        self.raises = raises
        self.value = value
        self.evidence = evidence
        self.suggestion = suggestion
        self.cost = cost
        self.fetch_calls = 0

    def cache_key(self, question):
        return self.key

    def fetch(self, question):
        self.fetch_calls += 1
        if self.raises:
            raise self.raises
        return RawVerdict(value=self.value, evidence=self.evidence,
                          suggestion=self.suggestion, cost=self.cost)


# --- cache-first shape --------------------------------------------------

def test_a_miss_executes_and_appends_one_row(db):
    backend = _FakeBackend(value=True, evidence="looks right", cost=0.002)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    v = assessor.ask("judge", AssessQuestion(subject="s1", role="picture-for-word"))
    assert isinstance(v, Verdict)
    assert v.value is True and v.evidence == "looks right" and v.cost == 0.002
    assert backend.fetch_calls == 1
    assert len(db.assessments_of("s1")) == 1


def test_a_hit_does_not_execute_and_appends_nothing(db):
    backend = _FakeBackend(value=True)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    assessor.ask("judge", AssessQuestion(subject="s1", role="picture-for-word"))
    v = assessor.ask("judge", AssessQuestion(subject="s1", role="picture-for-word"))
    assert backend.fetch_calls == 1
    assert v.cost == 0.0
    assert len(db.assessments_of("s1")) == 1


def test_a_transport_error_is_not_cached_and_propagates(db):
    backend = _FakeBackend(raises=TransportError("down"))
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    with pytest.raises(TransportError):
        assessor.ask("judge", AssessQuestion(subject="s1", role="picture-for-word"))
    assert db.assessments_of("s1") == []


def test_learner_backend_raises_without_touching_cache_or_record(db):
    assessor = Assessor(record=db, cache=db, backends={})
    with pytest.raises(LearnerAskNotSupported):
        assessor.ask("learner", AssessQuestion(subject="s1", role="picture-fit"))
    assert db.assessments_of("s1") == []


def test_listener_backend_is_not_implemented(db):
    assessor = Assessor(record=db, cache=db, backends={})
    with pytest.raises(NotImplementedError):
        assessor.ask("listener", AssessQuestion(subject="s1", role="recording-for-word"))


# --- judge: key = judge:sha(RUBRIC):ARTIFACT_SHA:ROLE -------------------

def test_judge_cache_key_shape():
    backend = JudgeBackend(model="claude-opus-5", transport="cli",
                           complete=lambda p: "true")
    q = AssessQuestion(subject="s", role="picture-for-word",
                       artifact_sha="deadbeef", rubric="does this fit?")
    key = backend.cache_key(q)
    assert key.startswith("judge:")
    assert key.endswith(":deadbeef:picture-for-word")


def test_judge_key_changes_when_the_rubric_changes_but_not_the_artifact():
    backend = JudgeBackend(model="m", transport="cli", complete=lambda p: "true")
    q1 = AssessQuestion(subject="s", role="r", artifact_sha="a", rubric="rubric A")
    q2 = AssessQuestion(subject="s", role="r", artifact_sha="a", rubric="rubric B")
    assert backend.cache_key(q1) != backend.cache_key(q2)


def test_judge_key_reuses_the_artifact_sha_verbatim_not_double_hashed():
    backend = JudgeBackend(model="m", transport="cli", complete=lambda p: "true")
    q = AssessQuestion(subject="s", role="r", artifact_sha="deadbeef", rubric="x")
    assert "deadbeef" in backend.cache_key(q)


def test_judge_key_falls_back_to_subject_when_there_is_no_artifact():
    # A text-only judged question (no artifact_sha, e.g. a sentence quality
    # verdict) must still distinguish different subjects under the same
    # rubric/role -- a bare '-' placeholder would collide them.
    backend = JudgeBackend(model="m", transport="cli", complete=lambda p: "true")
    q1 = AssessQuestion(subject="sentence-1", role="r", rubric="x")
    q2 = AssessQuestion(subject="sentence-2", role="r", rubric="x")
    assert backend.cache_key(q1) != backend.cache_key(q2)
    assert "sentence-1" in backend.cache_key(q1)


def test_judge_fetch_builds_a_prompt_and_parses_the_response():
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return '{"value": true, "evidence": "clear picture"}'

    backend = JudgeBackend(model="m", transport="cli", complete=complete)
    q = AssessQuestion(subject="s", role="picture-for-word", artifact_sha="sha1",
                       rubric="does the picture evoke the word?")
    raw = backend.fetch(q)
    assert raw.value is True
    assert raw.evidence == "clear picture"
    assert "does the picture evoke the word?" in prompts[0]


def test_judge_backend_without_a_transport_refuses_single_question_fetch():
    backend = JudgeBackend(model="m", transport="batch")  # no `complete`
    with pytest.raises(RuntimeError, match="ask_batch"):
        backend.fetch(AssessQuestion(subject="s", role="r"))


# --- judge batch: submit -> persist pending -> resume -------------------

class _FakeBatchTransport:
    def __init__(self):
        self.submitted = None
        self.batch_id = "batch-1"
        self._status = "in_progress"
        self._results = {}

    def submit(self, requests):
        self.submitted = dict(requests)
        return self.batch_id

    def status(self, batch_id):
        return self._status

    def results(self, batch_id):
        return dict(self._results)


def test_ask_batch_submits_once_and_persists_a_batch_pending_row(db):
    bt = _FakeBatchTransport()
    backend = JudgeBackend(model="m", transport="batch", batch_transport=bt)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    q1 = AssessQuestion(subject="w1", role="picture-for-word", artifact_sha="a1", rubric="r")
    q2 = AssessQuestion(subject="w2", role="picture-for-word", artifact_sha="a2", rubric="r")

    with pytest.raises(BatchPending) as exc:
        assessor.ask_batch("judge", [q1, q2])
    assert exc.value.batch_id == "batch-1"
    assert set(exc.value.pending_subjects) == {"w1", "w2"}
    assert set(bt.submitted) == {"w1", "w2"}

    # calling again while still in_progress does NOT resubmit
    with pytest.raises(BatchPending):
        assessor.ask_batch("judge", [q1, q2])
    assert bt.submitted is not None
    submitted_once = bt.submitted
    bt.submitted = None
    with pytest.raises(BatchPending):
        assessor.ask_batch("judge", [q1, q2])
    assert bt.submitted is None  # not called again -- resumed the pending row


def test_ask_batch_resumes_and_writes_individual_verdicts_once_ended(db):
    bt = _FakeBatchTransport()
    backend = JudgeBackend(model="m", transport="batch", batch_transport=bt)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    q1 = AssessQuestion(subject="w1", role="picture-for-word", artifact_sha="a1", rubric="r")
    q2 = AssessQuestion(subject="w2", role="picture-for-word", artifact_sha="a2", rubric="r")

    with pytest.raises(BatchPending):
        assessor.ask_batch("judge", [q1, q2])

    bt._status = "ended"
    bt._results = {"w1": '{"value": true, "evidence": "good"}',
                   "w2": '{"value": false, "evidence": "blurry"}'}
    results = assessor.ask_batch("judge", [q1, q2])
    assert results["w1"].value is True
    assert results["w2"].value is False
    assert len(db.assessments_of("w1")) == 1
    assert len(db.assessments_of("w2")) == 1

    # a further call is now a pure cache hit -- no batch transport touched
    bt.submitted = None
    results2 = assessor.ask_batch("judge", [q1, q2])
    assert results2["w1"].value is True


def test_ask_batch_returns_cached_verdicts_without_touching_the_transport(db):
    bt = _FakeBatchTransport()
    backend = JudgeBackend(model="m", transport="batch", batch_transport=bt)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    # pre-seed a cache hit for w1 via the single-question path's own key
    key = backend.cache_key(AssessQuestion(subject="w1", role="r", artifact_sha="a1", rubric="x"))
    db.append(port="assess", backend="judge", key=key, subject="w1",
             question={}, answer={"value": True})
    q1 = AssessQuestion(subject="w1", role="r", artifact_sha="a1", rubric="x")
    results = assessor.ask_batch("judge", [q1])
    assert results["w1"].value is True
    assert bt.submitted is None  # never touched the transport


# --- mechanical: duration/format checks ---------------------------------

def test_duration_mechanical_key_is_parameter_explicit():
    backend = duration_mechanical_backend(lo=0.2, hi=5.0, resolve_path=lambda sha: sha)
    key = backend.cache_key(AssessQuestion(subject="s", role="recording-for-word",
                                           artifact_sha="deadbeef"))
    assert key == "mech:duration:0.2-5.0:deadbeef"


def test_duration_mechanical_passes_within_range():
    backend = duration_mechanical_backend(
        lo=0.2, hi=5.0, resolve_path=lambda sha: f"/media/{sha}.mp3",
        duration_of=lambda path: 1.5)
    raw = backend.fetch(AssessQuestion(subject="s", role="recording-for-word",
                                       artifact_sha="deadbeef"))
    assert raw.value is True


def test_duration_mechanical_fails_outside_range():
    backend = duration_mechanical_backend(
        lo=0.2, hi=5.0, resolve_path=lambda sha: f"/media/{sha}.mp3",
        duration_of=lambda path: 9.9)
    raw = backend.fetch(AssessQuestion(subject="s", role="recording-for-word",
                                       artifact_sha="deadbeef"))
    assert raw.value is False


def test_format_mechanical_key_uses_code_version_when_no_params_express_it():
    backend = format_mechanical_backend(expected_ext="mp3", code_version="v2",
                                        resolve_ext=lambda sha: "mp3")
    key = backend.cache_key(AssessQuestion(subject="s", role="recording-for-word",
                                           artifact_sha="deadbeef"))
    assert key == "mech:format:v2:deadbeef"


def test_format_mechanical_evaluates_extension_match():
    backend = format_mechanical_backend(expected_ext="mp3", resolve_ext=lambda sha: "wav")
    raw = backend.fetch(AssessQuestion(subject="s", role="r", artifact_sha="x"))
    assert raw.value is False


def test_ffprobe_backend_failure_is_a_transport_error_and_uncached(db):
    import subprocess as sp

    def failing_runner(cmd, **kwargs):
        return sp.CompletedProcess(cmd, 1, "", "no such file")

    backend = duration_mechanical_backend(resolve_path=lambda sha: "/nope.mp3",
                                          runner=failing_runner)
    assessor = Assessor(record=db, cache=db, backends={"mechanical": backend})
    with pytest.raises(TransportError):
        assessor.ask("mechanical", AssessQuestion(subject="s", role="recording-for-word",
                                                  artifact_sha="x"))
    assert db.assessments_of("s") == []


# --- authority table ------------------------------------------------------

def test_authority_order_puts_learner_ahead_of_judge_on_fit_roles():
    assert AUTHORITY_ORDER["picture-for-word"][0] == "learner"


def test_authority_order_puts_mechanical_ahead_of_judge_on_recording_roles():
    assert AUTHORITY_ORDER["recording-for-word"][0] == "mechanical"
    assert "learner" not in AUTHORITY_ORDER["recording-for-word"]
