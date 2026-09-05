"""Tests for assessor.py (spec 3 sections 1-2): Assessor.ask()'s cache-first
shape, judge's three transports behind one implementation (incl. batch
persistence/resume), the mechanical backend, and the learner/listener
non-implementations. Real SyllabusDb for cache-first behavior; fakes for
transports -- no network, no subprocess, no anthropic import.
"""
import importlib
import pkgutil
from pathlib import Path

import pytest

import thai_syllabus
from thai_syllabus.assessor import (
    AssessQuestion,
    Assessor,
    BatchPending,
    JudgeBackend,
    LearnerAskNotSupported,
    ManyResult,
    Price,
    RawVerdict,
    Verdict,
    duration_mechanical_backend,
    format_mechanical_backend,
    parse_preference,
    picture_fit_prompt,
    picture_preference_prompt,
    sentence_prompt,
)
from thai_syllabus.authority import AUTHORITY_ORDER, ROLE_FOR_KIND, role_for
from thai_syllabus.cachekeys import sha
from thai_syllabus.store import SyllabusDb
from thai_syllabus.transport import Completion, TransportError


def test_no_module_imports_the_old_packages():
    for m in pkgutil.iter_modules(thai_syllabus.__path__):
        src = Path(importlib.import_module(f"thai_syllabus.{m.name}").__file__).read_text()
        assert "from thai_deck_eval" not in src and "import thai_deck_eval" not in src, m.name
        assert "from thai_deck_gen" not in src and "import thai_deck_gen" not in src, m.name


def test_assessor_has_no_authority_data():
    import thai_syllabus.assessor as a
    assert not hasattr(a, "AUTHORITY_ORDER") and not hasattr(a, "ROLE_FOR_KIND")


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
    v = assessor.ask("judge", AssessQuestion(subject="s1", role="picture-for-word", kind="picture"))
    assert isinstance(v, Verdict)
    assert v.value is True and v.evidence == "looks right" and v.cost == 0.002
    assert backend.fetch_calls == 1
    rows = db.assessments_of("s1")
    assert len(rows) == 1
    assert rows[0].question["kind"] == "picture"  # record.rows_for reads this back


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
    assert key.encode().startswith("judge:")
    assert key.encode().endswith(":deadbeef:picture-for-word")


def test_judge_key_changes_when_the_rubric_changes_but_not_the_artifact():
    backend = JudgeBackend(model="m", transport="cli", complete=lambda p: "true")
    q1 = AssessQuestion(subject="s", role="r", artifact_sha="a", rubric="rubric A")
    q2 = AssessQuestion(subject="s", role="r", artifact_sha="a", rubric="rubric B")
    assert backend.cache_key(q1) != backend.cache_key(q2)


def test_judge_key_reuses_the_artifact_sha_verbatim_not_double_hashed():
    backend = JudgeBackend(model="m", transport="cli", complete=lambda p: "true")
    q = AssessQuestion(subject="s", role="r", artifact_sha="deadbeef", rubric="x")
    assert "deadbeef" in backend.cache_key(q).encode()


def test_judge_key_falls_back_to_subject_when_there_is_no_artifact():
    # A text-only judged question (no artifact_sha, e.g. a sentence quality
    # verdict) must still distinguish different subjects under the same
    # rubric/role -- a bare '-' placeholder would collide them.
    backend = JudgeBackend(model="m", transport="cli",
                           complete=lambda p, a=(): Completion(text="true"))
    q1 = AssessQuestion(subject="sentence-1", role="r", rubric="x")
    q2 = AssessQuestion(subject="sentence-2", role="r", rubric="x")
    assert backend.cache_key(q1) != backend.cache_key(q2)
    assert "sentence-1" in backend.cache_key(q1).encode()


def test_judge_fetch_builds_a_prompt_and_parses_the_response():
    prompts = []

    def complete(prompt, attachments=()):
        prompts.append(prompt)
        return Completion(text='{"value": true, "evidence": "clear picture"}')

    backend = JudgeBackend(model="m", transport="cli", complete=complete)
    q = AssessQuestion(subject="s", role="picture-for-word", artifact_sha="sha1",
                       rubric="does the picture evoke the word?")
    raw = backend.fetch(q)
    assert raw.value is True
    assert raw.evidence == "clear picture"
    assert "does the picture evoke the word?" in prompts[0]


def test_judge_backend_without_a_transport_refuses_single_question_fetch():
    backend = JudgeBackend(model="m", transport="batch")  # no `complete`
    with pytest.raises(RuntimeError, match="ask_many"):
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
    k1, k2 = backend.cache_key(q1), backend.cache_key(q2)
    cid1, cid2 = "q" + sha(k1.encode()), "q" + sha(k2.encode())

    with pytest.raises(BatchPending) as exc:
        assessor.ask_batch("judge", [q1, q2])
    assert exc.value.batch_id == "batch-1"
    assert set(exc.value.pending_keys) == {k1, k2}
    assert exc.value.resolved == {}
    assert set(bt.submitted) == {cid1, cid2}

    # calling again while still in_progress does NOT resubmit
    with pytest.raises(BatchPending):
        assessor.ask_batch("judge", [q1, q2])
    assert bt.submitted is not None
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
    k1, k2 = backend.cache_key(q1), backend.cache_key(q2)

    with pytest.raises(BatchPending):
        assessor.ask_batch("judge", [q1, q2])

    bt._status = "ended"
    bt._results = {"q" + sha(k1.encode()): Completion(text='{"value": true, "evidence": "good"}'),
                   "q" + sha(k2.encode()): Completion(text='{"value": false, "evidence": "blurry"}')}
    results = assessor.ask_batch("judge", [q1, q2])
    assert results[k1].value is True
    assert results[k2].value is False
    # 2 rows each: the per-subject batch-pending marker plus the verdict
    assert len(db.assessments_of("w1")) == 2
    assert len(db.assessments_of("w2")) == 2
    marker = db.latest("assess", "judge", "judge-batch-pending:w1")
    assert marker.question["kind"] == "batch"  # record.rows_for never returns this for a need kind

    # a further call is now a pure cache hit -- no batch transport touched
    bt.submitted = None
    results2 = assessor.ask_batch("judge", [q1, q2])
    assert results2[k1].value is True


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
    assert results[key].value is True
    assert bt.submitted is None  # never touched the transport


# --- batch starvation: an ended/terminal batch must not wedge its subjects
# forever -- a key with no successful result must drop out of the per-
# subject `judge-batch-pending:{subject}` marker so derivations.pending()
# stops reporting it forever-pending (it was never cached, so a later run
# asks it again as a fresh question).

def test_ask_batch_drops_an_errored_key_from_the_pending_marker_once_ended(db):
    bt = _FakeBatchTransport()
    backend = JudgeBackend(model="m", transport="batch", batch_transport=bt)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    q_ok = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="a1", rubric="r")
    q_bad = AssessQuestion(subject="w", role="picture-preference", artifact_sha="a2", rubric="r")
    k_ok, k_bad = backend.cache_key(q_ok), backend.cache_key(q_bad)

    with pytest.raises(BatchPending):
        assessor.ask_batch("judge", [q_ok, q_bad])

    bt._status = "ended"
    # q_bad's result errored/canceled/expired -- results() omits it, exactly
    # as ClaudeBatchTransport.results does for a non-succeeded result.
    bt._results = {"q" + sha(k_ok.encode()): Completion(text='{"value": true, "evidence": "good"}')}

    with pytest.raises(BatchPending) as exc:
        assessor.ask_batch("judge", [q_ok, q_bad])
    assert exc.value.resolved[k_ok].value is True
    assert exc.value.pending_keys == [k_bad]

    marker = db.latest("assess", "judge", "judge-batch-pending:w")
    assert marker is not None
    assert marker.question["keys"] == [k_ok.encode()]  # the errored key is absent
    assert marker.answer["abandoned"] == [k_bad.encode()]

    from thai_syllabus.derivations import pending as derive_pending
    assert derive_pending(db, "w", "picture") is False


def test_ask_batch_treats_a_non_ended_terminal_status_as_all_abandoned_and_logs(db, caplog):
    class BT:
        def __init__(self):
            self.batch_id = "batch-expired"
            self.status_calls = 0

        def submit(self, requests):
            return self.batch_id

        def status(self, batch_id):
            self.status_calls += 1
            return "expired"

        def results(self, batch_id):
            raise AssertionError("results() must not be called for a non-ended status")

    bt = BT()
    backend = JudgeBackend(model="m", transport="batch", batch_transport=bt)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="a1", rubric="r")
    k = backend.cache_key(q)

    with pytest.raises(BatchPending):
        assessor.ask_batch("judge", [q])  # submits, does not check status yet

    with caplog.at_level("WARNING", logger="thai_syllabus.assessor"):
        with pytest.raises(BatchPending) as exc:
            assessor.ask_batch("judge", [q])
    assert exc.value.pending_keys == [k]
    assert bt.status_calls == 1
    assert "expired" in caplog.text.lower()

    marker = db.latest("assess", "judge", "judge-batch-pending:w")
    assert marker.question["keys"] == []
    assert marker.answer["abandoned"] == [k.encode()]

    from thai_syllabus.derivations import pending as derive_pending
    assert derive_pending(db, "w", "picture") is False


# --- mechanical: duration/format checks ---------------------------------

def test_duration_mechanical_key_is_parameter_explicit():
    backend = duration_mechanical_backend(lo=0.2, hi=5.0, resolve_path=lambda sha: sha)
    key = backend.cache_key(AssessQuestion(subject="s", role="recording-for-word",
                                           artifact_sha="deadbeef"))
    assert key.encode() == "mech:duration:0.2-5.0:deadbeef"


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
    assert key.encode() == "mech:format:v2:deadbeef"


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


def test_role_for_returns_the_mapped_role():
    assert role_for("picture") == "picture-for-word"
    assert role_for("grapheme-keyword") == "grapheme-keyword-for-grapheme"


def test_role_for_raises_keyerror_naming_the_kind():
    with pytest.raises(KeyError, match="unknown-kind"):
        role_for("unknown-kind")


# --- spec 3 section 1/2: judge attaches artifacts, prices verdicts, --------
# --- Assessor.ask_many, batch keyed by cache key ---------------------------

def test_role_for_kind_and_rendition_authority():
    assert ROLE_FOR_KIND["picture"] == "picture-for-word"
    assert AUTHORITY_ORDER["rendition-for-pair"] == ("mechanical",)


def test_price_costs_a_completion():
    p = Price(input_per_mtok=2.0, output_per_mtok=10.0)
    assert p.cost(Completion(text="", input_tokens=1_000_000, output_tokens=100_000)) == 3.0


def test_judge_fetch_attaches_the_artifact_and_prices_the_verdict(tmp_path):
    img = tmp_path / "abc.jpg"
    img.write_bytes(b"x")
    seen = {}

    def complete(prompt, attachments=()):
        seen["attachments"] = list(attachments)
        return Completion(text='{"value": true, "evidence": "fits"}', input_tokens=500, output_tokens=50)

    jb = JudgeBackend(model="m", transport="api", complete=complete,
                      resolve_path=lambda sha: img if sha == "abc" else None,
                      price=Price(2.0, 10.0))
    raw = jb.fetch(AssessQuestion(subject="w", role="picture-for-word", artifact_sha="abc", rubric="r"))
    assert raw.value is True and seen["attachments"] == [img]
    assert abs(raw.cost - (500 * 2.0 + 50 * 10.0) / 1_000_000) < 1e-12


def test_judge_cli_cost_is_one_quota_call():
    jb = JudgeBackend(model="m", transport="cli",
                      complete=lambda p, a=(): Completion(text="true"), quota_cost_per_call=1.0)
    assert jb.fetch(AssessQuestion(subject="w", role="picture-for-word", rubric="r")).cost == 1.0


def test_preference_question_key_and_attachments(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"a"); b.write_bytes(b"b")
    paths = {"sha-a": a, "sha-b": b}
    seen = {}

    def complete(prompt, attachments=()):
        seen["n"] = len(list(attachments))
        return Completion(text='{"ranking": ["sha-b", "sha-a"]}')

    jb = JudgeBackend(model="m", transport="api", complete=complete,
                      resolve_path=paths.get, parse_response=parse_preference,
                      prompt_builder=picture_preference_prompt)
    q = AssessQuestion(subject="w", role="picture-preference", rubric="pref",
                       params={"candidates": ["sha-b", "sha-a"], "word": "x", "meaning": "y"})
    assert jb.cache_key(q).encode().endswith(":picture-preference")
    assert jb.cache_key(q) == jb.cache_key(AssessQuestion(
        subject="w", role="picture-preference", rubric="pref",
        params={"candidates": ["sha-a", "sha-b"]}))
    raw = jb.fetch(q)
    assert raw.value == ["sha-b", "sha-a"] and seen["n"] == 2


def test_picture_fit_prompt_delimits_fields_and_names_the_rubric():
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s", rubric="RUBRIC",
                       params={"word": "ส้ม", "meaning": "orange", "gloss_shown": "orange",
                               "phrase": "oranges on a table"})
    p = picture_fit_prompt(q)
    assert "RUBRIC" in p and "<deck-field>ส้ม</deck-field>" in p and "oranges on a table" in p
    assert '"value"' in p


def test_ask_many_inline_resolves_each_and_skips_transport_errors(db):
    calls = []

    def complete(prompt, attachments=()):
        calls.append(prompt)
        if "boom" in prompt:
            raise TransportError("boom")
        return Completion(text="true")

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    # rubric text (not artifact_sha) is what the default picture-for-word
    # prompt embeds verbatim -- "boom" has to land there to trip `complete`.
    qs = [AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r"),
          AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s2", rubric="boom")]
    res = a.ask_many("judge", qs)
    assert isinstance(res, ManyResult)
    assert set(res.resolved) == {jb.cache_key(qs[0])} and res.pending == []
    assert len(calls) == 2  # both questions were attempted, not short-circuited


def test_ask_many_batch_returns_pending_keys_and_writes_per_subject_marker(db):
    class BT:
        def submit(self, requests):
            assert all(cid.startswith("q") for cid in requests)
            return "batch_9"

        def status(self, batch_id):
            return "in_progress"

    jb = JudgeBackend(model="m", transport="batch", batch_transport=BT())
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    qs = [AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r"),
          AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s2", rubric="r")]
    res = a.ask_many("judge", qs)
    assert res.resolved == {} and set(res.pending) == {jb.cache_key(q) for q in qs}
    marker = db.latest("assess", "judge", "judge-batch-pending:w")
    assert marker is not None and marker.answer["batch_id"] == "batch_9"
    assert set(marker.question["keys"]) == {jb.cache_key(q).encode() for q in qs}


def test_ask_many_batch_writes_one_marker_row_per_subject_with_only_its_own_keys(db):
    class BT:
        def submit(self, requests):
            return "batch_7"

        def status(self, batch_id):
            return "in_progress"

    jb = JudgeBackend(model="m", transport="batch", batch_transport=BT())
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q1 = AssessQuestion(subject="w1", role="picture-for-word", artifact_sha="s1", rubric="r")
    q2 = AssessQuestion(subject="w2", role="picture-for-word", artifact_sha="s2", rubric="r")
    a.ask_many("judge", [q1, q2])

    m1 = db.latest("assess", "judge", "judge-batch-pending:w1")
    m2 = db.latest("assess", "judge", "judge-batch-pending:w2")
    assert m1 is not None and m1.question["keys"] == [jb.cache_key(q1).encode()]
    assert m2 is not None and m2.question["keys"] == [jb.cache_key(q2).encode()]


def test_ask_many_batch_excludes_a_question_whose_sha_cannot_be_resolved(db, tmp_path):
    img = tmp_path / "ok.jpg"
    img.write_bytes(b"x")

    class BT:
        def __init__(self):
            self.submitted = None

        def submit(self, requests):
            self.submitted = dict(requests)
            return "batch_5"

        def status(self, batch_id):
            return "in_progress"

    bt = BT()
    jb = JudgeBackend(model="m", transport="batch", batch_transport=bt,
                      resolve_path=lambda s: img if s == "ok" else None)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q_ok = AssessQuestion(subject="w1", role="picture-for-word", artifact_sha="ok", rubric="r")
    q_bad = AssessQuestion(subject="w2", role="picture-for-word", artifact_sha="missing", rubric="r")

    res = a.ask_many("judge", [q_ok, q_bad])

    assert set(bt.submitted) == {"q" + sha(jb.cache_key(q_ok).encode())}  # only the resolvable one submitted
    marker = db.latest("assess", "judge", "judge-batch-pending:w1")
    assert marker is not None and marker.question["keys"] == [jb.cache_key(q_ok).encode()]
    assert db.latest("assess", "judge", "judge-batch-pending:w2") is None  # excluded, no marker
    assert res.resolved == {}
    assert res.pending == [jb.cache_key(q_ok)]
    assert res.excluded == [jb.cache_key(q_bad)]  # reported, not silently absent


# --- two controller-decided additions: key_of, two-parameter parse_response

def test_assessor_key_of_matches_the_backend_cache_key(db):
    jb = JudgeBackend(model="m", transport="api", complete=lambda p, a=(): Completion(text="true"))
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r")
    assert a.key_of("judge", q) == jb.cache_key(q)


def test_parse_response_receives_the_question_when_it_declares_two_params():
    def parse_two(text, question):
        return RawVerdict(value=question.subject, evidence=text)

    jb = JudgeBackend(model="m", transport="api",
                      complete=lambda p, a=(): Completion(text="ignored"),
                      parse_response=parse_two)
    raw = jb.fetch(AssessQuestion(subject="w9", role="picture-for-word", rubric="r"))
    assert raw.value == "w9" and raw.evidence == "ignored"


# --- review fix: an unresolvable sha must raise, never be silently dropped

def test_fit_fetch_raises_when_the_artifact_sha_cannot_be_resolved():
    jb = JudgeBackend(model="m", transport="api",
                      complete=lambda p, a=(): Completion(text="true"),
                      resolve_path=lambda sha: None)
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="missing", rubric="r")
    with pytest.raises(TransportError):
        jb.fetch(q)


def test_preference_fetch_raises_when_one_candidate_cannot_be_resolved(tmp_path):
    img_a = tmp_path / "a.jpg"
    img_a.write_bytes(b"a")
    paths = {"sha-a": img_a}  # sha-b deliberately left unresolvable

    jb = JudgeBackend(model="m", transport="api",
                      complete=lambda p, attachments=(): Completion(text='{"ranking": []}'),
                      resolve_path=paths.get, parse_response=parse_preference,
                      prompt_builder=picture_preference_prompt)
    q = AssessQuestion(subject="w", role="picture-preference", rubric="pref",
                       params={"candidates": ["sha-a", "sha-b"]})
    with pytest.raises(TransportError):
        jb.fetch(q)


# --- review fix: the default prompt_builder/parse_response dispatch by role,
# one role -> (prompt_builder, parser) table, so a preference completion's
# {"ranking": [...]} is never silently parsed as value=None -----------------

def test_default_dispatch_builds_the_picture_fit_prompt_and_parses_its_value():
    prompts = []

    def complete(prompt, attachments=()):
        prompts.append(prompt)
        return Completion(text='{"value": true, "evidence": "e"}')

    jb = JudgeBackend(model="m", transport="api", complete=complete)  # no custom builder/parser
    q = AssessQuestion(subject="w", role="picture-for-word", rubric="fit rubric",
                       params={"word": "ก", "meaning": "m"})
    raw = jb.fetch(q)
    assert prompts[0] == picture_fit_prompt(q)
    assert raw.value is True and raw.evidence == "e"


def test_default_dispatch_builds_the_sentence_prompt_and_parses_its_value():
    prompts = []

    def complete(prompt, attachments=()):
        prompts.append(prompt)
        return Completion(text='{"value": false, "evidence": "not natural"}')

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    q = AssessQuestion(subject="w", role="sentence-for-target", rubric="sentence rubric",
                       params={"text": "some sentence", "word": "ก"})
    raw = jb.fetch(q)
    assert prompts[0] == sentence_prompt(q)
    assert raw.value is False and raw.evidence == "not natural"


def test_default_dispatch_builds_the_preference_prompt_and_parses_the_ranking():
    prompts = []

    def complete(prompt, attachments=()):
        prompts.append(prompt)
        return Completion(text='{"ranking": ["a", "b"], "evidence": "e"}')

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    q = AssessQuestion(subject="w", role="picture-preference", rubric="pref rubric",
                       params={"candidates": ["a", "b"], "word": "ก", "meaning": "m"})
    raw = jb.fetch(q)
    assert prompts[0] == picture_preference_prompt(q)
    assert raw.value == ["a", "b"]


# --- hit/miss is the port's answer, not the caller's timestamp guess ------

def test_a_miss_is_not_a_hit_and_the_re_read_is(db):
    backend = _FakeBackend(value=True, cost=0.002)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    q = AssessQuestion(subject="s1", role="picture-for-word")
    assert assessor.ask("judge", q).hit is False
    assert assessor.ask("judge", q).hit is True


def test_ask_many_marks_cached_verdicts_as_hits(db):
    backend = _FakeBackend(value=True)
    assessor = Assessor(record=db, cache=db, backends={"judge": backend})
    q = AssessQuestion(subject="s1", role="picture-for-word")
    first = assessor.ask_many("judge", [q])
    second = assessor.ask_many("judge", [q])
    assert [v.hit for v in first.resolved.values()] == [False]
    assert [v.hit for v in second.resolved.values()] == [True]


def test_ask_batch_marks_cached_verdicts_as_hits(db):
    jb = JudgeBackend(model="m", transport="batch", batch_transport=object())
    assessor = Assessor(record=db, cache=db, backends={"judge": jb})
    q = AssessQuestion(subject="s1", role="picture-for-word", artifact_sha="a", rubric="r")
    db.append(port="assess", backend="judge", key=jb.cache_key(q), subject="s1",
              question={"role": q.role, "artifact_sha": "a", "rubric": "r"},
              answer={"value": True})
    resolved = assessor.ask_batch("judge", [q])
    assert [v.hit for v in resolved.values()] == [True]


# --- a swallowed per-question transport error is logged, with its key ------

def test_ask_many_logs_a_warning_naming_the_dropped_question_key(db, caplog):
    import logging

    def complete(prompt, attachments=()):
        raise TransportError("api transport failed: 401")

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r")
    with caplog.at_level(logging.WARNING, logger="thai_syllabus.assessor"):
        res = a.ask_many("judge", [q])
    assert res.resolved == {} and res.pending == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings and jb.cache_key(q).encode() in warnings[0].getMessage()


# --- excluded: a question the backend cannot PREPARE is not a dead wire ----

def test_ask_many_inline_excludes_a_question_whose_sha_cannot_be_resolved(db, tmp_path):
    """An artifact sha that resolves to no file is a question the judge
    cannot prepare -- reported in `excluded`, distinct from a wire that
    will not answer, so a caller can tell "this candidate is unusable"
    from "the judge is unreachable"."""
    img = tmp_path / "ok.jpg"
    img.write_bytes(b"x")
    calls = []

    def complete(prompt, attachments=()):
        calls.append(prompt)
        return Completion(text='{"value": true}')

    jb = JudgeBackend(model="m", transport="api", complete=complete,
                      resolve_path=lambda s: img if s == "ok" else None)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q_ok = AssessQuestion(subject="w1", role="picture-for-word", artifact_sha="ok", rubric="r")
    q_bad = AssessQuestion(subject="w2", role="picture-for-word", artifact_sha="gone", rubric="r")

    res = a.ask_many("judge", [q_ok, q_bad])

    assert set(res.resolved) == {jb.cache_key(q_ok)}
    assert res.excluded == [jb.cache_key(q_bad)]
    assert len(calls) == 1        # the unpreparable question never reached the wire


def test_ask_many_inline_does_not_call_a_wire_failure_an_exclusion(db):
    """A transport that will not answer is not an unpreparable question:
    `excluded` stays empty, which is how a caller recognises a dead wire."""
    def complete(prompt, attachments=()):
        raise TransportError("api transport failed: 401")

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r")
    res = a.ask_many("judge", [q])
    assert res.resolved == {} and res.pending == [] and res.excluded == []


def test_ask_many_never_re_prepares_a_cached_verdict(db):
    """A cached verdict needs no preparation -- a candidate whose file has
    since vanished must still read its verdict back, not be excluded."""
    def resolve(sha):
        raise AssertionError("preparation must not run for a cache hit")

    jb = JudgeBackend(model="m", transport="api", resolve_path=resolve,
                      complete=lambda p, a=(): Completion(text="true"))
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r")
    db.append(port="assess", backend="judge", key=jb.cache_key(q), subject="w",
              question={"role": q.role, "artifact_sha": "s1", "rubric": "r"},
              answer={"value": True})
    res = a.ask_many("judge", [q])
    assert [v.value for v in res.resolved.values()] == [True] and res.excluded == []
