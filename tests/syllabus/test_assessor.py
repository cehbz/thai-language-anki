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
    JudgeBackend,
    JudgeUnreachable,
    LearnerAskNotSupported,
    ManyResult,
    PreparationError,
    Price,
    RawVerdict,
    Verdict,
    MechanicalBackend,
    duration_mechanical_backend,
    format_mechanical_backend,
    rendition_mechanical_backend,
    parse_preference,
    picture_fit_prompt,
    picture_preference_prompt,
    sentence_prompt,
)
from thai_syllabus.authority import AUTHORITY_ORDER, ROLE_FOR_KIND, role_for
from thai_syllabus.cachekeys import MechanicalKey, rendition_identity, sha
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
    assert AUTHORITY_ORDER["rendition-for-pair"] == ("rendition",)


def test_role_for_reads_a_sentence_subject_into_its_own_role():
    assert role_for("picture") == "picture-for-word"
    assert role_for("picture", "sentence") == "scene-for-sentence"
    assert role_for("recording", "sentence") == "recording-for-sentence"
    assert role_for("rendition", "pair") == "rendition-for-pair"


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
    assert set(res.resolved) == {jb.cache_key(qs[0])} and res.collected == []
    assert len(calls) == 2  # both questions were attempted, not short-circuited


# --- one judge batch per run: ask_many collects, submit/resolve release ----

def fit_question(subject: str, artifact_sha: str) -> AssessQuestion:
    return AssessQuestion(subject=subject, role="picture-for-word", artifact_sha=artifact_sha,
                          rubric="fit rubric", kind="picture")


class _FakeBatch:
    """A batch transport test double: `submit` hands back a fresh batch id
    every call; a test drives its outcome with `complete`/`expire` before
    Assessor.resolve() reads `status`/`results`.
    """
    def __init__(self):
        self._requests: dict[str, dict[str, tuple[str, list]]] = {}
        self._status: dict[str, str] = {}
        self._texts: dict[str, dict[str, Completion]] = {}

    def submit(self, requests):
        batch_id = f"batch-{len(self._requests) + 1}"
        self._requests[batch_id] = dict(requests)
        self._status[batch_id] = "in_progress"
        return batch_id

    def complete(self, batch_id: str, text_by_key: dict) -> None:
        """text_by_key: CacheKey -> completion text, for every question
        this batch should answer successfully."""
        self._status[batch_id] = "ended"
        self._texts[batch_id] = {"q" + sha(key.encode()): Completion(text=text)
                                 for key, text in text_by_key.items()}

    def expire(self, batch_id: str) -> None:
        self._status[batch_id] = "expired"

    def status(self, batch_id):
        return self._status[batch_id]

    def results(self, batch_id):
        return dict(self._texts.get(batch_id, {}))


@pytest.fixture
def fake_batch():
    return _FakeBatch()


@pytest.fixture
def assessor_with_batch_transport(db, fake_batch):
    jb = JudgeBackend(model="m", transport="batch", batch_transport=fake_batch)
    return Assessor(record=db, cache=db, backends={"judge": jb})


@pytest.fixture
def assessor_inline(db, tmp_path):
    img = tmp_path / "rice.jpg"
    img.write_bytes(b"x")

    def complete(prompt, attachments=()):
        return Completion(text='{"value": true}')

    jb = JudgeBackend(model="m", transport="api", complete=complete,
                      resolve_path=lambda s: img if s == "a" * 64 else None)
    return Assessor(record=db, cache=db, backends={"judge": jb})


@pytest.fixture
def assessor_inline_with_dead_transport(db):
    def complete(prompt, attachments=()):
        raise TransportError("api transport failed: 401")

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    return Assessor(record=db, cache=db, backends={"judge": jb})


def test_ask_many_collects_misses_under_a_batch_transport(assessor_with_batch_transport):
    a = assessor_with_batch_transport
    res = a.ask_many("judge", [fit_question("rice", "a" * 64), fit_question("rice", "b" * 64)])
    assert res.resolved == {} and len(res.collected) == 2
    assert a.unresolved_batch() is None  # nothing submitted until submit()


def test_submit_then_resolve_writes_verdicts_and_releases_the_marker(
        assessor_with_batch_transport, fake_batch):
    a = assessor_with_batch_transport
    res = a.ask_many("judge", [fit_question("rice", "a" * 64)])
    bid = a.submit(res.collected)
    assert a.unresolved_batch() == (bid, frozenset({"rice"}))
    fake_batch.complete(bid, {res.collected[0].key: '{"value": true}'})
    got = a.resolve(bid)
    assert got and a.unresolved_batch() is None
    assert a.ask_many("judge", [fit_question("rice", "a" * 64)]).resolved  # now a cache hit


def test_expired_batch_releases_and_questions_reask(assessor_with_batch_transport, fake_batch):
    a = assessor_with_batch_transport
    bid = a.submit(a.ask_many("judge", [fit_question("rice", "a" * 64)]).collected)
    fake_batch.expire(bid)
    assert a.resolve(bid) == {} and a.unresolved_batch() is None
    assert a.ask_many("judge", [fit_question("rice", "a" * 64)]).collected  # asked again


def test_ask_many_and_submit_build_each_question_exactly_once(db, fake_batch):
    """ask_many prepares a batch-transport miss once (into `collected`);
    submit() must consume that, never calling prompt_builder/attachments
    again for the same question."""
    prompt_calls = []
    attachment_calls = []

    def prompt_builder(q):
        prompt_calls.append(q.subject)
        return "prompt text"

    jb = JudgeBackend(model="m", transport="batch", batch_transport=fake_batch,
                      prompt_builder=prompt_builder)
    original_attachments = jb.attachments

    def counting_attachments(q):
        attachment_calls.append(q.subject)
        return original_attachments(q)
    jb.attachments = counting_attachments

    a = Assessor(record=db, cache=db, backends={"judge": jb})
    questions = [fit_question("rice", "a" * 64), fit_question("corn", "b" * 64)]
    res = a.ask_many("judge", questions)
    a.submit(res.collected)

    assert prompt_calls == ["rice", "corn"]
    assert attachment_calls == ["rice", "corn"]


def test_unpreparable_question_is_excluded_not_asked(assessor_inline):
    res = assessor_inline.ask_many("judge", [fit_question("rice", "missing" * 8)])
    assert list(res.excluded.values()) == ["artifact not found: " + "missing" * 8]


def test_all_questions_failing_on_the_wire_raises(assessor_inline_with_dead_transport):
    with pytest.raises(JudgeUnreachable):
        assessor_inline_with_dead_transport.ask_many("judge", [fit_question("rice", "a" * 64)])


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
    with pytest.raises(PreparationError):
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
    with pytest.raises(PreparationError):
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


def test_ask_many_batch_marks_cached_verdicts_as_hits(db):
    jb = JudgeBackend(model="m", transport="batch", batch_transport=object())
    assessor = Assessor(record=db, cache=db, backends={"judge": jb})
    q = AssessQuestion(subject="s1", role="picture-for-word", artifact_sha="a", rubric="r")
    db.append(port="assess", backend="judge", key=jb.cache_key(q), subject="s1",
              question={"role": q.role, "artifact_sha": "a", "rubric": "r"},
              answer={"value": True})
    res = assessor.ask_many("judge", [q])
    assert [v.hit for v in res.resolved.values()] == [True] and res.collected == []


# --- a swallowed per-question transport error is logged, with its key ------

def test_ask_many_logs_a_warning_naming_the_dropped_question_key(db, caplog):
    import logging

    def complete(prompt, attachments=()):
        if "boom" in prompt:
            raise TransportError("api transport failed: 401")
        return Completion(text="true")

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q_ok = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r")
    q_bad = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s2", rubric="boom")
    with caplog.at_level(logging.WARNING, logger="thai_syllabus.assessor"):
        res = a.ask_many("judge", [q_ok, q_bad])
    assert set(res.resolved) == {jb.cache_key(q_ok)} and res.collected == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings and jb.cache_key(q_bad).encode() in warnings[0].getMessage()


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
    assert res.excluded == {jb.cache_key(q_bad).encode(): "artifact not found: gone"}
    assert len(calls) == 1        # the unpreparable question never reached the wire


def test_ask_many_inline_does_not_call_a_wire_failure_an_exclusion(db):
    """A transport that will not answer is not an unpreparable question:
    `excluded` stays empty, which is how a caller recognises a dead wire."""
    def complete(prompt, attachments=()):
        if "boom" in prompt:
            raise TransportError("api transport failed: 401")
        return Completion(text="true")

    jb = JudgeBackend(model="m", transport="api", complete=complete)
    a = Assessor(record=db, cache=db, backends={"judge": jb})
    q_ok = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s1", rubric="r")
    q_bad = AssessQuestion(subject="w", role="picture-for-word", artifact_sha="s2", rubric="boom")
    res = a.ask_many("judge", [q_ok, q_bad])
    assert set(res.resolved) == {jb.cache_key(q_ok)} and res.collected == [] and res.excluded == {}


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
    assert [v.value for v in res.resolved.values()] == [True] and res.excluded == {}


# --- the rendition backend: one speaker across a pair's members -------------

def _rendition(speakers):
    return rendition_mechanical_backend(speaker_of=lambda sha: speakers.get(sha))


def _rendition_question(members, checks=None):
    return AssessQuestion(subject="p1", role="rendition-for-pair", kind="rendition",
                          subject_kind="pair",
                          params={"members": members,
                                  "member_checks": {m: True for m in members}
                                  if checks is None else checks})


def test_a_rendition_by_one_speaker_passes():
    backend = _rendition({"a": "forvo:somchai", "b": "forvo:somchai"})
    verdict = backend.fetch(_rendition_question({"near": "a", "far": "b"}))
    assert verdict.value is True and "forvo:somchai" in verdict.evidence


def test_a_rendition_by_two_speakers_fails():
    backend = _rendition({"a": "forvo:somchai", "b": "forvo:malee"})
    verdict = backend.fetch(_rendition_question({"near": "a", "far": "b"}))
    assert verdict.value is False
    assert "forvo:malee" in verdict.evidence and "forvo:somchai" in verdict.evidence


def test_a_member_recording_with_no_speaker_fails_the_rendition():
    backend = _rendition({"a": "forvo:somchai"})
    verdict = backend.fetch(_rendition_question({"near": "a", "far": "b"}))
    assert verdict.value is False and "far" in verdict.evidence


def test_a_rendition_whose_member_recording_failed_its_own_checks_fails():
    """One speaker is not enough: a member recording that failed duration
    is not part of a usable rendition, and the rendition check is the one
    decider on that."""
    backend = _rendition({"a": "forvo:somchai", "b": "forvo:somchai"})
    verdict = backend.fetch(_rendition_question({"near": "a", "far": "b"},
                                                checks={"near": True, "far": False}))
    assert verdict.value is False and "far" in verdict.evidence


def test_a_rendition_cannot_be_judged_without_its_members_own_verdicts():
    backend = _rendition({"a": "forvo:somchai", "b": "forvo:somchai"})
    with pytest.raises(PreparationError, match="far"):
        backend.fetch(_rendition_question({"near": "a", "far": "b"}, checks={"near": True}))


def test_the_rendition_key_identifies_the_member_set_not_its_order():
    backend = _rendition({})
    one = backend.cache_key(_rendition_question({"near": "a", "far": "b"}))
    other = backend.cache_key(_rendition_question({"far": "b", "near": "a"}))
    assert one == other
    assert one.artifact_sha == rendition_identity({"near": "a", "far": "b"})
    assert one.params == "p1"


def test_a_rendition_with_no_members_cannot_be_prepared():
    with pytest.raises(PreparationError):
        _rendition({}).fetch(_rendition_question({}))


# --- the judge's sentence prompt puts the gloss to the judge ---------------

def test_the_sentence_prompt_renders_the_gloss_it_asks_about():
    prompt = sentence_prompt(AssessQuestion(
        subject="s", role="sentence-for-target", rubric="R", kind="sentence",
        subject_kind="sentence",
        params={"text": "กินข้าว", "gloss": "eat rice", "word": "กิน"}))   # กินข้าว: eat rice
    assert "eat rice" in prompt and "กินข้าว" in prompt


def test_the_sentence_prompt_says_so_when_no_gloss_was_offered():
    prompt = sentence_prompt(AssessQuestion(
        subject="s", role="sentence-for-target", rubric="R", kind="sentence",
        params={"text": "กินข้าว", "word": "กิน"}))   # กินข้าว: eat rice
    assert "(none given)" in prompt


# --- Assessor.inline: the transport, not the shape of one result -----------

def test_an_api_judge_is_inline_and_a_batch_judge_is_not(tmp_path):
    db = SyllabusDb(tmp_path / "s.db")
    api = Assessor(record=db, cache=db, backends={
        "judge": JudgeBackend(model="m", transport="api", complete=lambda p, a=(): None)})
    batch = Assessor(record=db, cache=db, backends={
        "judge": JudgeBackend(model="m", transport="batch", batch_transport=object())})
    assert api.inline is True and batch.inline is False


def test_an_assessor_with_no_judge_at_all_is_not_inline(tmp_path):
    db = SyllabusDb(tmp_path / "s.db")
    assert Assessor(record=db, cache=db, backends={}).inline is False


# --- a verdict row carries the params its question was asked with ----------

def test_a_verdict_row_keeps_the_params_the_question_carried(tmp_path):
    db = SyllabusDb(tmp_path / "s.db")
    backend = MechanicalBackend(
        key_fn=lambda q: MechanicalKey(check="c", params="v1", artifact_sha="a"),
        evaluate=lambda q: RawVerdict(value=True))
    assessor = Assessor(record=db, cache=db, backends={"mech": backend})
    assessor.ask("mech", AssessQuestion(subject="s", role="r", kind="picture",
                                        subject_kind="sentence", params={"target": "t1"}))
    row = db.assessments_of("s")[-1]
    assert row.question["params"] == {"target": "t1"}
    assert row.question["subject_kind"] == "sentence"
