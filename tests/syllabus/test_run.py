"""Tests for run.py (spec 3 section 7): one pass per source and one judge
batch -- the previous batch resolved and its drafts adopted, one sentence
attempt, each queued need tried at its next source, every collected
question submitted as one batch, and a report accounting for every need
gaps() lists.

The first group runs the real Sourcing (build_sourcing over a fixture
deck) against fake search/Forvo/LLM backends and a fake batch transport;
the second drives run() over fake attempts to pin one report field at a
time.
"""
import dataclasses
import hashlib
import io
import json
from datetime import date

import pytest
from PIL import Image as PILImage

from thai_syllabus import run as run_mod
from thai_syllabus.assessor import JudgeUnreachable
from thai_syllabus.cachekeys import BatchMarkerKey
from thai_syllabus.attempts import AttemptResult, Sourcing, Spend
from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
from thai_syllabus.entities import Category
from thai_syllabus.profile import Profile
from thai_syllabus.provider import FetchBackend, RawAnswer
from thai_syllabus.run import (
    FORVO_DEFAULT_DAILY_BUDGET,
    LEARNER_DEFAULT_SESSION_BUDGET,
    Budget,
    run,
)
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.transport import Completion, TransportError
from thai_syllabus.wiring import build_sourcing

from .builders import sentence, target, word
from .fakes import FakeTokenizer

# --- a fixture deck and fake backends --------------------------------------

RICE = word("rice", "ข้าว", "rice")       # ข้าว = rice
FISH = word("fish", "ปลา", "fish")        # ปลา = fish
EAT = word("eat", "กิน", "eat")           # กิน = eat
EAT_RICE = "กินข้าว"                       # กินข้าว = eat rice


def _deck(tmp_path, words, targets, *, transport="batch"):
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=words, targets=targets, graphemes=(), confusions=(), pairs=(),
        profile=Profile(register="male_colloquial"), rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset(w.id for w in words)),)))
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n"
        "secrets: {anthropic: op://Shared/Anthropic/API Key}\n"
        f"judge: {{transport: {transport}, model: m, "
        "price_per_mtok: {input: 2.0, output: 10.0}}\n",
        encoding="utf-8")
    SyllabusDb(root / "syllabus.db").close()
    MediaStore(root / "media")
    return root


def _jpeg_bytes(seed: str) -> bytes:
    """A decodable JPEG whose colour derives from `seed`, so distinct urls
    ingest as distinct shas (MediaStore.add_image decodes what it stores).
    """
    digest = hashlib.sha256(seed.encode()).digest()
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2), tuple(digest[:3])).save(buf, format="JPEG")
    return buf.getvalue()


class _Search:
    """One image-search backend, recording its asks as (subject, source)
    in a list shared with every other source in the roster. Two hits per
    query: one picture alone can never need a preference question."""

    def __init__(self, name: str, asks: list):
        self.name, self.asks = name, asks

    def cache_key(self, q):
        return f"{self.name}:{q.params['query']}"

    def fetch(self, q):
        self.asks.append((q.subject, self.name))
        return RawAnswer(items=(
            {"url": f"https://{self.name}/{q.subject}-1.jpg", "source": self.name,
             "licence": "by"},
            {"url": f"https://{self.name}/{q.subject}-2.jpg", "source": self.name,
             "licence": "by"}))


class _Searches:
    """The image-search roster over one ordered ask log."""

    def __init__(self):
        self.asks: list[tuple[str, str]] = []

    def backend(self, name: str) -> _Search:
        return _Search(name, self.asks)


class _Silent:
    """A Source with nothing to offer -- a recording need attempts it and
    comes back empty."""

    def __init__(self, name: str):
        self.name = name

    def cache_key(self, q):
        return f"{self.name}:{sorted(q.params.items())}"

    def fetch(self, q):
        return RawAnswer()


class _Llm:
    """The sentence drafter: `drafts` is the JSON its one answer carries."""

    def __init__(self, drafts: str = '{"sentences": []}'):
        self.drafts = drafts

    def cache_key(self, q):
        return "llm:sentence-drafter:m:x"

    def fetch(self, q):
        return RawAnswer(items=(self.drafts,))


class _BatchTransport:
    """The judge's batch transport: submit() records a batch and answers
    "in_progress" until complete_all() gives every question the same
    verdict."""

    def __init__(self):
        self.submitted = 0
        self._requests: dict[str, dict] = {}
        self._results: dict[str, dict] = {}
        self._status: dict[str, str] = {}

    def submit(self, requests):
        self.submitted += 1
        batch_id = f"batch-{self.submitted}"
        self._requests[batch_id] = dict(requests)
        self._status[batch_id] = "in_progress"
        return batch_id

    def status(self, batch_id):
        return self._status[batch_id]

    def results(self, batch_id):
        return self._results.get(batch_id, {})

    def complete_all(self, batch_id, *, passed: bool):
        self._status[batch_id] = "ended"
        self._results[batch_id] = {
            custom_id: Completion(text=json.dumps({"value": passed, "evidence": "ok"}))
            for custom_id in self._requests[batch_id]}


@pytest.fixture
def fake_search():
    return _Searches()


@pytest.fixture
def fake_batch():
    return _BatchTransport()


def _wire(ctx, fake_search, *, llm=None, batch=None, complete=None):
    """Replaces every backend that would touch the network."""
    ctx.provider._backends.update({
        "openverse": fake_search.backend("openverse"),
        "wikimedia": fake_search.backend("wikimedia"),
        "pexels": fake_search.backend("pexels"),
        "forvo": _Silent("forvo"), "tts": _Silent("tts"),
        "llm-sentence": llm if llm is not None else _Llm(),
        "imgfetch": FetchBackend(media=ctx.media_store,
                                 fetcher=lambda url: (_jpeg_bytes(url), "jpg")),
        "audiofetch": FetchBackend(media=ctx.media_store,
                                   fetcher=lambda url: (url.encode(), "mp3"))})
    if batch is not None:
        ctx.assessor._backends["judge"].batch_transport = batch
    if complete is not None:
        ctx.assessor._backends["judge"].complete = complete
    return ctx


@pytest.fixture
def ctx_batch_two_needs(tmp_path, fake_search, fake_batch):
    """Two targeted words, a batch judge, and no sentence draft on offer."""
    root = _deck(tmp_path, (RICE, FISH),
                 (target("rice/receptive", "rice"), target("fish/receptive", "fish")))
    return _wire(build_sourcing(root), fake_search, batch=fake_batch)


@pytest.fixture
def ctx_inline_dead_judge(tmp_path, fake_search):
    """One targeted word and an inline judge whose wire is down."""
    def dead(prompt, attachments=()):
        raise TransportError("no judge")

    root = _deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),), transport="api")
    return _wire(build_sourcing(root), fake_search, complete=dead)


@pytest.fixture
def ctx_batch_sentences(tmp_path, fake_search, fake_batch):
    """One targeted word whose Target a drafted sentence claims, a batch
    judge, and no picture candidate to judge alongside it."""
    root = _deck(tmp_path, (RICE, EAT),
                 (target("rice/receptive", "rice"), target("eat/receptive", "eat")))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch,
                llm=_Llm(json.dumps({"sentences": [
                    {"text": EAT_RICE, "gloss": "eat rice",
                     "targets": ["rice/receptive", "eat/receptive"]}]})))
    ctx.provider._backends["openverse"] = _Silent("openverse")
    ctx.provider._backends["wikimedia"] = _Silent("wikimedia")
    ctx.provider._backends["pexels"] = _Silent("pexels")
    # กิน = eat, ข้าว = rice
    ctx.syllabus = dataclasses.replace(
        ctx.syllabus, tokenizer=FakeTokenizer({EAT_RICE: ["กิน", "ข้าว"]}))
    return ctx


# --- one pass per source, one batch ---------------------------------------

def test_run_tries_one_source_per_need_and_submits_one_batch(
        ctx_batch_two_needs, fake_search, fake_batch):
    report = run(ctx_batch_two_needs, budgets={})
    # queue() orders equal-rank needs by subject, so fish precedes rice.
    assert fake_search.asks == [("fish", "openverse"), ("rice", "openverse")]
    assert fake_batch.submitted == 1
    assert report.pending == 2 and report.batch_id


def test_second_run_resolves_then_escalates(ctx_batch_two_needs, fake_search, fake_batch):
    r1 = run(ctx_batch_two_needs, budgets={})
    fake_batch.complete_all(r1.batch_id, passed=False)   # every candidate failed fit
    r2 = run(ctx_batch_two_needs, budgets={})
    assert fake_search.asks[-2:] == [("fish", "wikimedia"), ("rice", "wikimedia")]
    assert r2.pending == 2 and r2.improved == 0


def test_run_asks_the_preference_question_once_the_fits_resolve(
        ctx_batch_two_needs, fake_batch):
    r1 = run(ctx_batch_two_needs, budgets={})
    fake_batch.complete_all(r1.batch_id, passed=True)    # both candidates fit
    r2 = run(ctx_batch_two_needs, budgets={})
    marker = ctx_batch_two_needs.db.latest(
        "assess", "judge", BatchMarkerKey(r2.batch_id))
    assert "picture-preference" in marker.question["roles"]


def test_a_batch_still_in_progress_holds_the_next_run_back(
        ctx_batch_two_needs, fake_search, fake_batch):
    """At most one batch is ever outstanding: until the last one answers,
    a run attempts nothing and submits nothing."""
    r1 = run(ctx_batch_two_needs, budgets={})
    asked = list(fake_search.asks)
    r2 = run(ctx_batch_two_needs, budgets={})
    assert fake_batch.submitted == 1 and r2.batch_id == r1.batch_id
    assert fake_search.asks == asked
    assert r2.attempted == 0 and r2.pending == 2


def test_run_stops_at_the_first_unreachable_judge(ctx_inline_dead_judge):
    report = run(ctx_inline_dead_judge, budgets={})
    assert report.unreachable and report.attempted == 1


def test_run_adopts_sentences_whose_verdicts_resolved(ctx_batch_sentences, fake_batch):
    r1 = run(ctx_batch_sentences, budgets={})
    fake_batch.complete_all(r1.batch_id, passed=True)
    r2 = run(ctx_batch_sentences, budgets={})
    assert r2.sentences_adopted == 1
    assert ctx_batch_sentences.db.all_sentences()[0].gloss == "eat rice"


# --- the report, one field at a time ---------------------------------------

@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _Gaps:
    def __init__(self, pictures=(), recordings=(), sentences=(), graphemes=()):
        self.words_missing_pictures, self.words_missing_recordings = pictures, recordings
        self.unfilled_targets, self.missing_renditions = sentences, ()
        self.graphemes_missing_keyword_data = graphemes


class _Syl:
    def __init__(self, gaps):
        self._gaps, self.targets, self.sentences = gaps, [], ()

    def gaps(self):
        return self._gaps

    def cover(self, drafts):
        return []

    def with_sentences(self, new):
        return self


class _Assessor:
    """The Assess port as run() uses it: an outstanding batch to resolve,
    and one submission per run."""
    inline = True

    def __init__(self, outstanding=None):
        self._outstanding = outstanding
        self.resolved: list[str] = []
        self.submitted: list[list] = []

    def unresolved_batch(self):
        return self._outstanding

    def resolve(self, batch_id):
        self.resolved.append(batch_id)
        self._outstanding = None
        return {}

    def submit(self, prepared):
        if not prepared:
            return None
        self.submitted.append(list(prepared))
        return f"batch-{len(self.submitted)}"


def _ctx(db, syl, assessor=None):
    return Sourcing(syllabus=syl, provider=None, assessor=assessor or _Assessor(),
                    db=db, media_store=None, rubrics={}, provenance_prior=())


class _Q:
    """A collected question as run() reads it: its own question's subject."""

    def __init__(self, subject: str):
        self.question = type("_AskedAbout", (), {"subject": subject})()


def _patch(monkeypatch, results, sentence_result=AttemptResult(attempted=False), drafts=(),
           preference=AttemptResult(attempted=False)):
    """Replaces attempt/sentence_attempt/preference_attempt/adoptable_drafts.
    A `results` entry that is an exception class is raised instead of
    returned."""
    calls = []

    def fake_attempt(ctx, need, source):
        calls.append((need, source))
        result = results.get((need.subject, source),
                             AttemptResult(True, spend={source: Spend(1, 0.0)}))
        if isinstance(result, type) and issubclass(result, Exception):
            raise result("no source")
        return result

    def fake_sentence_attempt(ctx, max_targets=40):
        if isinstance(sentence_result, type) and issubclass(sentence_result, Exception):
            raise sentence_result("no judge")
        return sentence_result

    monkeypatch.setattr(run_mod, "attempt", fake_attempt)
    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "preference_attempt", lambda ctx, subjects: preference)
    monkeypatch.setattr(run_mod, "adoptable_drafts",
                        lambda cache, syllabus, **kwargs: list(drafts))
    return calls


def test_run_asks_one_source_per_need_and_leaves_escalation_to_the_next_run(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("w",)))), {})
    assert [s for _need, s in calls] == ["openverse"]
    assert report.attempted == 1


def test_run_reports_every_need_gaps_lists_as_available(db, monkeypatch):
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",), recordings=("a",), sentences=("t1",)))), {})
    assert report.available == 3


def test_run_counts_a_need_with_no_source_left_as_exhausted(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "next_source", lambda *a, **k: None)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert calls == [] and report.exhausted == 1 and report.attempted == 0


def test_run_counts_a_need_the_queue_dropped_as_exhausted(db, monkeypatch):
    _patch(monkeypatch, {})
    for source in ("openverse", "wikimedia", "pexels"):
        db.append(port="provide", backend=source, key=f"{source}:a", subject="a",
                  question={"kind": "picture", "subject_kind": "word"},
                  answer={"items": []})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert report.exhausted == 1 and report.available == 1 and report.attempted == 0


def test_run_never_attempts_sentence_needs_per_subject(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert calls == []


def test_run_skips_a_need_whose_source_budget_is_spent(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(recordings=("a", "b")))), {"forvo": Budget(max_asks=1)})
    assert [(n.subject, s) for n, s in calls] == [("a", "forvo")]


def test_run_sums_excluded_candidates_across_attempts(db, monkeypatch):
    _patch(monkeypatch, {
        ("a", "openverse"): AttemptResult(True, excluded={"k1": "gone"}),
        ("b", "openverse"): AttemptResult(True, excluded={"k2": "gone", "k3": "gone"})},
        sentence_result=AttemptResult(False, excluded={"k4": "gone"}))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert report.excluded == 4
    assert db.latest("run", "runreport", "runreport").answer["excluded"] == 4


def test_a_run_with_nothing_wrong_reports_zero_excluded_and_reachable(db, monkeypatch):
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert report.excluded == 0 and report.unreachable is False
    assert report.source_failures == {}


def test_run_counts_a_need_whose_current_best_changed_as_improved(db, monkeypatch):
    def lands_a_picture(ctx, need, source):
        db.append(port="provide", backend=source, key=f"{source}:{need.subject}",
                  subject=need.subject,
                  question={"kind": "picture", "subject_kind": "word"},
                  answer={"items": [{"sha": "pic-1"}]})
        db.append(port="assess", backend="judge", key="judge:0:pic-1:picture-for-word",
                  subject=need.subject,
                  question={"role": "picture-for-word", "artifact_sha": "pic-1",
                            "rubric": None, "kind": "picture", "subject_kind": "word"},
                  answer={"value": True})
        return AttemptResult(True)

    _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "attempt", lands_a_picture)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert report.improved == 1


def test_run_records_the_spend_each_attempt_incurred(db, monkeypatch):
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert report.spend["openverse"].asks == 2


def test_run_submits_every_collected_question_as_one_batch(db, monkeypatch):
    assessor = _Assessor()
    _patch(monkeypatch, {("a", "openverse"): AttemptResult(True, questions=[_Q("a")]),
                         ("b", "openverse"): AttemptResult(True, questions=[_Q("b")])})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b"))), assessor), {})
    assert len(assessor.submitted) == 1 and len(assessor.submitted[0]) == 2
    assert report.batch_id == "batch-1" and report.pending == 2


def test_a_run_that_collected_nothing_submits_no_batch(db, monkeypatch):
    assessor = _Assessor()
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert assessor.submitted == [] and report.batch_id is None and report.pending == 0


def test_run_resolves_the_previous_batch_and_counts_one_still_out_as_pending(db, monkeypatch):
    class _Stuck(_Assessor):
        def resolve(self, batch_id):
            self.resolved.append(batch_id)   # still in progress: nothing released
            return {}

    assessor = _Stuck(outstanding=("batch-0", frozenset({"a", "b"})))
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert assessor.resolved == ["batch-0"] and report.pending == 2
    assert report.batch_id == "batch-0"
    assert calls == [] and assessor.submitted == [] and report.attempted == 0


def test_run_collects_the_preference_questions_of_a_resolved_batch(db, monkeypatch):
    assessor = _Assessor(outstanding=("batch-0", frozenset({"a"})))
    _patch(monkeypatch, {}, preference=AttemptResult(True, questions=[_Q("a")]))
    report = run(_ctx(db, _Syl(_Gaps()), assessor), {})
    assert len(assessor.submitted[0]) == 1 and report.pending == 1


def test_unfilled_targets_are_available_work_and_never_exhausted(db, monkeypatch):
    """The sentence attempt serves every open Target once per run: no
    Source is asked per Target, so none of them is ever out of options."""
    _patch(monkeypatch, {}, sentence_result=AttemptResult(True, drafted=2))
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1", "t2", "t3")))), {})
    assert report.available == 3 and report.exhausted == 0 and report.attempted == 0


def test_run_reports_the_drafts_the_sentence_attempt_produced(db, monkeypatch):
    _patch(monkeypatch, {}, sentence_result=AttemptResult(True, drafted=2))
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert report.drafted == 2
    assert db.latest("run", "runreport", "runreport").answer["drafted"] == 2


def _row_today(db, backend, subject, *, ts):
    db.append(port="provide", backend=backend, key=f"{backend}:{subject}:{ts}",
              subject=subject, question={"kind": "recording", "subject_kind": "word"},
              answer={"items": []}, cost=0.0, ts=ts)


def test_todays_asks_already_on_record_count_against_a_per_day_budget(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    midnight = run_mod.day_start_ns(date.today())
    _row_today(db, "forvo", "x", ts=midnight + 1)
    _row_today(db, "forvo", "y", ts=midnight + 2)
    run(_ctx(db, _Syl(_Gaps(recordings=("a",)))), {"forvo": Budget(max_asks=2)})
    assert calls == []


def test_yesterdays_asks_do_not_count_against_it(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    midnight = run_mod.day_start_ns(date.today())
    _row_today(db, "forvo", "x", ts=midnight - 2)
    _row_today(db, "forvo", "y", ts=midnight - 1)
    run(_ctx(db, _Syl(_Gaps(recordings=("a",)))), {"forvo": Budget(max_asks=2)})
    assert [(n.subject, s) for n, s in calls] == [("a", "forvo")]


# --- what went wrong -------------------------------------------------------

def test_run_stops_the_loop_at_the_first_unreachable_judge(db, monkeypatch):
    """An unreachable judge is a dead wire, not a per-need failure: every
    remaining need would fail the same way."""
    calls = _patch(monkeypatch, {("a", "openverse"): JudgeUnreachable})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert [n.subject for n, _s in calls] == ["a"]   # no second need
    assert report.unreachable is True and report.attempted == 1
    assert report.available == 2
    assert db.latest("run", "runreport", "runreport").answer["unreachable"] is True


def test_an_unreachable_sentence_attempt_stops_the_run_too(db, monkeypatch):
    assessor = _Assessor()
    calls = _patch(monkeypatch, {}, sentence_result=JudgeUnreachable)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert calls == []                               # no need attempted at all
    assert report.unreachable is True
    assert assessor.submitted == []                  # nothing goes out after that


def test_a_judge_that_cannot_be_reached_to_resolve_stops_the_run(db, monkeypatch):
    class _DeadResolve(_Assessor):
        def resolve(self, batch_id):
            raise JudgeUnreachable("batch status failed")

    assessor = _DeadResolve(outstanding=("batch-0", frozenset({"a"})))
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert report.unreachable is True and calls == [] and assessor.submitted == []
    assert report.batch_id == "batch-0" and report.pending == 1
    assert db.latest("run", "runreport", "runreport").answer["unreachable"] is True


def test_a_judge_that_cannot_be_reached_to_submit_stops_the_run(db, monkeypatch):
    class _DeadSubmit(_Assessor):
        def submit(self, prepared):
            raise JudgeUnreachable("batch submit failed")

    assessor = _DeadSubmit()
    _patch(monkeypatch, {("a", "openverse"): AttemptResult(True, questions=[_Q("a")])})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert report.unreachable is True and report.batch_id is None and report.pending == 0
    assert db.latest("run", "runreport", "runreport").answer["unreachable"] is True


def test_a_dead_source_is_counted_and_skipped_for_the_rest_of_the_run(db, monkeypatch):
    calls = _patch(monkeypatch, {("a", "openverse"): TransportError})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert [n.subject for n, _s in calls] == ["a"]   # b's next source is the dead one
    assert report.source_failures == {"openverse": 1}
    assert report.unreachable is False
    assert db.latest("run", "runreport", "runreport").answer["source_failures"] == {
        "openverse": 1}


# --- adoption: the cover over what the judge passed ------------------------

class _AdoptingSyl:
    """Stands in for the real Syllabus's adoption effect: cover() takes the
    verified drafts whose Targets are still open, and with_sentences()
    closes those Targets."""

    def __init__(self, unfilled_targets):
        self.targets, self.sentences = [], ()
        self._unfilled = tuple(unfilled_targets)
        self._covered: tuple[str, ...] = ()

    def gaps(self):
        return _Gaps(sentences=self._unfilled)

    def cover(self, drafts):
        chosen = [(s, ts) for s, ts in drafts if any(t in self._unfilled for t in ts)]
        self._covered = tuple(t for _s, ts in chosen for t in ts)
        return chosen

    def with_sentences(self, new):
        return _AdoptingSyl(t for t in self._unfilled if t not in self._covered)


def test_run_adopts_a_cover_of_the_adoptable_drafts_before_computing_the_queue(db, monkeypatch):
    captured = []

    def fake_queued(syllabus, cache, **kwargs):
        captured.append(syllabus)
        return run_mod.QueuedNeeds(entries=[], available=len(syllabus.gaps().unfilled_targets),
                                   exhausted=0)

    monkeypatch.setattr(run_mod, "queued", fake_queued)
    _patch(monkeypatch, {}, drafts=[(sentence("x", gloss="ex"), ("t1",))])

    report = run(_ctx(db, _AdoptingSyl(("t1", "t2"))), {})

    assert report.sentences_adopted == 1
    assert [s.text for s in db.all_sentences()] == ["x"]
    assert captured[0].gaps().unfilled_targets == ("t2",)   # t1 already covered


# --- documented defaults (spec 3 section 7) -------------------------------

def test_forvo_default_daily_budget_is_450_asks():
    assert FORVO_DEFAULT_DAILY_BUDGET.max_asks == 450


def test_learner_default_session_budget_is_20_questions():
    assert LEARNER_DEFAULT_SESSION_BUDGET.max_asks == 20


# --- RunReport persistence: /stats history needs a source ----------------

def test_a_run_that_does_almost_nothing_still_appends_a_row(db, monkeypatch):
    _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps())), {})
    rows = [r for r in db.assessments_of("run") if r.port == "run"]
    assert len(rows) == 1
    assert rows[0].question["kind"] == "runreport"


def test_two_runs_each_get_their_own_keyed_row(db, monkeypatch):
    _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps())), {})
    run(_ctx(db, _Syl(_Gaps())), {})
    rows = [r for r in db.assessments_of("run") if r.port == "run"]
    assert len(rows) == 2
    assert rows[0].ts != rows[1].ts


def test_the_persisted_row_carries_every_report_field(db, monkeypatch):
    _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    answer = db.latest("run", "runreport", "runreport").answer
    assert set(answer) == {"attempted", "improved", "exhausted", "available", "pending",
                           "sentences_adopted", "drafted", "excluded", "unreachable",
                           "batch_id", "source_failures", "spend"}


# --- Spend ------------------------------------------------------------

def test_spend_add_increments_by_given_asks():
    s = Spend()
    s.add(3, 0.6)
    assert s.asks == 3 and s.cost == pytest.approx(0.6)


def test_a_budget_says_when_a_spend_has_reached_it():
    assert Budget(max_asks=5).exceeded_by(Spend(asks=5))
    assert not Budget(max_asks=5).exceeded_by(Spend(asks=4))
    assert Budget(max_cost=1.0).exceeded_by(Spend(cost=1.0))
    assert not Budget(max_cost=1.0).exceeded_by(Spend(cost=0.5))
