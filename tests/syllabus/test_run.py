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
from thai_syllabus.assessor import Excluded, JudgeUnreachable, RawVerdict
from thai_syllabus.cachekeys import (AttemptOutcomeKey, BatchMarkerKey, DirectionKey, JudgeKey,
                                    LearnerKey, LlmPromptKey, MechanicalKey, ProvideKey,
                                    RunReportKey, sha)
from thai_syllabus.attempts import AttemptResult, Sourcing, Spend, sources_for
from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
from thai_syllabus.derivations import available_need_keys, current_best, next_source, open_words
from thai_syllabus.entities import Category, MinimalPair, SoundConfusion, text_sha
from thai_syllabus.ids import ConfusionId, PairId, WordId
from thai_syllabus.profile import Profile
from thai_syllabus.provider import FetchBackend, RawAnswer
from thai_syllabus.record import rows_for
from thai_syllabus.run import (
    FORVO_DEFAULT_DAILY_BUDGET,
    LEARNER_DEFAULT_SESSION_BUDGET,
    Budget,
    run,
)
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.transport import Completion, TransportError
from thai_syllabus.wiring import build_sourcing

from .builders import sentence, syl, target, thai_of, word

# --- a fixture deck and fake backends --------------------------------------

RICE = word("rice", "ข้าว", "rice")       # ข้าว = rice
FISH = word("fish", "ปลา", "fish")        # ปลา = fish
EAT = word("eat", "กิน", "eat")           # กิน = eat
EAT_RICE = "กินข้าว"                       # กินข้าว = eat rice


def _deck(tmp_path, words, targets, *, transport="batch", pairs=(), confusions=()):
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=words, targets=targets, graphemes=(), confusions=confusions, pairs=pairs,
        profile=Profile(register="male_colloquial"), rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset(w.id for w in words)),)))
    (root / "curated" / "frequency_th.txt").write_text("", encoding="utf-8")
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
        return ProvideKey(source=self.name, kind="", query=q.params["query"])

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
        return ProvideKey(source=self.name, kind="", query=str(sorted(q.params.items())))

    def fetch(self, q):
        return RawAnswer()


class _Llm:
    """The sentence drafter: `drafts` is the JSON its one answer carries."""

    def __init__(self, drafts: str = '{"sentences": []}'):
        self.drafts = drafts

    def cache_key(self, q):
        return LlmPromptKey(producer="sentence-drafter", model="m", prompt_sha="x")

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
                # กิน = eat, ข้าว = rice, one clause so it renders กินข้าว with no space
                llm=_Llm(json.dumps({"sentences": [
                    {"clauses": [["eat", "rice"]], "text": EAT_RICE, "gloss": "eat rice"}]})))
    ctx.provider._backends["openverse"] = _Silent("openverse")
    ctx.provider._backends["wikimedia"] = _Silent("wikimedia")
    ctx.provider._backends["pexels"] = _Silent("pexels")
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
    # 1 for rice's still-open Target (the sentence attempt ran and handed
    # it over, drafting nothing) + 1 for the picture need that hit the
    # dead judge before the loop stopped.
    assert report.unreachable and report.attempted == 2


def test_run_adopts_sentences_whose_verdicts_resolved(ctx_batch_sentences, fake_batch):
    r1 = run(ctx_batch_sentences, budgets={})
    fake_batch.complete_all(r1.batch_id, passed=True)
    r2 = run(ctx_batch_sentences, budgets={})
    assert r2.sentences_adopted == 1
    assert ctx_batch_sentences.db.all_sentences()[0].gloss == "eat rice"


def test_an_adopted_sentence_reaches_its_recording_and_picture_needs_this_run(
        ctx_batch_sentences, fake_batch):
    """C1: an adopted sentence carries a gloss (spec 1); with neither a
    recording nor a scene picture yet, both needs are queued and
    attempted in the run that adopts it, and the run's own accounting
    identity holds."""
    r1 = run(ctx_batch_sentences, budgets={})
    fake_batch.complete_all(r1.batch_id, passed=True)
    r2 = run(ctx_batch_sentences, budgets={})
    assert r2.sentences_adopted == 1

    sentence_sha = text_sha(EAT_RICE)
    recording_rows = rows_for(ctx_batch_sentences.db, sentence_sha, "recording")
    picture_rows = rows_for(ctx_batch_sentences.db, sentence_sha, "picture")
    assert any(r.port == "provide" and r.backend == "forvo" for r in recording_rows)
    assert any(r.port == "provide" and r.backend == "openverse" for r in picture_rows)

    assert (r2.available == r2.attempted + r2.exhausted + r2.pending
           + r2.unserved + r2.budgeted + r2.deferred)


@pytest.fixture
def ctx_inline_sentences(tmp_path, fake_search):
    """Two targeted words, one drafted sentence claiming both their
    Targets, and an inline judge that passes: the draft is verified and
    adopted inside the same run that drafts it."""
    root = _deck(tmp_path, (RICE, EAT),
                 (target("rice/receptive", "rice"), target("eat/receptive", "eat")),
                 transport="api")
    ctx = _wire(build_sourcing(root), fake_search,
                # กิน = eat, ข้าว = rice, one clause so it renders กินข้าว with no space
                llm=_Llm(json.dumps({"sentences": [
                    {"clauses": [["eat", "rice"]], "text": EAT_RICE, "gloss": "eat rice"}]})),
                complete=lambda prompt, attachments=(): Completion(
                    text=json.dumps({"value": True, "evidence": "ok"})))
    ctx.provider._backends["openverse"] = _Silent("openverse")
    ctx.provider._backends["wikimedia"] = _Silent("wikimedia")
    ctx.provider._backends["pexels"] = _Silent("pexels")
    return ctx


def test_a_word_whose_targets_this_run_adopted_leaves_available_and_every_bucket(
        ctx_inline_sentences):
    """The draft the sentence attempt verified inline closed both words'
    Targets in the run that produced it: neither word's "sentence" need
    is available afterwards, and neither reaches attempted, budgeted or
    deferred -- over a real queued(), the identity still holds."""
    report = run(ctx_inline_sentences, budgets={})

    assert report.sentences_adopted == 1
    assert open_words(ctx_inline_sentences.syllabus) == frozenset()
    assert not [need for need in available_need_keys(ctx_inline_sentences.syllabus)
                if need[1] == "sentence"]
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


# --- a pair's rendition need reaches the attempt (F1 defect 1) -------------

def test_a_pairs_rendition_need_reaches_the_attempt(tmp_path, fake_search, fake_batch):
    """derivations.available_needs must key a rendition need by the
    pair's own id, not its confusion id: attempts._rendition_attempt
    looks the pair up by PairId (ctx.syllabus.pair(...)).
    """
    near = word("near_tone", "ใกล้", "near", syllables=(syl(tone="low"),))  # near
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = MinimalPair.create(id=PairId("p-rice-near"), confusion=confusion,
                              members=(RICE, near))
    root = _deck(tmp_path, (RICE, near), (), pairs=(pair,), confusions=(confusion,))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch)

    report = run(ctx, budgets={})  # must not raise KeyError

    # the attempt appended its ask under the pair's own subject.
    assert rows_for(ctx.db, "p-rice-near", "rendition")
    assert report.available >= 1


# --- a transient failure counts as tried only at the transient cap (spec 3
# section 6a, r7, B8) -------------------------------------------------------

class _LookupOnceForvo:
    """Forvo answers once with one item; the per-item download always
    fails on the wire, never the lookup itself."""
    def __init__(self, item):
        self.item, self.calls = item, 0

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params.get("word", q.subject))

    def fetch(self, q):
        self.calls += 1
        return RawAnswer(items=(self.item,), cost=1.0)


class _DeadAudiofetch:
    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        raise TransportError("audiofetch is down")


def _silence_pictures(ctx):
    """Every picture search backend answers with no candidates."""
    ctx.provider._backends["openverse"] = _Silent("openverse")
    ctx.provider._backends["wikimedia"] = _Silent("wikimedia")
    ctx.provider._backends["pexels"] = _Silent("pexels")


def test_a_transient_download_failure_retries_the_same_source_next_run(
        tmp_path, fake_search, fake_batch):
    """The forvo lookup succeeds but its only download fails on the wire
    (caught inside the attempt, spec 3 section 3 -- the ask itself never
    raises): the attempt writes a transient-failure outcome (ruling 2),
    which next_source counts as tried only at the transient cap (spec 3
    section 6a); a second run's _try_each_need asks forvo again rather
    than escalating to tts -- through the real record, run.py itself
    unchanged.
    """
    root = _deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch)
    _silence_pictures(ctx)
    forvo = _LookupOnceForvo({"username": "somchai", "pathmp3": "https://f/u.mp3"})
    ctx.provider._backends["forvo"] = forvo
    ctx.provider._backends["audiofetch"] = _DeadAudiofetch()

    r1 = run(ctx, budgets={})
    assert r1.source_failures == {}          # the download's own catch never propagates
    assert next_source(ctx.db, "rice", "recording", sources_for("recording"),
                       transient_cap=ctx.transient_cap) == "forvo"

    r2 = run(ctx, budgets={})
    assert next_source(ctx.db, "rice", "recording", sources_for("recording"),
                       transient_cap=ctx.transient_cap) == "forvo"
    assert forvo.calls == 1   # forvo's own lookup is cached forever, never re-fetched


def test_a_nothing_outcome_advances_to_the_next_source(tmp_path, fake_search, fake_batch):
    """A source that answers with nothing usable writes a "nothing"
    outcome, which next_source does count as tried -- the next run's
    attempt escalates to the next-cheapest source.
    """
    root = _deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch)
    _silence_pictures(ctx)
    # forvo and tts are already _Silent by _wire's own default, answering
    # with nothing.

    run(ctx, budgets={})
    assert next_source(ctx.db, "rice", "recording", sources_for("recording"),
                       transient_cap=ctx.transient_cap) == "tts"


# --- r8 fix round 2: a learner supply reopens an exhausted need over a
# real run() (architecture section 4) ---------------------------------------

class _EmptyForvo:
    """A forvo lookup answered once with nothing; a repeat ask on the same
    key is Provider's own cache hit (spec 3 section 2), zero cost and no
    second call to fetch().
    """
    def __init__(self):
        self.calls = 0

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params.get("word", ""))

    def fetch(self, q):
        self.calls += 1
        return RawAnswer()


class _EmptyTts:
    def cache_key(self, q):
        return ProvideKey(source="tts", kind="", query=str(sorted(q.params.items())))

    def fetch(self, q):
        return RawAnswer()


class _PassingMechanical:
    """Passes every recording it is asked about -- stands in for ffprobe
    over the real recording attempt pipeline."""
    def cache_key(self, q):
        return MechanicalKey(check="duration", params="0.2-5.0",
                             artifact_sha=q.artifact_sha or "-")

    def fetch(self, q):
        return RawVerdict(value=True, evidence="ok")


def test_a_learner_supply_reopens_an_exhausted_recording_need_over_a_real_run(tmp_path):
    """r8 fix round 2, ruling 1(b): two runs exhaust "rice"'s recording
    sources (forvo, tts, both empty). A learner supply nominates a sha
    (an "unacceptable-use-this" rating, which never vetoes). A third run's
    re-ask of forvo is Provider's own cache hit -- no new fetch -- but the
    attempt's mechanical check still covers every candidate sha under the
    need, the supplied one included: current_best lands on it, source
    "mechanical", and the run's own accounting identity holds.
    """
    root = _deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = build_sourcing(root)
    forvo = _EmptyForvo()
    ctx.provider._backends.update({
        "openverse": _Silent("openverse"), "wikimedia": _Silent("wikimedia"),
        "pexels": _Silent("pexels"), "forvo": forvo, "tts": _EmptyTts(),
        "llm-sentence": _Llm(),
    })
    ctx.assessor._backends["mechanical"] = _PassingMechanical()

    run(ctx, budgets={})   # forvo tried, nothing
    run(ctx, budgets={})   # tts tried, nothing -- the recording need is now exhausted

    supplied_sha = "s" * 64
    ctx.db.append(port="provide", backend="learner",
                 key=ProvideKey(source="learner", kind="", query="supplied.mp3"),
                 subject="rice",
                 question={"provides": "recording-bytes", "kind": "recording",
                          "subject_kind": "word", "params": {"path": "supplied.mp3"}},
                 answer={"items": [{"sha": supplied_sha, "ext": "mp3"}]})
    ctx.db.append(port="assess", backend="learner",
                 key=LearnerKey(artifact_sha=supplied_sha, role="recording-for-word"),
                 subject="rice",
                 question={"role": "recording-for-word", "artifact_sha": supplied_sha,
                          "rubric": None, "kind": "rating", "subject_kind": "word"},
                 answer={"value": "unacceptable-use-this"})

    report = run(ctx, budgets={})
    assert forvo.calls == 1   # the re-ask was Provider's cache hit, not a new fetch

    best = current_best(ctx.db, "rice", "recording", current_rubric={}, prior=(),
                        provenance_source=lambda artifact_sha: None)
    assert best.artifact_sha == supplied_sha
    assert best.source == "mechanical"
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


# --- the report, one field at a time ---------------------------------------

@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _Gaps:
    def __init__(self, pictures=(), recordings=(), sentences=(), graphemes=()):
        self.words_missing_pictures, self.words_missing_recordings = pictures, recordings
        self.unfilled_targets, self.missing_renditions = sentences, ()
        self.graphemes_missing_keyword_data = graphemes
        self.sentence_recordings, self.scene_pictures = (), ()


class _Syl:
    def __init__(self, gaps):
        self._gaps, self.targets, self.sentences, self.pairs = gaps, [], (), ()

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
    """A collected question as run() reads it: its own question's subject
    and kind."""

    def __init__(self, subject: str, kind: str = "picture"):
        self.question = type("_AskedAbout", (), {"subject": subject, "kind": kind})()


def _patch(monkeypatch, results, sentence_result=AttemptResult(attempted=False), drafts=(),
           preference=AttemptResult(attempted=False), assess=None):
    """Replaces attempt/assess_first/sentence_attempt/preference_attempt/
    adoptable_drafts. A `results` entry that is an exception class is
    raised instead of returned. `assess` is assess_first's fixed return
    for every need -- None keeps the fall-through to the source."""
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
    monkeypatch.setattr(run_mod, "assess_first", lambda ctx, need: assess)
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


def test_run_reports_unserved_needs_from_the_queue(db, monkeypatch):
    """grapheme-keyword needs (no Source, no per-run pass) are queued()'s
    own count, threaded through unchanged."""
    def fake_queued(syllabus, cache, **kwargs):
        return run_mod.QueuedNeeds(entries=[], available=2, exhausted=0, unserved=2)

    monkeypatch.setattr(run_mod, "queued", fake_queued)
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps())), {})
    assert report.unserved == 2
    assert db.latest("run", "runreport", RunReportKey()).answer["unserved"] == 2


def test_a_cap_of_one_on_llm_sentence_with_no_prior_spend_lets_the_sentence_attempt_run(
        db, monkeypatch):
    calls = []

    def fake_sentence_attempt(ctx, max_targets=40):
        calls.append(1)
        return AttemptResult(True, drafted=1)

    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "adoptable_drafts", lambda cache, syllabus, **kwargs: [])
    report = run(_ctx(db, _Syl(_Gaps())), {"llm-sentence": Budget(max_asks=1)})
    assert calls == [1]
    assert report.drafted == 1


def test_a_cap_of_one_on_llm_sentence_already_spent_today_skips_the_sentence_attempt(
        db, monkeypatch):
    calls = []

    def fake_sentence_attempt(ctx, max_targets=40):
        calls.append(1)
        return AttemptResult(True, drafted=1)

    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "adoptable_drafts", lambda cache, syllabus, **kwargs: [])
    midnight = run_mod.day_start_ns(date.today())
    _row_today(db, "llm-sentence", "x", ts=midnight + 1)
    report = run(_ctx(db, _Syl(_Gaps())), {"llm-sentence": Budget(max_asks=1)})
    assert calls == []
    assert report.drafted == 0


def test_run_counts_a_need_with_no_source_left_as_exhausted(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "next_source", lambda *a, **k: None)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert calls == [] and report.exhausted == 1 and report.attempted == 0


def test_run_counts_a_need_the_queue_dropped_as_exhausted(db, monkeypatch):
    _patch(monkeypatch, {})
    _exhaust(db, "a")
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert report.exhausted == 1 and report.available == 1 and report.attempted == 0


def test_a_need_whose_candidate_awaits_a_verdict_is_assessed_and_no_source_asked(db, monkeypatch):
    calls = _patch(monkeypatch, {}, assess=AttemptResult(True, questions=[_Q("a")]))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert calls == []
    assert report.pending == 1 and report.attempted == 0 and report.available == 1


def test_an_assess_first_need_answered_inline_counts_as_attempted(db, monkeypatch):
    calls = _patch(monkeypatch, {}, assess=AttemptResult(True))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert calls == [] and report.attempted == 1


def test_assess_first_returning_none_falls_through_to_the_source(db, monkeypatch):
    calls = _patch(monkeypatch, {}, assess=None)
    run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert [(n.subject, s) for n, s in calls] == [("a", "openverse")]


def test_assess_first_runs_before_the_source_budget_check(db, monkeypatch):
    calls = _patch(monkeypatch, {}, assess=AttemptResult(True, questions=[_Q("a")]))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {"openverse": Budget(max_asks=0)})
    assert calls == [] and report.budgeted == 0 and report.pending == 1


def test_assess_first_runs_for_a_need_with_no_source_left(db, monkeypatch):
    calls = _patch(monkeypatch, {}, assess=AttemptResult(True, questions=[_Q("a")]))
    monkeypatch.setattr(run_mod, "next_source", lambda *a, **k: None)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert calls == [] and report.exhausted == 0 and report.pending == 1


def test_an_unreachable_judge_in_assess_first_stops_the_run(db, monkeypatch):
    def dead(ctx, need):
        raise JudgeUnreachable("judge down")
    calls = _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "assess_first", dead)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert calls == []
    assert report.unreachable is True and report.attempted == 1 and report.deferred == 1


def test_run_never_attempts_sentence_needs_per_subject(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert calls == []


def test_run_keeps_the_identity_for_a_directed_sentence_need(db, monkeypatch):
    """I1: a directed subject with an unfilled Target is still served only
    by the sentence attempt's own accounting -- queued() never doubles it
    with a "sentence"-kind entry of its own."""
    calls = _patch(monkeypatch, {})
    db.append(port="assess", backend="learner",
             key=DirectionKey(subject="t1", role="sentence-for-target",
                              text_sha=sha("try again")),
             subject="t1",
             question={"kind": "direction"}, answer={"direction": "try again"})
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert calls == []
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_run_skips_a_need_whose_source_budget_is_spent(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(recordings=("a", "b")))), {"forvo": Budget(max_asks=1)})
    assert [(n.subject, s) for n, s in calls] == [("a", "forvo")]
    assert report.budgeted == 1


def test_a_spent_learner_budget_never_counts_toward_the_run_s_budgeted_bucket(db, monkeypatch):
    """spec 3 section 7: "learner" names the question session's own
    Budget, never a Source -- next_source/sources_for never return it, so
    a spent budgets["learner"] entry gates no real need's attempt."""
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(recordings=("a",)))), {"learner": Budget(max_asks=0)})
    assert [(n.subject, s) for n, s in calls] == [("a", "forvo")]
    assert report.budgeted == 0


def test_run_sums_excluded_candidates_across_attempts(db, monkeypatch):
    _patch(monkeypatch, {
        ("a", "openverse"): AttemptResult(True, excluded={
            "k1": Excluded(subject="a", artifact_sha="s1", reason="gone")}),
        ("b", "openverse"): AttemptResult(True, excluded={
            "k2": Excluded(subject="b", artifact_sha="s2", reason="gone"),
            "k3": Excluded(subject="b", artifact_sha="s3", reason="gone")})},
        sentence_result=AttemptResult(False, excluded={
            "k4": Excluded(subject="sentence-drafts", artifact_sha=None, reason="gone")}))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert report.excluded == 4
    assert db.latest("run", "runreport", RunReportKey()).answer["excluded"] == 4
    assert len(report.excluded_items) == 4
    assert {item["subject"] for item in report.excluded_items} == {"a", "b", "sentence-drafts"}


def test_run_keeps_two_no_artifact_exclusions_on_one_subject_distinct(db, monkeypatch):
    """Two questions under the same subject that both name no artifact_sha
    (e.g. one `fills` question per Target under one sentence draft) must
    not collide under the same excluded key: each keeps its own
    excluded_items entry (assessor.ManyResult.excluded is keyed by the
    question's own typed CacheKey, not by subject/artifact_sha).
    """
    _patch(monkeypatch, {
        ("a", "openverse"): AttemptResult(True, excluded={
            "key-1": Excluded(subject="a", artifact_sha=None, reason="no target t1"),
            "key-2": Excluded(subject="a", artifact_sha=None, reason="no target t2")})})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert report.excluded == 2
    assert [(item["subject"], item["reason"]) for item in report.excluded_items] == [
        ("a", "no target t1"), ("a", "no target t2")]


def test_a_run_with_nothing_wrong_reports_zero_excluded_and_reachable(db, monkeypatch):
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert report.excluded == 0 and report.unreachable is False
    assert report.source_failures == {}


def test_run_counts_a_need_whose_current_best_changed_as_improved(db, monkeypatch):
    def lands_a_picture(ctx, need, source):
        db.append(port="provide", backend=source,
                  key=ProvideKey(source=source, kind="", query=need.subject),
                  subject=need.subject,
                  question={"kind": "picture", "subject_kind": "word"},
                  answer={"items": [{"sha": "pic-1"}]})
        db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(None, "pic-1", need.subject, "picture-for-word"),
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


def test_pending_counts_one_words_two_kinds_as_two_needs(db, monkeypatch):
    """`pending` counts needs, not subjects: one word whose picture and
    whose recording both land a question in the same batch is two pending
    needs, so the identity still holds."""
    assessor = _Assessor()
    _patch(monkeypatch, {
        ("a", "openverse"): AttemptResult(True, questions=[_Q("a", "picture")]),
        ("a", "forvo"): AttemptResult(True, questions=[_Q("a", "recording")])})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",), recordings=("a",))), assessor), {})
    assert report.available == 2 and report.pending == 2
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_run_that_collected_nothing_submits_no_batch(db, monkeypatch):
    assessor = _Assessor()
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert assessor.submitted == [] and report.batch_id is None and report.pending == 0


class _Stuck(_Assessor):
    """A batch still in progress: resolve() releases nothing and the same
    batch is still out afterwards."""

    def resolve(self, batch_id):
        self.resolved.append(batch_id)
        return {}


class _DeadResolve(_Assessor):
    """The judge's wire is down at the resolve: the batch stays out."""

    def resolve(self, batch_id):
        raise JudgeUnreachable("batch status failed")


def _exhaust(db, subject):
    """Every picture Source asked for `subject` and none of them offering
    anything: queued() counts the need out of options. The "nothing"
    outcome row is what next_source/exhausted fold over (spec 3 section
    6); the provide row alongside it is the ask itself, on the record
    like any real attempt's.
    """
    for source in ("openverse", "wikimedia", "pexels"):
        db.append(port="provide", backend=source,
                  key=ProvideKey(source=source, kind="", query=subject), subject=subject,
                  question={"kind": "picture", "subject_kind": "word"},
                  answer={"items": []})
        db.append(port="attempt", backend=source,
                  key=AttemptOutcomeKey(subject=subject, kind="picture", source=source),
                  subject=subject,
                  question={"kind": "picture", "subject_kind": "word", "source": source},
                  answer={"outcome": "nothing", "candidates": []})


def test_run_resolves_the_previous_batch_and_counts_one_still_out_as_pending(db, monkeypatch):
    assessor = _Stuck(outstanding=("batch-0", frozenset({("a", "picture"), ("b", "picture")})))
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b", "c"))), assessor), {})
    assert assessor.resolved == ["batch-0"] and report.pending == 2
    assert report.batch_id == "batch-0"
    assert calls == [] and assessor.submitted == [] and report.attempted == 0
    # The run ended here, before it ever looked at anything: "a"/"b" are
    # pending (still out in the batch), "c" is deferred (never even that).
    assert report.deferred == 1
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_dead_resolve_counts_the_exhausted_and_unserved_needs_once(db, monkeypatch):
    """The judge died at the resolve, with an exhausted picture need and
    an unserved grapheme need in the same syllabus: `pending` is the
    outstanding batch's own needs, and the leftover is deferred once --
    never the exhausted and unserved needs a second time.
    """
    assessor = _DeadResolve(outstanding=("batch-0", frozenset({("a", "picture")})))
    _patch(monkeypatch, {})
    _exhaust(db, "b")
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b", "c"), graphemes=("g",))), assessor), {})
    assert report.available == 4 and report.pending == 1
    assert report.exhausted == 1 and report.unserved == 1 and report.deferred == 1
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_batch_still_out_after_the_resolve_counts_each_need_once(db, monkeypatch):
    """The same accounting where the resolve reached a batch that has not
    ended: the run looked at no need, and the exhausted and unserved ones
    are not deferred on top of their own buckets."""
    assessor = _Stuck(outstanding=("batch-0", frozenset({("a", "picture")})))
    _patch(monkeypatch, {})
    _exhaust(db, "b")
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b", "c"), graphemes=("g",))), assessor), {})
    assert report.batch_id == "batch-0" and report.available == 4 and report.pending == 1
    assert report.exhausted == 1 and report.unserved == 1 and report.deferred == 1
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_pending_at_a_dead_resolve_counts_a_words_two_kinds_as_two_needs(db, monkeypatch):
    """The outstanding batch names one word's picture and its recording:
    two pending needs on this path too, the measure the submitted path
    uses."""
    assessor = _DeadResolve(
        outstanding=("batch-0", frozenset({("a", "picture"), ("a", "recording")})))
    _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",), recordings=("a",))), assessor), {})
    assert report.available == 2 and report.pending == 2 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_run_ending_before_any_attempt_refuses_buckets_over_available():
    """A `deferred` below zero is a defect in the counts, never something
    to clamp: the fold names every count it was given and refuses."""
    needs = run_mod.QueuedNeeds(entries=[], available=2, exhausted=1, unserved=1)
    with pytest.raises(ValueError) as raised:
        run_mod._unconsidered(needs, pending=1)
    message = str(raised.value)
    assert "available=2" in message and "pending=1" in message
    assert "exhausted=1" in message and "unserved=1" in message


def test_run_collects_the_preference_questions_of_a_resolved_batch(db, monkeypatch):
    """A resolved batch's preference question is submitted and counted
    under `preferences`, not `pending`: the picture it ranks already has
    its need satisfied (`_Gaps()` here names no gap at all), so it never
    was a member of `available`."""
    assessor = _Assessor(outstanding=("batch-0", frozenset({("a", "picture")})))
    _patch(monkeypatch, {}, preference=AttemptResult(True, questions=[_Q("a")]))
    report = run(_ctx(db, _Syl(_Gaps()), assessor), {})
    assert len(assessor.submitted[0]) == 1
    assert report.pending == 0 and report.preferences == 1


class _TargetedSyl(_Syl):
    """A syllabus whose open Targets name the word they belong to: two
    Targets on one word are one (word, "sentence") need."""

    def __init__(self, targets):
        super().__init__(_Gaps(sentences=tuple(t.id for t in targets)))
        self.targets = list(targets)


def _one_word_two_targets():
    return (target("w/receptive", "w"), target("w/productive", "w", skill="productive"))


def test_one_words_two_open_targets_are_one_attempted_sentence_need(db, monkeypatch):
    """available_needs counts one (word, "sentence") need per word: a word
    with a receptive and a productive Target still open is available once,
    and the attempt handed both its Targets attempts that one need once --
    never once per Target."""
    _patch(monkeypatch, {}, sentence_result=AttemptResult(
        True, drafted=1, targets_handed=2, subjects_handed=frozenset({"w"})))
    report = run(_ctx(db, _TargetedSyl(_one_word_two_targets())), {})
    assert report.available == 1 and report.attempted == 1 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_one_words_two_open_targets_are_one_budgeted_sentence_need(db, monkeypatch):
    """The same word under a spent llm-sentence budget: one need
    budgeted, not one per Target."""
    calls = []

    def fake_sentence_attempt(ctx, max_targets=40):
        calls.append(1)
        return AttemptResult(True)

    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "adoptable_drafts", lambda cache, syllabus, **kwargs: [])
    midnight = run_mod.day_start_ns(date.today())
    _row_today(db, "llm-sentence", "x", ts=midnight + 1)
    report = run(_ctx(db, _TargetedSyl(_one_word_two_targets())),
                {"llm-sentence": Budget(max_asks=1)})
    assert calls == []
    assert report.available == 1 and report.budgeted == 1 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_unfilled_targets_are_available_work_and_never_exhausted(db, monkeypatch):
    """The sentence attempt serves every open Target once per run: no
    Source is asked per Target, so none of them is ever out of options --
    every open Target within the per-run cap it was handed counts under
    `attempted` instead (the drafts not closing any of them here), and
    the identity
    available == attempted + exhausted + pending + unserved + budgeted + deferred
    still holds.
    """
    _patch(monkeypatch, {},
          sentence_result=AttemptResult(True, drafted=2, targets_handed=3,
                                        subjects_handed=frozenset({"t1", "t2", "t3"})))
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1", "t2", "t3")))), {})
    assert report.available == 3 and report.exhausted == 0 and report.attempted == 3
    assert report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_the_llm_sentence_gate_skipping_the_attempt_leaves_open_targets_budgeted(
        db, monkeypatch):
    """When the llm-sentence budget keeps the sentence attempt from
    running at all, every open Target need within the per-run cap it
    would have been handed is budget-constrained, not silently missing
    from the identity."""
    calls = []

    def fake_sentence_attempt(ctx, max_targets=40):
        calls.append(1)
        return AttemptResult(True, drafted=1, targets_handed=2)

    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "adoptable_drafts", lambda cache, syllabus, **kwargs: [])
    midnight = run_mod.day_start_ns(date.today())
    _row_today(db, "llm-sentence", "x", ts=midnight + 1)
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1", "t2")))),
                {"llm-sentence": Budget(max_asks=1)})
    assert calls == []
    assert report.available == 2 and report.attempted == 0 and report.budgeted == 2
    assert report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_the_sentence_attempts_per_run_cap_defers_the_excess_when_it_runs(db, monkeypatch):
    """45 open Targets, the default 40-per-run cap: the attempt hands over
    only 40 (`attempted`), the other 5 were never even considered this
    run (`deferred`), and the identity still holds."""
    sentences = tuple(f"t{i}" for i in range(45))
    _patch(monkeypatch, {},
          sentence_result=AttemptResult(True, drafted=40, targets_handed=40,
                                        subjects_handed=frozenset(sentences[:40])))
    report = run(_ctx(db, _Syl(_Gaps(sentences=sentences))), {})
    assert report.attempted == 40 and report.deferred == 5
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_the_llm_sentence_gate_counts_every_open_word_budgeted_when_it_skips(
        db, monkeypatch):
    """45 open Targets, the llm-sentence budget already spent: the per-run
    Target cap bounds what one attempt is handed, so with no attempt to
    run it bounds nothing -- all 45 needs are budget-constrained and none
    of them deferred."""
    calls = []

    def fake_sentence_attempt(ctx, max_targets=40):
        calls.append(1)
        return AttemptResult(True, drafted=40, targets_handed=40)

    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "adoptable_drafts", lambda cache, syllabus, **kwargs: [])
    midnight = run_mod.day_start_ns(date.today())
    _row_today(db, "llm-sentence", "x", ts=midnight + 1)
    sentences = tuple(f"t{i}" for i in range(45))
    report = run(_ctx(db, _Syl(_Gaps(sentences=sentences))),
                {"llm-sentence": Budget(max_asks=1)})
    assert calls == []
    assert report.budgeted == 45 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_run_reports_the_drafts_the_sentence_attempt_produced(db, monkeypatch):
    _patch(monkeypatch, {}, sentence_result=AttemptResult(True, drafted=2))
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert report.drafted == 2
    assert db.latest("run", "runreport", RunReportKey()).answer["drafted"] == 2


def _row_today(db, backend, subject, *, ts):
    key = (ProvideKey(source="forvo", kind="", query=subject) if backend == "forvo"
          else LlmPromptKey(producer="sentence-drafter", model="m", prompt_sha=sha(subject)))
    db.append(port="provide", backend=backend, key=key,
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
    remaining need would fail the same way -- and the queued needs the
    loop never reached because of it are `deferred`, never looked at this
    run."""
    calls = _patch(monkeypatch, {("a", "openverse"): JudgeUnreachable})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b", "c")))), {})
    assert [n.subject for n, _s in calls] == ["a"]   # no second need
    assert report.unreachable is True and report.attempted == 1
    assert report.available == 3 and report.deferred == 2
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)
    assert db.latest("run", "runreport", RunReportKey()).answer["unreachable"] is True


def test_an_unreachable_sentence_attempt_stops_the_run_too(db, monkeypatch):
    assessor = _Assessor()
    calls = _patch(monkeypatch, {}, sentence_result=JudgeUnreachable)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert calls == []                               # no need attempted at all
    assert report.unreachable is True
    assert assessor.submitted == []                  # nothing goes out after that


def test_a_judge_that_cannot_be_reached_to_resolve_stops_the_run(db, monkeypatch):
    assessor = _DeadResolve(outstanding=("batch-0", frozenset({("a", "picture")})))
    calls = _patch(monkeypatch, {})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b"))), assessor), {})
    assert report.unreachable is True and calls == [] and assessor.submitted == []
    assert report.batch_id == "batch-0" and report.pending == 1
    assert db.latest("run", "runreport", RunReportKey()).answer["unreachable"] is True
    # "a" is pending (in the batch the dead resolve never released); "b"
    # is deferred -- this run never got far enough to even ask about it.
    assert report.deferred == 1
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


class _DeadSubmit(_Assessor):
    """The judge's wire is down at the submit: nothing this run collected
    goes out."""

    def submit(self, prepared):
        raise JudgeUnreachable("batch submit failed")


def test_a_judge_that_cannot_be_reached_to_submit_stops_the_run(db, monkeypatch):
    assessor = _DeadSubmit()
    _patch(monkeypatch, {("a", "openverse"): AttemptResult(True, questions=[_Q("a")])})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert report.unreachable is True and report.batch_id is None and report.pending == 0
    assert db.latest("run", "runreport", RunReportKey()).answer["unreachable"] is True


def test_an_unsubmitted_preference_question_is_a_preference_not_a_deferred_need(
        db, monkeypatch):
    """The resolve raised a preference question ranking "a"'s picture,
    whose need is already satisfied, and the judge then died at the
    submit: that question is measured against the same `available`
    snapshot the submitted path uses, so it counts under `preferences`,
    outside the identity, rather than deferring a need that is not
    available at all."""
    assessor = _DeadSubmit(outstanding=("batch-0", frozenset({("a", "picture")})))
    _patch(monkeypatch, {}, preference=AttemptResult(True, questions=[_Q("a", "picture")]))
    report = run(_ctx(db, _Syl(_Gaps(recordings=("b",))), assessor), {})
    assert report.unreachable is True and report.preferences == 1
    assert report.available == 1 and report.attempted == 1 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_submitted_question_on_a_satisfied_need_is_a_preference_not_pending(db, monkeypatch):
    """`pending` is the batch's needs that `available` still counts, need
    key for need key: a resolve-time question on "a"'s satisfied picture
    counts under `preferences` even while "a"'s recording need is open
    and pending in the same batch."""
    assessor = _Assessor(outstanding=("batch-0", frozenset({("a", "picture")})))
    _patch(monkeypatch, {("a", "forvo"): AttemptResult(True, questions=[_Q("a", "recording")])},
           preference=AttemptResult(True, questions=[_Q("a", "picture")]))
    report = run(_ctx(db, _Syl(_Gaps(recordings=("a",))), assessor), {})
    assert len(assessor.submitted[0]) == 2
    assert report.pending == 1 and report.preferences == 1
    assert report.available == 1 and report.attempted == 0 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_an_unreachable_sentence_attempt_defers_what_the_resolve_collected(db, monkeypatch):
    """The resolve raised a preference question for "a"'s still-open
    picture need, then the sentence attempt met a dead judge: nothing was
    submitted, so "a" is neither pending nor attempted -- its question is
    collected again next run, which is `deferred`."""
    assessor = _Assessor(outstanding=("batch-0", frozenset({("a", "picture")})))
    calls = _patch(monkeypatch, {}, sentence_result=JudgeUnreachable,
                   preference=AttemptResult(True, questions=[_Q("a", "picture")]))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",))), assessor), {})
    assert calls == [] and assessor.submitted == []
    assert report.unreachable is True and report.available == 1
    assert report.pending == 0 and report.attempted == 0 and report.deferred == 1
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_an_unreachable_judge_in_the_loop_defers_what_the_resolve_collected(db, monkeypatch):
    """Same fold at the loop's own dead judge: "a"'s resolve-time question
    never went out (`deferred`), while "b" (whose own attempt raised a
    question that never went out either) and "c" (whose attempt met the
    dead judge) were both asked (`attempted`)."""
    assessor = _Assessor(outstanding=("batch-0", frozenset({("a", "picture")})))
    calls = _patch(monkeypatch, {("b", "openverse"): AttemptResult(True,
                                                                  questions=[_Q("b")]),
                                 ("c", "openverse"): JudgeUnreachable},
                   preference=AttemptResult(True, questions=[_Q("a", "picture")]))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b", "c"))), assessor), {})
    assert [n.subject for n, _s in calls] == ["b", "c"]   # "a" is never re-attempted
    assert report.unreachable is True and report.available == 3
    assert report.pending == 0 and report.attempted == 2 and report.deferred == 1
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_an_unreachable_sentence_attempt_defers_the_queued_needs_it_never_reached(
        db, monkeypatch):
    """The judge died inside the sentence attempt, before the loop ever
    ran: the two queued needs behind it were never looked at, so they are
    deferred rather than missing from the account."""
    calls = _patch(monkeypatch, {}, sentence_result=JudgeUnreachable)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b"), sentences=("t1",)))), {})
    assert calls == [] and report.unreachable is True
    assert report.available == 3 and report.attempted == 1 and report.deferred == 2
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_an_unreachable_sentence_attempt_attempts_every_open_word_uncapped(db, monkeypatch):
    """45 open words against the default 40-per-run Target cap: the cap
    bounds the Targets one attempt is handed, never the needs, and the
    drafts for all 45 were asked for before the judge died."""
    sentences = tuple(f"t{i}" for i in range(45))
    _patch(monkeypatch, {}, sentence_result=JudgeUnreachable)
    report = run(_ctx(db, _Syl(_Gaps(sentences=sentences))), {})
    assert report.available == 45 and report.attempted == 45 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_dead_source_defers_the_needs_it_left_untouched(db, monkeypatch):
    """The first of three needs on one Source failed on the wire and the
    Source is skipped for the rest of the run: none of the three wrote a
    row, so all three are deferred and retried next run, while the need
    on a live Source is attempted."""
    calls = _patch(monkeypatch, {("a", "openverse"): TransportError})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b", "c"), recordings=("r",)))), {})
    assert [n.subject for n, _s in calls] == ["a", "r"]
    assert report.source_failures == {"openverse": 1}
    assert report.available == 4 and report.attempted == 1 and report.deferred == 3
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_dead_source_is_counted_and_skipped_for_the_rest_of_the_run(db, monkeypatch):
    calls = _patch(monkeypatch, {("a", "openverse"): TransportError})
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a", "b")))), {})
    assert [n.subject for n, _s in calls] == ["a"]   # b's next source is the dead one
    assert report.source_failures == {"openverse": 1}
    assert report.unreachable is False
    assert db.latest("run", "runreport", RunReportKey()).answer["source_failures"] == {
        "openverse": 1}


def test_a_drafter_transport_failure_is_a_source_failure_and_the_loop_runs(db, monkeypatch):
    """The drafter died on the wire: its failure is counted, every word
    with an open Target is deferred, and the queued picture need is
    still attempted."""
    calls = _patch(monkeypatch, {}, sentence_result=TransportError)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",), sentences=("t1", "t2")))), {})
    assert [n.subject for n, _s in calls] == ["a"]
    assert report.source_failures == {"llm-sentence": 1}
    assert report.unreachable is False
    assert report.available == 3 and report.attempted == 1 and report.deferred == 2
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)
    assert db.latest("run", "runreport", RunReportKey()).answer["source_failures"] == {
        "llm-sentence": 1}


def test_a_drafter_transport_failure_defers_every_open_word_beyond_the_cap_too(db, monkeypatch):
    sentences = tuple(f"t{i}" for i in range(45))
    _patch(monkeypatch, {}, sentence_result=TransportError)
    report = run(_ctx(db, _Syl(_Gaps(sentences=sentences))), {})
    assert report.available == 45 and report.attempted == 0 and report.deferred == 45
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


# --- adoption: the cover over what the judge passed ------------------------

class _AdoptingSyl:
    """Stands in for the real Syllabus's adoption effect: cover() takes the
    verified drafts whose Targets are still open, and with_sentences()
    closes those Targets."""

    def __init__(self, unfilled_targets):
        self.targets, self.sentences, self.pairs = [], (), ()
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
    _patch(monkeypatch, {}, drafts=[
        (sentence(((WordId("x"),),), thai_of(word("x", "x")), gloss="ex"), ("t1",))])

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
    answer = db.latest("run", "runreport", RunReportKey()).answer
    assert set(answer) == {"attempted", "improved", "exhausted", "available", "pending",
                           "sentences_adopted", "drafted", "excluded", "excluded_items",
                           "unreachable", "batch_id", "source_failures", "spend", "unserved",
                           "budgeted", "deferred", "preferences"}


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
