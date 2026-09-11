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
import time
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image as PILImage

from thai_syllabus import run as run_mod
from thai_syllabus.assessor import Excluded, JudgeUnreachable, RawVerdict
from thai_syllabus.cachekeys import (AttemptOutcomeKey, BatchMarkerKey, DirectionKey, JudgeKey,
                                    LearnerKey, LlmPromptKey, MechanicalKey, PhraseKey, ProvideKey,
                                    RunReportKey, sha)
from thai_syllabus.attempts import AttemptResult, Sourcing, Spend, sources_for
from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
from thai_syllabus.derivations import (
    available_need_keys,
    available_needs,
    current_best,
    next_source,
    open_words,
)
from thai_syllabus.entities import Category, MinimalPair, SoundConfusion, text_sha
from thai_syllabus.ids import ConfusionId, PairId, WordId
from thai_syllabus.profile import Profile
from thai_syllabus.provider import FetchBackend, LlmBackend, RawAnswer, TtsBackend
from thai_syllabus.record import drafted_phrase, parse_phrases, rows_for
from thai_syllabus.safety import Guard
from thai_syllabus.tts import pick_voice
from thai_syllabus.run import (
    FORVO_DEFAULT_DAILY_BUDGET,
    LEARNER_DEFAULT_SESSION_BUDGET,
    Budget,
    run,
)
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.transport import Completion, QuotaExhausted, TransportError
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


class _LlmPhrase:
    """The phrase drafter: `phrases` is the JSON its one answer carries --
    empty by default, so a picture need still searches the gloss fallback
    unless a test asks for drafted phrases specifically."""

    def __init__(self, phrases: str = '{"phrases": []}'):
        self.phrases = phrases
        self.prompts: list[str] = []

    def cache_key(self, q):
        return LlmPromptKey(producer="phrase-drafter", model="m",
                            prompt_sha=sha(q.params["prompt"]))

    def fetch(self, q):
        self.prompts.append(q.params["prompt"])
        return RawAnswer(items=(self.phrases,))


class _FixedCompletionTransport:
    """A drafter transport (LlmBackend's own `.complete(prompt)`
    contract) answering one fixed completion; records every prompt it
    was asked."""

    def __init__(self, text):
        self.text, self.prompts = text, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return Completion(text=self.text)


def _real_llm_phrase(text: str) -> LlmBackend:
    """The production llm-phrase wiring (wiring.build_provider's own
    recognizer for producer "phrase-drafter", fix round 2 finding 2):
    recognizes only a completion naming at least one asked item's
    phrase."""
    return LlmBackend(producer="phrase-drafter", model="m", transport=_FixedCompletionTransport(text),
                      recognize=lambda t: bool(parse_phrases(t)))


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


def _wire(ctx, fake_search, *, llm=None, batch=None, complete=None, phrase=None):
    """Replaces every backend that would touch the network."""
    ctx.provider._backends.update({
        "openverse": fake_search.backend("openverse"),
        "wikimedia": fake_search.backend("wikimedia"),
        "pexels": fake_search.backend("pexels"),
        "forvo": _Silent("forvo"), "tts": _Silent("tts"),
        "llm-sentence": llm if llm is not None else _Llm(),
        "llm-phrase": phrase if phrase is not None else _LlmPhrase(),
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


def _seed_no_fit(db, word_id, target_id, *, times):
    """`times` no-fit outcome rows on a word's sentence need, in the shape
    attempts.sentence_attempt appends them (spec 3 r19 section 5)."""
    for _ in range(times):
        db.append(port="attempt", backend="llm",
                  key=AttemptOutcomeKey(subject=word_id, kind="sentence", source="llm"),
                  subject=word_id,
                  question={"kind": "sentence", "source": "llm", "subject_kind": "word",
                           "targets": [target_id]},
                  answer={"outcome": "nothing", "candidates": [], "reason": "nothing fits"})


# --- the phrase attempt: one drafted search phrase per open picture need ---
# (spec 3 r24 section 5) ------------------------------------------------

def test_run_drafts_a_phrase_for_every_open_picture_need_and_searches_it(
        ctx_batch_two_needs, fake_search, fake_batch):
    llm_phrase = _LlmPhrase(json.dumps({"phrases": [
        {"subject": "rice", "phrase": "a bowl of steamed rice"},
        {"subject": "fish", "phrase": "a grilled whole fish"}]}))
    ctx_batch_two_needs.provider._backends["llm-phrase"] = llm_phrase
    report = run(ctx_batch_two_needs, budgets={})
    assert len(llm_phrase.prompts) == 1        # one ask covers both open needs
    assert report.spend["llm-phrase"].asks == 1

    def query_of(subject):
        return [r.question["params"]["query"] for r in rows_for(ctx_batch_two_needs.db, subject,
                                                                 "picture")
               if r.port == "provide" and r.backend == "openverse"]

    assert query_of("rice") == ["a bowl of steamed rice"]
    assert query_of("fish") == ["a grilled whole fish"]


def test_run_skips_the_phrase_ask_when_every_picture_need_already_has_one(
        ctx_batch_two_needs, fake_search, fake_batch):
    for subject in ("rice", "fish"):
        ctx_batch_two_needs.db.append(
            port="provide", backend="llm", key=PhraseKey(subject=subject), subject=subject,
            question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
            answer={"phrase": "already drafted"})
    report = run(ctx_batch_two_needs, budgets={})
    assert ctx_batch_two_needs.provider._backends["llm-phrase"].prompts == []
    assert "llm-phrase" not in report.spend


def test_run_treats_an_empty_phrases_answer_as_unrecognized_and_re_asks_next_run(
        ctx_batch_two_needs, fake_search, fake_batch):
    """Fix round 2 finding 2 (spec 3 section 2): an answer phrasing none
    of the asked items is not a cacheable answer -- the real llm-phrase
    wiring's own recognizer (wiring.build_provider) raises, so
    phrase_attempt appends no per-subject row and the run counts a
    source failure instead of a permanent empty cache hit for a stable
    lacking set. Uses the real LlmBackend/recognize path (not the bare
    _LlmPhrase fake), so this exercises the actual production contract."""
    backend = _real_llm_phrase(json.dumps({"phrases": []}))
    ctx_batch_two_needs.provider._backends["llm-phrase"] = backend
    report = run(ctx_batch_two_needs, budgets={})
    assert len(backend.transport.prompts) == 1
    assert report.source_failures == {"llm-phrase": 1}
    assert report.unreachable is False
    assert drafted_phrase(ctx_batch_two_needs.db.assessments_of("rice")) is None
    assert drafted_phrase(ctx_batch_two_needs.db.assessments_of("fish")) is None
    # nothing cached -- a second run (its own batch resolved first, same
    # as every other two-run test in this module) re-asks rather than
    # getting a permanent empty cache hit for the same stable lacking set
    fake_batch.complete_all(report.batch_id, passed=False)
    run(ctx_batch_two_needs, budgets={})
    assert len(backend.transport.prompts) == 2


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


def test_a_word_with_three_no_fit_rows_is_reported_exhausted_not_attempted(
        tmp_path, fake_search, fake_batch):
    """End to end over the real Sourcing: three no-fit answers on record
    and the drafter is not handed rice's Target again -- the run reports
    the word under `exhausted`, and the identity over its three needs
    (picture, recording, sentence) holds."""
    root = _deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch,
                llm=_Llm(json.dumps({"sentences": [], "reason": "nothing fits"})))
    for name in ("openverse", "wikimedia", "pexels"):
        ctx.provider._backends[name] = _Silent(name)
    _seed_no_fit(ctx.db, "rice", "rice/receptive", times=3)

    report = run(ctx, budgets={})
    assert report.available == 3          # rice's picture, recording and sentence needs
    assert report.exhausted == 1 and report.attempted == 2 and report.deferred == 0
    assert report.drafted == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)
    # The drafter was never asked, so no fourth no-fit row landed.
    assert len([r for r in rows_for(ctx.db, "rice", "sentence") if r.port == "attempt"]) == 3


def test_a_no_fit_answer_leaves_one_nothing_row_per_handed_word(
        tmp_path, fake_search, fake_batch):
    root = _deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch,
                llm=_Llm(json.dumps({"sentences": [], "reason": "no natural sentence"})))
    for name in ("openverse", "wikimedia", "pexels"):
        ctx.provider._backends[name] = _Silent(name)

    report = run(ctx, budgets={})
    rows = [r for r in rows_for(ctx.db, "rice", "sentence") if r.port == "attempt"]
    assert len(rows) == 1 and rows[0].answer["reason"] == "no natural sentence"
    assert report.attempted == 3 and report.exhausted == 0   # the word was handed over
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


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
        return MechanicalKey(check="duration", params="0.2-5.0", subject=q.subject,
                             artifact_sha=q.artifact_sha or "-")

    def fetch(self, q):
        return RawVerdict(value=True, evidence="ok")


class _FailingMechanical:
    """Fails every recording it is asked about -- stands in for ffprobe
    rejecting a candidate over the real recording attempt pipeline."""
    def cache_key(self, q):
        return MechanicalKey(check="duration", params="0.2-5.0", subject=q.subject,
                             artifact_sha=q.artifact_sha or "-")

    def fetch(self, q):
        return RawVerdict(value=False, evidence="too long")


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


def test_a_recording_needs_unjudged_candidate_is_assessed_before_any_source_over_a_real_run(
        tmp_path):
    """spec 3 r23 section 5: assess-first now covers recording needs too
    (attempts._assess_recordings). A migrated candidate (spec 2 section 4
    r7's legacy-current provide row) on record with no mechanical verdict
    is checked before forvo/tts is ever asked; the run counts the need
    `attempted` (assess-first's own AttemptResult.attempted, resolved
    inline -- the same run.py accounting an inline picture assess-first
    result already uses, run._try_each_need) and `improved` once
    current_best lands on it.
    """
    root = _deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = build_sourcing(root)
    forvo, tts = _EmptyForvo(), _EmptyTts()
    ctx.provider._backends.update({
        "openverse": _Silent("openverse"), "wikimedia": _Silent("wikimedia"),
        "pexels": _Silent("pexels"), "forvo": forvo, "tts": tts,
        "llm-sentence": _Llm(),
    })
    ctx.assessor._backends["mechanical"] = _PassingMechanical()

    candidate_sha = "c" * 64
    ctx.db.append(port="provide", backend="legacy-current",
                 key=ProvideKey(source="legacy-current", kind="recording", query="rice"),
                 subject="rice",
                 question={"provides": "recording", "kind": "recording", "subject_kind": "word",
                          "params": {"audio": "audio/rice.mp3"}},
                 answer={"items": [{"sha": candidate_sha, "ext": "mp3"}]})

    before = current_best(ctx.db, "rice", "recording", current_rubric={}, prior=(),
                          provenance_source=lambda artifact_sha: None)
    assert before.artifact_sha is None

    report = run(ctx, budgets={})

    assert forvo.calls == 0   # assess-first resolved it; no source was ever asked
    assert not [r for r in rows_for(ctx.db, "rice", "recording") if r.port == "attempt"]
    best = current_best(ctx.db, "rice", "recording", current_rubric={}, prior=(),
                        provenance_source=lambda artifact_sha: None)
    assert best.artifact_sha == candidate_sha and best.source == "mechanical"
    assert report.improved == 1
    assert report.exhausted == 0 and report.pending == 0 and report.budgeted == 0
    assert report.deferred == 0 and report.unserved == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


# --- the report, one field at a time ---------------------------------------

@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class _Gaps:
    def __init__(self, pictures=(), recordings=(), sentences=(), graphemes=(),
                sentence_recordings=()):
        self.words_missing_pictures, self.words_missing_recordings = pictures, recordings
        self.unfilled_targets, self.missing_renditions = sentences, ()
        self.graphemes_missing_keyword_data = graphemes
        self.sentence_recordings, self.scene_pictures = sentence_recordings, ()


@dataclasses.dataclass
class _Syl:
    """A dataclass (not just a plain class) so run._retire_exhausted_sentence's
    own `dataclasses.replace(ctx.syllabus, sentences=...)` (F13 fix round 1)
    works over this fake the same way it does over the real Syllabus."""
    _gaps: object
    targets: list = dataclasses.field(default_factory=list)
    sentences: tuple = ()
    pairs: tuple = ()

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
           preference=AttemptResult(attempted=False), assess=None,
           phrase_result=AttemptResult(attempted=False)):
    """Replaces attempt/assess_first/sentence_attempt/phrase_attempt/
    preference_attempt/adoptable_drafts. A `results` entry that is an
    exception class is raised instead of returned. `assess` is
    assess_first's fixed return for every need -- None keeps the
    fall-through to the source."""
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

    def fake_phrase_attempt(ctx):
        if isinstance(phrase_result, type) and issubclass(phrase_result, Exception):
            raise phrase_result("no drafter")
        return phrase_result

    monkeypatch.setattr(run_mod, "attempt", fake_attempt)
    monkeypatch.setattr(run_mod, "assess_first", lambda ctx, need: assess)
    monkeypatch.setattr(run_mod, "sentence_attempt", fake_sentence_attempt)
    monkeypatch.setattr(run_mod, "phrase_attempt", fake_phrase_attempt)
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
    midnight = run_mod.day_start_ns(datetime.now().astimezone(), None)
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


def test_assess_first_exclusions_reach_the_report_on_fall_through(db, monkeypatch):
    """When assess-first's own result is unattempted (every awaiting
    candidate excluded, attempts.assess_first), the run still asks a
    source but must not drop the exclusion on the floor: it lands in
    RunReport.excluded (spec 3 section 7) and the accounting identity
    still holds.
    """
    calls = _patch(monkeypatch, {}, assess=AttemptResult(
        attempted=False, excluded={"k": Excluded(subject="a", artifact_sha="s", reason="gone")}))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert [(n.subject, s) for n, s in calls] == [("a", "openverse")]
    assert report.excluded == 1
    assert report.available == (report.attempted + report.exhausted + report.pending
                                + report.unserved + report.budgeted + report.deferred)


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


def test_run_counts_a_word_the_sentence_attempt_withheld_under_exhausted(db, monkeypatch):
    """Spec 3 r19 section 5: a word at the no-fit cap is not handed to the
    drafter, and is neither attempted nor deferred -- it is `exhausted`,
    once, and the run's accounting identity still holds."""
    withheld = AttemptResult(attempted=False, subjects_exhausted=frozenset({"t1"}))
    _patch(monkeypatch, {}, sentence_result=withheld)
    report = run(_ctx(db, _Syl(_Gaps(sentences=("t1",)))), {})
    assert report.available == 1
    assert report.exhausted == 1 and report.attempted == 0 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_run_reads_its_own_clock_once_and_re_offers_forvo_after_its_nothing_aged_out(db, monkeypatch):
    """spec 3 r19 section 6a/9 threading guard: run() reads the clock once
    and hands it with the Sourcing's nothing_ttl to every fold, so a forvo
    `nothing` row 200 days old (ttl 180) leaves forvo the next source
    while a tts `nothing` row keeps tts tried."""
    calls = _patch(monkeypatch, {})
    day = 86_400 * 1_000_000_000
    for source in ("forvo", "tts"):
        db.append(port="attempt", backend=source,
                  key=AttemptOutcomeKey(subject="a", kind="recording", source=source),
                  subject="a", question={"kind": "recording", "subject_kind": "word",
                                         "source": source},
                  answer={"outcome": "nothing", "candidates": []}, cost=0.0,
                  ts=time.time_ns() - 200 * day)
    ctx = _ctx(db, _Syl(_Gaps(recordings=("a",))))
    ctx.nothing_ttl = {"forvo": 180}
    report = run(ctx, {})
    assert [(n.subject, s) for n, s in calls] == [("a", "forvo")]
    assert report.exhausted == 0
    ctx_no_ttl = _ctx(db, _Syl(_Gaps(recordings=("a",))))
    calls.clear()
    report = run(ctx_no_ttl, {})
    assert calls == [] and report.exhausted == 1


def test_run_threads_its_own_clock_read_to_every_attempt_via_ctx_now_ns(db, monkeypatch):
    """spec 3 r19 section 6a/9: run() reads the clock once and threads it
    through ctx.now_ns for the pass's duration (restoring the original
    callable once the pass ends), so every attempt sees the same instant
    instead of each re-reading the clock."""
    seen = []

    def fake_attempt(ctx, need, source):
        seen.append(ctx.now_ns())
        return AttemptResult(True, spend={source: Spend(1, 0.0)})

    monkeypatch.setattr(run_mod, "attempt", fake_attempt)
    monkeypatch.setattr(run_mod, "assess_first", lambda ctx, need: None)
    monkeypatch.setattr(run_mod, "sentence_attempt",
                        lambda ctx, max_targets=40: AttemptResult(attempted=False))
    monkeypatch.setattr(run_mod, "phrase_attempt", lambda ctx: AttemptResult(attempted=False))
    monkeypatch.setattr(run_mod, "preference_attempt",
                        lambda ctx, subjects: AttemptResult(attempted=False))
    monkeypatch.setattr(run_mod, "adoptable_drafts", lambda cache, syllabus, **kwargs: [])
    monkeypatch.setattr(run_mod, "time_ns", lambda: 12345)
    ctx = _ctx(db, _Syl(_Gaps(pictures=("a", "b"))))
    original_now_ns = ctx.now_ns
    run(ctx, {})
    assert seen == [12345, 12345]
    assert ctx.now_ns is original_now_ns   # restored once the pass ends


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


def test_run_counts_the_same_excluded_candidate_once_across_assess_first_and_the_attempt(
        db, monkeypatch):
    """Important fix: on fall-through, assess-first's own exclusion
    (collected before the source is asked) and the attempt's exclusion
    for the same candidate (e.g. _judge_pictures re-asking the fit
    question for every candidate on record, not only the freshly fetched
    ones) name the same (subject, artifact_sha). RunReport.excluded and
    excluded_items must agree and count it once, not twice.
    """
    same_exclusion = {"k": Excluded(subject="a", artifact_sha="s", reason="gone")}
    calls = _patch(monkeypatch, {
        ("a", "openverse"): AttemptResult(True, excluded=same_exclusion)},
        assess=AttemptResult(attempted=False, excluded=same_exclusion))
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert [(n.subject, s) for n, s in calls] == [("a", "openverse")]
    assert report.excluded == 1
    assert report.excluded_items == ({"subject": "a", "artifact_sha": "s", "reason": "gone"},)


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
    midnight = run_mod.day_start_ns(datetime.now().astimezone(), None)
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
    midnight = run_mod.day_start_ns(datetime.now().astimezone(), None)
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
    midnight = run_mod.day_start_ns(datetime.now().astimezone(), None)
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
    midnight = run_mod.day_start_ns(datetime.now().astimezone(), None)
    _row_today(db, "forvo", "x", ts=midnight + 1)
    _row_today(db, "forvo", "y", ts=midnight + 2)
    run(_ctx(db, _Syl(_Gaps(recordings=("a",)))), {"forvo": Budget(max_asks=2)})
    assert calls == []


def test_todays_forvo_downloads_on_record_count_against_its_per_day_budget(db, monkeypatch):
    """A Forvo mp3 download is a Forvo request (spec 3 section 4's forvo
    row): one lookup and one download on today's record spend a budget
    of two."""
    calls = _patch(monkeypatch, {})
    midnight = run_mod.day_start_ns(datetime.now().astimezone(), None)
    _row_today(db, "forvo", "x", ts=midnight + 1)
    db.append(port="provide", backend="audiofetch",
              key=ProvideKey(source="", kind="", query="k1"), subject="x",
              question={"kind": "recording", "subject_kind": "word",
                        "params": {"source": "forvo", "url": "https://f/x.mp3"}},
              answer={"items": []}, cost=0.0, ts=midnight + 2)
    run(_ctx(db, _Syl(_Gaps(recordings=("a",)))), {"forvo": Budget(max_asks=2)})
    assert calls == []


def test_yesterdays_asks_do_not_count_against_it(db, monkeypatch):
    calls = _patch(monkeypatch, {})
    midnight = run_mod.day_start_ns(datetime.now().astimezone(), None)
    _row_today(db, "forvo", "x", ts=midnight - 2)
    _row_today(db, "forvo", "y", ts=midnight - 1)
    run(_ctx(db, _Syl(_Gaps(recordings=("a",)))), {"forvo": Budget(max_asks=2)})
    assert [(n.subject, s) for n, s in calls] == [("a", "forvo")]


class _FixedNow(datetime):
    """Stands in for run.py's module-level `datetime` name so
    `_spent_today` reads a fixed, zone-aware instant instead of the real
    wall clock."""
    _fixed = datetime(2026, 9, 9, 4, 0, tzinfo=timezone(timedelta(hours=7)))

    @classmethod
    def now(cls, tz=None):
        return cls._fixed if tz is None else cls._fixed.astimezone(tz)


def test_spent_today_sums_each_budget_from_its_own_day_starts(db, monkeypatch):
    """A row logged after forvo's own 22:00Z reset but before local
    midnight the next ICT day counts against forvo's own budget and not
    against a budget with no `day_starts` of its own -- the two windows
    differ, and `_spent_today` reads each budget's own.
    """
    monkeypatch.setattr(run_mod, "datetime", _FixedNow)
    now = _FixedNow.now()
    forvo_reset = run_mod.day_start_ns(now, "22:00Z")
    local_midnight = run_mod.day_start_ns(now, None)
    assert forvo_reset < local_midnight
    _row_today(db, "forvo", "x", ts=forvo_reset + 1)
    _row_today(db, "other", "x", ts=forvo_reset + 1)
    spent = run_mod._spent_today(
        _ctx(db, _Syl(_Gaps())),
        {"forvo": Budget(max_asks=1, day_starts="22:00Z"), "other": Budget(max_asks=1)})
    assert spent["forvo"].asks == 1     # forvo_reset <= its own ts: counts
    assert spent["other"].asks == 0     # its ts is before local midnight: does not


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


def test_a_quota_exhausted_source_budgets_the_need_and_every_later_one_on_it(db, monkeypatch):
    """Forvo's own quota statement (spec 3 section 6a) is not a source
    failure: the need that hit it counts budgeted, not deferred, no
    source_failures entry is recorded, and a later need whose next
    source is the same one is skipped as budgeted too, without ever
    calling attempt() for it -- the source is budgeted for the rest of
    the run, the same bucket a spent day budget uses."""
    calls = _patch(monkeypatch, {("a", "forvo"): QuotaExhausted})
    report = run(_ctx(db, _Syl(_Gaps(recordings=("a", "b")))), {})
    assert [n.subject for n, _s in calls] == ["a"]   # "b" never reaches attempt()
    assert report.budgeted == 2 and report.deferred == 0
    assert report.source_failures == {}
    assert report.available == 2 and report.attempted == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)
    assert db.latest("run", "runreport", RunReportKey()).answer["source_failures"] == {}


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


def test_a_phrase_drafter_transport_failure_is_a_source_failure_and_the_loop_runs(db, monkeypatch):
    """spec 3 r24 section 5: the phrase drafter died on the wire -- its
    failure is counted, but unlike the sentence drafter it owns no need
    bucket of its own, so the queued picture need is still attempted
    normally (on its plain gloss fallback) and nothing is deferred."""
    calls = _patch(monkeypatch, {}, phrase_result=TransportError)
    report = run(_ctx(db, _Syl(_Gaps(pictures=("a",)))), {})
    assert [n.subject for n, _s in calls] == ["a"]
    assert report.source_failures == {"llm-phrase": 1}
    assert report.unreachable is False
    assert report.available == 1 and report.attempted == 1 and report.deferred == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)
    assert db.latest("run", "runreport", RunReportKey()).answer["source_failures"] == {
        "llm-phrase": 1}


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


def test_forvo_default_daily_budget_resets_at_22_00_utc():
    assert FORVO_DEFAULT_DAILY_BUDGET.day_starts == "22:00Z"


def test_learner_default_session_budget_is_20_questions():
    assert LEARNER_DEFAULT_SESSION_BUDGET.max_asks == 20


# --- F13: the run retires an adopted sentence whose recording is
# exhausted (spec 3 section 5) ----------------------------------------------

_RETIRED_TEXT = "กินข้าว"
_RETIRED_SHA = text_sha(_RETIRED_TEXT)
_RETIRED_CANDIDATE = "c" * 64


def _seed_retired_sentence_row(db):
    db.add_sentence(text_sha=_RETIRED_SHA, text=_RETIRED_TEXT, clauses=((WordId("eat"),),),
                    gloss="eat rice", voice="learner_voice", source="llm", origin="m",
                    licence="generated", acquired=datetime.now().date())


def _seed_forvo_nothing(db):
    """forvo's own attempt, an earlier run's: nothing usable (the shape
    attempts._recording_attempt's own outcome row leaves on record)."""
    db.append(port="attempt", backend="forvo",
             key=AttemptOutcomeKey(subject=_RETIRED_SHA, kind="recording", source="forvo"),
             subject=_RETIRED_SHA,
             question={"kind": "recording", "subject_kind": "sentence", "source": "forvo"},
             answer={"outcome": "nothing", "candidates": [], "tried": []})


def _seed_tts_candidate(db, *, passed: bool):
    """tts's own outcome row and mechanical verdict -- the shape a real
    attempt(ctx, need, "tts") leaves on record, minus the media/speaker
    rows this test never reads."""
    db.append(port="attempt", backend="tts",
             key=AttemptOutcomeKey(subject=_RETIRED_SHA, kind="recording", source="tts"),
             subject=_RETIRED_SHA,
             question={"kind": "recording", "subject_kind": "sentence", "source": "tts"},
             answer={"outcome": "candidates", "candidates": [_RETIRED_CANDIDATE], "tried": []})
    db.append(port="assess", backend="mechanical",
             key=MechanicalKey(check="duration", params="0.2-5.0", subject=_RETIRED_SHA,
                               artifact_sha=_RETIRED_CANDIDATE),
             subject=_RETIRED_SHA,
             question={"role": "recording-for-sentence", "artifact_sha": _RETIRED_CANDIDATE,
                      "rubric": None, "kind": "recording", "subject_kind": "sentence",
                      "params": {}},
             answer={"value": passed, "evidence": "ok" if passed else "too long"})


def _fake_tts_attempt(db, *, passed: bool):
    """Stands in for attempts.attempt(): when this run's own next_source
    picks tts (forvo already tried, seeded nothing), writes the same
    outcome/verdict rows a real attempt would have, inline -- the run's
    own post-attempt check (run._maybe_retire_exhausted_sentence) then
    reads them back for real, over the real db.
    """
    def fake_attempt(ctx, need, source):
        if source == "tts":
            _seed_tts_candidate(db, passed=passed)
        return AttemptResult(attempted=True)
    return fake_attempt


def _retirable_ctx(db):
    ctx = _ctx(db, _Syl(_Gaps(sentence_recordings=(_RETIRED_SHA,))))
    ctx.guard = Guard()
    return ctx


def test_run_retires_an_adopted_sentence_whose_recording_is_exhausted(db, monkeypatch):
    """forvo already answered nothing (an earlier run's own attempt,
    seeded directly); this run's own tts attempt stores the one
    remaining candidate and it fails mechanical -- next_source is None
    afterwards, current_best is None, and no learner row outlives the
    rule: the run deletes the sentence, reports it retired, and the
    writing command's Guard sees the removal.
    """
    _seed_retired_sentence_row(db)
    _seed_forvo_nothing(db)
    _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "attempt", _fake_tts_attempt(db, passed=False))
    ctx = _retirable_ctx(db)

    report = run(ctx, {})

    assert report.retired == 1
    assert _RETIRED_SHA not in {s.text_sha for s in db.all_sentences()}
    assert ctx.guard.removals == {"sentences": 1}
    assert report.attempted == 1 and report.exhausted == 0
    assert (report.available == report.attempted + report.exhausted + report.pending
           + report.unserved + report.budgeted + report.deferred)


def test_a_sentence_with_a_passing_candidate_is_untouched(db, monkeypatch):
    """Every source is now tried, but this run's tts candidate passed
    mechanical: current_best is not None, so F13 never applies."""
    _seed_retired_sentence_row(db)
    _seed_forvo_nothing(db)
    _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "attempt", _fake_tts_attempt(db, passed=True))
    ctx = _retirable_ctx(db)

    report = run(ctx, {})

    assert report.retired == 0
    assert report.improved == 1   # current_best landed on the passing candidate
    assert _RETIRED_SHA in {s.text_sha for s in db.all_sentences()}
    assert ctx.guard.removals == {}


def test_a_sentence_with_a_learner_use_this_row_is_untouched(db, monkeypatch):
    """F9: every source now tried, nothing passing, but a learner row
    nominates a recording (LearnerKey/value "unacceptable-use-this",
    reviewserver.append_supply's own rating shape, role
    recording-for-sentence) -- the sentence outlives the rule change
    instead of being retired.
    """
    _seed_retired_sentence_row(db)
    _seed_forvo_nothing(db)
    db.append(port="assess", backend="learner",
             key=LearnerKey(artifact_sha=_RETIRED_CANDIDATE, role="recording-for-sentence"),
             subject=_RETIRED_SHA,
             question={"role": "recording-for-sentence", "artifact_sha": _RETIRED_CANDIDATE,
                      "rubric": None, "kind": "rating", "subject_kind": "sentence"},
             answer={"value": "unacceptable-use-this"})
    _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "attempt", _fake_tts_attempt(db, passed=False))
    ctx = _retirable_ctx(db)

    report = run(ctx, {})

    assert report.retired == 0
    assert _RETIRED_SHA in {s.text_sha for s in db.all_sentences()}
    assert ctx.guard.removals == {}


def test_a_sentence_with_a_learner_supplied_recording_provide_row_is_untouched(db, monkeypatch):
    """F9's other shape (spec 3 section 5): a provide row the learner
    supplied directly (backend "learner", reviewserver.append_supply's
    local-path shape) also keeps the sentence, on its own -- no rating
    row names it here.
    """
    _seed_retired_sentence_row(db)
    _seed_forvo_nothing(db)
    db.append(port="provide", backend="learner",
             key=ProvideKey(source="learner", kind="", query="supplied.mp3"),
             subject=_RETIRED_SHA,
             question={"provides": "recording-bytes", "kind": "recording",
                      "subject_kind": "sentence", "params": {"path": "supplied.mp3"}},
             answer={"items": [{"sha": "s" * 64, "ext": "mp3"}]})
    _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "attempt", _fake_tts_attempt(db, passed=False))
    ctx = _retirable_ctx(db)

    report = run(ctx, {})

    assert report.retired == 0
    assert _RETIRED_SHA in {s.text_sha for s in db.all_sentences()}
    assert ctx.guard.removals == {}


def test_a_directed_sentence_with_no_passing_candidate_is_untouched(db, monkeypatch):
    """I1 (F9, fix round 2): a directed sentence (derivations.directed --
    here a learner direction row) is the learner's, even exhausted and
    with nothing passing -- the feedback screen still shows it, so it is
    kept, not retired.
    """
    _seed_retired_sentence_row(db)
    _seed_forvo_nothing(db)
    db.append(port="assess", backend="learner",
             key=DirectionKey(subject=_RETIRED_SHA, role="recording-for-sentence",
                              text_sha=sha("supply a recording")),
             subject=_RETIRED_SHA,
             question={"kind": "direction", "role": "recording-for-sentence",
                      "subject_kind": "sentence"},
             answer={"direction": "supply a recording"})
    _patch(monkeypatch, {})
    monkeypatch.setattr(run_mod, "attempt", _fake_tts_attempt(db, passed=False))
    ctx = _retirable_ctx(db)

    report = run(ctx, {})

    assert report.retired == 0
    assert _RETIRED_SHA in {s.text_sha for s in db.all_sentences()}
    assert ctx.guard.removals == {}


# --- F13 fix round 1: a same-pass retirement replaces ctx.syllabus, so a
# later need for the same sentence is skipped, not attempted against
# stale, already-deleted data (build_sourcing builds Sourcing.syllabus once
# per CLI invocation; --cycles reuses the same ctx across every cycle) ------

class _FakeTtsEngine:
    def synthesize(self, text, voice):
        return f"{text}-{voice}".encode()


def test_a_same_pass_retirement_skips_the_sentences_other_still_queued_needs(
        ctx_batch_sentences, fake_batch):
    """The sentence's recording need retires it mid-pass (forvo already
    nothing from r2's own attempt; this run's tts attempt is the one that
    exhausts it and fails mechanical); its scene picture need -- given one
    extra `nothing` outcome directly on record so it sorts after the
    recording need in this run's own queue (derivations.queue's bucket/
    attempts tie-break) -- is still queued when the retirement happens.
    ctx.syllabus is replaced in place (run._retire_exhausted_sentence,
    dataclasses.replace) so the picture need is skipped rather than
    attempted against the now-deleted sentence: no new attempt-outcome row
    under it, and the run still reports retired=1. A later cycle over the
    SAME ctx (run._run_pass directly, standing in for cli._cmd_run's own
    --cycles loop, which never rebuilds Sourcing) lists no needs for it
    either -- the retirement row (record.retired_texts) keeps
    adoptable_drafts from re-adopting the same still-passing draft now
    that its Targets have reopened (C1 fix round 2).
    """
    ctx = ctx_batch_sentences
    ctx.guard = Guard()
    r1 = run(ctx, budgets={})
    fake_batch.complete_all(r1.batch_id, passed=True)
    r2 = run(ctx, budgets={})   # adopts; forvo (recording) and openverse
                                 # (scene picture) each get one silent,
                                 # "nothing" attempt.
    assert r2.sentences_adopted == 1
    sentence_sha = text_sha(EAT_RICE)

    # One more picture attempt directly on record, standing in for a run
    # this test does not otherwise need: the picture need's own attempt
    # count now outranks the recording need's, so it sorts after it in
    # r3's own queue.
    ctx.db.append(port="attempt", backend="wikimedia",
                 key=AttemptOutcomeKey(subject=sentence_sha, kind="picture", source="wikimedia"),
                 subject=sentence_sha,
                 question={"kind": "picture", "subject_kind": "sentence", "source": "wikimedia"},
                 answer={"outcome": "nothing", "candidates": [], "tried": []})

    ctx.provider._backends["tts"] = TtsBackend(
        tts=_FakeTtsEngine(), voices=["m1"], media=ctx.media_store, pick_voice=pick_voice)
    ctx.assessor._backends["mechanical"] = _FailingMechanical()
    picture_attempts_before = len([r for r in rows_for(ctx.db, sentence_sha, "picture")
                                   if r.port == "attempt"])

    r3 = run(ctx, budgets={})

    assert r3.retired == 1
    assert sentence_sha not in {s.text_sha for s in ctx.db.all_sentences()}
    picture_attempts_after = len([r for r in rows_for(ctx.db, sentence_sha, "picture")
                                  if r.port == "attempt"])
    assert picture_attempts_after == picture_attempts_before   # never attempted this pass
    assert not [n for n in available_needs(ctx.syllabus) if n[0] == sentence_sha]
    assert (r3.available == r3.attempted + r3.exhausted + r3.pending
           + r3.unserved + r3.budgeted + r3.deferred)

    # A later cycle over the same ctx (cli._cmd_run's own --cycles: one
    # Sourcing built once, run_pipeline called again): the still-passing
    # draft's Targets reopened, so without the retirement row it would be
    # readopted right back (derivations.adoptable_drafts) -- it is not.
    attempt_rows_before = len([r for r in ctx.db.assessments_of(sentence_sha)
                               if r.port == "attempt"])
    r4 = run_mod._run_pass(ctx, {}, time.time_ns(), sentence_targets_per_run=40)
    assert r4.sentences_adopted == 0
    assert sentence_sha not in {s.text_sha for s in ctx.db.all_sentences()}
    attempt_rows_after = len([r for r in ctx.db.assessments_of(sentence_sha)
                              if r.port == "attempt"])
    assert attempt_rows_after == attempt_rows_before
    assert not [n for n in available_needs(ctx.syllabus) if n[0] == sentence_sha]


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
                           "sentences_adopted", "drafted", "retired", "excluded",
                           "excluded_items", "unreachable", "batch_id", "source_failures",
                           "spend", "unserved", "budgeted", "deferred", "preferences"}


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


def test_a_budget_carries_no_day_starts_by_default():
    assert Budget(max_asks=5).day_starts is None


# --- day_start_ns (spec 3 section 7): a window from the source's own
# reset time, never the machine's local midnight when day_starts names
# another zone --------------------------------------------------------

def test_day_start_ns_at_06_27_ict_with_22_00z_is_05_00_ict_the_same_day():
    ict = timezone(timedelta(hours=7))
    now = datetime(2026, 9, 9, 6, 27, tzinfo=ict)
    since = run_mod.day_start_ns(now, "22:00Z")
    assert since == int(datetime(2026, 9, 9, 5, 0, tzinfo=ict).timestamp() * 1_000_000_000)


def test_day_start_ns_at_04_00_ict_with_22_00z_is_the_previous_days_05_00():
    ict = timezone(timedelta(hours=7))
    now = datetime(2026, 9, 9, 4, 0, tzinfo=ict)
    since = run_mod.day_start_ns(now, "22:00Z")
    assert since == int(datetime(2026, 9, 8, 5, 0, tzinfo=ict).timestamp() * 1_000_000_000)


def test_day_start_ns_with_no_day_starts_is_local_midnight():
    ict = timezone(timedelta(hours=7))
    now = datetime(2026, 9, 9, 6, 27, tzinfo=ict)
    since = run_mod.day_start_ns(now, None)
    assert since == int(datetime(2026, 9, 9, 0, 0, tzinfo=ict).timestamp() * 1_000_000_000)


def test_day_start_ns_accepts_a_numeric_offset_zone():
    ict = timezone(timedelta(hours=7))
    now = datetime(2026, 9, 9, 6, 27, tzinfo=ict)
    since = run_mod.day_start_ns(now, "22:00+00:00")
    assert since == int(datetime(2026, 9, 9, 5, 0, tzinfo=ict).timestamp() * 1_000_000_000)


def test_parse_day_starts_accepts_an_offset_hour_past_nineteen():
    wall, zone = run_mod.parse_day_starts("22:00+23:30")
    assert (wall.hour, wall.minute) == (22, 0)
    assert zone.utcoffset(None) == timedelta(hours=23, minutes=30)


def test_day_start_ns_refuses_an_unparseable_day_starts():
    with pytest.raises(ValueError, match="day_starts"):
        run_mod.day_start_ns(datetime.now().astimezone(), "22")
