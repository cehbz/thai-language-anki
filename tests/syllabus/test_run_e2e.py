"""End to end over a fixture deck: build_sourcing + run with fake search, Forvo, LLM, judge and
mechanical backends; picture, recording and sentence needs close; the gate opens."""
import hashlib
import io
import json
from datetime import date
from pathlib import Path

from PIL import Image as PILImage

from thai_syllabus.assessor import AssessQuestion
from thai_syllabus.attempts import current_best_of
from thai_syllabus.cachekeys import (JudgeKey, LearnerKey, LlmPromptKey, MechanicalKey,
                                    ProvideKey, sha)
from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
from thai_syllabus.entities import Category, text_sha
from thai_syllabus.media import Speaker
from thai_syllabus.provider import FetchBackend, RawAnswer
from thai_syllabus.profile import Profile
from thai_syllabus.run import run
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.transport import Completion
from thai_syllabus.wiring import build_sourcing, load_syllabus

from .builders import sentence, target, thai_of, word
from .test_run import RICE, _Llm as _DraftLlm, _deck as _batch_fixture_deck, _wire, fake_batch, fake_search


def _deck(tmp_path):
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=(word("orange", "ส้ม", "orange"), word("eat", "กิน", "eat")),
        targets=(target("eat/receptive", "eat"), target("orange/receptive", "orange")),
        graphemes=(), confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset({"orange"})),
                   Category(name="Verbs", members=frozenset({"eat"})))))
    (root / "curated" / "frequency_th.txt").write_text("", encoding="utf-8")
    # imgfetch/audiofetch paths and the anthropic secret reference are what
    # load_providers_config now requires of any real providers.yaml; the
    # test replaces both fetch backends (and the judge's transport) with
    # fakes below, so neither binary nor secret is ever touched.
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n"
        "secrets: {anthropic: op://Shared/Anthropic/API Key}\n"
        "judge: {transport: api, model: m, price_per_mtok: {input: 2.0, output: 10.0}}\n",
        encoding="utf-8")
    SyllabusDb(root / "syllabus.db").close()
    MediaStore(root / "media")
    return root


class _Search:
    """Two hits per query -- one image alone can never exercise the
    preference branch (_assess_all_candidates only asks a preference
    question over 2+ passing candidates)."""
    def cache_key(self, q):
        return ProvideKey(source="openverse", kind="", query=q.params["query"])

    def fetch(self, q):
        return RawAnswer(items=(
            {"url": f"https://x/{q.subject}-good.jpg", "source": "openverse", "licence": "by"},
            {"url": f"https://x/{q.subject}-good2.jpg", "source": "openverse", "licence": "by"}))


class _Forvo:
    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        return RawAnswer(items=({"pathmp3": f"https://f/{q.params['word']}.mp3", "username": "kris"},), cost=1.0)


class _Llm:
    def cache_key(self, q):
        return LlmPromptKey(producer="sentence-drafter", model="m", prompt_sha="x")

    def fetch(self, q):
        # กิน = eat, ส้ม = orange, one clause renders กินส้ม with no space
        return RawAnswer(items=(json.dumps({"sentences": [
            {"clauses": [["eat", "orange"]], "text": "กินส้ม", "gloss": "eat orange"}]}),))


def _jpeg_bytes(seed: str) -> bytes:
    """A valid, decodable JPEG whose color derives from `seed` -- distinct
    urls decode to distinct images, unlike a bare url-as-bytes stand-in.
    """
    digest = hashlib.sha256(seed.encode()).digest()
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2), tuple(digest[:3])).save(buf, format="JPEG")
    return buf.getvalue()


def _judge_complete(prompt, attachments=()):
    """The judge's default prompt builders dispatch per role -- fit
    questions attach at most one artifact, a picture-preference question
    attaches every candidate -- so this fakes the same distinction by
    attachment count rather than sniffing prompt text.
    """
    if len(attachments) > 1:
        shas = [Path(p).stem for p in attachments]
        return Completion(text=json.dumps({"ranking": shas}))
    return Completion(text='{"value": true, "evidence": "ok"}')


def test_run_closes_picture_recording_and_sentence_needs(tmp_path):
    root = _deck(tmp_path)
    ctx = build_sourcing(root)
    ctx.provider._backends.update({
        "openverse": _Search(), "forvo": _Forvo(), "llm-sentence": _Llm(),
        "imgfetch": FetchBackend(media=ctx.media_store, fetcher=lambda url: (_jpeg_bytes(url), "jpg")),
        "audiofetch": FetchBackend(media=ctx.media_store, fetcher=lambda url: (url.encode(), "mp3"))})
    ctx.assessor._backends["judge"].complete = _judge_complete
    ctx.assessor._backends["mechanical"].duration_of = lambda path: 1.0

    before = load_syllabus(root).gaps()
    assert before.unfilled_targets and before.words_missing_pictures and before.words_missing_recordings

    report = run(ctx, {})
    assert report.improved >= 4 and report.pending == 0 and report.sentences_adopted == 1

    # sentence/recording-required (F7): Syllabus.gaps() does not enumerate
    # an adopted sentence's own audio need, so this run's queue never
    # attempts it -- seed it directly, the same way its word recordings
    # are seeded (build_sourcing shares ctx.db with load_syllabus's db).
    sentence_sha = text_sha("กินส้ม")  # eat orange
    ctx.db.add_speaker(Speaker(id="somchai", kind="native"))
    ctx.db.add_media(sha="sentence-rec", kind="recording", ext="mp3", source="forvo",
                     origin="https://forvo.com/x", licence="cc-by",
                     acquired=date(2026, 1, 1), speaker_id="somchai")
    ctx.db.append(port="provide", backend="forvo",
                 key=ProvideKey(source="forvo", kind="", query=sentence_sha),
                 subject=sentence_sha, question={"kind": "recording"},
                 answer={"items": [{"sha": "sentence-rec"}]})
    rec_question = AssessQuestion(subject=sentence_sha, role="recording-for-word",
                                  artifact_sha="sentence-rec", kind="recording")
    ctx.db.append(port="assess", backend="judge", key=JudgeKey.for_question(rec_question),
                 subject=sentence_sha,
                 question={"role": "recording-for-word", "artifact_sha": "sentence-rec", "rubric": None,
                          "kind": "recording"},
                 answer={"value": True})

    after = load_syllabus(root)
    g = after.gaps()
    assert g.words_missing_pictures == () and g.words_missing_recordings == () and g.unfilled_targets == ()
    rep = after.report()
    assert "recording/synthetic" not in {f.rule for f in rep.findings}
    assert rep.gate is True

    # two candidate pictures per word -> a preference question ran and its
    # positional bonus lifted the winner above a bare judge pass (50.0).
    assert current_best_of(ctx, "orange", "picture").rank > 50


class _OneHit:
    """One picture-search hit -- unlike the two-hit search fakes elsewhere,
    this never leaves more than one candidate passing fit, so it can never
    trigger a preference question: the fit verdict alone resolves the need
    in one batch round trip."""

    def cache_key(self, q):
        return ProvideKey(source="openverse", kind="", query=q.params["query"])

    def fetch(self, q):
        return RawAnswer(items=({"url": f"https://x/{q.subject}.jpg",
                                 "source": "openverse", "licence": "by"},))


class _Tts:
    """A synthesized recording: writes real bytes to the media store and
    answers with the resulting sha -- the shape provider.TtsBackend.fetch
    answers with for real."""

    def __init__(self, media_store):
        self._media_store = media_store

    def cache_key(self, q):
        return ProvideKey(source="tts", kind=q.params["voice"], query=sha(q.params["text"]))

    def fetch(self, q):
        sha = self._media_store.write(q.params["text"].encode(), "mp3")
        return RawAnswer(items=({"sha": sha, "ext": "mp3"},))


def test_two_runs_over_a_batch_judge_resolve_a_picture_and_escalate_a_recording(
        tmp_path, fake_search, fake_batch):
    """The picture need pends in a judge batch (spec 3 section 7); the
    second run resolves that batch first, and separately escalates the
    still-open recording need to its next Source (forvo silent, tts
    answers), which closes inline under the mechanical authority.
    """
    root = _batch_fixture_deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch)
    ctx.provider._backends["openverse"] = _OneHit()
    ctx.provider._backends["tts"] = _Tts(ctx.media_store)
    ctx.assessor._backends["mechanical"].duration_of = lambda path: 1.0

    r1 = run(ctx, budgets={})
    assert r1.batch_id is not None
    assert r1.pending >= 1
    assert r1.improved == 0

    fake_batch.complete_all(r1.batch_id, passed=True)   # the one candidate fits
    r2 = run(ctx, budgets={})

    assert r2.improved >= 1
    assert r2.pending == 0
    assert current_best_of(ctx, "rice", "picture").artifact_sha is not None

    for report in (r1, r2):
        assert (report.available == report.attempted + report.exhausted + report.pending
               + report.unserved + report.budgeted + report.deferred)


def test_a_resolved_batch_leaving_two_passing_pictures_submits_a_preference_batch(
        tmp_path, fake_search, fake_batch):
    """Fix round 1: a picture whose fit verdicts both pass leaves two
    ranked-nothing candidates; the preference question that ranks them,
    raised while resolving the batch that carried those fits, is counted
    under `preferences`, not `pending` -- the picture's own need is
    already satisfied by the time it is asked. The word "rice" itself is
    out of every other gap by r2, but the sentence it adopts there owes
    its own recording and scene picture (Task F1 items 1-2 make running
    them, in the same run, safe): the recording escalates past its one
    prior forvo candidate to tts, and the scene picture is a fresh
    openverse ask whose fit questions ride r2's own new batch, pending.
    The identity still holds on both reports.
    """
    root = _batch_fixture_deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch,
               llm=_DraftLlm(json.dumps({"sentences": [
                   {"clauses": [["rice"]], "text": "ข้าว", "gloss": "rice"}]})))   # ข้าว: rice
    ctx.provider._backends["forvo"] = _Forvo()
    ctx.assessor._backends["mechanical"].duration_of = lambda path: 1.0

    r1 = run(ctx, budgets={})
    assert r1.batch_id is not None
    assert r1.preferences == 0   # nothing has fit yet -- no ranking to ask for

    # The sentence adopts in r2 (its sentence-for-target verdict resolves
    # there); once adopted it owes its own recording and scene picture
    # (F1 defect 2), and C1/I1 (Task F1 items 1-2) let them run in this
    # same run: a prior forvo candidate is already on record, unrated, so
    # the recording need escalates straight to tts.
    sentence_sha = text_sha("ข้าว")  # rice
    ctx.db.add_speaker(Speaker(id="somchai", kind="native"))
    ctx.db.add_media(sha="rice-sentence-rec", kind="recording", ext="mp3", source="forvo",
                     origin="https://forvo.com/x", licence="cc-by",
                     acquired=date(2026, 1, 1), speaker_id="somchai")
    ctx.db.append(port="provide", backend="forvo",
                 key=ProvideKey(source="forvo", kind="", query="ข้าว"),  # rice: the sentence's own text
                 subject=sentence_sha,
                 question={"kind": "recording", "subject_kind": "sentence"},
                 answer={"items": [{"sha": "rice-sentence-rec"}]})

    fake_batch.complete_all(r1.batch_id, passed=True)   # both pictures fit; the sentence is natural
    r2 = run(ctx, budgets={})

    assert r2.batch_id is not None and r2.batch_id != r1.batch_id
    assert r2.preferences == 1
    assert r2.pending == 1   # the sentence's own scene picture: a fresh need, genuinely pending

    for report in (r1, r2):
        assert (report.available == report.attempted + report.exhausted + report.pending
               + report.unserved + report.budgeted + report.deferred)


def test_a_learner_rejection_with_no_floor_keeps_a_reranked_picture_pending_once(
        tmp_path, fake_search, fake_batch):
    """Fix round 2: a learner "unacceptable-none" on the current picture,
    with no acceptable floor, reopens the need -- the machine's own
    passing candidates are untouched by that rejection. When the batch
    this run resolves lands a third passing candidate, the preference
    question that ranks all three is for a subject still an available
    need: it must land in `pending` exactly once, never also in
    `preferences`, and the loop must not also re-attempt the same
    subject this same run (one bucket per subject).
    """
    root = _batch_fixture_deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch)
    ctx.provider._backends["forvo"] = _Forvo()
    ctx.assessor._backends["mechanical"].duration_of = lambda path: 1.0
    # No LLM/judge round trip needed to close the Target: with_sentences
    # is exactly the effect _adopt_sentences has on ctx.syllabus.
    ctx.syllabus = ctx.syllabus.with_sentences(
        (sentence(((RICE.id,),), thai_of(RICE), gloss="rice"),))   # ข้าว: rice

    # The word's own recording, already settled.
    ctx.db.add_speaker(Speaker(id="somchai", kind="native"))
    ctx.db.add_media(sha="rice-rec", kind="recording", ext="mp3", source="forvo",
                     origin="https://forvo.com/x", licence="cc-by",
                     acquired=date(2026, 1, 1), speaker_id="somchai")
    ctx.db.append(port="provide", backend="forvo",
                 key=ProvideKey(source="forvo", kind="", query="rice"), subject="rice",
                 question={"kind": "recording", "subject_kind": "word"},
                 answer={"items": [{"sha": "rice-rec"}]})
    ctx.db.append(port="assess", backend="mechanical",
                 key=MechanicalKey(check="duration", params="0.2-5.0", subject="rice",
                                   artifact_sha="rice-rec"),
                 subject="rice",
                 question={"role": "recording-for-word", "artifact_sha": "rice-rec",
                          "kind": "recording", "subject_kind": "word"},
                 answer={"value": True})

    # Two already-passing picture candidates from two already-tried
    # sources (real media: a re-attempt over all candidates on record --
    # the reopened need's own -- would otherwise need to prepare them
    # too), then the learner's rejection of the first -- no acceptable
    # floor, so the need stays open despite two machine passes on record.
    shas = {}
    for source, seed in (("openverse", "pic-a"), ("wikimedia", "pic-b")):
        shas[seed] = ctx.media_store.add_image(_jpeg_bytes(seed), "jpg").sha
        ctx.db.add_media(sha=shas[seed], kind="picture", ext="jpg", source=source,
                         origin=f"https://{source}/{seed}.jpg", licence="by",
                         acquired=date(2026, 1, 1))
        ctx.db.append(port="provide", backend=source,
                     key=ProvideKey(source=source, kind="", query=f"rice-{seed}"),
                     subject="rice",
                     question={"kind": "picture", "subject_kind": "word"},
                     answer={"items": [{"sha": shas[seed]}]})
        pic_question = AssessQuestion(subject="rice", role="picture-for-word",
                                      artifact_sha=shas[seed], kind="picture",
                                      subject_kind="word")
        ctx.db.append(port="assess", backend="judge",
                     key=JudgeKey.for_question(pic_question),
                     subject="rice",
                     question={"role": "picture-for-word", "artifact_sha": shas[seed],
                              "rubric": None, "kind": "picture", "subject_kind": "word"},
                     answer={"value": True})
    ctx.db.append(port="assess", backend="learner",
                 key=LearnerKey(artifact_sha=shas["pic-a"], role="picture-for-word"),
                 subject="rice",
                 question={"role": "picture-for-word", "artifact_sha": shas["pic-a"],
                          "kind": "rating"},
                 answer={"value": "unacceptable-none"})
    assert current_best_of(ctx, "rice", "picture").artifact_sha is None   # rejected, no floor

    r0 = run(ctx, budgets={})   # escalates the reopened need to its next (third) source
    assert r0.batch_id is not None and r0.pending >= 1

    asks_before = list(fake_search.asks)
    fake_batch.complete_all(r0.batch_id, passed=True)   # the third candidate fits too
    r1 = run(ctx, budgets={})

    assert fake_search.asks == asks_before   # the loop did not re-attempt "rice" this run
    assert r1.batch_id is not None   # the preference question over all three, its own new batch
    assert r1.pending == 1
    assert r1.preferences == 0
    assert current_best_of(ctx, "rice", "picture").artifact_sha is None   # still rejected

    for report in (r0, r1):
        assert (report.available == report.attempted + report.exhausted + report.pending
               + report.unserved + report.budgeted + report.deferred)


def test_a_words_open_recording_is_still_attempted_alongside_its_resolve_time_preference(
        tmp_path, fake_search, fake_batch):
    """Fix round 3: `collected_this_run` is keyed per (subject, kind), not
    just subject -- pending() itself only ever names subjects, but the
    loop-skip this feeds must not hold back a word's OTHER still-open
    need merely because its picture also raised a resolve-time
    preference question this same run.
    """
    root = _batch_fixture_deck(tmp_path, (RICE,), (target("rice/receptive", "rice"),))
    ctx = _wire(build_sourcing(root), fake_search, batch=fake_batch)
    ctx.provider._backends["tts"] = _Tts(ctx.media_store)   # forvo stays silent (_wire's default)
    ctx.assessor._backends["mechanical"].duration_of = lambda path: 1.0
    ctx.syllabus = ctx.syllabus.with_sentences(
        (sentence(((RICE.id,),), thai_of(RICE), gloss="rice"),))   # ข้าว: rice

    # The sentence's text is the word's own thai spelling: its own
    # recording/picture needs (F1 defect 2) share Forvo/TTS's
    # content-addressed artifact sha with the word's -- outside this
    # test's own subject (Fix round 3's own concern). Each is seeded
    # here already carrying the verdict a real attempt leaves
    # (mechanical for the recording, judge for the picture), not a
    # learner override, so neither reaches `available_needs()` again.
    sentence_sha = text_sha("ข้าว")  # rice
    ctx.db.add_speaker(Speaker(id="somchai", kind="native"))
    ctx.db.add_media(sha="sentence-rec-seed", kind="recording", ext="mp3", source="forvo",
                     origin="https://forvo.com/x", licence="cc-by",
                     acquired=date(2026, 1, 1), speaker_id="somchai")
    ctx.db.append(port="provide", backend="forvo",
                 key=ProvideKey(source="forvo", kind="", query="ข้าว"),  # rice: the sentence's own text
                 subject=sentence_sha,
                 question={"kind": "recording", "subject_kind": "sentence"},
                 answer={"items": [{"sha": "sentence-rec-seed"}]})
    ctx.db.append(port="assess", backend="mechanical",
                 key=MechanicalKey(check="duration", params="seed", subject=sentence_sha,
                                   artifact_sha="sentence-rec-seed"),
                 subject=sentence_sha,
                 question={"role": "recording-for-sentence", "artifact_sha": "sentence-rec-seed",
                          "kind": "recording", "subject_kind": "sentence"},
                 answer={"value": True})
    ctx.db.add_media(sha="sentence-pic-seed", kind="picture", ext="jpg", source="openverse",
                     origin="https://x/rice.jpg", licence="by", acquired=date(2026, 1, 1))
    ctx.db.append(port="provide", backend="openverse",
                 # "rice": the sentence's own gloss -- _picture_query_for's
                 # fallback query for a scene picture with no drafted phrase.
                 key=ProvideKey(source="openverse", kind="", query="rice"),
                 subject=sentence_sha,
                 question={"kind": "picture", "subject_kind": "sentence"},
                 answer={"items": [{"sha": "sentence-pic-seed"}]})
    scene_question = AssessQuestion(subject=sentence_sha, role="scene-for-sentence",
                                    artifact_sha="sentence-pic-seed",
                                    rubric=ctx.rubrics["scene-for-sentence"], kind="picture",
                                    subject_kind="sentence")
    ctx.db.append(port="assess", backend="judge",
                 key=JudgeKey.for_question(scene_question),
                 subject=sentence_sha,
                 question={"role": "scene-for-sentence", "artifact_sha": "sentence-pic-seed",
                          "rubric": ctx.rubrics["scene-for-sentence"], "kind": "picture",
                          "subject_kind": "sentence"},
                 answer={"value": True})

    # Two already-passing picture candidates, then the learner's rejection
    # of the first -- no acceptable floor, so the picture need stays open
    # (its own resolve-time preference question lands in `pending`, not
    # `preferences`); the recording need is left untouched, open too.
    shas = {}
    for source, seed in (("openverse", "pic-a"), ("wikimedia", "pic-b")):
        shas[seed] = ctx.media_store.add_image(_jpeg_bytes(seed), "jpg").sha
        ctx.db.add_media(sha=shas[seed], kind="picture", ext="jpg", source=source,
                         origin=f"https://{source}/{seed}.jpg", licence="by",
                         acquired=date(2026, 1, 1))
        ctx.db.append(port="provide", backend=source,
                     key=ProvideKey(source=source, kind="", query=f"rice-{seed}"),
                     subject="rice",
                     question={"kind": "picture", "subject_kind": "word"},
                     answer={"items": [{"sha": shas[seed]}]})
        pic_question = AssessQuestion(subject="rice", role="picture-for-word",
                                      artifact_sha=shas[seed], kind="picture",
                                      subject_kind="word")
        ctx.db.append(port="assess", backend="judge",
                     key=JudgeKey.for_question(pic_question),
                     subject="rice",
                     question={"role": "picture-for-word", "artifact_sha": shas[seed],
                              "rubric": None, "kind": "picture", "subject_kind": "word"},
                     answer={"value": True})
    ctx.db.append(port="assess", backend="learner",
                 key=LearnerKey(artifact_sha=shas["pic-a"], role="picture-for-word"),
                 subject="rice",
                 question={"role": "picture-for-word", "artifact_sha": shas["pic-a"],
                          "kind": "rating"},
                 answer={"value": "unacceptable-none"})
    assert current_best_of(ctx, "rice", "picture").artifact_sha is None   # rejected, no floor

    r0 = run(ctx, budgets={})   # escalates the picture to a third source; forvo finds nothing
    assert r0.batch_id is not None and r0.pending >= 1

    fake_batch.complete_all(r0.batch_id, passed=True)   # the third candidate fits too
    r1 = run(ctx, budgets={})   # resolves it -- a preference question over all three -- and,
                                # in the SAME run, escalates the still-open recording to tts

    assert r1.pending == 1 and r1.preferences == 0        # the picture's preference, still a need
    assert r1.improved == 1                               # the recording, attempted this run
    assert current_best_of(ctx, "rice", "picture").artifact_sha is None      # still rejected
    assert current_best_of(ctx, "rice", "recording").artifact_sha is not None

    for report in (r0, r1):
        assert (report.available == report.attempted + report.exhausted + report.pending
               + report.unserved + report.budgeted + report.deferred)
