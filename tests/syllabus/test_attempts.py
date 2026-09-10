"""attempts.py: one Source asked under the need's own subject, what it
returns ingested, the speaker recorded, and the judge questions collected.
Real SyllabusDb + MediaStore; fake Provide/Assess backends; no network."""
import hashlib
import io
import json
import logging
from datetime import date

import pytest
from PIL import Image as PILImage

from thai_syllabus.assessor import (Assessor, JudgeBackend, JudgeUnreachable,
                                    RawVerdict, RenditionBackend)
from thai_syllabus.attempts import (AttemptResult, Need, Sourcing, _sentence_prompt, assess_first,
                                    attempt, current_best_of, sentence_attempt, sources_for)
from thai_syllabus.cachekeys import (JudgeKey, LlmPromptKey, MechanicalKey, ProvideKey,
                                    rendition_identity, sha)
from thai_syllabus.derivations import exhausted
from thai_syllabus.record import DRAFT_SUBJECT, drafts_in, rows_for, sentence_drafts
from thai_syllabus.entities import Category, Clauses, MinimalPair, Sentence, SoundConfusion, text_sha
from thai_syllabus.ids import WordId
from thai_syllabus.media import Provenance, Speaker
from thai_syllabus.provider import FetchBackend, LlmBackend, Provider, RawAnswer, TtsBackend
from thai_syllabus.rulebook import (PICTURE_FIT_RUBRIC, PICTURE_PREFERENCE_RUBRIC,
                                    SENTENCE_FOR_TARGET_RUBRIC)
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.transport import (Completion, FetchRefused, QuotaExhausted, SynthesisRefused,
                                     TransportError)
from thai_syllabus.tts import pick_voice

from .builders import target, word

# This fixture's own role -> rubric map (rulebook.rubrics_for covers only
# roles a judged Rule registers).
_RUBRICS = {"picture-for-word": PICTURE_FIT_RUBRIC,
            "picture-preference": PICTURE_PREFERENCE_RUBRIC,
            "scene-for-sentence": PICTURE_FIT_RUBRIC,
            "sentence-for-target": SENTENCE_FOR_TARGET_RUBRIC}

_MALE = ("th-M-a", "th-M-b")
_FEMALE = ("th-F-a", "th-F-b")


# --- fake backends ----------------------------------------------------------

class _Search:
    """Records every query it was asked, answers one hit per url."""
    def __init__(self, urls):
        self.urls, self.queries = list(urls), []

    def cache_key(self, q):
        return ProvideKey(source="openverse", kind="", query=q.params["query"])

    def fetch(self, q):
        self.queries.append(q.params["query"])
        return RawAnswer(items=tuple({"url": u, "source": "openverse", "licence": "by"}
                                     for u in self.urls))


def _jpeg_bytes(url: str) -> bytes:
    """A decodable JPEG unique to `url`; green for a "good" url, red
    otherwise -- the fit signal _Judge reads back off the pixels."""
    digest = hashlib.sha256(url.encode()).digest()
    colour = ((digest[0] % 50, 200 + digest[1] % 56, digest[2] % 50) if "good" in url
              else (200 + digest[0] % 56, digest[1] % 50, digest[2] % 50))
    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4), colour).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class _Judge:
    """More than one attachment is a preference question; one attachment is
    a fit question, answered from the image's own colour."""
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, attachments=()):
        self.calls.append([str(a) for a in attachments])
        if len(attachments) > 1:
            names = [str(a).rsplit("/", 1)[-1].split(".")[0] for a in attachments]
            return Completion(text='{"ranking": ' + str(names).replace("'", '"') + "}")
        green = any(_is_green(a) for a in attachments)
        return Completion(text='{"value": %s, "evidence": "e"}' % ("true" if green else "false"))


def _is_green(path) -> bool:
    r, g, b = PILImage.open(path).convert("RGB").getpixel((0, 0))
    return g > r


class _Forvo:
    def __init__(self, items_by_word):
        self.items_by_word = dict(items_by_word)

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        return RawAnswer(items=tuple(self.items_by_word.get(q.params["word"], ())), cost=1.0)


class _QuotaForvo:
    """Forvo stating its own daily quota is spent (spec 3 section 6a)."""
    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        raise QuotaExhausted("forvo")


class _Tts:
    def __init__(self):
        self.voices = []

    def synthesize(self, text, voice):
        self.voices.append(voice)
        return f"{text}-{voice}".encode()

    @property
    def last_voice(self):
        return self.voices[-1]


class _Llm:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def cache_key(self, q):
        return LlmPromptKey(producer="sentence-drafter", model="m",
                            prompt_sha=sha(q.params["prompt"]))

    def fetch(self, q):
        self.prompts.append(q.params["prompt"])
        return RawAnswer(items=(self.text,))


class _Mechanical:
    """A duration-shaped mechanical backend passing every artifact but
    `failing_subject`'s.
    """

    def __init__(self, ok=True, failing_subject=None):
        self.ok, self.failing_subject = ok, failing_subject

    def cache_key(self, q):
        return MechanicalKey(check="duration", params="0.2-5.0", artifact_sha=q.artifact_sha)

    def fetch(self, q):
        passes = self.ok and q.subject != self.failing_subject
        return RawVerdict(value=passes, evidence="duration=1.0s" if passes else "too short")


def _mechanical(ok=True, failing_subject=None):
    return _Mechanical(ok=ok, failing_subject=failing_subject)


def _rendition_backend(db):
    def speaker_of(sha):
        prov = db.media_provenance(sha)
        return prov.get("speaker_id") if prov else None
    return RenditionBackend(speaker_of=speaker_of)


def _batch_judge():
    class _NeverSubmits:
        def submit(self, requests):
            raise AssertionError("ask_many must not submit a batch")
    return JudgeBackend(model="m", transport="batch", batch_transport=_NeverSubmits())


# --- contexts ---------------------------------------------------------------

def _sourcing(tmp_path, syllabus, *, backends, assess, media=None) -> Sourcing:
    db = SyllabusDb(tmp_path / "syllabus.db")
    return Sourcing(syllabus=syllabus, provider=Provider(record=db, cache=db, backends=backends),
                    assessor=Assessor(record=db, cache=db, backends=assess), db=db,
                    media_store=media or MediaStore(tmp_path / "media"), rubrics=dict(_RUBRICS),
                    provenance_prior=("commission", "forvo", "tts"), image_candidates=3,
                    today=lambda: date(2026, 9, 3),
                    voices={"male": _MALE, "female": _FEMALE},
                    query_hints={"Food": "food", "Colors": "color swatch"})


def _word_syllabus(*, productive=False) -> Syllabus:
    skill = "productive" if productive else "receptive"
    return Syllabus(words=(word("rice", "ข้าว", "rice (cooked)"),),   # ข้าว: rice
                    targets=(target(f"rice/{skill}", "rice", skill=skill),),
                    categories=(Category(name="Food", members=frozenset({"rice"})),))


def _picture_ctx(tmp_path, syllabus=None, *, judge=None, urls=("https://x/bad.jpg",
                                                               "https://x/good.jpg",
                                                               "https://x/good2.jpg")):
    media = MediaStore(tmp_path / "media")
    search = _Search(urls)
    complete = _Judge() if judge is None else judge
    holder: list[Sourcing] = []

    def resolve(sha):
        prov = holder[0].db.media_provenance(sha)
        path = media.path_for(sha, prov["ext"]) if prov else None
        return path if path is not None and path.exists() else None

    ctx = _sourcing(tmp_path, syllabus or _word_syllabus(), media=media, backends={
        "openverse": search,
        "imgfetch": FetchBackend(media=media, fetcher=lambda url: (_jpeg_bytes(url), "jpg"))},
        assess={"judge": JudgeBackend(model="m", transport="api", complete=complete,
                                      resolve_path=resolve)})
    holder.append(ctx)
    return ctx, search, complete


def _recording_ctx(tmp_path, syllabus, forvo_items=(), *, mechanical=None):
    media = MediaStore(tmp_path / "media")
    tts = _Tts()
    db = SyllabusDb(tmp_path / "syllabus.db")
    ctx = _sourcing(tmp_path, syllabus, media=media, backends={
        "forvo": _Forvo(forvo_items),
        "audiofetch": FetchBackend(media=media, fetcher=lambda url: (url.encode(), "mp3")),
        "tts": TtsBackend(tts=tts, voices=list(_MALE) + list(_FEMALE), media=media,
                          pick_voice=pick_voice)},
        assess={"mechanical": mechanical or _mechanical(),
                "rendition": _rendition_backend(db)})
    return ctx, tts


def _pair_syllabus() -> Syllabus:
    confusion = SoundConfusion(id="tone:rising-vs-low", dimension="tone", sounds=("rising", "low"))
    return Syllabus(words=(word("white", "ขาว", "white"), word("news", "ข่าว", "news")),
                    targets=(target("white/receptive", "white"), target("news/receptive", "news")),
                    confusions=(confusion,),
                    pairs=(MinimalPair(id="p1", confusion=confusion.id,
                                       members=("white", "news")),))


# --- the source roster ------------------------------------------------------

def test_sources_for_picture_is_cost_ordered():
    assert sources_for("picture") == ("openverse", "wikimedia", "pexels")


def test_a_sentence_and_a_word_share_the_recording_source_roster():
    # the artifact kind is the same; only the subject differs
    assert sources_for("recording") == ("forvo", "tts")


def test_a_need_knows_the_role_its_subject_kind_puts_it_under():
    assert Need("rice", "picture").role == "picture-for-word"
    assert Need("sha", "picture", "sentence").role == "scene-for-sentence"
    assert Need("rice", "recording").role == "recording-for-word"
    assert Need("sha", "recording", "sentence").role == "recording-for-sentence"
    assert Need("p1", "rendition", "pair").role == "rendition-for-pair"


def test_attempt_refuses_an_artifact_kind_it_has_no_attempt_for(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    with pytest.raises(ValueError, match="grapheme-keyword"):
        attempt(ctx, Need("g1", "grapheme-keyword", "grapheme"), "llm")


# --- picture: the query, the ingest, the fit questions ----------------------

def test_picture_attempt_searches_the_gloss_head_term_with_the_category_qualifier(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path)
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == ["rice food"]


def test_picture_attempt_searches_a_judge_suggestion_once_one_is_on_record(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path)
    ctx.db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(None, None, "rice", "picture-for-word"), subject="rice",
                  question={"role": "picture-for-word", "kind": "picture"},
                  answer={"value": False, "suggestion": "a bowl of steamed jasmine rice"})
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == ["a bowl of steamed jasmine rice"]


def test_picture_attempt_ingests_each_hit_with_its_provenance(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    res = attempt(ctx, Need("rice", "picture"), "openverse")
    shas = [i["sha"] for r in rows_for(ctx.db, "rice", "picture") if r.port == "provide"
            for i in r.answer["items"] if "sha" in i]
    assert res.attempted and len(shas) == 3
    for sha in shas:
        prov = ctx.db.media_provenance(sha)
        assert prov["source"] == "openverse" and prov["ext"] == "jpg"


def test_picture_attempt_under_an_inline_judge_resolves_fit_and_preference_in_one_pass(tmp_path):
    ctx, _search, judge = _picture_ctx(tmp_path)
    res = attempt(ctx, Need("rice", "picture"), "openverse")
    assert res.questions == [] and res.excluded == {}
    fits = [c for c in judge.calls if len(c) == 1]
    preferences = [c for c in judge.calls if len(c) == 2]
    assert len(fits) == 3 and len(preferences) == 1
    best = current_best_of(ctx, "rice", "picture")
    assert best.artifact_sha is not None and best.rank > 50.0


def test_picture_attempt_under_batch_collects_fit_questions(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path, urls=("https://x/a.jpg", "https://x/b.jpg"))
    ctx.assessor = Assessor(record=ctx.db, cache=ctx.db, backends={"judge": _batch_judge()})
    res = attempt(ctx, Need("rice", "picture"), "openverse")
    assert res.attempted
    assert {q.question.role for q in res.questions} == {"picture-for-word"}
    assert len(res.questions) == 2


def test_a_batch_deck_leaves_the_preference_question_to_the_run(tmp_path):
    """Under a batch transport the preference ask belongs to the run's
    resolve step. Gating it on "this attempt collected nothing" misfired
    exactly here: on the second pass every fit is a cache hit, so the
    attempt collects nothing and used to ask a preference inline anyway."""
    ctx, _search, _judge = _picture_ctx(tmp_path, urls=("https://x/good.jpg",
                                                        "https://x/good2.jpg"))
    ctx.assessor = Assessor(record=ctx.db, cache=ctx.db, backends={"judge": _batch_judge()})
    collected = attempt(ctx, Need("rice", "picture"), "openverse").questions
    assert len(collected) == 2
    for prepared in collected:                       # the batch's verdicts land
        ctx.db.append(port="assess", backend="judge", key=prepared.key,
                      subject=prepared.question.subject,
                      question={"role": prepared.question.role, "kind": "picture",
                                "artifact_sha": prepared.question.artifact_sha,
                                "rubric": prepared.question.rubric,
                                "subject_kind": "word", "params": {}},
                      answer={"value": True})
    assert attempt(ctx, Need("rice", "picture"), "openverse").questions == []


def test_a_second_picture_attempt_spends_nothing_new(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    attempt(ctx, Need("rice", "picture"), "openverse")
    again = attempt(ctx, Need("rice", "picture"), "openverse")
    assert again.attempted
    assert all(s.asks == 0 for s in again.spend.values())


def test_the_search_ask_stays_on_the_record_when_the_judge_cannot_be_reached(tmp_path):
    def boom(prompt, attachments=()):
        raise TransportError("api transport failed: 401")
    ctx, _search, _judge = _picture_ctx(tmp_path, judge=boom, urls=("https://x/good.jpg",))
    with pytest.raises(JudgeUnreachable):
        attempt(ctx, Need("rice", "picture"), "openverse")
    assert rows_for(ctx.db, "rice", "picture")     # the search ask is on the record


def test_a_candidate_the_judge_cannot_prepare_is_excluded_and_the_rest_are_judged(tmp_path):
    ctx, _search, judge = _picture_ctx(tmp_path, urls=("https://x/good.jpg",))
    ctx.db.add_media(sha="ghost", kind="picture", ext="jpg", source="legacy", origin="",
                     licence="?", acquired=date(2026, 1, 1))
    ctx.db.append(port="provide", backend="openverse",
                  key=ProvideKey(source="openverse", kind="", query="legacy"), subject="rice",
                  question={"provides": "picture", "kind": "picture", "params": {}},
                  answer={"items": [{"sha": "ghost"}]})
    res = attempt(ctx, Need("rice", "picture"), "openverse")
    assert [x.reason for x in res.excluded.values()] == ["artifact not found: ghost"]
    assert current_best_of(ctx, "rice", "picture").artifact_sha != "ghost"


# --- assess-first (spec 3 section 5) ----------------------------------------

def _seed_current_picture(ctx, subject, url="https://x/good-legacy.jpg"):
    """A candidate on record with no verdict: the row spec 2 section 4
    r7 writes for the old deck's current picture, and its media row."""
    ingest = ctx.media_store.add_image(_jpeg_bytes(url), ext="jpg")
    ctx.db.add_media(sha=ingest.sha, kind="picture", ext=ingest.ext, source="openverse",
                     origin=url, licence="unknown", acquired=date(2026, 9, 3))
    ctx.db.append(port="provide", backend="legacy-current",
                  key=ProvideKey(source="legacy-current", kind="picture", query=subject),
                  subject=subject,
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"image": "images/pw-1.jpg"}},
                  answer={"items": [{"sha": ingest.sha, "ext": ingest.ext}]})
    return ingest.sha


def test_assess_first_judges_the_waiting_candidate_and_asks_no_source(tmp_path):
    ctx, search, judge = _picture_ctx(tmp_path)
    legacy_sha = _seed_current_picture(ctx, "rice")
    res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None and res.attempted and res.questions == []
    assert len(judge.calls) == 1 and search.queries == []
    assert all(r.port != "attempt" for r in rows_for(ctx.db, "rice", "picture"))
    assert current_best_of(ctx, "rice", "picture").artifact_sha == legacy_sha


def test_assess_first_is_none_once_every_candidate_is_judged(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    _seed_current_picture(ctx, "rice")
    assert assess_first(ctx, Need("rice", "picture")) is not None
    assert assess_first(ctx, Need("rice", "picture")) is None


def test_assess_first_is_none_with_no_candidate_on_record(tmp_path):
    ctx, _search, judge = _picture_ctx(tmp_path)
    assert assess_first(ctx, Need("rice", "picture")) is None
    assert judge.calls == []


def test_assess_first_is_none_when_every_waiting_candidate_is_excluded(tmp_path):
    ctx, _search, judge = _picture_ctx(tmp_path)
    ctx.db.append(port="provide", backend="legacy-current",
                  key=ProvideKey(source="legacy-current", kind="picture", query="rice"),
                  subject="rice",
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"image": "images/pw-1.jpg"}},
                  answer={"items": [{"sha": "0" * 64, "ext": "jpg"}]})  # bytes never stored
    assert assess_first(ctx, Need("rice", "picture")) is None
    assert judge.calls == []


def test_assess_first_logs_the_candidates_it_excluded_before_falling_through(tmp_path, caplog):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    ctx.db.append(port="provide", backend="legacy-current",
                  key=ProvideKey(source="legacy-current", kind="picture", query="rice"),
                  subject="rice",
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"image": "images/pw-1.jpg"}},
                  answer={"items": [{"sha": "0" * 64, "ext": "jpg"}]})
    with caplog.at_level(logging.WARNING, logger="thai_syllabus.attempts"):
        assert assess_first(ctx, Need("rice", "picture")) is None
    assert "every awaiting candidate was excluded" in caplog.text and "0" * 64 in caplog.text


def test_assess_first_under_batch_collects_the_fit_question(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    ctx.assessor = Assessor(record=ctx.db, cache=ctx.db, backends={"judge": _batch_judge()})
    legacy_sha = _seed_current_picture(ctx, "rice")
    res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None
    assert [q.question.artifact_sha for q in res.questions] == [legacy_sha]
    assert {q.question.role for q in res.questions} == {"picture-for-word"}


def test_assess_first_asks_nothing_for_a_kind_the_judge_does_not_rank(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    assert assess_first(ctx, Need("rice", "recording")) is None


# --- scene picture: the same attempt, subject = text_sha --------------------

def _sentence(text="ข้าว", gloss="the rice is tasty",   # ข้าว: rice
              clauses: Clauses = ((WordId("rice"),),)) -> Sentence:
    return Sentence(clauses=clauses, text=text, gloss=gloss, voice="learner_voice",
                    provenance=Provenance(source="llm", origin="m", licence="generated",
                                          acquired=date(2026, 9, 3)))


def test_scene_picture_attempt_searches_the_sentence_gloss_not_its_thai(tmp_path):
    """Both image corpora index English metadata, so a Thai query matches
    only the handful of Thai-captioned items they hold."""
    sentence = _sentence()
    ctx, search, _judge = _picture_ctx(
        tmp_path, _word_syllabus().with_sentences([sentence]), urls=("https://x/good.jpg",))
    res = attempt(ctx, Need(sentence.text_sha, "picture", "sentence"), "openverse")
    assert search.queries == ["the rice is tasty"]
    assert res.attempted and rows_for(ctx.db, sentence.text_sha, "picture")
    verdicts = [r for r in ctx.db.assessments_of(sentence.text_sha) if r.port == "assess"]
    assert {r.question["role"] for r in verdicts} == {"scene-for-sentence"}
    assert {r.question["subject_kind"] for r in verdicts} == {"sentence"}


def test_a_scene_picture_attempt_refuses_a_sentence_with_no_gloss(tmp_path):
    sentence = _sentence(gloss="")
    ctx, _search, _judge = _picture_ctx(
        tmp_path, _word_syllabus().with_sentences([sentence]), urls=("https://x/good.jpg",))
    with pytest.raises(ValueError, match="gloss"):
        attempt(ctx, Need(sentence.text_sha, "picture", "sentence"), "openverse")


# --- recording: the voice constraint and the speaker attributes -------------

def test_recording_attempt_draws_any_sex_without_a_productive_target(tmp_path):
    ctx, tts = _recording_ctx(tmp_path, _word_syllabus())
    attempt(ctx, Need("rice", "recording"), "tts")
    assert tts.last_voice in _MALE + _FEMALE
    assert ctx.db.speaker(f"tts:{tts.last_voice}").sex in ("male", "female")


def test_recording_attempt_draws_a_male_voice_for_a_productive_target(tmp_path):
    ctx, tts = _recording_ctx(tmp_path, _word_syllabus(productive=True))
    attempt(ctx, Need("rice", "recording"), "tts")
    assert tts.last_voice in _MALE
    assert ctx.db.speaker(f"tts:{tts.last_voice}") == Speaker(
        id=f"tts:{tts.last_voice}", kind="synthetic", sex="male")


def test_forvo_attempt_records_sex_and_country(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3", "sex": "m",
                  "country": "Thailand"}]})   # ข้าว: rice
    attempt(ctx, Need("rice", "recording"), "forvo")
    assert ctx.db.speaker("forvo:somchai") == Speaker("forvo:somchai", "native", sex="male",
                                                      region="Thailand")
    assert current_best_of(ctx, "rice", "recording").source == "mechanical"


def test_a_productive_word_takes_only_a_forvo_speaker_forvo_calls_male(tmp_path):
    """A recording that plays on a productive back has to be in the
    learner's register (E2); an unstated sex is not a claim that it is."""
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(productive=True), {
        "ข้าว": [{"username": "malee", "pathmp3": "https://f/1.mp3", "sex": "f"},   # ข้าว: rice
                  {"username": "anon", "pathmp3": "https://f/2.mp3"},
                  {"username": "somchai", "pathmp3": "https://f/3.mp3", "sex": "m"}]})
    attempt(ctx, Need("rice", "recording"), "forvo")
    assert ctx.db.speaker("forvo:malee") is None and ctx.db.speaker("forvo:anon") is None
    assert ctx.db.speaker("forvo:somchai").sex == "male"


def test_a_receptive_word_takes_a_forvo_speaker_of_any_sex(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "malee", "pathmp3": "https://f/1.mp3", "sex": "f"}]})  # ข้าว: rice
    attempt(ctx, Need("rice", "recording"), "forvo")
    assert ctx.db.speaker("forvo:malee").sex == "female"


def test_forvo_attempt_leaves_an_attribute_forvo_did_not_give_unknown(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "anon", "pathmp3": "https://f/u.mp3"}]})   # ข้าว: rice
    attempt(ctx, Need("rice", "recording"), "forvo")
    assert ctx.db.speaker("forvo:anon") == Speaker("forvo:anon", "native")


def test_a_recording_attempt_raises_quota_exhausted_and_appends_no_row(tmp_path):
    """Forvo's own quota statement is not a transient failure: it
    reraises without a fetches.failed() and without an outcome row."""
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    ctx.provider._backends["forvo"] = _QuotaForvo()
    with pytest.raises(QuotaExhausted):
        attempt(ctx, Need("rice", "recording"), "forvo")
    assert not [r for r in rows_for(ctx.db, "rice", "recording") if r.port == "attempt"]


def test_a_forvo_attempt_that_found_nothing_is_still_on_the_record(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    res = attempt(ctx, Need("rice", "recording"), "forvo")
    assert res.attempted
    assert exhausted(ctx.db, "rice", "recording", sources=("forvo", "tts"),
                     attempt_cap=8, transient_cap=ctx.transient_cap).attempts == 1


def test_a_sentence_recording_keeps_the_recording_artifact_kind(tmp_path):
    """compile and the media index look a sentence's audio up as a
    "recording" under its text_sha; the subject kind, not the artifact
    kind, is what makes it a sentence's."""
    sentence = _sentence()
    ctx, tts = _recording_ctx(tmp_path, _word_syllabus().with_sentences([sentence]))
    attempt(ctx, Need(sentence.text_sha, "recording", "sentence"), "tts")
    assert tts.voices == [pick_voice(sentence.text_sha, list(_MALE) + list(_FEMALE))]
    best = current_best_of(ctx, sentence.text_sha, "recording")
    assert ctx.db.media_provenance(best.artifact_sha)["speaker"].kind == "synthetic"
    verdicts = [r for r in ctx.db.assessments_of(sentence.text_sha) if r.port == "assess"]
    assert {r.question["role"] for r in verdicts} == {"recording-for-sentence"}


def test_a_sentence_filling_a_productive_target_draws_a_male_voice(tmp_path):
    sentence = _sentence(text="ข้าว")   # ข้าว: rice
    syllabus = Syllabus(words=(word("rice", "ข้าว", "rice"),),
                        targets=(target("rice/productive", "rice", skill="productive"),)
                        ).with_sentences([sentence])
    ctx, tts = _recording_ctx(tmp_path, syllabus)
    attempt(ctx, Need(sentence.text_sha, "recording", "sentence"), "tts")
    assert tts.last_voice in _MALE


# --- rendition: one answer under the pair, one speaker across the members ---

def test_rendition_attempt_appends_under_the_pair(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"},   # ขาว: white
                {"username": "malee", "pathmp3": "https://f/a2.mp3"}],
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]})  # ข่าว: news
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    rows = rows_for(ctx.db, "p1", "rendition")
    provided = [r for r in rows if r.port == "provide"]
    assert provided and set(provided[-1].answer["items"][0]) >= {"member", "sha", "speaker"}
    assert {i["speaker"]["id"] for i in provided[-1].answer["items"]} == {"forvo:somchai"}
    assert exhausted(ctx.db, "p1", "rendition", sources=("forvo",), attempt_cap=8,
                     transient_cap=ctx.transient_cap).attempts == 1


def test_rendition_attempt_ranks_the_member_set_by_the_one_speaker_check(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]})  # ข่าว: news
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    verdict = [r for r in ctx.db.assessments_of("p1") if r.backend == "rendition"][-1]
    assert verdict.answer["value"] is True
    assert set(verdict.question["params"]["members"]) == {"white", "news"}
    assert current_best_of(ctx, "p1", "rendition").artifact_sha is not None
    # the members' own recording needs read the same verdicts
    assert current_best_of(ctx, "white", "recording").artifact_sha is not None
    assert current_best_of(ctx, "news", "recording").artifact_sha is not None


def test_rendition_attempt_fails_the_check_when_a_member_recording_does_not(tmp_path):
    """One speaker across the members is not enough: a member recording
    that failed its own mechanical check must not leave the pair with a
    current-best rendition."""
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]},  # ข่าว: news
        mechanical=_mechanical(failing_subject="news"))
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    verdict = [r for r in ctx.db.assessments_of("p1") if r.backend == "rendition"][-1]
    assert verdict.answer["value"] is False
    assert "news" in verdict.answer["evidence"]
    assert current_best_of(ctx, "p1", "rendition").artifact_sha is None


def test_a_member_check_that_never_resolved_excludes_the_rendition_instead_of_failing_it(tmp_path):
    """One member's mechanical check failed on the wire: the rendition
    question is excluded for the run and no verdict is cached against the
    member set."""
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]})  # ข่าว: news
    mech = ctx.assessor._backends["mechanical"]
    original = mech.fetch

    def flaky(q):
        if q.subject == "news":
            raise TransportError("ffprobe failed")
        return original(q)

    mech.fetch = flaky
    result = attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    assert result.excluded and not result.questions
    assert not [r for r in rows_for(ctx.db, "p1", "rendition") if r.backend == "rendition"]


def test_the_rendition_verdict_identifies_the_member_set_it_judged(tmp_path):
    """The rendition backend, not the attempt, computes the artifact the
    member set forms -- the identity current_best then ranks."""
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]})  # ข่าว: news
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    row = [r for r in ctx.db.assessments_of("p1") if r.backend == "rendition"][-1]
    members = row.question["params"]["members"]
    assert row.question["artifact_sha"] == rendition_identity(members)
    assert current_best_of(ctx, "p1", "rendition").artifact_sha == rendition_identity(members)


def test_a_source_that_cannot_guarantee_one_speaker_answers_empty(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "malee", "pathmp3": "https://f/b.mp3"}]})    # ข่าว: news
    res = attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    provided = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"]
    assert res.attempted and provided[-1].answer["items"] == []
    assert current_best_of(ctx, "p1", "rendition").artifact_sha is None


def test_a_rendition_attempt_raises_quota_exhausted_and_appends_no_row(tmp_path):
    """Same reraise-without-appending contract as a word's own recording
    attempt (spec 3 section 6a's Quota state)."""
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus())
    ctx.provider._backends["forvo"] = _QuotaForvo()
    with pytest.raises(QuotaExhausted):
        attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    assert not [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "attempt"]


def test_rendition_attempt_falls_to_one_tts_voice_across_the_members(tmp_path):
    ctx, tts = _recording_ctx(tmp_path, _pair_syllabus())
    attempt(ctx, Need("p1", "rendition", "pair"), "tts")
    assert set(tts.voices) == {pick_voice("p1", list(_MALE) + list(_FEMALE))}
    provided = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"]
    assert {i["speaker"]["kind"] for i in provided[-1].answer["items"]} == {"synthetic"}


class _PairForvo:
    """Two members share one speaker; each member's own lookup counts its
    own re-asks (spec 3 section 5: one lookup per member, shared with the
    recording need)."""
    def __init__(self, thai_of):
        self.thai_of = dict(thai_of)      # member -> thai
        self.lookups: dict[str, int] = {}

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        thai = q.params["word"]
        member = next(m for m, t in self.thai_of.items() if t == thai)
        self.lookups[member] = self.lookups.get(member, 0) + 1
        n = self.lookups[member]
        return RawAnswer(items=({"id": 7, "username": "somchai", "sex": "m",
                                 "country": "Thailand",
                                 "pathmp3": f"https://forvo/{member}/{n}.mp3"},), cost=1.0)


class _RefusesOneMembersFirstUrl:
    """Refuses one member's first-round url as content-type; every other
    url, including that member's retried one, answers."""
    def __init__(self, media, refused_url):
        self.media, self.asked, self.refused_url = media, [], refused_url

    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        url = q.params["url"]
        self.asked.append(url)
        if url == self.refused_url:
            raise FetchRefused(reason="content-type",
                               detail='content-type "application/json" is not allowed')
        sha = self.media.write(f"ID3-{url}".encode(), "mp3")
        return RawAnswer(items=({"sha": sha, "ext": "mp3", "speaker": q.params["speaker"],
                                 "speaker_kind": "native", "source": "forvo"},), cost=0.0)


def test_a_served_refusal_in_a_rendition_re_asks_once_for_that_member_only(tmp_path):
    forvo = _PairForvo({"white": "ขาว", "news": "ข่าว"})   # ขาว: white, ข่าว: news
    media = MediaStore(tmp_path / "media")
    audiofetch = _RefusesOneMembersFirstUrl(media, "https://forvo/white/1.mp3")
    db = SyllabusDb(tmp_path / "syllabus.db")
    ctx = _sourcing(tmp_path, _pair_syllabus(),
                    backends={"forvo": forvo, "audiofetch": audiofetch},
                    assess={"mechanical": _mechanical(), "rendition": _rendition_backend(db)},
                    media=media)
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    assert forvo.lookups == {"white": 2, "news": 1}   # one re-ask for white, none for news
    provided = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"]
    items = provided[-1].answer["items"]
    assert {i["member"] for i in items} == {"white", "news"}
    assert {i["speaker"]["id"] for i in items} == {"forvo:somchai"}
    for i in items:
        assert ctx.db.media_provenance(i["sha"])["speaker_id"] == "forvo:somchai"


# --- the sentence attempt: draft, verify by fill_set(), collect questions ---

def _draft_json(word_ids, text, gloss) -> str:
    """One drafting-answer item: a single clause of `word_ids`, `text`,
    `gloss` -- the new clauses-shaped answer (spec 1 section 1)."""
    return json.dumps({"sentences": [
        {"clauses": [list(word_ids)], "text": text, "gloss": gloss}]})


def _sentence_ctx(tmp_path, llm_text, *, judge_value="true", batch=False, syllabus=None):
    syllabus = syllabus if syllabus is not None else Syllabus(
        words=(word("rice", "ข้าว", "rice"), word("eat", "กิน", "eat"),   # ข้าว: rice, กิน: eat
              word("tasty", "อร่อย", "tasty")),   # อร่อย: tasty -- registered, no Target
        targets=(target("eat/receptive", "eat"), target("rice/receptive", "rice")),
        frequency={"eat": 1, "rice": 2})
    judge = (_batch_judge() if batch else JudgeBackend(
        model="m", transport="api",
        complete=lambda p, a=(): Completion(text='{"value": %s, "evidence": "e"}' % judge_value)))
    ctx = _sourcing(tmp_path, syllabus, backends={"llm-sentence": _Llm(llm_text)},
                    assess={"judge": judge})
    return ctx


def test_sentence_attempt_collects_a_judge_question_carrying_the_text_and_gloss(tmp_path):
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat", "rice"), "กินข้าว", "eat rice"),
                        batch=True)   # กินข้าว: eat rice
    res = sentence_attempt(ctx)
    assert res.attempted and len(res.questions) == 1
    question = res.questions[0].question
    assert question.role == "sentence-for-target"
    assert question.params["text"] == "กินข้าว" and question.params["gloss"] == "eat rice"
    assert question.artifact_sha is None       # a sentence judgment attaches no artifact


def test_sentence_attempt_reports_the_drafts_it_produced(tmp_path):
    """`drafted` counts the drafts that fill an open Target, whatever the
    transport did with their judge questions."""
    text = _draft_json(("eat", "rice"), "กินข้าว", "eat rice")   # กินข้าว: eat rice
    # one deck per call: the drafting ask is cached, and a shared db would
    # hand the later calls the first call's answer.
    assert sentence_attempt(_sentence_ctx(tmp_path / "batch", text, batch=True)).drafted == 1
    assert sentence_attempt(_sentence_ctx(tmp_path / "inline", text)).drafted == 1
    assert sentence_attempt(_sentence_ctx(tmp_path / "none", '{"sentences": []}')).drafted == 0


def test_sentence_attempt_reports_how_many_open_targets_it_was_handed(tmp_path):
    """`targets_handed` is min(open Targets, max_targets): the per-run
    Target cap's own count. The run accounts in needs and reads
    `subjects_handed` instead.
    """
    text = _draft_json(("eat", "rice"), "กินข้าว", "eat rice")   # กินข้าว: eat rice
    ctx = _sentence_ctx(tmp_path / "uncapped", text)
    assert sentence_attempt(ctx).targets_handed == 2   # both open Targets, well under the cap
    ctx = _sentence_ctx(tmp_path / "capped", text)
    assert sentence_attempt(ctx, max_targets=1).targets_handed == 1


def test_sentence_attempt_reports_the_words_it_was_handed_targets_for(tmp_path):
    """`subjects_handed` names those Targets' words, one entry per word:
    a word with a receptive and a productive Target open is one
    (word, "sentence") need, which is the unit run.py accounts for."""
    one_word = Syllabus(
        words=(word("rice", "ข้าว", "rice"),),                          # ข้าว: rice
        targets=(target("rice/receptive", "rice"),
                 target("rice/productive", "rice", skill="productive")),
        frequency={"rice": 1})
    result = sentence_attempt(_sentence_ctx(tmp_path, '{"sentences": []}', syllabus=one_word))
    assert result.targets_handed == 2 and result.subjects_handed == frozenset({"rice"})


def test_sentence_attempt_adopts_nothing_itself(tmp_path):
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat", "rice"), "กินข้าว", "eat rice"))   # กินข้าว: eat rice
    res = sentence_attempt(ctx)
    assert res.questions == []                 # the inline judge answered
    assert ctx.db.all_sentences() == []        # ...and the run, not the attempt, adopts


def test_sentence_attempt_fills_every_open_target_its_clauses_use(tmp_path):
    """Fills is membership (spec 1 section 3 clause 1): a draft naming
    two open targets' words in its own clauses fills both, and the judge
    sees the draft once -- not once per target."""
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat", "rice"), "กินข้าว", "eat rice"))   # กินข้าว: eat rice
    res = sentence_attempt(ctx)
    assert res.drafted == 1


def test_sentence_attempt_checks_an_open_target_beyond_the_handed_batch(tmp_path):
    """max_targets caps the handed batch (targets_handed, the prompt),
    but a draft's own fill_set is checked against every open Target the
    whole syllabus still has -- including one the capped batch left out.
    The draft here uses only the capped-out word ("rice"), never the
    handed one ("eat"): a batch-only check sees no open Target and drafts
    nothing (drafted == 0); the whole open set sees rice/receptive and
    drafts one."""
    syllabus = Syllabus(
        words=(word("eat", "กิน", "eat"), word("rice", "ข้าว", "rice")),   # กิน: eat, ข้าว: rice
        targets=(target("eat/receptive", "eat"), target("rice/receptive", "rice")),
        frequency={"eat": 1, "rice": 2})
    text = _draft_json(("rice",), "ข้าว", "rice")   # ข้าว: rice
    judge = JudgeBackend(model="m", transport="api",
                         complete=lambda p, a=(): Completion(text='{"value": true, "evidence": "e"}'))
    ctx = _sourcing(tmp_path, syllabus, backends={"llm-sentence": _Llm(text)},
                    assess={"judge": judge})
    res = sentence_attempt(ctx, max_targets=1)
    assert res.targets_handed == 1   # only eat/receptive handed (rice/receptive is capped out)
    assert res.drafted == 1          # the draft fills rice/receptive, beyond the handed batch


def test_the_judge_question_names_the_last_used_word(tmp_path):
    """params["word"] is the last used word (Syllabus.last_used_word),
    not the first-mentioned target's word: "rice" is ordered after "eat"
    (frequency), so it is the sentence's last used word."""
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat", "rice"), "กินข้าว", "eat rice"),
                        batch=True)   # กินข้าว: eat rice
    res = sentence_attempt(ctx)
    assert res.questions[0].question.params["word"] == "ข้าว"   # ข้าว: rice -- the last used word


def test_sentence_attempt_merges_a_duplicated_draft_into_one_judge_question(tmp_path):
    text = json.dumps({"sentences": [
        {"clauses": [["eat", "rice"]], "text": "กินข้าว", "gloss": ""},
        {"clauses": [["eat", "rice"]], "text": "กินข้าว", "gloss": "eat rice"}]})  # กินข้าว: eat rice
    ctx = _sentence_ctx(tmp_path, text, batch=True)
    res = sentence_attempt(ctx)
    assert len(res.questions) == 1


def test_sentence_attempt_drops_a_repeated_draft_whose_glosses_disagree(tmp_path, caplog):
    text = json.dumps({"sentences": [
        {"clauses": [["eat", "rice"]], "text": "กินข้าว", "gloss": "eat rice"},
        {"clauses": [["eat", "rice"]], "text": "กินข้าว",
         "gloss": "rice is eaten"}]})   # กินข้าว: eat rice
    ctx = _sentence_ctx(tmp_path, text, batch=True)
    with caplog.at_level(logging.WARNING):
        res = sentence_attempt(ctx)
    assert res.questions == [] and res.drafted == 0
    assert any("conflicting glosses" in r.message for r in caplog.records)


class _MultiItemLlm:
    """An llm-sentence backend answering with several items, each its own
    piece of the drafting JSON -- one text split across two items."""

    def __init__(self, *item_texts):
        self.item_texts = item_texts

    def cache_key(self, q):
        return LlmPromptKey(producer="sentence-drafter", model="m",
                            prompt_sha=sha(q.params["prompt"]))

    def fetch(self, q):
        return RawAnswer(items=self.item_texts)


def test_sentence_attempt_drops_a_text_whose_glosses_disagree_across_answer_items(tmp_path):
    """A text listed in two different answer items is one candidate: a
    gloss conflict between the items drops it, the same as a conflict
    within one item -- the merge spans every item, not just one."""
    syllabus = Syllabus(
        words=(word("rice", "ข้าว", "rice"), word("eat", "กิน", "eat")),   # ข้าว: rice, กิน: eat
        targets=(target("eat/receptive", "eat"), target("rice/receptive", "rice")),
        frequency={"eat": 1, "rice": 2})
    judge = _batch_judge()
    ctx = _sourcing(tmp_path, syllabus,
                    backends={"llm-sentence": _MultiItemLlm(
                        json.dumps({"sentences": [{"clauses": [["eat", "rice"]],
                                                   "text": "กินข้าว",   # กินข้าว: eat rice
                                                   "gloss": "eat rice"}]}),
                        json.dumps({"sentences": [{"clauses": [["eat", "rice"]],
                                                   "text": "กินข้าว",
                                                   "gloss": "rice is eaten"}]}))},
                    assess={"judge": judge})
    res = sentence_attempt(ctx)
    assert res.questions == []


def test_sentence_attempt_does_not_judge_a_draft_that_fills_nothing(tmp_path):
    """"tasty" is registered but carries no Target: a draft using only it
    passes check_sentence but fill_set() gates it out."""
    ctx = _sentence_ctx(tmp_path, _draft_json(("tasty",), "อร่อย", "tasty"), batch=True)
    res = sentence_attempt(ctx)
    assert res.questions == [] and res.drafted == 0


def test_sentence_attempt_refuses_a_draft_naming_an_unregistered_word(tmp_path, caplog):
    """Acceptance is the Sentence invariant (Syllabus.check_sentence): a
    draft whose clauses name an id the syllabus has no Word for is
    refused, logged, and never reaches the judge."""
    ctx = _sentence_ctx(tmp_path, _draft_json(("ghost",), "ผี", "a ghost"), batch=True)  # ผี: ghost
    with caplog.at_level(logging.WARNING):
        res = sentence_attempt(ctx)
    assert res.questions == [] and res.drafted == 0
    assert "draft refused" in caplog.text and "ghost" in caplog.text


def test_sentence_attempt_refuses_a_draft_whose_text_does_not_match_its_clauses(tmp_path, caplog):
    """The other half of the Sentence invariant: clauses that render to
    something other than the given text refuse the draft too."""
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat",), "ข้าว", "eat"), batch=True)  # กิน renders, not ข้าว
    with caplog.at_level(logging.WARNING):
        res = sentence_attempt(ctx)
    assert res.questions == [] and res.drafted == 0
    assert "draft refused" in caplog.text


def test_sentence_attempt_is_not_attempted_when_no_target_is_open(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": []}')
    ctx.syllabus = ctx.syllabus.with_sentences([_sentence(   # กินข้าว: eat rice
        text="กินข้าว", gloss="eat rice",
        clauses=((WordId("eat"), WordId("rice")),))])
    res = sentence_attempt(ctx)
    assert res == AttemptResult(attempted=False)
    assert ctx.provider._backends["llm-sentence"].prompts == []


def test_the_drafting_prompt_carries_the_vocabulary_met_and_asks_for_a_gloss(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": []}')
    sentence_attempt(ctx)
    prompt = ctx.provider._backends["llm-sentence"].prompts[0]
    assert "target rice/receptive" in prompt and "กิน" in prompt   # กิน: eat
    assert "male_colloquial" in prompt and '"gloss"' in prompt


# --- the outcome row (ruling 2, spec 3 section 6) ---------------------------

def _outcome(db, subject, kind, source):
    """The newest attempt-outcome row (port "attempt") for (subject, kind,
    source)."""
    rows = [r for r in rows_for(db, subject, kind)
           if r.port == "attempt" and r.backend == source]
    return max(rows, key=lambda r: r.ts)


class _DeadSearch:
    """A source whose own search ask fails on the wire."""
    def cache_key(self, q):
        return ProvideKey(source="openverse", kind="", query=q.params["query"])

    def fetch(self, q):
        raise TransportError("openverse down")


class _DeadImgfetch:
    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        raise TransportError("imgfetch refused")


class _QuotaSearch:
    """A picture search source stating its own quota is spent. Forvo is
    the only real quota source (spec 3 section 3); this stands in for it
    to check the picture attempt's catch is source-agnostic."""
    def cache_key(self, q):
        return ProvideKey(source="openverse", kind="", query=q.params["query"])

    def fetch(self, q):
        raise QuotaExhausted("openverse")


def test_a_picture_attempt_writes_a_candidates_outcome_when_a_hit_is_stored(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path, urls=("https://x/good.jpg",))
    attempt(ctx, Need("rice", "picture"), "openverse")
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.question == {"kind": "picture", "subject_kind": "word", "source": "openverse"}
    assert row.answer["outcome"] == "candidates"
    assert len(row.answer["candidates"]) == 1


def test_a_picture_attempt_writes_a_nothing_outcome_when_the_search_finds_nothing(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path, urls=())
    attempt(ctx, Need("rice", "picture"), "openverse")
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer == {"outcome": "nothing", "candidates": [], "tried": []}


def test_a_picture_attempt_writes_transient_failure_then_reraises_when_the_search_fails(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    ctx.provider._backends["openverse"] = _DeadSearch()
    with pytest.raises(TransportError):
        attempt(ctx, Need("rice", "picture"), "openverse")
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer == {"outcome": "transient-failure", "candidates": [], "tried": []}


def test_a_picture_attempt_writes_transient_failure_when_every_fetch_it_needed_fails(tmp_path):
    """The attempt completes without raising and its outcome row reads
    `transient-failure`.
    """
    ctx, _search, _judge = _picture_ctx(tmp_path, urls=("https://x/good.jpg",
                                                        "https://x/good2.jpg"))
    ctx.provider._backends["imgfetch"] = _DeadImgfetch()
    result = attempt(ctx, Need("rice", "picture"), "openverse")
    assert result.attempted
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer == {"outcome": "transient-failure", "candidates": [], "tried": []}


def test_a_picture_attempt_raises_quota_exhausted_and_appends_no_row(tmp_path):
    """The source's own quota statement is not a transient failure: it
    reraises without a fetches.failed() and without an outcome row."""
    ctx, _search, _judge = _picture_ctx(tmp_path)
    ctx.provider._backends["openverse"] = _QuotaSearch()
    with pytest.raises(QuotaExhausted):
        attempt(ctx, Need("rice", "picture"), "openverse")
    assert not [r for r in rows_for(ctx.db, "rice", "picture") if r.port == "attempt"]


class _RotatingSearch(_Search):
    """First answer: the given urls; every later answer: `fresh`."""
    def __init__(self, urls, fresh):
        super().__init__(urls)
        self.fresh, self.fetches = tuple(fresh), 0

    def fetch(self, q):
        self.fetches += 1
        urls = self.urls if self.fetches == 1 else self.fresh
        return RawAnswer(items=tuple({"url": u, "source": "openverse", "origin": u, "licence": "cc0"}
                                     for u in urls), cost=0.0)


class _ServedRefusingImgfetch:
    """Refuses every url containing "rotted" with an http refusal; stores the rest."""
    def __init__(self, media):
        self.media, self.asked = media, []

    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        url = q.params["url"]
        self.asked.append(url)
        if "rotted" in url:
            raise FetchRefused(reason="http", detail="http 404")
        ingest = self.media.add_image(_jpeg_bytes(url), "jpg")
        return RawAnswer(items=({"sha": ingest.sha, "ext": ingest.ext},), cost=0.0)


def test_a_picture_attempt_re_asks_the_search_once_when_every_hit_is_refused_by_its_server(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    search = _RotatingSearch(("https://x/rotted1.jpg", "https://x/rotted2.jpg"),
                             ("https://x/rotted1.jpg", "https://x/fresh.jpg"))
    ctx.provider._backends["openverse"] = search
    imgfetch = _ServedRefusingImgfetch(ctx.media_store)
    ctx.provider._backends["imgfetch"] = imgfetch
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.fetches == 2
    assert imgfetch.asked == ["https://x/rotted1.jpg", "https://x/rotted2.jpg", "https://x/fresh.jpg"]
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 1


def test_a_picture_attempt_does_not_re_ask_when_a_hit_was_stored(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/rotted1.jpg", "https://x/good.jpg"))
    search = _RotatingSearch(("https://x/rotted1.jpg", "https://x/good.jpg"), ("https://x/never.jpg",))
    ctx.provider._backends["openverse"] = search
    ctx.provider._backends["imgfetch"] = _ServedRefusingImgfetch(ctx.media_store)
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.fetches == 1


def test_a_picture_attempt_does_not_re_ask_on_wire_refusals(tmp_path):
    class _Wire(_ServedRefusingImgfetch):
        def fetch(self, q):
            self.asked.append(q.params["url"])
            raise FetchRefused(reason="wire", detail="timeout")

    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/a.jpg",))
    search = _RotatingSearch(("https://x/a.jpg",), ("https://x/b.jpg",))
    ctx.provider._backends["openverse"] = search
    ctx.provider._backends["imgfetch"] = _Wire(ctx.media_store)
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.fetches == 1
    assert _outcome(ctx.db, "rice", "picture", "openverse").answer["outcome"] == "transient-failure"


def test_a_picture_attempt_caps_the_re_ask_at_image_candidates_and_spends_on_it(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    fresh = tuple(f"https://x/fresh{i}.jpg" for i in range(5))
    search = _RotatingSearch(("https://x/rotted1.jpg", "https://x/rotted2.jpg"), fresh)
    ctx.provider._backends["openverse"] = search
    imgfetch = _ServedRefusingImgfetch(ctx.media_store)
    ctx.provider._backends["imgfetch"] = imgfetch
    result = attempt(ctx, Need("rice", "picture"), "openverse")
    assert imgfetch.asked == ["https://x/rotted1.jpg", "https://x/rotted2.jpg",
                              *fresh[:ctx.image_candidates]]
    assert result.spend["openverse"].asks == 2


class _SearchThenDead(_Search):
    """First answer: the given urls; the second fetch raises TransportError."""
    def __init__(self, urls):
        super().__init__(urls)
        self.fetches = 0

    def fetch(self, q):
        self.fetches += 1
        if self.fetches == 1:
            return RawAnswer(items=tuple({"url": u, "source": "openverse", "origin": u,
                                          "licence": "cc0"} for u in self.urls), cost=0.0)
        raise TransportError("openverse down")


def test_a_picture_attempt_writes_transient_failure_then_reraises_when_the_re_ask_fails(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    search = _SearchThenDead(("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    ctx.provider._backends["openverse"] = search
    ctx.provider._backends["imgfetch"] = _ServedRefusingImgfetch(ctx.media_store)
    with pytest.raises(TransportError):
        attempt(ctx, Need("rice", "picture"), "openverse")
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer == {"outcome": "transient-failure", "candidates": [],
                         "tried": ["https://x/rotted1.jpg", "https://x/rotted2.jpg"]}


# --- an attempt fetches only hits no earlier attempt fetched -----------

def test_a_second_attempt_fetches_only_the_hits_the_first_did_not_try(tmp_path):
    urls = ("https://x/good1.jpg", "https://x/good2.jpg",
           "https://x/good3.jpg", "https://x/good4.jpg")
    ctx, search, _judge = _picture_ctx(tmp_path, urls=urls)
    ctx.image_candidates = 2
    attempt(ctx, Need("rice", "picture"), "openverse")
    first = _outcome(ctx.db, "rice", "picture", "openverse")
    assert first.answer["tried"] == list(urls[:2])
    assert len(first.answer["candidates"]) == 2

    attempt(ctx, Need("rice", "picture"), "openverse")
    second = _outcome(ctx.db, "rice", "picture", "openverse")
    assert second.answer["tried"] == list(urls[2:4])
    assert len(second.answer["candidates"]) == 2
    assert search.queries == ["rice food"]           # the search itself was a cache hit both times

    shas = [i["sha"] for r in rows_for(ctx.db, "rice", "picture") if r.port == "provide"
           for i in r.answer["items"] if "sha" in i]
    assert len(shas) == 4                             # every hit ingested, none refetched


def test_a_refused_url_counts_as_tried_and_is_not_retried_by_a_later_attempt(tmp_path):
    ctx, search, _judge = _picture_ctx(
        tmp_path, urls=("https://x/rotted.jpg", "https://x/good.jpg"))
    ctx.image_candidates = 2
    imgfetch = _ServedRefusingImgfetch(ctx.media_store)
    ctx.provider._backends["imgfetch"] = imgfetch
    attempt(ctx, Need("rice", "picture"), "openverse")
    first = _outcome(ctx.db, "rice", "picture", "openverse")
    assert first.answer["tried"] == ["https://x/rotted.jpg", "https://x/good.jpg"]
    assert len(first.answer["candidates"]) == 1        # only the stored one

    imgfetch.asked = []                                # observe only the second attempt's asks
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert imgfetch.asked == []                        # the refused url is not retried
    second = _outcome(ctx.db, "rice", "picture", "openverse")
    assert second.answer == {"outcome": "nothing", "candidates": [], "tried": []}


class _WireRefusingImgfetch:
    """Every url is refused on the wire (spec 3 section 6a): no server
    answered, so the refusal is transient, not a served one."""
    def __init__(self):
        self.asked = []

    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        self.asked.append(q.params["url"])
        raise FetchRefused(reason="wire", detail="timeout")


def test_a_wire_refused_url_is_not_tried_and_is_fetched_again_by_a_later_attempt(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/a.jpg",))
    imgfetch = _WireRefusingImgfetch()
    ctx.provider._backends["imgfetch"] = imgfetch
    attempt(ctx, Need("rice", "picture"), "openverse")
    first = _outcome(ctx.db, "rice", "picture", "openverse")
    assert first.answer == {"outcome": "transient-failure", "candidates": [], "tried": []}

    attempt(ctx, Need("rice", "picture"), "openverse")
    assert imgfetch.asked == ["https://x/a.jpg", "https://x/a.jpg"]   # fetched again


class _DeadForvo:
    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        raise TransportError("forvo down")


class _DeadAudiofetch:
    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        raise TransportError("audiofetch refused")


class _DeadTts:
    def synthesize(self, text, voice):
        raise TransportError("tts down")


class _RefusingTts:
    def __init__(self):
        self.calls = 0

    def synthesize(self, text, voice):
        self.calls += 1
        raise SynthesisRefused(f"google tts refused {voice}: 400")


class _PartialTts:
    """Synthesizes the first ask; every later ask fails on the wire --
    the rendition attempt's partial-success case."""
    def __init__(self):
        self.calls = 0

    def synthesize(self, text, voice):
        self.calls += 1
        if self.calls > 1:
            raise TransportError("tts down after the first member")
        return f"{text}-{voice}".encode()


def test_a_forvo_recording_attempt_writes_a_candidates_outcome(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3"}]})   # ข้าว: rice
    attempt(ctx, Need("rice", "recording"), "forvo")
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 1


def test_a_forvo_download_counts_as_a_forvo_request_in_the_attempt_s_spend(tmp_path):
    """Forvo's daily limit counts the mp3 downloads as requests, so the
    attempt tallies each download under forvo as well as audiofetch."""
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3"},
                 {"username": "malee", "pathmp3": "https://f/v.mp3"}]})   # ข้าว: rice
    result = attempt(ctx, Need("rice", "recording"), "forvo")
    assert result.spend["forvo"].asks == 3   # one lookup, two downloads
    assert result.spend["audiofetch"].asks == 2


def test_a_forvo_recording_attempt_writes_a_nothing_outcome_when_forvo_has_nothing(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    attempt(ctx, Need("rice", "recording"), "forvo")
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer == {"outcome": "nothing", "candidates": [], "tried": []}


def test_a_forvo_recording_attempt_writes_transient_failure_when_the_lookup_raises(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    ctx.provider._backends["forvo"] = _DeadForvo()
    with pytest.raises(TransportError):
        attempt(ctx, Need("rice", "recording"), "forvo")
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer == {"outcome": "transient-failure", "candidates": [], "tried": []}


def test_a_forvo_recording_attempt_writes_transient_failure_when_every_download_fails(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3"}]})   # ข้าว: rice
    ctx.provider._backends["audiofetch"] = _DeadAudiofetch()
    result = attempt(ctx, Need("rice", "recording"), "forvo")
    assert result.attempted
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer == {"outcome": "transient-failure", "candidates": [], "tried": []}


class _ExpiringForvo:
    """The first lookup's url is refused by the server; the re-asked
    lookup carries a fresh url for the same item id."""
    def __init__(self):
        self.lookups = 0

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        self.lookups += 1
        return RawAnswer(items=({"id": 7, "username": "somchai", "sex": "m", "country": "Thailand",
                                 "pathmp3": f"https://forvo/audio/{self.lookups}.mp3"},), cost=1.0)


class _RefusingAudiofetch:
    """Refuses every url it has not seen served fresh: the first url of an
    item is refused as content-type, the second answers."""
    def __init__(self, media):
        self.media, self.asked = media, []

    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        self.asked.append(q.params["url"])
        if q.params["url"].endswith("/1.mp3"):
            raise FetchRefused(reason="content-type", detail='content-type "application/json" is not allowed')
        sha = self.media.write(b"ID3fresh", "mp3")
        return RawAnswer(items=({"sha": sha, "ext": "mp3", "speaker": q.params["speaker"],
                                 "speaker_kind": "native", "source": "forvo"},), cost=0.0)


def test_a_served_refusal_of_a_forvo_url_re_asks_the_lookup_once_and_retries(tmp_path):
    forvo = _ExpiringForvo()
    media = MediaStore(tmp_path / "media")
    audiofetch = _RefusingAudiofetch(media)
    ctx = _sourcing(tmp_path, _word_syllabus(),
                    backends={"forvo": forvo, "audiofetch": audiofetch},
                    assess={"mechanical": _mechanical()}, media=media)
    result = attempt(ctx, Need("rice", "recording"), "forvo")
    assert forvo.lookups == 2
    assert audiofetch.asked == ["https://forvo/audio/1.mp3", "https://forvo/audio/2.mp3"]
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 1
    forvo_rows = [r for r in rows_for(ctx.db, "rice", "recording")
                 if r.backend == "forvo" and r.port == "provide"]
    assert len(forvo_rows) == 2


def test_a_second_served_refusal_is_transient_and_re_asks_no_further(tmp_path):
    class _AlwaysRefusing(_RefusingAudiofetch):
        def fetch(self, q):
            self.asked.append(q.params["url"])
            raise FetchRefused(reason="http", detail="http 404")

    forvo = _ExpiringForvo()
    media = MediaStore(tmp_path / "media")
    audiofetch = _AlwaysRefusing(media)
    ctx = _sourcing(tmp_path, _word_syllabus(),
                    backends={"forvo": forvo, "audiofetch": audiofetch},
                    assess={"mechanical": _mechanical()}, media=media)
    attempt(ctx, Need("rice", "recording"), "forvo")
    assert forvo.lookups == 2 and len(audiofetch.asked) == 2
    assert _outcome(ctx.db, "rice", "recording", "forvo").answer["outcome"] == "transient-failure"


def test_a_wire_refusal_re_asks_nothing(tmp_path):
    class _WireDead(_RefusingAudiofetch):
        def fetch(self, q):
            self.asked.append(q.params["url"])
            raise FetchRefused(reason="wire", detail="request failed: timeout")

    forvo = _ExpiringForvo()
    media = MediaStore(tmp_path / "media")
    audiofetch = _WireDead(media)
    ctx = _sourcing(tmp_path, _word_syllabus(),
                    backends={"forvo": forvo, "audiofetch": audiofetch},
                    assess={"mechanical": _mechanical()}, media=media)
    attempt(ctx, Need("rice", "recording"), "forvo")
    assert forvo.lookups == 1 and len(audiofetch.asked) == 1
    assert _outcome(ctx.db, "rice", "recording", "forvo").answer["outcome"] == "transient-failure"


class _NoIdForvo:
    """A lookup whose item carries no Forvo id: nothing a re-ask's item
    can be matched back to."""
    def __init__(self):
        self.lookups = 0

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        self.lookups += 1
        return RawAnswer(items=({"username": "somchai", "sex": "m", "country": "Thailand",
                                 "pathmp3": f"https://forvo/audio/{self.lookups}.mp3"},), cost=1.0)


def test_a_served_refusal_of_an_item_with_no_id_is_not_retried(tmp_path):
    forvo = _NoIdForvo()
    media = MediaStore(tmp_path / "media")
    audiofetch = _RefusingAudiofetch(media)
    ctx = _sourcing(tmp_path, _word_syllabus(),
                    backends={"forvo": forvo, "audiofetch": audiofetch},
                    assess={"mechanical": _mechanical()}, media=media)
    attempt(ctx, Need("rice", "recording"), "forvo")
    # relookup() still runs once (the id-less item can never match); the
    # download itself is not retried
    assert forvo.lookups == 2
    assert audiofetch.asked == ["https://forvo/audio/1.mp3"]
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer["outcome"] == "transient-failure" and row.answer["candidates"] == []


class _TwoItemForvoThenDead:
    """One lookup returns two items; a re-lookup (the second item's served
    refusal triggers one) fails on the wire."""
    def __init__(self):
        self.lookups = 0

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        self.lookups += 1
        if self.lookups > 1:
            raise TransportError("forvo lookup failed")
        return RawAnswer(items=(
            {"id": 1, "username": "a", "sex": "m", "country": "Thailand",
             "pathmp3": "https://forvo/audio/1.mp3"},
            {"id": 2, "username": "b", "sex": "m", "country": "Thailand",
             "pathmp3": "https://forvo/audio/2.mp3"}), cost=1.0)


class _FirstOkSecondRefusedAudiofetch:
    """The first item's url downloads; the second is refused by the
    server (content-type)."""
    def __init__(self, media):
        self.media = media

    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        if q.params["url"].endswith("/2.mp3"):
            raise FetchRefused(reason="content-type",
                               detail='content-type "application/json" is not allowed')
        sha = self.media.write(b"ID3ok", "mp3")
        return RawAnswer(items=({"sha": sha, "ext": "mp3", "speaker": q.params["speaker"],
                                 "speaker_kind": "native", "source": "forvo"},), cost=0.0)


def test_a_failed_relookup_after_a_stored_item_is_a_transient_fetch(tmp_path):
    """A served refusal on the second item re-asks the lookup (spec 3
    section 6a); when that re-lookup itself fails on the wire, the first
    item's stored candidate still stands and the attempt does not raise."""
    forvo = _TwoItemForvoThenDead()
    media = MediaStore(tmp_path / "media")
    audiofetch = _FirstOkSecondRefusedAudiofetch(media)
    ctx = _sourcing(tmp_path, _word_syllabus(),
                    backends={"forvo": forvo, "audiofetch": audiofetch},
                    assess={"mechanical": _mechanical()}, media=media)
    result = attempt(ctx, Need("rice", "recording"), "forvo")
    assert result.attempted
    assert forvo.lookups == 2
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 1


class _QuotaOnRelookupForvo:
    """The first lookup succeeds; the re-lookup a served refusal triggers
    hits Forvo's own daily quota (spec 3 section 6a) instead of a fresh
    item -- the live failure: audiofetch refused Forvo's JSON error body,
    the re-lookup itself answered 400."""
    def __init__(self):
        self.lookups = 0

    def cache_key(self, q):
        return ProvideKey(source="forvo", kind="", query=q.params["word"])

    def fetch(self, q):
        self.lookups += 1
        if self.lookups > 1:
            raise QuotaExhausted("forvo")
        return RawAnswer(items=({"id": 7, "username": "somchai", "sex": "m",
                                 "country": "Thailand",
                                 "pathmp3": "https://forvo/audio/1.mp3"},), cost=1.0)


def test_a_quota_exhausted_relookup_appends_no_row_and_reaches_the_caller(tmp_path):
    """A served refusal of the first download re-asks the lookup (spec 3
    section 6a's re-ask rule); when that re-lookup hits Forvo's own quota
    instead of answering fresh, it is not a transient fetch failure --
    no outcome row is appended and QuotaExhausted reaches attempt()'s own
    caller unchanged, the same contract a quota hit on the first lookup
    has."""
    forvo = _QuotaOnRelookupForvo()
    media = MediaStore(tmp_path / "media")
    audiofetch = _RefusingAudiofetch(media)
    ctx = _sourcing(tmp_path, _word_syllabus(),
                    backends={"forvo": forvo, "audiofetch": audiofetch},
                    assess={"mechanical": _mechanical()}, media=media)
    with pytest.raises(QuotaExhausted):
        attempt(ctx, Need("rice", "recording"), "forvo")
    assert forvo.lookups == 2
    assert not [r for r in rows_for(ctx.db, "rice", "recording") if r.port == "attempt"]


def test_a_tts_recording_attempt_writes_a_candidates_outcome(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    attempt(ctx, Need("rice", "recording"), "tts")
    row = _outcome(ctx.db, "rice", "recording", "tts")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 1


def test_a_tts_recording_attempt_writes_transient_failure_when_synthesis_raises(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    ctx.provider._backends["tts"] = TtsBackend(
        tts=_DeadTts(), voices=list(_MALE) + list(_FEMALE), media=ctx.media_store,
        pick_voice=pick_voice)
    with pytest.raises(TransportError):
        attempt(ctx, Need("rice", "recording"), "tts")
    row = _outcome(ctx.db, "rice", "recording", "tts")
    assert row.answer == {"outcome": "transient-failure", "candidates": [], "tried": []}


def test_a_tts_refusal_is_a_nothing_outcome_not_a_transient_one(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    ctx.provider._backends["tts"] = TtsBackend(
        tts=_RefusingTts(), voices=list(_MALE) + list(_FEMALE), media=ctx.media_store,
        pick_voice=pick_voice)
    result = attempt(ctx, Need("rice", "recording"), "tts")
    assert result.attempted
    row = _outcome(ctx.db, "rice", "recording", "tts")
    assert row.answer == {"outcome": "nothing", "candidates": [], "tried": []}


def test_a_rendition_attempt_writes_a_candidates_outcome(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]})  # ข่าว: news
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    row = _outcome(ctx.db, "p1", "rendition", "forvo")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 2


def test_a_rendition_attempt_writes_a_nothing_outcome_with_no_shared_speaker(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "malee", "pathmp3": "https://f/b.mp3"}]})    # ข่าว: news
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    row = _outcome(ctx.db, "p1", "rendition", "forvo")
    assert row.answer == {"outcome": "nothing", "candidates": [], "tried": []}


def test_a_rendition_attempt_writes_transient_failure_when_a_members_lookup_raises(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus())
    ctx.provider._backends["forvo"] = _DeadForvo()
    with pytest.raises(TransportError):
        attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    row = _outcome(ctx.db, "p1", "rendition", "forvo")
    assert row.answer == {"outcome": "transient-failure", "candidates": [], "tried": []}


def test_a_rendition_tts_attempt_reports_candidates_from_a_partial_success_then_reraises(tmp_path):
    """The first member's synthesis stores a candidate; the second's
    fails on the wire. The outcome row reads `candidates` and the call
    raises.
    """
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus())
    ctx.provider._backends["tts"] = TtsBackend(
        tts=_PartialTts(), voices=list(_MALE) + list(_FEMALE), media=ctx.media_store,
        pick_voice=pick_voice)
    with pytest.raises(TransportError):
        attempt(ctx, Need("p1", "rendition", "pair"), "tts")
    row = _outcome(ctx.db, "p1", "rendition", "tts")
    assert row.answer["outcome"] == "candidates"
    assert len(row.answer["candidates"]) == 1


def test_a_rendition_tts_attempt_stops_at_the_first_member_the_service_refuses(tmp_path):
    """The first member's synthesis is refused. The second member is
    never asked, and the outcome row reads `nothing`.
    """
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus())
    refusing = _RefusingTts()
    ctx.provider._backends["tts"] = TtsBackend(
        tts=refusing, voices=list(_MALE) + list(_FEMALE), media=ctx.media_store,
        pick_voice=pick_voice)
    result = attempt(ctx, Need("p1", "rendition", "pair"), "tts")
    assert result.attempted
    assert refusing.calls == 1
    row = _outcome(ctx.db, "p1", "rendition", "tts")
    assert row.answer == {"outcome": "nothing", "candidates": [], "tried": []}


# --- ruling 1: the outcome row is written before the check call ------------

class _DeadMechanical:
    def cache_key(self, q):
        return MechanicalKey(check="duration", params="0.2-5.0", artifact_sha=q.artifact_sha or "-")

    def fetch(self, q):
        raise TransportError("mechanical check unreachable")


def test_a_picture_attempt_writes_its_outcome_row_when_the_judge_is_unreachable(tmp_path):
    def boom(prompt, attachments=()):
        raise TransportError("api transport failed")
    ctx, _search, _judge = _picture_ctx(tmp_path, judge=boom, urls=("https://x/good.jpg",))
    with pytest.raises(JudgeUnreachable):
        attempt(ctx, Need("rice", "picture"), "openverse")
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer["outcome"] == "candidates"


def test_a_forvo_recording_attempt_writes_its_outcome_row_when_the_check_is_unreachable(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3"}]},   # ข้าว: rice
        mechanical=_DeadMechanical())
    with pytest.raises(JudgeUnreachable):
        attempt(ctx, Need("rice", "recording"), "forvo")
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer["outcome"] == "candidates"


def test_a_rendition_attempt_writes_its_outcome_row_when_the_check_is_unreachable(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]},  # ข่าว: news
        mechanical=_DeadMechanical())
    with pytest.raises(JudgeUnreachable):
        attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    row = _outcome(ctx.db, "p1", "rendition", "forvo")
    assert row.answer["outcome"] == "candidates"


def test_sentence_drafts_reads_back_every_draft_the_run_asked_for(tmp_path):
    ctx = _sentence_ctx(tmp_path, _draft_json(("rice",), "ข้าว", "eat rice"))   # ข้าว: rice
    sentence_attempt(ctx)
    drafts = sentence_drafts(ctx.db)
    assert [(d.text, d.gloss, d.clauses) for d in drafts] == [
        ("ข้าว", "eat rice", ((WordId("rice"),),))]   # ข้าว: rice
    assert rows_for(ctx.db, DRAFT_SUBJECT, "sentence")


class _ProseTransport:
    """A drafter transport that answers with prose, never the drafting
    prompt's JSON -- what LlmBackend.recognize (wired to drafts_in) rejects
    (spec 3 r10 section 2)."""

    def complete(self, prompt):
        return Completion(text="I can't think of a sentence right now.")


def test_sentence_attempt_raises_on_a_drafter_answer_that_carries_no_draft(tmp_path):
    syllabus = Syllabus(
        words=(word("rice", "ข้าว", "rice"), word("eat", "กิน", "eat")),   # ข้าว: rice, กิน: eat
        targets=(target("eat/receptive", "eat"), target("rice/receptive", "rice")),
        frequency={"eat": 1, "rice": 2})
    llm = LlmBackend(producer="sentence-drafter", model="m", transport=_ProseTransport(),
                     recognize=lambda text: bool(drafts_in(text)))
    ctx = _sourcing(tmp_path, syllabus, backends={"llm-sentence": llm},
                    assess={"judge": JudgeBackend(model="m", transport="api",
                                                  complete=lambda p, a=(): Completion(
                                                      text='{"value": true, "evidence": "e"}'))})
    with pytest.raises(TransportError, match="recognizable answer"):
        sentence_attempt(ctx)
    assert not rows_for(ctx.db, DRAFT_SUBJECT, "sentence")


# --- _sentence_prompt: the vocabulary listed once, a cutoff per target -----

def _three_word_syllabus():
    # กิน: eat, ข้าว: rice, อร่อย: tasty; entry order by frequency: eat, rice, tasty
    return Syllabus(
        words=(word("eat", "กิน", "eat"), word("rice", "ข้าว", "rice"),
               word("tasty", "อร่อย", "tasty")),
        targets=(target("tasty/receptive", "tasty"), target("rice/receptive", "rice"),
                 target("eat/receptive", "eat")),      # list order differs from entry order
        frequency={"eat": 1, "rice": 2, "tasty": 3})


def test_sentence_prompt_lists_each_vocabulary_word_once_in_entry_order():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets))
    vocabulary = prompt.split("Vocabulary, in the order met:\n")[1].split("\nTargets:")[0]
    assert vocabulary.splitlines() == ["- eat  กิน  (eat)", "- rice  ข้าว  (rice)",
                                       "- tasty  อร่อย  (tasty)"]
    assert prompt.count("อร่อย") == 2          # once in the list, once on its own target line


def test_sentence_prompt_lists_a_targets_line_per_handed_target():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets))
    assert "- target eat/receptive: eat  กิน  (eat)" in prompt
    assert "- target rice/receptive: rice  ข้าว  (rice)" in prompt
    assert "- target tasty/receptive: tasty  อร่อย  (tasty)" in prompt


def test_sentence_prompt_gives_the_required_covering_instruction_verbatim():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets))
    assert ("Each JSON item is one sentence. Write the fewest natural sentences that together "
           "cover the targets below; a sentence may cover several targets. A sentence may "
           "introduce at most one word from the Introducible list and must otherwise use only "
           "the vocabulary below.") in prompt


def test_sentence_prompt_gives_the_clause_rendering_rule_and_json_shape_verbatim():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets))
    assert ("Write each sentence as clauses of vocabulary ids in order; a clause renders as "
           "its words' Thai concatenated, clauses are separated by one space; write a repeated "
           'word as [id, "ๆ"]; standard spelling (ครับ, never คับ); numbers as number words; '
           "no punctuation or digits.") in prompt
    assert ('Output JSON only: {"sentences": [{"clauses": [["id", ...], ...], "text": "...", '
           '"gloss": "..."}]}') in prompt


def _glue_word_syllabus():
    # กิน: eat, ข้าว: rice -- picture-introduced; แล้ว: already, ก็: also --
    # sentence-introduced glue words; ก็ already met via a sentence that
    # fills ก็/receptive (its only adopted sentence, so nothing else meets
    # แล้ว/receptive).
    return Syllabus(
        words=(word("eat", "กิน", "eat"), word("rice", "ข้าว", "rice"),
               word("glue1", "แล้ว", "already"), word("glue2", "ก็", "also")),
        targets=(target("eat/receptive", "eat"),
                 target("rice/receptive", "rice"),
                 target("glue1/receptive", "glue1", introduction="sentence"),
                 target("glue2/receptive", "glue2", introduction="sentence")),
        frequency={"eat": 1, "rice": 2, "glue1": 3, "glue2": 4},
        sentences=(_sentence(text="ก็กิน", gloss="also eats",   # ก็กิน: also eats
                             clauses=((WordId("glue2"), WordId("eat")),)),))


def test_sentence_prompt_omits_an_unmet_glue_word_from_vocabulary_and_lists_it_introducible():
    syllabus = _glue_word_syllabus()
    glue1, glue2 = (t for t in syllabus.targets if t.id in ("glue1/receptive", "glue2/receptive"))
    prompt = _sentence_prompt(syllabus, [glue1, glue2])
    vocabulary = prompt.split("Vocabulary, in the order met:\n")[1].split("\nIntroducible")[0]
    assert "แล้ว" not in vocabulary   # แล้ว: already -- unmet, left out of vocabulary
    introducible = prompt.split("Introducible (at most one per sentence):\n")[1]
    assert "- target glue1/receptive: glue1  แล้ว  (already)" in prompt
    assert "glue1" in introducible.splitlines()[0]   # the word id appears on the introducible line
    assert "Introducible (at most one per sentence):" in prompt


def test_sentence_prompt_shows_a_target_an_adopted_sentence_fills_as_a_targets_line():
    """A handed sentence-introduced Target some adopted sentence already
    fills (spec 3 r15 section 5) is an ordinary Targets line -- present,
    not dropped, and not Introducible."""
    syllabus = _glue_word_syllabus()
    glue2 = next(t for t in syllabus.targets if t.id == "glue2/receptive")
    prompt = _sentence_prompt(syllabus, [glue2])
    vocabulary = prompt.split("Vocabulary, in the order met:\n")[1].split("\nTargets:")[0]
    assert "ก็" in vocabulary   # ก็: also -- met by the adopted sentence
    assert "- target glue2/receptive: glue2  ก็  (also)" in prompt
    assert "Introducible (at most one per sentence):" not in prompt


def test_sentence_prompt_appends_a_met_glue_word_whose_target_lies_beyond_the_handed_batch():
    """ก็/receptive's own entry sits after แล้ว/receptive's (the only
    handed target's) own entry; the bounded walk stops at the furthest
    handed target, so a second pass over the whole order appends a met
    sentence-introduced word wherever its own entry falls."""
    syllabus = _glue_word_syllabus()
    glue1 = next(t for t in syllabus.targets if t.id == "glue1/receptive")
    prompt = _sentence_prompt(syllabus, [glue1])
    vocabulary = prompt.split("Vocabulary, in the order met:\n")[1].split("\nIntroducible")[0]
    assert "ก็" in vocabulary   # ก็: also -- appended though its own entry is beyond the batch


def test_sentence_prompt_shows_a_picture_introduced_target_though_its_word_is_already_met():
    """A picture-introduced Target is always a Targets line: a met
    sentence-introduced Target of the same word plays no part in it."""
    met_sentence = Sentence(
        clauses=((WordId("help"),),), text="ช่วย", gloss="a person helps",   # ช่วย: help
        voice="other_voice",
        provenance=Provenance(source="llm", origin="m", licence="generated",
                              acquired=date(2026, 9, 3)))
    syllabus = Syllabus(
        words=(word("help", "ช่วย", "help"),),   # ช่วย: help
        targets=(target("help/receptive", "help", introduction="sentence"),
                 target("help/productive", "help", skill="productive")),
        frequency={"help": 1},
        sentences=(met_sentence,))
    productive = next(t for t in syllabus.targets if t.id == "help/productive")
    prompt = _sentence_prompt(syllabus, [productive])
    assert "- target help/productive: help  ช่วย  (help)" in prompt


def test_sentence_prompt_lists_only_the_vocabulary_the_handed_targets_met():
    syllabus = _three_word_syllabus()
    first_only = [t for t in syllabus.targets if t.id == "eat/receptive"]
    vocabulary = _sentence_prompt(syllabus, first_only).split(
        "Vocabulary, in the order met:\n")[1].split("\nTargets:")[0]
    assert vocabulary.splitlines() == ["- eat  กิน  (eat)"]
