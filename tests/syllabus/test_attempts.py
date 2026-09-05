"""attempts.py: one Source asked under the need's own subject, what it
returns ingested, the speaker recorded, and the judge questions collected.
Real SyllabusDb + MediaStore; fake Provide/Assess backends; no network."""
import hashlib
import io
from datetime import date

import pytest
from PIL import Image as PILImage

from thai_syllabus.assessor import (Assessor, JudgeBackend, JudgeUnreachable, MechanicalBackend,
                                    RawVerdict, fills_mechanical_backend,
                                    rendition_mechanical_backend)
from thai_syllabus.attempts import (AttemptResult, Need, Sourcing, attempt, current_best_of,
                                    sentence_attempt, sources_for)
from thai_syllabus.cachekeys import rendition_identity
from thai_syllabus.derivations import exhausted
from thai_syllabus.record import DRAFT_SUBJECT, sentence_drafts
from thai_syllabus.entities import Category, MinimalPair, Sentence, SoundConfusion, text_sha
from thai_syllabus.media import Provenance, Speaker
from thai_syllabus.provider import FetchBackend, Provider, RawAnswer, TtsBackend
from thai_syllabus.record import rows_for
from thai_syllabus.rulebook import (PICTURE_FIT_RUBRIC, PICTURE_PREFERENCE_RUBRIC,
                                    SENTENCE_FOR_TARGET_RUBRIC)
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.transport import Completion, TransportError
from thai_syllabus.tts import pick_voice

from .builders import target, word
from .fakes import FakeTokenizer

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
        return f"search:{q.params['query']}"

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
        return f"forvo:{q.params['word']}"

    def fetch(self, q):
        return RawAnswer(items=tuple(self.items_by_word.get(q.params["word"], ())), cost=1.0)


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
        return "llm:sentence-drafter:m:" + hashlib.sha256(
            q.params["prompt"].encode()).hexdigest()[:16]

    def fetch(self, q):
        self.prompts.append(q.params["prompt"])
        return RawAnswer(items=(self.text,))


def _mechanical(ok=True, failing_subject=None):
    def key_fn(q):
        return f"mech:duration:0.2-5.0:{q.artifact_sha}"

    def evaluate(q):
        passes = ok and q.subject != failing_subject
        return RawVerdict(value=passes, evidence="duration=1.0s" if passes else "too short")
    return MechanicalBackend(key_fn=key_fn, evaluate=evaluate)


def _rendition_backend(db):
    def speaker_of(sha):
        prov = db.media_provenance(sha)
        return prov.get("speaker_id") if prov else None
    return rendition_mechanical_backend(speaker_of=speaker_of)


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
                    categories=(Category(name="Food", members=frozenset({"rice"})),),
                    tokenizer=FakeTokenizer())


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
                                       members=("white", "news")),),
                    tokenizer=FakeTokenizer())


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
    ctx.db.append(port="assess", backend="judge", key="k1", subject="rice",
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
    ctx.db.append(port="provide", backend="openverse", key="legacy", subject="rice",
                  question={"provides": "picture", "kind": "picture", "params": {}},
                  answer={"items": [{"sha": "ghost"}]})
    res = attempt(ctx, Need("rice", "picture"), "openverse")
    assert list(res.excluded.values()) == ["artifact not found: ghost"]
    assert current_best_of(ctx, "rice", "picture").artifact_sha != "ghost"


# --- scene picture: the same attempt, subject = text_sha --------------------

def _sentence(text="ข้าวอร่อย", gloss="the rice is tasty") -> Sentence:  # ข้าวอร่อย: tasty rice
    return Sentence(text=text, gloss=gloss, voice="learner_voice",
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


def test_a_forvo_attempt_that_found_nothing_is_still_on_the_record(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    res = attempt(ctx, Need("rice", "recording"), "forvo")
    assert res.attempted
    assert exhausted(ctx.db, "rice", "recording", sources=("forvo", "tts"),
                     attempt_cap=8).attempts == 1


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
                        targets=(target("rice/productive", "rice", skill="productive"),),
                        tokenizer=FakeTokenizer()).with_sentences([sentence])
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
    assert exhausted(ctx.db, "p1", "rendition", sources=("forvo",), attempt_cap=8).attempts == 1


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


def test_rendition_attempt_falls_to_one_tts_voice_across_the_members(tmp_path):
    ctx, tts = _recording_ctx(tmp_path, _pair_syllabus())
    attempt(ctx, Need("p1", "rendition", "pair"), "tts")
    assert set(tts.voices) == {pick_voice("p1", list(_MALE) + list(_FEMALE))}
    provided = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"]
    assert {i["speaker"]["kind"] for i in provided[-1].answer["items"]} == {"synthetic"}


# --- the sentence attempt: draft, verify with fills(), collect questions ----

def _sentence_ctx(tmp_path, llm_text, *, judge_value="true", batch=False):
    syllabus = Syllabus(
        words=(word("rice", "ข้าว", "rice"), word("eat", "กิน", "eat")),   # ข้าว: rice, กิน: eat
        targets=(target("eat/receptive", "eat"), target("rice/receptive", "rice")),
        frequency={"eat": 1, "rice": 2},
        tokenizer=FakeTokenizer({"กินข้าว": ["กิน", "ข้าว"],      # กินข้าว: eat rice
                                 "ข้าวอร่อย": ["ข้าว", "อร่อย"],   # ข้าวอร่อย: tasty rice
                                 "กิน": ["กิน"]}))                # กิน: eat
    judge = (_batch_judge() if batch else JudgeBackend(
        model="m", transport="api",
        complete=lambda p, a=(): Completion(text='{"value": %s, "evidence": "e"}' % judge_value)))
    holder = []
    ctx = _sourcing(tmp_path, syllabus, backends={"llm-sentence": _Llm(llm_text)},
                    assess={"judge": judge,
                            "fills": fills_mechanical_backend(lambda: holder[0].syllabus)})
    holder.append(ctx)
    return ctx


def test_sentence_attempt_collects_a_judge_question_carrying_the_text_and_gloss(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'
                                  ' "targets": ["rice/receptive", "eat/receptive"]}]}',
                        batch=True)
    res = sentence_attempt(ctx)
    assert res.attempted and len(res.questions) == 1
    question = res.questions[0].question
    assert question.role == "sentence-for-target"
    assert question.params["text"] == "กินข้าว" and question.params["gloss"] == "eat rice"
    assert question.artifact_sha is None       # a sentence judgment attaches no artifact


def test_sentence_attempt_reports_the_drafts_it_produced(tmp_path):
    """`drafted` counts the drafts that fill an open Target, whatever the
    transport did with their judge questions."""
    text = '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'\
           ' "targets": ["rice/receptive", "eat/receptive"]}]}'   # กินข้าว: eat rice
    # one deck per call: the drafting ask is cached, and a shared db would
    # hand the later calls the first call's answer.
    assert sentence_attempt(_sentence_ctx(tmp_path / "batch", text, batch=True)).drafted == 1
    assert sentence_attempt(_sentence_ctx(tmp_path / "inline", text)).drafted == 1
    assert sentence_attempt(_sentence_ctx(tmp_path / "none", '{"sentences": []}')).drafted == 0


def test_sentence_attempt_reports_how_many_open_targets_it_was_handed(tmp_path):
    """`targets_handed` is min(open Targets, max_targets) -- the per-run
    cap run.py needs to tell "handed" apart from "left for another run".
    """
    text = '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'\
           ' "targets": ["rice/receptive", "eat/receptive"]}]}'   # กินข้าว: eat rice
    ctx = _sentence_ctx(tmp_path / "uncapped", text)
    assert sentence_attempt(ctx).targets_handed == 2   # both open Targets, well under the cap
    ctx = _sentence_ctx(tmp_path / "capped", text)
    assert sentence_attempt(ctx, max_targets=1).targets_handed == 1


def test_sentence_attempt_adopts_nothing_itself(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'
                                  ' "targets": ["rice/receptive", "eat/receptive"]}]}')
    res = sentence_attempt(ctx)
    assert res.questions == []                 # the inline judge answered
    assert ctx.db.all_sentences() == []        # ...and the run, not the attempt, adopts


def test_sentence_attempt_records_a_fills_verdict_per_claimed_target(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'
                                  ' "targets": ["rice/receptive", "eat/receptive"]}]}')
    sentence_attempt(ctx)
    fills = [r for r in ctx.db.assessments_of(text_sha("กินข้าว"))   # กินข้าว: eat rice
             if r.backend == "fills"]
    assert {r.answer["value"] for r in fills} == {True}
    assert len(fills) == 2


def test_sentence_attempt_does_not_judge_a_draft_that_fills_nothing(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": [{"text": "ข้าวอร่อย", "gloss": "tasty rice",'
                                  ' "targets": ["rice/receptive"]}]}', batch=True)
    res = sentence_attempt(ctx)
    assert res.questions == []                 # อร่อย (tasty) is not a registered word
    fills = [r for r in ctx.db.assessments_of(text_sha("ข้าวอร่อย"))   # ข้าวอร่อย: tasty rice
             if r.backend == "fills"]
    assert fills and fills[0].answer["value"] is False


def test_sentence_attempt_is_not_attempted_when_no_target_is_open(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": []}')
    ctx.syllabus = ctx.syllabus.with_sentences(   # กินข้าว: eat rice
        [_sentence(text="กินข้าว", gloss="eat rice")])
    res = sentence_attempt(ctx)
    assert res == AttemptResult(attempted=False)
    assert ctx.provider._backends["llm-sentence"].prompts == []


def test_the_drafting_prompt_carries_the_vocabulary_met_and_asks_for_a_gloss(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": []}')
    sentence_attempt(ctx)
    prompt = ctx.provider._backends["llm-sentence"].prompts[0]
    assert "target rice/receptive" in prompt and "กิน" in prompt   # กิน: eat
    assert "male_colloquial" in prompt and '"gloss"' in prompt


def test_sentence_drafts_reads_back_every_draft_the_run_asked_for(tmp_path):
    ctx = _sentence_ctx(tmp_path, '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'
                                  ' "targets": ["rice/receptive"]}]}')
    sentence_attempt(ctx)
    drafts = sentence_drafts(ctx.db)
    assert [(d.text, d.gloss, d.claimed) for d in drafts] == [
        ("กินข้าว", "eat rice", ("rice/receptive",))]   # กินข้าว: eat rice
    assert rows_for(ctx.db, DRAFT_SUBJECT, "sentence")
