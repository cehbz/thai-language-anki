"""attempts.py: one Source asked under the need's own subject, what it
returns ingested, the speaker recorded, and the judge questions collected.
Real SyllabusDb + MediaStore; fake Provide/Assess backends; no network."""
import hashlib
import io
import json
import logging
from dataclasses import replace
from datetime import date

import pytest
from PIL import Image as PILImage

from thai_syllabus.assessor import (UNTRUSTED, Assessor, Excluded, JudgeBackend, JudgeUnreachable,
                                    ManyResult, RawVerdict, RenditionBackend, deck_field)
from thai_syllabus.attempts import (COMMENTS_PER_ASK, GRAPHEME_NAME_MEANING, AttemptResult,
                                    ChartCell, Need, Sourcing,
                                    _picture_params, _pool, _sentence_prompt,
                                    adjudication_attempt, assess_first, attempt, chart_cell,
                                    comment_attempt, current_best_of, grapheme_attempt,
                                    pair_search_attempt, phrase_attempt,
                                    picture_query_for, retire_sentence, sentence_attempt,
                                    sources_for, sources_for_need)
from thai_syllabus.cachekeys import (AttemptOutcomeKey, DirectionKey, JudgeKey, LlmPromptKey,
                                    MechanicalKey, PhraseKey, ProvideKey, RunReportKey,
                                    rendition_identity, sha)
from thai_syllabus.derivations import CANDIDATE_SUBJECT_PREFIX, attempts_since_change, exhausted
from thai_syllabus.learner import CommentRef, append_comment, append_direction
from thai_syllabus.record import (DRAFT_SUBJECT, QUERY_FORMS, DraftedQuery, candidate_shas,
                                  comments, drafted_phrase, drafted_queries, drafts_in,
                                  gloss_on_requested, latest_phrase, parse_phrases,
                                  reading_of, retired_texts, retirements, rows_for,
                                  sentence_drafts)
from thai_syllabus import attempts as attempts_module
from thai_syllabus import curated as curated_module
from thai_syllabus.curated import (build_categories, load_graphemes, load_pairs, load_targets,
                                   load_words, save_confusions, save_graphemes, save_pairs,
                                   save_targets, save_words)
from thai_syllabus.entities import (Category, Clauses, Grapheme, MinimalPair, Sentence,
                                    SoundConfusion, Syllable, text_sha)
from thai_syllabus.ids import ConfusionId, WordId
from thai_syllabus.inventory import ConsonantRow
from thai_syllabus.phonology import Engines
from thai_syllabus.media import Provenance, Speaker
from thai_syllabus.provider import FetchBackend, LlmBackend, Provider, RawAnswer, TtsBackend
from thai_syllabus.rulebook import (PICTURE_FIT_RUBRIC, PICTURE_PREFERENCE_RUBRIC,
                                    SENTENCE_FOR_TARGET_RUBRIC)
from thai_syllabus.run import _Tally
from thai_syllabus.safety import Guard
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.transport import (Completion, FetchRefused, QuotaExhausted, SynthesisRefused,
                                     TransportError)
from thai_syllabus.tts import pick_voice

from .builders import sentence as compose_sentence
from .builders import syl, target, thai_of, word
from .fakes import FakeMediaIndex

# This fixture's own role -> rubric map (rulebook.rubrics_for covers only
# roles a judged Rule registers).
_RUBRICS = {"picture-for-word": PICTURE_FIT_RUBRIC,
            "picture-preference": PICTURE_PREFERENCE_RUBRIC,
            "scene-for-sentence": PICTURE_FIT_RUBRIC,
            "sentence-for-target": SENTENCE_FOR_TARGET_RUBRIC,
            "pronunciation-for-word": "R"}

_MALE = ("th-M-a", "th-M-b")
_FEMALE = ("th-F-a", "th-F-b")


# --- fake backends ----------------------------------------------------------

class _Search:
    """Records every query it was asked -- and the whole params mapping,
    which spec 3 r41 section 5 keeps bare for an ordinary source -- and
    answers one hit per url."""
    def __init__(self, urls):
        self.urls, self.queries = list(urls), []
        self.asked: list[dict] = []

    def cache_key(self, q):
        return ProvideKey(source="openverse", kind="", query=q.params["query"])

    def fetch(self, q):
        self.queries.append(q.params["query"])
        self.asked.append(dict(q.params))
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


def _png_bytes(colour=(10, 200, 10)) -> bytes:
    """A decodable PNG, the bytes a drawn chart cell arrives as."""
    buf = io.BytesIO()
    PILImage.new("RGB", (4, 4), colour).save(buf, format="PNG")
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
        # An item that names no `word` is one Forvo recorded as the asked
        # form -- the fixtures' default; a test about a tone sibling sets it.
        items = tuple({**i, "word": i.get("word", q.params["word"])}
                      for i in self.items_by_word.get(q.params["word"], ()))
        return RawAnswer(items=items, cost=1.0)


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
        return MechanicalKey(check="duration", params="0.2-5.0", subject=q.subject,
                             artifact_sha=q.artifact_sha)

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


def _batch_judge(resolve_path=None):
    class _NeverSubmits:
        def submit(self, requests):
            raise AssertionError("ask_many must not submit a batch")
    return JudgeBackend(model="m", transport="batch", batch_transport=_NeverSubmits(),
                        resolve_path=resolve_path)


# --- contexts ---------------------------------------------------------------

def _sourcing(tmp_path, syllabus, *, backends, assess, media=None) -> Sourcing:
    db = SyllabusDb(tmp_path / "syllabus.db")
    return Sourcing(syllabus=syllabus, provider=Provider(record=db, cache=db, backends=backends),
                    assessor=Assessor(record=db, cache=db, backends=assess), db=db,
                    media_store=media or MediaStore(tmp_path / "media"), rubrics=dict(_RUBRICS),
                    provenance_prior=("commission", "forvo", "tts"), image_candidates=3,
                    today=lambda: date(2026, 9, 3),
                    voices={"male": _MALE, "female": _FEMALE})


def _word_syllabus(*, productive=False) -> Syllabus:
    skill = "productive" if productive else "receptive"
    return Syllabus(words=(word("rice", "ข้าว", "rice (cooked)"),),   # ข้าว: rice
                    targets=(target(f"rice/{skill}", "rice", skill=skill),),
                    categories=(Category(name="Food", members=frozenset({"rice"})),))


def _picture_ctx(tmp_path, syllabus=None, *, judge=None, urls=("https://x/bad.jpg",
                                                               "https://x/good.jpg",
                                                               "https://x/good2.jpg"),
                 phrase="rice food"):
    """A picture-sourcing ctx over one word, rice, with `phrase` on record
    as its drafted search phrase (spec 3 r25 section 5: a need with no
    query on record is not searched) -- None seeds no phrase."""
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
    if phrase is not None:
        _seed_phrase(ctx, "rice", phrase)
    return ctx, search, complete


def _seed_phrase(ctx, subject, phrase, subject_kind="word"):
    ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject=subject), subject=subject,
                  question={"provides": "phrase", "kind": "picture", "subject_kind": subject_kind},
                  answer={"phrase": phrase})


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
    with pytest.raises(ValueError, match="pronunciation"):
        attempt(ctx, Need("rice", "pronunciation"), "llm")


# --- picture: the query, the ingest, the fit questions ----------------------

def test_picture_sources_are_asked_pexels_first(tmp_path):
    """Spec 3 r26 section 5: pexels, openverse, wikimedia -- the keyed
    corpus first, the challenge-prone anonymous-tier corpus second; brave,
    the metered paid search, last of the corpora; and the illustrator
    (spec 3 r34) after brave, the answer of last resort."""
    assert sources_for("picture") == ("pexels", "openverse", "wikimedia", "brave", "illustrator")


def test_the_illustrator_is_the_last_picture_source_after_brave(tmp_path):
    """One source per need per run (spec 3 r26 `next_source`): brave is
    asked only for a need the three free corpora have all failed, and the
    illustrator (spec 3 r34) only once every corpus is tried -- a drawn
    picture is the answer of last resort."""
    assert sources_for("picture")[-2:] == ("brave", "illustrator")


def test_a_picture_need_with_nothing_on_record_has_no_query(tmp_path):
    """Spec 3 r25 section 5: no direction, no suggestion, no drafted
    phrase -- no query. The gloss is the drafter's input, never a
    search."""
    ctx, _search, _judge = _picture_ctx(tmp_path, phrase=None)
    assert picture_query_for(ctx, Need("rice", "picture")) is None


def test_a_picture_attempt_refuses_a_need_with_no_query(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, phrase=None)
    with pytest.raises(ValueError, match="rice"):
        attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == []


def test_picture_attempt_searches_a_judge_suggestion_once_one_is_on_record(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path)
    ctx.db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(None, None, "rice", "picture-for-word"), subject="rice",
                  question={"role": "picture-for-word", "kind": "picture"},
                  answer={"value": False, "suggestion": "a bowl of steamed jasmine rice"})
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == ["a bowl of steamed jasmine rice"]


def test_picture_attempt_searches_the_drafted_phrase_when_one_is_on_record(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path)
    ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject="rice"), subject="rice",
                  question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                  answer={"phrase": "a bowl of steamed rice"})
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == ["a bowl of steamed rice"]


def test_a_learner_direction_still_outranks_the_drafted_phrase(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path)
    ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject="rice"), subject="rice",
                  question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                  answer={"phrase": "a bowl of steamed rice"})
    ctx.db.append(port="assess", backend="learner",
                  key=DirectionKey(subject="rice", role="picture-for-word", text_sha=sha("try red")),
                  subject="rice", question={"kind": "direction", "role": "picture-for-word"},
                  answer={"direction": "try red"})
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == ["try red"]


def test_picture_attempt_ingests_each_hit_with_its_provenance(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    res = attempt(ctx, Need("rice", "picture"), "openverse")
    shas = [i["sha"] for r in rows_for(ctx.db, "rice", "picture") if r.port == "provide"
            for i in r.answer.get("items", ()) if "sha" in i]
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


def test_assess_first_hands_the_judge_no_search_phrase(tmp_path):
    """A candidate already on record was not found by this attempt's
    query, so its fit question names no phrase -- even with a drafted
    phrase on record (the rubric's "pass if no phrase is given" applies)."""
    ctx, _search, _judge = _picture_ctx(tmp_path)
    _seed_current_picture(ctx, "rice")
    ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject="rice"), subject="rice",
                  question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                  answer={"phrase": "a bowl of steamed rice"})
    assess_first(ctx, Need("rice", "picture"))
    verdicts = [r for r in rows_for(ctx.db, "rice", "picture")
                if r.port == "assess" and r.backend == "judge"]
    assert len(verdicts) == 1
    assert verdicts[0].question["params"]["phrase"] is None


def test_assess_first_returns_the_exclusion_when_every_waiting_candidate_is_excluded(tmp_path):
    ctx, _search, judge = _picture_ctx(tmp_path)
    ctx.db.append(port="provide", backend="legacy-current",
                  key=ProvideKey(source="legacy-current", kind="picture", query="rice"),
                  subject="rice",
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"image": "images/pw-1.jpg"}},
                  answer={"items": [{"sha": "0" * 64, "ext": "jpg"}]})  # bytes never stored
    res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None and not res.attempted and res.questions == []
    assert [e.artifact_sha for e in res.excluded.values()] == ["0" * 64]
    assert judge.calls == []
    # Minor fix: the unattempted result is dataclasses.replace(result,
    # attempted=False), not a bare AttemptResult(attempted=False,
    # excluded=...) -- every other field of the assess step's own result
    # (spend included) must survive onto it. Preparation fails here
    # before the judge backend is ever invoked (judge.calls == [] above),
    # so this fixture's own assess step never accrues judge spend to
    # begin with; spend == {} either way, so this only pins the
    # replace(...)-shape (same excluded, attempted False, questions []),
    # not spend surviving a nonzero value.
    assert res.spend == {}
    assert res.drafted == 0 and res.targets_handed == 0 and res.subjects_handed == frozenset()


def test_assess_first_logs_the_candidates_it_excluded_before_falling_through(tmp_path, caplog):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    ctx.db.append(port="provide", backend="legacy-current",
                  key=ProvideKey(source="legacy-current", kind="picture", query="rice"),
                  subject="rice",
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"image": "images/pw-1.jpg"}},
                  answer={"items": [{"sha": "0" * 64, "ext": "jpg"}]})
    with caplog.at_level(logging.WARNING, logger="thai_syllabus.attempts"):
        res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None and not res.attempted
    assert "every awaiting candidate was excluded" in caplog.text and "0" * 64 in caplog.text


def test_assess_first_under_batch_collects_the_fit_question(tmp_path):
    ctx, _search, _judge = _picture_ctx(tmp_path)
    ctx.assessor = Assessor(record=ctx.db, cache=ctx.db, backends={"judge": _batch_judge()})
    legacy_sha = _seed_current_picture(ctx, "rice")
    res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None
    assert [q.question.artifact_sha for q in res.questions] == [legacy_sha]
    assert {q.question.role for q in res.questions} == {"picture-for-word"}


def test_assess_first_is_none_for_a_recording_need_with_no_candidate_on_record(tmp_path):
    """Not "the judge does not rank it" (r23 makes mechanical
    assess-first cover recording regardless of the judge's own rubric) --
    empty because unjudged_candidates has no candidate sha to offer."""
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    assert assess_first(ctx, Need("rice", "recording")) is None


# --- assess-first: recording (mechanical, spec 3 r23 section 5) -------------

def _seed_current_recording(ctx, subject, sha="c" * 64):
    """A candidate on record with no mechanical verdict: the row spec 2
    section 4 r7 writes for the old deck's current recording, and its
    media row -- the same shape _seed_current_picture uses for pictures."""
    ctx.db.append(port="provide", backend="legacy-current",
                  key=ProvideKey(source="legacy-current", kind="recording", query=subject),
                  subject=subject,
                  question={"provides": "recording", "kind": "recording", "subject_kind": "word",
                            "params": {"audio": "audio/pw-1.mp3"}},
                  answer={"items": [{"sha": sha, "ext": "mp3"}]})
    return sha


def test_assess_first_asks_mechanical_for_an_unjudged_recording_candidate_and_no_source(tmp_path):
    mech = _mechanical()
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), mechanical=mech)
    sha = _seed_current_recording(ctx, "rice")
    res = assess_first(ctx, Need("rice", "recording"))
    assert res is not None and res.attempted and res.questions == []
    verdicts = [r for r in rows_for(ctx.db, "rice", "recording")
               if r.port == "assess" and r.backend == "mechanical"]
    assert [r.question.get("artifact_sha") for r in verdicts] == [sha]
    assert all(r.port != "attempt" for r in rows_for(ctx.db, "rice", "recording"))
    assert current_best_of(ctx, "rice", "recording").artifact_sha == sha


def test_assess_first_is_none_once_every_recording_candidate_is_judged(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    _seed_current_recording(ctx, "rice")
    assert assess_first(ctx, Need("rice", "recording")) is not None
    assert assess_first(ctx, Need("rice", "recording")) is None


# --- assess-first under a rubric change: the incumbent alone (r33) ----------

_OLD_RUBRIC = "a previous picture rubric"


def _judged_under_a_previous_rubric(tmp_path, urls):
    """A picture need whose candidates were all judged under an older
    rubric, then the rubric changed: every verdict on record is stale.
    Returns (ctx, the candidate shas in record order)."""
    ctx, _search, _judge = _picture_ctx(tmp_path, urls=urls)
    ctx.rubrics["picture-for-word"] = _OLD_RUBRIC
    attempt(ctx, Need("rice", "picture"), "openverse")
    ctx.rubrics["picture-for-word"] = PICTURE_FIT_RUBRIC          # the rubric changes

    def resolve(sha):
        prov = ctx.db.media_provenance(sha)
        path = ctx.media_store.path_for(sha, prov["ext"]) if prov else None
        return path if path is not None and path.exists() else None

    ctx.assessor = Assessor(record=ctx.db, cache=ctx.db,
                            backends={"judge": _batch_judge(resolve)})
    return ctx, candidate_shas(rows_for(ctx.db, "rice", "picture"))


def _fresh_verdict(ctx, prepared, value):
    """The verdict a batch would bring back for one prepared question."""
    ctx.db.append(port="assess", backend="judge", key=prepared.key,
                  subject=prepared.question.subject,
                  question={"role": prepared.question.role, "kind": "picture",
                            "artifact_sha": prepared.question.artifact_sha,
                            "rubric": prepared.question.rubric,
                            "subject_kind": "word", "params": {}},
                  answer={"value": value})


def test_assess_first_under_a_rubric_change_asks_the_incumbent_alone(tmp_path):
    """Fix round 1 finding 1: the fit questions come from
    unjudged_candidates, so the fold's incumbent rule actually shortens
    the ask -- three stale candidates cost one question, not three."""
    ctx, shas = _judged_under_a_previous_rubric(
        tmp_path, ("https://x/good.jpg", "https://x/bad.jpg", "https://x/good2.jpg"))
    res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None and res.attempted
    assert [q.question.artifact_sha for q in res.questions] == [shas[2]]   # the newest pass


def test_assess_first_asks_the_rest_once_the_incumbent_failed_afresh(tmp_path):
    ctx, shas = _judged_under_a_previous_rubric(
        tmp_path, ("https://x/good.jpg", "https://x/bad.jpg", "https://x/good2.jpg"))
    _fresh_verdict(ctx, assess_first(ctx, Need("rice", "picture")).questions[0], False)
    res = assess_first(ctx, Need("rice", "picture"))
    assert [q.question.artifact_sha for q in res.questions] == [shas[0], shas[1]]


def test_assess_first_asks_every_candidate_when_none_passed_the_old_rubric(tmp_path):
    ctx, shas = _judged_under_a_previous_rubric(
        tmp_path, ("https://x/bad1.jpg", "https://x/bad2.jpg", "https://x/bad3.jpg"))
    res = assess_first(ctx, Need("rice", "picture"))
    assert [q.question.artifact_sha for q in res.questions] == list(shas)


def test_assess_first_asks_the_rest_when_the_incumbent_cannot_be_prepared(tmp_path):
    """Fix round 1 finding 3: an incumbent whose bytes are gone is
    excluded and never gets a verdict row, so the candidates it beat
    would wait on a verdict that can never arrive. They are asked on the
    same call instead."""
    ctx, shas = _judged_under_a_previous_rubric(
        tmp_path, ("https://x/good.jpg", "https://x/bad.jpg", "https://x/good2.jpg"))
    ctx.media_store.path_for(shas[2], "jpg").unlink()      # the incumbent's bytes vanish
    res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None and res.attempted
    assert [e.artifact_sha for e in res.excluded.values()] == [shas[2]]
    assert [q.question.artifact_sha for q in res.questions] == [shas[0], shas[1]]


def test_assess_first_falls_through_when_the_second_round_is_excluded_too(tmp_path):
    """Fix round 2 finding A: the second round gets the same
    total-exclusion check as the first, or a need whose every candidate
    is unpreparable comes back attempted with nothing asked and never
    reaches a source again."""
    ctx, shas = _judged_under_a_previous_rubric(
        tmp_path, ("https://x/good.jpg", "https://x/bad.jpg", "https://x/good2.jpg"))
    for sha in shas:
        ctx.media_store.path_for(sha, "jpg").unlink()
    res = assess_first(ctx, Need("rice", "picture"))
    assert res is not None and not res.attempted          # the caller goes on to a source
    assert sorted(e.artifact_sha for e in res.excluded.values()) == sorted(shas)
    assert res.questions == []


def test_a_source_attempt_judges_new_hits_even_while_an_incumbent_is_stuck(tmp_path):
    """Fix round 2 finding B: a hit this attempt just stored carries no
    verdict at all, so it is not one the incumbent beat and the gate
    never holds it -- even when the incumbent itself can never answer."""
    ctx, shas = _judged_under_a_previous_rubric(
        tmp_path, ("https://x/good.jpg", "https://x/bad.jpg", "https://x/good2.jpg",
                   "https://x/good3.jpg"))                # a fourth hit no attempt has tried
    ctx.media_store.path_for(shas[2], "jpg").unlink()     # the incumbent cannot be prepared
    res = attempt(ctx, Need("rice", "picture"), "openverse")
    fetched = [s for s in candidate_shas(rows_for(ctx.db, "rice", "picture")) if s not in shas]
    assert len(fetched) == 1
    assert fetched[0] in [q.question.artifact_sha for q in res.questions]


def test_assess_first_then_source_attempt_excludes_the_same_unpreparable_sha_once(tmp_path):
    """spec 3 section 7 / run.py's _Tally.collect: a picture need whose
    only candidate on record is unpreparable (a provide row naming a sha
    with no media object -- same "bytes never stored" shape as
    test_assess_first_returns_the_exclusion_when_every_waiting_candidate_is_excluded
    above -- so the judge backend's resolve_path finds nothing to attach)
    falls through assess_first to the source. The source attempt that
    follows (_picture_attempt) ends by judging every candidate awaiting a
    verdict, including the still-unpreparable one (it never got a verdict
    row, and PreparationError is never cached), so the real assess_first
    and attempt calls this test
    drives by hand -- no _patch, a real Sourcing ctx -- each produce an
    Excluded naming the same (subject, artifact_sha). _Tally.collect must
    fold those into exactly one RunReport.excluded / one excluded_items
    entry for that sha, not two.
    """
    ctx, _search, _judge = _picture_ctx(tmp_path, urls=("https://x/good.jpg",))
    ghost_sha = "0" * 64
    ctx.db.append(port="provide", backend="legacy-current",
                  key=ProvideKey(source="legacy-current", kind="picture", query="rice"),
                  subject="rice",
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"image": "images/pw-1.jpg"}},
                  answer={"items": [{"sha": ghost_sha, "ext": "jpg"}]})  # bytes never stored

    need = Need("rice", "picture")
    first = assess_first(ctx, need)
    assert first is not None and not first.attempted
    assert [e.artifact_sha for e in first.excluded.values()] == [ghost_sha]

    second = attempt(ctx, need, "openverse")
    assert ghost_sha in [e.artifact_sha for e in second.excluded.values()]

    tally = _Tally()
    tally.collect(first)
    tally.collect(second)

    assert tally.excluded == 1
    assert [item["artifact_sha"] for item in tally.excluded_items] == [ghost_sha]


# --- scene picture: the same attempt, subject = text_sha --------------------

def _sentence(text="ข้าว", gloss="the rice is tasty",   # ข้าว: rice
              clauses: Clauses = ((WordId("rice"),),)) -> Sentence:
    return Sentence(clauses=clauses, text=text, gloss=gloss, voice="learner_voice",
                    provenance=Provenance(source="llm", origin="m", licence="generated",
                                          acquired=date(2026, 9, 3)))


def test_scene_picture_attempt_searches_the_scene_s_drafted_phrase(tmp_path):
    """A scene need's query is its own drafted phrase, under the
    sentence's subject and the scene role."""
    sentence = _sentence()
    ctx, search, _judge = _picture_ctx(
        tmp_path, _word_syllabus().with_sentences([sentence]), urls=("https://x/good.jpg",))
    _seed_phrase(ctx, sentence.text_sha, "a family enjoying a rice meal", "sentence")
    res = attempt(ctx, Need(sentence.text_sha, "picture", "sentence"), "openverse")
    assert search.queries == ["a family enjoying a rice meal"]
    assert res.attempted and rows_for(ctx.db, sentence.text_sha, "picture")
    verdicts = [r for r in ctx.db.assessments_of(sentence.text_sha) if r.port == "assess"]
    assert {r.question["role"] for r in verdicts} == {"scene-for-sentence"}
    assert {r.question["subject_kind"] for r in verdicts} == {"sentence"}


def test_scene_fit_params_carry_the_sentences_target_word_and_its_gloss(tmp_path):
    """Spec 3 r33: the scene fit question names the word the production
    card blanks -- the last used word (the one the sentence introduces)."""
    sentence = _sentence()
    ctx, _search, _judge = _picture_ctx(
        tmp_path, _word_syllabus().with_sentences([sentence]))
    params = _picture_params(ctx, Need(sentence.text_sha, "picture", "sentence"), "a rice meal")
    assert params["word"] == sentence.text and params["meaning"] == sentence.gloss
    assert params["target"] == "ข้าว"            # ข้าว: rice
    assert params["target_gloss"] == "rice (cooked)"


def test_word_fit_params_carry_no_target(tmp_path):
    """A word's picture is judged on its own; there is nothing to blank."""
    ctx, _search, _judge = _picture_ctx(tmp_path)
    params = _picture_params(ctx, Need("rice", "picture"), "bowl of rice")
    assert "target" not in params and "target_gloss" not in params
    assert params["word"] == "ข้าว" and params["meaning"] == "rice (cooked)"   # ข้าว: rice


def test_a_scene_picture_attempt_refuses_a_sentence_with_no_phrase_on_record(tmp_path):
    """The sentence's gloss is the drafter's input, never a search (r25)."""
    sentence = _sentence()
    ctx, search, _judge = _picture_ctx(
        tmp_path, _word_syllabus().with_sentences([sentence]), urls=("https://x/good.jpg",))
    with pytest.raises(ValueError, match="no query"):
        attempt(ctx, Need(sentence.text_sha, "picture", "sentence"), "openverse")
    assert search.queries == []


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


def test_a_word_marked_male_draws_a_male_voice_without_a_productive_target(tmp_path):
    """The marking decides the constraint before the productive-back
    fallback is even consulted (spec 1 section 1 (r10))."""
    phom = word("phom", "ผม", "I (male speaker)", speaker="male")
    syllabus = Syllabus(words=(phom,), targets=(target("phom/receptive", "phom"),))
    ctx, tts = _recording_ctx(tmp_path, syllabus)
    attempt(ctx, Need("phom", "recording"), "tts")
    assert tts.last_voice in _MALE


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


def test_a_sentence_marked_female_draws_a_female_voice_and_forvo_admits_only_female(tmp_path):
    """The marking outranks the productive-back fallback: rice carries
    no productive Target here, yet ค่ะ's own female marking still picks
    the voice (spec 1 section 1 (r10))."""
    rice = word("rice", "ข้าว", "rice")   # ข้าว: rice
    kha = word("kha", "ค่ะ", "female politeness particle", speaker="female")
    draft = compose_sentence(((WordId("rice"), WordId("kha")),), thai_of(rice, kha),
                             gloss="the rice is good, politely")
    syllabus = Syllabus(words=(rice, kha), targets=(target("rice/receptive", "rice"),)
                        ).with_sentences([draft])
    ctx, tts = _recording_ctx(tmp_path, syllabus, {
        draft.text: [{"username": "malee", "pathmp3": "https://f/1.mp3", "sex": "f"},
                     {"username": "somchai", "pathmp3": "https://f/2.mp3", "sex": "m"}]})
    attempt(ctx, Need(draft.text_sha, "recording", "sentence"), "tts")
    assert tts.last_voice in _FEMALE

    attempt(ctx, Need(draft.text_sha, "recording", "sentence"), "forvo")
    assert ctx.db.speaker("forvo:malee").sex == "female"
    assert ctx.db.speaker("forvo:somchai") is None


def test_an_unmarked_receptive_only_sentence_stays_any(tmp_path):
    sentence = _sentence()
    ctx, tts = _recording_ctx(tmp_path, _word_syllabus().with_sentences([sentence]))
    attempt(ctx, Need(sentence.text_sha, "recording", "sentence"), "tts")
    assert tts.voices == [pick_voice(sentence.text_sha, list(_MALE) + list(_FEMALE))]


def test_pool_raises_on_an_empty_pool(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    ctx.voices = {"male": _MALE, "female": ()}
    with pytest.raises(ValueError, match="female"):
        _pool(ctx, "female")


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
    # spec 3 section 6a: this attempt's own outcome row anchors escalation
    # on the rendition identity it produced (current_best's artifact), so
    # it does not itself count toward exhaustion.
    assert exhausted(ctx.db, "p1", "rendition", sources=("forvo",), attempt_cap=8,
                     transient_cap=ctx.transient_cap).attempts == 0


def test_a_forvo_item_recorded_as_a_tone_sibling_is_not_a_candidate(tmp_path):
    """Forvo's lookup ignores tone marks: asking for ห่า (a classifier)
    returns หา ("to look for") and ห้า ("five") too, each item naming the
    word it records. Only the item recording the asked form is a
    candidate; the others are never downloaded."""
    syllabus = Syllabus(words=(word("classifier:ห่า", "ห่า", "classifier"),),
                        targets=(target("classifier:ห่า/receptive", "classifier:ห่า"),))
    ctx, _tts = _recording_ctx(tmp_path, syllabus, {
        "ห่า": [{"username": "master0z", "word": "หา", "pathmp3": "https://f/haa.mp3"},
                {"username": "skyton", "word": "ห่า", "pathmp3": "https://f/haa-low.mp3"},
                {"username": "deepindark", "word": "ห้า", "pathmp3": "https://f/haa-falling.mp3"}]})
    attempt(ctx, Need("classifier:ห่า", "recording"), "forvo")
    fetched = [r for r in rows_for(ctx.db, "classifier:ห่า", "recording")
               if r.port == "provide" and r.backend == "audiofetch"]
    assert [r.question["params"]["url"] for r in fetched] == ["https://f/haa-low.mp3"]


def test_a_forvo_item_differing_only_by_a_zero_width_mark_is_a_candidate(tmp_path):
    syllabus = Syllabus(words=(word("date", "วันที่", "date"),),   # วันที่: date
                        targets=(target("date/receptive", "date"),))
    ctx, _tts = _recording_ctx(tmp_path, syllabus, {
        "วันที่": [{"username": "mattissa", "word": "วันที่‎", "pathmp3": "https://f/d.mp3"}]})
    attempt(ctx, Need("date", "recording"), "forvo")
    fetched = [r for r in rows_for(ctx.db, "date", "recording")
               if r.port == "provide" and r.backend == "audiofetch"]
    assert len(fetched) == 1


def test_the_recorded_word_rides_the_download(tmp_path):
    """attempts._fetch_forvo_item's audiofetch params now carry the
    item's own recorded word (r49): the download row can be joined back
    to which Thai form the clip records without re-reading the lookup
    (record.recorded_form)."""
    syllabus = Syllabus(words=(word("classifier:ห่า", "ห่า", "classifier"),),
                        targets=(target("classifier:ห่า/receptive", "classifier:ห่า"),))
    ctx, _tts = _recording_ctx(tmp_path, syllabus, {
        "ห่า": [{"username": "skyton", "word": "ห่า", "pathmp3": "https://f/haa-low.mp3"}]})
    attempt(ctx, Need("classifier:ห่า", "recording"), "forvo")
    fetched = [r for r in rows_for(ctx.db, "classifier:ห่า", "recording")
              if r.port == "provide" and r.backend == "audiofetch"]
    assert fetched[0].question["params"]["word"] == "ห่า"


def test_an_item_naming_no_word_downloads_with_no_word_param(tmp_path):
    """An item with no `word` at all (a lookup row written before r49,
    or a served item Forvo answered with none) downloads with no "word"
    key in the audiofetch params at all -- never a literal None. Called
    directly: _forvo_lookup's own same-form filter (Task 1, spec 3 r49)
    would drop a wordless item before it ever reaches a download."""
    syllabus = Syllabus(words=(word("classifier:ห่า", "ห่า", "classifier"),),
                        targets=(target("classifier:ห่า/receptive", "classifier:ห่า"),))
    ctx, _tts = _recording_ctx(tmp_path, syllabus)
    item = {"username": "skyton", "pathmp3": "https://f/haa-low.mp3"}
    fetches = attempts_module._Fetches()
    attempts_module._fetch_forvo_item(ctx, "classifier:ห่า", item, fetches, subject_kind="word")
    fetched = [r for r in rows_for(ctx.db, "classifier:ห่า", "recording")
              if r.port == "provide" and r.backend == "audiofetch"]
    assert len(fetched) == 1 and "word" not in fetched[0].question["params"]


def test_a_rendition_intersection_ignores_items_recording_other_words(tmp_path):
    """Both members' lookups hold master0z, but his items record หา for
    both asks; the filtered lookups share no speaker and forvo answers
    empty (a rendition of one clip for both members was the live defect)."""
    confusion = SoundConfusion(id="tone:low-falling", dimension="tone", sounds=("low", "falling"))
    syllabus = Syllabus(
        words=(word("classifier:ห่า", "ห่า", "classifier", syllables=(syl(onset="h", vowel="a", length="long", tone="low"),)),
               word("five", "ห้า", "five", syllables=(syl(onset="h", vowel="a", length="long", tone="falling"),))),
        confusions=(confusion,),
        pairs=(MinimalPair(id="p-haa", confusion=confusion.id, members=("classifier:ห่า", "five")),))
    ctx, _tts = _recording_ctx(tmp_path, syllabus, {
        "ห่า": [{"username": "master0z", "word": "หา", "pathmp3": "https://f/1.mp3"}],
        "ห้า": [{"username": "master0z", "word": "หา", "pathmp3": "https://f/1.mp3"}]})
    attempt(ctx, Need("p-haa", "rendition", "pair"), "forvo")
    provided = [r for r in rows_for(ctx.db, "p-haa", "rendition") if r.port == "provide"]
    assert provided[-1].answer["items"] == []


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


def test_a_vetoed_tts_rendition_is_resynthesized_in_a_voice_not_yet_vetoed(tmp_path):
    """Design §4 (2026-09-19): 1 on a TTS rendition means "another
    voice", not the same bytes again. pick_voice is deterministic per
    pair, so the pool handed to it must exclude the vetoed voice."""
    from thai_syllabus.cachekeys import rendition_identity
    from thai_syllabus.learner import append_rating
    ctx, tts = _recording_ctx(tmp_path, _pair_syllabus())
    attempt(ctx, Need("p1", "rendition", "pair"), "tts")
    first = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"][-1]
    shas = {i["member"]: i["sha"] for i in first.answer["items"]}
    first_voice = first.answer["items"][0]["speaker"]["id"].removeprefix("tts:")
    append_rating(ctx.db, subject="p1", role="rendition-for-pair", rating="unacceptable-none",
                  artifact_sha=rendition_identity(shas), subject_kind="pair")

    attempt(ctx, Need("p1", "rendition", "pair"), "tts")

    second = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"][-1]
    voices = {i["speaker"]["id"].removeprefix("tts:") for i in second.answer["items"]}
    assert len(voices) == 1 and first_voice not in voices
    assert voices == {pick_voice("p1", [v for v in list(_MALE) + list(_FEMALE) if v != first_voice])}


def test_a_tts_rendition_answers_empty_once_every_voice_is_vetoed(tmp_path):
    from thai_syllabus.cachekeys import rendition_identity
    from thai_syllabus.learner import append_rating
    ctx, tts = _recording_ctx(tmp_path, _pair_syllabus())
    for _ in range(len(_MALE) + len(_FEMALE)):
        attempt(ctx, Need("p1", "rendition", "pair"), "tts")
        last = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"][-1]
        shas = {i["member"]: i["sha"] for i in last.answer["items"]}
        append_rating(ctx.db, subject="p1", role="rendition-for-pair",
                      rating="unacceptable-none", artifact_sha=rendition_identity(shas),
                      subject_kind="pair")
    attempt(ctx, Need("p1", "rendition", "pair"), "tts")
    last = [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "provide"][-1]
    assert last.answer["items"] == []


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
                                 "country": "Thailand", "word": thai,
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


class _ForvoLimitOnMembersFirstUrl:
    """Forvo's daily allowance is spent partway through a rendition's
    per-member downloads: the first member's url serves the limit JSON
    body (task 7 brief, spec 3 section 6a). Propagates through
    _forvo_rendition exactly as _download_forvo's own contract requires
    -- no relookup, no fetches.failed()."""
    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        raise FetchRefused(reason="content-type",
                           detail='content-type "application/json; charset=utf-8" is not allowed',
                           body='["Limit/day reached."]')


def test_a_rendition_attempt_raises_quota_exhausted_on_a_members_download(tmp_path):
    """_forvo_rendition (unlike _recording_attempt's forvo branch) calls
    _download_forvo from inside a loop over pair.members with no try of
    its own; this confirms QuotaExhausted still propagates unchanged
    through _rendition_attempt's outer catch."""
    forvo = _PairForvo({"white": "ขาว", "news": "ข่าว"})   # ขาว: white, ข่าว: news
    media = MediaStore(tmp_path / "media")
    db = SyllabusDb(tmp_path / "syllabus.db")
    ctx = _sourcing(tmp_path, _pair_syllabus(),
                    backends={"forvo": forvo, "audiofetch": _ForvoLimitOnMembersFirstUrl()},
                    assess={"mechanical": _mechanical(), "rendition": _rendition_backend(db)},
                    media=media)
    with pytest.raises(QuotaExhausted):
        attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    assert not [r for r in rows_for(ctx.db, "p1", "rendition") if r.port == "attempt"]


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


def _introducible_and_receptive_syllabus(n_introducible: int, n_receptive: int) -> Syllabus:
    """`n_introducible` words with a sentence-introduced, unmet Target
    (order: intro0 .. introN-1), followed by `n_receptive` words with a
    picture_card Target (recept0 .. receptN-1) -- syllabus.targets order
    is what gaps().unfilled_targets follows (rulebook._check_target_sentence
    walks syllabus.targets in order)."""
    introducible_words = tuple(word(f"intro{i}", f"อ{i}", f"intro {i}")
                               for i in range(n_introducible))
    receptive_words = tuple(word(f"recept{i}", f"ร{i}", f"recept {i}")
                            for i in range(n_receptive))
    introducible_targets = tuple(target(f"intro{i}/t", f"intro{i}", introduction="sentence")
                                 for i in range(n_introducible))
    receptive_targets = tuple(target(f"recept{i}/t", f"recept{i}") for i in range(n_receptive))
    return Syllabus(words=introducible_words + receptive_words,
                    targets=introducible_targets + receptive_targets)


def test_sentence_attempt_hands_at_most_the_introducible_cap(tmp_path):
    """Spec 3 r24 section 5: the handed targets are the next open Targets
    in order, at most `max_targets`, of which at most
    `ctx.sentence_introducible_per_ask` are sentence-introduced and unmet
    (introducible) -- the remainder are the next non-introduced open
    Targets in order. 36 introducible targets then 10 receptive ones,
    default cap 5: the ask hands 5 introducible + 10 receptive (15,
    under the 40 max_targets cap), skipping the other 31 introducible
    ones (they do not count against max_targets)."""
    syllabus = _introducible_and_receptive_syllabus(36, 10)
    ctx = _sentence_ctx(tmp_path, '{"sentences": []}', syllabus=syllabus)
    assert ctx.sentence_introducible_per_ask == 5
    res = sentence_attempt(ctx)
    assert res.targets_handed == 15
    assert res.subjects_handed == (frozenset(f"intro{i}" for i in range(5))
                                   | frozenset(f"recept{i}" for i in range(10)))


def test_sentence_attempt_honors_a_lowered_introducible_cap(tmp_path):
    """`ctx.sentence_introducible_per_ask` is Sourcing's own knob, read
    fresh each attempt -- lowering it hands fewer introducible targets
    without touching the non-introduced ones."""
    syllabus = _introducible_and_receptive_syllabus(36, 10)
    ctx = _sentence_ctx(tmp_path, '{"sentences": []}', syllabus=syllabus)
    ctx.sentence_introducible_per_ask = 2
    res = sentence_attempt(ctx)
    assert res.targets_handed == 12
    assert res.subjects_handed == (frozenset(f"intro{i}" for i in range(2))
                                   | frozenset(f"recept{i}" for i in range(10)))


# --- the no-fit answer (spec 3 r19 section 5) -------------------------------

_NO_FIT = json.dumps({"sentences": [], "reason": "no natural sentence covers both"})


def _seed_no_fit(db, word_id, target_id, *, times=1, reason="earlier no-fit"):
    """`times` no-fit outcome rows on one word's sentence need, in the
    shape sentence_attempt appends them."""
    for _ in range(times):
        db.append(port="attempt", backend="llm",
                  key=AttemptOutcomeKey(subject=word_id, kind="sentence", source="llm"),
                  subject=word_id,
                  question={"kind": "sentence", "source": "llm", "subject_kind": "word",
                           "targets": [target_id]},
                  answer={"outcome": "nothing", "candidates": [], "reason": reason})


def test_a_no_fit_answer_appends_one_nothing_row_per_handed_word(tmp_path):
    """Spec 3 r19 section 5: the drafter's own "nothing fits" answer is
    recognized and cached -- one outcome row per handed Target's WORD,
    the Target ids in the row's question, and no draft to judge."""
    ctx = _sentence_ctx(tmp_path, _NO_FIT)
    res = sentence_attempt(ctx)
    assert res.attempted is True and res.drafted == 0
    assert res.targets_handed == 2 and res.subjects_handed == frozenset({"eat", "rice"})
    assert res.questions == []
    for word_id in ("eat", "rice"):
        rows = [r for r in rows_for(ctx.db, word_id, "sentence") if r.port == "attempt"]
        assert len(rows) == 1
        assert rows[0].backend == "llm"
        assert rows[0].question == {"kind": "sentence", "source": "llm",
                                    "subject_kind": "word",
                                    "targets": [f"{word_id}/receptive"]}
        assert rows[0].answer == {"outcome": "nothing", "candidates": [],
                                  "reason": "no natural sentence covers both"}
        assert ctx.db.latest("attempt", "llm", AttemptOutcomeKey(
            subject=word_id, kind="sentence", source="llm")) is not None


def test_a_no_fit_answer_asks_the_judge_nothing(tmp_path):
    def never(prompt, attachments=()):
        raise AssertionError("a no-fit answer has no draft to judge")

    ctx = _sourcing(tmp_path, _sentence_ctx(tmp_path / "unused", _NO_FIT).syllabus,
                    backends={"llm-sentence": _Llm(_NO_FIT)},
                    assess={"judge": JudgeBackend(model="m", transport="api", complete=never)})
    assert sentence_attempt(ctx).questions == []


def test_a_cached_no_fit_is_re_asked_so_the_cap_counts_refusals_not_runs(tmp_path):
    """Spec 3 r19 section 6a's re-ask rule: a no-fit adds nothing to the
    prompt's refused block, so the next run's ask would hit the very same
    cached answer and cache one more `nothing` row for a refusal that
    never happened. A recognized no-fit served from the cache is re-asked
    once, and only the fresh answer is recorded."""
    ctx = _sentence_ctx(tmp_path, _NO_FIT)
    drafter = ctx.provider._backends["llm-sentence"]
    sentence_attempt(ctx)
    sentence_attempt(ctx)
    assert len(drafter.prompts) == 2            # the second run re-asked
    assert drafter.prompts[0] == drafter.prompts[1]    # ...on the same prompt
    for word_id in ("eat", "rice"):
        rows = [r for r in rows_for(ctx.db, word_id, "sentence") if r.port == "attempt"]
        assert len(rows) == 2                   # one per refusal, not one per run


def test_a_re_asked_no_fit_that_comes_back_with_drafts_records_no_nothing_row(tmp_path):
    """The re-ask is the run's answer: drafts go to the judge and the
    cached no-fit leaves no second outcome row behind."""
    ctx = _sentence_ctx(tmp_path, _NO_FIT, batch=True)
    drafter = ctx.provider._backends["llm-sentence"]
    sentence_attempt(ctx)
    drafter.text = _draft_json(("eat", "rice"), "กินข้าว", "eat rice")   # กินข้าว: eat rice
    res = sentence_attempt(ctx)
    assert res.attempted is True and res.drafted == 1 and len(res.questions) == 1
    for word_id in ("eat", "rice"):
        rows = [r for r in rows_for(ctx.db, word_id, "sentence") if r.port == "attempt"]
        assert len(rows) == 1                   # only the first run's refusal


def test_a_re_asked_no_fit_counts_both_drafter_asks_under_spend(tmp_path):
    ctx = _sentence_ctx(tmp_path, _NO_FIT)
    sentence_attempt(ctx)
    # the hit itself is free; the re-ask it triggers is the run's one ask
    assert sentence_attempt(ctx).spend["llm-sentence"].asks == 1


def test_a_sentence_exhausted_word_s_targets_are_not_handed_to_the_drafter(tmp_path):
    """At the no-fit cap the word stops being drafted for: it is neither
    handed over nor counted as deferred -- `subjects_exhausted` is how the
    run buckets it."""
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat",), "กิน", "eat"))   # กิน: eat
    _seed_no_fit(ctx.db, "rice", "rice/receptive", times=3)
    res = sentence_attempt(ctx)
    assert res.targets_handed == 1 and res.subjects_handed == frozenset({"eat"})
    assert res.subjects_exhausted == frozenset({"rice"})
    prompt = ctx.provider._backends["llm-sentence"].prompts[-1]
    assert "target eat/receptive" in prompt and "target rice/receptive" not in prompt


def test_a_word_below_the_no_fit_cap_is_still_handed_to_the_drafter(tmp_path):
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat",), "กิน", "eat"))   # กิน: eat
    _seed_no_fit(ctx.db, "rice", "rice/receptive", times=2)
    res = sentence_attempt(ctx)
    assert res.subjects_handed == frozenset({"eat", "rice"})
    assert res.subjects_exhausted == frozenset()


def test_the_sourcing_s_own_no_fit_cap_decides_when_a_word_is_withheld(tmp_path):
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat",), "กิน", "eat"))   # กิน: eat
    ctx.sentence_nothing_cap = 2
    _seed_no_fit(ctx.db, "rice", "rice/receptive", times=2)
    assert sentence_attempt(ctx).subjects_exhausted == frozenset({"rice"})


def test_the_drafter_is_not_asked_at_all_when_every_open_word_is_exhausted(tmp_path):
    ctx = _sentence_ctx(tmp_path, _NO_FIT)
    _seed_no_fit(ctx.db, "rice", "rice/receptive", times=3)
    _seed_no_fit(ctx.db, "eat", "eat/receptive", times=3)
    res = sentence_attempt(ctx)
    assert res.attempted is False and res.targets_handed == 0
    assert res.subjects_exhausted == frozenset({"eat", "rice"})
    assert ctx.provider._backends["llm-sentence"].prompts == []


def test_a_learner_direction_hands_a_withheld_word_back_to_the_drafter(tmp_path):
    """Spec 3 r19 section 6a: a learner row on the word reopens it."""
    ctx = _sentence_ctx(tmp_path, _draft_json(("eat",), "กิน", "eat"))   # กิน: eat
    _seed_no_fit(ctx.db, "rice", "rice/receptive", times=3)
    ctx.db.append(port="assess", backend="learner",
                  key=DirectionKey(subject="rice", role="sentence-for-target",
                                   text_sha=sha("pair it with a verb")),
                  subject="rice",
                  question={"kind": "direction", "role": "sentence-for-target",
                           "subject_kind": "word"},
                  answer={"direction": "pair it with a verb"})
    res = sentence_attempt(ctx)
    assert res.subjects_handed == frozenset({"eat", "rice"})
    assert res.subjects_exhausted == frozenset()


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


def test_sentence_attempt_keeps_the_first_gloss_when_a_repeated_draft_s_glosses_disagree(tmp_path,
                                                                                          caplog):
    """Spec 3 r19 section 5: a text listed twice with differing glosses is
    one candidate, keyed by the text -- the first gloss stands and the
    draft still reaches the judge."""
    text = json.dumps({"sentences": [
        {"clauses": [["eat", "rice"]], "text": "กินข้าว", "gloss": "eat rice"},
        {"clauses": [["eat", "rice"]], "text": "กินข้าว",
         "gloss": "rice is eaten"}]})   # กินข้าว: eat rice
    ctx = _sentence_ctx(tmp_path, text, batch=True)
    with caplog.at_level(logging.DEBUG):
        res = sentence_attempt(ctx)
    assert len(res.questions) == 1
    assert res.questions[0].question.params["gloss"] == "eat rice"
    assert any("กินข้าว" in r.message and r.levelno == logging.DEBUG for r in caplog.records)


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


def test_sentence_attempt_keeps_the_first_gloss_across_answer_items(tmp_path):
    """A text listed in two different answer items is one candidate: a
    gloss conflict between the items keeps the first gloss, the same as
    a conflict within one item -- the merge spans every item, not just
    one."""
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
    assert len(res.questions) == 1
    assert res.questions[0].question.params["gloss"] == "eat rice"


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


def _clauses_draft_json(clause_word_ids, text, gloss) -> str:
    """One drafting-answer item with one clause per element of
    `clause_word_ids` (each a single-word clause)."""
    return json.dumps({"sentences": [
        {"clauses": [[w] for w in clause_word_ids], "text": text, "gloss": gloss}]})


def test_sentence_attempt_refuses_a_draft_over_the_clause_cap(tmp_path, caplog):
    """Spec 3 section 5: a draft with more clauses than
    `ctx.sentence_max_clauses` is refused like the Sentence invariant --
    logged and skipped, never reaching the judge -- though it is
    otherwise a valid, target-filling draft."""
    syllabus = _three_word_syllabus()   # eat, rice, tasty each carry an open Target
    text = _clauses_draft_json(("eat", "rice", "tasty"), "กิน ข้าว อร่อย", "eats tasty rice")
    ctx = _sentence_ctx(tmp_path, text, batch=True, syllabus=syllabus)
    assert ctx.sentence_max_clauses == 2
    with caplog.at_level(logging.WARNING):
        res = sentence_attempt(ctx)
    assert res.questions == [] and res.drafted == 0
    assert ctx.db.all_sentences() == []
    assert "draft refused" in caplog.text and "3 clauses" in caplog.text and "cap 2" in caplog.text


def _four_word_syllabus():
    # กิน eat, ข้าว rice, อร่อย tasty, มาก very -- four open receptive Targets
    return Syllabus(
        words=(word("eat", "กิน", "eat"), word("rice", "ข้าว", "rice"),
               word("tasty", "อร่อย", "tasty"), word("very", "มาก", "very")),
        targets=(target("eat/receptive", "eat"), target("rice/receptive", "rice"),
                 target("tasty/receptive", "tasty"), target("very/receptive", "very")),
        frequency={"eat": 1, "rice": 2, "tasty": 3, "very": 4})


def test_sentence_attempt_refuses_a_draft_filling_more_open_targets_than_the_cap(tmp_path, caplog):
    """Spec 3 r27 section 5: a draft that fills more open Targets than
    `ctx.sentence_targets_per_sentence` is a word list in disguise --
    refused like the clause cap, logged, never reaching the judge."""
    syllabus = _four_word_syllabus()
    text = _draft_json(("eat", "rice", "tasty", "very"), "กินข้าวอร่อยมาก", "eats very tasty rice")
    ctx = _sentence_ctx(tmp_path, text, batch=True, syllabus=syllabus)
    assert ctx.sentence_targets_per_sentence == 3
    with caplog.at_level(logging.WARNING):
        res = sentence_attempt(ctx)
    assert res.questions == [] and res.drafted == 0
    assert "draft refused" in caplog.text and "4 targets" in caplog.text and "cap 3" in caplog.text


def test_sentence_attempt_accepts_the_same_draft_at_a_raised_target_cap(tmp_path):
    syllabus = _four_word_syllabus()
    text = _draft_json(("eat", "rice", "tasty", "very"), "กินข้าวอร่อยมาก", "eats very tasty rice")
    ctx = _sentence_ctx(tmp_path, text, syllabus=syllabus)
    ctx.sentence_targets_per_sentence = 4
    assert sentence_attempt(ctx).drafted == 1


def test_sentence_attempt_counts_only_open_targets_against_the_cap(tmp_path):
    """Met words are filler (r27): once an adopted sentence fills "very",
    a four-word draft fills three open Targets and passes the cap of 3."""
    syllabus = _four_word_syllabus().with_sentences(
        [_sentence(text="มาก", gloss="very", clauses=((WordId("very"),),))])
    assert len(syllabus.gaps().unfilled_targets) == 3
    text = _draft_json(("eat", "rice", "tasty", "very"), "กินข้าวอร่อยมาก", "eats very tasty rice")
    ctx = _sentence_ctx(tmp_path, text, syllabus=syllabus)
    assert sentence_attempt(ctx).drafted == 1


def test_sentence_attempt_accepts_a_draft_at_a_raised_clause_cap(tmp_path):
    """The very same three-clause draft, judged and drafted once
    `ctx.sentence_max_clauses` is raised to admit it."""
    syllabus = _three_word_syllabus()
    text = _clauses_draft_json(("eat", "rice", "tasty"), "กิน ข้าว อร่อย", "eats tasty rice")
    ctx = _sentence_ctx(tmp_path, text, syllabus=syllabus)
    ctx.sentence_max_clauses = 3
    res = sentence_attempt(ctx)
    assert res.drafted == 1


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
    assert row.question == {"kind": "picture", "subject_kind": "word", "source": "openverse", "query": "rice food"}
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


def _cache_search_hit(ctx, urls, *, query="rice food"):
    """A provide row already on record for this query: ask() serves it as
    a cache hit (`hit=True`), making a later served refusal of every hit
    the "cached answer" case spec 3 section 5/6a re-asks."""
    ctx.db.append(port="provide", backend="openverse",
                  key=ProvideKey(source="openverse", kind="", query=query), subject="rice",
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"query": query}},
                  answer={"items": [{"url": u, "source": "openverse", "origin": u,
                                     "licence": "cc0"} for u in urls]})


def test_a_picture_attempt_does_not_re_ask_a_search_asked_live_in_this_attempt(tmp_path):
    """spec 3 section 5 (r19): a search asked live in this attempt is not
    re-asked -- its hits were just served, so a served refusal of all of
    them is the attempt's own outcome, not cause for a second ask."""
    ctx, search, _judge = _picture_ctx(
        tmp_path, urls=("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    ctx.provider._backends["imgfetch"] = _ServedRefusingImgfetch(ctx.media_store)
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert len(search.queries) == 1                   # one live ask, no reask
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer["outcome"] == "transient-failure"


def test_a_picture_attempt_re_asks_the_search_once_when_its_cached_hits_are_all_refused(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/fresh.jpg",))
    _cache_search_hit(ctx, ("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    imgfetch = _ServedRefusingImgfetch(ctx.media_store)
    ctx.provider._backends["imgfetch"] = imgfetch
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert len(search.queries) == 1                   # only the reask calls the backend
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
    fresh = tuple(f"https://x/fresh{i}.jpg" for i in range(5))
    ctx, search, _judge = _picture_ctx(tmp_path, urls=fresh)
    _cache_search_hit(ctx, ("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    imgfetch = _ServedRefusingImgfetch(ctx.media_store)
    ctx.provider._backends["imgfetch"] = imgfetch
    result = attempt(ctx, Need("rice", "picture"), "openverse")
    assert imgfetch.asked == ["https://x/rotted1.jpg", "https://x/rotted2.jpg",
                              *fresh[:ctx.image_candidates]]
    assert result.spend["openverse"].asks == 1        # the cache hit costs nothing; only the reask asks


def test_a_picture_attempt_writes_transient_failure_then_reraises_when_the_re_ask_fails(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path, urls=("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    _cache_search_hit(ctx, ("https://x/rotted1.jpg", "https://x/rotted2.jpg"))
    ctx.provider._backends["openverse"] = _DeadSearch()
    ctx.provider._backends["imgfetch"] = _ServedRefusingImgfetch(ctx.media_store)
    with pytest.raises(TransportError):
        attempt(ctx, Need("rice", "picture"), "openverse")
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer == {"outcome": "transient-failure", "candidates": [],
                         "tried": ["https://x/rotted1.jpg", "https://x/rotted2.jpg"]}


def test_a_picture_attempt_asks_openverse_afresh_once_its_nothing_aged_out(tmp_path):
    """spec 3 r19 section 6a: an aged-out `nothing` re-offers the source
    and the attempt makes a fresh search (the reask path) instead of
    reading the cached empty answer, so a corpus that grew is seen."""
    day = 86_400 * 1_000_000_000
    ctx, search, _judge = _picture_ctx(tmp_path, phrase=None)
    # the cached empty search and the stale nothing row of an earlier attempt
    ctx.db.append(port="provide", backend="openverse",
                  key=ProvideKey(source="openverse", kind="", query="rice food"), subject="rice",
                  question={"provides": "picture", "kind": "picture", "subject_kind": "word",
                            "params": {"query": "rice food"}},
                  answer={"items": []}, cost=1.0, ts=1 * day)
    ctx.db.append(port="attempt", backend="openverse",
                  key=AttemptOutcomeKey(subject="rice", kind="picture", source="openverse"),
                  subject="rice", question={"kind": "picture", "subject_kind": "word",
                                            "source": "openverse"},
                  answer={"outcome": "nothing", "candidates": []}, cost=0.0, ts=1 * day)
    # the phrase row last: the store keeps ts monotonic, so seeding it
    # first would drag the "old" rows above forward past the ttl
    _seed_phrase(ctx, "rice", "rice food")
    ctx.nothing_ttl = {"openverse": 180}
    ctx.now_ns = lambda: 201 * day
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == ["rice food"], "a fresh search, not the cached empty answer"
    row = _outcome(ctx.db, "rice", "picture", "openverse")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 3


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
           for i in r.answer.get("items", ()) if "sha" in i]
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


def test_a_recording_attempt_asks_forvo_afresh_once_its_nothing_aged_out(tmp_path):
    """spec 3 r19 section 6a: an aged-out `nothing` re-offers the source
    and the attempt makes a fresh lookup (the reask path) instead of
    reading the cached empty answer, so a corpus that grew is seen."""
    day = 86_400 * 1_000_000_000
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3"}]})   # ข้าว: rice
    forvo = ctx.provider._backends["forvo"]
    calls = []
    original_fetch = forvo.fetch
    forvo.fetch = lambda q: calls.append(q) or original_fetch(q)
    # the cached empty lookup and the stale nothing row of an earlier attempt
    ctx.db.append(port="provide", backend="forvo",
                  key=ProvideKey(source="forvo", kind="", query="ข้าว"), subject="rice",
                  question={"kind": "recording", "subject_kind": "word", "params": {"word": "ข้าว"}},
                  answer={"items": []}, cost=1.0, ts=1 * day)
    ctx.db.append(port="attempt", backend="forvo",
                  key=AttemptOutcomeKey(subject="rice", kind="recording", source="forvo"),
                  subject="rice", question={"kind": "recording", "subject_kind": "word", "source": "forvo"},
                  answer={"outcome": "nothing", "candidates": []}, cost=0.0, ts=1 * day)
    ctx.nothing_ttl = {"forvo": 180}
    ctx.now_ns = lambda: 201 * day
    attempt(ctx, Need("rice", "recording"), "forvo")
    assert len(calls) == 1, "a fresh lookup, not the cached empty answer"
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 1


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


class _ForvoLimitAtDownloadAudiofetch:
    """Forvo's daily allowance is spent: its mp3 url serves the limit
    JSON body `["Limit/day reached."]` as application/json, refused by
    audiofetch as content-type (task 7 brief, spec 3 section 6a)."""
    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        raise FetchRefused(reason="content-type",
                           detail='content-type "application/json; charset=utf-8" is not allowed',
                           body='["Limit/day reached."]')


def test_a_recording_attempt_raises_quota_exhausted_on_forvos_limit_body_at_download(tmp_path):
    """Compare to a plain content-type refusal (transient-failure, no
    raise): a content-type refusal whose body is Forvo's own daily-limit
    statement is Quota, not a served refusal (spec 3 section 6a) --
    reraises without a fetches.failed() and without an outcome row."""
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3"}]})   # ข้าว: rice
    ctx.provider._backends["audiofetch"] = _ForvoLimitAtDownloadAudiofetch()
    with pytest.raises(QuotaExhausted):
        attempt(ctx, Need("rice", "recording"), "forvo")
    assert not [r for r in rows_for(ctx.db, "rice", "recording") if r.port == "attempt"]


class _ContentTypeRefusalWithAnotherBodyAudiofetch:
    """A served content-type refusal whose body is not Forvo's limit
    statement: stays a plain served refusal (transient-failure), not
    Quota."""
    def cache_key(self, q):
        return ProvideKey(source="", kind="", query=q.params["url"])

    def fetch(self, q):
        raise FetchRefused(reason="content-type",
                           detail='content-type "text/html" is not allowed',
                           body='["some other message"]')


def test_a_content_type_refusal_with_another_body_stays_a_served_refusal(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus(), {
        "ข้าว": [{"username": "somchai", "pathmp3": "https://f/u.mp3"}]})   # ข้าว: rice
    ctx.provider._backends["audiofetch"] = _ContentTypeRefusalWithAnotherBodyAudiofetch()
    result = attempt(ctx, Need("rice", "recording"), "forvo")
    assert result.attempted
    row = _outcome(ctx.db, "rice", "recording", "forvo")
    assert row.answer["outcome"] == "transient-failure"


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
                                 "word": q.params["word"],
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
                                 "word": q.params["word"],
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
             "word": q.params["word"], "pathmp3": "https://forvo/audio/1.mp3"},
            {"id": 2, "username": "b", "sex": "m", "country": "Thailand",
             "word": q.params["word"], "pathmp3": "https://forvo/audio/2.mp3"}), cost=1.0)


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
                                 "country": "Thailand", "word": q.params["word"],
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
    assert row.answer["outcome"] == "candidates" and len(row.answer["candidates"]) == 3


def test_a_rendition_attempts_candidates_end_with_the_rendition_identity(tmp_path):
    """spec 3 section 6a: escalation anchors on the attempt-outcome row
    that produced current-best's artifact. A rendition's current-best
    artifact_sha is the rendition identity over its members
    (cachekeys.rendition_identity), so the outcome row's candidates must
    include that identity alongside the member recording shas for
    derivations._anchor_ts to ever find a producing row."""
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]})  # ข่าว: news
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    row = _outcome(ctx.db, "p1", "rendition", "forvo")
    member_shas = row.answer["candidates"][:2]
    assert row.answer["candidates"][2] == rendition_identity(
        {"white": member_shas[0], "news": member_shas[1]})


def test_a_passing_rendition_attempt_anchors_escalation_on_its_own_row(tmp_path):
    """spec 3 section 6a: the row that produced current-best's artifact
    anchors escalation -- attempts_since_change is empty right after the
    producing attempt, and a later outcome row counts as one attempt
    since that anchor."""
    ctx, _tts = _recording_ctx(tmp_path, _pair_syllabus(), {
        "ขาว": [{"username": "somchai", "pathmp3": "https://f/a.mp3"}],   # ขาว: white
        "ข่าว": [{"username": "somchai", "pathmp3": "https://f/b.mp3"}]})  # ข่าว: news
    attempt(ctx, Need("p1", "rendition", "pair"), "forvo")
    assert attempts_since_change(ctx.db, "p1", "rendition") == []

    ctx.db.append(port="attempt", backend="tts",
                  key=AttemptOutcomeKey(subject="p1", kind="rendition", source="tts"),
                  subject="p1",
                  question={"kind": "rendition", "source": "tts", "subject_kind": "pair"},
                  answer={"outcome": "nothing", "candidates": []})
    assert len(attempts_since_change(ctx.db, "p1", "rendition")) == 1


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
        return MechanicalKey(check="duration", params="0.2-5.0", subject=q.subject,
                             artifact_sha=q.artifact_sha or "-")

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
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2)
    vocabulary = prompt.split("Vocabulary, in the order met:\n")[1].split("\nTargets:")[0]
    assert vocabulary.splitlines() == ["- eat  กิน  (eat)", "- rice  ข้าว  (rice)",
                                       "- tasty  อร่อย  (tasty)"]
    assert prompt.count("อร่อย") == 2          # once in the list, once on its own target line


def test_sentence_prompt_lists_a_targets_line_per_handed_target():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2)
    assert "- target eat/receptive: eat  กิน  (eat)" in prompt
    assert "- target rice/receptive: rice  ข้าว  (rice)" in prompt
    assert "- target tasty/receptive: tasty  อร่อย  (tasty)" in prompt


def test_sentence_prompt_gives_the_required_covering_instruction_verbatim():
    """Spec 3 r27 section 5: as many sentences as it takes, each filling
    at most the cap -- never "the fewest sentences", which drafts word
    lists."""
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2,
                              sentence_targets_per_sentence=3)
    assert ("Each JSON item is one sentence. Write as many natural sentences as it takes to "
           "cover the targets below; a sentence fills at most 3 of the targets (two or three "
           "is right) and may use any other listed vocabulary besides. A sentence may "
           "introduce at most one word from the Introducible list and must otherwise use only "
           "the vocabulary below.") in prompt
    assert "fewest" not in prompt


def test_sentence_prompt_gives_the_clause_rendering_rule_and_json_shape_verbatim():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2)
    assert ("Write each sentence as clauses of vocabulary ids in order; a clause renders as "
           "its words' Thai concatenated, clauses are separated by one space; write a repeated "
           'word as [id, "ๆ"]; standard spelling (ครับ, never คับ); numbers as number words; '
           "no punctuation or digits.") in prompt
    assert ('Output JSON only: {"sentences": [{"clauses": [["id", ...], ...], "text": "...", '
           '"gloss": "..."}]}') in prompt


def test_sentence_prompt_requires_ids_exactly_as_listed():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2)
    assert "Use vocabulary ids exactly as listed, suffix included" in prompt


def _repeated_word_syllabus():
    # rice first in entry order, then eat; no word id carries a
    # -<digit> suffix.
    return Syllabus(
        words=(word("rice", "ข้าว", "rice"), word("eat", "กิน", "eat")),
        targets=(target("rice/receptive", "rice"), target("eat/receptive", "eat")),
        frequency={"rice": 1, "eat": 2})


def test_sentence_prompt_worked_example_picks_a_real_repeated_id_and_falls_back_to_the_literal_suffixed_id():
    """Spec 3 r24 section 5: the worked example is built from ids in the
    prompt's own vocabulary where possible -- the repeated word takes the
    first vocabulary id; the suffixed id falls back to the literal
    `delicious-2` when no vocabulary word carries a -<digit> suffix."""
    syllabus = _repeated_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2)
    assert ('{"clauses": [["i-male-speaker", "eat", ["rice", "ๆ"], "delicious-2"]], '
           '"text": "...", "gloss": "..."}') in prompt


def test_sentence_prompt_worked_example_picks_a_real_suffixed_id_from_vocabulary_when_one_exists():
    syllabus = Syllabus(
        words=(word("rice", "ข้าว", "rice"), word("tasty-2", "อร่อย", "tasty")),
        targets=(target("rice/receptive", "rice"), target("tasty-2/receptive", "tasty-2")),
        frequency={"rice": 1, "tasty-2": 2})
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2)
    assert ('{"clauses": [["i-male-speaker", "eat", ["rice", "ๆ"], "tasty-2"]], '
           '"text": "...", "gloss": "..."}') in prompt


def test_sentence_prompt_worked_example_falls_back_to_the_literal_repeated_word_when_vocabulary_is_empty():
    syllabus = Syllabus(words=(), targets=(), frequency={})
    prompt = _sentence_prompt(syllabus, [], sentence_max_clauses=2)
    assert ('{"clauses": [["i-male-speaker", "eat", ["little", "ๆ"], "delicious-2"]], '
           '"text": "...", "gloss": "..."}') in prompt


def test_sentence_prompt_states_the_clause_cap_from_the_given_value():
    """Spec 3 r23 section 5/8: the drafting prompt asks for at most
    `sentence_max_clauses` clauses, the value the ctx hands it."""
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=2)
    assert "Each sentence has at most 2 clauses." in prompt
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), sentence_max_clauses=3)
    assert "Each sentence has at most 3 clauses." in prompt


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
    prompt = _sentence_prompt(syllabus, [glue1, glue2], sentence_max_clauses=2)
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
    prompt = _sentence_prompt(syllabus, [glue2], sentence_max_clauses=2)
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
    prompt = _sentence_prompt(syllabus, [glue1], sentence_max_clauses=2)
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
    prompt = _sentence_prompt(syllabus, [productive], sentence_max_clauses=2)
    assert "- target help/productive: help  ช่วย  (help)" in prompt


def test_sentence_prompt_lists_only_the_vocabulary_the_handed_targets_met():
    syllabus = _three_word_syllabus()
    first_only = [t for t in syllabus.targets if t.id == "eat/receptive"]
    vocabulary = _sentence_prompt(syllabus, first_only, sentence_max_clauses=2).split(
        "Vocabulary, in the order met:\n")[1].split("\nTargets:")[0]
    assert vocabulary.splitlines() == ["- eat  กิน  (eat)"]


def test_sentence_prompt_appends_the_refused_block_when_refused_texts_exist():
    """Spec 3 r19 section 5: the prompt also lists, as sentences not to
    propose, the texts `refused` names -- unadopted drafts the judge has
    failed, newest first, at most `derivations.refused_drafts`' own
    `limit` (`_sentence_prompt` itself is agnostic to how `refused` was
    selected) -- each with the verdict's evidence, before the
    output-format sentence."""
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), refused=[("กินข้าว", "too formal")],
                              sentence_max_clauses=2)   # กินข้าว: eat rice
    assert ("Do not propose these sentences; each failed review:\n"
           f"{UNTRUSTED}\n"
           f"- กินข้าว — {deck_field('too formal')}") in prompt
    assert prompt.index("Do not propose these sentences") < prompt.index(
        "Write each sentence as clauses")


def test_sentence_prompt_lists_each_refused_text_and_omits_the_block_when_empty():
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), refused=[("กิน", "e1"), ("ข้าว", "e2")],
                              sentence_max_clauses=2)   # กิน: eat, ข้าว: rice
    assert f"- กิน — {deck_field('e1')}" in prompt and f"- ข้าว — {deck_field('e2')}" in prompt
    assert "Do not propose these sentences" not in _sentence_prompt(
        syllabus, list(syllabus.targets), sentence_max_clauses=2)


def test_sentence_prompt_renders_a_refused_text_with_no_evidence_with_no_dangling_separator():
    """An empty evidence string renders `- <text>` alone -- no trailing
    ' — ' with nothing after it."""
    syllabus = _three_word_syllabus()
    prompt = _sentence_prompt(syllabus, list(syllabus.targets), refused=[("กิน", "")],
                              sentence_max_clauses=2)   # กิน: eat
    assert "- กิน\n" in prompt
    assert "— " not in prompt and "—\n" not in prompt


# --- phrase_attempt: one drafted image-search phrase per open picture need --

class _LlmPhrase:
    """The phrase drafter: `text` is the JSON its one answer carries;
    every prompt it was asked is recorded."""

    def __init__(self, text):
        self.text, self.prompts = text, []

    def cache_key(self, q):
        return LlmPromptKey(producer="phrase-drafter", model="m",
                            prompt_sha=sha(q.params["prompt"]))

    def fetch(self, q):
        self.prompts.append(q.params["prompt"])
        return RawAnswer(items=(self.text,))


def _phrase_ctx(tmp_path, syllabus, text) -> Sourcing:
    return _sourcing(tmp_path, syllabus, backends={"llm-phrase": _LlmPhrase(text)}, assess={})


def _phrase_drafter(ctx: Sourcing) -> _LlmPhrase:
    return ctx.provider._backends["llm-phrase"]


class _FixedCompletionTransport:
    """A drafter transport (LlmBackend's own `.complete(prompt)`
    contract) answering one fixed completion; records every prompt it
    was asked."""

    def __init__(self, text):
        self.text, self.prompts = text, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return Completion(text=self.text)


def _real_phrase_backend(text: str) -> LlmBackend:
    """The production llm-phrase wiring (spec 3 section 2,
    wiring.build_provider's own recognizer for producer "phrase-drafter",
    fix round 2 finding 2): recognizes only a completion naming at least
    one asked item's phrase -- an answer phrasing none of them (garbage,
    or an empty `{"phrases": []}`) is not recognized, so LlmBackend.fetch
    raises and caches nothing.
    """
    return LlmBackend(producer="phrase-drafter", model="m", transport=_FixedCompletionTransport(text),
                      recognize=lambda t: bool(parse_phrases(t)))


def test_phrase_attempt_drafts_one_phrase_each_for_a_word_and_a_sentence_need(tmp_path):
    scene = _sentence()   # subject_kind "sentence", subject = its text_sha
    syllabus = _word_syllabus().with_sentences([scene])
    text = json.dumps({"phrases": [
        {"subject": "rice", "phrase": "bowl of steamed rice"},
        {"subject": scene.text_sha, "phrase": "a family eating rice together"}]})
    ctx = _phrase_ctx(tmp_path, syllabus, text)
    res = phrase_attempt(ctx)
    assert res.attempted
    assert len(_phrase_drafter(ctx).prompts) == 1   # one ask for both needs
    assert drafted_phrase(ctx.db.assessments_of("rice")) == "bowl of steamed rice"
    assert drafted_phrase(ctx.db.assessments_of(scene.text_sha)) == "a family eating rice together"


def test_phrase_attempt_does_not_reask_a_subject_that_already_has_a_phrase(tmp_path):
    syllabus = _word_syllabus()
    ctx = _phrase_ctx(tmp_path, syllabus,
                      json.dumps({"phrases": [{"subject": "rice", "phrase": "should not be asked"}]}))
    ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject="rice"), subject="rice",
                  question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                  answer={"phrase": "already drafted"})
    res = phrase_attempt(ctx)
    assert res.attempted is False
    assert _phrase_drafter(ctx).prompts == []
    assert drafted_phrase(ctx.db.assessments_of("rice")) == "already drafted"


def test_phrase_attempt_skips_a_subject_that_already_carries_a_learner_direction(tmp_path):
    """The direction always wins over a drafted phrase (record.latest_phrase),
    so drafting one is dead weight -- minor fix, fix round 2."""
    syllabus = _word_syllabus()
    ctx = _phrase_ctx(tmp_path, syllabus,
                      json.dumps({"phrases": [{"subject": "rice", "phrase": "should not be asked"}]}))
    ctx.db.append(port="assess", backend="learner",
                  key=DirectionKey(subject="rice", role="picture-for-word", text_sha=sha("try red")),
                  subject="rice", question={"kind": "direction", "role": "picture-for-word"},
                  answer={"direction": "try red"})
    res = phrase_attempt(ctx)
    assert res.attempted is False
    assert _phrase_drafter(ctx).prompts == []
    assert drafted_phrase(ctx.db.assessments_of("rice")) is None


def test_phrase_attempt_is_skipped_when_nothing_lacks_a_phrase(tmp_path):
    scene = _sentence()
    syllabus = _word_syllabus().with_sentences([scene])
    ctx = _phrase_ctx(tmp_path, syllabus, json.dumps({"phrases": []}))
    for subject in ("rice", scene.text_sha):
        ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject=subject),
                      subject=subject,
                      question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                      answer={"phrase": "already drafted"})
    res = phrase_attempt(ctx)
    assert res.attempted is False and res.spend == {}
    assert _phrase_drafter(ctx).prompts == []


def test_phrase_attempt_records_spend_under_llm_phrase(tmp_path):
    syllabus = _word_syllabus()
    ctx = _phrase_ctx(tmp_path, syllabus,
                      json.dumps({"phrases": [{"subject": "rice", "phrase": "bowl of rice"}]}))
    res = phrase_attempt(ctx)
    assert res.spend["llm-phrase"].asks == 1


def test_phrase_attempt_raises_and_caches_nothing_for_an_answer_phrasing_no_asked_item(
        tmp_path, caplog):
    """Fix round 2 finding 2 (spec 3 section 2): an answer phrasing none
    of the asked items is not a recognized answer -- the real llm-phrase
    wiring (LlmBackend.recognize) raises, so no per-subject row is
    appended and a warning names how many items were asked, rather than
    the empty answer becoming a permanent cache hit for a stable lacking
    set."""
    syllabus = _word_syllabus()
    backend = _real_phrase_backend(json.dumps({"phrases": []}))
    ctx = _sourcing(tmp_path, syllabus, backends={"llm-phrase": backend}, assess={})
    with caplog.at_level(logging.WARNING):
        with pytest.raises(TransportError):
            phrase_attempt(ctx)
    assert drafted_phrase(ctx.db.assessments_of("rice")) is None
    assert any("1" in r.message for r in caplog.records)   # names the count asked


def test_phrase_attempt_caches_a_partial_answers_rows_and_leaves_the_rest_lacking(tmp_path):
    """An answer phrasing SOME of the asked items is recognized and
    cached as today; the omitted subject simply re-asks next run (a
    shrunken lacking set)."""
    scene = _sentence()
    syllabus = _word_syllabus().with_sentences([scene])   # two open picture needs
    backend = _real_phrase_backend(json.dumps({"phrases": [
        {"subject": "rice", "phrase": "bowl of rice"}]}))   # scene.text_sha omitted
    ctx = _sourcing(tmp_path, syllabus, backends={"llm-phrase": backend}, assess={})
    res = phrase_attempt(ctx)
    assert res.attempted
    assert drafted_phrase(ctx.db.assessments_of("rice")) == "bowl of rice"
    assert drafted_phrase(ctx.db.assessments_of(scene.text_sha)) is None


def test_phrase_prompt_delimits_each_need_as_deck_data(tmp_path):
    scene = _sentence()
    syllabus = _word_syllabus().with_sentences([scene])
    ctx = _phrase_ctx(tmp_path, syllabus, json.dumps({"phrases": []}))
    phrase_attempt(ctx)
    prompt = _phrase_drafter(ctx).prompts[0]
    assert UNTRUSTED in prompt
    assert "subject: rice" in prompt and deck_field("ข้าว") in prompt
    assert deck_field("rice (cooked)") in prompt
    assert f"subject: {scene.text_sha}" in prompt and deck_field(scene.text) in prompt
    assert deck_field(scene.gloss) in prompt
    assert 'Output JSON only: {"phrases":' in prompt


def test_phrase_attempt_ignores_an_answer_naming_a_subject_it_never_asked_for(tmp_path):
    syllabus = _word_syllabus()
    ctx = _phrase_ctx(tmp_path, syllabus,
                      json.dumps({"phrases": [{"subject": "not-asked", "phrase": "irrelevant"}]}))
    phrase_attempt(ctx)
    assert drafted_phrase(ctx.db.assessments_of("not-asked")) is None


def test_phrase_prompt_names_the_target_word_the_category_and_asks_for_two_forms(tmp_path):
    """Spec 3 r36 section 5: the drafter searches for the cue, not the
    topic -- it is told what the target word contributes (a sentence)
    or the category (a word), the cue criteria, and answers a phrase
    and keywords."""
    scene = _sentence()                       # its target word is the last used word
    syllabus = _word_syllabus().with_sentences([scene])
    ctx = _phrase_ctx(tmp_path, syllabus, json.dumps({"phrases": []}))
    phrase_attempt(ctx)
    prompt = _phrase_drafter(ctx).prompts[0]
    target = syllabus.word(syllabus.last_used_word(scene))
    assert f"target: {deck_field(target.thai)} ({deck_field(target.meaning)})" in prompt
    assert f"kind: word  thai: {deck_field('ข้าว')}" in prompt       # ข้าว: rice
    assert f"category: {deck_field('Food')}" in prompt
    assert "memory cue" in prompt and "keywords" in prompt and "at most ten words" in prompt
    assert 'Output JSON only: {"phrases": [{"subject": "...", "phrase": "...", "keywords": "..."}]}' in prompt
    assert UNTRUSTED in prompt


def test_a_scene_using_no_targeted_word_is_still_asked_for_a_query(tmp_path):
    """Fix round 1: Syllabus.last_used_word raises for a sentence using
    no targeted word (as `candidate_targets` and `_sentence_order_key`
    already tolerate). Its item line falls back to text and gloss with no
    `target:` clause -- the scene still deserves a query -- and the ask
    still goes out for every other need beside it."""
    orphan = _sentence(text="กิน", gloss="someone eats",   # กิน: eat
                       clauses=((WordId("eat"),),))
    syllabus = _word_syllabus().with_words(
        (word("rice", "ข้าว", "rice (cooked)"), word("eat", "กิน", "eat"))
    ).with_sentences([orphan])
    with pytest.raises(ValueError):                        # the fixture's own premise
        syllabus.last_used_word(orphan)
    ctx = _phrase_ctx(tmp_path, syllabus, json.dumps({"phrases": []}))
    assert phrase_attempt(ctx).attempted
    prompt = _phrase_drafter(ctx).prompts[0]
    line = next(ln for ln in prompt.splitlines()
                if ln.startswith(f"- subject: {orphan.text_sha}"))
    assert f"text: {deck_field('กิน')}" in line and f"gloss: {deck_field('someone eats')}" in line
    assert "target:" not in line
    assert "- subject: rice  kind: word" in prompt        # the other need still asked


def test_phrase_attempt_records_both_forms_and_only_the_phrase_when_none_came(tmp_path):
    syllabus = _word_syllabus()
    ctx = _phrase_ctx(tmp_path, syllabus, json.dumps({"phrases": [
        {"subject": "rice", "phrase": "a bowl of steamed rice", "keywords": "rice bowl steam"}]}))
    phrase_attempt(ctx)
    assert drafted_queries(ctx.db.assessments_of("rice")) == DraftedQuery("a bowl of steamed rice",
                                                                          "rice bowl steam")
    row = [r for r in ctx.db.assessments_of("rice") if r.question.get("provides") == "phrase"][-1]
    assert row.answer == {"phrase": "a bowl of steamed rice", "keywords": "rice bowl steam"}
    ctx2 = _phrase_ctx(tmp_path / "two", syllabus, json.dumps({"phrases": [
        {"subject": "rice", "phrase": "rice on a banana leaf"}]}))
    phrase_attempt(ctx2)
    row = [r for r in ctx2.db.assessments_of("rice") if r.question.get("provides") == "phrase"][-1]
    assert row.answer == {"phrase": "rice on a banana leaf"}


def test_a_source_is_asked_in_the_form_it_declares(tmp_path, monkeypatch):
    """Every current source consumes the phrase; a keywords source (Flickr,
    when wired) gets the head terms, and a direction for either."""
    ctx, search, _judge = _picture_ctx(tmp_path, phrase=None)
    ctx.db.append(port="provide", backend="llm", key=PhraseKey(subject="rice"), subject="rice",
                  question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                  answer={"phrase": "a bowl of steamed rice", "keywords": "rice bowl steam"})
    assert picture_query_for(ctx, Need("rice", "picture"), "openverse") == "a bowl of steamed rice"
    assert picture_query_for(ctx, Need("rice", "picture"), "illustrator") == "a bowl of steamed rice"
    monkeypatch.setitem(QUERY_FORMS, "openverse", "keywords")
    assert picture_query_for(ctx, Need("rice", "picture"), "openverse") == "rice bowl steam"
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.queries == ["rice bowl steam"]
    assert _outcome(ctx.db, "rice", "picture", "openverse").question["query"] == "rice bowl steam"


# --- adjudication_attempt: one pronunciation ask per uncorroborated word ----

def test_adjudication_attempt_collects_one_question_per_disputed_word(tmp_path):
    syllabus = Syllabus(words=(word("rice", "ข้าว", "rice", corroboration="disputed"),
                               word("eat", "กิน", "eat")),
                        targets=(target("rice/receptive", "rice"),))
    ctx = _sourcing(tmp_path, syllabus, backends={}, assess={"judge": _batch_judge()})
    res = adjudication_attempt(ctx)
    assert res.attempted and [q.question.subject for q in res.questions] == ["rice"]
    q = res.questions[0].question
    assert q.role == "pronunciation-for-word" and q.kind == "pronunciation"
    assert q.params == {"thai": "ข้าว", "meaning": "rice"}


def test_adjudication_attempt_is_not_attempted_when_every_word_is_corroborated(tmp_path):
    ctx = _sourcing(tmp_path, _word_syllabus(), backends={}, assess={"judge": _batch_judge()})
    assert adjudication_attempt(ctx).attempted is False


# --- retire_sentence: one retirement mechanism (F13 and the comment pass) ----

def _adopted_sentence_ctx(tmp_path):
    rice, eat = word("rice", "ข้าว", "rice"), word("eat", "กิน", "eat")   # ข้าว: rice, กิน: eat
    s = compose_sentence(((eat.id, rice.id),), thai_of(rice, eat), gloss="eat rice")
    syllabus = Syllabus(words=(rice, eat), targets=(target("rice/receptive", "rice"),
                                                    target("eat/receptive", "eat")),
                        sentences=(s,), frequency={"eat": 1, "rice": 2})
    ctx = _sourcing(tmp_path, syllabus, backends={}, assess={"judge": _batch_judge()})
    ctx.db.add_sentence(text_sha=s.text_sha, text=s.text, clauses=s.clauses, gloss=s.gloss,
                        voice=s.voice, source="test", origin="fixture", licence="cc0",
                        acquired=date(2026, 1, 1))
    ctx.guard = Guard()
    return ctx, s


def test_retire_sentence_leaves_a_row_with_reason_hint_text_and_origin_then_deletes(tmp_path):
    ctx, s = _adopted_sentence_ctx(tmp_path)
    retire_sentence(ctx, s.text_sha, reason="unnatural", replacement_hint="shorter",
                    derived_from=CommentRef("c1c1c1c1c1c1c1c1", "1"))
    (row,) = [r for r in ctx.db.assessments_of(s.text_sha) if r.port == "attempt"]
    assert row.backend == "run"
    assert row.question == {"kind": "retirement", "subject_kind": "sentence",
                            "reason": "unnatural", "candidates": 0, "text": s.text,
                            "replacement_hint": "shorter", "comment_sha": "c1c1c1c1c1c1c1c1",
                            "prompt_version": "1"}
    assert row.answer == {"retired": True}
    assert ctx.db.all_sentences() == [] and ctx.syllabus.sentences == ()
    assert ctx.guard.removals == {"sentences": 1}
    assert retired_texts(ctx.db) == frozenset({s.text_sha})


def test_retire_sentence_for_f13_carries_no_hint_and_no_origin(tmp_path):
    ctx, s = _adopted_sentence_ctx(tmp_path)
    retire_sentence(ctx, s.text_sha, reason="recording exhausted")
    (row,) = [r for r in ctx.db.assessments_of(s.text_sha) if r.port == "attempt"]
    assert row.question == {"kind": "retirement", "subject_kind": "sentence",
                            "reason": "recording exhausted", "candidates": 0, "text": s.text}


# --- the comment pass (spec 3 r30 section 5) --------------------------------

def _readings_json(*items):
    return json.dumps({"readings": [
        {"comment": sha_, "reading": reading, "actions": list(actions),
         "unactionable": list(unactionable)}
        for sha_, reading, actions, unactionable in items]}, ensure_ascii=False)


def _comment_ctx(tmp_path, *, syllabus=None, judge=None, parse_text='{"parses": []}'):
    """A ctx whose llm-comment answer is set after the comment is written
    (the answer names the comment's own sha): `ctx.provider._backends`
    is a dict, so the test swaps the backend in."""
    syllabus = syllabus if syllabus is not None else _word_syllabus()
    return _sourcing(tmp_path, syllabus,
                     backends={"llm-comment": _Llm('{"readings": []}'),
                               "llm-parse": _Llm(parse_text)},
                     assess={"judge": judge or _batch_judge()})


def _answer_with(ctx, text):
    ctx.provider._backends["llm-comment"] = _Llm(text)


def _pass_picture(ctx, subject, sha_):
    """A judge pass on `sha_` under the ctx's own rubric, so current_best
    ranks it."""
    rubric = ctx.rubrics["picture-for-word"]
    ctx.db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(rubric, sha_, subject, "picture-for-word"), subject=subject,
                  question={"role": "picture-for-word", "artifact_sha": sha_, "rubric": rubric,
                            "kind": "picture", "subject_kind": "word"},
                  answer={"value": True, "evidence": "ok"})


def test_comment_attempt_is_not_attempted_with_no_unread_comment(tmp_path):
    ctx = _comment_ctx(tmp_path)
    assert comment_attempt(ctx).attempted is False


def test_comment_attempt_hands_the_comment_with_its_card_meaning_and_subject_facts(tmp_path):
    ctx = _comment_ctx(tmp_path)
    _seed_phrase(ctx, "rice", "steamed rice")
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="that is noodles",
                   shown={"picture": "a" * 64, "recordings": []}, subject_kind="word")
    (c,) = comments(ctx.db)
    llm = _Llm(_readings_json((c.comment_sha, "noodles shown", [], [])))
    ctx.provider._backends["llm-comment"] = llm
    res = comment_attempt(ctx)
    assert res.attempted and res.comments_read == 1 and res.comment_actions == 0
    prompt = llm.prompts[0]
    assert f"comment {c.comment_sha}" in prompt
    assert "<deck-field>that is noodles</deck-field>" in prompt
    assert "word / reading" in prompt and "Front shows the Thai" in prompt
    assert "<deck-field>ข้าว</deck-field>" in prompt and "rice (cooked)" in prompt   # ข้าว: rice
    assert f"picture {'a' * 64}" in prompt and "steamed rice" in prompt
    for name in ("direction", "retire_sentence", "replacement_sentence", "rate", "gloss_on",
                 "none"):
        assert name in prompt
    row = reading_of(ctx.db.assessments_of("rice"), c.comment_sha, "1")
    assert row.answer == {"reading": "noodles shown", "actions": [], "unactionable": []}
    assert row.question["kind"] == "comment-reading" and row.question["card_kind"] == "reading"
    # read once: the next call hands nothing
    assert comment_attempt(ctx).attempted is False


def test_direction_and_rate_actions_write_the_typed_rows_marked_with_the_comment(tmp_path):
    ctx = _comment_ctx(tmp_path)
    _pass_picture(ctx, "rice", _seed_current_picture(ctx, "rice"))   # a current-best picture
    current = current_best_of(ctx, "rice", "picture").artifact_sha
    assert current is not None
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="noodles",
                   shown={"picture": current, "recordings": []}, subject_kind="word")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "wrong food", [
        {"action": "direction", "kind": "picture", "text": "bowl of steamed jasmine rice"},
        {"action": "rate", "kind": "picture", "value": 1}], ["make it bigger"])))
    res = comment_attempt(ctx)
    assert res.comment_actions == 2 and res.comment_unactionable == 1
    rows = ctx.db.assessments_of("rice")
    assert latest_phrase(rows) == "bowl of steamed jasmine rice"
    direction = next(r for r in rows if r.question.get("kind") == "direction")
    assert direction.question["comment_sha"] == c.comment_sha
    assert direction.question["prompt_version"] == "1"
    rating = next(r for r in rows if r.question.get("kind") == "rating")
    assert rating.answer["value"] == "unacceptable-none" and rating.question["artifact_sha"] == current
    assert current_best_of(ctx, "rice", "picture").artifact_sha is None
    reading = reading_of(rows, c.comment_sha)
    assert [a["outcome"] for a in reading.answer["actions"]] == ["done", "done"]


def test_a_rejection_of_a_picture_no_longer_current_is_refused_not_written(tmp_path):
    """The screen refuses a stale rejection (reviewserver._refuses_stale_rejection):
    so does the pass, else a newer current-best would be wiped by a
    comment about an older one."""
    ctx = _comment_ctx(tmp_path)
    _pass_picture(ctx, "rice", _seed_current_picture(ctx, "rice"))
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="noodles",
                   shown={"picture": "f" * 64, "recordings": []}, subject_kind="word")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "wrong food",
                                      [{"action": "rate", "kind": "picture", "value": 1}], [])))
    res = comment_attempt(ctx)
    assert res.comment_actions == 0 and res.comment_unactionable == 1
    assert not [r for r in ctx.db.assessments_of("rice") if r.question.get("kind") == "rating"]
    (action,) = reading_of(ctx.db.assessments_of("rice"), c.comment_sha).answer["actions"]
    assert action["outcome"] == "refused" and "no longer" in action["reason"]


def test_retire_sentence_action_retires_with_reason_and_hint_and_a_replacement_is_drafted(tmp_path):
    ctx, s = _adopted_sentence_ctx(tmp_path)
    ctx.provider._backends["llm-parse"] = _Llm(json.dumps({"parses": [
        {"text": "กินข้าวครับ", "clauses": [["eat", "rice", "polite-particle"]]}]},
        ensure_ascii=False))   # กินข้าวครับ: eat rice (polite)
    ctx.provider._backends["llm-comment"] = _Llm('{"readings": []}')
    particle = word("polite-particle", "ครับ", "polite particle (male)", speaker="male")  # ครับ: kráp
    # a glue word carries its own sentence-introduced Target (spec 1
    # section 3's gate: every word a sentence names is targeted)
    ctx.syllabus = replace(ctx.syllabus.with_words((*ctx.syllabus.words, particle)),
                           targets=(*ctx.syllabus.targets,
                                    target("polite-particle/receptive", "polite-particle",
                                           introduction="sentence")))
    append_comment(ctx.db, subject=s.text_sha, card_id=s.text_sha, kind="listening",
                   text="too blunt", shown={"text_sha": s.text_sha}, subject_kind="sentence")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "the sentence lacks a particle", [
        {"action": "retire_sentence", "reason": "too blunt", "replacement_hint": "add a particle"},
        {"action": "replacement_sentence", "thai": "กินข้าวครับ", "gloss": "eat rice (polite)"}],
        [])))
    res = comment_attempt(ctx)
    assert res.retired == 1 and res.comment_actions == 2
    assert ctx.syllabus.sentences == () and ctx.db.all_sentences() == []
    retirement = retirements(ctx.db)[s.text_sha]
    assert retirement.reason == "too blunt" and retirement.replacement_hint == "add a particle"
    # the replacement is a draft in the record with a judge question collected for the batch
    drafts = [r for r in rows_for(ctx.db, DRAFT_SUBJECT, "sentence") if r.port == "provide"]
    assert len(drafts) == 1 and drafts[0].question["comment_sha"] == c.comment_sha
    assert '"กินข้าวครับ"' in drafts[0].answer["items"][0]
    assert [d.text for d in sentence_drafts(ctx.db)] == ["กินข้าวครับ"]
    (q,) = res.questions
    assert q.question.role == "sentence-for-target" and q.question.params["text"] == "กินข้าวครับ"
    assert q.question.params["gloss"] == "eat rice (polite)"
    (retire, replaced) = reading_of(ctx.db.assessments_of(s.text_sha),
                                    c.comment_sha).answer["actions"]
    assert retire["outcome"] == "done" and replaced["outcome"] == "done"


def test_a_judge_that_dies_at_the_replacement_check_reports_the_counts_it_already_wrote(tmp_path):
    """Fix round 1 finding 1: the retirement, the rows and the reading are
    already on the record when the judge is asked about the replacement
    draft, so a dead judge there is REPORTED, not raised -- the run must
    be able to say a sentence was deleted and a comment read. The drafts
    keep their provide rows and the run's D2 recovery re-raises their
    questions next run."""
    def dead(prompt, attachments=()):
        raise TransportError("no judge")

    ctx, s = _adopted_sentence_ctx(tmp_path)
    ctx.assessor = Assessor(record=ctx.db, cache=ctx.db, backends={
        "judge": JudgeBackend(model="m", transport="api", complete=dead)})
    ctx.provider._backends["llm-parse"] = _Llm(json.dumps({"parses": [
        {"text": "กินข้าวครับ", "clauses": [["eat", "rice", "polite-particle"]]}]},
        ensure_ascii=False))   # กินข้าวครับ: eat rice (polite)
    particle = word("polite-particle", "ครับ", "polite particle (male)", speaker="male")  # ครับ: kráp
    ctx.syllabus = replace(ctx.syllabus.with_words((*ctx.syllabus.words, particle)),
                           targets=(*ctx.syllabus.targets,
                                    target("polite-particle/receptive", "polite-particle",
                                           introduction="sentence")))
    append_comment(ctx.db, subject=s.text_sha, card_id=s.text_sha, kind="listening",
                   text="too blunt", shown={"text_sha": s.text_sha}, subject_kind="sentence")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "the sentence lacks a particle", [
        {"action": "retire_sentence", "reason": "too blunt", "replacement_hint": "add a particle"},
        {"action": "replacement_sentence", "thai": "กินข้าวครับ", "gloss": "eat rice (polite)"}],
        [])))
    res = comment_attempt(ctx)
    assert res.judge_unreachable is True and res.questions == []
    assert res.retired == 1 and res.comments_read == 1 and res.comment_actions == 2
    assert ctx.db.all_sentences() == []                     # the retirement stands
    assert [d.text for d in sentence_drafts(ctx.db)] == ["กินข้าวครับ"]   # the draft stands
    assert reading_of(ctx.db.assessments_of(s.text_sha), c.comment_sha) is not None


def test_a_replacement_that_does_not_parse_or_fails_acceptance_is_refused_with_the_reason(tmp_path):
    ctx, s = _adopted_sentence_ctx(tmp_path)
    ctx.provider._backends["llm-parse"] = _Llm('{"parses": []}')
    append_comment(ctx.db, subject=s.text_sha, card_id=s.text_sha, kind="listening",
                   text="odd", shown={}, subject_kind="sentence")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "odd", [
        {"action": "replacement_sentence", "thai": "ข้าวกิน", "gloss": "rice eats"}], [])))
    # ข้าวกิน: rice eat (reversed)
    res = comment_attempt(ctx)
    assert res.comment_actions == 0 and res.comment_unactionable == 1 and res.questions == []
    (action,) = reading_of(ctx.db.assessments_of(s.text_sha), c.comment_sha).answer["actions"]
    assert action["outcome"] == "refused" and action["reason"] == "no parse returned for this text"


def test_an_answer_naming_an_unhanded_comment_is_ignored_and_an_omitted_one_is_recorded(tmp_path):
    """D7: a handed comment the answer names no reading for gets its own
    reading row saying so -- the same cached answer would otherwise leave
    it unread for ever."""
    ctx = _comment_ctx(tmp_path)
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="a",
                   shown={}, subject_kind="word")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json(("0000000000000000", "not asked", [
        {"action": "rate", "kind": "picture", "value": 4}], [])))
    res = comment_attempt(ctx)
    assert res.attempted and res.comments_read == 1 and res.comment_actions == 0
    assert res.comment_unactionable == 1
    assert not [r for r in ctx.db.assessments_of("rice") if r.question.get("kind") == "rating"]
    reading = reading_of(ctx.db.assessments_of("rice"), c.comment_sha, "1")
    assert reading.answer == {"reading": "", "actions": [],
                              "unactionable": ["the model gave no reading"]}
    assert comment_attempt(ctx).attempted is False


def test_gloss_on_writes_a_gloss_on_row_not_a_direction(tmp_path):
    ctx = _comment_ctx(tmp_path)
    _seed_phrase(ctx, "rice", "steamed rice")
    append_comment(ctx.db, subject="rice", card_id="rice", kind="production", text="show the word",
                   shown={}, subject_kind="word")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "wants the gloss on the front",
                                      [{"action": "gloss_on", "word": "rice"}], [])))
    comment_attempt(ctx)
    rows = ctx.db.assessments_of("rice")
    assert gloss_on_requested(rows) is True
    assert latest_phrase(rows_for(ctx.db, "rice", "picture")) == "steamed rice"   # query untouched


def test_a_none_action_is_an_act_the_reading_records_not_an_unactionable_request(tmp_path):
    """D5: `none(remark)` executes nothing and still counts as an action."""
    ctx = _comment_ctx(tmp_path)
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="nice card",
                   shown={}, subject_kind="word")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "praise", [
        {"action": "none", "remark": "nothing for the deck to do"}], [])))
    res = comment_attempt(ctx)
    assert res.comments_read == 1 and res.comment_actions == 1 and res.comment_unactionable == 0
    (action,) = reading_of(ctx.db.assessments_of("rice"), c.comment_sha).answer["actions"]
    assert action["action"] == "none" and action["outcome"] == "done"


def test_a_comment_on_a_subject_no_longer_in_the_syllabus_is_closed_unactionable(tmp_path):
    """It is never handed (this ctx has no llm-comment backend at all, so
    an ask would raise) and never raises building facts for a subject
    that is gone: one reading row closes it."""
    ctx, s = _adopted_sentence_ctx(tmp_path)
    append_comment(ctx.db, subject=s.text_sha, card_id=s.text_sha, kind="listening",
                   text="gone", shown={}, subject_kind="sentence")
    (c,) = comments(ctx.db)
    retire_sentence(ctx, s.text_sha, reason="recording exhausted")
    res = comment_attempt(ctx)
    assert res.comments_read == 1 and res.comment_unactionable == 1 and res.comment_actions == 0
    reading = reading_of(ctx.db.assessments_of(s.text_sha), c.comment_sha, "1")
    assert reading.answer == {"reading": "", "actions": [],
                              "unactionable": ["subject not in the syllabus"]}
    assert comment_attempt(ctx).attempted is False   # closed, not re-read


def test_a_comment_whose_recorded_subject_kind_disagrees_with_the_syllabus_is_closed(tmp_path):
    ctx = _comment_ctx(tmp_path)
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="hm",
                   shown={}, subject_kind="grapheme")   # the syllabus holds "rice" as a word
    (c,) = comments(ctx.db)
    res = comment_attempt(ctx)
    assert res.comments_read == 1 and res.comment_unactionable == 1
    assert ctx.provider._backends["llm-comment"].prompts == []   # nothing was asked
    reading = reading_of(ctx.db.assessments_of("rice"), c.comment_sha, "1")
    assert reading.answer["unactionable"] == ["subject kind mismatch"]


def test_a_pair_and_a_grapheme_subject_build_their_facts_for_the_prompt(tmp_path):
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: gài
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",   # ก: the letter k
                               consonant_class="mid", keyword_word=chicken)
    pairs = _pair_syllabus()
    syllabus = replace(pairs, words=(*pairs.words, chicken), graphemes=(grapheme,))
    ctx = _comment_ctx(tmp_path, syllabus=syllabus)
    append_comment(ctx.db, subject="p1", card_id="p1/white", kind="recognition",
                   text="they sound the same", shown={}, subject_kind="pair")
    append_comment(ctx.db, subject="ก", card_id="ก", kind="reading",
                   text="the keyword is odd", shown={}, subject_kind="grapheme")
    pair_comment, grapheme_comment = comments(ctx.db)
    llm = _Llm(_readings_json((pair_comment.comment_sha, "hard pair", [], []),
                              (grapheme_comment.comment_sha, "keyword", [], [])))
    ctx.provider._backends["llm-comment"] = llm
    res = comment_attempt(ctx)
    assert res.comments_read == 2 and res.comment_unactionable == 0
    prompt = llm.prompts[0]
    assert "minimal pair p1 on confusion tone:rising-vs-low" in prompt
    assert "minimal_pair / recognition" in prompt and "<deck-field>ขาว</deck-field>" in prompt
    assert "grapheme <deck-field>ก</deck-field> (consonant), sound k" in prompt
    assert "grapheme / reading" in prompt and "keyword word chicken" in prompt


def test_direction_and_rate_are_refused_on_a_subject_that_has_no_such_need(tmp_path):
    """A grapheme (or a pair) has no picture and no recording need: only a
    word and a sentence do (derivations.available_needs), and the comment
    vocabulary names no other artifact kind (record._COMMENT_ARTIFACT_KINDS).
    Without the guard `authority.role_for` falls back to the word role and
    both rows are written under a subject no fold over them ever reads --
    dead rows the screen reports as actions taken."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: gài
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",   # ก: the letter k
                               consonant_class="mid", keyword_word=chicken)
    syllabus = replace(_word_syllabus(), words=(word("rice", "ข้าว", "rice (cooked)"), chicken),
                       graphemes=(grapheme,))
    ctx = _comment_ctx(tmp_path, syllabus=syllabus)
    append_comment(ctx.db, subject="ก", card_id="ก", kind="reading", text="wrong picture",
                   shown={"picture": "a" * 64, "recordings": []}, subject_kind="grapheme")
    (c,) = comments(ctx.db)
    _answer_with(ctx, _readings_json((c.comment_sha, "the keyword picture is wrong", [
        {"action": "rate", "kind": "picture", "value": 1},
        {"action": "direction", "kind": "picture", "text": "a hen"}], [])))
    res = comment_attempt(ctx)
    assert res.comments_read == 1 and res.comment_actions == 0 and res.comment_unactionable == 2
    rows = ctx.db.assessments_of("ก")
    assert not [r for r in rows if r.question.get("kind") in ("rating", "direction")]
    actions = reading_of(rows, c.comment_sha, "1").answer["actions"]
    assert [a["outcome"] for a in actions] == ["refused", "refused"]
    assert {a["reason"] for a in actions} == {"the subject has no picture need"}


def test_only_the_oldest_comments_up_to_the_cap_are_handed_in_one_run(tmp_path):
    ctx = _comment_ctx(tmp_path)
    for i in range(COMMENTS_PER_ASK + 1):
        append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text=f"note {i}",
                       shown={}, subject_kind="word")
    written = comments(ctx.db)
    assert len(written) == COMMENTS_PER_ASK + 1
    llm = _Llm(_readings_json(*((c.comment_sha, "read", [], []) for c in written)))
    ctx.provider._backends["llm-comment"] = llm
    res = comment_attempt(ctx)
    assert res.comments_read == COMMENTS_PER_ASK
    rows = ctx.db.assessments_of("rice")
    assert reading_of(rows, written[-1].comment_sha) is None    # held back, no reading at all
    assert all(reading_of(rows, c.comment_sha) is not None for c in written[:COMMENTS_PER_ASK])
    assert f"comment {written[-1].comment_sha}" not in llm.prompts[0]
    # the next run hands the one held back
    assert comment_attempt(ctx).comments_read == 1


def test_every_answer_item_is_read_and_the_first_reading_of_a_comment_wins(tmp_path):
    ctx = _comment_ctx(tmp_path)
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="a",
                   shown={}, subject_kind="word")
    (c,) = comments(ctx.db)

    class _TwoItems(_Llm):
        def fetch(self, q):
            self.prompts.append(q.params["prompt"])
            return RawAnswer(items=(_readings_json((c.comment_sha, "first", [], [])),
                                    _readings_json((c.comment_sha, "second", [], []))))

    ctx.provider._backends["llm-comment"] = _TwoItems("")
    assert comment_attempt(ctx).comments_read == 1
    reading = reading_of(ctx.db.assessments_of("rice"), c.comment_sha)
    assert reading.answer["reading"] == "first"


def test_a_learner_direction_is_the_query_named_beside_the_shown_picture(tmp_path):
    ctx = _comment_ctx(tmp_path)
    _seed_phrase(ctx, "rice", "steamed rice")
    append_direction(ctx.db, subject="rice", role="picture-for-word", text="a bowl of rice")
    append_comment(ctx.db, subject="rice", card_id="rice", kind="reading", text="wrong",
                   shown={"picture": "b" * 64, "recordings": []}, subject_kind="word")
    (c,) = comments(ctx.db)
    llm = _Llm(_readings_json((c.comment_sha, "wrong picture", [], [])))
    ctx.provider._backends["llm-comment"] = llm
    comment_attempt(ctx)
    assert "search query: <deck-field>a bowl of rice</deck-field>" in llm.prompts[0]


# --- the illustrator (spec 3 r34 section 5) ---------------------------------

class _Illustrator:
    """A fake illustrator backend answering the item shape
    provider.IllustratorBackend answers: the sha of bytes already in the
    media store, provenance generated."""

    def __init__(self, media, answer="image"):
        self.media, self.answer, self.queries = media, answer, []

    def cache_key(self, q):
        return ProvideKey(source="illustrator", kind="m", query=q.params["query"])

    def fetch(self, q):
        self.queries.append(q.params["query"])
        if self.answer == "quota":
            raise QuotaExhausted("illustrator")
        if self.answer == "declined":
            return RawAnswer(items=(), cost=0.0)
        ingest = self.media.add_image(_jpeg_bytes("https://x/good-drawn.jpg"), "jpg")
        return RawAnswer(items=({"sha": ingest.sha, "ext": ingest.ext, "source": "generated",
                                 "origin": "m", "licence": "generated"},), cost=0.067)


def _illustrated_ctx(tmp_path, answer="image"):
    ctx, _search, _judge = _picture_ctx(tmp_path, phrase="rice food")
    ctx.provider._backends["illustrator"] = _Illustrator(ctx.media_store, answer)
    return ctx


def test_the_illustrator_attempt_ingests_the_generated_picture_with_provenance_generated(tmp_path):
    """Spec 3 r34 section 5: the item already carries its sha, so no
    imgfetch; the media row says generated/generated; the candidate is
    judged like any other (inline judge here: a green image passes)."""
    ctx = _illustrated_ctx(tmp_path)
    res = attempt(ctx, Need("rice", "picture"), "illustrator")
    assert res.attempted and res.questions == []
    row = _outcome(ctx.db, "rice", "picture", "illustrator")
    (sha,) = row.answer["candidates"]
    assert row.answer["outcome"] == "candidates" and row.answer["tried"] == []
    prov = ctx.db.media_provenance(sha)
    assert prov["source"] == "generated" and prov["licence"] == "generated" and prov["origin"] == "m"
    assert not [r for r in rows_for(ctx.db, "rice", "picture") if r.backend == "imgfetch"]
    assert current_best_of(ctx, "rice", "picture").artifact_sha == sha
    assert res.spend["illustrator"].asks == 1 and res.spend["illustrator"].cost == 0.067


def test_the_illustrator_is_asked_under_the_needs_query_and_collects_a_fit_question_under_batch(
        tmp_path):
    ctx = _illustrated_ctx(tmp_path)
    ctx.assessor = Assessor(record=ctx.db, cache=ctx.db, backends={"judge": _batch_judge()})
    res = attempt(ctx, Need("rice", "picture"), "illustrator")
    assert ctx.provider._backends["illustrator"].queries == ["rice food"]
    assert [q.question.role for q in res.questions] == ["picture-for-word"]


def test_a_declined_generation_is_a_nothing_outcome(tmp_path):
    ctx = _illustrated_ctx(tmp_path, answer="declined")
    res = attempt(ctx, Need("rice", "picture"), "illustrator")
    row = _outcome(ctx.db, "rice", "picture", "illustrator")
    assert res.attempted and row.answer["outcome"] == "nothing" and row.answer["candidates"] == []


def test_the_illustrators_quota_propagates_with_no_outcome_row(tmp_path):
    ctx = _illustrated_ctx(tmp_path, answer="quota")
    with pytest.raises(QuotaExhausted):
        attempt(ctx, Need("rice", "picture"), "illustrator")
    assert not [r for r in rows_for(ctx.db, "rice", "picture")
                if r.port == "attempt" and r.backend == "illustrator"]


def test_re_ingesting_the_same_generated_sha_neither_fails_nor_duplicates_the_media_row(tmp_path):
    """`add_media` is idempotent on sha (store.add_media: insert or
    ignore), so a second attempt over the cached answer re-records the
    same candidate without a second provenance row or an error."""
    ctx = _illustrated_ctx(tmp_path)
    first = attempt(ctx, Need("rice", "picture"), "illustrator")
    (sha,) = _outcome(ctx.db, "rice", "picture", "illustrator").answer["candidates"]
    before = ctx.db.media_provenance(sha)
    second = attempt(ctx, Need("rice", "picture"), "illustrator")
    assert first.attempted and second.attempted
    assert _outcome(ctx.db, "rice", "picture", "illustrator").answer["candidates"] == [sha]
    assert ctx.db.media_provenance(sha) == before
    assert ctx.db.add_media(sha=sha, kind="picture", ext="jpg", source="generated",
                            origin="m", licence="generated", acquired=date(2026, 9, 3)) is False


# --- the outcome row carries its query (spec 3 r35 section 6) ------------------

def test_a_picture_outcome_row_carries_the_query_it_was_asked_with(tmp_path):
    """Spec 3 r35 section 6: the row names the query, so tried_sources can
    fold per (need, query)."""
    ctx, _search, _judge = _picture_ctx(tmp_path, phrase="rice food")
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert _outcome(ctx.db, "rice", "picture", "openverse").question["query"] == "rice food"


def test_the_illustrators_outcome_row_carries_the_query_too(tmp_path):
    ctx = _illustrated_ctx(tmp_path)
    attempt(ctx, Need("rice", "picture"), "illustrator")
    assert _outcome(ctx.db, "rice", "picture", "illustrator").question["query"] == "rice food"


def test_a_recording_outcome_row_carries_no_query(tmp_path):
    ctx, _tts = _recording_ctx(tmp_path, _word_syllabus())
    attempt(ctx, Need("rice", "recording"), "tts")
    assert "query" not in _outcome(ctx.db, "rice", "recording", "tts").question


def test_a_picture_attempt_at_a_source_already_asked_under_another_query_is_a_requery(tmp_path):
    """Spec 3 r35 section 7: RunReport.requeried counts these; the first
    ask under the phrase is not one; the tried-url filter still applies
    across queries (section 5), so the same hit is not fetched twice."""
    ctx, search, _judge = _picture_ctx(tmp_path, phrase="rice food",
                                       urls=("https://x/bad.jpg",))
    first = attempt(ctx, Need("rice", "picture"), "openverse")
    assert first.requeried is False
    ctx.db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(None, None, "rice", "picture-for-word"), subject="rice",
                  question={"role": "picture-for-word", "kind": "picture"},
                  answer={"value": False, "suggestion": "a heap of rice grains"})
    second = attempt(ctx, Need("rice", "picture"), "openverse")
    assert second.requeried is True and second.attempted
    assert search.queries == ["rice food", "a heap of rice grains"]
    fetched = [r for r in rows_for(ctx.db, "rice", "picture")
               if r.port == "provide" and r.backend == "imgfetch"]
    assert len(fetched) == 1
    assert _outcome(ctx.db, "rice", "picture", "openverse").answer["outcome"] == "nothing"


# --- the grapheme pass: the consonant inventory adopted (spec 3 r40 §5) ----

KO = ConsonantRow(symbol="ก", consonant_class="mid", sound="k", name_thai="กอ ไก่",
                  keyword_thai="ไก่", keyword_gloss="chicken")   # ก: k; ไก่: chicken


def _engines(g2p=None, tone=None):
    """Two fake engines: one monosyllable they agree on unless a test says
    otherwise (thai -> syllables; thai -> tone)."""
    one = (Syllable(segments=("k", "a", ""), vowel_length="short", tone="mid"),)
    return Engines(g2p=(g2p or (lambda thai: one),), tone=tone or (lambda thai: "mid"))


def _reads_normally_except(*unreadable_thai: str):
    """A g2p that reads the same monosyllable as `_engines`'s own default
    for anything but the given strings -- which it reads as thaig2p reads
    a form it cannot read at all. Naming both a phrase and one of its own
    whitespace-separated tokens blocks phonology.py's token-wise fallback
    too (spec 3 r43), so a test can still make a recited name unreadable
    end to end."""
    one = (Syllable(segments=("k", "a", ""), vowel_length="short", tone="mid"),)

    def g2p(thai: str):
        return None if thai in unreadable_thai else one
    return g2p


def _grapheme_ctx(tmp_path, syllabus, *, engines=None,
                  files=("words.yaml", "targets.yaml", "graphemes.yaml",
                        "pairs.yaml", "confusions.yaml")):
    """A ctx with a curated/ directory of its own (the run's writing
    command owns it) and injected engines -- pythainlp never loads. The
    files the adoption passes add rows to are written first, from the
    syllabus's own rows (R-P2: the passes add rows, they never create the
    store); `files` names which of them exist, so a test can leave one
    out. A batch judge backend is wired in unconditionally (unused by the
    grapheme pass, which asks nothing) so the pair search's own judge ask
    has somewhere to land.
    """
    ctx = _sourcing(tmp_path, syllabus, backends={}, assess={"judge": _batch_judge()})
    ctx.curated_dir = tmp_path / "curated"
    ctx.curated_dir.mkdir(parents=True, exist_ok=True)
    if "words.yaml" in files:
        save_words(ctx.curated_dir / "words.yaml",
                   [(w, syllabus.category_of(w.id)) for w in syllabus.words])
    if "targets.yaml" in files:
        save_targets(ctx.curated_dir / "targets.yaml", list(syllabus.targets))
    if "graphemes.yaml" in files:
        save_graphemes(ctx.curated_dir / "graphemes.yaml", list(syllabus.graphemes))
    if "pairs.yaml" in files:
        save_pairs(ctx.curated_dir / "pairs.yaml", list(syllabus.pairs))
    if "confusions.yaml" in files:
        save_confusions(ctx.curated_dir / "confusions.yaml", list(syllabus.confusions))
    ctx.engines = engines or _engines()
    return ctx


def test_a_consonant_whose_keyword_is_vocabulary_adopts_its_name_word_and_row(tmp_path):
    """Design 2026-09-12 §1: 20 of the 44 acrophonic keywords are already
    vocabulary words, matched by `thai`; the recited name is the new Word,
    with both Targets and the category Letter names (spec 1 r16)."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    syllabus = Syllabus(words=(chicken,), targets=(target("chicken/receptive", "chicken"),),
                        categories=(Category(name="Animals", members=frozenset({"chicken"})),))
    ctx = _grapheme_ctx(tmp_path, syllabus)

    result = grapheme_attempt(ctx, consonants=[KO])

    assert result.attempted is True
    assert (result.adopted_graphemes, result.adopted_words, result.adoption_skipped) == (1, 1, 0)
    name = ctx.syllabus.word("name-chicken")
    assert name.thai == "กอ ไก่"                                   # กอ ไก่: the name of ก
    assert name.meaning == GRAPHEME_NAME_MEANING.format(symbol="ก")
    assert ctx.syllabus.category_of("name-chicken") == "Letter names"
    assert [t.id for t in ctx.syllabus.targets if t.word == "name-chicken"] == [
        "name-chicken/receptive", "name-chicken/productive"]
    g = ctx.syllabus.graphemes[0]
    assert (g.symbol, g.kind, g.sound, g.consonant_class) == ("ก", "consonant", "k", "mid")
    assert (g.keyword, g.name_word) == ("chicken", "name-chicken")
    assert ctx.syllabus.name_word_ids == frozenset({"name-chicken"})


def test_the_pass_writes_the_three_curated_files(tmp_path):
    """Spec 2 r17: the run adds rows to the learner's own files under its
    writing command -- rows added, none removed."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    syllabus = Syllabus(words=(chicken,), targets=(target("chicken/receptive", "chicken"),),
                        categories=(Category(name="Animals", members=frozenset({"chicken"})),))
    ctx = _grapheme_ctx(tmp_path, syllabus)

    grapheme_attempt(ctx, consonants=[KO])

    rows = load_words(ctx.curated_dir / "words.yaml")
    assert [(w.id, c) for w, c in rows] == [("chicken", "Animals"),
                                            ("name-chicken", "Letter names")]
    assert [t.id for t in load_targets(ctx.curated_dir / "targets.yaml")] == [
        "chicken/receptive", "name-chicken/receptive", "name-chicken/productive"]
    saved = load_graphemes(ctx.curated_dir / "graphemes.yaml", {w.id: w for w, _ in rows})
    assert [(g.symbol, g.keyword, g.name_word) for g in saved] == [("ก", "chicken",
                                                                    "name-chicken")]


def test_a_pass_without_the_three_files_writes_nothing_and_counts_the_skip(tmp_path):
    """R-P2: the pass adds rows to files the deck already has; a store
    missing one of them is not a store it may rewrite (a truncated
    targets.yaml would lose every Target the deck owns), so the rows are
    skipped with a logged reason and nothing is written."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    syllabus = Syllabus(words=(chicken,), targets=(target("chicken/receptive", "chicken"),),
                        categories=(Category(name="Animals", members=frozenset({"chicken"})),))
    ctx = _grapheme_ctx(tmp_path, syllabus, files=("words.yaml", "graphemes.yaml"))

    result = grapheme_attempt(ctx, consonants=[KO])

    assert result == AttemptResult(attempted=False, adoption_skipped=1)
    assert [(w.id, c) for w, c in load_words(ctx.curated_dir / "words.yaml")] == [
        ("chicken", "Animals")]
    assert not (ctx.curated_dir / "targets.yaml").exists()
    assert ctx.syllabus.graphemes == ()


def test_a_multi_syllable_recited_name_is_written_disputed(tmp_path):
    """R2: the rule tone engine settles one syllable's tone only, so a
    recited name is disputed and the adjudication pass asks the judge next
    run (E4 blocks its cards until then)."""
    two = (Syllable(segments=("k", "ɔ", ""), vowel_length="long", tone="mid"),
           Syllable(segments=("k", "a", ""), vowel_length="short", tone="low"))
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    syllabus = Syllabus(words=(chicken,), targets=(target("chicken/receptive", "chicken"),),
                        categories=(Category(name="Animals", members=frozenset({"chicken"})),))
    ctx = _grapheme_ctx(tmp_path, syllabus,
                        engines=_engines(g2p=lambda thai: two if " " in thai else two[:1]))

    grapheme_attempt(ctx, consonants=[KO])

    assert ctx.syllabus.word("name-chicken").pron.corroboration == "disputed"


NGO = ConsonantRow(symbol="ง", consonant_class="low", sound="ŋ", name_thai="งอ งู",
                   keyword_thai="งู", keyword_gloss="snake")   # ง: ŋ; งู: snake


def test_a_keyword_new_to_the_vocabulary_enters_as_a_closure_word(tmp_path):
    """Design 2026-09-12 §1: a keyword not already a vocabulary word is a
    closure Word -- no category, no Target -- with the slug of its gloss
    for an id."""
    rice = word("rice", "ข้าว", "rice")   # ข้าว: rice
    syllabus = Syllabus(words=(rice,), targets=(target("rice/receptive", "rice"),),
                        categories=(Category(name="Food", members=frozenset({"rice"})),))
    ctx = _grapheme_ctx(tmp_path, syllabus)

    result = grapheme_attempt(ctx, consonants=[NGO])

    assert (result.adopted_graphemes, result.adopted_words) == (1, 2)
    snake = ctx.syllabus.word("snake")
    assert (snake.thai, snake.meaning) == ("งู", "snake")       # งู: snake
    assert snake.pron.corroboration == "engines_agree"
    assert ctx.syllabus.category_of("snake") is None
    assert [t.id for t in ctx.syllabus.targets if t.word == "snake"] == []
    assert ctx.syllabus.graphemes[0].keyword == "snake"
    assert ctx.syllabus.word("name-snake").thai == "งอ งู"       # งอ งู: the name of ง


def test_a_keyword_gloss_already_taken_is_suffixed(tmp_path):
    """The live convention (`delicious-2`): the slug is suffixed until it
    is free, so an existing `snake` word with another form keeps its id."""
    other = word("snake", "อสรพิษ", "snake")   # อสรพิษ: snake (venomous)
    syllabus = Syllabus(words=(other,))
    ctx = _grapheme_ctx(tmp_path, syllabus)

    grapheme_attempt(ctx, consonants=[NGO])

    assert ctx.syllabus.word("snake-2").thai == "งู"            # งู: snake
    assert ctx.syllabus.graphemes[0].keyword == "snake-2"
    assert ctx.syllabus.word("name-snake-2").thai == "งอ งู"     # งอ งู: the name of ง


# ฃ (kho khuat) is obsolete and its acrophonic keyword ขวด (bottle) is
# spelled with the modern ข, so the containment invariant refuses it.
KHO_KHUAT = ConsonantRow(symbol="ฃ", consonant_class="high", sound="kʰ", name_thai="ฃอ ขวด",
                         keyword_thai="ขวด", keyword_gloss="bottle")


def test_a_keyword_that_does_not_contain_the_symbol_is_skipped_and_counted(tmp_path):
    """Decision 12: Grapheme.create enforces containment and
    grapheme/keyword-contains-symbol is an error-severity rule, so the two
    obsolete letters are reported, not adopted -- the other rows still
    are."""
    ctx = _grapheme_ctx(tmp_path, Syllabus())

    result = grapheme_attempt(ctx, consonants=[KHO_KHUAT, NGO])

    assert (result.adopted_graphemes, result.adoption_skipped) == (1, 1)
    assert [g.symbol for g in ctx.syllabus.graphemes] == ["ง"]        # ง: ŋ
    assert ctx.syllabus.find_word("bottle") is None
    assert load_words(ctx.curated_dir / "words.yaml") != []


def test_a_name_no_engine_reads_leaves_the_row_with_no_name_word(tmp_path):
    """R2: a Word is never written with an empty syllable tuple. The
    grapheme row still stands (compile drops its Reading card, counted)
    and the skip is reported. Both the phrase and one of its own tokens
    ("งอ") are unreadable, so phonology.py's token-wise fallback (r43)
    cannot read it either -- this is genuinely no engine reading, not the
    two-token case r43 fixes."""
    one = (Syllable(segments=("ŋ", "u", ""), vowel_length="long", tone="mid"),)
    ctx = _grapheme_ctx(tmp_path, Syllabus(),
                        engines=_engines(g2p=lambda thai: None if " " in thai or thai == "งอ"
                                         else one))

    result = grapheme_attempt(ctx, consonants=[NGO])

    assert (result.adopted_graphemes, result.adopted_words, result.adoption_skipped) == (1, 1, 1)
    assert ctx.syllabus.graphemes[0].name_word is None
    assert ctx.syllabus.find_word("name-snake") is None
    assert ctx.syllabus.name_word_ids == frozenset()


def test_a_keyword_no_engine_reads_leaves_the_row_unadopted(tmp_path):
    ctx = _grapheme_ctx(tmp_path, Syllabus(), engines=_engines(g2p=lambda thai: None))

    result = grapheme_attempt(ctx, consonants=[NGO])

    assert (result.adopted_graphemes, result.adoption_skipped) == (0, 1)
    assert ctx.syllabus.graphemes == ()


def test_a_second_pass_adopts_nothing(tmp_path):
    """R7: the pass is idempotent -- the second run over the same
    inventory finds every symbol already in the syllabus, asks nothing and
    writes nothing."""
    ctx = _grapheme_ctx(tmp_path, Syllabus())
    first = grapheme_attempt(ctx, consonants=[NGO])
    before = (ctx.curated_dir / "words.yaml").read_bytes()

    second = grapheme_attempt(ctx, consonants=[NGO])

    assert first.adopted_graphemes == 1
    assert second == AttemptResult(attempted=False)
    assert (ctx.curated_dir / "words.yaml").read_bytes() == before
    assert [g.symbol for g in ctx.syllabus.graphemes] == ["ง"]        # ง: ŋ


def test_a_pass_with_no_curated_store_adopts_nothing(tmp_path):
    """Outside a wired deck (run._materialize_adjudications takes the same
    guard): nothing to write the rows to, so nothing is derived."""
    ctx = _grapheme_ctx(tmp_path, Syllabus())
    ctx.curated_dir = None

    assert grapheme_attempt(ctx, consonants=[NGO]) == AttemptResult(attempted=False)
    assert ctx.syllabus.graphemes == ()


def test_the_phrase_drafter_is_not_handed_a_chart_cells_need(tmp_path):
    """Decision 17: a chart cell's query is its symbol, so drafting an
    English photograph description for it would spend the drafter's own
    ask on an item no corpus will ever be searched for."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    name = word("name-chicken", "กอ ไก่",          # กอ ไก่: the name of ก
                GRAPHEME_NAME_MEANING.format(symbol="ก"))
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    syllabus = Syllabus(words=(chicken, name), graphemes=(g,),
                        targets=(target("name-chicken/receptive", "name-chicken"),),
                        categories=(Category(name="Letter names",
                                             members=frozenset({"name-chicken"})),),
                        media=FakeMediaIndex(pictures={"chicken"}))
    ctx = _phrase_ctx(tmp_path, syllabus, json.dumps(
        {"phrases": [{"subject": "name-chicken", "phrase": "a chicken in a yard"}]}))

    result = phrase_attempt(ctx)

    assert result == AttemptResult(attempted=False)
    assert _phrase_drafter(ctx).prompts == []


def test_a_grapheme_on_file_without_a_name_word_that_still_does_not_read_survives_unchanged(
        tmp_path):
    """Spec 2 r17 section 6, for the third file too, and spec 3 r43: a row
    on file with no name word is rewritten only once the engines can read
    it (below); until then it comes back out of graphemes.yaml unchanged
    beside the newly adopted row -- rows added, none removed."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    already = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                              keyword_word=chicken)          # ก: k, no name word on file
    syllabus = Syllabus(words=(chicken,), graphemes=(already,))
    ctx = _grapheme_ctx(tmp_path, syllabus,
                        engines=_engines(g2p=_reads_normally_except("กอ ไก่", "กอ")))

    result = grapheme_attempt(ctx, consonants=[KO, NGO])

    assert (result.adopted_graphemes, result.adopted_words, result.adoption_skipped) == (1, 2, 1)
    rows = load_words(ctx.curated_dir / "words.yaml")
    saved = load_graphemes(ctx.curated_dir / "graphemes.yaml", {w.id: w for w, _ in rows})
    assert [(g.symbol, g.keyword, g.name_word) for g in saved] == [
        ("ก", "chicken", None), ("ง", "snake", "name-snake")]     # ก kept as it was; ง added


def test_a_grapheme_on_file_without_a_name_word_completes_when_the_engines_now_read_it(tmp_path):
    """Spec 3 r43: a recited name thaig2p cannot read as a phrase but can
    read token by token (2026-09-17 evidence, phonology.engines_pronunciation)
    leaves a row on file with `name_word: None`; a later pass, over the
    same table, mints (or re-uses) that name Word and its two Targets and
    REPLACES the Grapheme row in place -- the one case a curated row
    changes rather than being added. The Guard's row counts are unchanged
    (the row is replaced, not added); `adopted_graphemes` does not count
    it (it was adopted before) and `adopted_words` counts the minted name
    Word alone.
    """
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    already = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                              keyword_word=chicken)          # ก: k, no name word on file
    syllabus = Syllabus(words=(chicken,), graphemes=(already,))
    ctx = _grapheme_ctx(tmp_path, syllabus)

    result = grapheme_attempt(ctx, consonants=[KO])

    assert (result.adopted_graphemes, result.adopted_words, result.adoption_skipped) == (0, 1, 0)
    name = ctx.syllabus.word("name-chicken")
    assert name.thai == "กอ ไก่"                                   # กอ ไก่: the name of ก
    assert ctx.syllabus.category_of("name-chicken") == "Letter names"
    assert [t.id for t in ctx.syllabus.targets if t.word == "name-chicken"] == [
        "name-chicken/receptive", "name-chicken/productive"]
    g = ctx.syllabus.graphemes[0]
    assert (g.symbol, g.kind, g.sound, g.consonant_class) == ("ก", "consonant", "k", "mid")
    assert (g.keyword, g.name_word) == ("chicken", "name-chicken")
    rows = load_words(ctx.curated_dir / "words.yaml")
    saved = load_graphemes(ctx.curated_dir / "graphemes.yaml", {w.id: w for w, _ in rows})
    assert [(g2.symbol, g2.keyword, g2.name_word) for g2 in saved] == [
        ("ก", "chicken", "name-chicken")]


def test_a_second_pass_after_a_completion_changes_nothing(tmp_path):
    """Idempotent, as R7 is for a brand-new adoption: once the row is
    complete, the next pass over the same table finds no incomplete row
    and no new symbol, so it asks nothing and writes nothing."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    already = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                              keyword_word=chicken)
    syllabus = Syllabus(words=(chicken,), graphemes=(already,))
    ctx = _grapheme_ctx(tmp_path, syllabus)
    first = grapheme_attempt(ctx, consonants=[KO])
    before = (ctx.curated_dir / "words.yaml").read_bytes()

    second = grapheme_attempt(ctx, consonants=[KO])

    assert first.adopted_words == 1
    assert second == AttemptResult(attempted=False)
    assert (ctx.curated_dir / "words.yaml").read_bytes() == before


def test_a_symbol_twice_in_the_table_is_adopted_once(tmp_path):
    """R7 within one pass: the symbol is a Grapheme's identity, so a table
    naming it twice yields one row and one set of Words, not a duplicate
    pair under suffixed ids."""
    ctx = _grapheme_ctx(tmp_path, Syllabus())

    result = grapheme_attempt(ctx, consonants=[NGO, NGO])

    assert (result.adopted_graphemes, result.adopted_words) == (1, 2)
    assert [g.symbol for g in ctx.syllabus.graphemes] == ["ง"]        # ง: ŋ
    assert [w.id for w, _ in load_words(ctx.curated_dir / "words.yaml")] == [
        "snake", "name-snake"]


def _rewire(ctx):
    """The next run's wiring: a Syllabus rebuilt from the three curated
    files exactly as `wiring.load_syllabus` builds it, so a pass that died
    between two of those writes is followed by a pass that sees only what
    reached the disk."""
    rows = load_words(ctx.curated_dir / "words.yaml")
    words_by_id = {w.id: w for w, _ in rows}
    ctx.syllabus = Syllabus(
        words=tuple(w for w, _ in rows),
        targets=tuple(load_targets(ctx.curated_dir / "targets.yaml")),
        graphemes=tuple(load_graphemes(ctx.curated_dir / "graphemes.yaml", words_by_id)),
        categories=build_categories(rows))
    return ctx


def _crash_after(monkeypatch, name):
    """Turns one of curated.py's three writers into the crash: the writes
    before it land, this one and the ones after it never happen. The pass
    imports them from `curated` when it runs, so patching the module
    reaches the call."""
    def boom(*_args, **_kwargs):
        raise OSError(f"disk full writing {name}")
    monkeypatch.setattr(curated_module, name, boom)


def test_a_pass_that_died_before_targets_yaml_re_adopts_without_duplicating(
        tmp_path, monkeypatch):
    """C1, spec 3 r40 §5: the three curated writes are not one
    transaction. A run interrupted after words.yaml names the recited
    name in the vocabulary already, so the next pass must find it by its
    Thai text -- exactly as it finds the keyword -- and finish the row,
    not mint `name-chicken-2` beside it.
    """
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    syllabus = Syllabus(words=(chicken,), targets=(target("chicken/receptive", "chicken"),),
                        categories=(Category(name="Animals", members=frozenset({"chicken"})),))
    ctx = _grapheme_ctx(tmp_path, syllabus)
    _crash_after(monkeypatch, "save_targets")
    with pytest.raises(OSError):
        grapheme_attempt(ctx, consonants=[KO])
    monkeypatch.undo()
    _rewire(ctx)

    result = grapheme_attempt(ctx, consonants=[KO])

    rows = load_words(ctx.curated_dir / "words.yaml")
    assert [w.id for w, _ in rows] == ["chicken", "name-chicken"]
    assert not [w.id for w, _ in rows if str(w.id).endswith("-2")]
    assert (result.adopted_graphemes, result.adopted_words) == (1, 0)
    assert [t.id for t in load_targets(ctx.curated_dir / "targets.yaml")] == [
        "chicken/receptive", "name-chicken/receptive", "name-chicken/productive"]
    saved = load_graphemes(ctx.curated_dir / "graphemes.yaml", {w.id: w for w, _ in rows})
    assert [(g.symbol, g.keyword, g.name_word) for g in saved] == [
        ("ก", "chicken", "name-chicken")]


def test_a_pass_that_died_before_graphemes_yaml_writes_only_the_missing_row(
        tmp_path, monkeypatch):
    """C1, the second interruption: words.yaml and targets.yaml both
    landed, so only the Grapheme row is absent. The Targets are already
    listed under their own ids, so the pass writes neither a second Word
    nor a second Target -- it writes the row that is missing.
    """
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    syllabus = Syllabus(words=(chicken,), targets=(target("chicken/receptive", "chicken"),),
                        categories=(Category(name="Animals", members=frozenset({"chicken"})),))
    ctx = _grapheme_ctx(tmp_path, syllabus)
    _crash_after(monkeypatch, "save_graphemes")
    with pytest.raises(OSError):
        grapheme_attempt(ctx, consonants=[KO])
    monkeypatch.undo()
    _rewire(ctx)

    result = grapheme_attempt(ctx, consonants=[KO])

    assert (result.adopted_graphemes, result.adopted_words) == (1, 0)
    rows = load_words(ctx.curated_dir / "words.yaml")
    assert [w.id for w, _ in rows] == ["chicken", "name-chicken"]
    assert [t.id for t in load_targets(ctx.curated_dir / "targets.yaml")] == [
        "chicken/receptive", "name-chicken/receptive", "name-chicken/productive"]
    saved = load_graphemes(ctx.curated_dir / "graphemes.yaml", {w.id: w for w, _ in rows})
    assert [(g.symbol, g.keyword, g.name_word) for g in saved] == [
        ("ก", "chicken", "name-chicken")]
    assert ctx.syllabus.name_word_ids == frozenset({"name-chicken"})


def test_a_pass_that_died_before_targets_yaml_re_adopts_a_closure_keyword_too(
        tmp_path, monkeypatch):
    """C1 for the row whose keyword is new too: the keyword was already
    re-used by its Thai text before this fix, and the recited name now
    joins it, so the interrupted row is completed under its original two
    ids."""
    ctx = _grapheme_ctx(tmp_path, Syllabus())
    _crash_after(monkeypatch, "save_targets")
    with pytest.raises(OSError):
        grapheme_attempt(ctx, consonants=[NGO])
    monkeypatch.undo()
    _rewire(ctx)

    result = grapheme_attempt(ctx, consonants=[NGO])

    assert (result.adopted_graphemes, result.adopted_words) == (1, 0)
    assert [w.id for w, _ in load_words(ctx.curated_dir / "words.yaml")] == ["snake", "name-snake"]
    assert [t.id for t in load_targets(ctx.curated_dir / "targets.yaml")] == [
        "name-snake/receptive", "name-snake/productive"]
    assert ctx.syllabus.graphemes[0].name_word == "name-snake"


# --- the pair search: the adoption pass for minimal pairs (spec 3 r47) -----

TONE = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone", sounds=("mid", "low"),
                      weight=2)


def _confusion_syllabus(*words, confusions=(TONE,), pairs=()):
    return Syllabus(words=tuple(words), targets=(target(f"{words[0].id}/receptive", words[0].id),),
                    categories=(Category(name="Food", members=frozenset({words[0].id})),),
                    confusions=tuple(confusions), pairs=tuple(pairs))


def _engines_reading(readings: dict[str, tuple[Syllable, ...]]):
    """Two engines that agree on every reading in `readings` (an outside
    form's `engines_pronunciation` is then `engines_agree`), built the
    same shape `_engines` builds its own fake pair."""
    def g2p(thai: str):
        return readings.get(thai)
    return Engines(g2p=(g2p, g2p), tone=lambda thai: None)


def test_pair_search_adopts_a_vocabulary_pair_and_writes_pairs_yaml(tmp_path):
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", length="short",
                                                       tone="mid"),))
    far = word("far", "ไกล", "far", syllables=(syl(onset="kl", vowel="a", length="short",
                                                   tone="low"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near, far))
    result = pair_search_attempt(ctx)
    assert result.attempted is True
    assert (result.adopted_pairs, result.adopted_words, result.candidate_asks) == (1, 0, 0)
    assert [p.id for p in ctx.syllabus.pairs] == ["tone:mid-low/far-near"]
    saved = load_pairs(ctx.curated_dir / "pairs.yaml", {w.id: w for w in ctx.syllabus.words},
                       {TONE.id: TONE})
    assert [(p.confusion, p.members) for p in saved] == [("tone:mid-low", ("near", "far"))]


def test_pair_search_asks_the_judge_about_an_outside_form_it_needs(tmp_path):
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ไกล",)
    ctx.engines = _engines_reading({"ไกล": (syl(onset="kl", vowel="a", tone="low"),)})
    result = pair_search_attempt(ctx)
    assert (result.adopted_pairs, result.candidate_asks) == (0, 1)
    assert [q.question.subject for q in result.questions] == [f"{CANDIDATE_SUBJECT_PREFIX}ไกล"]
    assert result.questions[0].question.subject_kind == "candidate"


def test_pair_search_adopts_an_outside_form_once_the_judge_glossed_it(tmp_path):
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ไกล",)
    ctx.engines = _engines_reading({"ไกล": (syl(onset="kl", vowel="a", tone="low"),)})
    ctx.db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(ctx.rubrics["pronunciation-for-word"], None,
                                        f"{CANDIDATE_SUBJECT_PREFIX}ไกล", "pronunciation-for-word"),
                  subject=f"{CANDIDATE_SUBJECT_PREFIX}ไกล",
                  question={"role": "pronunciation-for-word", "artifact_sha": None,
                            "rubric": ctx.rubrics["pronunciation-for-word"],
                            "kind": "pronunciation", "subject_kind": "candidate"},
                  answer={"value": {"syllables": [{"segments": ["kl", "a", ""],
                                                   "vowel_length": "short", "tone": "low"}],
                                    "gloss": "far"}})
    result = pair_search_attempt(ctx)
    assert (result.adopted_pairs, result.adopted_words, result.candidate_asks) == (1, 1, 0)
    far = ctx.syllabus.word("far")
    assert (far.thai, far.meaning, far.pron.corroboration) == ("ไกล", "far", "adjudicated")
    assert ctx.syllabus.category_of("far") is None
    assert [p.members for p in ctx.syllabus.pairs] == [("near", "far")]
    assert [w.id for w, _ in load_words(ctx.curated_dir / "words.yaml")] == ["near", "far"]


def test_pair_search_stops_at_the_wanted_count_and_never_reuses_a_member(tmp_path):
    a = word("a", "กา", "crow", syllables=(syl(onset="k", vowel="a", tone="mid"),))
    b = word("b", "ก่า", "b", syllables=(syl(onset="k", vowel="a", tone="low"),))
    c = word("c", "ก้า", "c", syllables=(syl(onset="k", vowel="a", tone="low"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(a, b, c))
    assert pair_search_attempt(ctx).adopted_pairs == 1
    assert pair_search_attempt(ctx) == AttemptResult(attempted=False)   # wanted 1 (weight 2), have 1


def test_pair_search_does_nothing_without_pairs_yaml(tmp_path):
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    far = word("far", "ไกล", "far", syllables=(syl(onset="kl", vowel="a", tone="low"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near, far),
                        files=("words.yaml", "targets.yaml", "graphemes.yaml"))
    assert pair_search_attempt(ctx) == AttemptResult(attempted=False, adoption_skipped=1)


# Review fix round 1: a weight-5 confusion (pair_count 4) with two formable
# pairs, so a second run still has `wanted > 0` and the old `taken` bug (built
# from `by_thai`, a WordId never matching a Thai-keyed dict) could silently
# reuse an adopted pair's members and duplicate its row.
TONE5 = SoundConfusion(id=ConfusionId("tone:mid-low-5"), dimension="tone", sounds=("mid", "low"),
                       weight=5)


def test_pair_search_run_twice_does_not_duplicate_an_adopted_pair(tmp_path):
    a = word("a", "กา", "a", syllables=(syl(onset="k", vowel="a", tone="mid"),))
    b = word("b", "ก่า", "b", syllables=(syl(onset="k", vowel="a", tone="low"),))
    c = word("c", "งา", "c", syllables=(syl(onset="ng", vowel="a", tone="mid"),))
    d = word("d", "ง่า", "d", syllables=(syl(onset="ng", vowel="a", tone="low"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(a, b, c, d, confusions=(TONE5,)))

    first = pair_search_attempt(ctx)
    assert first.adopted_pairs == 2
    assert sorted(p.id for p in ctx.syllabus.pairs) == [
        "tone:mid-low-5/a-b", "tone:mid-low-5/c-d"]

    second = pair_search_attempt(ctx)
    assert second == AttemptResult(attempted=False)
    assert sorted(p.id for p in ctx.syllabus.pairs) == [
        "tone:mid-low-5/a-b", "tone:mid-low-5/c-d"]
    saved = load_pairs(ctx.curated_dir / "pairs.yaml", {w.id: w for w in ctx.syllabus.words},
                       {TONE5.id: TONE5})
    assert sorted(p.id for p in saved) == ["tone:mid-low-5/a-b", "tone:mid-low-5/c-d"]


def test_pair_search_adopts_a_late_pair_once_its_member_is_corroborated(tmp_path):
    """The first run can only form a-b (c is not yet corroborated); once c
    becomes corroborated the second run adopts c-d alone -- a-b, already
    adopted, is never reconsidered."""
    a = word("a", "กา", "a", syllables=(syl(onset="k", vowel="a", tone="mid"),))
    b = word("b", "ก่า", "b", syllables=(syl(onset="k", vowel="a", tone="low"),))
    c_disputed = word("c", "งา", "c", syllables=(syl(onset="ng", vowel="a", tone="mid"),),
                      corroboration="disputed")
    d = word("d", "ง่า", "d", syllables=(syl(onset="ng", vowel="a", tone="low"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(a, b, c_disputed, d, confusions=(TONE5,)))

    first = pair_search_attempt(ctx)
    assert first.adopted_pairs == 1
    assert [p.id for p in ctx.syllabus.pairs] == ["tone:mid-low-5/a-b"]

    c_ready = word("c", "งา", "c", syllables=(syl(onset="ng", vowel="a", tone="mid"),))
    ctx.syllabus = replace(ctx.syllabus, words=tuple(
        c_ready if w.id == "c" else w for w in ctx.syllabus.words))

    second = pair_search_attempt(ctx)
    assert second.adopted_pairs == 1
    assert [p.id for p in ctx.syllabus.pairs] == [
        "tone:mid-low-5/a-b", "tone:mid-low-5/c-d"]


def test_pair_search_gives_an_outside_members_pair_the_minted_word_id(tmp_path):
    """The PairId of a pair with an outside member is built from the ids
    the mint gives it -- not the pre-mint Candidates' Thai text -- so it
    reads exactly like the all-vocabulary case."""
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ไกล",)
    ctx.engines = _engines_reading({"ไกล": (syl(onset="kl", vowel="a", tone="low"),)})
    ctx.db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(ctx.rubrics["pronunciation-for-word"], None,
                                        f"{CANDIDATE_SUBJECT_PREFIX}ไกล", "pronunciation-for-word"),
                  subject=f"{CANDIDATE_SUBJECT_PREFIX}ไกล",
                  question={"role": "pronunciation-for-word", "artifact_sha": None,
                            "rubric": ctx.rubrics["pronunciation-for-word"],
                            "kind": "pronunciation", "subject_kind": "candidate"},
                  answer={"value": {"syllables": [{"segments": ["kl", "a", ""],
                                                   "vowel_length": "short", "tone": "low"}],
                                    "gloss": "far"}})
    result = pair_search_attempt(ctx)
    assert result.adopted_pairs == 1
    assert [p.id for p in ctx.syllabus.pairs] == ["tone:mid-low/far-near"]


def test_pair_search_asks_about_both_outside_forms_of_a_selected_pair(tmp_path):
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ตา", "ต่า")
    ctx.engines = _engines_reading({"ตา": (syl(onset="t", vowel="a", tone="mid"),),
                                   "ต่า": (syl(onset="t", vowel="a", tone="low"),)})
    result = pair_search_attempt(ctx)
    assert result.adopted_pairs == 0
    assert result.candidate_asks == 2
    assert sorted(q.question.subject for q in result.questions) == sorted(
        f"{CANDIDATE_SUBJECT_PREFIX}{form}" for form in ("ตา", "ต่า"))


def test_pair_search_caps_asks_at_pair_search_asks(tmp_path):
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ตา", "ต่า")
    ctx.engines = _engines_reading({"ตา": (syl(onset="t", vowel="a", tone="mid"),),
                                   "ต่า": (syl(onset="t", vowel="a", tone="low"),)})
    ctx.pair_search_asks = 1
    result = pair_search_attempt(ctx)
    assert result.adopted_pairs == 0
    assert result.candidate_asks == 1
    assert len(result.questions) == 1
    assert result.questions[0].question.subject in {
        f"{CANDIDATE_SUBJECT_PREFIX}ตา", f"{CANDIDATE_SUBJECT_PREFIX}ต่า"}


def test_pair_search_resolves_a_homograph_by_word_id_not_thai(tmp_path):
    """Two Words sharing one `thai` (a homograph; load_words dedupes ids
    only, never thai) -- a vocabulary-member lookup keyed on `thai` alone
    could resolve the wrong Word, or mint a pair from it that itself
    violates the confusion and raises. Keyed on word_id, the pass adopts
    the pair the search actually chose and never raises."""
    w1a = word("w1a", "กา", "a-meaning", syllables=(syl(onset="k", vowel="a", tone="mid"),))
    w1b = word("w1b", "กา", "b-meaning", syllables=(syl(onset="k", vowel="a", tone="falling"),))
    w3 = word("w3", "ก่า", "partner", syllables=(syl(onset="k", vowel="a", tone="low"),))
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low-homograph"), dimension="tone",
                               sounds=("mid", "low"), weight=1)
    syllabus = Syllabus(words=(w1a, w1b, w3),
                        targets=(target("w1a/receptive", "w1a"),),
                        categories=(Category(name="Food", members=frozenset({"w1a"})),),
                        confusions=(confusion,))
    ctx = _grapheme_ctx(tmp_path, syllabus)

    result = pair_search_attempt(ctx)

    assert result.adopted_pairs == 1
    assert [p.members for p in ctx.syllabus.pairs] == [("w1a", "w3")]


def _candidate_verdict(ctx, thai, syllables, gloss):
    """A fresh judge verdict on an outside form, as the pair search's own
    ask records it. Returns the row's ts."""
    return ctx.db.append(port="assess", backend="judge",
                  key=JudgeKey.for_rule(ctx.rubrics["pronunciation-for-word"], None,
                                        f"{CANDIDATE_SUBJECT_PREFIX}{thai}",
                                        "pronunciation-for-word"),
                  subject=f"{CANDIDATE_SUBJECT_PREFIX}{thai}",
                  question={"role": "pronunciation-for-word", "artifact_sha": None,
                            "rubric": ctx.rubrics["pronunciation-for-word"],
                            "kind": "pronunciation", "subject_kind": "candidate"},
                  answer={"value": {"syllables": syllables, "gloss": gloss}})


def test_pair_search_drops_a_candidate_whose_fresh_verdict_does_not_corroborate(tmp_path):
    """I3: a form the engines agree on, whose judge verdict then disagrees
    with them, is out of the pool -- not kept on its `engines_agree`
    reading, where it would be selected every run and re-asked for ever as
    a cache hit."""
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ไกล",)
    ctx.engines = _engines_reading({"ไกล": (syl(onset="kl", vowel="a", tone="low"),)})
    _candidate_verdict(ctx, "ไกล",
                       [{"segments": ["kl", "a", ""], "vowel_length": "short",
                         "tone": "falling"}], "far")

    result = pair_search_attempt(ctx)

    assert result == AttemptResult(attempted=False, candidates_dropped=1)
    assert ctx.syllabus.pairs == ()


def test_pair_search_candidates_dropped_is_a_per_run_delta(tmp_path):
    """spec 3's RunReport row: candidates_dropped is what this run's pool
    dropped, not a standing gauge. A form whose disagreeing verdict a
    previous run already reported (its row's ts at or before that run's
    RunReport row) does not count again just for still sitting there --
    it still leaves the pool (the `continue` stays)."""
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ไกล",)
    ctx.engines = _engines_reading({"ไกล": (syl(onset="kl", vowel="a", tone="low"),)})
    verdict_ts = _candidate_verdict(ctx, "ไกล",
                       [{"segments": ["kl", "a", ""], "vowel_length": "short",
                         "tone": "falling"}], "far")

    first = pair_search_attempt(ctx)

    assert first == AttemptResult(attempted=False, candidates_dropped=1)

    ctx.db.append(port="run", backend="runreport", key=RunReportKey(), subject="run",
                 question={"kind": "runreport"}, answer={"attempted": False},
                 ts=verdict_ts + 1)

    second = pair_search_attempt(ctx)

    assert second == AttemptResult(attempted=False, candidates_dropped=0)


def test_pair_search_reads_each_frequency_form_once_across_two_passes(tmp_path):
    """I6: the engines' reading of a frequency form is memoised on the
    ctx, so a second pass of the same invocation pays nothing for it."""
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ไกล",)
    calls: list[str] = []

    def g2p(thai: str):
        calls.append(thai)
        return {"ไกล": (syl(onset="kl", vowel="a", tone="low"),)}.get(thai)

    ctx.engines = Engines(g2p=(g2p, g2p), tone=lambda thai: None)

    pair_search_attempt(ctx)
    after_first = len(calls)
    pair_search_attempt(ctx)

    assert after_first > 0
    assert len(calls) == after_first


def test_pair_search_never_loads_the_engines_for_a_vocabulary_only_deck(tmp_path, monkeypatch):
    """I6: `default_engines` pulls in pythainlp/torch, so it is resolved
    only when there is a frequency form to read -- a deck whose pairs come
    out of the vocabulary never touches it."""
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    far = word("far", "ไกล", "far", syllables=(syl(onset="kl", vowel="a", tone="low"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near, far))
    ctx.engines = None

    def boom():
        raise AssertionError("the engines were loaded for a vocabulary-only pass")

    monkeypatch.setattr(attempts_module, "default_engines", boom)

    assert pair_search_attempt(ctx).adopted_pairs == 1


class _ExcludingAssessor:
    """An assessor that prepares nothing: every ask is excluded, so
    `collected` is empty though the asks were made."""

    def __init__(self):
        self.asked: list = []

    def ask_many(self, backend: str, asks):
        self.asked = list(asks)
        return ManyResult(resolved={}, collected=[],
                          excluded={str(i): Excluded(subject=a.subject, artifact_sha=None,
                                                     reason="no rubric")
                                    for i, a in enumerate(self.asked)})


def test_pair_search_counts_the_asks_it_made_not_the_questions_collected(tmp_path):
    """`candidate_asks` is what the pass asked the judge about this run
    (spec 3 r47 §7); a question the assessor could not prepare was still an
    ask."""
    near = word("near", "ใกล้", "near", syllables=(syl(onset="kl", vowel="a", tone="mid"),))
    ctx = _grapheme_ctx(tmp_path, _confusion_syllabus(near))
    ctx.frequency_words = lambda: ("ไกล",)
    ctx.engines = _engines_reading({"ไกล": (syl(onset="kl", vowel="a", tone="low"),)})
    ctx.assessor = _ExcludingAssessor()

    result = pair_search_attempt(ctx)

    assert result.questions == [] and len(result.excluded) == 1
    assert result.candidate_asks == 1


# --- sources_for_need: the roster one need is asked (spec 3 r41 §5) ---------

def _named_ctx(tmp_path):
    """A ctx whose syllabus holds one grapheme with a name word."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    name = word("name-chicken", "กอ ไก่", "the letter ก's recited name")  # กอ ไก่
    rice = word("rice", "ข้าว", "rice")           # ข้าว: rice
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    syllabus = Syllabus(words=(chicken, name, rice), graphemes=(g,),
                        targets=(target("rice/receptive", "rice"),))
    return _sourcing(tmp_path, syllabus, backends={}, assess={})


def test_a_name_words_picture_need_is_asked_the_glyph_source_alone(tmp_path):
    ctx = _named_ctx(tmp_path)
    assert sources_for_need(ctx, Need("name-chicken", "picture")) == ("glyph",)


def test_every_other_need_is_asked_the_ctxs_own_roster(tmp_path):
    ctx = _named_ctx(tmp_path)
    assert sources_for_need(ctx, Need("rice", "picture")) == ctx.sources_for("picture")
    assert sources_for_need(ctx, Need("name-chicken", "recording")) == ("forvo", "tts")
    assert sources_for_need(ctx, Need("sha", "picture", "sentence")) == ctx.sources_for("picture")


def test_the_decks_configured_roster_is_honoured(tmp_path):
    ctx = _named_ctx(tmp_path)
    ctx.sources_for = lambda kind: ("pexels",) if kind == "picture" else ()
    assert sources_for_need(ctx, Need("rice", "picture")) == ("pexels",)
    assert sources_for_need(ctx, Need("name-chicken", "picture")) == ("glyph",)


# --- the chart cell and its query (spec 3 r41 §5) ---------------------------

def _cell_ctx(tmp_path, *, keyword_has_a_picture=True):
    """A ctx over one grapheme whose keyword's current-best picture is (or
    is not) in the store."""
    chicken = word("chicken", "ไก่", "chicken")   # ไก่: chicken
    name = word("name-chicken", "กอ ไก่", "the letter ก's recited name")  # กอ ไก่
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k", consonant_class="mid",
                        keyword_word=chicken, name_word=name)
    media = FakeMediaIndex(pictures={"chicken"} if keyword_has_a_picture else set())
    syllabus = Syllabus(words=(chicken, name), graphemes=(g,),
                        targets=(target("name-chicken/receptive", "name-chicken"),),
                        media=media)
    store = MediaStore(tmp_path / "media")
    holder: list[Sourcing] = []

    def resolve(sha):
        prov = holder[0].db.media_provenance(sha)
        path = store.path_for(sha, prov["ext"]) if prov else None
        return path if path is not None and path.exists() else None

    ctx = _sourcing(tmp_path, syllabus, media=store, backends={},
                    assess={"judge": JudgeBackend(model="m", transport="api", complete=_Judge(),
                                                  resolve_path=resolve)})
    holder.append(ctx)
    if keyword_has_a_picture:
        ctx.db.add_media(sha="sha-chicken", kind="picture", ext="jpg", source="pexels",
                         origin="https://x/chicken.jpg", licence="by", acquired=date(2026, 9, 17))
    return ctx


def test_a_name_words_picture_need_has_a_chart_cell(tmp_path):
    ctx = _cell_ctx(tmp_path)
    assert chart_cell(ctx, Need("name-chicken", "picture")) == ChartCell(
        symbol="ก", picture_sha="sha-chicken", picture_ext="jpg")


def test_a_name_word_whose_keyword_has_no_picture_has_no_cell(tmp_path):
    ctx = _cell_ctx(tmp_path, keyword_has_a_picture=False)
    assert chart_cell(ctx, Need("name-chicken", "picture")) is None


def test_no_other_need_has_a_chart_cell(tmp_path):
    ctx = _cell_ctx(tmp_path)
    assert chart_cell(ctx, Need("name-chicken", "recording")) is None
    assert chart_cell(ctx, Need("chicken", "picture")) is None
    assert chart_cell(ctx, Need("name-chicken", "picture", "sentence")) is None


def test_the_chart_cells_query_is_the_symbol(tmp_path):
    """R5/decision 17: the cell is drawn from the symbol, so the query is
    a fact of curated data -- never drafted, never a learner direction."""
    ctx = _cell_ctx(tmp_path)
    _seed_phrase(ctx, "name-chicken", "a chicken in a yard")
    assert picture_query_for(ctx, Need("name-chicken", "picture")) == "ก"


def test_a_chart_cell_need_with_no_keyword_picture_has_no_query(tmp_path):
    """Spec 3 r41 §5: the need waits -- the r25 path, deferred, until the
    keyword's own picture arrives."""
    ctx = _cell_ctx(tmp_path, keyword_has_a_picture=False)
    _seed_phrase(ctx, "name-chicken", "a chicken in a yard")
    assert picture_query_for(ctx, Need("name-chicken", "picture")) is None


def test_a_direction_on_a_chart_cell_need_is_logged_and_the_symbol_still_wins(tmp_path, caplog):
    """I3, spec 3 r41 §5: a learner direction outranks every other query
    (§5's precedence) -- except here, where the cell is drawn from the
    symbol and the keyword's picture rather than searched for. Silently
    dropping the direction would leave the learner watching a card that
    never changes, so the pass says so once, at WARNING, and carries on
    with the symbol.
    """
    ctx = _cell_ctx(tmp_path)
    append_direction(ctx.db, subject="name-chicken", role="picture-for-word",
                     text="use a photo of a rooster")

    with caplog.at_level(logging.WARNING, logger="thai_syllabus.attempts"):
        assert picture_query_for(ctx, Need("name-chicken", "picture")) == "ก"

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "name-chicken" in warnings[0].getMessage()
    assert "direction" in warnings[0].getMessage()


def test_a_chart_cell_need_with_no_direction_logs_nothing(tmp_path, caplog):
    ctx = _cell_ctx(tmp_path)
    with caplog.at_level(logging.WARNING, logger="thai_syllabus.attempts"):
        assert picture_query_for(ctx, Need("name-chicken", "picture")) == "ก"
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_an_ordinary_needs_query_is_still_the_record(tmp_path):
    ctx = _cell_ctx(tmp_path)
    _seed_phrase(ctx, "chicken", "a chicken in a yard")
    assert picture_query_for(ctx, Need("chicken", "picture")) == "a chicken in a yard"


class _RecordingGlyph:
    """A glyph backend that records what it was asked and answers a cell
    already in the store."""

    def __init__(self, media):
        self.media = media
        self.asked = []

    def cache_key(self, q):
        return ProvideKey(source="glyph", kind=str(q.params.get("cell_picture") or ""),
                          query=q.params["query"])

    def fetch(self, q):
        self.asked.append(dict(q.params))
        sha = self.media.add_image(_png_bytes(), "png").sha
        return RawAnswer(items=({"sha": sha, "ext": "png", "source": "glyph",
                                 "origin": q.params["query"], "licence": "generated"},))


def test_the_glyph_source_is_asked_with_the_cells_symbol_and_keyword_picture(tmp_path):
    """Spec 3 r41 §5: the attempt names the artifact the cell is composed
    from, so the backend needs no syllabus and the key carries the sha."""
    ctx = _cell_ctx(tmp_path)
    glyph = _RecordingGlyph(ctx.media_store)
    ctx.provider._backends["glyph"] = glyph

    attempt(ctx, Need("name-chicken", "picture"), "glyph")

    assert glyph.asked == [{"query": "ก", "cell_picture": "sha-chicken",
                            "cell_picture_ext": "jpg"}]


def test_the_cell_is_ingested_without_imgfetch_and_keeps_its_provenance(tmp_path):
    ctx = _cell_ctx(tmp_path)
    glyph = _RecordingGlyph(ctx.media_store)
    ctx.provider._backends["glyph"] = glyph

    attempt(ctx, Need("name-chicken", "picture"), "glyph")

    outcome = _outcome(ctx.db, "name-chicken", "picture", "glyph")
    (sha,) = outcome.answer["candidates"]
    assert outcome.answer["tried"] == []                     # no imgfetch
    assert ctx.db.media_provenance(sha)["source"] == "glyph"


def test_an_ordinary_picture_source_is_asked_the_query_alone(tmp_path):
    ctx, search, _judge = _picture_ctx(tmp_path)
    attempt(ctx, Need("rice", "picture"), "openverse")
    assert search.asked == [{"query": "rice food"}]
