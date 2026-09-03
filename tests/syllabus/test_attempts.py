"""attempts.py: assess-first, then one Source; every candidate judged; current-best re-derived.
Real SyllabusDb + MediaStore; fake Provider/Assessor backends; no network."""
from datetime import date
from pathlib import Path

import pytest

from thai_syllabus.assessor import AssessQuestion, Assessor, JudgeBackend, MechanicalBackend, RawVerdict
from thai_syllabus.attempts import (Need, Outcome, Sourcing, _phrase, attempt, candidates_of,
                                    sources_for)
from thai_syllabus.entities import MinimalPair, SoundConfusion
from thai_syllabus.provider import FetchBackend, Provider, Question, RawAnswer, TtsBackend
from thai_syllabus.rulebook import RUBRICS_BY_ROLE
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.transport import Completion, TransportError

from .builders import target, word


class _Search:
    def __init__(self, urls):
        self.urls, self.calls = urls, 0

    def cache_key(self, q):
        return f"search:{q.params['query']}"

    def fetch(self, q):
        self.calls += 1
        return RawAnswer(items=tuple({"url": u, "source": "openverse", "licence": "by"} for u in self.urls))


def _fetcher(url):
    return url.encode(), "jpg"


class _Judge:
    """Verdict by attachment count: more than one attachment means a
    preference (ranking) question; one attachment is a fit question,
    decided by its file's bytes containing 'good'."""
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, attachments=()):
        self.calls.append([Path(a).name for a in attachments])
        if len(attachments) > 1:
            names = [Path(a).stem for a in attachments]
            return Completion(text='{"ranking": ' + str(names).replace("'", '"') + '}')
        ok = any(b"good" in Path(a).read_bytes() for a in attachments)
        return Completion(text='{"value": %s, "evidence": "e"}' % ("true" if ok else "false"))


@pytest.fixture
def ctx(tmp_path):
    db = SyllabusDb(tmp_path / "syllabus.db")
    media = MediaStore(tmp_path / "media")
    judge = _Judge()
    syl = Syllabus(words=(word("orange", "ส้ม", "orange"),),
                   targets=(target("orange/receptive", "orange"),))

    def resolve(sha):
        prov = db.media_provenance(sha)
        return media.path_for(sha, prov["ext"]) if prov else None

    search = _Search(["https://x/bad1.jpg", "https://x/good.jpg", "https://x/good2.jpg"])
    provider = Provider(record=db, cache=db, backends={
        "openverse": search,
        "imgfetch": FetchBackend(media=media, fetcher=_fetcher)})
    jb = JudgeBackend(model="m", transport="api", complete=judge, resolve_path=resolve)
    assessor = Assessor(record=db, cache=db, backends={"judge": jb})
    sourcing = Sourcing(syllabus=syl, provider=provider, assessor=assessor, db=db, media_store=media,
                        rubrics=dict(RUBRICS_BY_ROLE), provenance_prior=("commission", "forvo", "tts"),
                        image_candidates=3, today=lambda: date(2026, 9, 3))
    return sourcing, search, judge


def _prime_legacy_passer(c: Sourcing, thai_word: str = "ส้ม") -> str:
    """A picture already on record, already fit-verified (as if judged in
    an earlier run) -- pre-seeds current_best without going through a
    whole attempt(). The cache key for a fit verdict is rubric+artifact+
    role only (JudgeBackend.cache_key), so this matches what
    _fit_pictures will later read regardless of params.
    """
    legacy_sha = c.media_store.write(b"good-legacy", "jpg")
    c.db.add_media(sha=legacy_sha, kind="picture", ext="jpg", source="legacy", origin="", licence="?",
                   acquired=date(2026, 1, 1))
    c.db.append(port="assess", backend="machine-chosen", key=f"machine-chosen:orange:{legacy_sha}",
               subject="orange", question={"note_id": "pw-1", "word": thai_word},
               answer={"marker": "machine-chosen", "sha": legacy_sha})
    c.assessor.ask("judge", AssessQuestion(subject="orange", role="picture-for-word",
                                           artifact_sha=legacy_sha, rubric=c.rubrics["picture-for-word"]))
    return legacy_sha


def test_sources_for_picture_is_cost_ordered():
    assert sources_for("picture") == ("openverse", "wikimedia", "pexels")


def test_picture_attempt_fetches_judges_all_and_improves(ctx):
    c, search, judge = ctx
    out = attempt(c, Need("orange", "picture"), "openverse")
    assert (out.attempted, out.pending, out.improved) == (True, False, True)
    fit_calls = [call for call in judge.calls if len(call) == 1]
    assert len(fit_calls) == 3                       # every candidate judged, not stop-at-first-pass
    assert any(len(call) == 2 for call in judge.calls)  # one preference call over the two passes
    assert len(candidates_of(c.db, "orange", "picture")) == 3
    from thai_syllabus.attempts import current_best_of
    best = current_best_of(c, "orange", "picture")
    assert best.artifact_sha is not None and best.rank > 50.0


def test_picture_attempt_spend_accounting(ctx):
    c, search, judge = ctx
    out1 = attempt(c, Need("orange", "picture"), "openverse")
    assert out1.spend["judge"][0] == 4          # 3 fit + 1 preference
    assert out1.spend["imgfetch"][0] == 3

    out2 = attempt(c, Need("orange", "picture"), "openverse")
    assert all(asks == 0 for asks, _cost in out2.spend.values())
    assert out2.improved is False


def test_picture_attempt_assesses_existing_candidates_before_searching(ctx):
    # a migrated/unjudged candidate already on record
    c, search, judge = ctx
    sha = c.media_store.write(b"good-old", "jpg")
    c.db.add_media(sha=sha, kind="picture", ext="jpg", source="legacy", origin="", licence="?",
                   acquired=date(2026, 1, 1))
    c.db.append(port="assess", backend="machine-chosen", key=f"machine-chosen:orange:{sha}",
               subject="orange", question={"note_id": "pw-1", "word": "ส้ม"},
               answer={"marker": "machine-chosen", "sha": sha})
    out = attempt(c, Need("orange", "picture"), "openverse")
    assert out.improved and search.calls == 0


def test_picture_attempt_records_provenance_for_each_fetched_candidate(ctx):
    c, search, judge = ctx
    attempt(c, Need("orange", "picture"), "openverse")
    for sha in candidates_of(c.db, "orange", "picture"):
        prov = c.db.media_provenance(sha)
        assert prov and prov["source"] == "openverse" and prov["ext"] == "jpg"


def test_picture_attempt_reports_pending_under_a_batch_judge(ctx):
    c, search, judge = ctx

    class BT:
        def submit(self, requests):
            return "b1"

        def status(self, batch_id):
            return "in_progress"
    c.assessor = Assessor(record=c.db, cache=c.db, backends={
        "judge": JudgeBackend(model="m", transport="batch", batch_transport=BT())})
    out = attempt(c, Need("orange", "picture"), "openverse")
    assert out.pending and not out.improved and out.attempted


def test_unknown_kind_is_not_attempted(ctx):
    c, search, judge = ctx
    assert attempt(c, Need("x", "grapheme-keyword"), "llm") == Outcome(False, False, False, {})


# --- preference runs over the WHOLE passing set, not just what's newly fit -

def test_preference_runs_over_the_whole_passing_set_not_just_new_shas(ctx):
    c, search, judge = ctx
    legacy_sha = _prime_legacy_passer(c)  # already-verified, ranks equal to before -- doesn't short-circuit
    out = attempt(c, Need("orange", "picture"), "openverse")
    assert out.improved
    pref_calls = [call for call in judge.calls if len(call) > 1]
    assert len(pref_calls) == 1
    assert len(pref_calls[0]) == 3                       # legacy + both new passers
    assert any(legacy_sha in name for name in pref_calls[0])


def test_preference_runs_with_one_legacy_and_one_new_passer(ctx):
    c, search, judge = ctx
    search.urls = ["https://x/bad1.jpg", "https://x/good.jpg"]  # exactly one new passer
    _prime_legacy_passer(c)
    attempt(c, Need("orange", "picture"), "openverse")
    pref_calls = [call for call in judge.calls if len(call) > 1]
    assert len(pref_calls) == 1
    assert len(pref_calls[0]) == 2


# --- an unavailable Assessor ends the attempt before any Source is tried ---

def test_unavailable_judge_backend_ends_the_attempt_before_any_search(ctx):
    c, search, judge = ctx
    c.assessor = Assessor(record=c.db, cache=c.db, backends={})  # no "judge" backend at all
    out = attempt(c, Need("orange", "picture"), "openverse")
    assert search.calls == 0
    assert out == Outcome(False, False, False, {})


# --- per-url fetch failure is skipped, not fatal to the attempt -----------

def test_per_url_fetch_transport_error_is_skipped(ctx):
    c, search, judge = ctx
    search.urls = ["https://x/bad1.jpg", "https://x/boom.jpg", "https://x/good.jpg"]

    def flaky_fetcher(url):
        if "boom" in url:
            raise TransportError("refused")
        return url.encode(), "jpg"

    c.provider = Provider(record=c.db, cache=c.db, backends={
        "openverse": search, "imgfetch": FetchBackend(media=c.media_store, fetcher=flaky_fetcher)})
    attempt(c, Need("orange", "picture"), "openverse")
    assert len(candidates_of(c.db, "orange", "picture")) == 2  # boom.jpg's url produced no candidate


# --- candidates_of: de-dup and marker-row exclusion -------------------------

def test_candidates_of_dedups_shas_and_ignores_batch_pending_markers(ctx):
    c, search, judge = ctx
    c.db.append(port="provide", backend="imgfetch", key="url1", subject="orange",
               question={"provides": "picture-bytes", "params": {"url": "url1"}},
               answer={"items": [{"sha": "sha1", "ext": "jpg"}]})
    c.db.append(port="provide", backend="imgfetch", key="url2", subject="orange",
               question={"provides": "picture-bytes", "params": {"url": "url2"}},
               answer={"items": [{"sha": "sha1", "ext": "jpg"}]})  # same sha, different url -- not a dup
    c.db.append(port="assess", backend="judge", key="judge-batch-pending:orange", subject="orange",
               question={"keys": ["k1"]}, answer={"kind": "batch-pending", "batch_id": "b1"})
    assert candidates_of(c.db, "orange", "picture") == ["sha1"]


# --- _phrase precedence: direction > suggestion newer than last provide > None (caller falls back) -

def test_phrase_is_none_when_nothing_is_on_record(ctx):
    c, _search, _judge = ctx
    assert _phrase(c, "orange") is None


def test_phrase_direction_wins_over_a_judge_suggestion(ctx):
    c, _search, _judge = ctx
    c.db.append(port="assess", backend="judge", key="k1", subject="orange",
               question={"role": "picture-for-word"}, answer={"value": False, "suggestion": "sugg"})
    c.db.append(port="assess", backend="learner", key="k2", subject="orange",
               question={}, answer={"direction": "dir"})
    assert _phrase(c, "orange") == "dir"


def test_phrase_falls_back_to_a_suggestion_newer_than_the_last_provide(ctx):
    c, _search, _judge = ctx
    c.db.append(port="provide", backend="openverse", key="search:x", subject="orange",
               question={"provides": "picture", "params": {"query": "x"}}, answer={"items": []})
    c.db.append(port="assess", backend="judge", key="k1", subject="orange",
               question={"role": "picture-for-word"}, answer={"value": False, "suggestion": "fresh phrase"})
    assert _phrase(c, "orange") == "fresh phrase"


def test_phrase_ignores_a_suggestion_older_than_the_last_provide(ctx):
    c, _search, _judge = ctx
    c.db.append(port="assess", backend="judge", key="k1", subject="orange",
               question={"role": "picture-for-word"}, answer={"value": False, "suggestion": "stale phrase"})
    c.db.append(port="provide", backend="openverse", key="search:x", subject="orange",
               question={"provides": "picture", "params": {"query": "x"}}, answer={"items": []})
    assert _phrase(c, "orange") is None


# --- Task 8: recording and rendition attempts; mechanical verdicts rank recordings

def _mech_duration(ok=True):
    def key_fn(q):
        return f"mech:duration:0.2-5.0:{q.artifact_sha}"

    def evaluate(q):
        return RawVerdict(value=ok, evidence="duration=1.0s")
    return MechanicalBackend(key_fn=key_fn, evaluate=evaluate)


def _mech_duration_failing_for(bad_subject):
    """Fails only the member whose AssessQuestion.subject is bad_subject --
    exercises that _assess_members asks under each member's OWN subject
    (not the pair id), the only way a per-member fake can single one out.
    """
    def key_fn(q):
        return f"mech:duration:0.2-5.0:{q.artifact_sha}"

    def evaluate(q):
        ok = q.subject != bad_subject
        return RawVerdict(value=ok, evidence="duration=1.0s" if ok else "too short")
    return MechanicalBackend(key_fn=key_fn, evaluate=evaluate)


class _Forvo:
    def __init__(self, items_by_word):
        self.items_by_word = items_by_word
        self.calls = 0

    def cache_key(self, q):
        return f"forvo:{q.params['word']}"

    def fetch(self, q):
        self.calls += 1
        return RawAnswer(items=tuple(self.items_by_word.get(q.params["word"], ())), cost=1.0)


class _Tts:
    def synthesize(self, text, voice):
        return f"{text}-{voice}".encode()


def _recording_backends(c, forvo_items):
    media = c.media_store
    forvo = _Forvo(forvo_items)
    provider = Provider(record=c.db, cache=c.db, backends={
        "forvo": forvo,
        "audiofetch": FetchBackend(media=media, fetcher=lambda url: (url.encode(), "mp3")),
        "tts": TtsBackend(tts=_Tts(), voices=["v1", "v2"], media=media, pick_voice=lambda s, v: v[0])})
    return provider, forvo


def _recording_ctx(c, forvo_items):
    provider, _forvo = _recording_backends(c, forvo_items)
    c.provider = provider
    c.assessor = Assessor(record=c.db, cache=c.db, backends={"mechanical": _mech_duration()})
    return c


def _pair_syllabus(conf):
    return Syllabus(
        words=(word("white", "ขาว", "white"), word("news", "ข่าว", "news")),
        targets=(target("white/receptive", "white"), target("news/receptive", "news")),
        confusions=(conf,),
        pairs=(MinimalPair(id="pair-1", confusion=conf.id, members=("white", "news")),))


def test_forvo_recording_attempt_downloads_checks_and_ranks(ctx):
    c, _search, _judge = ctx
    c = _recording_ctx(c, {"ส้ม": [{"pathmp3": "https://f/1.mp3", "username": "kris"}]})
    out = attempt(c, Need("orange", "recording"), "forvo")
    assert out.improved and out.spend["forvo"] == (1, 1.0)
    from thai_syllabus.attempts import current_best_of
    best = current_best_of(c, "orange", "recording")
    prov = c.db.media_provenance(best.artifact_sha)
    assert best.source == "mechanical" and prov["speaker_id"] == "forvo:kris" and prov["speaker_kind"] == "native"


def test_tts_recording_attempt_marks_synthetic_provenance(ctx):
    c, _search, _judge = ctx
    c = _recording_ctx(c, {})
    assert not attempt(c, Need("orange", "recording"), "forvo").improved
    out = attempt(c, Need("orange", "recording"), "tts")
    assert out.improved
    from thai_syllabus.attempts import current_best_of
    prov = c.db.media_provenance(current_best_of(c, "orange", "recording").artifact_sha)
    assert prov["speaker_kind"] == "synthetic" and prov["source"] == "tts"


def test_rendition_attempt_intersects_forvo_speakers(ctx):
    c, _search, _judge = ctx
    c = _recording_ctx(c, {
        "ขาว": [{"pathmp3": "https://f/a1.mp3", "username": "kris"}, {"pathmp3": "https://f/a2.mp3", "username": "x"}],
        "ข่าว": [{"pathmp3": "https://f/b1.mp3", "username": "kris"}]})
    conf = SoundConfusion(id="tone:rising-vs-low", dimension="tone", sounds=("rising", "low"))
    c.syllabus = _pair_syllabus(conf)
    out = attempt(c, Need("pair-1", "rendition"), "forvo")
    assert out.improved
    from thai_syllabus.attempts import current_best_of
    best = current_best_of(c, "pair-1", "rendition")
    rows = [r for r in c.db.assessments_of("pair-1") if r.backend == "mechanical"]
    members = rows[-1].question["params"]["members"]
    assert set(members) == {"white", "news"}
    assert {c.db.media_provenance(s)["speaker_id"] for s in members.values()} == {"forvo:kris"}
    assert rows[-1].answer["value"] is True
    # each member's own recording need benefits too -- mechanical verdicts
    # are recorded under the member's own subject, not the pair id.
    assert current_best_of(c, "white", "recording").artifact_sha is not None
    assert current_best_of(c, "news", "recording").artifact_sha is not None


def test_rendition_attempt_falls_to_one_tts_voice(ctx):
    c, _search, _judge = ctx
    c = _recording_ctx(c, {"ขาว": [{"pathmp3": "https://f/a.mp3", "username": "p"}],
                           "ข่าว": [{"pathmp3": "https://f/b.mp3", "username": "q"}]})
    conf = SoundConfusion(id="tone:rising-vs-low", dimension="tone", sounds=("rising", "low"))
    c.syllabus = _pair_syllabus(conf)
    assert not attempt(c, Need("pair-1", "rendition"), "forvo").improved
    out = attempt(c, Need("pair-1", "rendition"), "tts")
    assert out.improved
    rows = [r for r in c.db.assessments_of("pair-1") if r.backend == "mechanical"]
    speakers = {c.db.media_provenance(s)["speaker_id"] for s in rows[-1].question["params"]["members"].values()}
    assert len(speakers) == 1
    from thai_syllabus.tts import pick_voice
    expected_voice = pick_voice("pair-1", list(c.tts_voices))
    origins = {c.db.media_provenance(s)["origin"] for s in rows[-1].question["params"]["members"].values()}
    assert origins == {expected_voice}
    # symmetry with forvo: the tts branch mechanically checks each member
    # too, under the member's own subject.
    from thai_syllabus.attempts import current_best_of
    assert current_best_of(c, "white", "recording").artifact_sha is not None
    assert current_best_of(c, "news", "recording").artifact_sha is not None


def test_rendition_attempt_records_false_when_a_member_fails_mechanically(ctx):
    c, _search, _judge = ctx
    c = _recording_ctx(c, {"ขาว": [{"pathmp3": "https://f/a1.mp3", "username": "kris"}],
                           "ข่าว": [{"pathmp3": "https://f/b1.mp3", "username": "kris"}]})
    c.assessor = Assessor(record=c.db, cache=c.db, backends={"mechanical": _mech_duration_failing_for("news")})
    conf = SoundConfusion(id="tone:rising-vs-low", dimension="tone", sounds=("rising", "low"))
    c.syllabus = _pair_syllabus(conf)
    out = attempt(c, Need("pair-1", "rendition"), "forvo")
    assert not out.improved
    rows = [r for r in c.db.assessments_of("pair-1") if r.backend == "mechanical"]
    assert rows[-1].answer["value"] is False
    assert "news" in rows[-1].answer["evidence"]


def test_recording_attempt_second_run_is_fully_cached(ctx):
    c, _search, _judge = ctx
    c = _recording_ctx(c, {"ส้ม": [{"pathmp3": "https://f/1.mp3", "username": "kris"}]})
    attempt(c, Need("orange", "recording"), "forvo")
    out2 = attempt(c, Need("orange", "recording"), "forvo")
    assert out2.attempted is False
    assert all(asks == 0 for asks, _cost in out2.spend.values())


def test_recording_attempt_ends_before_any_source_when_assessor_unavailable(ctx):
    c, _search, _judge = ctx
    provider, forvo = _recording_backends(c, {"ส้ม": [{"pathmp3": "https://f/1.mp3", "username": "kris"}]})
    c.provider = provider
    c.assessor = Assessor(record=c.db, cache=c.db, backends={})  # no "mechanical" backend at all
    out = attempt(c, Need("orange", "recording"), "forvo")
    assert forvo.calls == 0
    assert out == Outcome(False, False, False, {})


def test_rendition_attempt_ends_before_any_source_when_assessor_unavailable(ctx):
    c, _search, _judge = ctx
    provider, forvo = _recording_backends(c, {
        "ขาว": [{"pathmp3": "https://f/a.mp3", "username": "kris"}],
        "ข่าว": [{"pathmp3": "https://f/b.mp3", "username": "kris"}]})
    c.provider = provider
    c.assessor = Assessor(record=c.db, cache=c.db, backends={})  # no "mechanical" backend at all
    conf = SoundConfusion(id="tone:rising-vs-low", dimension="tone", sounds=("rising", "low"))
    c.syllabus = _pair_syllabus(conf)
    out = attempt(c, Need("pair-1", "rendition"), "forvo")
    assert forvo.calls == 0
    assert out == Outcome(False, False, False, {})
