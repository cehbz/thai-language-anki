"""Tests for reviewserver.py (spec 5): the feedback-screen server.

Real SyllabusDb (tmp_path sqlite, both CacheReader and RecordWriter) and a
synthetic Syllabus built from builders.py -- no real network, no apkg, no
judge. HTTP endpoints are exercised against a live server on an ephemeral
loopback port (spec 5's "handler logic ... via urllib against a live
server on an ephemeral port (loopback only)"); everything else calls the
handler functions directly.
"""
from __future__ import annotations

import dataclasses
import http.client
import io
import json
import threading
from datetime import date
from http.server import HTTPServer

import pytest

from PIL import Image as PILImage

from thai_syllabus import reviewserver as rs
from thai_syllabus.attempts import sources_for
from thai_syllabus.authority import role_for
from thai_syllabus.compile import CARD_CSS
from thai_syllabus.derivations import DEFAULT_ATTEMPT_CAP
from thai_syllabus.entities import Grapheme, MinimalPair, SoundConfusion
from thai_syllabus.ids import ConfusionId, PairId, WordId
from thai_syllabus.ports import StudyRecord
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.wiring import Derivations

from .builders import PROV, syl, pron, target, word
from .fakes import FakeTokenizer


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


@pytest.fixture
def media_store(tmp_path):
    return MediaStore(tmp_path / "media")


@pytest.fixture
def confusion():
    return SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                          sounds=("mid", "low"))


@pytest.fixture
def w1():
    return word("rice", "ข้าว", "rice", syllables=(syl(onset="kh", tone="mid"),))


@pytest.fixture
def w2():
    return word("near", "ใกล้", "near", syllables=(syl(onset="kh", tone="low"),))


@pytest.fixture
def keyword_word():
    return word("chicken", "ไก่", "chicken")


@pytest.fixture
def grapheme(keyword_word):
    return Grapheme.create(symbol="ไก่"[0], kind="consonant", sound="k",
                           consonant_class="mid", keyword_word=keyword_word)


@pytest.fixture
def pair(confusion, w1, w2):
    return MinimalPair.create(id=PairId("pair-rice-near"), confusion=confusion,
                              members=(w1, w2))


@pytest.fixture
def syllabus(w1, w2, keyword_word, confusion, pair, grapheme, db):
    targets = (target("t-rice", w1.id), target("t-near", w2.id))
    return Syllabus(words=(w1, w2, keyword_word), targets=targets, pairs=(pair,),
                    graphemes=(grapheme,), confusions=(confusion,), assessments=db,
                    tokenizer=FakeTokenizer())


@pytest.fixture
def derivations(syllabus, db, media_store):
    """The bundle wiring.load_derivations builds from a deck, with an
    empty rubric mapping (no verdict is stale on a role's account) and no
    provenance prior -- the deck-independent parameters these unit tests
    measure their folds under.
    """
    return Derivations(syllabus=syllabus, db=db, media_store=media_store,
                       current_rubric={}, prior=(), provenance_source=lambda sha: None,
                       sources_for=sources_for, attempt_cap=DEFAULT_ATTEMPT_CAP)


# --- cache-row helpers (mirrors test_derivations.py's) ----------------------

def _provide(db, subject, kind, backend="openverse", items=(), query=None):
    """One whole attempt, in the two row shapes a real one writes: the
    Source ask (carrying the query and the search hits) and then one
    bytes row per fetched candidate (backend imgfetch, one sha each --
    the only row shape that ever carries a sha; same `kind` as the ask,
    distinguished from it by backend, not by a suffixed kind). The Source
    ask is the attempt; the bytes rows are the candidates it produced.
    """
    params = {"query": query} if query else {}
    ts = db.append(port="provide", backend=backend, key=f"{backend}:{subject}:{query}",
                   subject=subject, question={"kind": kind, "params": params},
                   answer={"items": [i for i in items if not i.get("sha")]})
    for item in items:
        if not item.get("sha"):
            continue
        url = f"https://x/{item['sha']}.jpg"
        ts = db.append(port="provide", backend="imgfetch", key=url, subject=subject,
                       question={"kind": kind, "params": {"url": url}},
                       answer={"items": [dict(item)]})
    return ts


def _judge(db, subject, kind, artifact_sha, value, rubric="rubric-v1", evidence=None):
    role = role_for(kind)
    answer = {"value": value}
    if evidence:
        answer["evidence"] = evidence
    return db.append(port="assess", backend="judge", key=f"judge:{subject}:{artifact_sha}",
                     subject=subject,
                     question={"role": role, "artifact_sha": artifact_sha, "rubric": rubric,
                              "kind": kind},
                     answer=answer)


def _learner(db, subject, kind, artifact_sha, rating):
    role = role_for(kind)
    return db.append(port="assess", backend="learner",
                     key=f"learner:{artifact_sha}:{role}", subject=subject,
                     question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                              "kind": "rating"},
                     answer={"value": rating})


# --- build_queue: budget + F10 order ----------------------------------------

def test_build_queue_respects_budget(derivations, db):
    items = rs.build_queue(derivations, budget=1)
    assert len(items) == 1


def test_build_queue_rate_order_matches_derivations_queue(derivations, syllabus, db):
    from thai_syllabus.attempts import sources_for
    from thai_syllabus.derivations import DEFAULT_ATTEMPT_CAP
    from thai_syllabus.derivations import queue as derive_queue
    entries = derive_queue(syllabus, db, current_rubric={}, prior=(), sources_for=sources_for,
                           attempt_cap=DEFAULT_ATTEMPT_CAP, provenance_source=lambda s: None)
    items = rs.build_queue(derivations, budget=len(entries))
    rate_items = [i for i in items if i["type"] == "rate"]
    assert [(i["subject"], i["kind"]) for i in rate_items] == \
           [(e.subject, e.kind) for e in entries]


def test_build_queue_rate_item_carries_gloss_query_verdict_and_thumbnails(derivations, db, w1):
    _provide(db, w1.id, "picture", query="rice photo", items=[{"sha": "sA"}, {"sha": "sB"}])
    _judge(db, w1.id, "picture", "sA", True, evidence="clear rice bowl")
    items = rs.build_queue(derivations, budget=50)
    rated = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                and i["kind"] == "picture")
    assert rated["gloss"] == "rice"
    assert rated["query"] == "rice photo"
    assert rated["current"]["sha"] == "sA"
    assert "judge: pass" in rated["current"]["verdict"]
    assert rated["rejected"] == [{"sha": "sB", "url": "/media/sB"}]


def test_build_queue_rate_item_lists_excluded_candidates_and_card_flags(derivations, db, w1):
    """spec 5 section 1 kind 1 / section 3: the rate question carries the
    candidates the judge could never even prepare (this run's own
    RunReport row) and the subject's card-level flags (anki_import.py's
    own card-flag row shape) -- never a live judge call, both read back.
    """
    _provide(db, w1.id, "picture", items=[{"sha": "sA"}])
    _judge(db, w1.id, "picture", "sA", True)
    missing_sha = "c" * 64
    db.append(port="run", backend="runreport", key="runreport", subject="run",
             question={"kind": "runreport"},
             answer={"excluded_items": [
                 {"subject": w1.id, "artifact_sha": missing_sha,
                  "reason": "artifact not found: " + missing_sha},
                 {"subject": "some-other-word", "artifact_sha": "x", "reason": "irrelevant"},
             ]})
    db.append(port="assess", backend="learner", key=f"flag:word:{w1.id}:reading:1",
             subject=w1.id,
             question={"kind": "card-flag", "role": "card-flag", "family": "word",
                      "anchor": w1.id, "card_kind": "reading", "flags": 1},
             answer={"flagged": True, "flag": 1})

    items = rs.build_queue(derivations, budget=50)
    rated = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                and i["kind"] == "picture")
    assert rated["excluded"] == [{"sha": missing_sha, "reason": "artifact not found: " + missing_sha}]
    assert rated["flags"] == [f"{w1.id}::reading"]


def test_build_queue_direction_kind_for_exhausted_subject(derivations, db, w1):
    # No candidate ever passes and every source in the roster (spec 3's
    # openverse/wikimedia/pexels) has been asked -- next_source has
    # nothing left, so the subject is exhausted (spec 3 section 6).
    _provide(db, w1.id, "picture", backend="openverse", items=[])
    _provide(db, w1.id, "picture", backend="wikimedia", items=[])
    _provide(db, w1.id, "picture", backend="pexels", items=[])
    items = rs.build_queue(derivations, budget=50)
    direction = [i for i in items if i["type"] == "direction" and i["subject"] == w1.id
                and i["kind"] == "picture"]
    assert len(direction) == 1
    assert direction[0]["attempts"] == 3
    assert "openverse" in {t["source"] for t in direction[0]["tried"]}


def test_build_queue_direction_tried_lists_source_asks_only_never_fetch_rows(derivations, db, w1):
    """spec 5 section 1 kind 2: "what was tried" is the phrases and Sources
    a need's Source asks carried -- never a bytes-fetch row (imgfetch) or
    the url it carried.
    """
    _provide(db, w1.id, "picture", backend="openverse", query="a bowl of rice",
            items=[{"sha": "sA"}])
    _provide(db, w1.id, "picture", backend="wikimedia", items=[])
    _provide(db, w1.id, "picture", backend="pexels", items=[])
    items = rs.build_queue(derivations, budget=50)
    direction = next(i for i in items if i["type"] == "direction" and i["subject"] == w1.id)
    assert all(t["source"] in ("openverse", "wikimedia", "pexels") for t in direction["tried"])
    assert {"source": "openverse", "query": "a bowl of rice"} in direction["tried"]
    assert not any("url" in t for t in direction["tried"])


def test_build_queue_direction_candidates_carry_judge_verdicts(derivations, db, w1):
    """spec 5 section 1 kind 2: the best candidates a Source produced, each
    with the judge's own verdict (derivations.judge_verdict) -- pass and
    its evidence (the judge's reason), or None when the judge never spoke.
    """
    _provide(db, w1.id, "picture", backend="openverse", items=[{"sha": "sA"}, {"sha": "sB"}])
    _judge(db, w1.id, "picture", "sA", False, evidence="no rice visible")
    _provide(db, w1.id, "picture", backend="wikimedia", items=[])
    _provide(db, w1.id, "picture", backend="pexels", items=[])
    items = rs.build_queue(derivations, budget=50)
    direction = next(i for i in items if i["type"] == "direction" and i["subject"] == w1.id)
    by_sha = {c["sha"]: c for c in direction["candidates"]}
    assert by_sha["sA"]["verdict"] == {"passed": False, "evidence": "no rice visible"}
    assert by_sha["sB"]["verdict"] is None


def test_build_queue_challenger_kind_when_rubric_change_outranks_learner_pick(derivations, db,
                                                                              w1):
    _learner(db, w1.id, "picture", "s-old", "acceptable")
    _judge(db, w1.id, "picture", "s-new", True, rubric="rubric-v2")
    items = rs.build_queue(
        dataclasses.replace(derivations, current_rubric={"picture-for-word": "rubric-v2"}),
        budget=50)
    challengers = [i for i in items if i["type"] == "challenger" and i["subject"] == w1.id]
    assert len(challengers) == 1
    assert challengers[0]["current"]["sha"] == "s-old"
    assert challengers[0]["challenger"]["sha"] == "s-new"


# --- the verdict line: the rubric the deck runs under ----------------------

def test_rate_question_drops_the_verdict_line_once_the_rubric_moved_on(derivations, db, w1):
    _provide(db, w1.id, "picture", items=[{"sha": "sA"}])
    _judge(db, w1.id, "picture", "sA", True, rubric="rubric-v2", evidence="clear")

    under_same_rubric = dataclasses.replace(
        derivations, current_rubric={"picture-for-word": "rubric-v2"})
    rated = next(i for i in rs.build_queue(under_same_rubric, budget=50)
                if i["type"] == "rate" and i["subject"] == w1.id and i["kind"] == "picture")
    assert "judge: pass" in rated["current"]["verdict"]

    under_new_rubric = dataclasses.replace(
        derivations, current_rubric={"picture-for-word": "some other text"})
    rated = next(i for i in rs.build_queue(under_new_rubric, budget=50)
                if i["type"] == "rate" and i["subject"] == w1.id and i["kind"] == "picture")
    # the verdict is stale, so current_best no longer ranks the artifact
    # and no verdict line stands beside it.
    assert rated["current"] is None


def test_verdict_line_never_shows_a_preference_rank_as_pass(db, w1):
    """spec 5 section 1: the verdict line is filtered by role -- a
    picture-preference row is never read as the fit verdict a screen
    prints "judge: pass" from (derivations.judge_verdict reads the fit
    role only).
    """
    from thai_syllabus.derivations import judge_verdict
    db.append(port="assess", backend="judge", key="judge:pref:sA,sB", subject=w1.id,
             question={"role": "picture-preference", "artifact_sha": None, "rubric": "r",
                      "kind": "picture", "params": {"candidates": ["sA", "sB"]}},
             answer={"value": ["sA", "sB"]})
    assert judge_verdict(db, w1.id, "picture", "sA", current_rubric={}) is None
    assert rs._verdict_line(judge_verdict(db, w1.id, "picture", "sA", current_rubric={})) is None


def _pair_study_row(pair, **overrides) -> StudyRecord:
    fields = {"family": "minimal_pair", "anchor": pair.id, "card_kind": "recognition",
             "compile_id": "c1", "ts": 1, "grade": 1, "time_ms": 900}
    fields.update(overrides)
    return StudyRecord(**fields)


def test_build_queue_reask_kind_on_study_lapse_contradicting_learner_rating(
        derivations, db, pair, confusion):
    db.append_study(_pair_study_row(pair))
    _learner(db, confusion.id, "rendition", "rend-sha", "acceptable")
    items = rs.build_queue(derivations, study=db, budget=50)
    reasks = [i for i in items if i["type"] == "reask" and i["subject"] == confusion.id]
    assert len(reasks) == 1
    assert reasks[0]["original_answer"] == "acceptable"
    assert reasks[0]["evidence"][0]["anchor"] == pair.id
    assert reasks[0]["evidence"][0]["card_kind"] == "recognition"


def _word_study_row(word_id: str, **overrides) -> StudyRecord:
    fields = {"family": "word", "anchor": word_id, "card_kind": "production",
             "compile_id": "c1", "ts": 1, "grade": 1, "time_ms": 900}
    fields.update(overrides)
    return StudyRecord(**fields)


def test_build_queue_reask_kind_on_a_word_card_lapse(derivations, db, w1):
    """derivations.reasks over a word: a "good" picture rating whose
    Production card has lapsed is a contradiction worth re-asking, same
    as a pair's rendition (spec 5 section 1 kind 4).
    """
    db.append_study(_word_study_row(w1.id))
    _learner(db, w1.id, "picture", "sA", "good")
    items = rs.build_queue(derivations, study=db, budget=50)
    reasks = [i for i in items if i["type"] == "reask" and i["subject"] == w1.id]
    assert len(reasks) == 1
    assert reasks[0]["kind"] == "picture"
    assert reasks[0]["subject_kind"] == "word"
    assert reasks[0]["original_answer"] == "good"
    assert reasks[0]["current"]["sha"] == "sA"
    assert reasks[0]["evidence"][0]["anchor"] == w1.id
    assert reasks[0]["evidence"][0]["card_kind"] == "production"


def test_build_queue_yields_no_reask_without_studyreader(derivations, db, pair, confusion):
    db.append_study(_pair_study_row(pair))
    _learner(db, confusion.id, "rendition", "rend-sha", "acceptable")
    items = rs.build_queue(derivations, study=None, budget=50)
    assert not [i for i in items if i["type"] == "reask"]


# --- append_answer: keys, ratings, idempotence ------------------------------

def test_append_answer_action_maps_to_rating_and_learner_key(db, w1):
    result = rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": 4,
                                   "artifact_sha": "sA"})
    assert result["ok"] is True
    assert result["rating"] == "good"
    rows = db.assessments_of(w1.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.port == "assess" and row.backend == "learner"
    assert row.key == "learner:sA:picture-for-word"
    assert row.answer["value"] == "good"
    assert row.question["kind"] == "rating"  # record.learner_ratings reads this back


def test_append_answer_unacceptable_none_records_the_rejected_artifact(db, w1):
    # spec 5 section 1 kind 1: a rejection names the artifact it rejects --
    # the screen's own question passes the current-best artifact_sha it
    # displayed.
    rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": 1,
                          "artifact_sha": "sA"})
    row = db.assessments_of(w1.id)[0]
    assert row.question["artifact_sha"] == "sA"
    assert row.answer["value"] == "unacceptable-none"
    assert row.key == "learner:sA:picture-for-word"


def test_append_answer_unacceptable_none_with_no_artifact_named_records_none(db, w1):
    # A rejection over a subject with no current artifact at all (nothing
    # to name) still appends -- LearnerKey's own "-" placeholder.
    rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": 1})
    row = db.assessments_of(w1.id)[0]
    assert row.question["artifact_sha"] is None
    assert row.key == "learner:-:picture-for-word"


def test_append_answer_carries_optional_note(db, w1):
    rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": 3,
                          "artifact_sha": "sA", "note": "too blurry"})
    row = db.assessments_of(w1.id)[0]
    assert row.answer["note"] == "too blurry"


def test_typed_direction_is_a_direction_row_not_a_rating(db, w1):
    """spec 5 section 1 kind 2: "a typed direction is recorded as a
    direction, not a rating" -- record.directions reads it back, and it
    makes the subject directed (derivations.directed), never a
    LEARNER_RANK-carrying rating row.
    """
    from thai_syllabus.derivations import directed
    from thai_syllabus.record import directions, learner_ratings

    result = rs.append_answer(db, {"subject": w1.id, "kind": "picture",
                                   "direction": "a woman pointing at herself"})
    assert result == {"ok": True, "ts": result["ts"], "kind": "direction"}
    rows = db.assessments_of(w1.id)
    assert directions(rows)[-1].answer["direction"] == "a woman pointing at herself"
    assert not learner_ratings(rows)          # a direction is never a rating row
    assert directed(db, w1.id)


def test_typed_direction_key_is_typed_not_a_rating_key(db, w1):
    rs.append_answer(db, {"subject": w1.id, "kind": "picture",
                          "direction": "a woman pointing at herself"})
    row = db.assessments_of(w1.id)[0]
    from thai_syllabus.cachekeys import DirectionKey, sha
    expected = DirectionKey(subject=w1.id, role="picture-for-word",
                            text_sha=sha("a woman pointing at herself"))
    assert row.key == expected.encode()


def test_append_answer_challenger_switch_rates_the_challenger(db, w1):
    result = rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": "switch",
                                   "artifact_sha": "s-new"})
    assert result["ok"] is True
    row = db.assessments_of(w1.id)[0]
    assert row.question["artifact_sha"] == "s-new"
    assert row.answer["value"] == "acceptable"


def test_append_answer_challenger_keep_appends_nothing(db, w1):
    result = rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": "keep"})
    assert result["ok"] is True
    assert db.assessments_of(w1.id) == []


def test_append_answer_waiver_uses_finding_identity_key(db):
    result = rs.append_answer(db, {"finding": {"rule": "r1", "note_id": "n1",
                                               "artifact_sha": "sA"}, "waived": True,
                                   "reason": "known false positive"})
    assert result["kind"] == "waiver"
    from thai_syllabus.rules import Finding
    assert db.is_waived(Finding(rule="r1", note_id="n1", artifact_sha="sA", evidence=""))


def test_waivers_from_either_write_path_appear_under_the_note_subject(db):
    # store.append_waiver and reviewserver.append_answer's waiver branch
    # are two separate RecordWriter callers; both must land the waiver
    # under subject=note_id, so assessments_of(note_id) sees it either way.
    db.append_waiver(rule_id="r1", note_id="n1", artifact_sha=None,
                     waived=True, reason="from the store path")
    rs.append_answer(db, {"finding": {"rule": "r2", "note_id": "n1",
                                      "artifact_sha": None}, "waived": True,
                          "reason": "from the reviewserver path"})
    kinds_and_reasons = [(a.question.get("kind"), a.answer.get("reason"))
                        for a in db.assessments_of("n1")]
    assert ("waiver", "from the store path") in kinds_and_reasons
    assert ("waiver", "from the reviewserver path") in kinds_and_reasons


def test_append_answer_rejects_unknown_rating(db, w1):
    with pytest.raises(ValueError):
        rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": 3,
                              "rating": "not-a-real-rating", "artifact_sha": "sA"})


def test_append_answer_same_answer_twice_is_idempotent_in_derived_state(derivations, db, w1):
    from thai_syllabus.derivations import current_best
    payload = {"subject": w1.id, "kind": "picture", "action": 4, "artifact_sha": "sA"}
    rs.append_answer(db, payload)
    best_after_first = current_best(db, w1.id, "picture", current_rubric={}, prior=(),
                                    provenance_source=lambda s: None)
    rs.append_answer(db, dict(payload))
    best_after_second = current_best(db, w1.id, "picture", current_rubric={}, prior=(),
                                     provenance_source=lambda s: None)
    assert best_after_first.artifact_sha == best_after_second.artifact_sha
    assert best_after_first.rank == best_after_second.rank
    assert best_after_first.source == best_after_second.source
    # append-only: the row count DOES grow even though derived state doesn't.
    assert len(db.assessments_of(w1.id)) == 2


def _png_bytes() -> bytes:
    """A valid, decodable PNG -- add_image now requires one, unlike an
    arbitrary bytes stand-in."""
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


# --- append_supply: path and url flows --------------------------------------

def test_supplied_picture_is_normalized_recorded_and_visible(tmp_path, derivations, db,
                                                             media_store, w1):
    """spec 5 section 1 kind 2: a supplied picture goes through the media
    ingest path (normalized, provenance row source=learner) and is what
    current_best picks (the implicit use-this)."""
    src = tmp_path / "candidate.png"
    src.write_bytes(_png_bytes())
    ctx = rs.ReviewContext(derivations=derivations)
    out = rs.append_supply(ctx, {"subject": w1.id, "kind": "picture", "source": "path",
                                 "value": str(src)})
    assert out["ok"] is True
    sha = out["artifact_sha"]
    assert media_store.has(sha, "png")  # add_image normalized it, not a raw write
    assert db.media_provenance(sha)["source"] == "learner"
    assert db.media_provenance(sha)["kind"] == "picture"
    assert ctx.current_best(w1.id, "picture").artifact_sha == sha
    row = db.assessments_of(w1.id)[0]
    assert row.answer["value"] == "unacceptable-use-this"
    assert row.answer["provenance"]["source"] == "learner"
    assert row.question["kind"] == "rating"  # record.learner_ratings reads this back
    # no provide row for a local path -- nothing to cache-key against.
    assert not [r for r in db.assessments_of(w1.id) if r.port == "provide"]


def test_append_supply_from_url_goes_through_imgfetch_provider(derivations, db, media_store, w1):
    def fake_fetcher(url):
        assert url == "https://example.test/pic.png"
        return _png_bytes(), "png"

    ctx = rs.ReviewContext(derivations=derivations, url_fetchers={"picture": fake_fetcher})
    result = rs.append_supply(ctx, {"subject": w1.id, "kind": "picture", "source": "url",
                                    "value": "https://example.test/pic.png"})
    assert result["ok"] is True
    sha = result["artifact_sha"]
    assert media_store.has(sha, "png")
    assert db.media_provenance(sha)["source"] == "learner"
    rows = db.assessments_of(w1.id)
    assert any(r.port == "provide" and r.backend == "imgfetch" for r in rows)
    assert any(r.port == "assess" and r.answer["value"] == "unacceptable-use-this" for r in rows)


def test_append_supply_url_fetch_is_cache_first(derivations, db, media_store, w1):
    calls = []

    def counting_fetcher(url):
        calls.append(url)
        return _png_bytes(), "png"

    ctx = rs.ReviewContext(derivations=derivations, url_fetchers={"picture": counting_fetcher})
    payload = {"subject": w1.id, "kind": "picture", "source": "url",
              "value": "https://example.test/pic.png"}
    rs.append_supply(ctx, payload)
    rs.append_supply(ctx, dict(payload))
    assert len(calls) == 1  # 2nd ask hits the cache, no 2nd fetch


def test_supplied_recording_url_uses_audiofetch(derivations, db, media_store, w1):
    """spec 5 section 1 kind 2: a recording URL is fetched by kind, through
    audiofetch, never imgfetch (which refuses non-images)."""
    calls = []

    def fake_audio_fetcher(url):
        calls.append(url)
        return b"fake-mp3-bytes", "mp3"

    ctx = rs.ReviewContext(derivations=derivations, url_fetchers={"recording": fake_audio_fetcher})
    out = rs.append_supply(ctx, {"subject": w1.id, "kind": "recording", "source": "url",
                                 "value": "https://x/y.mp3"})
    assert out["ok"] is True
    assert calls == ["https://x/y.mp3"]
    sha = out["artifact_sha"]
    assert media_store.has(sha, "mp3")
    provenance = db.media_provenance(sha)
    assert provenance["source"] == "learner"
    assert provenance["kind"] == "recording"
    assert provenance["speaker_id"] == "learner"
    assert db.speaker("learner").kind == "native"


def test_supplied_recording_from_local_path_writes_the_real_ext_unnormalized(
        tmp_path, derivations, db, media_store, w1):
    """A local recording is a direct learner act (no Provider backend, no
    cache key) -- MediaStore.write, not add_image: recordings are never
    normalized (spec 4 section 3 normalizes pictures only)."""
    src = tmp_path / "candidate.wav"
    src.write_bytes(b"fake-wav-bytes")
    ctx = rs.ReviewContext(derivations=derivations)
    out = rs.append_supply(ctx, {"subject": w1.id, "kind": "recording", "source": "path",
                                 "value": str(src)})
    assert out["ok"] is True
    sha = out["artifact_sha"]
    assert media_store.has(sha, "wav")
    assert media_store.path_for(sha, "wav").read_bytes() == b"fake-wav-bytes"
    assert db.media_provenance(sha)["speaker_id"] == "learner"


# --- gallery / notes / drills ------------------------------------------------

def test_gallery_cards_come_from_the_compile(deck_with_history):
    """spec 5 section 1: the proof gallery renders every card the Compile
    would write. Card kinds are compile.py's own template names (family-
    prefixed only where two families share a name); every card carries
    its model's CSS, so the gallery shows what Anki shows (F4).
    """
    ctx = rs.load_context(deck_with_history)
    kinds = {c["kind"] for c in ctx.cards()}
    assert kinds <= {"listening", "production", "reading", "spelling", "recognition",
                     "grapheme-reading", "cloze", "sentence-listening"}
    assert all(c["css"] == CARD_CSS for c in ctx.cards())


def test_gallery_cards_render_front_and_back_html_in_introduction_order(derivations, db, w1):
    # pair has no seeded rendition and grapheme has no name_word (this
    # module's fixtures), so build_deck drops both (spec 4 section 1) --
    # this test measures the word family's own Reading card, which needs
    # neither.
    from thai_syllabus.compile import build_deck

    _judge(db, w1.id, "picture", "sha-w1", True)
    cards = rs.compiled_cards(derivations)
    built_deck = build_deck(derivations.syllabus, derivations.db, derivations.media_store)

    # sequential, in the due order build_deck assigns from Syllabus.order()
    # (spec 5 section 1: "every card rendered ... in introduction order") --
    # build_deck.built itself chains family by family (word, pair,
    # grapheme, sentence), so it's compiled_cards's own sort that recovers
    # introduction order, checked here against each card's base_due.
    assert [c["index"] for c in cards] == list(range(len(cards)))
    base_due_by_subject = {built.subject: built.base_due for built in built_deck.built}
    dues = [base_due_by_subject[c["id"]] for c in cards]
    assert dues == sorted(dues)

    reading_card = next(c for c in cards if c["kind"] == "reading" and c["id"] == w1.id)
    assert w1.thai in reading_card["front_html"]
    assert w1.meaning in reading_card["back_html"]
    assert reading_card["family"] == "word"
    assert reading_card["subject"] == w1.id
    assert reading_card["gloss"] == w1.meaning

    # no bespoke card shape: only the compile's own front/back/css plus
    # the note's own field/tag metadata, never structured fields like
    # "thai"/"picture" composed by the gallery itself.
    for card in cards:
        assert set(card) == {"index", "id", "family", "kind", "subject",
                             "front_html", "back_html", "css", "gloss"}


def test_compiled_cards_carry_pair_confusion_and_stimulus_member(
        db, media_store, w1, w2, pair, confusion):
    """spec 5 section 1: the pair drill's per-confusion accuracy logging
    reads `confusion`/`stimulus_member` straight off each minimal_pair
    note's own tags (compile.py's tag_value) -- the pair id (`subject`)
    is shared by both member notes; `stimulus_member` (the member index
    this note's own Stimulus is) tells the two notes' recognition cards
    apart.
    """
    from datetime import date

    from thai_syllabus.cachekeys import rendition_identity
    from thai_syllabus.media import Speaker
    from thai_syllabus.wiring import _DbMediaIndex

    speaker = "somchai"
    db.add_speaker(Speaker(id=speaker, kind="native"))
    shas = {}
    for member in pair.members:
        sha = media_store.write(f"rendition:{pair.id}:{member}".encode(), ext="mp3")
        db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                     origin="https://forvo.com/x", licence="cc-by",
                     acquired=date(2026, 1, 1), speaker_id=speaker)
        shas[member] = sha
    db.append(port="assess", backend="rendition", key=f"rendition:{pair.id}", subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": rendition_identity(shas),
                      "rubric": None, "kind": "rendition", "subject_kind": "pair",
                      "params": {"members": shas}},
             answer={"value": True})

    syllabus = Syllabus(words=(w1, w2), pairs=(pair,), confusions=(confusion,),
                        media=_DbMediaIndex(db=db, pairs=(pair,)), assessments=db,
                        tokenizer=FakeTokenizer())
    derivations = Derivations(syllabus=syllabus, db=db, media_store=media_store,
                              current_rubric={}, prior=(), provenance_source=lambda sha: None,
                              sources_for=sources_for, attempt_cap=DEFAULT_ATTEMPT_CAP)

    pair_cards = [c for c in rs.compiled_cards(derivations) if c["family"] == "minimal_pair"]
    assert len(pair_cards) == 2
    for card in pair_cards:
        assert card["kind"] == "recognition"
        assert card["subject"] == pair.id
        assert card["confusion"] == confusion.id
    assert {c["stimulus_member"] for c in pair_cards} == {0, 1}


def test_resolve_media_for_web_rewrites_img_and_sound_to_the_media_route():
    html = ('<img src="deadbeef.jpg">'
           '<div>[sound:cafef00d.mp3]</div>')
    resolved = rs._resolve_media_for_web(html)
    assert '<img src="/media/deadbeef">' in resolved
    assert '<audio controls src="/media/cafef00d"></audio>' in resolved
    assert ".jpg" not in resolved
    assert ".mp3" not in resolved


def test_append_gallery_note_appends_learner_row_not_a_file(db):
    ts = rs.append_gallery_note(db, card_id="t-rice", kind="target", text="lovely bowl of rice")
    assert isinstance(ts, int)
    rows = db.assessments_of("t-rice")
    assert len(rows) == 1
    assert rows[0].answer == {"kind": "rating", "rating": None, "note": "lovely bowl of rice"}
    assert rows[0].key == "learner:t-rice:card-flag"


def test_append_drill_result_is_study_adjacent_not_study_table(db, confusion, pair):
    rs.append_drill_result(db, confusion=confusion.id, pair_id=pair.id, correct=True)
    rs.append_drill_result(db, confusion=confusion.id, pair_id=pair.id, correct=False)
    rows = db.assessments_of(confusion.id)
    assert len(rows) == 2
    assert all(r.question.get("kind") == "drill" for r in rows)
    import sqlite3
    con = sqlite3.connect(db.path)
    assert con.execute("select count(*) from study").fetchone()[0] == 0


# --- stats -------------------------------------------------------------------

def test_compute_stats_counts_ratings_coverage_exhausted_and_drills(
        derivations, db, w1, w2, confusion, pair):
    _learner(db, w1.id, "picture", "s1", "good")
    _learner(db, w2.id, "picture", "s2", "acceptable")
    rs.append_drill_result(db, confusion=confusion.id, pair_id=pair.id, correct=True)
    rs.append_drill_result(db, confusion=confusion.id, pair_id=pair.id, correct=False)

    session = rs.SessionStats(answered=3, queued=10)
    stats = rs.compute_stats(derivations, session=session)

    assert stats["session"] == {"answered": 3, "queued": 10}
    assert stats["ratings"]["good"] == 1
    assert stats["ratings"]["acceptable"] == 1
    assert stats["coverage"]["picture"]["covered"] == 2
    assert stats["coverage"]["picture"]["total"] == 2
    assert stats["drills"][confusion.id] == {"correct": 1, "total": 2}
    assert stats["run_report_history"] == []
    assert stats["pending"] == 0
    assert stats["sentences_adopted"] == 0


def test_compute_stats_reads_pending_and_sentences_adopted_from_the_newest_runreport(
        derivations, syllabus, db):
    # run.py's _persist_report convention: port="run", backend="runreport",
    # key="runreport", subject="run" -- one row per run() call, newest wins.
    db.append(port="run", backend="runreport", key="runreport", subject="run",
             question={}, answer={"attempted": 1, "improved": 0, "exhausted": 0,
                                  "available": 2, "pending": 3, "sentences_adopted": 4},
             cost=0.0)
    stats = rs.compute_stats(derivations)
    assert stats["pending"] == 3
    assert stats["sentences_adopted"] == 4


# --- HTTP layer (spec 5 section 2 endpoints, live loopback server) ---------

@pytest.fixture
def live_server(derivations, syllabus, tmp_path, media_store):
    # sqlite3 connections are single-thread by default (store.py's
    # SyllabusDb doesn't override that -- out of this module's scope to
    # change), and HTTPServer.serve_forever() handles requests on the
    # thread that calls it, not the fixture/test thread -- so the
    # RecordWriter/CacheReader db must be OPENED on the server thread.
    # `syllabus`/`media_store` are plain dataclasses with no thread
    # affinity and are safe to build in the fixture thread and hand over.
    db_path = tmp_path / "syllabus.db"
    ready = threading.Event()
    state: dict = {}

    def run() -> None:
        db_local = SyllabusDb(db_path)
        # syllabus.assessments must be db_local too: Syllabus.gaps() reads
        # waivers/verdicts through it, and that query must run on this
        # server thread, not the fixture thread that built `syllabus`.
        server_syllabus = dataclasses.replace(syllabus, assessments=db_local)
        server_derivations = dataclasses.replace(derivations, syllabus=server_syllabus,
                                                 db=db_local)
        ctx = rs.ReviewContext(derivations=server_derivations, study=db_local)
        httpd = HTTPServer(("127.0.0.1", 0), rs.build_app(ctx))
        state["port"] = httpd.server_address[1]
        state["httpd"] = httpd
        ready.set()
        httpd.serve_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(5), "server thread did not start in time"
    yield state["port"], db_path
    state["httpd"].shutdown()
    thread.join(timeout=5)


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def _post(port, path, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload).encode("utf-8")
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = resp.read()
    conn.close()
    return resp.status, out


def test_http_index_serves_html(live_server):
    port, _db_path = live_server
    status, body = _get(port, "/")
    assert status == 200
    assert b"<title>Review</title>" in body


def test_http_api_queue_returns_json_list(live_server, w1):
    port, _db_path = live_server
    status, body = _get(port, "/api/queue")
    assert status == 200
    items = json.loads(body)
    assert isinstance(items, list)
    assert any(i["subject"] == w1.id for i in items)


def test_http_api_cards_returns_json_list(live_server):
    port, _db_path = live_server
    status, body = _get(port, "/api/cards")
    assert status == 200
    assert isinstance(json.loads(body), list)


def test_http_answer_post_appends_row(live_server, w1):
    port, db_path = live_server
    status, body = _post(port, "/api/answer", {"subject": w1.id, "kind": "picture",
                                                "action": 4, "artifact_sha": "sA"})
    assert status == 200
    assert json.loads(body)["ok"] is True
    # verify persistence via a fresh connection to the same file -- the
    # server thread's own SyllabusDb object is not safe to touch from the
    # test/main thread (sqlite3's check_same_thread rule).
    verify_db = SyllabusDb(db_path)
    assert len(verify_db.assessments_of(w1.id)) == 1


def test_http_answer_accepts_a_rejection_naming_current_best(live_server, w1):
    port, db_path = live_server
    _post(port, "/api/answer", {"subject": w1.id, "kind": "picture", "action": 4,
                                "artifact_sha": "sA"})
    status, body = _post(port, "/api/answer", {"subject": w1.id, "kind": "picture",
                                                "action": 1, "artifact_sha": "sA"})
    assert status == 200
    assert json.loads(body)["ok"] is True
    verify_db = SyllabusDb(db_path)
    assert len(verify_db.assessments_of(w1.id)) == 2


def test_http_answer_refuses_a_rejection_naming_a_stale_artifact(live_server, w1):
    """spec 5 section 1 kind 1: the server verifies a rejection's own
    artifact_sha is still current-best and refuses (never appends) when
    the screen's question has moved on since it was displayed.
    """
    port, db_path = live_server
    _post(port, "/api/answer", {"subject": w1.id, "kind": "picture", "action": 4,
                                "artifact_sha": "sA"})
    status, body = _post(port, "/api/answer", {"subject": w1.id, "kind": "picture",
                                                "action": 1, "artifact_sha": "sB"})
    assert status == 400
    assert json.loads(body)["ok"] is False
    verify_db = SyllabusDb(db_path)
    assert len(verify_db.assessments_of(w1.id)) == 1   # the refused rejection never appended


def test_http_media_serves_bytes_by_sha(live_server, media_store):
    sha = media_store.write(b"hello-media", "jpg")
    port, db_path = live_server
    SyllabusDb(db_path).add_media(sha=sha, kind="picture", ext="jpg", source="test",
                                  origin="test", licence="test", acquired=date.today())
    status, body = _get(port, f"/media/{sha}")
    assert status == 200
    assert body == b"hello-media"


def test_http_media_404_for_unknown_sha(live_server):
    port, _db_path = live_server
    status, _body = _get(port, "/media/does-not-exist")
    assert status == 404


def test_http_media_404_when_file_exists_but_has_no_provenance_row(live_server, media_store):
    """`_find_media_file` reads the ext from the media table (spec 2
    section 2): an object with no provenance row is not served by
    globbing the objects directory for it."""
    sha = media_store.write(b"orphan-bytes", "jpg")
    port, _db_path = live_server
    status, _body = _get(port, f"/media/{sha}")
    assert status == 404


def test_http_stats_endpoint(live_server):
    port, _db_path = live_server
    status, body = _get(port, "/stats")
    assert status == 200
    stats = json.loads(body)
    assert "session" in stats and "coverage" in stats


# --- the screen derives what the run derives (spec 5 section 3) ------------

@pytest.fixture
def deck_with_history(tmp_path):
    """A deck whose record holds two judged pictures -- one judged under a
    rubric the deck has since moved off, one under the deck's current
    rubric -- and an attempt cap of 1 in providers.yaml. A screen deriving
    under parameters of its own rather than the deck's disagrees with the
    run about current-best, the queue and exhaustion.
    """
    from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
    from thai_syllabus.entities import Category
    from thai_syllabus.profile import Profile
    from thai_syllabus.wiring import build_sourcing

    rice = word("rice", "ข้าว", "rice")     # ข้าว = rice
    fish = word("fish", "ปลา", "fish")      # ปลา = fish
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=(rice, fish),
        targets=(target("t-rice", rice.id), target("t-fish", fish.id)),
        graphemes=(), confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset({rice.id, fish.id})),)))
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n"
        "attempt_cap: 1\n"
        "secrets: {anthropic: op://Shared/Anthropic/API Key}\n"
        "judge: {transport: batch, model: m, "
        "price_per_mtok: {input: 2.0, output: 10.0}}\n", encoding="utf-8")
    MediaStore(root / "media")
    SyllabusDb(root / "syllabus.db").close()

    current_rubric = build_sourcing(root).rubrics["picture-for-word"]
    deck_db = SyllabusDb(root / "syllabus.db")
    for subject, sha, rubric in (("rice", "pic-rice", "a rubric this deck has moved off"),
                                 ("fish", "pic-fish", current_rubric)):
        deck_db.append(port="provide", backend="openverse", key=f"openverse:{subject}",
                       subject=subject,
                       question={"kind": "picture", "params": {"query": subject}},
                       answer={"items": [{"sha": sha}]})
        deck_db.append(port="assess", backend="judge", key=f"judge:{sha}:picture-for-word",
                       subject=subject,
                       question={"role": "picture-for-word", "artifact_sha": sha,
                                "rubric": rubric, "kind": "picture"},
                       answer={"value": True})
    # one more source asked for rice since its candidate arrived: an attempt
    # the deck's own cap of 1 counts as its last.
    deck_db.append(port="provide", backend="wikimedia", key="wikimedia:rice", subject="rice",
                   question={"kind": "picture", "params": {"query": "rice"}},
                   answer={"items": []})
    deck_db.close()
    return root


def test_screen_and_run_agree_on_current_best_queue_and_exhaustion(deck_with_history):
    from thai_syllabus.attempts import provenance_source_for
    from thai_syllabus.derivations import current_best, exhausted, queue
    from thai_syllabus.wiring import build_sourcing

    ctx = rs.load_context(deck_with_history)
    src = build_sourcing(deck_with_history)
    provenance_source = provenance_source_for(src.db)

    for subject in ("rice", "fish"):
        assert ctx.current_best(subject, "picture") == current_best(
            src.db, subject, "picture", current_rubric=src.rubrics,
            prior=src.provenance_prior, provenance_source=provenance_source)

    assert [(e.subject, e.kind) for e in ctx.queue()] == [
        (e.subject, e.kind) for e in queue(
            src.syllabus, src.db, current_rubric=src.rubrics, prior=src.provenance_prior,
            sources_for=src.sources_for, attempt_cap=src.attempt_cap,
            provenance_source=provenance_source)]

    assert ctx.exhausted("rice", "picture") == exhausted(
        src.db, "rice", "picture", sources=src.sources_for("picture"),
        attempt_cap=src.attempt_cap)


def test_screen_stats_count_coverage_and_exhaustion_as_the_run_does(deck_with_history):
    from thai_syllabus.attempts import provenance_source_for
    from thai_syllabus.derivations import LEARNER_RANK, available_needs, current_best, exhausted
    from thai_syllabus.wiring import build_sourcing

    ctx = rs.load_context(deck_with_history)
    src = build_sourcing(deck_with_history)
    provenance_source = provenance_source_for(src.db)

    coverage: dict[str, dict[str, int]] = {}
    exhausted_remaining = 0
    for subject, kind, _subject_kind in available_needs(src.syllabus):
        best = current_best(src.db, subject, kind, current_rubric=src.rubrics,
                            prior=src.provenance_prior, provenance_source=provenance_source)
        bucket = coverage.setdefault(kind, {"covered": 0, "total": 0})
        bucket["total"] += 1
        if best.rank >= LEARNER_RANK["acceptable"]:
            bucket["covered"] += 1
        if exhausted(src.db, subject, kind, sources=src.sources_for(kind),
                     attempt_cap=src.attempt_cap).exhausted:
            exhausted_remaining += 1

    stats = ctx.stats()
    assert stats["coverage"] == coverage
    assert stats["exhausted_remaining"] == exhausted_remaining


# --- load_context: one assembly path with the run and the compiler ---------

def test_load_context_builds_its_syllabus_through_the_shared_loader(tmp_path):
    """load_context used to construct a bare Syllabus inline -- no media
    index, no sentences, no frequency map, no rulebook overlay -- so the
    review screen reported gaps the run had already closed. It must build
    the same Syllabus wiring.load_syllabus builds for the run and the
    compiler.
    """
    from datetime import date

    from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
    from thai_syllabus.entities import Category
    from thai_syllabus.profile import Profile
    from thai_syllabus.rulebook import PICTURE_FIT_RUBRIC

    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=(word("orange", "ส้ม", "orange"),),
        targets=(target("orange/receptive", "orange"),),
        graphemes=(), confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset({"orange"})),)))
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n",
        encoding="utf-8")
    deck_db = SyllabusDb(root / "syllabus.db")
    deck_db.append(port="provide", backend="openverse", key="openverse:orange",
                   subject="orange", question={"kind": "picture", "params": {}},
                   answer={"items": [{"sha": "pic1"}]})
    deck_db.append(port="assess", backend="judge", key="judge:x:pic1:picture-for-word",
                   subject="orange",
                   question={"role": "picture-for-word", "artifact_sha": "pic1",
                             "rubric": PICTURE_FIT_RUBRIC, "kind": "picture"},
                   answer={"value": True})
    deck_db.add_media(sha="pic1", kind="picture", ext="jpg", source="openverse",
                      origin="https://example.com/x.jpg", licence="cc0",
                      acquired=date(2026, 1, 1))
    deck_db.close()

    ctx = rs.load_context(root)

    assert "orange" not in ctx.syllabus.gaps().words_missing_pictures
    assert ctx.syllabus.media.picture_sha(WordId("orange")) == "pic1"
    # the rest of ReviewContext is unchanged
    assert ctx.cache is ctx.record
    assert ctx.syllabus.assessments is ctx.cache
