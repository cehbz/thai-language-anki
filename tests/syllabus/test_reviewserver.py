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
from http.server import HTTPServer

import pytest

from PIL import Image as PILImage

from thai_syllabus import reviewserver as rs
from thai_syllabus.entities import Grapheme, MinimalPair, SoundConfusion
from thai_syllabus.ids import ConfusionId, PairId, WordId
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus

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
    role = rs._role(kind)
    answer = {"value": value}
    if evidence:
        answer["evidence"] = evidence
    return db.append(port="assess", backend="judge", key=f"judge:{subject}:{artifact_sha}",
                     subject=subject,
                     question={"role": role, "artifact_sha": artifact_sha, "rubric": rubric,
                              "kind": kind},
                     answer=answer)


def _learner(db, subject, kind, artifact_sha, rating):
    role = rs._role(kind)
    return db.append(port="assess", backend="learner",
                     key=f"learner:{artifact_sha}:{role}", subject=subject,
                     question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                              "kind": "rating"},
                     answer={"value": rating})


# --- build_queue: budget + F10 order ----------------------------------------

def test_build_queue_respects_budget(syllabus, db):
    items = rs.build_queue(syllabus, db, budget=1)
    assert len(items) == 1


def test_build_queue_rate_order_matches_derivations_queue(syllabus, db):
    from thai_syllabus.derivations import queue as derive_queue
    entries = derive_queue(syllabus, db)
    items = rs.build_queue(syllabus, db, budget=len(entries))
    rate_items = [i for i in items if i["type"] == "rate"]
    assert [(i["subject"], i["kind"]) for i in rate_items] == \
           [(e.subject, e.kind) for e in entries]


def test_build_queue_rate_item_carries_gloss_query_verdict_and_thumbnails(syllabus, db, w1):
    _provide(db, w1.id, "picture", query="rice photo", items=[{"sha": "sA"}, {"sha": "sB"}])
    _judge(db, w1.id, "picture", "sA", True, evidence="clear rice bowl")
    items = rs.build_queue(syllabus, db, budget=50)
    rated = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                and i["kind"] == "picture")
    assert rated["gloss"] == "rice"
    assert rated["query"] == "rice photo"
    assert rated["current"]["sha"] == "sA"
    assert "judge: pass" in rated["current"]["verdict"]
    assert rated["rejected"] == [{"sha": "sB", "url": "/media/sB"}]


def test_build_queue_direction_kind_for_exhausted_subject(syllabus, db, w1):
    _provide(db, w1.id, "picture", items=[{"sha": "s1"}])
    _judge(db, w1.id, "picture", "s1", True)
    _provide(db, w1.id, "picture", items=[{"sha": "s2"}])  # 2nd attempt, no judge verdict
    items = rs.build_queue(syllabus, db, budget=50, k=1, attempt_cap=2)
    direction = [i for i in items if i["type"] == "direction" and i["subject"] == w1.id
                and i["kind"] == "picture"]
    assert len(direction) == 1
    assert direction[0]["attempts"] == 2
    assert "openverse" in direction[0]["tried"]["sources"]


def test_build_queue_challenger_kind_when_rubric_change_outranks_learner_pick(syllabus, db, w1):
    _learner(db, w1.id, "picture", "s-old", "acceptable")
    _judge(db, w1.id, "picture", "s-new", True, rubric="rubric-v2")
    items = rs.build_queue(syllabus, db, budget=50, current_rubric="rubric-v2")
    challengers = [i for i in items if i["type"] == "challenger" and i["subject"] == w1.id]
    assert len(challengers) == 1
    assert challengers[0]["current"]["sha"] == "s-old"
    assert challengers[0]["challenger"]["sha"] == "s-new"


# --- _judge_verdict_line: a role -> rubric mapping current_rubric ----------

def test_judge_verdict_line_honors_a_role_scoped_rubric_mapping(db, w1):
    _provide(db, w1.id, "picture", items=[{"sha": "sA"}])
    _judge(db, w1.id, "picture", "sA", True, rubric="rubric-v2", evidence="clear")
    rows = rs._rows_for(db, w1.id, "picture")

    # a mapping naming this row's role (picture-for-word) under the same
    # rubric text -- the verdict line is shown.
    assert rs._judge_verdict_line(rows, "sA", {"picture-for-word": "rubric-v2"}) is not None

    # a mapping naming the SAME role under different rubric text -- the
    # row is stale under that role, so no verdict line.
    assert rs._judge_verdict_line(rows, "sA", {"picture-for-word": "some other text"}) is None


def test_build_queue_reask_kind_on_study_lapse_contradicting_learner_rating(syllabus, db, pair, confusion):
    db.append_study(card_key=f"{pair.id}::recognition", compile_id="c1", grade=1, time_ms=900)
    _learner(db, confusion.id, "rendition", "rend-sha", "acceptable")
    items = rs.build_queue(syllabus, db, study=db, budget=50)
    reasks = [i for i in items if i["type"] == "reask" and i["subject"] == confusion.id]
    assert len(reasks) == 1
    assert reasks[0]["original_answer"] == "acceptable"
    assert reasks[0]["evidence"][0]["card_key"] == f"{pair.id}::recognition"


def test_build_queue_yields_no_reask_without_studyreader(syllabus, db, pair, confusion):
    db.append_study(card_key=f"{pair.id}::recognition", compile_id="c1", grade=1, time_ms=900)
    _learner(db, confusion.id, "rendition", "rend-sha", "acceptable")
    items = rs.build_queue(syllabus, db, study=None, budget=50)
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


def test_append_answer_action_1_has_no_artifact_sha(db, w1):
    rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": 1,
                          "artifact_sha": "irrelevant"})
    row = db.assessments_of(w1.id)[0]
    assert row.question["artifact_sha"] is None
    assert row.key == "learner:-:picture-for-word"


def test_append_answer_carries_optional_note(db, w1):
    rs.append_answer(db, {"subject": w1.id, "kind": "picture", "action": 3,
                          "artifact_sha": "sA", "note": "too blurry"})
    row = db.assessments_of(w1.id)[0]
    assert row.answer["note"] == "too blurry"


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


def test_append_answer_same_answer_twice_is_idempotent_in_derived_state(db, w1):
    from thai_syllabus.derivations import current_best
    payload = {"subject": w1.id, "kind": "picture", "action": 4, "artifact_sha": "sA"}
    rs.append_answer(db, payload)
    best_after_first = current_best(db, w1.id, "picture")
    rs.append_answer(db, dict(payload))
    best_after_second = current_best(db, w1.id, "picture")
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

def test_append_supply_from_local_path(tmp_path, db, media_store, syllabus, w1):
    src = tmp_path / "candidate.jpg"
    src.write_bytes(b"fake-jpeg-bytes")
    ctx = rs.ReviewContext(syllabus=syllabus, cache=db, record=db, media_store=media_store)
    result = rs.append_supply(ctx, {"subject": w1.id, "kind": "picture", "source": "path",
                                    "value": str(src)})
    assert result["ok"] is True
    sha = result["artifact_sha"]
    assert media_store.has(sha, "jpg")
    row = db.assessments_of(w1.id)[0]
    assert row.answer["value"] == "unacceptable-use-this"
    assert row.answer["provenance"]["source"] == "learner"
    assert row.question["kind"] == "rating"  # record.learner_ratings reads this back
    # no provide row for a local path -- nothing to cache-key against.
    assert not [r for r in db.assessments_of(w1.id) if r.port == "provide"]


def test_append_supply_from_url_goes_through_imgfetch_provider(db, media_store, syllabus, w1):
    def fake_fetcher(url):
        assert url == "https://example.test/pic.png"
        return _png_bytes(), "png"

    ctx = rs.ReviewContext(syllabus=syllabus, cache=db, record=db, media_store=media_store,
                           url_fetcher=fake_fetcher)
    result = rs.append_supply(ctx, {"subject": w1.id, "kind": "picture", "source": "url",
                                    "value": "https://example.test/pic.png"})
    assert result["ok"] is True
    sha = result["artifact_sha"]
    assert media_store.has(sha, "png")
    rows = db.assessments_of(w1.id)
    assert any(r.port == "provide" and r.backend == "imgfetch" for r in rows)
    assert any(r.port == "assess" and r.answer["value"] == "unacceptable-use-this" for r in rows)


def test_append_supply_url_fetch_is_cache_first(db, media_store, syllabus, w1):
    calls = []

    def counting_fetcher(url):
        calls.append(url)
        return _png_bytes(), "png"

    ctx = rs.ReviewContext(syllabus=syllabus, cache=db, record=db, media_store=media_store,
                           url_fetcher=counting_fetcher)
    payload = {"subject": w1.id, "kind": "picture", "source": "url",
              "value": "https://example.test/pic.png"}
    rs.append_supply(ctx, payload)
    rs.append_supply(ctx, dict(payload))
    assert len(calls) == 1  # 2nd ask hits the cache, no 2nd fetch


# --- gallery / notes / drills ------------------------------------------------

def test_simplified_cards_orders_and_renders_target_pair_grapheme(syllabus, db, w1, w2, pair, grapheme, keyword_word):
    _judge(db, w1.id, "picture", "sha-w1", True)
    cards = rs.simplified_cards(syllabus, db)
    kinds = [c["kind"] for c in cards]
    assert "pair" in kinds and "grapheme" in kinds and "target" in kinds
    # order() puts sounds (pairs, then graphemes) before words (spec 1 order()).
    assert kinds.index("pair") < kinds.index("target")
    assert kinds.index("grapheme") < kinds.index("target")

    target_card = next(c for c in cards if c["kind"] == "target" and c["id"] == "t-rice")
    assert target_card["front"]["thai"] == w1.thai
    assert target_card["front"]["picture"] == "/media/sha-w1"
    assert target_card["gloss"] == "rice"

    pair_card = next(c for c in cards if c["kind"] == "pair")
    assert pair_card["drill"]["thai"] == w1.thai
    assert pair_card["drill"]["other_thai"] == w2.thai
    assert pair_card["drill"]["contrast"] == pair.confusion

    grapheme_card = next(c for c in cards if c["kind"] == "grapheme")
    assert grapheme_card["front"]["symbol"] == grapheme.symbol
    assert grapheme_card["back"]["keyword"] == keyword_word.thai


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

def test_compute_stats_counts_ratings_coverage_exhausted_and_drills(syllabus, db, w1, w2, confusion, pair):
    _learner(db, w1.id, "picture", "s1", "good")
    _learner(db, w2.id, "picture", "s2", "acceptable")
    rs.append_drill_result(db, confusion=confusion.id, pair_id=pair.id, correct=True)
    rs.append_drill_result(db, confusion=confusion.id, pair_id=pair.id, correct=False)

    session = rs.SessionStats(answered=3, queued=10)
    stats = rs.compute_stats(syllabus, db, session=session)

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
        syllabus, db):
    # run.py's _persist_report convention: port="run", backend="runreport",
    # key="runreport", subject="run" -- one row per run() call, newest wins.
    db.append(port="run", backend="runreport", key="runreport", subject="run",
             question={}, answer={"attempted": 1, "improved": 0, "exhausted": 0,
                                  "available": 2, "pending": 3, "sentences_adopted": 4},
             cost=0.0)
    stats = rs.compute_stats(syllabus, db)
    assert stats["pending"] == 3
    assert stats["sentences_adopted"] == 4


# --- HTTP layer (spec 5 section 2 endpoints, live loopback server) ---------

@pytest.fixture
def live_server(syllabus, tmp_path, media_store):
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
        ctx = rs.ReviewContext(syllabus=server_syllabus, cache=db_local, record=db_local,
                               media_store=media_store, study=db_local)
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


def test_http_media_serves_bytes_by_sha(live_server, media_store):
    sha = media_store.write(b"hello-media", "jpg")
    port, _db_path = live_server
    status, body = _get(port, f"/media/{sha}")
    assert status == 200
    assert body == b"hello-media"


def test_http_media_404_for_unknown_sha(live_server):
    port, _db_path = live_server
    status, _body = _get(port, "/media/does-not-exist")
    assert status == 404


def test_http_stats_endpoint(live_server):
    port, _db_path = live_server
    status, body = _get(port, "/stats")
    assert status == 200
    stats = json.loads(body)
    assert "session" in stats and "coverage" in stats


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
