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
import time
from datetime import date
from http.server import HTTPServer

import pytest

from PIL import Image as PILImage

from thai_syllabus import record as record_mod
from thai_syllabus import reviewserver as rs
from thai_syllabus.attempts import sources_for
from thai_syllabus.authority import role_for
from thai_syllabus.cachekeys import (AttemptOutcomeKey, CommentReadingKey, DirectionKey, FlagKey,
                                    JudgeKey, LearnerKey, MechanicalKey, PhraseKey, ProvideKey,
                                    RunReportKey, comment_identity, preference_identity, sha)
from thai_syllabus.compile import CARD_CSS
from thai_syllabus.derivations import DEFAULT_ATTEMPT_CAP, DEFAULT_TRANSIENT_CAP, directed
from thai_syllabus.entities import Grapheme, MinimalPair, Sentence, SoundConfusion
from thai_syllabus.media import Provenance
from thai_syllabus.ids import ConfusionId, PairId, WordId
from thai_syllabus.ports import StudyRecord
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.wiring import Derivations

from .builders import PROV, sentence, syl, pron, target, thai_of, word


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
                    graphemes=(grapheme,), confusions=(confusion,), assessments=db)


@pytest.fixture
def derivations(syllabus, db, media_store):
    """The bundle wiring.load_derivations builds from a deck, with an
    empty rubric mapping (no verdict is stale on a role's account) and no
    provenance prior -- the deck-independent parameters these unit tests
    measure their folds under.
    """
    return Derivations(syllabus=syllabus, db=db, media_store=media_store,
                       current_rubric={}, prior=(), provenance_source=lambda sha: None,
                       sources_for=sources_for, attempt_cap=DEFAULT_ATTEMPT_CAP,
                       transient_cap=DEFAULT_TRANSIENT_CAP)


# --- cache-row helpers (mirrors test_derivations.py's) ----------------------

def _provide(db, subject, kind, backend="openverse", items=(), query=None):
    """One whole attempt, in the row shapes a real one writes: the Source
    ask (carrying the query and the search hits), one bytes row per
    fetched candidate (backend imgfetch, one sha each -- the only row
    shape that ever carries a sha; same `kind` as the ask, distinguished
    from it by backend, not by a suffixed kind), and the attempt's own
    outcome row (port "attempt", spec 3 section 6) -- `candidates` when a
    sha was stored, `nothing` otherwise, the fold next_source/exhausted
    read.
    """
    params = {"query": query} if query else {}
    ts = db.append(port="provide", backend=backend,
                   key=ProvideKey(source=backend, kind="", query=str(query)),
                   subject=subject, question={"kind": kind, "params": params},
                   answer={"items": [i for i in items if not i.get("sha")]})
    stored: list[str] = []
    for item in items:
        if not item.get("sha"):
            continue
        url = f"https://x/{item['sha']}.jpg"
        ts = db.append(port="provide", backend="imgfetch",
                       key=ProvideKey(source="", kind="", query=url), subject=subject,
                       question={"kind": kind, "params": {"url": url}},
                       answer={"items": [dict(item)]})
        stored.append(item["sha"])
    ts = db.append(port="attempt", backend=backend,
                   key=AttemptOutcomeKey(subject=subject, kind=kind, source=backend),
                   subject=subject,
                   question={"kind": kind, "subject_kind": "word", "source": backend},
                   answer={"outcome": "candidates" if stored else "nothing",
                           "candidates": stored})
    return ts


def _judge(db, subject, kind, artifact_sha, value, rubric="rubric-v1", evidence=None):
    role = role_for(kind)
    answer = {"value": value}
    if evidence:
        answer["evidence"] = evidence
    return db.append(port="assess", backend="judge",
                     key=JudgeKey.for_rule(rubric, artifact_sha, subject, role),
                     subject=subject,
                     question={"role": role, "artifact_sha": artifact_sha, "rubric": rubric,
                              "kind": kind},
                     answer=answer)


def _learner(db, subject, kind, artifact_sha, rating):
    role = role_for(kind)
    return db.append(port="assess", backend="learner",
                     key=LearnerKey(artifact_sha=artifact_sha, role=role), subject=subject,
                     question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                              "kind": "rating"},
                     answer={"value": rating})


# --- build_queue: budget + F10 order ----------------------------------------

def test_build_queue_threads_its_one_clock_read_so_an_aged_out_nothing_re_offers_forvo(
        derivations, db, w1):
    """spec 3 r19 section 6a/9 threading guard: build_queue reads the
    clock once and hands it with nothing_ttl to queue() and exhausted();
    a w1 recording need whose forvo `nothing` is 200 days old (ttl 180)
    and whose tts `nothing` is fresh has no candidate, so an unsearched
    source (forvo, aged out) leaves it queued for neither a rate nor a
    direction question (spec 5 r7 F4) -- only once every source is
    exhausted (the un-aged case below) does it surface as direction."""
    day = 86_400 * 1_000_000_000
    for source, age in (("forvo", 200), ("tts", 1)):
        db.append(port="attempt", backend=source,
                  key=AttemptOutcomeKey(subject=w1.id, kind="recording", source=source),
                  subject=w1.id, question={"kind": "recording", "subject_kind": "word",
                                           "source": source},
                  answer={"outcome": "nothing", "candidates": []}, cost=0.0,
                  ts=time.time_ns() - age * day)
    aged = dataclasses.replace(derivations, nothing_ttl={"forvo": 180})
    items = [i for i in rs.build_queue(aged, budget=50)
             if i["subject"] == w1.id and i["kind"] == "recording"]
    assert items == []
    items = [i for i in rs.build_queue(derivations, budget=50)
             if i["subject"] == w1.id and i["kind"] == "recording"]
    assert [i["type"] for i in items] == ["direction"]


def test_build_queue_respects_budget(derivations, db):
    items = rs.build_queue(derivations, budget=1)
    assert len(items) == 1


def test_build_queue_rate_order_matches_derivations_queue(derivations, syllabus, db):
    from thai_syllabus.attempts import sources_for
    from thai_syllabus.derivations import DEFAULT_ATTEMPT_CAP, DEFAULT_TRANSIENT_CAP
    from thai_syllabus.derivations import queue as derive_queue
    # Every queued need gets a candidate so none is skipped for having
    # nothing to rate (spec 5 r7 F4) -- this test is only about F10 order.
    for w in syllabus.words:
        _provide(db, w.id, "picture", items=[{"sha": f"p-{w.id}"}])
        _provide(db, w.id, "recording", items=[{"sha": f"r-{w.id}"}])
    for p in syllabus.pairs:
        _provide(db, p.id, "rendition", items=[{"sha": f"v-{p.id}"}])
    entries = derive_queue(syllabus, db, current_rubric={}, prior=(), sources_for=sources_for,
                           attempt_cap=DEFAULT_ATTEMPT_CAP, transient_cap=DEFAULT_TRANSIENT_CAP,
                           provenance_source=lambda s: None)
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
    assert rated["rejected"] == [{"sha": "sB", "url": "/media/sB", "verdict": None}]


def test_rate_question_marks_learner_ranks_false_for_a_recording(derivations, db, w1):
    """r8: recording-for-word names no "learner" in AUTHORITY_ORDER -- the
    rate question tells the client so it can label the buttons a veto,
    not a rank.
    """
    _provide(db, w1.id, "recording", items=[{"sha": "r1"}])
    items = rs.build_queue(derivations, budget=50)
    rated = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                and i["kind"] == "recording")
    assert rated["learner_ranks"] is False


def test_rate_question_marks_learner_ranks_true_for_a_picture(derivations, db, w1):
    _provide(db, w1.id, "picture", items=[{"sha": "p1"}])
    items = rs.build_queue(derivations, budget=50)
    rated = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                and i["kind"] == "picture")
    assert rated["learner_ranks"] is True


def test_rate_button_label_text_covers_the_veto_and_ranking_variants(derivations, db, w1):
    """r8 fix round 2, ruling 3: action 2 is a nomination on every role
    kind ("...use this one instead"); on a veto-only role (a recording's
    rate question, learner_ranks False) 1/3/4 read as a veto and a note;
    on a learner-ranking role (a picture's, learner_ranks True) they keep
    their original vocabulary.
    """
    _provide(db, w1.id, "recording", items=[{"sha": "r1"}])
    _provide(db, w1.id, "picture", items=[{"sha": "p1"}])
    items = rs.build_queue(derivations, budget=50)
    recording = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                     and i["kind"] == "recording")
    picture = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                  and i["kind"] == "picture")
    assert recording["learner_ranks"] is False
    assert picture["learner_ranks"] is True

    assert '"2 unacceptable, use this one instead"' in rs.INDEX_HTML
    assert '"1 unacceptable (veto)"' in rs.INDEX_HTML
    assert '"3 acceptable (note)"' in rs.INDEX_HTML
    assert '"4 good (note)"' in rs.INDEX_HTML
    assert '"1 unacceptable-none"' in rs.INDEX_HTML
    assert '"3 acceptable"' in rs.INDEX_HTML
    assert '"4 good"' in rs.INDEX_HTML

    # both renderers (renderRate and renderReask) share the one label
    # mapping -- the label text above is never duplicated per renderer.
    assert rs.INDEX_HTML.count("rateLabels(q.learner_ranks)") == 2


# --- the page: card type labels, subject headers, comments (spec 5 r9) -----

def test_the_page_embeds_the_card_meaning_table_from_compile():
    """Spec 5 r9 (design ruling 4): the one-line meaning of every card
    type reaches the page as embedded JSON, keyed "family/kind" the way
    /api/cards reports them -- compile.CARD_MEANINGS is the one table
    (spec 4 section 1), never a copy in the script.
    """
    from thai_syllabus.compile import CARD_MEANINGS
    assert 'var CARD_MEANINGS = {' in rs.INDEX_HTML
    for (family, kind), meaning in CARD_MEANINGS.items():
        assert json.dumps(f"{family}/{kind}") in rs.INDEX_HTML
        assert json.dumps(meaning, ensure_ascii=False) in rs.INDEX_HTML
    assert "n comment" in rs.INDEX_HTML


def test_every_rendered_card_carries_its_type_label_with_the_meaning_tooltip():
    """Design ruling 4: one kindLabel helper, used by the gallery card
    and by every compiled card a question shows -- the tooltip is the
    embedded meaning, never recomposed per call site.
    """
    assert "function kindLabel(family, kind)" in rs.INDEX_HTML
    assert 'CARD_MEANINGS[family + "/" + kind]' in rs.INDEX_HTML
    assert rs.INDEX_HTML.count("kindLabel(card.family, card.kind)") == 2
    assert 'el("div", { "class": "kind" }, card.family + " / " + card.kind)' not in rs.INDEX_HTML
    assert ".kind[title] { cursor: help; }" in rs.INDEX_HTML


def test_every_question_names_its_subject_in_words_never_a_bare_sha():
    """Design ruling 5: all four question renderers head with
    subjectHeader (q.label's Thai, then id and kind), never the old
    `q.subject + " (" + q.kind + ")"` sha line.
    """
    assert "function subjectHeader(q, suffix)" in rs.INDEX_HTML
    assert "var thai = (q.label && q.label.thai)" in rs.INDEX_HTML
    assert rs.INDEX_HTML.count("subjectHeader(q") == 5  # the definition + four renderers
    assert 'subjectHeader(q, " — exhausted, attempts=" + q.attempts)' in rs.INDEX_HTML
    assert 'subjectHeader(q, " — a new candidate outranks your pick")' in rs.INDEX_HTML
    assert 'subjectHeader(q, " — lapse evidence contradicts a past rating")' in rs.INDEX_HTML
    assert 'q.subject + " (" + q.kind + ")"' not in rs.INDEX_HTML
    assert ".subject-header {" in rs.INDEX_HTML


def test_a_sentence_question_never_prints_its_text_sha_as_an_id():
    """Design ruling 5, fix: a sentence subject's id IS its text sha
    (compile.py tags the sentence family by text_sha), so subjectHeader
    drops the id line for a sentence -- its text and gloss are above it
    -- and keeps the id for every other subject kind, whose ids are
    meaningful (a word's, a pair's, a grapheme's). subjectHeader is the
    one place the page prints a subject in visible text.
    """
    assert 'var id = q.subject_kind === "sentence" ? null' in rs.INDEX_HTML
    assert '(id ? id + " · " : "") + q.kind + (suffix || "")' in rs.INDEX_HTML
    assert rs.INDEX_HTML.count("q.label.id") == 1
    # ... and the large line's own fallback, where the label lookup found
    # nothing: a sentence falls back to its gloss, never to its subject.
    assert '(q.subject_kind === "sentence" ? (q.gloss || "(sentence)") : q.subject)' \
        in rs.INDEX_HTML
    # the header's Thai is scoped: compile.CARD_CSS's own .thai (48px)
    # loads second, so an unscoped page rule never applied.
    assert ".subject-header .thai { font-size: 44px; }" in rs.INDEX_HTML
    assert "\n  .thai { font-size: 44px; }" not in rs.INDEX_HTML


def test_the_subjects_comments_are_listed_under_a_question_each_unread():
    """Spec 5 r9: one renderComments for both modes (the gallery card's
    own notes and a question's subject comments), each row carrying its
    reading -- "unread" until spec 5 r10's reading pass fills one in.
    """
    assert "function renderComments(items, box, subject, subjectKind, reload)" in rs.INDEX_HTML
    assert "function renderCardNotes" not in rs.INDEX_HTML
    assert "renderComments(card.notes, box, card.subject," in rs.INDEX_HTML
    assert rs.INDEX_HTML.count("renderComments(q.comments, box, q.subject,") == 4
    assert 'el("div", { "class": "reading unread" }, "unread")' in rs.INDEX_HTML
    assert 'on a " + (n.question_kind || "") + " question"' in rs.INDEX_HTML


def test_the_page_renders_readings_and_a_strike_control():
    """Spec 5 r10: an unread comment still reads "unread"; a read one
    shows the reading line, one line per action taken and per request
    nothing could be done about, and -- while it stands -- a strike
    control that POSTs the whole reading reference to /api/veto and
    reloads the view it struck from.
    """
    assert "/api/veto" in rs.INDEX_HTML
    assert '"read as: "' in rs.INDEX_HTML
    assert "strike" in rs.INDEX_HTML and "(struck)" in rs.INDEX_HTML
    assert "function renderReading(n, subject, subjectKind, reload)" in rs.INDEX_HTML
    assert 'if (!n.reading) { return el("div", { "class": "reading unread" }, "unread"); }' \
        in rs.INDEX_HTML
    # the struck reading stays visible and says so -- a veto hides nothing
    assert 'r.vetoed ? "reading struck" : "reading"' in rs.INDEX_HTML
    # an empty reading (a comment the reader closed or passed over,
    # attempts._NO_READING) prints no bare "read as: " line -- only its
    # unactionable line speaks -- but a struck one still says so
    assert 'var head = (r.reading ? "read as: " + r.reading : "") ' \
        '+ (r.vetoed ? " (struck)" : "");' in rs.INDEX_HTML
    assert 'if (head.trim()) { wrap.appendChild(el("div", {}, head.trim())); }' in rs.INDEX_HTML
    # one line per action label, one per unactionable request (both
    # already prefixed by record.action_label / reading_view)
    assert '(r.actions || []).forEach' in rs.INDEX_HTML
    assert '(r.unactionable || []).forEach' in rs.INDEX_HTML
    # the strike is a button, not a new key: Esc/Enter are untouched
    assert 'el("button", { "class": "strike" }, "strike this reading")' in rs.INDEX_HTML
    assert 'if (!r.vetoed) {' in rs.INDEX_HTML
    # the POST names the subject the comment sits under AND both halves
    # of the (comment_sha, prompt_version) reference -- half a reference
    # strikes nothing, and the server answers ok to a wrong subject
    assert 'postJson("/api/veto", { subject: subject, subject_kind: subjectKind,' \
        in rs.INDEX_HTML
    assert "comment_sha: n.comment_sha, prompt_version: r.prompt_version })" in rs.INDEX_HTML
    assert "if (result && result.ok) { reload(); return; }" in rs.INDEX_HTML
    # a strike that does not save says so beside the control (the note
    # box's own failure wording and styling) and strikes nothing
    assert 'var err = el("div", { "class": "save-error" });' in rs.INDEX_HTML
    assert rs.INDEX_HTML.count(
        'err.textContent = (result && result.error) || "not saved -- server unreachable";') == 2
    assert "#noteError, .save-error { color: #d9534f; font-size: 13px; }" in rs.INDEX_HTML


def test_every_comment_list_names_the_subject_the_strike_would_veto():
    """The strike's subject is the comment's own: a question's
    `q.subject` and a gallery card's ENTITY subject (`card.subject`),
    never the card anchor (`card.id`) -- the comment views are built
    from that subject's rows, and a veto written under any other subject
    strikes nothing while the server still answers ok.
    """
    assert ('renderComments(card.notes, box, card.subject, '
            'SUBJECT_KIND_OF_FAMILY[card.family], loadGallery)') in rs.INDEX_HTML
    assert "renderComments(card.notes, box, card.id" not in rs.INDEX_HTML
    # SUBJECT_KIND_OF_FAMILY's keys ARE compile.CARD_MEANINGS' families,
    # so the lookup cannot miss and carries no dead fallback branch
    from thai_syllabus.compile import CARD_MEANINGS
    assert {family for family, _kind in CARD_MEANINGS} == {
        "word", "sentence", "minimal_pair", "grapheme"}
    assert '|| "word"' not in rs.INDEX_HTML
    assert rs.INDEX_HTML.count(
        "renderComments(q.comments, box, q.subject, q.subject_kind, loadQueue)") == 4
    # no call site left on the r9 two-argument form
    assert "renderComments(card.notes, box)" not in rs.INDEX_HTML
    assert "renderComments(q.comments, box)" not in rs.INDEX_HTML


def test_the_struck_reading_and_the_strike_control_carry_their_own_css():
    assert ".reading.struck { text-decoration: line-through; opacity: 0.7; }" in rs.INDEX_HTML
    assert ".reading .action { padding-left: 12px; }" in rs.INDEX_HTML
    assert ".reading .unactionable { color: #c98a3d; }" in rs.INDEX_HTML
    assert "button.strike {" in rs.INDEX_HTML


def test_n_in_a_session_comments_on_the_current_questions_subject():
    """Design ruling 1: `n` is live in both modes -- in a session it
    posts the gallery comment's shape anchored on the subject under card
    kind "question", carrying the question kind, the artifact kind, the
    subject kind and what the question showed. The box keeps r5's
    failure semantics (stays open, says why, retries on Enter).
    """
    assert 'mode === "session" && queueItems.length' in rs.INDEX_HTML
    assert 'card_id: q.subject, kind: "question"' in rs.INDEX_HTML
    assert "subject_kind: q.subject_kind, question_kind: q.type," in rs.INDEX_HTML
    assert "artifact_kind: q.kind, shown: q.shown" in rs.INDEX_HTML
    assert "var SUBJECT_KIND_OF_FAMILY = {" in rs.INDEX_HTML
    assert 'not saved -- server unreachable' in rs.INDEX_HTML
    assert 'placeholder="comment (Enter to save, Esc to cancel)"' in rs.INDEX_HTML
    # the keystroke opening the box must not also type "n" into it
    assert 'if (e.key === "n") { e.preventDefault(); openNoteBox(); return; }' in rs.INDEX_HTML
    # the empty-artifact block says what n does now -- not the pre-r9
    # "gives the next search a direction"
    assert "n leaves a comment on this question (the machine reads it next run)" \
        in rs.INDEX_HTML
    assert "n gives the next search a direction" not in rs.INDEX_HTML


# --- _gloss_for: sentence gloss on a scene question (spec 5 section 1 kind 1) ---

def test_gloss_for_a_sentence_subject_is_the_sentences_own_gloss(syllabus):
    scene_word = word("scene", "ข้าวอร่อย", "the rice is delicious")  # the rice is delicious
    s = sentence(((scene_word.id,),), thai_of(scene_word), gloss="the rice is delicious")
    with_sentence = dataclasses.replace(syllabus, sentences=(s,))
    assert rs._gloss_for(with_sentence, s.text_sha, "sentence") == "the rice is delicious"


def test_gloss_for_a_pair_subject_joins_the_members_meanings(syllabus, pair, w1, w2):
    assert rs._gloss_for(syllabus, pair.id, "pair") == f"{w1.meaning} / {w2.meaning}"


def test_gloss_for_an_unknown_sentence_or_pair_is_none(syllabus):
    assert rs._gloss_for(syllabus, "no-such-sha", "sentence") is None
    assert rs._gloss_for(syllabus, "no-such-pair", "pair") is None


def test_build_queue_rate_item_lists_excluded_candidates_and_card_flags(derivations, db, w1):
    """spec 5 section 1 kind 1 / section 3: the rate question carries the
    candidates the judge could never even prepare (this run's own
    RunReport row) and the subject's card-level flags (anki_import.py's
    own card-flag row shape) -- never a live judge call, both read back.
    """
    _provide(db, w1.id, "picture", items=[{"sha": "sA"}])
    _judge(db, w1.id, "picture", "sA", True)
    missing_sha = "c" * 64
    db.append(port="run", backend="runreport", key=RunReportKey(), subject="run",
             question={"kind": "runreport"},
             answer={"excluded_items": [
                 {"subject": w1.id, "artifact_sha": missing_sha,
                  "reason": "artifact not found: " + missing_sha},
                 {"subject": "some-other-word", "artifact_sha": "x", "reason": "irrelevant"},
             ]})
    db.append(port="assess", backend="learner",
             key=FlagKey(family="word", anchor=w1.id, card_kind="reading", flags=1),
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


def _no_fit(db, word_id, *, targets=("t-rice",), reason="nothing fits"):
    """One no-fit outcome row the sentence attempt appends for a word
    (spec 3 r19 section 5): subject the WORD, Target ids in the question.
    """
    return db.append(port="attempt", backend="llm",
                     key=AttemptOutcomeKey(subject=word_id, kind="sentence", source="llm"),
                     subject=word_id,
                     question={"kind": "sentence", "source": "llm", "subject_kind": "word",
                              "targets": list(targets)},
                     answer={"outcome": "nothing", "candidates": [], "reason": reason})


def test_build_queue_asks_a_direction_for_a_word_at_the_sentence_no_fit_cap(derivations, db, w1):
    """Spec 3 r19 section 5: below the cap a word's unfilled Target is the
    drafter's to serve and the screen says nothing about it; at the cap
    the screen asks the learner where to go instead."""
    def sentence_directions():
        return [i for i in rs.build_queue(derivations, budget=50)
                if i["type"] == "direction" and i["kind"] == "sentence"
                and i["subject"] == w1.id]

    _no_fit(db, w1.id)
    _no_fit(db, w1.id)
    assert sentence_directions() == []
    _no_fit(db, w1.id)
    asked = sentence_directions()
    assert len(asked) == 1
    assert asked[0]["role"] == "sentence-for-target" and asked[0]["attempts"] == 3
    assert asked[0]["gloss"] == w1.meaning


def test_a_sentence_direction_question_carries_the_drafter_s_own_reason(derivations, db, w1):
    """Spec 5 section 1 kind 2: the learner is told why the drafter
    declined -- the newest no-fit row's reason (spec 3 r19 section 5)."""
    _no_fit(db, w1.id, reason="the vocabulary has no verb yet")
    _no_fit(db, w1.id, reason="still no verb to use it with")
    _no_fit(db, w1.id, reason="still no verb to use it with")
    asked = next(i for i in rs.build_queue(derivations, budget=50)
                 if i["type"] == "direction" and i["kind"] == "sentence"
                 and i["subject"] == w1.id)
    assert asked.get("reason") == "still no verb to use it with"


def test_a_sentence_direction_questions_tried_lists_the_words_own_nothing_reasons(
        derivations, db, w1):
    """Spec 5 section 1 kind 2: a sentence need's drafting ask lives under
    record.DRAFT_SUBJECT, not the word, so _tried_summary's source-ask rows
    are never here -- "tried" is instead the word's own `nothing` outcome
    rows' reasons, newest first, each named as the llm source."""
    _no_fit(db, w1.id, reason="the vocabulary has no verb yet")
    _no_fit(db, w1.id, reason="still no verb to use it with")
    _no_fit(db, w1.id, reason="tried again, still nothing")
    asked = next(i for i in rs.build_queue(derivations, budget=50)
                 if i["type"] == "direction" and i["kind"] == "sentence"
                 and i["subject"] == w1.id)
    assert asked["tried"] == [
        {"source": "llm", "reason": "tried again, still nothing"},
        {"source": "llm", "reason": "still no verb to use it with"},
        {"source": "llm", "reason": "the vocabulary has no verb yet"},
    ]


def test_a_picture_direction_question_carries_no_reason(derivations, db, w1):
    """A picture attempt's `nothing` outcome states none, so the field is
    None rather than absent -- the screen reads one shape."""
    for backend in ("openverse", "wikimedia", "pexels"):
        _provide(db, w1.id, "picture", backend=backend, items=[])
    asked = next(i for i in rs.build_queue(derivations, budget=50)
                 if i["type"] == "direction" and i["kind"] == "picture"
                 and i["subject"] == w1.id)
    assert "reason" in asked and asked["reason"] is None


def test_build_queue_stops_asking_a_direction_once_the_learner_gave_one(derivations, db, w1):
    for _ in range(3):
        _no_fit(db, w1.id)
    db.append(port="assess", backend="learner",
              key=DirectionKey(subject=w1.id, role="sentence-for-target",
                               text_sha=sha("use it with กิน")),   # กิน: eat
              subject=w1.id,
              question={"kind": "direction", "role": "sentence-for-target",
                       "subject_kind": "word"},
              answer={"direction": "use it with กิน"})
    items = rs.build_queue(derivations, budget=50)
    assert not [i for i in items if i["type"] == "direction" and i["kind"] == "sentence"
                and i["subject"] == w1.id]


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


def test_tried_summary_lists_no_entry_for_a_drafted_phrase_row(db):
    """spec 3 r24 section 5: attempts.phrase_attempt's per-subject phrase
    row (backend "llm") is an answer, not a Source ask -- _tried_summary
    (record.source_asks underneath) must not show it as a phantom
    {"source": "llm", "query": None} entry on a phrased picture need's
    "what was tried" list."""
    _provide(db, "rice", "picture", backend="openverse", query="rice food", items=[])
    db.append(port="provide", backend="llm", key=PhraseKey(subject="rice"), subject="rice",
             question={"provides": "phrase", "kind": "picture", "subject_kind": "word"},
             answer={"phrase": "a bowl of steamed rice"})
    tried = rs._tried_summary(record_mod.rows_for(db, "rice", "picture"))
    assert tried == [{"source": "openverse", "query": "rice food"}]


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


def test_build_queue_lists_a_need_kept_queued_for_an_awaiting_candidate_once(derivations, db, w1):
    """spec 5 section 1: a need kept queued because a candidate awaits a
    verdict (derivations.queue's bucket 2) can also be exhausted on
    attempts -- it must appear once, not once as a rate question and
    again as a direction question. A current_rubric entry for the
    picture role is needed for unjudged_candidates to see sA as
    awaiting -- an empty mapping (this file's `derivations` fixture)
    judges no role at all (derivations.unjudged_candidates).
    """
    rubric_derivations = dataclasses.replace(
        derivations, current_rubric={"picture-for-word": "rubric-v1"})
    _provide(db, w1.id, "picture", items=[{"sha": "sA"}])
    for src in sources_for("picture"):
        db.append(port="attempt", backend=src,
                  key=AttemptOutcomeKey(subject=w1.id, kind="picture", source=src),
                  subject=w1.id,
                  question={"kind": "picture", "source": src, "subject_kind": "word"},
                  answer={"outcome": "nothing", "candidates": []})
    items = rs.build_queue(rubric_derivations, budget=50)
    matches = [i for i in items if i["subject"] == w1.id and i["kind"] == "picture"]
    assert len(matches) == 1


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
    key = JudgeKey(rubric_sha=sha("r"), subject=w1.id, identity=preference_identity(["sA", "sB"]),
                   role="picture-preference")
    db.append(port="assess", backend="judge", key=key, subject=w1.id,
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
    _learner(db, pair.id, "rendition", "rend-sha", "acceptable")
    items = rs.build_queue(derivations, study=db, budget=50)
    reasks = [i for i in items if i["type"] == "reask" and i["subject"] == pair.id]
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


def test_a_rating_on_a_sentence_subject_carries_its_subject_kind(db):
    """A learner rating row must name the kind of thing its subject is,
    the same as every other row (record.py's own docstring) --
    record.subject_kind_of misfiles a scene-for-sentence rating as
    "word" without it.
    """
    rs.append_answer(db, {"subject": "sent-sha", "kind": "picture", "subject_kind": "sentence",
                          "action": 4, "artifact_sha": "sha1"})
    rows = db.assessments_of("sent-sha")
    assert record_mod.subject_kind_of(rows) == "sentence"
    assert record_mod.ratings_for_role(rows, "scene-for-sentence") != []


def test_a_supplied_recording_on_a_sentence_subject_carries_its_subject_kind(
        tmp_path, derivations, db):
    ctx = rs.ReviewContext(derivations=derivations)
    src = tmp_path / "clip.mp3"
    src.write_bytes(b"fake-mp3-bytes")
    rs.append_supply(ctx, {"subject": "sent-sha-2", "kind": "recording", "source": "path",
                          "value": str(src), "subject_kind": "sentence"})
    rows = db.assessments_of("sent-sha-2")
    assert record_mod.subject_kind_of(rows) == "sentence"
    assert record_mod.ratings_for_role(rows, "recording-for-sentence") != []


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
    rating_row = next(r for r in db.assessments_of(w1.id) if r.port == "assess")
    assert rating_row.answer["value"] == "unacceptable-use-this"
    assert rating_row.answer["provenance"]["source"] == "learner"
    assert rating_row.question["kind"] == "rating"  # record.learner_ratings reads this back
    # a local path also appends its own provide row (F1 defect 8), so
    # record.candidate_shas and derivations._anchor_ts see the artifact.
    provide_rows = [r for r in db.assessments_of(w1.id) if r.port == "provide"]
    assert len(provide_rows) == 1
    assert provide_rows[0].answer["items"] == [{"sha": sha, "ext": "png"}]


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


def _mechanical_pass(db, subject, sha):
    db.append(port="assess", backend="mechanical",
             key=MechanicalKey(check="recording-for-word", params="", subject=subject,
                               artifact_sha=sha),
             subject=subject,
             question={"role": "recording-for-word", "artifact_sha": sha, "rubric": None,
                      "kind": "recording"},
             answer={"value": True})


def test_supplied_recording_url_uses_audiofetch(derivations, db, media_store, w1):
    """spec 5 section 1 kind 2: a recording URL is fetched by kind, through
    audiofetch, never imgfetch (which refuses non-images). recording-for-word
    names no "learner" in AUTHORITY_ORDER (r8): the implicit "unacceptable-
    use-this" rating a supply writes is a candidate nomination, not a veto
    (only "unacceptable-none" vetoes) -- the sha is not excluded from
    machine ranking, but a mechanical verdict still decides current-best.
    """
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

    # not vetoed: still a candidate, awaiting a mechanical verdict.
    assert sha in record_mod.candidate_shas(record_mod.rows_for(db, w1.id, "recording"))
    assert ctx.current_best(w1.id, "recording").artifact_sha is None

    _mechanical_pass(db, w1.id, sha)
    best = ctx.current_best(w1.id, "recording")
    assert best.artifact_sha == sha
    assert best.source == "mechanical"


def test_supplied_recording_from_local_path_writes_the_real_ext_unnormalized(
        tmp_path, derivations, db, media_store, w1):
    """A local recording is a direct learner act -- MediaStore.write, not
    add_image: recordings are never normalized (spec 4 section 3
    normalizes pictures only). It still appends its own `provide` row
    (backend="learner"), matching what a URL supply gets through
    Provider.ask; record.candidate_shas and derivations._anchor_ts read
    the artifact through that row (F1 defect 8). recording-for-word names
    no "learner" in AUTHORITY_ORDER (r8): the supply's implicit
    "unacceptable-use-this" rating nominates the sha rather than vetoing
    it -- it is a candidate (visible via candidate_shas) awaiting a
    mechanical verdict, same as any other Source-provided artifact, and
    becomes current-best once one passes it.
    """
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

    assert sha in record_mod.candidate_shas(record_mod.rows_for(db, w1.id, "recording"))
    assert ctx.current_best(w1.id, "recording").artifact_sha is None

    _mechanical_pass(db, w1.id, sha)
    best = ctx.current_best(w1.id, "recording")
    assert best.artifact_sha == sha
    assert best.source == "mechanical"


def test_a_supplied_picture_from_local_path_also_appends_a_provide_row(
        tmp_path, derivations, db, media_store, w1):
    """The same fix (F1 defect 8), on the picture ingest path."""
    src = tmp_path / "candidate.jpg"
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="JPEG")
    src.write_bytes(buf.getvalue())
    ctx = rs.ReviewContext(derivations=derivations)
    out = rs.append_supply(ctx, {"subject": w1.id, "kind": "picture", "source": "path",
                                 "value": str(src)})
    assert out["ok"] is True
    sha = out["artifact_sha"]
    assert sha in record_mod.candidate_shas(record_mod.rows_for(db, w1.id, "picture"))


# --- gallery / notes / drills ------------------------------------------------

def test_reviewserver_has_no_second_card_kind_vocabulary():
    """reviewserver.py must derive a card's kind the same way
    anki_import.py does (compile.card_kind_of), not keep a second,
    family-prefixed vocabulary of its own.
    """
    from thai_syllabus import anki_import
    from thai_syllabus.compile import card_kind_of

    assert not hasattr(rs, "_CARD_KIND_NAMES")
    for name in ("Reading", "Listening", "Cloze"):
        assert card_kind_of(name) == name.lower()
    # anki_import._identify_card's own kind_slug now runs through the
    # same helper, proven directly here rather than through a full
    # grapheme/sentence card built with real media on disk.
    assert anki_import.card_kind_of is card_kind_of


def test_gallery_cards_come_from_the_compile(deck_with_history):
    """spec 5 section 1: the proof gallery renders every card the Compile
    would write. A card's kind is compile.card_kind_of(template_name) --
    the same bare, lowered template name anki_import.py's own card
    identity uses (no family prefix; every card carries its model's CSS,
    so the gallery shows what Anki shows (F4).
    """
    ctx = rs.load_context(deck_with_history)
    kinds = {c["kind"] for c in ctx.cards()}
    assert kinds <= {"listening", "production", "reading", "spelling", "recognition", "cloze"}
    assert all(c["css"] == CARD_CSS for c in ctx.cards())


def test_gallery_cards_render_front_and_back_html_in_introduction_order(derivations, db, w1):
    # pair has no seeded rendition and grapheme has no name_word (this
    # module's fixtures), so build_deck drops both (spec 4 section 1) --
    # this test measures the word family's own Reading card, which needs
    # neither.
    from thai_syllabus.compile import build_deck

    _judge(db, w1.id, "picture", "sha-w1", True)
    cards = rs.compiled_cards(derivations)
    built_deck = build_deck(derivations.syllabus, derivations.db, derivations.media_store,
                            current_rubric=derivations.current_rubric, prior=derivations.prior,
                            provenance_source=derivations.provenance_source)

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
    # "thai"/"picture" composed by the gallery itself -- "shown"/"notes"
    # are the exception (spec 5 section 1 r5): what this card displays
    # right now (for the client to echo back on /api/note, C1 fix) and
    # this card's own notes list.
    for card in cards:
        assert set(card) == {"index", "id", "family", "kind", "subject",
                             "front_html", "back_html", "css", "gloss",
                             "shown", "notes"}
        assert card["notes"] == []  # no notes on record yet
        assert set(card["shown"]) == {"picture", "recordings", "text_sha", "syllabus_state_id"}


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
    rendition_key = MechanicalKey(check="rendition", params="v1", subject=str(pair.id),
                                  artifact_sha=rendition_identity(shas))
    db.append(port="assess", backend="rendition", key=rendition_key, subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": rendition_identity(shas),
                      "rubric": None, "kind": "rendition", "subject_kind": "pair",
                      "params": {"members": shas}},
             answer={"value": True})

    syllabus = Syllabus(words=(w1, w2), pairs=(pair,), confusions=(confusion,),
                        media=_DbMediaIndex(db=db, pairs=(pair,)), assessments=db)
    derivations = Derivations(syllabus=syllabus, db=db, media_store=media_store,
                              current_rubric={}, prior=(), provenance_source=lambda sha: None,
                              sources_for=sources_for, attempt_cap=DEFAULT_ATTEMPT_CAP,
                              transient_cap=DEFAULT_TRANSIENT_CAP)

    pair_cards = [c for c in rs.compiled_cards(derivations) if c["family"] == "minimal_pair"]
    assert len(pair_cards) == 2
    for card in pair_cards:
        assert card["kind"] == "recognition"
        assert card["subject"] == pair.id
        assert card["confusion"] == confusion.id
    assert {c["stimulus_member"] for c in pair_cards} == {0, 1}


def test_compiled_cards_notes_are_scoped_by_anchor_not_just_subject_and_kind(
        db, media_store, w1, w2, pair, confusion):
    """spec 5 section 1 r5 (C2 fix): a minimal-pair note's two member
    cards share both a subject (the pair id) and a card_kind
    ("recognition") -- a note written against member 0's card must not
    also be listed under member 1's, and vice versa. record.card_notes
    is scoped by (anchor, card_kind); compiled_cards passes each card's
    own id (its anchor) through.
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
    rendition_key = MechanicalKey(check="rendition", params="v1", subject=str(pair.id),
                                  artifact_sha=rendition_identity(shas))
    db.append(port="assess", backend="rendition", key=rendition_key, subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": rendition_identity(shas),
                      "rubric": None, "kind": "rendition", "subject_kind": "pair",
                      "params": {"members": shas}},
             answer={"value": True})

    syllabus = Syllabus(words=(w1, w2), pairs=(pair,), confusions=(confusion,),
                        media=_DbMediaIndex(db=db, pairs=(pair,)), assessments=db)
    pair_derivations = Derivations(syllabus=syllabus, db=db, media_store=media_store,
                                   current_rubric={}, prior=(), provenance_source=lambda sha: None,
                                   sources_for=sources_for, attempt_cap=DEFAULT_ATTEMPT_CAP,
                                   transient_cap=DEFAULT_TRANSIENT_CAP)

    pair_cards = [c for c in rs.compiled_cards(pair_derivations) if c["family"] == "minimal_pair"]
    assert len(pair_cards) == 2
    member0 = next(c for c in pair_cards if c["stimulus_member"] == 0)
    member1 = next(c for c in pair_cards if c["stimulus_member"] == 1)
    assert member0["id"] != member1["id"]  # distinct anchors, same subject and kind

    rs.append_gallery_note(db, subject=pair.id, card_id=member0["id"], kind="recognition",
                           text="static on this one", shown=member0["shown"])

    pair_cards = [c for c in rs.compiled_cards(pair_derivations) if c["family"] == "minimal_pair"]
    member0 = next(c for c in pair_cards if c["stimulus_member"] == 0)
    member1 = next(c for c in pair_cards if c["stimulus_member"] == 1)
    assert [n["text"] for n in member0["notes"]] == ["static on this one"]
    assert member1["notes"] == []  # never bleeds into the other member's card


def test_compiled_cards_pair_card_names_both_recordings_and_stales_on_the_second(
        db, media_store, w1, w2, pair, confusion):
    """spec 5 section 1 r5 (minor fix): a minimal-pair Recognition card
    plays two recordings -- this member's own, on `{{Audio}}` (front),
    and the other member's, on `{{OtherAudio}}` (back). `shown` must
    name both, and a note must go stale when EITHER changes, not just
    the first (the bug: only the first `<audio>` match was kept, so
    the second member's recording changing never staled a note).
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

    def _write_rendition(members_shas):
        key = MechanicalKey(check="rendition", params="v1", subject=str(pair.id),
                            artifact_sha=rendition_identity(members_shas))
        db.append(port="assess", backend="rendition", key=key, subject=pair.id,
                 question={"role": "rendition-for-pair",
                          "artifact_sha": rendition_identity(members_shas),
                          "rubric": None, "kind": "rendition", "subject_kind": "pair",
                          "params": {"members": members_shas}},
                 answer={"value": True})

    _write_rendition(shas)

    syllabus = Syllabus(words=(w1, w2), pairs=(pair,), confusions=(confusion,),
                        media=_DbMediaIndex(db=db, pairs=(pair,)), assessments=db)
    pair_derivations = Derivations(syllabus=syllabus, db=db, media_store=media_store,
                                   current_rubric={}, prior=(), provenance_source=lambda sha: None,
                                   sources_for=sources_for, attempt_cap=DEFAULT_ATTEMPT_CAP,
                                   transient_cap=DEFAULT_TRANSIENT_CAP)

    pair_cards = [c for c in rs.compiled_cards(pair_derivations) if c["family"] == "minimal_pair"]
    member0 = next(c for c in pair_cards if c["stimulus_member"] == 0)
    assert member0["shown"]["recordings"] == [shas[w1.id], shas[w2.id]]

    rs.append_gallery_note(db, subject=pair.id, card_id=member0["id"], kind="recognition",
                           text="clear both voices", shown=member0["shown"])

    # only the SECOND member's recording changes -- member 0's own sha
    # (the first, on Audio) is untouched.
    new_other_sha = media_store.write(f"rendition:{pair.id}:{w2.id}:v2".encode(), ext="mp3")
    db.add_media(sha=new_other_sha, kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id=speaker)
    _write_rendition({w1.id: shas[w1.id], w2.id: new_other_sha})

    pair_cards = [c for c in rs.compiled_cards(pair_derivations) if c["family"] == "minimal_pair"]
    member0 = next(c for c in pair_cards if c["stimulus_member"] == 0)
    assert member0["shown"]["recordings"] == [shas[w1.id], new_other_sha]
    assert member0["notes"][0]["stale"] is True


def _seed_picture(db, media_store, subject, payload=b"pic"):
    sha = media_store.write(payload, ext="jpg")
    db.add_media(sha=sha, kind="picture", ext="jpg", source="openverse",
                origin="https://example.com/x.jpg", licence="cc0", acquired=date(2026, 1, 1))
    return sha


def test_compiled_cards_lists_a_cards_notes_oldest_first_not_stale_while_matching(
        derivations, db, media_store, w1):
    """spec 5 section 1 r5 (b): the gallery card payload lists the
    card's existing notes, oldest first, each with its text and time --
    not stale while its own `shown` still matches what the card shows.
    """
    sha = _seed_picture(db, media_store, w1.id)
    _judge(db, w1.id, "picture", sha, True)
    reading_card = next(c for c in rs.compiled_cards(derivations)
                        if c["kind"] == "reading" and c["id"] == w1.id)
    shown = reading_card["shown"]  # the exact value the client would echo back
    assert shown["picture"] == sha

    ts1 = rs.append_gallery_note(db, subject=w1.id, card_id=reading_card["id"], kind="reading",
                                 text="clear picture", shown=shown)
    ts2 = rs.append_gallery_note(db, subject=w1.id, card_id=reading_card["id"], kind="reading",
                                 text="still good", shown=shown)

    reading_card = next(c for c in rs.compiled_cards(derivations)
                        if c["kind"] == "reading" and c["id"] == w1.id)
    identities = {r.ts: comment_identity(r.key_sha, r.ts) for r in db.assessments_of(w1.id)}
    assert reading_card["notes"] == [
        {"text": "clear picture", "ts": ts1, "stale": False, "comment_sha": identities[ts1],
         "reading": None},
        {"text": "still good", "ts": ts2, "stale": False, "comment_sha": identities[ts2],
         "reading": None},
    ]


def test_compiled_cards_marks_a_note_stale_once_a_different_picture_becomes_current_best(
        derivations, db, media_store, w1):
    """spec 5 section 1 r5: the note reads stale once the card no
    longer shows what it named (F9) -- here, a learner "good" rating on
    a second picture displaces the judged one the note was written
    against.
    """
    sha_a = _seed_picture(db, media_store, w1.id, b"picture-a")
    _judge(db, w1.id, "picture", sha_a, True)
    reading_card = next(c for c in rs.compiled_cards(derivations)
                        if c["kind"] == "reading" and c["id"] == w1.id)
    shown = reading_card["shown"]
    assert shown["picture"] == sha_a
    rs.append_gallery_note(db, subject=w1.id, card_id=reading_card["id"], kind="reading",
                           text="looks right", shown=shown)

    sha_b = _seed_picture(db, media_store, w1.id, b"picture-b")
    _learner(db, w1.id, "picture", sha_b, "good")

    reading_card = next(c for c in rs.compiled_cards(derivations)
                        if c["kind"] == "reading" and c["id"] == w1.id)
    assert reading_card["shown"]["picture"] == sha_b
    assert reading_card["notes"][0]["stale"] is True


def test_compiled_cards_note_with_a_legacy_singular_recording_is_not_stale_when_unchanged(
        derivations, db, media_store, w1):
    """fix round 4, end to end through compiled_cards: a note whose row
    carries the earlier singular `recording` key (round 1's own shape,
    before round 3 switched to a `recordings` list) reads NOT stale
    against a card still showing that same recording under the current
    shape -- card_notes' own normalize_shown makes the comparison shape-
    blind.
    """
    sha = media_store.write(b"w1-listening", ext="mp3")
    db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by", acquired=date(2026, 1, 1))
    _judge(db, w1.id, "recording", sha, True)
    listening_card = next(c for c in rs.compiled_cards(derivations)
                          if c["kind"] == "listening" and c["id"] == w1.id)
    assert listening_card["shown"]["recordings"] == [sha]

    # append_gallery_note stores whatever `shown` mapping it's given
    # verbatim -- this simulates a row an earlier server version wrote.
    rs.append_gallery_note(db, subject=w1.id, card_id=listening_card["id"], kind="listening",
                           text="clear audio", shown={"picture": None, "recording": sha,
                                                       "text_sha": None})

    listening_card = next(c for c in rs.compiled_cards(derivations)
                          if c["kind"] == "listening" and c["id"] == w1.id)
    assert listening_card["notes"][0]["stale"] is False


def test_resolve_media_for_web_rewrites_img_and_sound_to_the_media_route():
    html = ('<img src="deadbeef.jpg">'
           '<div>[sound:cafef00d.mp3]</div>')
    resolved = rs._resolve_media_for_web(html)
    assert '<img src="/media/deadbeef">' in resolved
    assert '<audio controls src="/media/cafef00d"></audio>' in resolved
    assert ".jpg" not in resolved
    assert ".mp3" not in resolved


def test_shown_of_reads_picture_and_recording_shas_off_the_resolved_html():
    """spec 5 section 1 r5: `shown` is read off the card's own resolved
    front/back media refs -- the same bytes the learner saw -- never
    re-derived from current_best (a note names what rendered, not what
    ranks highest right now).
    """
    entry = {"front_html": '<div class="thai">rice</div>',
             "back_html": ('<img src="/media/pic123">'
                          '<audio controls src="/media/rec456"></audio>'),
             "family": "word", "subject": "rice"}
    assert rs._shown_of(entry) == {"picture": "pic123", "recordings": ["rec456"], "text_sha": None}


def test_shown_of_collects_every_recording_in_document_order():
    """spec 5 section 1 r5 (minor fix): a minimal_pair Recognition card
    plays two recordings -- its own member's on `{{Audio}}` (front) and
    the other member's on `{{OtherAudio}}` (back) -- both must be named,
    front to back, not just the first found.
    """
    entry = {"front_html": '<audio controls src="/media/own-rec"></audio>',
             "back_html": '<audio controls src="/media/other-rec"></audio>',
             "family": "minimal_pair", "subject": "pair-id"}
    assert rs._shown_of(entry)["recordings"] == ["own-rec", "other-rec"]


def test_shown_of_has_no_artifacts_when_the_card_shows_neither():
    entry = {"front_html": "<div>x</div>", "back_html": "<div>y</div>",
             "family": "grapheme", "subject": "k"}
    assert rs._shown_of(entry) == {"picture": None, "recordings": [], "text_sha": None}


def test_shown_of_carries_text_sha_for_a_sentence_card():
    """A sentence family's entry["subject"] IS its text_sha already
    (compile.py tags the sentence family by text_sha) -- carried straight
    through, no recomputation.
    """
    entry = {"front_html": "", "back_html": "", "family": "sentence", "subject": "sentence-sha"}
    assert rs._shown_of(entry)["text_sha"] == "sentence-sha"


def test_is_stale_false_when_the_note_named_nothing():
    # a pre-r5 row (card_notes' own {} default) made no claim -- nothing
    # to check it against, so it never goes stale on this account.
    assert rs._is_stale({}, {"picture": "p1", "recordings": [], "text_sha": None}) is False


def test_is_stale_true_once_a_different_picture_is_shown():
    recorded = {"picture": "p1", "recordings": [], "text_sha": None}
    current = {"picture": "p2", "recordings": [], "text_sha": None}
    assert rs._is_stale(recorded, current) is True


def test_is_stale_false_while_the_shown_artifacts_still_match():
    shown = {"picture": "p1", "recordings": ["r1", "r2"], "text_sha": None}
    assert rs._is_stale(shown, dict(shown)) is False


def test_is_stale_true_once_only_the_second_recording_changes():
    """spec 5 section 1 r5 (minor fix): the first recording is
    unchanged -- only the second (a minimal_pair card's OtherAudio)
    differs -- and that alone must still mark the note stale.
    """
    recorded = {"picture": None, "recordings": ["own-rec", "other-rec"], "text_sha": None}
    current = {"picture": None, "recordings": ["own-rec", "a-new-other-rec"], "text_sha": None}
    assert rs._is_stale(recorded, current) is True


def test_is_stale_treats_a_normalized_legacy_recording_as_unchanged():
    """fix round 4, the reviewer's exact reproduction: a legacy `shown`
    naming a picture and a singular `recording` must not read stale
    against a current `shown` naming the same picture and the same
    recording under the current `recordings`-list shape -- once
    record.normalize_shown has run (card_notes' own contract; _is_stale
    itself only ever compares already-normalized shapes).
    """
    recorded = record_mod.normalize_shown({"picture": "p", "recording": "r"})
    current = {"picture": "p", "recordings": ["r"], "text_sha": None}
    assert rs._is_stale(recorded, current) is False


def test_is_stale_true_when_a_legacy_recording_actually_changed():
    recorded = record_mod.normalize_shown({"picture": "p", "recording": "r"})
    current = {"picture": "p", "recordings": ["r2"], "text_sha": None}
    assert rs._is_stale(recorded, current) is True


def test_validated_shown_accepts_a_mapping_of_hex_shas_and_nones():
    shown = {"picture": "a" * 64, "recordings": [], "text_sha": None,
            "syllabus_state_id": "b" * 64}
    assert rs._validated_shown(shown) == shown


def test_validated_shown_accepts_a_list_of_hex_shas():
    shown = {"recordings": ["a" * 64, "b" * 64]}
    assert rs._validated_shown(shown) == shown


def test_validated_shown_is_empty_when_absent():
    assert rs._validated_shown(None) == {}


def test_validated_shown_raises_on_a_non_hex_string():
    with pytest.raises(ValueError):
        rs._validated_shown({"picture": "not-a-64-hex-sha"})


def test_validated_shown_raises_on_a_list_containing_a_non_hex_value():
    with pytest.raises(ValueError):
        rs._validated_shown({"recordings": ["a" * 64, "not-a-64-hex-sha"]})


def test_validated_shown_raises_on_a_nested_value():
    with pytest.raises(ValueError):
        rs._validated_shown({"picture": {"nested": "object"}})


def test_validated_shown_raises_on_a_non_mapping():
    with pytest.raises(ValueError):
        rs._validated_shown(["not", "a", "mapping"])


def test_append_gallery_note_appends_learner_row_not_a_file(db):
    ts = rs.append_gallery_note(db, subject="t-rice", card_id="t-rice", kind="target",
                                text="lovely bowl of rice")
    assert isinstance(ts, int)
    rows = db.assessments_of("t-rice")
    assert len(rows) == 1
    assert rows[0].answer == {"kind": "rating", "rating": None, "note": "lovely bowl of rice"}
    assert rows[0].key == "learner:t-rice:card-flag"
    assert rows[0].question["kind"] == "card-flag"  # derivations.directed()'s own test
    assert rows[0].question["shown"] == {}  # no `shown` passed -- names nothing (spec 5 r5)


def test_append_gallery_note_question_carries_shown(db):
    """spec 5 section 1 r5: a gallery note records the card as shown --
    the artifact shas it displayed, the sentence text_sha for a sentence
    card, and the syllabus state id -- in its own question, never the
    answer (the answer stays the plain {kind, rating, note} shape every
    other reader of a card-flag row already expects).
    """
    shown = {"picture": "sha-w1", "recordings": [], "text_sha": None,
            "syllabus_state_id": "state-1"}
    rs.append_gallery_note(db, subject="rice", card_id="rice", kind="reading",
                           text="nice picture", shown=shown)
    rows = db.assessments_of("rice")
    assert rows[0].question["shown"] == shown
    assert rows[0].answer == {"kind": "rating", "rating": None, "note": "nice picture"}


def test_a_gallery_note_on_a_pair_member_card_directs_the_pair_not_the_member(db, pair):
    """A minimal-pair card's own MemberKey (e.g. "pair-id:speaker:0") is
    the row's per-card anchor, never the entity subject
    derivations.directed() and record.card_flags() key on -- that is the
    pair's own id (Built.subject vs. card.subject in reviewserver.py's
    own gallery data, spec 4 section 4).
    """
    member_key = f"{pair.id}:speaker-x:0"
    rs.append_gallery_note(db, subject=pair.id, card_id=member_key, kind="recognition",
                           text="static on this recording")
    assert directed(db, pair.id)
    assert record_mod.card_flags(db.assessments_of(pair.id)) == [f"{member_key}::recognition"]
    assert db.assessments_of(member_key) == []  # never filed under the per-card anchor


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


def test_compute_stats_accepted_counts_a_mechanically_passing_veto_only_need(
        derivations, db, w1):
    """r8 fix round 2, ruling 4: a mechanically-passing recording (rank
    50, source "mechanical") is accepted -- the learner-ranking
    "acceptable" (rank 80) floor has no meaning on a veto-only role's
    machine scale.
    """
    _provide(db, w1.id, "recording", backend="forvo", items=[{"sha": "s1"}])
    _mechanical_pass(db, w1.id, "s1")
    stats = rs.compute_stats(derivations)
    assert stats["coverage"]["recording"]["covered"] == 1
    assert stats["coverage"]["recording"]["accepted"] == 1


def test_compute_stats_accepted_counts_a_learner_nominated_passing_recording(
        derivations, db, w1):
    """r8 fix round 3, ruling 5: an "unacceptable-use-this" nomination on
    the mechanically-passing current sha is not a rejection -- accepted.
    """
    _provide(db, w1.id, "recording", backend="forvo", items=[{"sha": "s1"}])
    _mechanical_pass(db, w1.id, "s1")
    _learner(db, w1.id, "recording", "s1", "unacceptable-use-this")
    stats = rs.compute_stats(derivations)
    assert stats["coverage"]["recording"]["accepted"] == 1


def test_compute_stats_accepted_still_gates_a_learner_ranking_role_at_80(derivations, db, w1):
    """Unchanged: on a learner-ranking role (AUTHORITY_ORDER names
    "learner") a passing-only machine verdict does not reach "accepted"
    without a learner rating of "acceptable" or better.
    """
    _provide(db, w1.id, "picture", items=[{"sha": "sA"}])
    _judge(db, w1.id, "picture", "sA", True)
    stats = rs.compute_stats(derivations)
    assert stats["coverage"]["picture"]["covered"] == 1
    assert stats["coverage"]["picture"]["accepted"] == 0

    _learner(db, w1.id, "picture", "sA", "acceptable")
    stats = rs.compute_stats(derivations)
    assert stats["coverage"]["picture"]["accepted"] == 1


def test_compute_stats_reads_pending_and_sentences_adopted_from_the_newest_runreport(
        derivations, syllabus, db):
    # run.py's _persist_report convention: port="run", backend="runreport",
    # key=RunReportKey(), subject="run" -- one row per run() call, newest wins.
    db.append(port="run", backend="runreport", key=RunReportKey(), subject="run",
             question={}, answer={"attempted": 1, "improved": 0, "exhausted": 0,
                                  "available": 2, "pending": 3, "sentences_adopted": 4},
             cost=0.0)
    stats = rs.compute_stats(derivations)
    assert stats["pending"] == 3
    assert stats["sentences_adopted"] == 4


@pytest.fixture
def ctx_two_words_one_pictured(derivations, db, w1):
    """Two targeted words (the `syllabus` fixture's w1/w2), one of which
    (w1) has a learner-rated picture; w2's picture need has no artifact
    at all -- the coverage/total disagreement the old available_needs-based
    fold produced (spec 5 section 3).
    """
    _learner(db, w1.id, "picture", "s1", "good")
    return derivations


def _run_report_answer(**overrides):
    """The fields run._persist_report's answer carried before spec 3 r29
    (spec 3 section 7), defaulted so a test only names the fields it
    cares about -- an older row, the shape a deck's early runs left
    behind, is exactly this with nothing overridden.
    """
    answer = {"attempted": 0, "improved": 0, "exhausted": 0, "available": 0,
             "pending": 0, "sentences_adopted": 0, "drafted": 0,
             "excluded": 0, "excluded_items": [], "unreachable": False,
             "batch_id": None, "source_failures": {}, "spend": {},
             "unserved": 0, "budgeted": 0, "deferred": 0, "preferences": 0}
    answer.update(overrides)
    return answer


@pytest.fixture
def ctx_with_two_runs(derivations, db):
    db.append(port="run", backend="runreport", key=RunReportKey(), subject="run",
             question={"kind": "runreport"},
             answer=_run_report_answer(attempted=1, excluded=1, unreachable=False),
             cost=0.0)
    db.append(port="run", backend="runreport", key=RunReportKey(), subject="run",
             question={"kind": "runreport"},
             answer=_run_report_answer(attempted=2, excluded=0, unreachable=True),
             cost=0.0)
    return derivations


def test_stats_cover_every_need(ctx_two_words_one_pictured):
    s = rs.compute_stats(ctx_two_words_one_pictured)
    assert s["coverage"]["picture"] == {"total": 2, "covered": 1, "accepted": 1}


def test_stats_list_every_run_with_excluded_and_unreachable(ctx_with_two_runs):
    hist = rs.compute_stats(ctx_with_two_runs)["run_report_history"]
    assert len(hist) == 2 and {"excluded", "unreachable"} <= hist[0].keys()


@pytest.fixture
def ctx_with_an_old_and_a_new_run(derivations, db):
    """An older runreport row from before spec 3 r29 (neither
    `adjudicated` nor `stayed_disputed` on it) and a newer one carrying
    both -- the mix a deck that has been running since before r29 holds.
    """
    db.append(port="run", backend="runreport", key=RunReportKey(), subject="run",
             question={"kind": "runreport"}, answer=_run_report_answer(), cost=0.0)
    db.append(port="run", backend="runreport", key=RunReportKey(), subject="run",
             question={"kind": "runreport"},
             answer=_run_report_answer(adjudicated=3, stayed_disputed=5, comments_read=2,
                                       comment_actions=1, comment_unactionable=1), cost=0.0)
    return derivations


def test_stats_history_carries_adjudicated_and_stayed_disputed_per_run(
        ctx_with_an_old_and_a_new_run):
    """Spec 3 r29: both counts are per-run columns of the run history,
    and a row written before r29 reads as 0 rather than dropping the
    column for every run (the page takes its columns from the oldest
    row).
    """
    hist = rs.compute_stats(ctx_with_an_old_and_a_new_run)["run_report_history"]
    assert [(r["adjudicated"], r["stayed_disputed"]) for r in hist] == [(0, 0), (3, 5)]


def test_stats_history_carries_the_comment_counts_on_every_run(ctx_with_an_old_and_a_new_run):
    """Spec 3 r30: the comment pass's three counts are per-run columns of
    the history too, 0 on a row written before them."""
    hist = rs.compute_stats(ctx_with_an_old_and_a_new_run)["run_report_history"]
    assert [(r["comments_read"], r["comment_actions"], r["comment_unactionable"]) for r in hist] \
        == [(0, 0, 0), (2, 1, 1)]


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
    port, db_path = live_server
    # w1's needs otherwise have no candidate at all, and so no rate
    # question (spec 5 r7 F4) -- seed one so the queue names w1 (the
    # server thread owns the db connection; write through a second one
    # to the same WAL file, as test_http_answer_post_appends_row does in
    # reverse).
    seed_db = SyllabusDb(db_path)
    _provide(seed_db, w1.id, "picture", items=[{"sha": "p1"}])
    seed_db.close()
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


def test_http_api_cards_lists_shown_per_card(live_server, media_store, w1):
    """spec 5 section 1 r5: the gallery card payload carries `shown` --
    this exact card's own artifacts/state -- for the client to echo
    back verbatim on /api/note.
    """
    port, db_path = live_server
    verify_db = SyllabusDb(db_path)
    sha = media_store.write(b"w1-picture", ext="jpg")
    verify_db.add_media(sha=sha, kind="picture", ext="jpg", source="openverse",
                        origin="https://example.com/x.jpg", licence="cc0",
                        acquired=date(2026, 1, 1))
    _judge(verify_db, w1.id, "picture", sha, True)

    _status, body = _get(port, "/api/cards")
    reading_card = next(c for c in json.loads(body)
                        if c["kind"] == "reading" and c["id"] == w1.id)
    assert reading_card["shown"]["picture"] == sha
    assert isinstance(reading_card["shown"]["syllabus_state_id"], str)
    assert reading_card["shown"]["syllabus_state_id"]


def test_http_note_records_the_posted_shown_verbatim_and_lists_it_not_stale(
        live_server, media_store, w1):
    """spec 5 section 1 r5 end to end through the handler: /api/note
    records the client's own posted `shown` (the C1 fix -- never a
    server-side recompute), then /api/cards lists the note back, not
    stale, since nothing has changed.
    """
    port, db_path = live_server
    verify_db = SyllabusDb(db_path)
    sha = media_store.write(b"w1-picture", ext="jpg")
    verify_db.add_media(sha=sha, kind="picture", ext="jpg", source="openverse",
                        origin="https://example.com/x.jpg", licence="cc0",
                        acquired=date(2026, 1, 1))
    _judge(verify_db, w1.id, "picture", sha, True)

    _status, body = _get(port, "/api/cards")
    reading_card = next(c for c in json.loads(body)
                        if c["kind"] == "reading" and c["id"] == w1.id)

    status, body = _post(port, "/api/note",
                         {"subject": reading_card["subject"], "card_id": reading_card["id"],
                          "kind": "reading", "text": "clear picture",
                          "shown": reading_card["shown"]})
    assert status == 200
    assert json.loads(body)["ok"] is True

    saved = verify_db.assessments_of(w1.id)[-1]
    assert saved.question["shown"] == reading_card["shown"]

    _status, body = _get(port, "/api/cards")
    reading_card = next(c for c in json.loads(body)
                        if c["kind"] == "reading" and c["id"] == w1.id)
    assert reading_card["notes"] == [
        {"text": "clear picture", "ts": saved.ts, "stale": False,
         "comment_sha": comment_identity(saved.key_sha, saved.ts), "reading": None}]


def test_http_note_records_the_shown_the_page_rendered_not_a_later_current_best(
        live_server, media_store, w1):
    """C1 regression: reproduced as picture A shown, B rated good before
    the save, and a server-side recompute recording B -- /api/note must
    record the `shown` the client echoes back from the page it actually
    rendered, never recompute against current_best at save time.
    """
    port, db_path = live_server
    verify_db = SyllabusDb(db_path)
    sha_a = media_store.write(b"picture-a", ext="jpg")
    verify_db.add_media(sha=sha_a, kind="picture", ext="jpg", source="openverse",
                        origin="https://example.com/a.jpg", licence="cc0",
                        acquired=date(2026, 1, 1))
    _judge(verify_db, w1.id, "picture", sha_a, True)

    _status, body = _get(port, "/api/cards")
    reading_card = next(c for c in json.loads(body)
                        if c["kind"] == "reading" and c["id"] == w1.id)
    assert reading_card["shown"]["picture"] == sha_a
    rendered_shown = reading_card["shown"]

    # a second picture becomes current-best BEFORE the note is saved --
    # the page the learner is looking at still shows sha_a.
    sha_b = media_store.write(b"picture-b", ext="jpg")
    verify_db.add_media(sha=sha_b, kind="picture", ext="jpg", source="openverse",
                        origin="https://example.com/b.jpg", licence="cc0",
                        acquired=date(2026, 1, 1))
    _learner(verify_db, w1.id, "picture", sha_b, "good")

    status, body = _post(port, "/api/note",
                         {"subject": reading_card["subject"], "card_id": reading_card["id"],
                          "kind": "reading", "text": "clear picture", "shown": rendered_shown})
    assert status == 200
    saved = verify_db.assessments_of(w1.id)[-1]
    assert saved.question["shown"]["picture"] == sha_a  # what the page showed, not sha_b


def test_http_note_refuses_a_shown_value_that_is_not_a_hex_sha(live_server, w1):
    """fix round 2 ruling: a *present* but malformed `shown` is a bad
    request -- refused with 400 and no row appended -- never silently
    stored or emptied (round 1's leniency let a garbage claim through
    while telling the client it had saved).
    """
    port, db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "reading",
                          "text": "hmm", "shown": {"picture": "not-a-64-hex-sha"}})
    assert status == 400
    assert json.loads(body)["ok"] is False
    assert json.loads(body)["error"]
    verify_db = SyllabusDb(db_path)
    assert verify_db.assessments_of(w1.id) == []  # no row appended


def test_http_note_refuses_a_nested_object_in_shown(live_server, w1):
    port, db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "reading",
                          "text": "hmm", "shown": {"picture": {"nested": "object"}}})
    assert status == 400
    assert json.loads(body)["ok"] is False
    verify_db = SyllabusDb(db_path)
    assert verify_db.assessments_of(w1.id) == []  # no row appended


def test_http_note_records_empty_shown_when_the_field_is_absent(live_server, w1):
    """The one lenient case: no `shown` in the POST body at all (an old
    client, or a card with no media) -- {}, a note naming nothing, not
    a 400.
    """
    port, db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "reading", "text": "hmm"})
    assert status == 200
    assert json.loads(body)["ok"] is True
    verify_db = SyllabusDb(db_path)
    saved = verify_db.assessments_of(w1.id)[-1]
    assert saved.question["shown"] == {}


def test_http_note_records_a_valid_shown_mapping_verbatim(live_server, w1):
    port, db_path = live_server
    sha_a, sha_b, sha_c = "a" * 64, "b" * 64, "c" * 64
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "reading", "text": "hmm",
                          "shown": {"picture": sha_a, "recordings": [sha_b, sha_c],
                                   "text_sha": None, "syllabus_state_id": "d" * 64}})
    assert status == 200
    assert json.loads(body)["ok"] is True
    verify_db = SyllabusDb(db_path)
    saved = verify_db.assessments_of(w1.id)[-1]
    assert saved.question["shown"] == {"picture": sha_a, "recordings": [sha_b, sha_c],
                                       "text_sha": None, "syllabus_state_id": "d" * 64}


def test_http_note_on_a_question_records_the_question_kind_and_subject_kind(live_server, w1):
    """Spec 5 r9: `n` in a session posts the question's own facts; the
    row is the card-flag shape anchored on the subject."""
    port, db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "question",
                          "subject_kind": "word", "question_kind": "rate",
                          "artifact_kind": "picture", "text": "not rice",
                          "shown": {"picture": None, "recordings": [], "text_sha": None}})
    assert status == 200 and json.loads(body)["ok"] is True
    saved = SyllabusDb(db_path).assessments_of(w1.id)[-1]
    assert saved.question["card_kind"] == "question"
    assert saved.question["question_kind"] == "rate"
    assert saved.question["artifact_kind"] == "picture"
    assert saved.question["subject_kind"] == "word"


def test_http_note_refuses_a_question_kind_the_queue_never_emits(live_server, w1):
    """A present `question_kind` must be one of the four kinds the queue
    emits -- anything else is a bad request (400, no row appended), the
    same treatment a malformed `shown` gets: a comment whose recorded
    question kind is garbage can never be read back as evidence.
    """
    port, db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "question",
                          "question_kind": "ponder", "text": "hmm"})
    assert status == 400
    assert json.loads(body)["ok"] is False
    assert "question_kind" in json.loads(body)["error"]
    assert SyllabusDb(db_path).assessments_of(w1.id) == []  # no row appended


def test_http_note_refuses_an_artifact_kind_the_queue_never_emits(live_server, w1):
    """Likewise `artifact_kind`: picture, recording or rendition, the
    kinds a question actually carries.
    """
    port, db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "question",
                          "question_kind": "rate", "artifact_kind": "diagram", "text": "hmm"})
    assert status == 400
    assert json.loads(body)["ok"] is False
    assert "artifact_kind" in json.loads(body)["error"]
    assert SyllabusDb(db_path).assessments_of(w1.id) == []  # no row appended


def test_http_note_refuses_a_subject_kind_outside_the_vocabulary(live_server, w1):
    """`subject_kind` is validated the way /api/veto validates it: the
    comment pass resolves the kind off the syllabus and refuses a comment
    whose recorded kind disagrees (attempts._handable), so a kind that is
    no kind at all would close the comment unread -- a bad request, not a
    row. An absent one stays absent (a gallery comment names none).
    """
    port, db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "reading",
                          "subject_kind": "letter", "text": "hmm"})
    assert status == 400
    assert json.loads(body)["ok"] is False
    assert "subject_kind" in json.loads(body)["error"]
    assert SyllabusDb(db_path).assessments_of(w1.id) == []  # no row appended
    status, _body = _post(port, "/api/note",
                          {"subject": w1.id, "card_id": w1.id, "kind": "reading", "text": "hm"})
    assert status == 200
    assert "subject_kind" not in SyllabusDb(db_path).assessments_of(w1.id)[-1].question


def test_http_note_accepts_a_rendition_question_and_an_absent_kind(live_server, w1):
    """The kinds the queue does emit pass (rendition among them), and an
    absent question/artifact kind stays the gallery comment's own shape.
    """
    port, _db_path = live_server
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "question",
                          "question_kind": "challenger", "artifact_kind": "rendition",
                          "text": "the pair reads flat"})
    assert status == 200 and json.loads(body)["ok"] is True
    status, body = _post(port, "/api/note",
                         {"subject": w1.id, "card_id": w1.id, "kind": "reading", "text": "fine"})
    assert status == 200 and json.loads(body)["ok"] is True


def test_http_veto_writes_the_veto_row_and_the_reading_reads_vetoed(live_server, w1):
    """Spec 5 r10: the strike is one POST naming the reading; the reading
    stays on record, read back as vetoed."""
    port, db_path = live_server
    verify_db = SyllabusDb(db_path)
    rs.append_gallery_note(verify_db, subject=w1.id, card_id=w1.id, kind="reading", text="busy",
                           shown={}, subject_kind="word")
    (c,) = record_mod.comments(verify_db)
    _reading_row(verify_db, w1.id, c.comment_sha)
    assert record_mod.reading_view(verify_db.assessments_of(w1.id),
                                   c.comment_sha)["vetoed"] is False
    status, body = _post(port, "/api/veto", {"subject": w1.id, "comment_sha": c.comment_sha,
                                             "prompt_version": "1", "subject_kind": "word"})
    assert status == 200 and json.loads(body)["ok"] is True
    assert record_mod.reading_view(verify_db.assessments_of(w1.id),
                                   c.comment_sha)["vetoed"] is True


def test_http_a_strike_shows_as_vetoed_on_the_next_queue_fetch(live_server, w1):
    """Spec 5 r10 end to end: the page posts the strike and reloads, and
    what comes back on the reload marks that comment's reading struck --
    the one round trip the page's own reload depends on."""
    port, db_path = live_server
    seed_db = SyllabusDb(db_path)
    # w1's picture need has no candidate of its own, so no rate question
    # would carry the comment -- seed one (as the queue HTTP test does).
    _provide(seed_db, w1.id, "picture", items=[{"sha": "p1"}])
    rs.append_gallery_note(seed_db, subject=w1.id, card_id=w1.id, kind="reading", text="busy",
                           shown={}, subject_kind="word")
    (c,) = record_mod.comments(seed_db)
    _reading_row(seed_db, w1.id, c.comment_sha)
    seed_db.close()

    def queued_reading():
        status, body = _get(port, "/api/queue")
        assert status == 200
        (item,) = [i for i in json.loads(body) if i["type"] == "rate"
                   and i["subject"] == w1.id and i["kind"] == "picture"]
        (entry,) = [n for n in item["comments"] if n["comment_sha"] == c.comment_sha]
        return entry["reading"]

    assert queued_reading() == {"reading": "wants plain rice", "prompt_version": "1",
                                "vetoed": False, "actions": [], "unactionable": []}
    status, body = _post(port, "/api/veto", {"subject": w1.id, "comment_sha": c.comment_sha,
                                             "prompt_version": "1", "subject_kind": "word"})
    assert status == 200 and json.loads(body)["ok"] is True
    assert queued_reading()["vetoed"] is True


def test_http_veto_is_idempotent(live_server, w1):
    """A second strike writes a second row and changes nothing else."""
    port, db_path = live_server
    verify_db = SyllabusDb(db_path)
    rs.append_gallery_note(verify_db, subject=w1.id, card_id=w1.id, kind="reading", text="busy",
                           shown={}, subject_kind="word")
    (c,) = record_mod.comments(verify_db)
    _reading_row(verify_db, w1.id, c.comment_sha)
    payload = {"subject": w1.id, "comment_sha": c.comment_sha, "prompt_version": "1"}
    assert _post(port, "/api/veto", payload)[0] == 200
    assert _post(port, "/api/veto", payload)[0] == 200
    assert record_mod.vetoed_readings_all(verify_db) == frozenset({(c.comment_sha, "1")})


@pytest.mark.parametrize("payload", [
    {"subject": "rice"},
    {"subject": "rice", "comment_sha": "c1c1c1c1c1c1c1c1"},
    {"subject": "rice", "prompt_version": "1"},
    {"comment_sha": "c1c1c1c1c1c1c1c1", "prompt_version": "1"},
    {"subject": "rice", "comment_sha": "not-a-sha", "prompt_version": "1"},
    {"subject": "rice", "comment_sha": "C1C1C1C1C1C1C1C1", "prompt_version": "1"},
    {"subject": "rice", "comment_sha": ["c1c1c1c1c1c1c1c1"], "prompt_version": "1"},
    # a truthy prompt_version that is no version: str()-ing it would
    # write a durable veto row naming a reading no row can ever carry
    {"subject": "rice", "comment_sha": "c1c1c1c1c1c1c1c1", "prompt_version": ["1"]},
    {"subject": "rice", "comment_sha": "c1c1c1c1c1c1c1c1", "prompt_version": 1},
    {"subject": "rice", "comment_sha": "c1c1c1c1c1c1c1c1", "prompt_version": "  "},
    # a subject_kind outside the closed vocabulary record.subject_kind_of
    # reads back, refused the way /api/note refuses an unknown kind
    {"subject": "rice", "comment_sha": "c1c1c1c1c1c1c1c1", "prompt_version": "1",
     "subject_kind": "noun"},
])
def test_http_veto_refuses_a_body_that_does_not_name_one_reading(live_server, payload):
    """A strike names a subject and a whole (comment_sha,
    prompt_version) reference -- half a reference, or a comment_sha that
    is not one, would strike nothing and must not be stored as if it
    had."""
    port, db_path = live_server
    status, body = _post(port, "/api/veto", payload)
    assert status == 400 and json.loads(body)["ok"] is False
    assert record_mod.vetoed_readings_all(SyllabusDb(db_path)) == frozenset()


@pytest.mark.parametrize("body", [[], 5, "x", None])
def test_http_refuses_a_body_that_is_not_a_json_object(live_server, body):
    """Valid JSON that is not an object has no fields to read: `.get` on
    one raises AttributeError past the handlers' own (KeyError,
    ValueError) clause, so the shape is refused once, at the read, and
    every endpoint answers 400 rather than 500."""
    port, db_path = live_server
    status, out = _post(port, "/api/veto", body)
    assert status == 400 and json.loads(out)["ok"] is False
    assert record_mod.vetoed_readings_all(SyllabusDb(db_path)) == frozenset()


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
    (root / "curated" / "frequency_th.txt").write_text("", encoding="utf-8")
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
        deck_db.append(port="provide", backend="openverse",
                       key=ProvideKey(source="openverse", kind="", query=subject),
                       subject=subject,
                       question={"kind": "picture", "params": {"query": subject}},
                       answer={"items": [{"sha": sha}]})
        deck_db.append(port="assess", backend="judge",
                       key=JudgeKey.for_rule(rubric, sha, subject, "picture-for-word"),
                       subject=subject,
                       question={"role": "picture-for-word", "artifact_sha": sha,
                                "rubric": rubric, "kind": "picture"},
                       answer={"value": True})
    # one more source asked for rice since its candidate arrived: an attempt
    # the deck's own cap of 1 counts as its last.
    deck_db.append(port="provide", backend="wikimedia",
                   key=ProvideKey(source="wikimedia", kind="", query="rice"), subject="rice",
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
            transient_cap=src.transient_cap,
            provenance_source=provenance_source)]

    assert ctx.exhausted("rice", "picture") == exhausted(
        src.db, "rice", "picture", sources=src.sources_for("picture"),
        attempt_cap=src.attempt_cap, transient_cap=src.transient_cap)


def test_screen_stats_count_coverage_and_exhaustion_as_the_run_does(deck_with_history):
    from thai_syllabus.attempts import provenance_source_for
    from thai_syllabus.derivations import (
        LEARNER_RANK,
        all_needs,
        available_needs,
        current_best,
        exhausted,
    )
    from thai_syllabus.wiring import build_sourcing

    ctx = rs.load_context(deck_with_history)
    src = build_sourcing(deck_with_history)
    provenance_source = provenance_source_for(src.db)

    coverage: dict[str, dict[str, int]] = {}
    for subject, kind, _subject_kind in all_needs(src.syllabus):
        best = current_best(src.db, subject, kind, current_rubric=src.rubrics,
                            prior=src.provenance_prior, provenance_source=provenance_source)
        bucket = coverage.setdefault(kind, {"covered": 0, "accepted": 0, "total": 0})
        bucket["total"] += 1
        if best.artifact_sha is not None:
            bucket["covered"] += 1
        if best.rank >= LEARNER_RANK["acceptable"]:
            bucket["accepted"] += 1

    exhausted_remaining = 0
    for subject, kind, _subject_kind in available_needs(src.syllabus):
        if exhausted(src.db, subject, kind, sources=src.sources_for(kind),
                     attempt_cap=src.attempt_cap, transient_cap=src.transient_cap).exhausted:
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
    (root / "curated" / "frequency_th.txt").write_text("", encoding="utf-8")
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n",
        encoding="utf-8")
    deck_db = SyllabusDb(root / "syllabus.db")
    deck_db.append(port="provide", backend="openverse",
                   key=ProvideKey(source="openverse", kind="", query="orange"),
                   subject="orange", question={"kind": "picture", "params": {}},
                   answer={"items": [{"sha": "pic1"}]})
    deck_db.append(port="assess", backend="judge",
                   key=JudgeKey.for_rule(PICTURE_FIT_RUBRIC, "pic1", "orange",
                                        "picture-for-word"),
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


def test_load_context_session_cap_comes_from_providers_yaml_learner_quota(tmp_path):
    """spec 3 section 7's "learner 20/session" is wiring's own
    budgets["learner"].max_asks, providers.yaml-configurable through the
    same "quotas" path as forvo's day budget: load_context's
    ReviewContext takes its session cap from the loaded Derivations
    bundle, not a reviewserver-local default.
    """
    from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
    from thai_syllabus.entities import Category
    from thai_syllabus.profile import Profile

    root = tmp_path / "deck"
    words = tuple(word(f"w{i}", f"คำ{i}", f"word {i}") for i in range(5))
    save_curated(root / "curated", CuratedBundle(
        words=words,
        targets=tuple(target(f"w{i}/receptive", f"w{i}") for i in range(5)),
        graphemes=(), confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset(w.id for w in words)),)))
    (root / "curated" / "frequency_th.txt").write_text("", encoding="utf-8")
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n"
        "quotas:\n  learner: {max_asks: 3}\n", encoding="utf-8")

    ctx = rs.load_context(root)
    # Each word's needs otherwise have no candidate, so none is a rate
    # question yet (spec 5 r7 F4) -- give every word a picture candidate
    # so the session cap, not a lack of anything to ask, is what limits
    # questions() to 3.
    for i in range(5):
        _provide(ctx.derivations.db, f"w{i}", "picture", items=[{"sha": f"p{i}"}])

    assert ctx.learner_budget == 3
    assert len(ctx.questions()) == 3
    assert ctx.syllabus.assessments is ctx.cache


def test_rejected_candidates_carry_the_deciding_verdict(derivations, db, w1):
    _provide(db, w1.id, "picture", query="rice photo", items=[{"sha": "sA"}, {"sha": "sB"}])
    _judge(db, w1.id, "picture", "sA", True, evidence="clear rice bowl")
    _judge(db, w1.id, "picture", "sB", False, evidence="a cat")
    items = rs.build_queue(derivations, budget=50)
    rated = next(i for i in items if i["type"] == "rate" and i["subject"] == w1.id
                and i["kind"] == "picture")
    assert rated["rejected"] == [{"sha": "sB", "url": "/media/sB", "verdict": "judge: fail — a cat"}]


def test_a_need_with_no_candidate_and_a_source_left_is_not_a_rate_question(derivations, db, w1):
    """Spec 5 r7 section 1: nothing to rate -- the machine has sources
    left, so the learner is not asked yet (F4)."""
    _provide(db, w1.id, "picture", query="rice photo", items=[])   # one `nothing` outcome
    items = rs.build_queue(derivations, budget=50)
    assert not [i for i in items if i["subject"] == w1.id and i["kind"] == "picture"]


def test_a_need_with_no_candidate_and_no_source_left_is_a_direction_question(derivations, db, w1):
    for source in ("openverse", "wikimedia", "pexels"):
        _provide(db, w1.id, "picture", backend=source, query="rice photo", items=[])
    items = rs.build_queue(derivations, budget=50)
    mine = [i for i in items if i["subject"] == w1.id and i["kind"] == "picture"]
    assert [i["type"] for i in mine] == ["direction"]


# --- spec 5 r9: a question names its subject, carries `shown` and comments -

def test_a_question_names_its_subject_in_words_not_a_sha(derivations, db, w1):
    """Design ruling 5: a word's thai, id and meaning; a sentence's text
    and gloss."""
    _provide(db, w1.id, "picture", items=[{"sha": "p1"}])
    _judge(db, w1.id, "picture", "p1", True)      # p1 is current-best, so it is what was shown
    (item,) = [i for i in rs.build_queue(derivations, budget=50)
               if i["type"] == "rate" and i["subject"] == w1.id and i["kind"] == "picture"]
    assert item["label"] == {"thai": "ข้าว", "gloss": "rice", "id": "rice"}   # ข้าว: rice
    assert item["shown"]["picture"] == "p1" and item["shown"]["recordings"] == []
    assert item["shown"]["text_sha"] is None
    assert len(item["shown"]["syllabus_state_id"]) == 64
    assert item["comments"] == []


def test_subject_label_and_shown_for_a_sentence_and_a_pair(syllabus, w1, w2, pair):
    """A sentence names its text and gloss; a pair its members' Thai; an
    unknown sentence falls back to its id with no Thai."""
    s = sentence(((w1.id, w2.id),), thai_of(w1, w2), gloss="rice is near")
    with_sentence = dataclasses.replace(syllabus, sentences=(s,))
    assert rs._subject_label(with_sentence, s.text_sha, "sentence") == {
        "thai": s.text, "gloss": "rice is near", "id": s.text_sha}
    assert rs._subject_label(syllabus, s.text_sha, "sentence") == {
        "thai": None, "gloss": None, "id": s.text_sha}
    assert rs._subject_label(syllabus, pair.id, "pair") == {
        "thai": "ข้าว / ใกล้", "gloss": "rice / near", "id": pair.id}   # ข้าว: rice, ใกล้: near
    shown = rs._question_shown("recording", s.text_sha, "sentence", "r" * 64, "s" * 64)
    assert shown == {"picture": None, "recordings": ["r" * 64], "text_sha": s.text_sha,
                     "syllabus_state_id": "s" * 64}
    assert rs._question_shown("picture", "rice", "word", None, "s" * 64)["recordings"] == []


def test_a_question_lists_every_comment_on_its_subject_unread(derivations, db, w1):
    _provide(db, w1.id, "picture", items=[{"sha": "p1"}])
    rs.append_gallery_note(db, subject=w1.id, card_id=w1.id, kind="reading", text="blurry",
                           shown={}, subject_kind="word")
    rs.append_gallery_note(db, subject=w1.id, card_id=w1.id, kind="question", text="not rice",
                           shown={}, subject_kind="word", question_kind="rate",
                           artifact_kind="picture")
    (item,) = [i for i in rs.build_queue(derivations, budget=50)
               if i["type"] == "rate" and i["subject"] == w1.id and i["kind"] == "picture"]
    assert [(c["text"], c["card_kind"], c["question_kind"], c["reading"])
            for c in item["comments"]] == [("blurry", "reading", None, None),
                                          ("not rice", "question", "rate", None)]
    assert all(len(c["comment_sha"]) == 16 for c in item["comments"])


def _reading_row(db, subject, comment_sha, version="1", actions=(), unactionable=()):
    """The comment pass's reading row (record.reading_of reads it back)."""
    return db.append(port="assess", backend="llm",
                     key=CommentReadingKey(comment_sha, version), subject=subject,
                     question={"kind": "comment-reading", "comment_sha": comment_sha,
                               "prompt_version": version, "subject_kind": "word",
                               "anchor": subject, "card_kind": "reading"},
                     answer={"reading": "wants plain rice", "actions": list(actions),
                             "unactionable": list(unactionable)})


def test_a_questions_comments_carry_their_reading(derivations, db, w1):
    """Spec 5 r10: the comment view carries the run's reading of it --
    the text, the labelled actions, the unactionable requests and the
    prompt version the strike would name."""
    _provide(db, w1.id, "picture", items=[{"sha": "p1"}])
    rs.append_gallery_note(db, subject=w1.id, card_id=w1.id, kind="reading", text="busy",
                           shown={}, subject_kind="word")
    (c,) = record_mod.comments(db)
    _reading_row(db, w1.id, c.comment_sha,
                 actions=[{"action": "direction", "kind": "picture", "text": "plain rice",
                           "outcome": "done"}], unactionable=["bigger font"])
    (item,) = [i for i in rs.build_queue(derivations, budget=50)
               if i["type"] == "rate" and i["subject"] == w1.id and i["kind"] == "picture"]
    assert item["comments"][0]["reading"] == {
        "reading": "wants plain rice", "prompt_version": "1", "vetoed": False,
        "actions": ["direction for the picture search: plain rice"],
        "unactionable": ["no action available: bigger font"]}


def test_a_cards_notes_carry_their_reading(derivations, db, media_store, w1):
    sha_a = _seed_picture(db, media_store, w1.id)
    _judge(db, w1.id, "picture", sha_a, True)
    rs.append_gallery_note(db, subject=w1.id, card_id=w1.id, kind="reading", text="busy",
                           shown={}, subject_kind="word")
    (c,) = record_mod.comments(db)
    _reading_row(db, w1.id, c.comment_sha)
    reading_card = next(card for card in rs.compiled_cards(derivations)
                        if card["kind"] == "reading" and card["id"] == w1.id)
    assert reading_card["notes"][0]["reading"]["reading"] == "wants plain rice"
    assert reading_card["notes"][0]["reading"]["vetoed"] is False


def test_a_note_whose_identity_was_never_read_stays_unread(derivations, db, media_store, w1):
    """An Anki flag-import row carries no note text, so the comment pass
    never reads it, yet card_notes lists it with an identity of its own
    -- reading_view answers None for it and the entry shows unread."""
    sha_a = _seed_picture(db, media_store, w1.id)
    _judge(db, w1.id, "picture", sha_a, True)
    db.append(port="assess", backend="learner",
              key=FlagKey(family="word", anchor=w1.id, card_kind="reading", flags=1),
              subject=w1.id,
              question={"kind": "card-flag", "role": "card-flag", "family": "word",
                        "anchor": w1.id, "card_kind": "reading", "flags": 1},
              answer={"flagged": True, "flag": 1})
    reading_card = next(card for card in rs.compiled_cards(derivations)
                        if card["kind"] == "reading" and card["id"] == w1.id)
    assert [n["reading"] for n in reading_card["notes"]] == [None]


def test_direction_challenger_and_reask_items_carry_the_label(derivations, db, w1):
    for source in ("pexels", "openverse", "wikimedia"):
        _provide(db, w1.id, "picture", backend=source, query="rice photo", items=[])
    (item,) = [i for i in rs.build_queue(derivations, budget=50)
               if i["type"] == "direction" and i["subject"] == w1.id and i["kind"] == "picture"]
    assert item["label"]["thai"] == "ข้าว" and "comments" in item and "shown" in item   # ข้าว: rice


def test_challenger_item_carries_label_shown_and_comments(derivations, db, w1):
    _learner(db, w1.id, "picture", "s-old", "acceptable")
    _judge(db, w1.id, "picture", "s-new", True, rubric="rubric-v2")
    items = rs.build_queue(
        dataclasses.replace(derivations, current_rubric={"picture-for-word": "rubric-v2"}),
        budget=50)
    (challenger,) = [i for i in items if i["type"] == "challenger" and i["subject"] == w1.id]
    assert challenger["label"]["thai"] == "ข้าว"   # ข้าว: rice
    assert challenger["shown"]["picture"] == "s-old"
    assert challenger["comments"] == []


def test_reask_item_carries_label_shown_and_comments(derivations, db, w1):
    db.append_study(_word_study_row(w1.id))
    _learner(db, w1.id, "picture", "sA", "good")
    items = rs.build_queue(derivations, study=db, budget=50)
    (reask,) = [i for i in items if i["type"] == "reask" and i["subject"] == w1.id]
    assert reask["label"]["thai"] == "ข้าว"   # ข้าว: rice
    assert reask["shown"]["picture"] == "sA"
    assert reask["comments"] == []


def test_a_rendition_rate_items_shown_carries_the_member_recording_shas(
        derivations, db, w1, w2, pair):
    """r9 fix round 1: a rendition names no single artifact of its own --
    `current_best`'s artifact_sha is the compound identity
    (cachekeys.rendition_identity), never a real recording sha. The
    question's `shown` must instead name what the learner actually
    heard: the two member recordings backing that rendition, in the
    pair's own member order (mirroring the gallery's own pair-card
    `shown`, `_shown_of`'s `recordings` list)."""
    from thai_syllabus.cachekeys import rendition_identity

    shas = {w1.id: "rec-w1-sha", w2.id: "rec-w2-sha"}
    db.append(port="attempt", backend="forvo",
             key=AttemptOutcomeKey(subject=pair.id, kind="rendition", source="forvo"),
             subject=pair.id,
             question={"kind": "rendition", "subject_kind": "pair", "source": "forvo"},
             answer={"outcome": "candidates", "candidates": [rendition_identity(shas)]})
    db.append(port="assess", backend="rendition",
             key=MechanicalKey(check="rendition", params="v1", subject=pair.id,
                              artifact_sha=rendition_identity(shas)),
             subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": rendition_identity(shas),
                      "rubric": None, "kind": "rendition", "subject_kind": "pair",
                      "params": {"members": shas}},
             answer={"value": True})

    (item,) = [i for i in rs.build_queue(derivations, budget=50)
               if i["type"] == "rate" and i["subject"] == pair.id and i["kind"] == "rendition"]
    assert item["shown"]["recordings"] == [shas[w1.id], shas[w2.id]]
    assert item["shown"]["picture"] is None
