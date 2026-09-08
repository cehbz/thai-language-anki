"""Tests for record.py (spec 3 section 6): folds over cache rows, keyed
by each row's own explicit question["kind"] -- never inferred from a
`provides` or `role` string.
"""
import json
import logging

import pytest

from thai_syllabus.cachekeys import DirectionKey, JudgeKey, LearnerKey, ProvideKey, sha
from thai_syllabus.record import (
    DRAFT_SUBJECT,
    SentenceDraft,
    asks_since,
    candidate_shas,
    directions,
    drafts_in,
    judge_verdicts,
    latest_query,
    latest_rating,
    learner_ratings,
    merge_drafts,
    ratings_for_role,
    rows_for,
    sentence_drafts,
    source_asks,
    spend_since,
)
from thai_syllabus.store import SyllabusDb


@pytest.fixture
def cache(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


def test_rows_for_selects_by_explicit_kind(cache):
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="k1"),
                "rice", {"kind": "picture", "query": "rice"}, {"items": []}, 0)
    cache.append("provide", "imgfetch", ProvideKey(source="", kind="", query="k2"),
                "rice", {"kind": "picture", "url": "u"}, {"items": [{"sha": "a"*64}]}, 0)
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="k3"),
                "rice", {"kind": "recording"}, {"items": []}, 0)
    rows = rows_for(cache, "rice", "picture")
    assert [r.backend for r in rows] == ["openverse", "imgfetch"]
    assert [r.backend for r in source_asks(rows)] == ["openverse"]
    assert candidate_shas(rows) == ["a" * 64]


def test_rows_for_ignores_a_row_with_no_matching_kind(cache):
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="k1"),
                "rice", {"kind": "recording"}, {"items": []}, 0)
    assert rows_for(cache, "rice", "picture") == []


def test_source_asks_excludes_audiofetch_too(cache):
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="k1"),
                "w", {"kind": "recording"}, {"items": []}, 0)
    cache.append("provide", "audiofetch", ProvideKey(source="", kind="", query="k2"),
                "w", {"kind": "recording"},
                {"items": [{"sha": "b" * 64}]}, 0)
    rows = rows_for(cache, "w", "recording")
    assert [r.backend for r in source_asks(rows)] == ["forvo"]


def test_source_asks_excludes_a_learner_supply_row_too(cache):
    """I2: a learner supply is an answer, not a Source ask (spec 3
    section 1's vocabulary)."""
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="k1"),
                "w", {"kind": "picture", "params": {"query": "rice"}}, {"items": []}, 0)
    cache.append("provide", "learner", ProvideKey(source="learner", kind="", query="/tmp/x.jpg"),
                "w", {"kind": "picture", "params": {"path": "/tmp/x.jpg"}},
                {"items": [{"sha": "c" * 64}]}, 0)
    rows = rows_for(cache, "w", "picture")
    assert [r.backend for r in source_asks(rows)] == ["openverse"]


def test_source_asks_excludes_a_legacy_current_row(cache):
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="k1"),
                "w", {"kind": "picture", "params": {"query": "rice"}}, {"items": []}, 0)
    cache.append("provide", "legacy-current",
                ProvideKey(source="legacy-current", kind="picture", query="w"),
                "w", {"provides": "picture", "kind": "picture", "subject_kind": "word",
                      "params": {"image": "images/pw-1.jpg"}},
                {"items": [{"sha": "c" * 64, "ext": "jpg"}]}, 0)
    rows = rows_for(cache, "w", "picture")
    assert [r.backend for r in source_asks(rows)] == ["openverse"]
    assert candidate_shas(rows) == ["c" * 64]


def test_latest_query_ignores_a_newer_legacy_current_row(cache):
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="k1"),
                "w", {"kind": "picture", "params": {"query": "rice bowl"}}, {"items": []}, 0)
    cache.append("provide", "legacy-current",
                ProvideKey(source="legacy-current", kind="picture", query="w"),
                "w", {"provides": "picture", "kind": "picture", "subject_kind": "word",
                      "params": {"image": "images/pw-1.jpg"}},
                {"items": [{"sha": "c" * 64, "ext": "jpg"}]}, 0)
    assert latest_query(rows_for(cache, "w", "picture")) == "rice bowl"


def test_candidate_shas_is_first_seen_order_across_rows(cache):
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="k1"),
                "w", {"kind": "picture"}, {"items": []}, 0)
    cache.append("provide", "imgfetch", ProvideKey(source="", kind="", query="k2"),
                "w", {"kind": "picture"},
                {"items": [{"sha": "s1"}, {"sha": "s2"}]}, 0)
    cache.append("provide", "imgfetch", ProvideKey(source="", kind="", query="k3"),
                "w", {"kind": "picture"},
                {"items": [{"sha": "s2"}, {"sha": "s3"}]}, 0)
    rows = rows_for(cache, "w", "picture")
    assert candidate_shas(rows) == ["s1", "s2", "s3"]


def test_learner_ratings_selects_only_rating_kind_rows_newest_last(cache):
    cache.append("assess", "learner",
                DirectionKey(subject="w", role="picture-for-word", text_sha=sha("try red")),
                "w", {"kind": "direction", "role": "picture-for-word"},
                {"direction": "try red"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="s1", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "acceptable"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="s2", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s2"},
                {"value": "good"}, 0)
    rows = cache.assessments_of("w")
    ratings = learner_ratings(rows)
    assert [r.answer["value"] for r in ratings] == ["acceptable", "good"]


def test_directions_selects_only_direction_kind_rows(cache):
    cache.append("assess", "learner", LearnerKey(artifact_sha="s1", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "acceptable"}, 0)
    cache.append("assess", "learner",
                DirectionKey(subject="w", role="picture-for-word", text_sha=sha("try red")),
                "w", {"kind": "direction", "of": "image_query"}, {"direction": "try red"}, 0)
    rows = cache.assessments_of("w")
    result = directions(rows)
    assert len(result) == 1
    assert result[0].answer["direction"] == "try red"


def test_ratings_for_role_scopes_one_subjects_ratings_to_one_need(cache):
    # Two needs on the same subject (a word's picture and recording
    # ratings) must not bleed into each other -- both rows carry
    # kind="rating"; only the role tells them apart.
    cache.append("assess", "learner", LearnerKey(artifact_sha="s1", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "acceptable"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="s2", role="recording-for-word"),
                "w", {"kind": "rating", "role": "recording-for-word", "artifact_sha": "s2"},
                {"value": "good"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="s3", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s3"},
                {"value": "bogus-not-a-real-rating"}, 0)
    rows = cache.assessments_of("w")
    picture_ratings = ratings_for_role(rows, "picture-for-word")
    assert [r.question["artifact_sha"] for r in picture_ratings] == ["s1"]
    assert [r.question["artifact_sha"] for r in ratings_for_role(rows, "recording-for-word")] == ["s2"]


def test_latest_rating_is_the_newest_value_under_that_role(cache):
    cache.append("assess", "learner", LearnerKey(artifact_sha="s1", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "acceptable"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="s2", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s2"},
                {"value": "good"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="s3", role="recording-for-word"),
                "w", {"kind": "rating", "role": "recording-for-word", "artifact_sha": "s3"},
                {"value": "unacceptable-none"}, 0)
    rows = cache.assessments_of("w")
    assert latest_rating(rows, "picture-for-word") == "good"
    assert latest_rating(rows, "recording-for-word") == "unacceptable-none"


def test_latest_rating_is_none_when_the_role_has_no_rating(cache):
    cache.append("assess", "learner", LearnerKey(artifact_sha="s1", role="picture-for-word"),
                "w", {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "good"}, 0)
    assert latest_rating(cache.assessments_of("w"), "recording-for-word") is None


def test_judge_verdicts_selects_by_role(cache):
    cache.append("assess", "judge", JudgeKey.for_rule(None, "s1", "w", "picture-for-word"),
                "w", {"kind": "picture", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": True}, 0)
    cache.append("assess", "judge", JudgeKey.for_rule(None, None, "w-pref", "picture-preference"),
                "w", {"kind": "picture", "role": "picture-preference", "artifact_sha": None},
                {"value": ["s1"]}, 0)
    rows = cache.assessments_of("w")
    fit = judge_verdicts(rows, "picture-for-word")
    assert [r.question["role"] for r in fit] == ["picture-for-word"]


def test_latest_query_reads_the_newest_source_asks_params(cache):
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="old"),
                "w", {"kind": "picture", "params": {"query": "old"}}, {"items": []}, 0)
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="new"),
                "w", {"kind": "picture", "params": {"query": "new"}}, {"items": []}, 0)
    rows = rows_for(cache, "w", "picture")
    assert latest_query(rows) == "new"


def test_latest_query_is_none_with_no_source_ask(cache):
    assert latest_query([]) is None


def test_latest_query_ignores_a_later_learner_supply_row(cache):
    """I2: record.latest_query still returns the prior search phrase
    after a local-path supply -- the supply is not a Source ask."""
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="rice"),
                "w", {"kind": "picture", "params": {"query": "rice"}}, {"items": []}, 0)
    cache.append("provide", "learner", ProvideKey(source="learner", kind="", query="/tmp/x.jpg"),
                "w", {"kind": "picture", "params": {"path": "/tmp/x.jpg"}},
                {"items": [{"sha": "c" * 64}]}, 0)
    rows = rows_for(cache, "w", "picture")
    assert latest_query(rows) == "rice"


# --- the spend window a per-day budget is measured over --------------------

def test_asks_since_counts_that_backend_s_source_asks_in_the_window(cache):
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="rice"),
                "rice", {"kind": "recording"}, {"items": []}, 1.0, ts=100)
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="fish"),
                "fish", {"kind": "recording"}, {"items": []}, 2.0, ts=300)
    cache.append("provide", "tts", ProvideKey(source="tts", kind="", query=sha("rice")),
                "rice", {"kind": "recording"}, {"items": []}, 0.5, ts=300)
    cache.append("provide", "audiofetch", ProvideKey(source="", kind="", query="k4"),
                 "rice", {"kind": "recording"},
                 {"items": []}, 0.0, ts=300)   # bytes, not a Source ask
    assert asks_since(cache, "forvo", 200) == 1
    assert asks_since(cache, "forvo", 0) == 2


def test_spend_since_sums_the_cost_of_those_asks(cache):
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="rice"),
                "rice", {"kind": "recording"}, {"items": []}, 1.0, ts=100)
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="fish"),
                "fish", {"kind": "recording"}, {"items": []}, 2.5, ts=300)
    assert spend_since(cache, "forvo", 200) == pytest.approx(2.5)
    assert spend_since(cache, "forvo", 0) == pytest.approx(3.5)


# --- drafts_in: one draft per distinct text, claims merged ------------------

def test_drafts_in_merges_a_duplicated_text_unioning_claims_in_order():
    """A text listed twice is one draft: its target claims union in
    first-seen order, and the first non-empty gloss wins."""
    text = json.dumps({"sentences": [
        {"text": " กินข้าว ", "gloss": "", "targets": ["eat/receptive"]},
        {"text": "กินข้าว", "gloss": "eat rice",
         "targets": ["rice/receptive", "eat/receptive"]}]})   # กินข้าว: eat rice
    drafts = drafts_in(text)
    assert len(drafts) == 1
    assert drafts[0].text == "กินข้าว"     # กินข้าว: eat rice, whitespace stripped
    assert drafts[0].gloss == "eat rice"                       # the first non-empty gloss
    assert drafts[0].claimed == ("eat/receptive", "rice/receptive")


def test_drafts_in_drops_a_gloss_conflict_and_warns(caplog):
    """Two listings of one text carrying differing non-empty glosses drop
    the text -- fed no candidate downstream -- with a warning naming its
    first 40 characters."""
    text = json.dumps({"sentences": [
        {"text": "กินข้าว", "gloss": "eat rice", "targets": ["eat/receptive"]},
        {"text": "กินข้าว", "gloss": "rice is eaten",
         "targets": ["rice/receptive"]}]})   # กินข้าว: eat rice
    with caplog.at_level(logging.WARNING):
        drafts = drafts_in(text)
    assert drafts == []
    assert any("กินข้าว"[:40] in r.message for r in caplog.records)   # กินข้าว: eat rice


def test_drafts_in_keeps_distinct_texts_distinct():
    text = json.dumps({"sentences": [
        {"text": "กินข้าว", "gloss": "eat rice", "targets": ["eat/receptive"]},
        {"text": "ข้าวอร่อย", "gloss": "tasty rice",
         "targets": ["rice/receptive"]}]})   # กินข้าว: eat rice, ข้าวอร่อย: tasty rice
    drafts = drafts_in(text)
    assert [d.text for d in drafts] == ["กินข้าว", "ข้าวอร่อย"]   # eat rice, tasty rice


def test_merge_drafts_merges_a_text_across_two_lists_handed_together():
    """merge_drafts merges over whatever `drafts` it is given, not just
    one item's own listings -- spec 3 section 5's "a text listed twice is
    one candidate" holds across items, not per item."""
    from_item_a = [SentenceDraft(text="กินข้าว", gloss="eat rice",   # กินข้าว: eat rice
                                 claimed=("eat/receptive",))]
    from_item_b = [SentenceDraft(text="กินข้าว", gloss="eat rice",   # กินข้าว: eat rice
                                 claimed=("rice/receptive",))]
    merged = merge_drafts(from_item_a + from_item_b)
    assert len(merged) == 1
    assert merged[0].claimed == ("eat/receptive", "rice/receptive")


def test_sentence_drafts_merges_a_text_split_across_one_row_s_items(cache):
    """One provide row's own two items listing the same text with
    differing claims fold to one draft, its claims unioned -- the same
    merge `sentence_attempt` applies within a run."""
    cache.append("provide", "llm-sentence", ProvideKey(source="llm-sentence", kind="", query="q"),
                DRAFT_SUBJECT, {"kind": "sentence", "subject_kind": "sentence"},
                {"items": [
                    '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'   # กินข้าว: eat rice
                    ' "targets": ["eat/receptive"]}]}',
                    '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'   # กินข้าว: eat rice
                    ' "targets": ["rice/receptive"]}]}']}, 0)
    drafts = sentence_drafts(cache)
    assert len(drafts) == 1
    assert drafts[0].claimed == ("eat/receptive", "rice/receptive")


def test_sentence_drafts_merges_a_text_split_across_two_provide_rows(cache):
    """Two separate provide rows each listing the same text with
    differing claims fold to one draft, its claims unioned -- a text
    drafted in two runs is still one draft."""
    cache.append("provide", "llm-sentence", ProvideKey(source="llm-sentence", kind="", query="q1"),
                DRAFT_SUBJECT, {"kind": "sentence", "subject_kind": "sentence"},
                {"items": [
                    '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'   # กินข้าว: eat rice
                    ' "targets": ["eat/receptive"]}]}']}, 0, ts=100)
    cache.append("provide", "llm-sentence", ProvideKey(source="llm-sentence", kind="", query="q2"),
                DRAFT_SUBJECT, {"kind": "sentence", "subject_kind": "sentence"},
                {"items": [
                    '{"sentences": [{"text": "กินข้าว", "gloss": "eat rice",'   # กินข้าว: eat rice
                    ' "targets": ["rice/receptive"]}]}']}, 0, ts=200)
    drafts = sentence_drafts(cache)
    assert len(drafts) == 1
    assert drafts[0].claimed == ("eat/receptive", "rice/receptive")
