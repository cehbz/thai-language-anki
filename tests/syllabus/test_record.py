"""Tests for record.py (spec 3 section 6): folds over cache rows, keyed
by each row's own explicit question["kind"] -- never inferred from a
`provides` or `role` string.
"""
import pytest

from thai_syllabus.record import (
    candidate_shas,
    directions,
    judge_verdicts,
    latest_query,
    learner_ratings,
    ratings_for_role,
    rows_for,
    source_asks,
)
from thai_syllabus.store import SyllabusDb


@pytest.fixture
def cache(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


def test_rows_for_selects_by_explicit_kind(cache):
    cache.append("provide", "openverse", "k1", "rice", {"kind": "picture", "query": "rice"}, {"items": []}, 0)
    cache.append("provide", "imgfetch", "k2", "rice", {"kind": "picture", "url": "u"}, {"items": [{"sha": "a"*64}]}, 0)
    cache.append("provide", "forvo", "k3", "rice", {"kind": "recording"}, {"items": []}, 0)
    rows = rows_for(cache, "rice", "picture")
    assert [r.backend for r in rows] == ["openverse", "imgfetch"]
    assert [r.backend for r in source_asks(rows)] == ["openverse"]
    assert candidate_shas(rows) == ["a" * 64]


def test_rows_for_ignores_a_row_with_no_matching_kind(cache):
    cache.append("provide", "forvo", "k1", "rice", {"kind": "recording"}, {"items": []}, 0)
    assert rows_for(cache, "rice", "picture") == []


def test_source_asks_excludes_audiofetch_too(cache):
    cache.append("provide", "forvo", "k1", "w", {"kind": "recording"}, {"items": []}, 0)
    cache.append("provide", "audiofetch", "k2", "w", {"kind": "recording"},
                {"items": [{"sha": "b" * 64}]}, 0)
    rows = rows_for(cache, "w", "recording")
    assert [r.backend for r in source_asks(rows)] == ["forvo"]


def test_candidate_shas_is_first_seen_order_across_rows(cache):
    cache.append("provide", "openverse", "k1", "w", {"kind": "picture"},
                {"items": []}, 0)
    cache.append("provide", "imgfetch", "k2", "w", {"kind": "picture"},
                {"items": [{"sha": "s1"}, {"sha": "s2"}]}, 0)
    cache.append("provide", "imgfetch", "k3", "w", {"kind": "picture"},
                {"items": [{"sha": "s2"}, {"sha": "s3"}]}, 0)
    rows = rows_for(cache, "w", "picture")
    assert candidate_shas(rows) == ["s1", "s2", "s3"]


def test_learner_ratings_selects_only_rating_kind_rows_newest_last(cache):
    cache.append("assess", "learner", "learner:w:s1", "w",
                {"kind": "direction", "role": "picture-for-word"}, {"direction": "try red"}, 0)
    cache.append("assess", "learner", "learner:w:s1", "w",
                {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "acceptable"}, 0)
    cache.append("assess", "learner", "learner:w:s2", "w",
                {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s2"},
                {"value": "good"}, 0)
    rows = cache.assessments_of("w")
    ratings = learner_ratings(rows)
    assert [r.answer["value"] for r in ratings] == ["acceptable", "good"]


def test_directions_selects_only_direction_kind_rows(cache):
    cache.append("assess", "learner", "learner:w:1", "w",
                {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "acceptable"}, 0)
    cache.append("assess", "learner", "learner:direction:w", "w",
                {"kind": "direction", "of": "image_query"}, {"direction": "try red"}, 0)
    rows = cache.assessments_of("w")
    result = directions(rows)
    assert len(result) == 1
    assert result[0].answer["direction"] == "try red"


def test_ratings_for_role_scopes_one_subjects_ratings_to_one_need(cache):
    # Two needs on the same subject (a word's picture and recording
    # ratings) must not bleed into each other -- both rows carry
    # kind="rating"; only the role tells them apart.
    cache.append("assess", "learner", "learner:w:s1", "w",
                {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": "acceptable"}, 0)
    cache.append("assess", "learner", "learner:w:s2", "w",
                {"kind": "rating", "role": "recording-for-word", "artifact_sha": "s2"},
                {"value": "good"}, 0)
    cache.append("assess", "learner", "learner:w:s3", "w",
                {"kind": "rating", "role": "picture-for-word", "artifact_sha": "s3"},
                {"value": "bogus-not-a-real-rating"}, 0)
    rows = cache.assessments_of("w")
    picture_ratings = ratings_for_role(rows, "picture-for-word")
    assert [r.question["artifact_sha"] for r in picture_ratings] == ["s1"]
    assert [r.question["artifact_sha"] for r in ratings_for_role(rows, "recording-for-word")] == ["s2"]


def test_judge_verdicts_selects_by_role(cache):
    cache.append("assess", "judge", "judge:w:s1", "w",
                {"kind": "picture", "role": "picture-for-word", "artifact_sha": "s1"},
                {"value": True}, 0)
    cache.append("assess", "judge", "judge:w:pref", "w",
                {"kind": "picture", "role": "picture-preference", "artifact_sha": None},
                {"value": ["s1"]}, 0)
    rows = cache.assessments_of("w")
    fit = judge_verdicts(rows, "picture-for-word")
    assert [r.question["role"] for r in fit] == ["picture-for-word"]


def test_latest_query_reads_the_newest_source_asks_params(cache):
    cache.append("provide", "openverse", "k1", "w", {"kind": "picture", "params": {"query": "old"}},
                {"items": []}, 0)
    cache.append("provide", "openverse", "k2", "w", {"kind": "picture", "params": {"query": "new"}},
                {"items": []}, 0)
    rows = rows_for(cache, "w", "picture")
    assert latest_query(rows) == "new"


def test_latest_query_is_none_with_no_source_ask(cache):
    assert latest_query([]) is None
