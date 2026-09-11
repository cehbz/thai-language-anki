"""Tests for record.py (spec 3 section 6): folds over cache rows, keyed
by each row's own explicit question["kind"] -- never inferred from a
`provides` or `role` string.
"""
import json
import logging
from datetime import date

import pytest

from thai_syllabus.cachekeys import (AttemptOutcomeKey, DirectionKey, JudgeKey, LearnerKey,
                                     PhraseKey, ProvideKey, RetirementKey, sha)
from thai_syllabus.ids import WordId
from thai_syllabus.record import (
    DRAFT_SUBJECT,
    PARSE_SUBJECT,
    PHRASE_SUBJECT,
    SentenceDraft,
    asks_since,
    card_notes,
    cost_since,
    fetches_since,
    candidate_shas,
    directions,
    draft_sentence,
    drafted_phrase,
    drafts_in,
    judge_verdicts,
    latest_phrase,
    last_source_ask_ts,
    latest_query,
    latest_rating,
    learner_ratings,
    merge_drafts,
    normalize_shown,
    parse_drafts,
    parse_no_fit,
    parse_phrases,
    parse_prompt,
    parses_in,
    ratings_for_role,
    retired_texts,
    rows_for,
    sentence_drafts,
    source_asks,
    spend_since,
    tried_urls,
    vocabulary_line,
)
from thai_syllabus.store import SyllabusDb

from .builders import word


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


def test_card_notes_selects_by_anchor_and_card_kind_oldest_first(cache):
    """spec 5 section 1 r5 (C2 fix): a card's notes are its own
    card-flag rows, scoped to one (anchor, card_kind) pair -- a word
    note's several card kinds (reading, production) share one subject
    and must not bleed into each other, and a minimal-pair note's two
    member cards share both subject AND card_kind ("recognition") but
    carry distinct anchors (compile.py's MemberKey) that must not bleed
    into each other either. Oldest first, each carrying its own `shown`
    back for staleness.
    """
    cache.append("assess", "learner", LearnerKey(artifact_sha="c1", role="card-flag"), "rice",
                {"role": "card-flag", "kind": "card-flag", "anchor": "c1",
                 "card_kind": "reading", "shown": {"picture": "sha-a"}},
                {"kind": "rating", "rating": None, "note": "first note"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="c1", role="card-flag"), "rice",
                {"role": "card-flag", "kind": "card-flag", "anchor": "c1",
                 "card_kind": "production", "shown": {"picture": "sha-a"}},
                {"kind": "rating", "rating": None, "note": "wrong card kind"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="c2", role="card-flag"), "rice",
                {"role": "card-flag", "kind": "card-flag", "anchor": "c2",
                 "card_kind": "reading", "shown": {"picture": "sha-a"}},
                {"kind": "rating", "rating": None, "note": "wrong anchor"}, 0)
    cache.append("assess", "learner", LearnerKey(artifact_sha="c1", role="card-flag"), "rice",
                {"role": "card-flag", "kind": "card-flag", "anchor": "c1",
                 "card_kind": "reading", "shown": {"picture": "sha-b"}},
                {"kind": "rating", "rating": None, "note": "second note"}, 0)
    rows = cache.assessments_of("rice")
    notes = card_notes(rows, "c1", "reading")
    assert [n["text"] for n in notes] == ["first note", "second note"]
    assert notes[0]["shown"] == {"picture": "sha-a"}
    assert notes[0]["ts"] < notes[1]["ts"]


def test_card_notes_normalizes_a_legacy_singular_recording_into_a_recordings_list(cache):
    """fix round 4: a row written under the earlier singular `recording`
    key must not be compared against the current `recordings`-list
    shape as if the two were different claims -- record.card_notes
    normalizes it on the way out: `recording` becomes `recordings:
    [value]`, and the legacy key is dropped, so every caller (reviewserver.
    py's `_is_stale` included) only ever sees one shape.
    """
    cache.append("assess", "learner", LearnerKey(artifact_sha="c1", role="card-flag"), "rice",
                {"role": "card-flag", "kind": "card-flag", "anchor": "c1", "card_kind": "reading",
                 "shown": {"picture": "p1", "recording": "r1"}},
                {"kind": "rating", "rating": None, "note": "legacy note"}, 0)
    notes = card_notes(cache.assessments_of("rice"), "c1", "reading")
    assert notes[0]["shown"] == {"picture": "p1", "recordings": ["r1"]}


def test_card_notes_normalizes_a_legacy_recording_of_none_to_an_empty_list(cache):
    cache.append("assess", "learner", LearnerKey(artifact_sha="c1", role="card-flag"), "rice",
                {"role": "card-flag", "kind": "card-flag", "anchor": "c1", "card_kind": "reading",
                 "shown": {"picture": "p1", "recording": None}},
                {"kind": "rating", "rating": None, "note": "legacy note"}, 0)
    notes = card_notes(cache.assessments_of("rice"), "c1", "reading")
    assert notes[0]["shown"] == {"picture": "p1", "recordings": []}


def test_normalize_shown_leaves_the_current_shape_unchanged(cache):
    shown = {"picture": "p1", "recordings": ["r1", "r2"], "text_sha": None}
    assert normalize_shown(shown) == shown


def test_normalize_shown_prefers_an_existing_recordings_list_over_a_stray_recording_key(cache):
    # both present is not a shape any writer produces, but normalize_shown
    # never silently drops information it can't reconcile -- it keeps the
    # explicit `recordings` list and still discards the legacy key.
    assert normalize_shown({"recording": "r1", "recordings": ["r2"]}) == {"recordings": ["r2"]}


def test_card_notes_defaults_shown_to_empty_mapping_when_absent(cache):
    """A card-flag row written before spec 5 r5 (no `shown` field at
    all) folds to {}, not a KeyError -- it named nothing, so nothing to
    check it against later.
    """
    cache.append("assess", "learner", LearnerKey(artifact_sha="c1", role="card-flag"), "rice",
                {"role": "card-flag", "kind": "card-flag", "anchor": "c1", "card_kind": "reading"},
                {"kind": "rating", "rating": None, "note": "a pre-r5 note"}, 0)
    notes = card_notes(cache.assessments_of("rice"), "c1", "reading")
    assert notes == [{"text": "a pre-r5 note", "ts": notes[0]["ts"], "shown": {}}]


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


def test_source_asks_excludes_a_drafted_phrase_row_too(cache):
    """spec 3 r24 section 5: attempts.phrase_attempt's per-subject phrase
    row (backend "llm") records the phrase a Source ask already drafted
    (the batch prompt, backend "llm-phrase") -- it is an answer, not an
    ask of its own, the same rationale as imgfetch/audiofetch."""
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="k1"),
                "rice", {"kind": "picture", "params": {"query": "rice food"}}, {"items": []}, 0)
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "a bowl of steamed rice"}, 0)
    rows = rows_for(cache, "rice", "picture")
    assert [r.backend for r in source_asks(rows)] == ["openverse"]


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


def test_candidate_shas_includes_an_attempt_outcome_rows_candidates(cache):
    """spec 3 section 6 (r23): an outcome row names the artifacts its own
    attempt stored for this need -- a url-keyed fetch shared by two
    subjects appends a provide row under the first only (spec 3 section
    2), so the second subject's own outcome row is where its stored sha
    is named, and it is a candidate of the need on that account alone."""
    cache.append("attempt", "forvo",
                AttemptOutcomeKey(subject="w", kind="recording", source="forvo"),
                "w", {"kind": "recording", "subject_kind": "word", "source": "forvo"},
                {"outcome": "candidates", "candidates": ["d" * 64], "tried": []}, 0)
    rows = rows_for(cache, "w", "recording")
    assert candidate_shas(rows) == ["d" * 64]


def test_candidate_shas_lists_a_sha_in_both_a_provide_and_an_outcome_row_once(cache):
    cache.append("provide", "imgfetch", ProvideKey(source="", kind="", query="k1"),
                "w", {"kind": "picture"}, {"items": [{"sha": "e" * 64}]}, 0)
    cache.append("attempt", "openverse",
                AttemptOutcomeKey(subject="w", kind="picture", source="openverse"),
                "w", {"kind": "picture", "subject_kind": "word", "source": "openverse"},
                {"outcome": "candidates", "candidates": ["e" * 64], "tried": []}, 0)
    rows = rows_for(cache, "w", "picture")
    assert candidate_shas(rows) == ["e" * 64]


def test_tried_urls_unions_the_tried_lists_of_every_matching_attempt_row(cache):
    cache.append("attempt", "openverse",
                AttemptOutcomeKey(subject="w", kind="picture", source="openverse"),
                "w", {"kind": "picture", "subject_kind": "word", "source": "openverse"},
                {"outcome": "candidates", "candidates": ["s1"],
                 "tried": ["https://x/a.jpg", "https://x/b.jpg"]}, 0)
    cache.append("attempt", "openverse",
                AttemptOutcomeKey(subject="w", kind="picture", source="openverse"),
                "w", {"kind": "picture", "subject_kind": "word", "source": "openverse"},
                {"outcome": "candidates", "candidates": ["s2"],
                 "tried": ["https://x/b.jpg", "https://x/c.jpg"]}, 0)
    assert tried_urls(cache, "w", "picture", "openverse") == frozenset(
        {"https://x/a.jpg", "https://x/b.jpg", "https://x/c.jpg"})


def test_tried_urls_ignores_a_refused_urls_absence_of_a_candidate(cache):
    """A url a served refusal answered still counts as tried, even though
    it produced no candidate."""
    cache.append("attempt", "openverse",
                AttemptOutcomeKey(subject="w", kind="picture", source="openverse"),
                "w", {"kind": "picture", "subject_kind": "word", "source": "openverse"},
                {"outcome": "nothing", "candidates": [], "tried": ["https://x/refused.jpg"]}, 0)
    assert tried_urls(cache, "w", "picture", "openverse") == frozenset({"https://x/refused.jpg"})


def test_tried_urls_ignores_a_different_source_or_kind(cache):
    cache.append("attempt", "openverse",
                AttemptOutcomeKey(subject="w", kind="picture", source="openverse"),
                "w", {"kind": "picture", "subject_kind": "word", "source": "openverse"},
                {"outcome": "candidates", "candidates": ["s1"], "tried": ["https://x/a.jpg"]}, 0)
    cache.append("attempt", "wikimedia",
                AttemptOutcomeKey(subject="w", kind="picture", source="wikimedia"),
                "w", {"kind": "picture", "subject_kind": "word", "source": "wikimedia"},
                {"outcome": "candidates", "candidates": ["s2"], "tried": ["https://x/other.jpg"]}, 0)
    assert tried_urls(cache, "w", "picture", "openverse") == frozenset({"https://x/a.jpg"})
    assert tried_urls(cache, "w", "recording", "openverse") == frozenset()


def test_tried_urls_is_empty_with_no_attempt_row_on_record(cache):
    assert tried_urls(cache, "w", "picture", "openverse") == frozenset()


def test_tried_urls_folds_a_pre_tried_field_attempt_row_as_empty(cache):
    """An attempt row written before the `tried` field existed has no
    `tried` key at all; it must fold in as empty, not raise."""
    cache.append("attempt", "wikimedia",
                AttemptOutcomeKey(subject="w", kind="picture", source="wikimedia"),
                "w", {"kind": "picture", "subject_kind": "word", "source": "wikimedia"},
                {"outcome": "candidates", "candidates": ["s1"]}, 0)
    assert tried_urls(cache, "w", "picture", "wikimedia") == frozenset()


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


# --- latest_phrase / drafted_phrase (spec 3 section 5's picture query) -----

def test_drafted_phrase_reads_the_newest_phrase_provide_row(cache):
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "bowl of rice"}, 0)
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "steamed jasmine rice"}, 0)
    assert drafted_phrase(cache.assessments_of("rice")) == "steamed jasmine rice"


def test_drafted_phrase_is_none_with_no_phrase_row_on_record(cache):
    assert drafted_phrase(cache.assessments_of("rice")) is None


def test_drafted_phrase_ignores_a_row_of_a_different_provides(cache):
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="rice"),
                "rice", {"provides": "picture", "kind": "picture"}, {"items": []}, 0)
    assert drafted_phrase(cache.assessments_of("rice")) is None


def test_latest_phrase_prefers_the_latest_learner_direction(cache):
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "bowl of rice"}, 0)
    cache.append("assess", "learner",
                DirectionKey(subject="rice", role="picture-for-word", text_sha=sha("try red")),
                "rice", {"kind": "direction", "role": "picture-for-word"},
                {"direction": "try red"}, 0)
    assert latest_phrase(cache.assessments_of("rice")) == "try red"


def test_latest_phrase_prefers_a_suggestion_newer_than_the_last_provide(cache):
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "bowl of rice"}, 0)
    cache.append("assess", "judge", JudgeKey.for_rule(None, None, "rice", "picture-for-word"),
                "rice", {"role": "picture-for-word", "kind": "picture"},
                {"value": False, "suggestion": "a bowl of steamed jasmine rice"}, 0)
    assert latest_phrase(cache.assessments_of("rice")) == "a bowl of steamed jasmine rice"


def test_latest_phrase_prefers_a_suggestion_older_than_a_later_phrase_row(cache):
    """Fix round 2 finding 1: "the last provide row" means the last
    Source ask -- the search that produced the judged candidate -- never
    attempts.phrase_attempt's own per-subject phrase row (an answer, not
    an ask). A phrase drafted after a pending suggestion must not make
    that suggestion look stale."""
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="rice"),
                "rice", {"kind": "picture", "params": {"query": "rice"}}, {"items": []}, 0)
    cache.append("assess", "judge", JudgeKey.for_rule(None, None, "rice", "picture-for-word"),
                "rice", {"role": "picture-for-word", "kind": "picture"},
                {"value": False, "suggestion": "a bowl of steamed jasmine rice"}, 0)
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "bowl of rice"}, 0)
    assert latest_phrase(cache.assessments_of("rice")) == "a bowl of steamed jasmine rice"


def test_last_source_ask_ts_ignores_a_phrase_row(cache):
    cache.append("provide", "openverse", ProvideKey(source="openverse", kind="", query="rice"),
                "rice", {"kind": "picture", "params": {"query": "rice"}}, {"items": []}, 0, ts=1)
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "bowl of rice"}, 0, ts=2)
    assert last_source_ask_ts(cache.assessments_of("rice")) == 1


def test_last_source_ask_ts_is_minus_one_with_no_source_ask(cache):
    assert last_source_ask_ts([]) == -1


def test_latest_phrase_falls_back_to_the_drafted_phrase(cache):
    cache.append("provide", "llm", PhraseKey(subject="rice"), "rice",
                {"provides": "phrase", "kind": "picture", "subject_kind": "word"},
                {"phrase": "bowl of rice"}, 0)
    assert latest_phrase(cache.assessments_of("rice")) == "bowl of rice"


def test_latest_phrase_is_none_with_nothing_on_record(cache):
    assert latest_phrase(cache.assessments_of("rice")) is None


# --- parse_phrases: the phrase drafter's own answer shape (spec 3 s5) ------

def test_parse_phrases_reads_the_subject_to_phrase_map():
    text = json.dumps({"phrases": [{"subject": "rice", "phrase": "bowl of rice"},
                                   {"subject": "s1", "phrase": "a busy street market"}]})
    assert parse_phrases(text) == {"rice": "bowl of rice", "s1": "a busy street market"}


def test_parse_phrases_skips_an_item_lacking_a_subject_or_a_phrase():
    text = json.dumps({"phrases": [{"phrase": "no subject"}, {"subject": "rice", "phrase": ""},
                                   {"subject": "eat"}]})
    assert parse_phrases(text) == {}


def test_parse_phrases_is_empty_for_text_that_is_not_the_phrase_json():
    assert parse_phrases("not json") == {}
    assert parse_phrases(json.dumps({"sentences": []})) == {}


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


def test_fetches_since_counts_the_bytes_fetches_attributed_to_a_source(cache):
    """Forvo counts an mp3 download as a request against its daily limit
    (measured 2026-09-10: 232 lookups plus 268 downloads exhausted it), so
    a per-day budget in Forvo's currency sums the audiofetch rows whose
    params name forvo as the source, alongside the lookups."""
    forvo_mp3 = {"kind": "recording", "params": {"source": "forvo", "url": "https://f/a.mp3"}}
    cache.append("provide", "audiofetch", ProvideKey(source="", kind="", query="k1"),
                 "rice", forvo_mp3, {"items": []}, 0.0, ts=100)
    cache.append("provide", "audiofetch", ProvideKey(source="", kind="", query="k2"),
                 "fish", forvo_mp3, {"items": []}, 0.0, ts=300)
    cache.append("provide", "audiofetch", ProvideKey(source="", kind="", query="k3"),
                 "rice", {"kind": "recording", "params": {"url": "https://x/b.mp3"}},
                 {"items": []}, 0.0, ts=300)   # a fetch from elsewhere
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="rice"),
                "rice", {"kind": "recording"}, {"items": []}, 1.0, ts=300)   # a lookup
    assert fetches_since(cache, "forvo", 200) == 1
    assert fetches_since(cache, "forvo", 0) == 2


def test_spend_since_sums_the_cost_of_those_asks(cache):
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="rice"),
                "rice", {"kind": "recording"}, {"items": []}, 1.0, ts=100)
    cache.append("provide", "forvo", ProvideKey(source="forvo", kind="", query="fish"),
                "fish", {"kind": "recording"}, {"items": []}, 2.5, ts=300)
    assert spend_since(cache, "forvo", 200) == pytest.approx(2.5)
    assert spend_since(cache, "forvo", 0) == pytest.approx(3.5)


def test_cost_since_sums_a_ports_own_backend_rows_with_no_source_ask_filter(cache):
    """cli `run --spend-cap` (spec 3 section 7) needs a judge verdict's
    own cost, appended at port "assess" (assessor.py's _append_verdict)
    -- spend_since only ever reads "provide" rows, so it always reads
    0.0 for the judge. cost_since reads any (port, backend) pair and
    sums every row's cost as logged, with no Source-ask filter.
    """
    cache.append("assess", "judge",
                 JudgeKey(rubric_sha="r", subject="rice", identity="", role="picture-for-word"),
                 "rice", {"kind": "picture"}, {"value": True}, 0.5, ts=100)
    cache.append("provide", "tts", ProvideKey(source="tts", kind="", query="rice"),
                "rice", {"kind": "recording"}, {"items": []}, 0.25, ts=100)
    total = cost_since(cache, "assess", "judge", 0) + cost_since(cache, "provide", "tts", 0)
    assert total == pytest.approx(0.75)


# --- parse_drafts: clauses alongside text and gloss -------------------------

def test_parse_drafts_reads_clauses_alongside_text_and_gloss():
    text = json.dumps({"sentences": [
        {"clauses": [["eat"], ["rice"]], "text": " กินข้าว ",
         "gloss": "eat rice"}]})   # กินข้าว: eat rice
    drafts = parse_drafts(text)
    assert len(drafts) == 1
    assert drafts[0].clauses == ((WordId("eat"),), (WordId("rice"),))
    assert drafts[0].text == "กินข้าว"     # กินข้าว: eat rice, whitespace stripped
    assert drafts[0].gloss == "eat rice"


def test_parse_drafts_reads_a_repeated_word_element():
    text = json.dumps({"sentences": [
        {"clauses": [[["red", "ๆ"]]], "text": "แดงๆ", "gloss": "very red"}]})   # แดงๆ: very red
    drafts = parse_drafts(text)
    assert drafts[0].clauses == (((WordId("red"), "ๆ"),),)


def test_parse_drafts_skips_an_item_lacking_clauses():
    text = json.dumps({"sentences": [{"text": "กินข้าว", "gloss": "eat rice"}]})   # กินข้าว: eat rice
    assert parse_drafts(text) == []


def test_parse_drafts_skips_an_item_lacking_text():
    text = json.dumps({"sentences": [{"clauses": [["eat"]], "gloss": "eat"}]})
    assert parse_drafts(text) == []


def test_parse_drafts_skips_a_malformed_clause_and_warns(caplog):
    """An empty clause list is malformed (entities.clauses_from_json
    raises); the item is skipped and a warning names the text's first 40
    characters."""
    text = json.dumps({"sentences": [
        {"clauses": [["eat"], []], "text": "กินข้าว", "gloss": "eat rice"}]})   # กินข้าว: eat rice
    with caplog.at_level(logging.WARNING):
        drafts = parse_drafts(text)
    assert drafts == []
    assert any("กินข้าว"[:40] in r.message for r in caplog.records)   # กินข้าว: eat rice


def test_parse_drafts_skips_an_item_whose_clauses_are_an_empty_list_and_warns(caplog):
    """"clauses": [] carries the key -- distinct from a missing key --
    but names no clause at all; entities.clauses_from_json refuses it, and
    the item is skipped with the same malformed warning."""
    text = json.dumps({"sentences": [
        {"clauses": [], "text": "กินข้าว", "gloss": "eat rice"}]})   # กินข้าว: eat rice
    with caplog.at_level(logging.WARNING):
        drafts = parse_drafts(text)
    assert drafts == []
    assert any("กินข้าว"[:40] in r.message for r in caplog.records)   # กินข้าว: eat rice


# --- parse_no_fit: the drafter's own "nothing fits" answer (spec 3 r19 s5) --

def test_parse_no_fit_reads_the_reason_of_an_empty_sentences_answer():
    text = json.dumps({"sentences": [], "reason": "no natural sentence covers ข้าว"})
    assert parse_no_fit(text) == "no natural sentence covers ข้าว"   # ข้าว: rice


def test_parse_no_fit_reads_a_fenced_answer():
    text = '```json\n' + json.dumps({"sentences": [], "reason": "vocabulary too small"}) + '\n```'
    assert parse_no_fit(text) == "vocabulary too small"


def test_parse_no_fit_is_none_when_the_answer_carries_a_draft():
    text = json.dumps({"sentences": [{"clauses": [["eat"]], "text": "กิน", "gloss": "eat"}],
                       "reason": "only one"})   # กิน: eat
    assert parse_no_fit(text) is None


def test_parse_no_fit_is_none_without_a_non_empty_reason():
    assert parse_no_fit(json.dumps({"sentences": []})) is None
    assert parse_no_fit(json.dumps({"sentences": [], "reason": "   "})) is None
    assert parse_no_fit(json.dumps({"sentences": [], "reason": 7})) is None


def test_parse_no_fit_is_none_on_text_that_is_not_the_drafting_json():
    assert parse_no_fit("sorry, I cannot") is None
    assert parse_no_fit(json.dumps(["a", "b"])) is None


def test_drafts_in_keeps_distinct_texts_distinct():
    text = json.dumps({"sentences": [
        {"clauses": [["eat"], ["rice"]], "text": "กินข้าว", "gloss": "eat rice"},
        {"clauses": [["rice"], ["tasty"]], "text": "ข้าวอร่อย",
         "gloss": "tasty rice"}]})   # กินข้าว: eat rice, ข้าวอร่อย: tasty rice
    drafts = drafts_in(text)
    assert [d.text for d in drafts] == ["กินข้าว", "ข้าวอร่อย"]   # eat rice, tasty rice


# --- merge_drafts: one draft per distinct text -------------------------------

def test_merge_drafts_keeps_a_text_whose_repeated_clauses_agree():
    """A text listed twice with the same clauses is one draft; the first
    non-empty gloss wins."""
    a = SentenceDraft(clauses=((WordId("eat"),), (WordId("rice"),)), text="กินข้าว",   # กินข้าว: eat rice
                      gloss="")
    b = SentenceDraft(clauses=((WordId("eat"),), (WordId("rice"),)), text="กินข้าว",
                      gloss="eat rice")
    merged = merge_drafts([a, b])
    assert len(merged) == 1
    assert merged[0].clauses == a.clauses
    assert merged[0].gloss == "eat rice"


def test_merge_drafts_keeps_the_first_gloss_when_glosses_disagree_and_logs_at_debug(caplog):
    """Spec 3 r19 section 5: a text listed twice with differing non-empty
    glosses is one candidate, keyed by the text -- the verdict was given
    on the first gloss, so that gloss stands; the disagreement is logged
    at DEBUG, not dropped."""
    a = SentenceDraft(clauses=((WordId("eat"),), (WordId("rice"),)), text="กินข้าว",   # กินข้าว: eat rice
                      gloss="eat rice")
    b = SentenceDraft(clauses=((WordId("eat"),), (WordId("rice"),)), text="กินข้าว",
                      gloss="rice is eaten")
    with caplog.at_level(logging.DEBUG):
        merged = merge_drafts([a, b])
    assert len(merged) == 1
    assert merged[0].gloss == "eat rice"
    assert any("กินข้าว"[:40] in r.message and r.levelno == logging.DEBUG
              for r in caplog.records)


def test_merge_drafts_drops_a_text_whose_clauses_disagree_and_warns(caplog):
    """Spec 3 r16: differing clauses for one text drop it, the same as a
    gloss conflict -- two answers cannot both be the sentence's own
    parse."""
    a = SentenceDraft(clauses=((WordId("eat"),),), text="กินข้าว", gloss="eat rice")   # กินข้าว: eat rice
    b = SentenceDraft(clauses=((WordId("eat"),), (WordId("rice"),)), text="กินข้าว",
                      gloss="eat rice")
    with caplog.at_level(logging.WARNING):
        merged = merge_drafts([a, b])
    assert merged == []
    assert any("กินข้าว"[:40] in r.message for r in caplog.records)   # กินข้าว: eat rice


def test_merge_drafts_merges_a_text_across_two_lists_handed_together():
    """merge_drafts merges over whatever `drafts` it is given, not just
    one item's own listings -- spec 3 section 5's "a text listed twice is
    one candidate" holds across items, not per item."""
    clauses = ((WordId("eat"),), (WordId("rice"),))
    from_item_a = [SentenceDraft(clauses=clauses, text="กินข้าว", gloss="")]   # กินข้าว: eat rice
    from_item_b = [SentenceDraft(clauses=clauses, text="กินข้าว", gloss="eat rice")]
    merged = merge_drafts(from_item_a + from_item_b)
    assert len(merged) == 1
    assert merged[0].gloss == "eat rice"


def test_sentence_drafts_merges_a_text_split_across_one_row_s_items(cache):
    """One provide row's own two items listing the same text with
    agreeing clauses fold to one draft -- the same merge
    `sentence_attempt` applies within a run."""
    cache.append("provide", "llm-sentence", ProvideKey(source="llm-sentence", kind="", query="q"),
                DRAFT_SUBJECT, {"kind": "sentence", "subject_kind": "sentence"},
                {"items": [
                    '{"sentences": [{"clauses": [["eat"], ["rice"]], "text": "กินข้าว",'
                    ' "gloss": ""}]}',   # กินข้าว: eat rice
                    '{"sentences": [{"clauses": [["eat"], ["rice"]], "text": "กินข้าว",'
                    ' "gloss": "eat rice"}]}']}, 0)
    drafts = sentence_drafts(cache)
    assert len(drafts) == 1
    assert drafts[0].gloss == "eat rice"


def test_sentence_drafts_merges_a_text_split_across_two_provide_rows(cache):
    """Two separate provide rows each listing the same text with
    agreeing clauses fold to one draft -- a text drafted in two runs is
    still one draft."""
    cache.append("provide", "llm-sentence", ProvideKey(source="llm-sentence", kind="", query="q1"),
                DRAFT_SUBJECT, {"kind": "sentence", "subject_kind": "sentence"},
                {"items": [
                    '{"sentences": [{"clauses": [["eat"], ["rice"]], "text": "กินข้าว",'
                    ' "gloss": ""}]}']}, 0, ts=100)   # กินข้าว: eat rice
    cache.append("provide", "llm-sentence", ProvideKey(source="llm-sentence", kind="", query="q2"),
                DRAFT_SUBJECT, {"kind": "sentence", "subject_kind": "sentence"},
                {"items": [
                    '{"sentences": [{"clauses": [["eat"], ["rice"]], "text": "กินข้าว",'
                    ' "gloss": "eat rice"}]}']}, 0, ts=200)
    drafts = sentence_drafts(cache)
    assert len(drafts) == 1
    assert drafts[0].gloss == "eat rice"


# --- draft_sentence: a SentenceDraft as a Sentence value ---------------------

def test_draft_sentence_carries_the_drafts_own_clauses_learner_voice_and_provenance():
    draft = SentenceDraft(clauses=((WordId("eat"),),), text="กิน", gloss="eat")   # กิน: eat
    sentence = draft_sentence(draft, lambda: date(2026, 9, 9))
    assert sentence.clauses == draft.clauses
    assert sentence.text == "กิน" and sentence.gloss == "eat"
    assert sentence.voice == "learner_voice"
    assert sentence.provenance.source == "llm" and sentence.provenance.origin == "draft"
    assert sentence.provenance.acquired == date(2026, 9, 9)


# --- vocabulary_line / parse_prompt / parses_in (spec 3 r16 section 5) -----

def test_vocabulary_line_carries_no_leading_marker():
    w = word("eat", "กิน", "eat")   # กิน: eat
    assert vocabulary_line(w) == "eat  กิน  (eat)"


def test_parse_prompt_carries_every_vocabulary_id_and_every_text():
    vocabulary = [word("eat", "กิน", "eat"), word("rice", "ข้าว", "rice")]   # กิน: eat, ข้าว: rice
    texts = ["กินข้าว", "ข้าวอร่อย"]   # eat rice, tasty rice
    prompt = parse_prompt(texts, vocabulary)
    for w in vocabulary:
        assert w.id in prompt and w.thai in prompt
    for t in texts:
        assert t in prompt


def test_parse_subject_names_the_migration_worklist():
    assert PARSE_SUBJECT == "sentence-parses"


def test_parses_in_returns_the_text_to_clauses_map():
    text = json.dumps({"parses": [
        {"text": "กินข้าว", "clauses": [["eat"], ["rice"]]},   # กินข้าว: eat rice
        {"text": "ข้าวอร่อย", "clauses": [["rice"], ["tasty"]]}]})   # ข้าวอร่อย: tasty rice
    parses = parses_in(text)
    assert parses == {
        "กินข้าว": ((WordId("eat"),), (WordId("rice"),)),
        "ข้าวอร่อย": ((WordId("rice"),), (WordId("tasty"),))}


def test_parses_in_skips_an_entry_lacking_text_or_clauses():
    text = json.dumps({"parses": [
        {"clauses": [["eat"]]},
        {"text": "กินข้าว"}]})   # กินข้าว: eat rice
    assert parses_in(text) == {}


def test_parses_in_skips_a_malformed_clause_and_warns(caplog):
    text = json.dumps({"parses": [
        {"text": "กินข้าว", "clauses": [["eat"], []]}]})   # กินข้าว: eat rice, empty clause: invalid
    with caplog.at_level(logging.WARNING):
        parses = parses_in(text)
    assert parses == {}
    assert any("กินข้าว"[:40] in r.message for r in caplog.records)   # กินข้าว: eat rice


def test_parses_in_skips_an_entry_whose_clauses_are_an_empty_list_and_warns(caplog):
    """"clauses": [] carries the key -- distinct from a missing key --
    but names no clause at all; entities.clauses_from_json refuses it, and
    the entry is skipped with the same malformed warning."""
    text = json.dumps({"parses": [{"text": "กินข้าว", "clauses": []}]})   # กินข้าว: eat rice
    with caplog.at_level(logging.WARNING):
        parses = parses_in(text)
    assert parses == {}
    assert any("กินข้าว"[:40] in r.message for r in caplog.records)   # กินข้าว: eat rice


def test_parses_in_is_empty_for_text_that_is_not_the_parsing_json():
    assert parses_in("not json") == {}
    assert parses_in(json.dumps({"sentences": []})) == {}


# --- retired_texts (F13, spec 3 section 5) ----------------------------------

def test_retired_texts_names_every_subject_with_a_retirement_row(cache):
    sha1, sha2 = "e" * 64, "f" * 64
    cache.append("attempt", "run", RetirementKey(sha1), sha1,
                {"kind": "retirement", "subject_kind": "sentence",
                 "reason": "recording exhausted", "candidates": 2},
                {"retired": True}, 0)
    assert retired_texts(cache) == frozenset({sha1})
    cache.append("attempt", "run", RetirementKey(sha2), sha2,
                {"kind": "retirement", "subject_kind": "sentence",
                 "reason": "recording exhausted", "candidates": 0},
                {"retired": True}, 0)
    assert retired_texts(cache) == frozenset({sha1, sha2})


def test_retired_texts_ignores_an_attempt_run_row_of_a_different_kind(cache):
    cache.append("attempt", "run", RetirementKey("e" * 64), "e" * 64,
                {"kind": "something-else", "subject_kind": "sentence"},
                {"retired": True}, 0)
    assert retired_texts(cache) == frozenset()
