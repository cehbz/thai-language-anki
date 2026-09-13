"""learner.py: the learner backend's row writers, one shape per act,
shared by the feedback screen and the comment pass."""
import pytest

from thai_syllabus import learner
from thai_syllabus.cachekeys import CommentVetoKey, DirectionKey, LearnerKey, sha
from thai_syllabus.store import SyllabusDb


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


def test_append_comment_on_a_question_records_the_question_and_its_subject_kind(db):
    """Spec 5 r9: a session comment is the card-flag row shape with the
    question kind, the artifact kind and the subject kind in its
    question, anchored on the subject under card_kind "question"."""
    shown = {"picture": "a" * 64, "recordings": [], "text_sha": None}
    learner.append_comment(db, subject="rice", card_id="rice", kind="question",
                           text="wrong picture", shown=shown, subject_kind="word",
                           question_kind="rate", artifact_kind="picture")
    (row,) = db.assessments_of("rice")
    assert row.backend == "learner" and row.port == "assess"
    assert row.question == {"role": "card-flag", "kind": "card-flag", "anchor": "rice",
                            "card_kind": "question", "shown": shown, "subject_kind": "word",
                            "question_kind": "rate", "artifact_kind": "picture"}
    assert row.answer == {"kind": "rating", "rating": None, "note": "wrong picture"}
    # the gallery note's own key shape: LearnerKey(card_id, "card-flag")
    assert db.latest("assess", "learner", LearnerKey(artifact_sha="rice", role="card-flag")) is not None


def test_append_comment_on_a_gallery_card_omits_the_question_fields(db):
    learner.append_comment(db, subject="rice", card_id="rice", kind="reading",
                           text="clear", shown={}, subject_kind="word")
    (row,) = db.assessments_of("rice")
    assert "question_kind" not in row.question and "artifact_kind" not in row.question
    assert row.question["subject_kind"] == "word"


def test_append_rating_writes_the_screens_rating_row_and_marks_a_derived_one(db):
    learner.append_rating(db, subject="rice", role="picture-for-word", rating="good",
                          artifact_sha="a" * 64, note="fine")
    learner.append_rating(db, subject="rice", role="picture-for-word", rating="unacceptable-none",
                          artifact_sha="a" * 64,
                          derived_from=learner.CommentRef("c1c1c1c1c1c1c1c1", "1"))
    plain, derived = db.assessments_of("rice")
    assert plain.question == {"role": "picture-for-word", "artifact_sha": "a" * 64,
                              "rubric": None, "kind": "rating", "subject_kind": "word"}
    assert plain.answer == {"value": "good", "note": "fine"}
    assert derived.question["comment_sha"] == "c1c1c1c1c1c1c1c1"
    assert derived.question["prompt_version"] == "1"
    assert derived.answer == {"value": "unacceptable-none"}


def test_append_rating_refuses_a_value_outside_the_vocabulary(db):
    with pytest.raises(ValueError, match="unknown rating"):
        learner.append_rating(db, subject="rice", role="picture-for-word", rating="meh",
                              artifact_sha=None)


def test_append_direction_writes_the_typed_direction_row(db):
    learner.append_direction(db, subject="rice", role="picture-for-word",
                             text="a bowl of steamed rice",
                             derived_from=learner.CommentRef("c1c1c1c1c1c1c1c1", "1"))
    (row,) = db.assessments_of("rice")
    assert row.question == {"kind": "direction", "role": "picture-for-word",
                            "subject_kind": "word", "comment_sha": "c1c1c1c1c1c1c1c1",
                            "prompt_version": "1"}
    assert row.answer == {"direction": "a bowl of steamed rice"}
    assert db.latest("assess", "learner", DirectionKey(subject="rice", role="picture-for-word",
                                                       text_sha=sha("a bowl of steamed rice"))) is not None


def test_append_comment_veto_writes_the_veto_row(db):
    """Spec 5 r10: the learner's strike on one reading is a learner row
    under the comment's own subject, keyed by the reading it strikes --
    record.vetoed_readings reads the (comment_sha, prompt_version) pair
    straight off its question."""
    learner.append_comment_veto(db, subject="rice", comment_sha="c1c1c1c1c1c1c1c1",
                                prompt_version="1", subject_kind="word")
    (row,) = db.assessments_of("rice")
    assert row.backend == "learner" and row.port == "assess"
    assert row.question == {"kind": "comment-veto", "comment_sha": "c1c1c1c1c1c1c1c1",
                            "prompt_version": "1", "subject_kind": "word"}
    assert row.answer == {"vetoed": True}
    assert db.latest("assess", "learner",
                     CommentVetoKey(comment_sha="c1c1c1c1c1c1c1c1",
                                    prompt_version="1")) is not None


def test_append_comment_veto_defaults_the_subject_kind_to_word(db):
    learner.append_comment_veto(db, subject="rice", comment_sha="c1c1c1c1c1c1c1c1",
                                prompt_version="1")
    (row,) = db.assessments_of("rice")
    assert row.question["subject_kind"] == "word"
