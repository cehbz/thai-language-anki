"""The learner backend's row writers (spec 3 section 4: the learner is a
backend of both ports; spec 5: every answer appends one learner row).
One function per act, each writing the one row shape every fold over
the record already reads: the feedback screen (reviewserver) calls them
for the learner's own acts, and the comment pass (attempts.comment_attempt)
calls them on the learner's behalf, marking each row with the comment
it was read from (`derived_from`), which is how a struck reading's rows
are found again (record.without_vetoed_readings).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

from .cachekeys import CommentVetoKey, DirectionKey, LearnerKey, sha
from .ports import RecordWriter
from .record import LEARNER_RANK

__all__ = ["ACTION_RATINGS", "CommentRef", "append_comment", "append_comment_veto",
           "append_rating", "append_direction"]

# action 1-4 (spec 5 section 1 kind 1) -> the learner rating vocabulary
# derivations.py's current_best/exhausted fold over (LEARNER_RANK).
ACTION_RATINGS: dict[int, str] = {
    1: "unacceptable-none",
    2: "unacceptable-use-this",
    3: "acceptable",
    4: "good",
}


class CommentRef(NamedTuple):
    """The reading a derived row came from: the comment's identity
    (cachekeys.comment_identity) and the prompt version that read it."""
    comment_sha: str
    prompt_version: str


def _derived(question: dict[str, Any], derived_from: CommentRef | None) -> dict[str, Any]:
    if derived_from is not None:
        question["comment_sha"] = derived_from.comment_sha
        question["prompt_version"] = derived_from.prompt_version
    return question


def append_comment(record: RecordWriter, *, subject: str, card_id: str, kind: str, text: str,
                   shown: Mapping[str, Any] | None = None, subject_kind: str | None = None,
                   question_kind: str | None = None, artifact_kind: str | None = None) -> int:
    """One comment (spec 5 r9) as a card-level flag row (spec 4 section
    4's shape) under role "card-flag": `subject` is the card's entity
    subject, `card_id` the per-card anchor, `kind` its card_kind -- or,
    for a session question, `card_id` = the subject, `kind` "question",
    with `question_kind` (rate|direction|challenger|reask) and
    `artifact_kind` (the question's kind). `shown` records the card as
    shown (spec 5 r5); {} when omitted.
    """
    role = "card-flag"
    question: dict[str, Any] = {"role": role, "kind": "card-flag", "anchor": str(card_id),
                                "card_kind": kind,
                                "shown": dict(shown) if shown is not None else {}}
    if subject_kind is not None:
        question["subject_kind"] = subject_kind
    if question_kind is not None:
        question["question_kind"] = question_kind
    if artifact_kind is not None:
        question["artifact_kind"] = artifact_kind
    return record.append(port="assess", backend="learner",
                         key=LearnerKey(artifact_sha=str(card_id), role=role), subject=str(subject),
                         question=question,
                         answer={"kind": "rating", "rating": None, "note": text})


def append_rating(record: RecordWriter, *, subject: str, role: str, rating: str,
                  artifact_sha: str | None, subject_kind: str = "word", note: str | None = None,
                  derived_from: CommentRef | None = None) -> int:
    """One rating row (spec 3 section 4's learner key), the shape the
    screen's 1-4 answers write. Refuses a value outside LEARNER_RANK."""
    if rating not in LEARNER_RANK:
        raise ValueError(f"unknown rating {rating!r}")
    answer: dict[str, Any] = {"value": rating}
    if note:
        answer["note"] = note
    question = _derived({"role": role, "artifact_sha": artifact_sha, "rubric": None,
                         "kind": "rating", "subject_kind": subject_kind}, derived_from)
    return record.append(port="assess", backend="learner",
                         key=LearnerKey(artifact_sha=artifact_sha, role=role), subject=subject,
                         question=question, answer=answer)


def append_direction(record: RecordWriter, *, subject: str, role: str, text: str,
                     subject_kind: str = "word", derived_from: CommentRef | None = None) -> int:
    """One typed direction row (spec 5 section 1 kind 2: a direction, not
    a rating)."""
    question = _derived({"kind": "direction", "role": role, "subject_kind": subject_kind},
                        derived_from)
    return record.append(port="assess", backend="learner",
                         key=DirectionKey(subject=subject, role=role, text_sha=sha(text)),
                         subject=subject, question=question, answer={"direction": text})


def append_comment_veto(record: RecordWriter, *, subject: str, comment_sha: str,
                        prompt_version: str, subject_kind: str = "word") -> int:
    """The learner struck one reading (spec 5 r10): every row derived
    from it is ignored by every fold from here on
    (record.without_vetoed_readings for the subject's own rows,
    record.vetoed_readings_all for the ones written under another
    subject). The reading row itself stays, read back as vetoed
    (record.reading_view). Idempotent in effect: a second strike appends
    a second row naming the same reading, which strikes the same thing.
    """
    return record.append(port="assess", backend="learner",
                         key=CommentVetoKey(comment_sha=comment_sha,
                                            prompt_version=prompt_version),
                         subject=subject,
                         question={"kind": "comment-veto", "comment_sha": comment_sha,
                                   "prompt_version": prompt_version,
                                   "subject_kind": subject_kind},
                         answer={"vetoed": True})
