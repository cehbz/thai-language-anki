"""Check requests (spec 5 r22): an artifact a run or a session puts to the
learner with a note saying what to look at. A request is one row under
its subject (cachekeys.CheckRequestKey); the learner's answer is a rating
row naming the request's identity, which closes it. Whether its artifact
is still on a card of its subject is the review screen's to fold
(reviewserver.build_queue), from the compiled cards.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cachekeys import CheckRequestKey, check_request_identity, sha
from .ids import PairId
from .ports import CacheReader, RecordWriter

__all__ = ["CheckRequest", "resolve", "append_check_request", "check_requests",
           "is_answered", "open_check_requests"]

_PORT, _BACKEND, _KIND = "assess", "check-request", "check-request"


@dataclass(frozen=True)
class CheckRequest:
    """One check request: the subject (a word id, a sentence text_sha, a
    pair id), the artifact (a picture or recording sha, or a pair's
    rendition identity) with its kind, and the note. `identity` and `ts`
    are its row's, None until it is on record.
    """
    subject: str
    artifact_sha: str
    artifact_kind: str
    subject_kind: str
    note: str
    identity: str | None = None
    ts: int | None = None


def _subject_kind(syllabus, subject: str) -> str:
    if syllabus.find_word(subject) is not None:
        return "word"
    try:
        syllabus.pair(PairId(subject))
        return "pair"
    except KeyError:
        pass
    try:
        syllabus.sentence(subject)
        return "sentence"
    except KeyError:
        pass
    raise ValueError(f"no word, sentence or pair {subject!r} in the syllabus")


def resolve(syllabus, cache: CacheReader, *, subject: str, artifact_sha: str,
            note: str) -> CheckRequest:
    """The request `subject`, `artifact_sha` and `note` name, its kinds
    read off the syllabus (what the subject is) and the media table (what
    the artifact is; a pair's artifact is its rendition). Raises
    ValueError naming what it could not find.
    """
    subject_kind = _subject_kind(syllabus, subject)
    if subject_kind == "pair":
        artifact_kind = "rendition"
    else:
        prov = cache.media_provenance(artifact_sha)
        if prov is None or prov.get("kind") not in ("picture", "recording"):
            raise ValueError(f"no picture or recording {artifact_sha} in the media table")
        artifact_kind = prov["kind"]
    return CheckRequest(subject=subject, artifact_sha=artifact_sha, artifact_kind=artifact_kind,
                        subject_kind=subject_kind, note=note)


def append_check_request(record: RecordWriter, request: CheckRequest) -> int:
    """Appends `request` under its subject; returns the row's ts."""
    return record.append(port=_PORT, backend=_BACKEND,
                         key=CheckRequestKey(subject=request.subject,
                                             artifact_sha=request.artifact_sha,
                                             note_sha=sha(request.note)),
                         subject=request.subject,
                         question={"kind": _KIND, "artifact_sha": request.artifact_sha,
                                   "artifact_kind": request.artifact_kind,
                                   "subject_kind": request.subject_kind},
                         answer={"note": request.note})


def check_requests(cache: CacheReader) -> list[CheckRequest]:
    """Every check request on record, oldest first."""
    return [CheckRequest(subject=r.subject, artifact_sha=r.question["artifact_sha"],
                         artifact_kind=r.question["artifact_kind"],
                         subject_kind=r.question["subject_kind"], note=r.answer["note"],
                         identity=check_request_identity(r.key_sha, r.ts), ts=r.ts)
            for r in cache.rows_since(_PORT, _BACKEND, 0)
            if r.question.get("kind") == _KIND]


def is_answered(cache: CacheReader, request: CheckRequest) -> bool:
    """True once a learner row under the request's subject names it."""
    return any(r.backend == "learner" and r.question.get("check_request") == request.identity
               for r in cache.assessments_of(request.subject))


def open_check_requests(cache: CacheReader) -> list[CheckRequest]:
    """Every check request with no answer, oldest first."""
    return [r for r in check_requests(cache) if not is_answered(cache, r)]
