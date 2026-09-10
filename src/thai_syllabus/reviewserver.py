"""Spec 5: the feedback screen -- the local surface where the learner
answers the system's questions and reviews the deck.

Presents the run's own derivations and records the learner's acts. Every
parameter a fold is measured under -- the wired Syllabus and its media
index, the deck's rubric, provenance prior, Source roster and attempt cap
-- arrives as one wiring.Derivations bundle, the same one
wiring.build_sourcing hands the run; this module derives nothing itself.

Writes are RecordWriter appends only: no curated data is edited (spec 5
section 4) and no judge or provider backend is called except the media
ingest a supplied URL goes through (spec 5 section 1 kind 2).

One process: `python -m thai_syllabus.reviewserver --deck DIR
[--port 8877]`, which `thai-syllabus review` wires.
"""
from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import re
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .authority import role_for
from .cachekeys import DirectionKey, DrillKey, LearnerKey, ProvideKey, RunReportKey, WaiverKey, sha
from .compile import CARD_CSS, build_deck, card_kind_of, field_values, render_card, tag_value
from .derivations import (
    DEFAULT_REASK_LAPSES,
    LEARNER_RANK,
    Challenger,
    CurrentBest,
    ExhaustedStatus,
    JudgeVerdict,
    QueueEntry,
    all_needs,
    available_needs,
    challengers,
    current_best,
    exhausted,
    judge_verdict,
    learner_ranks,
    queue,
    reasks,
    vetoed,
)
from .ids import PairId
from .media import Speaker
from .ports import Answer, CacheReader, RecordWriter, StudyReader
from .provider import FetchBackend, Provider, Question, tool_fetcher
from .record import (
    candidate_shas,
    card_flags,
    excluded_candidates,
    latest_query,
    latest_rating,
    rows_for,
    run_reports,
    source_asks,
)
from .run import LEARNER_DEFAULT_SESSION_BUDGET
from .store import MediaStore
from .syllabus import Syllabus

if TYPE_CHECKING:                       # wiring reaches for provider/assessor; the
    from .wiring import Derivations     # screen needs only the bundle's values.

__all__ = [
    "ReviewContext", "SessionStats", "build_app", "serve", "load_context", "main",
    "build_queue", "compiled_cards", "compute_stats",
    "append_answer", "append_supply", "append_gallery_note", "append_drill_result",
]

DEFAULT_PORT = 8877          # 8765 is reserved for AnkiConnect / proof_gallery.py

# The rank an artifact must reach to count as covered (spec 5 section 3's
# current-best coverage per need).
_ACCEPTABLE_FLOOR = LEARNER_RANK["acceptable"]

# action 1-4 (spec 5 section 1 kind 1) -> the learner rating vocabulary
# derivations.py's current_best/exhausted already fold over (LEARNER_RANK).
ACTION_RATINGS: dict[int, str] = {
    1: "unacceptable-none",
    2: "unacceptable-use-this",
    3: "acceptable",
    4: "good",
}


def _best(d: "Derivations", subject: str, kind: str) -> CurrentBest:
    return current_best(d.db, subject, kind, current_rubric=d.current_rubric, prior=d.prior,
                        provenance_source=d.provenance_source)


def _exhausted(d: "Derivations", subject: str, kind: str) -> ExhaustedStatus:
    return exhausted(d.db, subject, kind, sources=d.sources_for(kind),
                     attempt_cap=d.attempt_cap, transient_cap=d.transient_cap)


def _gloss_for(syllabus: Syllabus, subject: str, subject_kind: str = "word") -> str | None:
    """The English gloss a question shows beside its subject (spec 5
    section 1 kind 1): a sentence's own gloss, a pair's members' meanings
    joined, else a word's meaning. None when nothing matches.
    """
    if subject_kind == "sentence":
        try:
            return syllabus.sentence(subject).gloss
        except KeyError:
            return None
    if subject_kind == "pair":
        try:
            pair = syllabus.pair(PairId(subject))
        except KeyError:
            return None
        meanings = [w.meaning for m in pair.members if (w := syllabus.find_word(m)) is not None]
        return " / ".join(meanings) if meanings else None
    word = syllabus.find_word(subject)
    return word.meaning if word is not None else None


def _verdict_line(verdict: JudgeVerdict | None) -> str | None:
    """The one line spec 5 section 1 kind 1 shows beside the current
    artifact, or None when derivations.judge_verdict has nothing fresh.
    """
    if verdict is None:
        return None
    line = f"judge: {'pass' if verdict.passed else 'fail'}"
    return f"{line} — {verdict.evidence}" if verdict.evidence else line


def _artifact(sha: str | None) -> dict[str, str] | None:
    return {"sha": sha, "url": f"/media/{sha}"} if sha else None


# --- question session (spec 5 section 1) -----------------------------------

def _rate_question(d: "Derivations", subject: str, kind: str, subject_kind: str,
                   *, directed: bool = False, rank: float = 0.0,
                   attempts: int = 0) -> dict[str, Any]:
    rows = rows_for(d.db, subject, kind)
    role = role_for(kind, subject_kind)
    best = _best(d, subject, kind)
    current = _artifact(best.artifact_sha)
    if current is not None:
        current["verdict"] = _verdict_line(
            judge_verdict(d.db, subject, kind, best.artifact_sha,
                          current_rubric=d.current_rubric))
        current["source"] = best.source
    rejected = [_artifact(s) for s in candidate_shas(rows) if s != best.artifact_sha]
    return {
        "type": "rate", "subject": subject, "kind": kind, "subject_kind": subject_kind,
        "role": role,
        # spec 5 section 1 kind 1 (r8): whether this role's rating orders
        # current_best (picture/scene/sentence roles) or only vetoes
        # (recording and rendition roles) -- the client labels "acceptable"/
        # "good" a note and "unacceptable" a veto when this is False.
        "learner_ranks": learner_ranks(role),
        "gloss": _gloss_for(d.syllabus, subject, subject_kind), "query": latest_query(rows),
        "current": current, "rejected": rejected, "directed": directed,
        "rank": best.rank, "attempts": attempts,
        # spec 5 section 1 kind 1 / section 3: candidates the judge could
        # never even prepare (record.excluded_candidates, this run's own
        # RunReport row) and the subject's card-level flags (spec 4
        # section 4, record.card_flags) -- shown beside the rejected
        # thumbnails, never fetched or recomputed here.
        "excluded": excluded_candidates(d.db, subject),
        "flags": card_flags(d.db.assessments_of(subject)),
    }


def _tried_summary(rows: Sequence[Answer]) -> list[dict[str, Any]]:
    """What spec 5 section 1 kind 2 shows an exhausted subject as "what
    was tried": every Source ask under the need, as the backend that made
    it and the phrase or text it carried.
    """
    tried: list[dict[str, Any]] = []
    for ask in source_asks(rows):
        params = ask.question.get("params", {}) or {}
        tried.append({"source": ask.backend, "query": params.get("query") or params.get("text")})
    return tried


def _tried_candidates(d: "Derivations", subject: str, kind: str,
                      rows: Sequence[Answer]) -> list[dict[str, Any]]:
    """The first 5 candidates those asks produced, each with the judge's
    verdict on it under the need's fit role (pass/fail and the judge's
    evidence), or None where the judge has not spoken.
    """
    candidates: list[dict[str, Any]] = []
    for artifact_sha in candidate_shas(rows)[:5]:
        verdict = judge_verdict(d.db, subject, kind, artifact_sha,
                                current_rubric=d.current_rubric)
        candidates.append({
            "sha": artifact_sha,
            "verdict": {"passed": verdict.passed, "evidence": verdict.evidence}
                       if verdict is not None else None,
        })
    return candidates


def _direction_question(d: "Derivations", subject: str, kind: str, subject_kind: str,
                        attempts: int) -> dict[str, Any]:
    rows = rows_for(d.db, subject, kind)
    return {
        "type": "direction", "subject": subject, "kind": kind, "subject_kind": subject_kind,
        "role": role_for(kind, subject_kind), "gloss": _gloss_for(d.syllabus, subject, subject_kind),
        "tried": _tried_summary(rows), "candidates": _tried_candidates(d, subject, kind, rows),
        "attempts": attempts,
    }


def _challenger_question(d: "Derivations", challenger: Challenger) -> dict[str, Any]:
    return {
        "type": "challenger", "subject": challenger.subject, "kind": challenger.kind,
        "subject_kind": challenger.subject_kind,
        "role": role_for(challenger.kind, challenger.subject_kind),
        "gloss": _gloss_for(d.syllabus, challenger.subject, challenger.subject_kind),
        "current": _artifact(challenger.current_sha),
        "challenger": _artifact(challenger.challenger_sha),
    }


def _reask_questions(d: "Derivations", study: StudyReader) -> list[dict[str, Any]]:
    """Spec 5 section 1 kind 4: every derivations.reasks contradiction as
    a question. The lapse threshold is rulebook.yaml's own
    "reask/lapses" where the deck sets one, else DEFAULT_REASK_LAPSES.
    """
    threshold = int(d.thresholds.get("reask/lapses", DEFAULT_REASK_LAPSES))
    out: list[dict[str, Any]] = []
    for found in reasks(d.db, study, d.syllabus, lapse_threshold=threshold):
        best = _best(d, found.subject, found.kind)
        role = role_for(found.kind, found.subject_kind)
        out.append({
            "type": "reask", "subject": found.subject, "kind": found.kind,
            "subject_kind": found.subject_kind,
            "role": role,
            "learner_ranks": learner_ranks(role),
            "gloss": _gloss_for(d.syllabus, found.subject, found.subject_kind),
            "original_answer": found.rating,
            "current": _artifact(best.artifact_sha),
            "evidence": [{"anchor": r.anchor, "card_kind": r.card_kind, "grade": r.grade,
                         "ts": r.ts} for r in found.evidence[-5:]],
        })
    return out


def build_queue(d: "Derivations", study: StudyReader | None = None, *,
                budget: int) -> list[dict[str, Any]]:
    """The question session (spec 5 section 1): four kinds from
    derivations.py under `d`'s parameters, capped by the learner-attention
    budget (spec 3 section 7's "learner" Budget, ReviewContext's own
    learner_budget). The F10-ordered rate questions fill it first; direction
    requests, challenger comparisons and re-asks fill what is left. A
    kind with no derivation input yields no questions.
    """
    entries = queue(d.syllabus, d.db, current_rubric=d.current_rubric, prior=d.prior,
                    sources_for=d.sources_for, attempt_cap=d.attempt_cap,
                    transient_cap=d.transient_cap,
                    provenance_source=d.provenance_source)
    items = [
        _rate_question(d, e.subject, e.kind, e.subject_kind, directed=e.directed,
                       rank=e.rank, attempts=e.attempts)
        for e in entries
    ][:budget]
    # A need kept queued for a candidate awaiting a verdict under the
    # current rubric (derivations.queue's bucket 2) can also be exhausted
    # on attempts -- already rated above, it is skipped here so the
    # screen lists it once (spec 5 section 1).
    queued = {(e.subject, e.kind) for e in entries}

    if len(items) < budget:
        for subject, kind, subject_kind in available_needs(d.syllabus):
            if (subject, kind) in queued:
                continue
            status = _exhausted(d, subject, kind)
            if status.exhausted:
                items.append(_direction_question(d, subject, kind, subject_kind,
                                                 status.attempts))
                if len(items) >= budget:
                    break

    if len(items) < budget:
        for challenger in challengers(d.db, d.syllabus, current_rubric=d.current_rubric,
                                      prior=d.prior, provenance_source=d.provenance_source):
            items.append(_challenger_question(d, challenger))
            if len(items) >= budget:
                break

    if len(items) < budget and study is not None:
        items.extend(_reask_questions(d, study))

    return items[:budget]


# --- gallery data provider (spec 5 section 1) -------------------------------
#
# Renders exactly the notes compile.build_deck would write, through each
# note's own model template -- the gallery composes no card shape of its
# own (principles F4: the learner judges the artifact the card will show).

_MEDIA_IMG_RE = re.compile(r'<img src="([^".]+)\.[A-Za-z0-9]+">')
_MEDIA_SOUND_RE = re.compile(r'\[sound:([^.\]]+)\.[A-Za-z0-9]+\]')


def _resolve_media_for_web(html: str) -> str:
    """Rewrites compile.py's apkg-relative media references ("sha.ext")
    into this server's own /media/SHA route, over the same bytes.
    """
    html = _MEDIA_IMG_RE.sub(lambda m: f'<img src="/media/{m.group(1)}">', html)
    html = _MEDIA_SOUND_RE.sub(
        lambda m: f'<audio controls src="/media/{m.group(1)}"></audio>', html)
    return html


# Each family's entity-identity tag prefix (spec 4 section 2): the card
# dict's "subject" reads from this, so a minimal_pair card's subject is
# the pair id, where Built.subject is one member's MemberKey.
_ENTITY_TAG_PREFIX: dict[str, str] = {
    "word": "word", "minimal_pair": "pair", "grapheme": "grapheme", "sentence": "sentence",
}


def compiled_cards(d: "Derivations") -> list[dict[str, Any]]:
    """Every card compile.build_deck would compile, in its due order
    (spec 5 section 1). One entry per card: its kind (the template name),
    front/back HTML rendered through the note's own model template, the
    model's CSS, and metadata read from the note's fields and tags --
    `family`, `subject` (the entity id), `gloss`, and for a minimal_pair
    note `confusion` and `stimulus_member`, which the gallery's pair
    drill logs against.
    """
    built_deck = build_deck(d.syllabus, d.db, d.media_store, current_rubric=d.current_rubric,
                            prior=d.prior, provenance_source=d.provenance_source)
    ordered = sorted(built_deck.built, key=lambda item: item.base_due)

    cards: list[dict[str, Any]] = []
    for item in ordered:
        values = field_values(item.model, item.note)
        gloss = values.get("Meaning") or values.get("Gloss") or None
        entity_subject = tag_value(item.note, _ENTITY_TAG_PREFIX[item.family])
        for card in item.note.cards:
            template_name = item.model.templates[card.ord]["name"]
            kind = card_kind_of(template_name)
            front, back = render_card(item.model, item.note, card.ord)
            entry: dict[str, Any] = {
                "index": len(cards), "id": item.subject, "family": item.family,
                "kind": kind, "subject": entity_subject,
                "front_html": _resolve_media_for_web(front),
                "back_html": _resolve_media_for_web(back),
                "css": item.model.css, "gloss": gloss,
            }
            if item.family == "minimal_pair":
                member = tag_value(item.note, "member")
                entry["confusion"] = tag_value(item.note, "confusion")
                entry["stimulus_member"] = int(member) if member is not None else None
            cards.append(entry)
    return cards


# --- writes: notes, drills, answers, supply ---------------------------------

def append_gallery_note(record: RecordWriter, *, subject: str, card_id: str, kind: str,
                        text: str) -> int:
    """One gallery note as a card-level flag row (spec 4 section 4's
    shape; spec 5 section 1), under role "card-flag" (AUTHORITY_ORDER's
    learner-only role). `subject` is the card's own entity subject
    (card.subject: a word/pair/grapheme/sentence id); `card_id` is the
    row's own per-card anchor, `kind` its card_kind.
    """
    role = "card-flag"
    key = LearnerKey(artifact_sha=str(card_id), role=role)
    return record.append(port="assess", backend="learner", key=key, subject=str(subject),
                         question={"role": role, "kind": "card-flag",
                                  "anchor": str(card_id), "card_kind": kind},
                         answer={"kind": "rating", "rating": None, "note": text})


def append_drill_result(record: RecordWriter, *, confusion: str, pair_id: str,
                        correct: bool) -> int:
    """One pair-drill result as a learner evidence row in `cache` (spec 5
    section 1; the `study` table is imported Anki revlog only). Subject is
    the confusion id, so /stats folds a confusion's drills in one read.
    """
    key = DrillKey(pair_id=pair_id, confusion=confusion)
    return record.append(port="assess", backend="learner", key=key, subject=confusion,
                         question={"kind": "drill", "pair": pair_id, "confusion": confusion},
                         answer={"correct": bool(correct)})


def _rating_of(payload: Mapping[str, Any]) -> str | None:
    """The rating value append_answer would write for this payload,
    appending nothing. A challenger "keep" carries none; "switch"
    defaults to "acceptable"; anything else takes an explicit `rating`,
    else its `action` (1-4) through ACTION_RATINGS. None when neither is
    present.
    """
    action = payload.get("action")
    if action == "keep":
        return None
    if action == "switch":
        return payload.get("rating", "acceptable")
    if payload.get("rating"):
        return payload["rating"]
    return ACTION_RATINGS.get(int(action)) if action is not None else None


def append_answer(record: RecordWriter, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The one write path for question-session answers (spec 5 section 1):
    every answer appends one learner cache row keyed by cachekeys.LearnerKey
    (cachekeys.DirectionKey for a typed direction, cachekeys.WaiverKey for
    a waiver). `payload` shapes:
      rate/reask:  {subject, kind, action: 1-4, artifact_sha?, note?}
      challenger:  {subject, kind, action: "keep"|"switch", artifact_sha?}
      direction:   {subject, kind, direction: TEXT, subject_kind?}
      waiver:      {finding: {rule, note_id, artifact_sha?}, waived?, reason?}
    The role a rating or direction is filed under is the question's own
    (`role`), else the need's kinds (`kind` plus `subject_kind`, "word"
    by default). A rejection names the artifact it rejects in
    `payload["artifact_sha"]`; whether that is still current-best is the
    caller's check (build_app's /api/answer handler). Append-only: two
    identical payloads append two rows, and every fold is newest-wins.
    """
    if "finding" in payload:
        finding = payload["finding"]
        rule_id, note_id = finding["rule"], finding["note_id"]
        artifact_sha = finding.get("artifact_sha")
        key = WaiverKey(rule_id=rule_id, note_id=note_id, artifact_sha=artifact_sha)
        ts = record.append(port="assess", backend="learner", key=key, subject=note_id,
                           question={"kind": "waiver", "rule": rule_id, "note_id": note_id,
                                    "artifact_sha": artifact_sha},
                           answer={"waived": bool(payload.get("waived", True)),
                                  "reason": payload.get("reason", "")})
        return {"ok": True, "ts": ts, "kind": "waiver"}

    subject = payload["subject"]
    kind = payload["kind"]
    subject_kind = payload.get("subject_kind", "word")
    role = payload.get("role") or role_for(kind, subject_kind)

    if "direction" in payload:
        text = payload["direction"]
        key = DirectionKey(subject=subject, role=role, text_sha=sha(text))
        ts = record.append(port="assess", backend="learner", key=key, subject=subject,
                           question={"kind": "direction", "role": role,
                                    "subject_kind": subject_kind},
                           answer={"direction": text})
        return {"ok": True, "ts": ts, "kind": "direction"}

    action = payload.get("action")

    if action == "keep":
        return {"ok": True, "kind": "challenger", "action": "keep"}
    artifact_sha = (payload.get("artifact_sha") or payload.get("challenger_sha")
                    if action == "switch" else payload.get("artifact_sha"))
    rating = _rating_of(payload)

    if rating not in LEARNER_RANK:
        raise ValueError(f"unknown rating {rating!r}")

    answer: dict[str, Any] = {"value": rating}
    if payload.get("note"):
        answer["note"] = payload["note"]
    key = LearnerKey(artifact_sha=artifact_sha, role=role)
    ts = record.append(port="assess", backend="learner", key=key, subject=subject,
                       question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                                "kind": "rating", "subject_kind": subject_kind},
                       answer=answer)
    return {"ok": True, "ts": ts, "rating": rating, "artifact_sha": artifact_sha}


def _refuses_stale_rejection(ctx: "ReviewContext", payload: Mapping[str, Any]) -> str | None:
    """The refusal message for a rejection naming an artifact_sha that is
    no longer `subject`'s current-best, else None (nothing named, or it
    still matches).
    """
    if _rating_of(payload) != "unacceptable-none":
        return None
    artifact_sha = payload.get("artifact_sha")
    if artifact_sha is None:
        return None
    subject, kind = payload.get("subject"), payload.get("kind")
    current_sha = ctx.current_best(subject, kind).artifact_sha
    if artifact_sha != current_sha:
        return (f"artifact {artifact_sha} is no longer current-best for {subject} ({kind}); "
               f"current-best is {current_sha!r}")
    return None


_LEARNER_SPEAKER = Speaker(id="learner", kind="native")

# Fallback extension when neither payload["ext"] nor the value's own
# filename suffix names one. A picture re-encodes through add_image
# regardless; a recording keeps the guess.
_DEFAULT_EXT = {"picture": "jpg", "recording": "mp3"}


def _guessed_ext(value: str, payload: Mapping[str, Any], kind: str) -> str:
    if payload.get("ext"):
        return str(payload["ext"])
    suffix = Path(value).suffix.lstrip(".").lower()
    return suffix or _DEFAULT_EXT[kind]


def _ingest_supplied_picture(ctx: "ReviewContext", payload: Mapping[str, Any], subject: str,
                             source: str, value: str) -> tuple[str, str]:
    """A picture always normalizes (spec 4 section 3): a URL's bytes
    normalize inside imgfetch's FetchBackend (provides="picture-bytes");
    a local file's bytes normalize through MediaStore.add_image directly.
    """
    if source == "url":
        provider = Provider(ctx.record, ctx.cache,
                            {"imgfetch": FetchBackend(media=ctx.media_store,
                                                      fetcher=ctx.url_fetchers["picture"])})
        answer = provider.ask("imgfetch", Question(subject=subject, provides="picture-bytes",
                                                    params={"url": value}, kind="picture",
                                                    subject_kind=payload.get("subject_kind",
                                                                            "word")))
        if not answer.items:
            raise ValueError(f"append_supply: imgfetch produced no artifact for {value!r}")
        item = answer.items[0]
        return str(item["sha"]), str(item["ext"])
    if source == "path":
        data = Path(value).read_bytes()
        ingest = ctx.media_store.add_image(data, _guessed_ext(value, payload, "picture"))
        _append_supply_provide_row(ctx, subject=subject, value=value, kind="picture",
                                   provides="picture-bytes", sha=ingest.sha, ext=ingest.ext,
                                   subject_kind=payload.get("subject_kind", "word"))
        return ingest.sha, ingest.ext
    raise ValueError(f"unknown supply source {source!r}")


def _append_supply_provide_row(ctx: "ReviewContext", *, subject: str, value: str, kind: str,
                               provides: str, sha: str, ext: str, subject_kind: str) -> None:
    """The provide row a local-path supply owes (spec 3 section 1: "an
    attempt appends"), matching what a URL supply already gets through
    Provider.ask. record.candidate_shas and derivations._anchor_ts read
    this row to see the artifact a path supply added.
    """
    ctx.record.append(port="provide", backend="learner",
                      key=ProvideKey(source="learner", kind="", query=value), subject=subject,
                      question={"provides": provides, "kind": kind, "subject_kind": subject_kind,
                               "params": {"path": value}},
                      answer={"items": [{"sha": sha, "ext": ext}]})


def _ingest_supplied_recording(ctx: "ReviewContext", payload: Mapping[str, Any], subject: str,
                               source: str, value: str) -> tuple[str, str]:
    """A recording is never normalized (spec 4 section 3 normalizes
    pictures only): a URL's bytes are fetched through audiofetch's
    FetchBackend, which stores them raw under their real ext; a local
    file's bytes go straight to MediaStore.write under the same real ext.
    """
    if source == "url":
        provider = Provider(ctx.record, ctx.cache,
                            {"audiofetch": FetchBackend(media=ctx.media_store,
                                                        fetcher=ctx.url_fetchers["recording"])})
        answer = provider.ask("audiofetch", Question(subject=subject, provides="recording-bytes",
                                                      params={"url": value}, kind="recording",
                                                      subject_kind=payload.get("subject_kind",
                                                                              "word")))
        if not answer.items:
            raise ValueError(f"append_supply: audiofetch produced no artifact for {value!r}")
        item = answer.items[0]
        return str(item["sha"]), str(item["ext"])
    if source == "path":
        data = Path(value).read_bytes()
        ext = _guessed_ext(value, payload, "recording")
        sha = ctx.media_store.write(data, ext)
        _append_supply_provide_row(ctx, subject=subject, value=value, kind="recording",
                                   provides="recording-bytes", sha=sha, ext=ext,
                                   subject_kind=payload.get("subject_kind", "word"))
        return sha, ext
    raise ValueError(f"unknown supply source {source!r}")


def append_supply(ctx: "ReviewContext", payload: Mapping[str, Any]) -> dict[str, Any]:
    """Spec 5 section 1 kind 2's supply action: {subject, kind:
    "picture"|"recording", source: "path"|"url", value, note?, ext?}. The
    bytes go through the ingest path for their kind (imgfetch/add_image
    for a picture, audiofetch/MediaStore.write for a recording), then a
    provenance row with source=learner and, for a recording, the
    "learner" Speaker row. A URL goes through Provider.ask, appending its
    own cache-first `provide` row; a local path appends its own
    (backend="learner"). Either way the artifact lands with an implicit
    use-this rating, so derivations.current_best picks it.
    """
    subject, kind = payload["subject"], payload["kind"]
    source, value = payload["source"], payload["value"]

    if kind == "picture":
        artifact_sha, ext = _ingest_supplied_picture(ctx, payload, subject, source, value)
        speaker_id = None
    elif kind == "recording":
        artifact_sha, ext = _ingest_supplied_recording(ctx, payload, subject, source, value)
        ctx.derivations.db.add_speaker(_LEARNER_SPEAKER)
        speaker_id = _LEARNER_SPEAKER.id
    else:
        raise ValueError(f"unknown supply kind {kind!r}")

    ctx.derivations.db.add_media(sha=artifact_sha, kind=kind, ext=ext, source="learner",
                                 origin=value, licence="learner", acquired=date.today(),
                                 speaker_id=speaker_id)

    subject_kind = payload.get("subject_kind", "word")
    role = payload.get("role") or role_for(kind, subject_kind)
    key = LearnerKey(artifact_sha=artifact_sha, role=role)
    answer_row: dict[str, Any] = {
        "value": "unacceptable-use-this",
        "provenance": {"source": "learner", "origin": value},
    }
    if payload.get("note"):
        answer_row["note"] = payload["note"]
    ts = ctx.record.append(port="assess", backend="learner", key=key, subject=subject,
                           question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                                    "kind": "rating", "subject_kind": subject_kind},
                           answer=answer_row)
    return {"ok": True, "ts": ts, "artifact_sha": artifact_sha}


# --- stats (spec 5 section 3) -----------------------------------------------

@dataclass
class SessionStats:
    """Per-session counters (spec 5 section 3), a session being this
    process's lifetime. Nothing here is persisted; every answer is
    already durable through RecordWriter.
    """
    answered: int = 0
    queued: int = 0


def _accepted(d: "Derivations", subject: str, role: str, best: CurrentBest) -> bool:
    """spec 5 section 3's "accepted": on a role the learner ranks, a
    current-best artifact ranked "acceptable" or better -- unchanged. On a
    veto-only role (spec 3 section 4 r8) current_best's own rank is a
    machine scale a rank-80 floor has no meaning on: accepted is a
    current-best artifact derivations.vetoed says is not vetoed -- the
    screen folds no row of its own here.
    """
    if best.artifact_sha is None:
        return False
    if learner_ranks(role):
        return best.rank >= _ACCEPTABLE_FLOOR
    return not vetoed(d.db, subject, role, best.artifact_sha)


def _drill_stats(d: "Derivations") -> dict[str, dict[str, int]]:
    """Per-confusion correct/total over the gallery drill rows
    append_drill_result appends.
    """
    drills: dict[str, dict[str, int]] = {}
    for confusion in d.syllabus.confusions:
        for r in d.db.assessments_of(confusion.id):
            if r.port == "assess" and r.backend == "learner" and r.question.get("kind") == "drill":
                bucket = drills.setdefault(confusion.id, {"correct": 0, "total": 0})
                bucket["total"] += 1
                if r.answer.get("correct"):
                    bucket["correct"] += 1
    return drills


def compute_stats(d: "Derivations", study: StudyReader | None = None, *,
                  session: SessionStats | None = None) -> dict[str, Any]:
    """Spec 5 section 3's stats, every count derived under `d`'s
    parameters: per-session (answered/queued, per-confusion drill
    accuracy, exhausted-remaining) and per-deck (coverage per need,
    learner rating counts, RunReport history).

    Coverage and the rating counts fold over derivations.all_needs, every
    need the deck has: `covered` is a need with a current-best artifact;
    `accepted` one whose current-best the learner rated acceptable or
    better on a role the learner ranks, else (spec 3 section 4 r8, a
    veto-only role) one with a current-best artifact not vetoed
    (see _accepted). `exhausted_remaining` is scoped to available_needs
    instead -- an outstanding need with no artifact and no source left.

    `pending`/`sentences_adopted` come from the newest run.py runreport
    row, else 0; `run_report_history` is every such row's answer, oldest
    first.
    """
    coverage: dict[str, dict[str, int]] = {}
    ratings = {"good": 0, "acceptable": 0, "unacceptable": 0}

    for subject, kind, subject_kind in all_needs(d.syllabus):
        best = _best(d, subject, kind)
        role = role_for(kind, subject_kind)
        bucket = coverage.setdefault(kind, {"covered": 0, "accepted": 0, "total": 0})
        bucket["total"] += 1
        if best.artifact_sha is not None:
            bucket["covered"] += 1
        if _accepted(d, subject, role, best):
            bucket["accepted"] += 1

        value = latest_rating(d.db.assessments_of(subject), role)
        if value == "good":
            ratings["good"] += 1
        elif value == "acceptable":
            ratings["acceptable"] += 1
        elif value is not None:
            ratings["unacceptable"] += 1

    exhausted_count = sum(1 for subject, kind, _ in available_needs(d.syllabus)
                         if _exhausted(d, subject, kind).exhausted)

    runreport = d.db.latest("run", "runreport", RunReportKey())
    runreport_answer = runreport.answer if runreport else {}

    return {
        "session": {"answered": session.answered if session else 0,
                   "queued": session.queued if session else 0},
        "exhausted_remaining": exhausted_count,
        "coverage": coverage,
        "ratings": ratings,
        "drills": _drill_stats(d),
        "pending": runreport_answer.get("pending", 0),
        "sentences_adopted": runreport_answer.get("sentences_adopted", 0),
        "run_report_history": [r.answer for r in run_reports(d.db)],
    }


# --- HTTP layer --------------------------------------------------------------

def _find_media_file(media_store: MediaStore, ext: str | None, sha: str) -> Path | None:
    """The media/objects file for `sha` at the extension its `media` row
    recorded (spec 2 section 2). An object with no provenance row is not
    served.
    """
    if ext is None:
        return None
    path = media_store.path_for(sha, ext)
    return path if path.exists() else None


def _learner_session_budget(d: "Derivations") -> int:
    """The question session's cap (spec 3 section 7's "learner 20/session",
    spec 5 section 1): budgets["learner"].max_asks off the same loaded
    Derivations bundle build_sourcing's own run() reads, falling back to
    run.LEARNER_DEFAULT_SESSION_BUDGET for a bundle carrying no "learner"
    entry (a bare Derivations built outside wiring.load_derivations).
    """
    budget = d.budgets.get("learner")
    cap = budget.max_asks if budget is not None else None
    return cap if cap is not None else LEARNER_DEFAULT_SESSION_BUDGET.max_asks


@dataclass
class ReviewContext:
    """One review session over one deck: its Derivations (the run's own
    parameters), the StudyReader the re-ask questions read, the
    learner-attention budget, and the session's counters. Every
    derivation the surface shows goes through the methods below.
    """
    derivations: "Derivations"
    study: StudyReader | None = None
    # None resolves in __post_init__ to derivations.budgets["learner"]'s
    # own max_asks (spec 3 section 7), the same one Budget build_sourcing's
    # run() reads -- an explicit value here (the CLI's --budget) overrides it.
    learner_budget: int | None = None
    # A supplied artifact's URL fetcher, by kind (spec 5 section 1 kind 2:
    # imgfetch for pictures, audiofetch for recordings -- a recording URL
    # sent to imgfetch is refused, since imgfetch expects an image).
    url_fetchers: Mapping[str, Callable[[str], tuple[bytes, str]]] = field(default_factory=dict)
    cards_provider: Callable[["Derivations"], list[dict[str, Any]]] = field(
        default=compiled_cards)
    session: SessionStats = field(default_factory=SessionStats)

    def __post_init__(self) -> None:
        fetchers = dict(self.url_fetchers)
        fetchers.setdefault("picture", tool_fetcher("imgfetch"))
        fetchers.setdefault("recording", tool_fetcher("audiofetch"))
        self.url_fetchers = fetchers
        if self.learner_budget is None:
            self.learner_budget = _learner_session_budget(self.derivations)

    @property
    def syllabus(self) -> Syllabus:
        return self.derivations.syllabus

    @property
    def cache(self) -> CacheReader:
        return self.derivations.db

    @property
    def record(self) -> RecordWriter:
        return self.derivations.db

    def media_ext(self, sha: str) -> str | None:
        """The ext its media table provenance row recorded, or None when
        `sha` has none (spec 2 section 2 -- provenance is the media table,
        never a filename glob)."""
        provenance = self.derivations.db.media_provenance(sha)
        return provenance["ext"] if provenance else None

    @property
    def media_store(self) -> MediaStore:
        return self.derivations.media_store

    def current_best(self, subject: str, kind: str) -> CurrentBest:
        return _best(self.derivations, subject, kind)

    def exhausted(self, subject: str, kind: str) -> ExhaustedStatus:
        return _exhausted(self.derivations, subject, kind)

    def queue(self) -> list[QueueEntry]:
        d = self.derivations
        return queue(d.syllabus, d.db, current_rubric=d.current_rubric, prior=d.prior,
                     sources_for=d.sources_for, attempt_cap=d.attempt_cap,
                     transient_cap=d.transient_cap,
                     provenance_source=d.provenance_source)

    def questions(self, budget: int | None = None) -> list[dict[str, Any]]:
        return build_queue(self.derivations, self.study,
                           budget=self.learner_budget if budget is None else budget)

    def cards(self) -> list[dict[str, Any]]:
        return self.cards_provider(self.derivations)

    def stats(self) -> dict[str, Any]:
        return compute_stats(self.derivations, self.study, session=self.session)


def build_app(ctx: ReviewContext) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "ReviewServer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: Any, status: int = 200) -> None:
            self._send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8", status)

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b"{}"
            return json.loads(body.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/":
                self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/queue":
                items = ctx.questions(int((qs.get("budget") or [ctx.learner_budget])[0]))
                ctx.session.queued = len(items)
                self._send_json(items)
            elif parsed.path == "/api/cards":
                self._send_json(ctx.cards())
            elif parsed.path == "/stats":
                self._send_json(ctx.stats())
            elif parsed.path.startswith("/media/"):
                self._serve_media(urllib.parse.unquote(parsed.path[len("/media/"):]))
            else:
                self.send_error(404, "not found")

        def _serve_media(self, sha: str) -> None:
            path = _find_media_file(ctx.media_store, ctx.media_ext(sha), sha)
            if path is None or not path.exists():
                self.send_error(404, f"missing media: {sha}")
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            try:
                self._send_bytes(path.read_bytes(), ctype)
            except OSError:
                self.send_error(404, f"missing media: {sha}")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            try:
                payload = self._read_json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            try:
                if parsed.path == "/api/answer":
                    refusal = _refuses_stale_rejection(ctx, payload)
                    if refusal is not None:
                        raise ValueError(refusal)
                    result = append_answer(ctx.record, payload)
                    ctx.session.answered += 1
                    self._send_json(result)
                elif parsed.path == "/api/supply":
                    result = append_supply(ctx, payload)
                    ctx.session.answered += 1
                    self._send_json(result)
                elif parsed.path == "/api/note":
                    card_id = payload.get("card_id", payload.get("id"))
                    subject = payload.get("subject") or card_id
                    ts = append_gallery_note(ctx.record, subject=str(subject),
                                             card_id=str(card_id),
                                             kind=payload.get("kind", "note"),
                                             text=payload.get("text", ""))
                    self._send_json({"ok": True, "ts": ts})
                elif parsed.path == "/api/drill":
                    ts = append_drill_result(ctx.record, confusion=payload["confusion"],
                                             pair_id=payload["pair"],
                                             correct=bool(payload.get("correct")))
                    self._send_json({"ok": True, "ts": ts})
                else:
                    self.send_error(404, "not found")
            except (KeyError, ValueError) as e:
                self._send_json({"ok": False, "error": str(e)}, status=400)

    return Handler


def serve(ctx: ReviewContext, port: int) -> None:
    handler = build_app(ctx)
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    print(f"review: http://127.0.0.1:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def load_context(deck_dir: str | Path, *, learner_budget: int | None = None
                 ) -> ReviewContext:
    """A ReviewContext over a deck directory (spec 2 section 1 layout).
    wiring.load_derivations supplies the same assembly build_sourcing
    hands the run: the Syllabus, one db connection as CacheReader/
    RecordWriter/StudyReader, the deck's rubric, provenance prior, Source
    roster, attempt cap and budgets; the screen adds no parameter of its
    own. `learner_budget` left None takes the session cap from that
    bundle's own budgets["learner"] (spec 3 section 7); an explicit value
    (the CLI's --budget) overrides it. The supplied-URL fetchers are
    providers.yaml's own imgfetch_path/audiofetch_path. wiring is imported
    inside the function, off cli.py's import path.
    """
    from .curated import load_providers_config
    from .wiring import load_derivations

    root = Path(deck_dir)
    cfg = load_providers_config(root / "curated" / "providers.yaml")
    derivations = load_derivations(root, cfg)
    url_fetchers = {"picture": tool_fetcher(cfg.imgfetch_path),
                   "recording": tool_fetcher(cfg.audiofetch_path)}
    return ReviewContext(derivations=derivations, study=derivations.db,
                         learner_budget=learner_budget, url_fetchers=url_fetchers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, type=Path, help="deck directory (spec 2 layout)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--budget", type=int, default=None,
                        help="learner-attention session budget override (spec 3 section 7's "
                             "\"learner\" Budget; default: providers.yaml's own quota)")
    args = parser.parse_args(argv)

    ctx = load_context(args.deck, learner_budget=args.budget)
    serve(ctx, args.port)
    return 0


# ---------------------------------------------------------------------------
# the page: inline CSS + JS, no external resources, keyboard-first
# (spec 5 section 2: "1-4 rate, n note, arrows navigate, g gloss, s stats")
# ---------------------------------------------------------------------------

_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: #111417; color: #e8e8e8;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  #bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #1b1f24; border-bottom: 1px solid #2a2f36;
    font-size: 13px; gap: 12px; flex-wrap: wrap;
  }
  #bar .left { display: flex; align-items: center; gap: 10px; }
  #bar button.mode {
    background: #262b31; color: #fff; border: 1px solid #3a4048; border-radius: 6px;
    padding: 4px 10px; cursor: pointer; font-size: 13px;
  }
  #bar button.mode.active { background: #2f5c8a; border-color: #4a7fb5; }
  #progress { font-weight: 600; }
  #help { color: #6b7480; }
  #main { flex: 1; overflow: auto; padding: 20px; }
  .card-box {
    max-width: 900px; margin: 0 auto; display: flex; flex-direction: column;
    align-items: center; gap: 12px; text-align: center;
  }
  .thai { font-size: 44px; }
  .gloss-chip {
    display: inline-block; border: 2px dashed #4fb3bf; color: #7fe0ea;
    padding: 4px 12px; border-radius: 8px; font-size: 18px; background: #10262a;
  }
  .current-artifact { max-width: min(90vw, 640px); }
  .verdict { color: #9aa4b1; font-size: 14px; }
  .query { color: #6b7480; font-size: 13px; font-family: monospace; }
  .thumbs { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
  .thumbs img {
    max-width: 120px; max-height: 120px; border-radius: 4px; cursor: zoom-in;
    border: 1px solid #3a4048;
  }
  .actions { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
  .actions button {
    font-size: 15px; padding: 10px 18px; background: #262b31; color: #fff;
    border: 1px solid #3a4048; border-radius: 8px; cursor: pointer;
  }
  .actions button:hover { background: #323942; }
  .actions button.good { border-color: #2f8a44; }
  .actions button.bad { border-color: #8a2f2f; }
  .side-by-side { display: flex; gap: 24px; justify-content: center; flex-wrap: wrap; }
  .side-by-side figure { margin: 0; }
  .side-by-side img { max-width: 320px; max-height: 320px; border-radius: 6px; }
  .tried { text-align: left; max-width: 640px; margin: 0 auto; font-size: 14px; color: #b9c2cd; }
  .tried h4 { margin: 8px 0 2px; color: #e8e8e8; }
  #noteInput, #directionInput, #supplyInput {
    position: fixed; left: 50%; bottom: 60px; transform: translateX(-50%);
    background: #1b1f24; border: 1px solid #3a4048; border-radius: 8px;
    padding: 10px 14px; display: flex; gap: 8px; align-items: center; z-index: 5;
  }
  #noteInput input, #directionInput input, #supplyInput input {
    width: 420px; background: #0e1114; color: #e8e8e8; border: 1px solid #333;
    border-radius: 4px; padding: 6px 10px; font-size: 15px;
  }
  #overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: flex;
    align-items: center; justify-content: center; z-index: 10; cursor: zoom-out;
  }
  #overlay img { max-width: 90vw; max-height: 90vh; }
  #statsOverlay { position: fixed; inset: 0; background: rgba(10,12,14,0.92);
    display: flex; align-items: center; justify-content: center; z-index: 10; }
  #statsOverlay .panel { background: #1b1f24; border: 1px solid #3a4048; border-radius: 10px;
    padding: 24px 32px; min-width: 320px; max-height: 80vh; overflow: auto; }
  #statsOverlay table { border-collapse: collapse; width: 100%; margin-top: 10px; }
  #statsOverlay td { padding: 4px 10px; border-bottom: 1px solid #2a2f36; font-size: 14px; }
  .empty { color: #6b7480; padding: 40px; }
  [hidden] { display: none !important; }
</style>
<style>__CARD_CSS__</style>
</head>
<body>
  <div id="bar">
    <div class="left">
      <button class="mode active" id="modeSession">session</button>
      <button class="mode" id="modeGallery">gallery</button>
      <span id="progress">- / -</span>
    </div>
    <div class="left">
      <span id="help">1-4 rate &middot; n note &middot; arrows navigate &middot; g gloss &middot; s stats</span>
    </div>
  </div>
  <div id="main"></div>
  <div id="noteInput" hidden><input id="noteText" placeholder="note (Enter to save, Esc to cancel)"></div>
  <div id="directionInput" hidden><input id="directionText" placeholder="direction (Enter to save, Esc to cancel)"></div>
  <div id="supplyInput" hidden>
    <input id="supplyValue" placeholder="file path or URL (Enter to save, Esc to cancel)">
  </div>
  <div id="overlay" hidden><img id="overlayImg" src=""></div>
  <div id="statsOverlay" hidden><div class="panel" id="statsPanel"></div></div>

<script>
(function () {
  "use strict";

  var POS_KEY = "review_pos";
  var GLOSS_KEY = "review_gloss";
  var MODE_KEY = "review_mode";

  var mode = localStorage.getItem(MODE_KEY) || "session";
  var glossOn = (localStorage.getItem(GLOSS_KEY) ?? "1") === "1";

  var queueItems = [];
  var qIdx = 0;
  var galleryCards = [];
  var gIdx = 0;
  var revealed = false;

  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) { for (var k in attrs) { e.setAttribute(k, attrs[k]); } }
    if (text !== undefined && text !== null) { e.textContent = text; }
    return e;
  }

  function saveProgress() {
    try {
      localStorage.setItem(POS_KEY, JSON.stringify({ session: qIdx, gallery: gIdx }));
    } catch (e) {}
  }

  function postJson(path, body) {
    return fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then(function (r) { return r.json(); }).catch(function () { return { ok: false }; });
  }

  // --- session (question queue) ------------------------------------------

  function loadQueue() {
    fetch("/api/queue").then(function (r) { return r.json(); }).then(function (items) {
      queueItems = items;
      if (qIdx >= queueItems.length) { qIdx = 0; }
      renderSession();
    });
  }

  function renderSession() {
    var main = document.getElementById("main");
    main.innerHTML = "";
    document.getElementById("progress").textContent = queueItems.length
      ? (qIdx + 1) + " / " + queueItems.length : "0 / 0";
    if (!queueItems.length) {
      main.appendChild(el("div", { "class": "empty" }, "queue is empty -- nothing to answer right now"));
      return;
    }
    var q = queueItems[qIdx];
    var box = el("div", { "class": "card-box" });
    if (q.gloss) { box.appendChild(el("div", { "class": "gloss-chip" }, q.gloss)); }

    if (q.type === "rate") { renderRate(q, box); }
    else if (q.type === "direction") { renderDirection(q, box); }
    else if (q.type === "challenger") { renderChallenger(q, box); }
    else if (q.type === "reask") { renderReask(q, box); }
    main.appendChild(box);
    saveProgress();
  }

  function rateLabels(learnerRanks) {
    // r8: action 2 nominates the picked candidate as the artifact to use
    // instead, on both a "rate" and a "reask" question, on every role --
    // it keeps the need directed, it never ranks the sha on its own. On a
    // veto-only role (learner_ranks false) 1 is the veto and 3/4 are a
    // note only, never a rank; a learner-ranking role keeps 1/3/4's
    // original vocabulary.
    return learnerRanks
      ? { 1: "1 unacceptable-none", 2: "2 unacceptable, use this one instead",
         3: "3 acceptable", 4: "4 good" }
      : { 1: "1 unacceptable (veto)", 2: "2 unacceptable, use this one instead",
         3: "3 acceptable (note)", 4: "4 good (note)" };
  }

  function thumb(art, cls) {
    var img = el("img", { src: art.url, "data-sha": art.sha });
    img.addEventListener("click", function () { openOverlay(art.url); });
    return img;
  }

  function renderRate(q, box) {
    box.appendChild(el("div", {}, q.subject + " (" + q.kind + ")"));
    if (q.query) { box.appendChild(el("div", { "class": "query" }, "query: " + q.query)); }
    if (q.current) {
      // "card" carries compile.CARD_CSS's own sizing (F4: judge the
      // artifact at the size the card will actually show it).
      var cur = el("div", { "class": "current-artifact card" });
      cur.appendChild(thumb(q.current));
      box.appendChild(cur);
      if (q.current.verdict) { box.appendChild(el("div", { "class": "verdict" }, q.current.verdict)); }
    } else {
      box.appendChild(el("div", { "class": "empty" }, "no current artifact"));
    }
    if (q.rejected && q.rejected.length) {
      var thumbs = el("div", { "class": "thumbs" });
      q.rejected.forEach(function (art) { thumbs.appendChild(thumb(art)); });
      box.appendChild(thumbs);
    }
    var actions = el("div", { "class": "actions" });
    var labels = rateLabels(q.learner_ranks);
    [1, 2, 3, 4].forEach(function (n) {
      var btn = el("button", { "class": n >= 3 ? "good" : "bad" }, labels[n]);
      btn.addEventListener("click", function () { answerRate(q, n); });
      actions.appendChild(btn);
    });
    box.appendChild(actions);
  }

  function pickCandidateForAction2(q, cb) {
    if (!q.rejected || !q.rejected.length) { cb(null); return; }
    // simplest usable UX for action 2: use the most recently shown
    // thumbnail the learner clicked to enlarge, defaulting to the first.
    cb((window.__lastThumbClick && window.__lastThumbClick.sha) || q.rejected[0].sha);
  }

  function answerRate(q, action, noteText) {
    var payload = { subject: q.subject, kind: q.kind, subject_kind: q.subject_kind,
                    role: q.role, action: action, note: noteText || "" };
    if (action === 2) {
      pickCandidateForAction2(q, function (sha) {
        payload.artifact_sha = sha;
        finishAnswer(payload);
      });
      return;
    }
    if (q.current) { payload.artifact_sha = q.current.sha; }
    finishAnswer(payload);
  }

  function finishAnswer(payload) {
    postJson("/api/answer", payload).then(function () { advanceQueue(); });
  }

  function renderDirection(q, box) {
    box.appendChild(el("div", {}, q.subject + " (" + q.kind + ") -- exhausted, attempts=" + q.attempts));
    var tried = el("div", { "class": "tried" });
    tried.appendChild(el("h4", {}, "tried"));
    if (q.tried && q.tried.length) {
      q.tried.forEach(function (t) {
        tried.appendChild(el("div", {}, t.source + (t.query ? ": " + t.query : "")));
      });
    } else {
      tried.appendChild(el("div", {}, "none"));
    }
    tried.appendChild(el("h4", {}, "best candidates"));
    if (q.candidates && q.candidates.length) {
      q.candidates.forEach(function (c) {
        var verdictText = c.verdict
          ? (c.verdict.passed ? "pass" : "fail") + (c.verdict.evidence ? " -- " + c.verdict.evidence : "")
          : "no verdict";
        tried.appendChild(el("div", {}, c.sha + ": " + verdictText));
      });
    } else {
      tried.appendChild(el("div", {}, "none"));
    }
    box.appendChild(tried);
    var actions = el("div", { "class": "actions" });
    var dirBtn = el("button", {}, "type a direction");
    dirBtn.addEventListener("click", function () { openDirectionBox(q); });
    var supplyBtn = el("button", {}, "supply an artifact");
    supplyBtn.addEventListener("click", function () { openSupplyBox(q); });
    actions.appendChild(dirBtn);
    actions.appendChild(supplyBtn);
    box.appendChild(actions);
  }

  function renderChallenger(q, box) {
    box.appendChild(el("div", {}, q.subject + " (" + q.kind + ") -- a new candidate outranks your pick"));
    var side = el("div", { "class": "side-by-side" });
    var cur = el("figure");
    cur.appendChild(thumb(q.current));
    cur.appendChild(el("figcaption", {}, "current"));
    var chal = el("figure");
    chal.appendChild(thumb(q.challenger));
    chal.appendChild(el("figcaption", {}, "challenger"));
    side.appendChild(cur);
    side.appendChild(chal);
    box.appendChild(side);
    var actions = el("div", { "class": "actions" });
    var keepBtn = el("button", { "class": "good" }, "keep");
    keepBtn.addEventListener("click", function () {
      postJson("/api/answer", { subject: q.subject, kind: q.kind,
                                subject_kind: q.subject_kind, role: q.role,
                                action: "keep" })
        .then(function () { advanceQueue(); });
    });
    var switchBtn = el("button", { "class": "bad" }, "switch");
    switchBtn.addEventListener("click", function () {
      postJson("/api/answer", { subject: q.subject, kind: q.kind,
                                subject_kind: q.subject_kind, role: q.role,
                                action: "switch", artifact_sha: q.challenger.sha })
        .then(function () { advanceQueue(); });
    });
    actions.appendChild(keepBtn);
    actions.appendChild(switchBtn);
    box.appendChild(actions);
  }

  function renderReask(q, box) {
    box.appendChild(el("div", {}, q.subject + " (" + q.kind + ") -- lapse evidence contradicts a past rating"));
    box.appendChild(el("div", { "class": "verdict" }, "original answer: " + q.original_answer));
    if (q.current) {
      var cur = el("div", { "class": "current-artifact" });
      cur.appendChild(thumb(q.current));
      box.appendChild(cur);
    }
    var ev = el("div", { "class": "tried" });
    ev.appendChild(el("h4", {}, "lapse evidence"));
    (q.evidence || []).forEach(function (e) {
      ev.appendChild(el("div", {}, e.anchor + " (" + e.card_kind + "): grade " + e.grade));
    });
    box.appendChild(ev);
    var actions = el("div", { "class": "actions" });
    var labels = rateLabels(q.learner_ranks);
    [1, 2, 3, 4].forEach(function (n) {
      var btn = el("button", { "class": n >= 3 ? "good" : "bad" }, labels[n]);
      btn.addEventListener("click", function () { answerRate(q, n); });
      actions.appendChild(btn);
    });
    box.appendChild(actions);
  }

  function advanceQueue() {
    loadQueue();
  }

  // --- gallery -------------------------------------------------------------

  function loadGallery() {
    fetch("/api/cards").then(function (r) { return r.json(); }).then(function (cards) {
      galleryCards = cards;
      if (gIdx >= galleryCards.length) { gIdx = 0; }
      renderGallery();
    });
  }

  function renderGallery() {
    revealed = false;
    var main = document.getElementById("main");
    main.innerHTML = "";
    document.getElementById("progress").textContent = galleryCards.length
      ? (gIdx + 1) + " / " + galleryCards.length : "0 / 0";
    if (!galleryCards.length) {
      main.appendChild(el("div", { "class": "empty" }, "no cards"));
      return;
    }
    var card = galleryCards[gIdx];
    var box = el("div", { "class": "card-box" });
    // the served front/back HTML and its model's own CSS -- what Anki
    // shows (spec 5 section 1, principles F4), nothing recomposed here.
    var style = document.createElement("style");
    style.textContent = card.css;
    box.appendChild(style);
    // "g" overlays the note's own gloss on the front (spec 5 section 2);
    // metadata beside the rendered HTML, not a recomposed card shape.
    if (glossOn && card.gloss) { box.appendChild(el("div", { "class": "gloss-chip" }, card.gloss)); }
    var face = el("div", { "class": "card" });
    face.innerHTML = card.front_html;
    box.appendChild(face);
    if (card.family === "minimal_pair") { renderPairDrill(card, box); }
    box.appendChild(el("div", { "class": "verdict" }, "space to reveal"));
    main.appendChild(box);
    saveProgress();
    var audio = face.querySelector("audio");
    if (audio) { audio.play().catch(function () {}); }
  }

  // Per-confusion accuracy logging (spec 5 section 1): a forced two-way
  // guess against `stimulus_member` (the member index this note's own
  // Stimulus is, read straight off the note's tags -- compiled_cards
  // carries it as metadata, never a recomposed card shape) before the
  // card's own reveal shows the answer.
  function renderPairDrill(card, box) {
    box.appendChild(el("div", { "class": "verdict" }, "which one did you hear?"));
    var choices = el("div", { "class": "actions" });
    [0, 1].forEach(function (idx) {
      var btn = el("button", {}, idx === 0 ? "1st" : "2nd");
      btn.addEventListener("click", function () {
        postJson("/api/drill", { confusion: card.confusion, pair: card.subject,
                                 correct: idx === card.stimulus_member });
        btn.parentNode.querySelectorAll("button").forEach(function (b) { b.disabled = true; });
      });
      choices.appendChild(btn);
    });
    box.appendChild(choices);
  }

  function revealGallery() {
    if (revealed || !galleryCards.length) { return; }
    revealed = true;
    var card = galleryCards[gIdx];
    var face = document.querySelector("#main .card-box .card");
    face.innerHTML = card.back_html;
  }

  function next() {
    if (mode === "session") { qIdx = Math.min(qIdx + 1, queueItems.length - 1); renderSession(); }
    else { gIdx = Math.min(gIdx + 1, galleryCards.length - 1); renderGallery(); }
  }
  function prev() {
    if (mode === "session") { qIdx = Math.max(qIdx - 1, 0); renderSession(); }
    else { gIdx = Math.max(gIdx - 1, 0); renderGallery(); }
  }

  // --- note / direction / supply input boxes --------------------------------

  function openBox(id, inputId, onSave) {
    var box = document.getElementById(id);
    var input = document.getElementById(inputId);
    box.hidden = false;
    input.value = "";
    input.focus();
    input.onkeydown = function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        var val = input.value;
        box.hidden = true;
        onSave(val);
      } else if (e.key === "Escape") {
        e.preventDefault();
        box.hidden = true;
      }
    };
  }

  function openNoteBox() {
    if (mode !== "gallery" || !galleryCards.length) { return; }
    var card = galleryCards[gIdx];
    openBox("noteInput", "noteText", function (text) {
      postJson("/api/note", { subject: card.subject, card_id: card.id, kind: card.kind, text: text });
    });
  }

  function openDirectionBox(q) {
    // A typed direction is recorded as a direction, not a rating (spec 5
    // section 1 kind 2): no action, no rating -- append_answer's own
    // "direction" branch.
    openBox("directionInput", "directionText", function (text) {
      postJson("/api/answer", { subject: q.subject, kind: q.kind,
                                subject_kind: q.subject_kind, role: q.role,
                                direction: text })
        .then(function () { advanceQueue(); });
    });
  }

  function openSupplyBox(q) {
    openBox("supplyInput", "supplyValue", function (val) {
      var source = /^https?:\\/\\//.test(val) ? "url" : "path";
      postJson("/api/supply", { subject: q.subject, kind: q.kind,
                                subject_kind: q.subject_kind, role: q.role,
                                source: source, value: val })
        .then(function () { advanceQueue(); });
    });
  }

  // --- overlay / stats -------------------------------------------------------

  function openOverlay(url) {
    window.__lastThumbClick = { sha: url.replace("/media/", "") };
    document.getElementById("overlayImg").src = url;
    document.getElementById("overlay").hidden = false;
  }
  document.getElementById("overlay").addEventListener("click", function () {
    document.getElementById("overlay").hidden = true;
  });

  function toggleStats() {
    var overlay = document.getElementById("statsOverlay");
    if (!overlay.hidden) { overlay.hidden = true; return; }
    fetch("/stats").then(function (r) { return r.json(); }).then(function (stats) {
      var panel = document.getElementById("statsPanel");
      panel.innerHTML = "";
      panel.appendChild(el("h2", {}, "Stats"));
      panel.appendChild(el("div", {},
        "answered " + stats.session.answered + " / queued " + stats.session.queued));
      panel.appendChild(el("div", {}, "exhausted remaining: " + stats.exhausted_remaining));
      panel.appendChild(el("div", {},
        "ratings: good " + stats.ratings.good + " / acceptable " + stats.ratings.acceptable +
        " / unacceptable " + stats.ratings.unacceptable));
      var table = el("table");
      Object.keys(stats.coverage).forEach(function (kind) {
        var c = stats.coverage[kind];
        var row = el("tr");
        row.appendChild(el("td", {}, kind));
        row.appendChild(el("td", {}, c.covered + " / " + c.total + " (accepted " + c.accepted + ")"));
        table.appendChild(row);
      });
      panel.appendChild(table);

      // RunReport history (spec 5 section 3): every run.py row, oldest
      // first, one table row per run with every field the row carries.
      panel.appendChild(el("h3", {}, "Run history"));
      var history = stats.run_report_history;
      if (history.length) {
        var fields = Object.keys(history[0]);
        var histTable = el("table");
        var head = el("tr");
        fields.forEach(function (f) { head.appendChild(el("th", {}, f)); });
        histTable.appendChild(head);
        history.forEach(function (report) {
          var row = el("tr");
          fields.forEach(function (f) {
            var value = report[f];
            var text = (value !== null && typeof value === "object")
              ? JSON.stringify(value) : String(value);
            row.appendChild(el("td", {}, text));
          });
          histTable.appendChild(row);
        });
        panel.appendChild(histTable);
      } else {
        panel.appendChild(el("div", {}, "no runs yet"));
      }

      panel.appendChild(el("div", { style: "margin-top:10px;color:#9aa4b1;" }, "press s to close"));
      overlay.hidden = false;
    });
  }

  function setMode(next) {
    mode = next;
    try { localStorage.setItem(MODE_KEY, mode); } catch (e) {}
    document.getElementById("modeSession").classList.toggle("active", mode === "session");
    document.getElementById("modeGallery").classList.toggle("active", mode === "gallery");
    if (mode === "session") { loadQueue(); } else { loadGallery(); }
  }

  document.getElementById("modeSession").addEventListener("click", function () { setMode("session"); });
  document.getElementById("modeGallery").addEventListener("click", function () { setMode("gallery"); });

  document.addEventListener("keydown", function (e) {
    var active = document.activeElement;
    if (active && active.tagName === "INPUT") { return; }
    if (!document.getElementById("overlay").hidden) {
      document.getElementById("overlay").hidden = true;
      return;
    }
    if (e.key === "Escape") {
      var so = document.getElementById("statsOverlay");
      if (!so.hidden) { so.hidden = true; }
      return;
    }
    if (e.key === "s") { toggleStats(); return; }
    if (e.key === "g") { glossOn = !glossOn; localStorage.setItem(GLOSS_KEY, glossOn ? "1" : "0");
      if (mode === "gallery") { renderGallery(); } return; }
    if (e.key === "ArrowRight" || e.key === "j") { next(); return; }
    if (e.key === "ArrowLeft" || e.key === "k") { prev(); return; }
    if (e.key === " " && mode === "gallery") { e.preventDefault(); revealGallery(); return; }
    if (e.key === "n") { openNoteBox(); return; }
    if (["1", "2", "3", "4"].indexOf(e.key) !== -1 && mode === "session" && queueItems.length) {
      var q = queueItems[qIdx];
      if (q.type === "rate" || q.type === "reask") { answerRate(q, parseInt(e.key, 10)); }
    }
  });

  // restore position (localStorage: position only, spec 5 section 2)
  try {
    var pos = JSON.parse(localStorage.getItem(POS_KEY) || "{}");
    qIdx = pos.session || 0;
    gIdx = pos.gallery || 0;
  } catch (e) {}

  setMode(mode);
})();
</script>
</body>
</html>
"""

# The compiled card's own CSS (compile.CARD_CSS, spec 4 section 1), so a
# ".card" element on the page -- the gallery face and the rate screen's
# current-artifact wrapper -- renders at the size and style Anki renders
# it at (principles F4), not a page-composed one.
INDEX_HTML = _INDEX_HTML_TEMPLATE.replace("__CARD_CSS__", CARD_CSS)


if __name__ == "__main__":
    raise SystemExit(main())
