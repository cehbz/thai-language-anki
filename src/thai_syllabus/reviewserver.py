"""Spec 5: the feedback screen -- the local surface where the learner
answers the system's questions and reviews the deck.

Presents the run's own derivations and records the learner's acts. The
parameters every fold is measured under -- the wired Syllabus and its
media index, the deck's rubric, provenance prior, Source roster and
attempt cap -- arrive as one wiring.Derivations bundle, the same one
wiring.build_sourcing hands the run, so current-best, exhausted, the
queue and coverage read here exactly as they read there; this module
computes none of them itself.

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
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .authority import role_for
from .cachekeys import DrillKey, LearnerKey, WaiverKey
from .derivations import (
    LEARNER_RANK,
    Challenger,
    CurrentBest,
    ExhaustedStatus,
    JudgeVerdict,
    QueueEntry,
    available_needs,
    challengers,
    current_best,
    exhausted,
    judge_verdict,
    queue,
)
from .ports import Answer, CacheReader, RecordWriter, StudyReader
from .provider import FetchBackend, Provider, Question, tool_fetcher
from .record import (
    candidate_shas,
    judge_verdicts,
    latest_query,
    ratings_for_role,
    rows_for,
    source_asks,
)
from .store import MediaStore
from .syllabus import Syllabus

if TYPE_CHECKING:                       # wiring reaches for provider/assessor; the
    from .wiring import Derivations     # screen needs only the bundle's values.

__all__ = [
    "ReviewContext", "SessionStats", "build_app", "serve", "load_context", "main",
    "build_queue", "simplified_cards", "compute_stats",
    "append_answer", "append_supply", "append_gallery_note", "append_drill_result",
]

DEFAULT_PORT = 8877          # 8765 is reserved for AnkiConnect / proof_gallery.py
DEFAULT_LEARNER_BUDGET = 20  # spec 3 section 4: session default, ~25 min

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
                     attempt_cap=d.attempt_cap)


def _gloss_for(syllabus: Syllabus, subject: str) -> str | None:
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
        "role": role_for(kind, subject_kind),
        "gloss": _gloss_for(d.syllabus, subject), "query": latest_query(rows),
        "current": current, "rejected": rejected, "directed": directed,
        "rank": best.rank, "attempts": attempts,
    }


def _tried_summary(rows: Sequence[Answer], role: str) -> dict[str, Any]:
    """What spec 5 section 1 kind 2 shows an exhausted subject: the phrases
    its Source asks carried, the Sources asked, the judge's reasons, and the
    candidates those asks produced.
    """
    asks = source_asks(rows)
    phrases: list[str] = []
    for ask in asks:
        params = ask.question.get("params", {}) or {}
        phrase = params.get("query") or params.get("text")
        if phrase:
            phrases.append(phrase)
    judge_reasons = [evidence for r in judge_verdicts(rows, role)
                    if (evidence := r.answer.get("evidence"))]
    return {"phrases": phrases, "sources": sorted({r.backend for r in asks}),
           "judge_reasons": judge_reasons, "best_candidates": candidate_shas(rows)[:5]}


def _direction_question(d: "Derivations", subject: str, kind: str, subject_kind: str,
                        attempts: int) -> dict[str, Any]:
    role = role_for(kind, subject_kind)
    rows = rows_for(d.db, subject, kind)
    return {
        "type": "direction", "subject": subject, "kind": kind, "subject_kind": subject_kind,
        "role": role, "gloss": _gloss_for(d.syllabus, subject),
        "tried": _tried_summary(rows, role), "attempts": attempts,
    }


def _challenger_question(d: "Derivations", challenger: Challenger) -> dict[str, Any]:
    return {
        "type": "challenger", "subject": challenger.subject, "kind": challenger.kind,
        "subject_kind": challenger.subject_kind,
        "role": role_for(challenger.kind, challenger.subject_kind),
        "gloss": _gloss_for(d.syllabus, challenger.subject),
        "current": _artifact(challenger.current_sha),
        "challenger": _artifact(challenger.challenger_sha),
    }


def _reask_questions(d: "Derivations", study: StudyReader) -> list[dict[str, Any]]:
    """Spec 5 section 1 kind 4 over confusions: a confusion whose
    StudyRecords hold lapses (grade <= 1) and whose rendition the learner
    has already rated is a contradiction worth re-asking. A kind with no
    matching derivation input yields no questions.
    """
    out: list[dict[str, Any]] = []
    grouped = d.syllabus.study_by_confusion(study)
    for confusion in d.syllabus.confusions:
        lapses = [r for r in grouped.get(confusion.id, []) if r.grade <= 1]
        if not lapses:
            continue
        subject, kind, subject_kind = confusion.id, "rendition", "pair"
        role = role_for(kind, subject_kind)
        rated = ratings_for_role(d.db.assessments_of(subject), role)
        if not rated:
            continue  # no prior answer to contradict -- nothing to re-ask
        latest = max(rated, key=lambda r: r.ts)
        best = _best(d, subject, kind)
        out.append({
            "type": "reask", "subject": subject, "kind": kind, "subject_kind": subject_kind,
            "role": role, "gloss": None, "original_answer": latest.answer.get("value"),
            "current": _artifact(best.artifact_sha),
            "evidence": [{"anchor": r.anchor, "card_kind": r.card_kind, "grade": r.grade,
                         "ts": r.ts} for r in lapses[-5:]],
        })
    return out


def build_queue(d: "Derivations", study: StudyReader | None = None, *,
                budget: int = DEFAULT_LEARNER_BUDGET) -> list[dict[str, Any]]:
    """The question session (spec 5 section 1): four kinds, data-driven
    from derivations.py under `d`'s parameters, capped by the session-wide
    learner-attention budget. Highest expected gain first: the F10-ordered
    rate questions (derivations.queue's own order) fill the budget first;
    direction requests, challenger comparisons and re-asks fill whatever
    the queue left. A kind with no matching derivation input yields no
    questions (no StudyRecords -> kind 4 is empty) rather than erroring.
    """
    entries = queue(d.syllabus, d.db, current_rubric=d.current_rubric, prior=d.prior,
                    sources_for=d.sources_for, attempt_cap=d.attempt_cap,
                    provenance_source=d.provenance_source)
    items = [
        _rate_question(d, e.subject, e.kind, e.subject_kind, directed=e.directed,
                       rank=e.rank, attempts=e.attempts)
        for e in entries
    ][:budget]

    if len(items) < budget:
        for subject, kind, subject_kind in available_needs(d.syllabus):
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


# --- gallery data provider (spec 5 section 1/4) -----------------------------
#
# Built from Syllabus state, not an apkg: compile() (spec 4) raises
# NotImplementedError today. Kept as a narrow, swappable provider function
# (ReviewContext.cards_provider) so a caller can hand build_app() an
# apkg-faithful renderer once compile() lands, without touching anything
# else here.

def simplified_cards(d: "Derivations") -> list[dict[str, Any]]:
    syllabus = d.syllabus
    words_by_id = {w.id: w for w in syllabus.words}
    targets_by_id = {t.id: t for t in syllabus.targets}
    pairs_by_id = {p.id: p for p in syllabus.pairs}
    graphemes_by_symbol = {g.symbol: g for g in syllabus.graphemes}
    confusions_by_id = {c.id: c for c in syllabus.confusions}

    cards: list[dict[str, Any]] = []
    for index, entry in enumerate(syllabus.order()):
        if entry.kind == "word_target":
            target = targets_by_id.get(entry.id)
            word = words_by_id.get(target.word) if target else None
            if target is None or word is None:
                continue
            best = _best(d, target.word, "picture")
            cards.append({
                "index": index, "id": target.id, "kind": "target",
                "front": {"thai": word.thai, "picture": (_artifact(best.artifact_sha) or {}).get("url")},
                "back": {"meaning": word.meaning},
                "gloss": word.meaning, "voice": None, "drill": None,
            })
        elif entry.kind == "pair":
            pair = pairs_by_id.get(entry.id)
            if pair is None:
                continue
            members = [words_by_id.get(m) for m in pair.members]
            if any(m is None for m in members):
                continue
            confusion = confusions_by_id.get(pair.confusion)
            best = _best(d, members[0].id, "recording")
            other = members[1].thai if len(members) > 1 else None
            cards.append({
                "index": index, "id": pair.id, "kind": "pair",
                "front": {"thai": members[0].thai},
                "back": {"other_thai": other},
                "gloss": " / ".join(f"{m.thai}: {m.meaning}" for m in members),
                "voice": None,
                "drill": {"audio": (_artifact(best.artifact_sha) or {}).get("url"),
                         "thai": members[0].thai, "other_thai": other,
                         "contrast": confusion.id if confusion else pair.confusion},
            })
        elif entry.kind == "grapheme":
            grapheme = graphemes_by_symbol.get(entry.id)
            if grapheme is None:
                continue
            keyword = words_by_id.get(grapheme.keyword)
            cards.append({
                "index": index, "id": entry.id, "kind": "grapheme",
                "front": {"symbol": grapheme.symbol},
                "back": {"sound": grapheme.sound, "keyword": keyword.thai if keyword else None},
                "gloss": keyword.meaning if keyword else None, "voice": None, "drill": None,
            })
        # else: entry.kind == "sentence" -- this provider renders no
        # sentence card yet, skipped rather than raising.
    return cards


# --- writes: notes, drills, answers, supply ---------------------------------

def append_gallery_note(record: RecordWriter, *, card_id: str, kind: str, text: str) -> int:
    """Gallery one-line notes append as learner assessment rows (spec 5
    section 1: "notes append as learner assessment rows via RecordWriter
    (not proof_notes.jsonl)"). role="card-flag" per spec 3's AUTHORITY_ORDER
    (the learner-only "cards (flags)" role).
    """
    role = "card-flag"
    key = LearnerKey(artifact_sha=str(card_id), role=role)
    return record.append(port="assess", backend="learner", key=key, subject=str(card_id),
                         question={"role": role, "kind": kind, "card_id": card_id},
                         answer={"kind": "rating", "rating": None, "note": text})


def append_drill_result(record: RecordWriter, *, confusion: str, pair_id: str,
                        correct: bool) -> int:
    """Pair-drill results append as study-adjacent learner evidence rows
    (spec 5 section 1) -- NOT the `study` table (that's real Anki revlog
    only, spec 2 section 2); this is live gallery-drill evidence, kept in
    `cache` under the learner backend like the rest of the learner's acts.
    Subject = confusion id, so /stats can fold every drill for a confusion
    with one assessments_of(confusion) read.
    """
    key = DrillKey(pair_id=pair_id, confusion=confusion)
    return record.append(port="assess", backend="learner", key=key, subject=confusion,
                         question={"kind": "drill", "pair": pair_id, "confusion": confusion},
                         answer={"correct": bool(correct)})


def append_answer(record: RecordWriter, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The one write path for question-session answers (spec 5 section 1):
    every answer appends one learner cache row keyed by cachekeys.LearnerKey,
    or cachekeys.WaiverKey for a waiver. `payload` shapes:
      rate/reask:  {subject, kind, action: 1-4, artifact_sha?, note?}
      challenger:  {subject, kind, action: "keep"|"switch", artifact_sha?}
      waiver:      {finding: {rule, note_id, artifact_sha?}, waived?, reason?}
    The role a rating is filed under comes back from the question that
    asked it (`role`), or from the need's own kinds (`kind` plus
    `subject_kind`, which is "word" for a payload naming neither).
    Never mutates or deletes a row (append-only, spec 2): calling this
    twice with an identical payload appends two rows, but every derivation
    over the cache folds newest-wins, so the DERIVED state (current_best,
    exhausted, ...) after the second call is identical to after the first
    -- idempotent in effect, not in row count.
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
    role = payload.get("role") or role_for(kind, payload.get("subject_kind", "word"))
    action = payload.get("action")

    if action in ("keep", "switch"):
        if action == "keep":
            return {"ok": True, "kind": "challenger", "action": "keep"}
        artifact_sha = payload.get("artifact_sha") or payload.get("challenger_sha")
        rating = payload.get("rating", "acceptable")
    else:
        rating = payload.get("rating") or ACTION_RATINGS.get(int(action))
        artifact_sha = None if rating == "unacceptable-none" else payload.get("artifact_sha")

    if rating not in LEARNER_RANK:
        raise ValueError(f"unknown rating {rating!r}")

    answer: dict[str, Any] = {"value": rating}
    if payload.get("note"):
        answer["note"] = payload["note"]
    key = LearnerKey(artifact_sha=artifact_sha, role=role)
    ts = record.append(port="assess", backend="learner", key=key, subject=subject,
                       question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                                "kind": "rating"},
                       answer=answer)
    return {"ok": True, "ts": ts, "rating": rating, "artifact_sha": artifact_sha}


def append_supply(ctx: "ReviewContext", payload: Mapping[str, Any]) -> dict[str, Any]:
    """spec 5 section 1 kind 2's supply action (also usable stand-alone for
    any subject): {subject, kind, source: "path"|"url", value, note?, ext?}.
    A URL goes through the spec-3 imgfetch Provider path -- cache-first,
    appends its own `provide` row via Provider.ask, exactly as any other
    backend would. A local file path is a direct learner act with no
    Provider backend behind it (there is no cache key to ask against: it is
    read once and written to MediaStore), so it appends no provide row --
    only the learner supply act below. Either way the artifact lands with
    learner provenance and an implicit use-this rating (spec 5 section 1).
    """
    subject, kind = payload["subject"], payload["kind"]
    source = payload["source"]

    if source == "url":
        provider = Provider(ctx.record, ctx.cache,
                            {"imgfetch": FetchBackend(media=ctx.media_store,
                                                      fetcher=ctx.url_fetcher)})
        answer = provider.ask("imgfetch", Question(subject=subject, provides=f"{kind}-bytes",
                                                    params={"url": payload["value"]}, kind=kind))
        if not answer.items:
            return {"ok": False, "error": "fetch produced no artifact"}
        artifact_sha = answer.items[0]["sha"]
    elif source == "path":
        data = Path(payload["value"]).read_bytes()
        artifact_sha = ctx.media_store.write(data, payload.get("ext", "jpg"))
    else:
        raise ValueError(f"unknown supply source {source!r}")

    role = payload.get("role") or role_for(kind, payload.get("subject_kind", "word"))
    key = LearnerKey(artifact_sha=artifact_sha, role=role)
    answer_row: dict[str, Any] = {
        "value": "unacceptable-use-this",
        "provenance": {"source": "learner", "origin": source},
    }
    if payload.get("note"):
        answer_row["note"] = payload["note"]
    ts = ctx.record.append(port="assess", backend="learner", key=key, subject=subject,
                           question={"role": role, "artifact_sha": artifact_sha, "rubric": None,
                                    "kind": "rating"},
                           answer=answer_row)
    return {"ok": True, "ts": ts, "artifact_sha": artifact_sha}


# --- stats (spec 5 section 3) -----------------------------------------------

@dataclass
class SessionStats:
    """Per-process, per-session counters -- "per-session" (spec 5 section
    3) is exactly the review server's process lifetime; nothing here is
    persisted (there is no store for it, and it is not the record of
    truth: every answer is already durable via RecordWriter before this
    counter is touched).
    """
    answered: int = 0
    queued: int = 0


def _drill_stats(d: "Derivations") -> dict[str, dict[str, int]]:
    """Per-confusion correct/total over the gallery drill rows this module
    itself appends (append_drill_result) -- live drill evidence, not a
    policy derivation.
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
    """Spec 5 section 3: per-session (answered/queued, per-confusion drill
    accuracy, exhausted-remaining count) and per-deck (current-best
    coverage per need, learner good/acceptable/unacceptable counts) --
    every count derived under `d`'s parameters, the run's own.
    `pending`/`sentences_adopted` come from the newest run.py
    (port="run", backend="runreport") row when one exists, else 0 -- the
    same row run._persist_report appends after every run() call.
    `run_report_history` stays empty: run._persist_report appends one row
    per run() call, but this module's read side only reads the newest one
    back -- no aggregation over the full history is implemented here yet.
    """
    coverage: dict[str, dict[str, int]] = {}
    ratings = {"good": 0, "acceptable": 0, "unacceptable": 0}
    exhausted_count = 0

    for subject, kind, subject_kind in available_needs(d.syllabus):
        best = _best(d, subject, kind)
        bucket = coverage.setdefault(kind, {"covered": 0, "total": 0})
        bucket["total"] += 1
        if best.rank >= _ACCEPTABLE_FLOOR:
            bucket["covered"] += 1
        if _exhausted(d, subject, kind).exhausted:
            exhausted_count += 1

        rated = ratings_for_role(d.db.assessments_of(subject), role_for(kind, subject_kind))
        if rated:
            value = max(rated, key=lambda r: r.ts).answer["value"]
            if value == "good":
                ratings["good"] += 1
            elif value == "acceptable":
                ratings["acceptable"] += 1
            else:
                ratings["unacceptable"] += 1

    runreport = d.db.latest("run", "runreport", "runreport")
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
        "run_report_history": [],
    }


# --- HTTP layer --------------------------------------------------------------

def _find_media_file(media_store: MediaStore, sha: str) -> Path | None:
    matches = sorted((media_store.root / "objects").glob(f"{sha}.*"))
    return matches[0] if matches else None


@dataclass
class ReviewContext:
    """One review session over one deck: the deck's Derivations (the run's
    own parameters), the StudyReader the re-ask questions read, the
    learner-attention budget one session serves, and the counters that
    session keeps. Every derivation the surface shows goes through the
    methods below, so no endpoint can measure a fold under anything but
    `derivations`.
    """
    derivations: "Derivations"
    study: StudyReader | None = None
    learner_budget: int = DEFAULT_LEARNER_BUDGET
    url_fetcher: Callable[[str], tuple[bytes, str]] | None = None
    cards_provider: Callable[["Derivations"], list[dict[str, Any]]] = field(
        default=simplified_cards)
    session: SessionStats = field(default_factory=SessionStats)

    def __post_init__(self) -> None:
        if self.url_fetcher is None:
            self.url_fetcher = tool_fetcher("imgfetch")

    @property
    def syllabus(self) -> Syllabus:
        return self.derivations.syllabus

    @property
    def cache(self) -> CacheReader:
        return self.derivations.db

    @property
    def record(self) -> RecordWriter:
        return self.derivations.db

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
            path = _find_media_file(ctx.media_store, sha)
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
                    result = append_answer(ctx.record, payload)
                    ctx.session.answered += 1
                    self._send_json(result)
                elif parsed.path == "/api/supply":
                    result = append_supply(ctx, payload)
                    ctx.session.answered += 1
                    self._send_json(result)
                elif parsed.path == "/api/note":
                    card_id = payload.get("card_id", payload.get("id"))
                    ts = append_gallery_note(ctx.record, card_id=str(card_id),
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


def load_context(deck_dir: str | Path, *, learner_budget: int = DEFAULT_LEARNER_BUDGET
                 ) -> ReviewContext:
    """Wire a ReviewContext from a real deck directory (spec 2 section 1
    layout: curated/*.yaml + syllabus.db + media/).

    wiring.load_derivations is the assembly build_sourcing hands the run:
    the Syllabus with its media index, one db connection serving as
    CacheReader, RecordWriter and StudyReader, and the deck's own rubric,
    provenance prior, Source roster and attempt cap. The screen adds no
    parameter of its own, so what it shows and what the run derives cannot
    drift apart.

    wiring is imported inside the function: reviewserver is on cli.py's
    import path and wiring reaches for provider/assessor/transport, which
    a module-level import would pull in to serve a page.
    """
    from .wiring import load_derivations

    derivations = load_derivations(deck_dir)
    return ReviewContext(derivations=derivations, study=derivations.db,
                         learner_budget=learner_budget)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, type=Path, help="deck directory (spec 2 layout)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--budget", type=int, default=DEFAULT_LEARNER_BUDGET,
                        help="learner-attention session budget (spec 3 section 4)")
    args = parser.parse_args(argv)

    ctx = load_context(args.deck, learner_budget=args.budget)
    serve(ctx, args.port)
    return 0


# ---------------------------------------------------------------------------
# the page: inline CSS + JS, no external resources, keyboard-first
# (spec 5 section 2: "1-4 rate, n note, arrows navigate, g gloss, s stats")
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
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
  .current-artifact img {
    max-width: min(90vw, 640px); max-height: 55vh; width: auto; height: auto;
    border-radius: 6px;
  }
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

  function thumb(art, cls) {
    var img = el("img", { src: art.url, "data-sha": art.sha });
    img.addEventListener("click", function () { openOverlay(art.url); });
    return img;
  }

  function renderRate(q, box) {
    box.appendChild(el("div", {}, q.subject + " (" + q.kind + ")"));
    if (q.query) { box.appendChild(el("div", { "class": "query" }, "query: " + q.query)); }
    if (q.current) {
      var cur = el("div", { "class": "current-artifact" });
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
    var labels = { 1: "1 unacceptable-none", 2: "2 unacceptable-use-this",
                  3: "3 acceptable", 4: "4 good" };
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
    tried.appendChild(el("h4", {}, "phrases tried"));
    tried.appendChild(el("div", {}, (q.tried.phrases || []).join(", ") || "none"));
    tried.appendChild(el("h4", {}, "sources"));
    tried.appendChild(el("div", {}, (q.tried.sources || []).join(", ") || "none"));
    tried.appendChild(el("h4", {}, "judge reasons"));
    tried.appendChild(el("div", {}, (q.tried.judge_reasons || []).join("; ") || "none"));
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
    var labels = { 1: "1 unacceptable-none", 2: "2 unacceptable-use-this",
                  3: "3 acceptable", 4: "4 good" };
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
    if (card.drill) { renderDrillCard(card, box); main.appendChild(box); saveProgress(); return; }

    if (card.front.thai) { box.appendChild(el("div", { "class": "thai" }, card.front.thai)); }
    if (card.front.symbol) { box.appendChild(el("div", { "class": "thai" }, card.front.symbol)); }
    if (card.front.picture) {
      var img = el("img", { src: card.front.picture });
      img.style.maxWidth = "min(90vw, 480px)";
      img.style.maxHeight = "50vh";
      box.appendChild(img);
    }
    if (glossOn && card.gloss) { box.appendChild(el("div", { "class": "gloss-chip" }, card.gloss)); }
    box.appendChild(el("div", { "class": "verdict" }, "space to reveal"));
    main.appendChild(box);
    saveProgress();
  }

  function revealGallery() {
    if (revealed || !galleryCards.length) { return; }
    var card = galleryCards[gIdx];
    if (card.drill) { return; }
    revealed = true;
    var box = document.querySelector("#main .card-box");
    var back = el("div", { "class": "thai" },
      card.back.meaning || card.back.other_thai || card.back.sound || card.back.keyword || "");
    box.appendChild(back);
  }

  function renderDrillCard(card, box) {
    box.appendChild(el("div", { "class": "thai" }, "which one did you hear?"));
    var choices = el("div", { "class": "actions" });
    var options = [card.drill.thai, card.drill.other_thai];
    if (Math.random() < 0.5) { options.reverse(); }
    options.forEach(function (text) {
      var btn = el("button", {}, text);
      btn.addEventListener("click", function () {
        var correct = text === card.drill.thai;
        postJson("/api/drill", { confusion: card.drill.contrast, pair: card.id, correct: correct })
          .then(function () { setTimeout(next, 400); });
      });
      choices.appendChild(btn);
    });
    box.appendChild(choices);
    if (card.drill.audio) {
      var audio = new Audio(card.drill.audio);
      audio.play().catch(function () {});
    }
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
      postJson("/api/note", { card_id: card.id, kind: card.kind, text: text });
    });
  }

  function openDirectionBox(q) {
    openBox("directionInput", "directionText", function (text) {
      postJson("/api/answer", { subject: q.subject, kind: q.kind,
                                subject_kind: q.subject_kind, role: q.role, action: 3,
                                rating: "unacceptable-use-this", note: text })
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
        row.appendChild(el("td", {}, c.covered + " / " + c.total));
        table.appendChild(row);
      });
      panel.appendChild(table);
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


if __name__ == "__main__":
    raise SystemExit(main())
