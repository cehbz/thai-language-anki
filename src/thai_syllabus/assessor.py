"""The Assess port (spec 3 section 1/2): Assessor.ask(backend, question)
-> Verdict, the same cache-first shape as provider.py's Provider.

Backends: judge (one implementation, three transports -- cli/api/batch --
selected by config), mechanical (ground truth for what it checks, e.g.
recording duration/format), listener (NOT implemented -- spec section 7:
"calibration first"), learner (read-side only; rows arrive via the
feedback surfaces, same as provider.py's learner -- ask() raises).
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .cachekeys import sha
from .ports import CacheReader, RecordWriter
from .transport import TransportError

__all__ = [
    "AssessQuestion", "Verdict", "RawVerdict", "AssessBackend",
    "Assessor", "LearnerAskNotSupported", "BatchPending",
    "JudgeBackend", "AUTHORITY_ORDER",
    "MechanicalBackend", "duration_mechanical_backend",
    "format_mechanical_backend", "ffprobe_duration_seconds",
]


# --- the port contract (spec 3 section 1) -----------------------------------

@dataclass(frozen=True)
class AssessQuestion:
    subject: str
    role: str
    artifact_sha: str | None = None
    rubric: str | None = None  # machine backends only
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    value: Any
    cost: float = 0.0
    ts: int = 0
    evidence: str | None = None
    suggestion: str | None = None


@dataclass(frozen=True)
class RawVerdict:
    value: Any
    cost: float = 0.0
    evidence: str | None = None
    suggestion: str | None = None


class LearnerAskNotSupported(RuntimeError):
    """Assessor.ask("learner", ...) always raises this: the learner
    backend is read-side only (newest-wins, authority per (backend, role)
    -- see AUTHORITY_ORDER); rows arrive via RecordWriter from the
    feedback surfaces, never through ask().
    """


class BatchPending(RuntimeError):
    """Assessor.ask_batch() raised this: a judge batch is submitted (or
    still processing) and some subjects remain unresolved this call. The
    batch id and its request set are already persisted as a batch-pending
    cache row (spec 3 section 2), so a later call with the same questions
    resumes rather than resubmitting.
    """
    def __init__(self, batch_id: str, pending_subjects: list[str]):
        super().__init__(f"batch {batch_id!r} not ready: "
                         f"{len(pending_subjects)} subject(s) still pending")
        self.batch_id = batch_id
        self.pending_subjects = pending_subjects


@runtime_checkable
class AssessBackend(Protocol):
    def cache_key(self, question: AssessQuestion) -> str: ...
    def fetch(self, question: AssessQuestion) -> RawVerdict: ...  # may raise -- not cached


# --- authority (spec 3 section 2's "authority" column, as data) ------------
#
# Per role, backends ordered most- to least-authoritative. Not a single
# global ranking -- authority is per (backend, role) (domain-language doc,
# "Recording QA placement" amendment): the learner is final on fit/quality/
# waivers but unqualified on tone correctness, where mechanical is ground
# truth and listener ranks only once calibrated (spec section 7 -- absent
# from this table until a deployment's providers.yaml supplies a measured
# rank, which is why "listener" does not appear in the recording-for-word
# row below: an uncalibrated listener contributes nothing to current_best).

AUTHORITY_ORDER: dict[str, tuple[str, ...]] = {
    "picture-for-word": ("learner", "judge"),
    "scene-for-sentence": ("learner", "judge"),
    "sentence-for-target": ("learner", "judge"),
    "finding-waiver": ("learner",),
    "card-flag": ("learner",),
    "recording-for-word": ("mechanical", "judge"),  # learner may flag, never outrank
}


class Assessor:
    """Cache-first ask() over injected backends (spec 3 section 1)."""

    def __init__(self, record: RecordWriter, cache: CacheReader,
                backends: Mapping[str, AssessBackend]):
        self._record = record
        self._cache = cache
        self._backends = dict(backends)

    def ask(self, backend: str, question: AssessQuestion) -> Verdict:
        if backend == "learner":
            raise LearnerAskNotSupported(
                "the learner Assess backend has no ask(); its rows arrive "
                "via RecordWriter from the feedback surfaces")
        if backend == "listener":
            raise NotImplementedError(
                "the listener Assess backend is not implemented (spec 3 "
                "section 7: calibration first)")
        impl = self._backends[backend]
        key = impl.cache_key(question)
        cached = self._cache.latest("assess", backend, key)
        if cached is not None:
            return _verdict_from_cached(cached)
        raw = impl.fetch(question)  # transport errors propagate uncached
        ts = self._append_verdict(backend, key, question, raw)
        return Verdict(value=raw.value, cost=raw.cost, ts=ts,
                       evidence=raw.evidence, suggestion=raw.suggestion)

    def _append_verdict(self, backend: str, key: str, question: AssessQuestion,
                        raw: RawVerdict) -> int:
        answer: dict[str, Any] = {"value": raw.value}
        if raw.evidence is not None:
            answer["evidence"] = raw.evidence
        if raw.suggestion is not None:
            answer["suggestion"] = raw.suggestion
        return self._record.append(
            port="assess", backend=backend, key=key, subject=question.subject,
            question={"role": question.role, "artifact_sha": question.artifact_sha,
                     "rubric": question.rubric}, answer=answer, cost=raw.cost)

    # --- judge's batch transport: many questions, one submission ---------

    def ask_batch(self, backend: str, questions: Sequence[AssessQuestion]
                  ) -> dict[str, Verdict]:
        """Resolves every question already cached; for the rest, resumes a
        pending batch for exactly this remaining request set (submitting a
        new one if none exists) -- persisted as a batch-pending cache row,
        never a sidecar file (spec 3 section 2). Raises BatchPending
        listing what remains unresolved this call.
        """
        impl = self._backends[backend]
        resolved: dict[str, Verdict] = {}
        misses: list[AssessQuestion] = []
        keys: dict[str, str] = {}
        for q in questions:
            key = impl.cache_key(q)
            keys[q.subject] = key
            cached = self._cache.latest("assess", backend, key)
            if cached is not None:
                resolved[q.subject] = _verdict_from_cached(cached)
            else:
                misses.append(q)
        if not misses:
            return resolved

        request_set_key = _batch_request_set_key(keys[q.subject] for q in misses)
        pending = self._cache.latest("assess", backend, request_set_key)
        if pending is not None and pending.answer.get("kind") == "batch-pending":
            batch_id = pending.answer["batch_id"]
        else:
            requests = {q.subject: impl.prompt_builder(q) for q in misses}
            batch_id = impl.batch_transport.submit(requests)
            self._record.append(
                port="assess", backend=backend, key=request_set_key,
                subject=request_set_key, question={"subjects": sorted(requests)},
                answer={"kind": "batch-pending", "batch_id": batch_id}, cost=0.0)
            raise BatchPending(batch_id, [q.subject for q in misses])

        status = impl.batch_transport.status(batch_id)
        if status != "ended":
            raise BatchPending(batch_id, [q.subject for q in misses])

        results = impl.batch_transport.results(batch_id)
        still_pending: list[str] = []
        for q in misses:
            text = results.get(q.subject)
            if text is None:
                still_pending.append(q.subject)
                continue
            raw = impl.parse_response(text)
            raw = RawVerdict(value=raw.value, evidence=raw.evidence,
                             suggestion=raw.suggestion, cost=impl.cost_per_call)
            ts = self._append_verdict(backend, keys[q.subject], q, raw)
            resolved[q.subject] = Verdict(value=raw.value, cost=raw.cost, ts=ts,
                                          evidence=raw.evidence, suggestion=raw.suggestion)
        if still_pending:
            raise BatchPending(batch_id, still_pending)
        return resolved


def _verdict_from_cached(cached) -> Verdict:
    a = cached.answer
    return Verdict(value=a["value"], cost=0.0, ts=cached.ts,
                   evidence=a.get("evidence"), suggestion=a.get("suggestion"))


def _batch_request_set_key(keys) -> str:
    return "judge-batch:" + sha(",".join(sorted(keys)))


# --- judge: one implementation, three transports ----------------------------
# key = "judge:sha(RUBRIC):ARTIFACT_SHA:ROLE". ARTIFACT_SHA is already a
# content hash of the artifact bytes (Picture/Recording are content-
# addressed, spec 1 section 1) -- re-hashing it would just reproduce the
# same value, so it goes into the key as-is rather than sha(sha(...)).

def _default_judge_prompt(question: AssessQuestion) -> str:
    lines = [f"Role: {question.role}", f"Rubric: {question.rubric or ''}"]
    if question.artifact_sha:
        lines.append(f"Artifact: {question.artifact_sha}")
    if question.params:
        lines.append(f"Params: {json.dumps(dict(question.params), sort_keys=True)}")
    lines.append(
        'Respond with a JSON object: {"value": <bool>, "evidence": <string>, '
        '"suggestion": <string or null>}.')
    return "\n".join(lines)


def _default_parse_judge_response(text: str) -> RawVerdict:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        stripped = text.strip().lower()
        if stripped in ("true", "false"):
            return RawVerdict(value=stripped == "true")
        return RawVerdict(value=text.strip())
    return RawVerdict(value=data.get("value"), evidence=data.get("evidence"),
                      suggestion=data.get("suggestion"))


@dataclass
class JudgeBackend:
    model: str
    transport: str  # "cli" | "api" | "batch" -- label, selects which of the below is used
    complete: Callable[[str], str] | None = None  # cli/api: ClaudeCliTransport/ClaudeApiTransport.complete
    batch_transport: Any = None  # ClaudeBatchTransport, batch transport only
    prompt_builder: Callable[[AssessQuestion], str] = field(default=_default_judge_prompt)
    parse_response: Callable[[str], RawVerdict] = field(default=_default_parse_judge_response)
    cost_per_call: float = 0.0

    def cache_key(self, question: AssessQuestion) -> str:
        return f"judge:{sha(question.rubric or '')}:{question.artifact_sha or '-'}:{question.role}"

    def fetch(self, question: AssessQuestion) -> RawVerdict:
        if self.complete is None:
            raise RuntimeError(
                "this JudgeBackend has no single-question transport "
                "(configured for batch only) -- use Assessor.ask_batch")
        prompt = self.prompt_builder(question)
        text = self.complete(prompt)
        raw = self.parse_response(text)
        return RawVerdict(value=raw.value, evidence=raw.evidence,
                          suggestion=raw.suggestion, cost=self.cost_per_call)


# --- mechanical: ground truth for what it checks ----------------------------

@dataclass
class MechanicalBackend:
    """Generic mechanical Assessor backend. key_fn/evaluate are both
    injectable so each concrete check supplies its own parameter-explicit
    key (spec 3 roster: "parameter-explicit where expressible ...
    mech:CHECK:CODE_VERSION:sha(ARTIFACT) only where no parameters express
    the question").
    """
    key_fn: Callable[[AssessQuestion], str]
    evaluate: Callable[[AssessQuestion], RawVerdict]

    def cache_key(self, question: AssessQuestion) -> str:
        return self.key_fn(question)

    def fetch(self, question: AssessQuestion) -> RawVerdict:
        return self.evaluate(question)


def ffprobe_duration_seconds(path: str, runner: Callable[..., Any] = subprocess.run) -> float:
    """Ported from thai_deck_gen/media/ffmpeg.py's duration_ok, split into
    a pure duration lookup so the mechanical backend owns the pass/fail
    range check (injectable `runner`, no thai_deck_gen import).
    """
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
          "-of", "json", str(path)]
    result = runner(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TransportError(f"ffprobe failed on {path!r}: {result.stderr}")
    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise TransportError(f"ffprobe returned unparseable output for {path!r}: {e}") from e


def duration_mechanical_backend(
        *, lo: float = 0.2, hi: float = 5.0,
        resolve_path: Callable[[str | None], str],
        duration_of: Callable[[str], float] | None = None,
        runner: Callable[..., Any] = subprocess.run) -> MechanicalBackend:
    duration_of = duration_of or (lambda path: ffprobe_duration_seconds(path, runner=runner))

    def key_fn(question: AssessQuestion) -> str:
        return f"mech:duration:{lo}-{hi}:{question.artifact_sha or '-'}"

    def evaluate(question: AssessQuestion) -> RawVerdict:
        path = resolve_path(question.artifact_sha)
        duration = duration_of(path)
        ok = lo <= duration <= hi
        return RawVerdict(value=ok, evidence=f"duration={duration:.3f}s")

    return MechanicalBackend(key_fn=key_fn, evaluate=evaluate)


def format_mechanical_backend(
        *, expected_ext: str, code_version: str = "v1",
        resolve_ext: Callable[[str | None], str]) -> MechanicalBackend:
    def key_fn(question: AssessQuestion) -> str:
        return f"mech:format:{code_version}:{question.artifact_sha or '-'}"

    def evaluate(question: AssessQuestion) -> RawVerdict:
        ext = resolve_ext(question.artifact_sha)
        ok = ext == expected_ext
        return RawVerdict(value=ok, evidence=f"ext={ext!r}, expected={expected_ext!r}")

    return MechanicalBackend(key_fn=key_fn, evaluate=evaluate)
