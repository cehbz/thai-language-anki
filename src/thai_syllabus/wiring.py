"""Wiring: build the Provide/Assess backend rosters and Budget defaults
from curated/providers.yaml (spec 3 section 5), and assemble a Syllabus
from a deck directory's curated files + db-backed ports (spec 1/2).
cli.py's module docstring names this gap: compile/run stayed library-level
"until their configs settle" -- this module is that settling.

Secrets are resolved lazily (secrets.SecretStore's own contract, spec 3
section 5): a backend whose secret is never touched must never cause a
file/1Password read merely by being constructed into the roster.
Secret-backed backends (pexels, forvo, tts, the judge/llm api transport)
are wrapped in `_LazyBackend` / `_LazyTransport`, which defer building the
real backend/transport object -- and therefore calling
`SecretStore.get()` -- until their first `cache_key`/`fetch`/`complete`
call. `Provider.__init__`/`Assessor.__init__` both do `dict(backends)`
over the mapping they're given, which forces every VALUE already in that
dict to exist as an object -- but never calls any METHOD on those
objects -- so wrapping only the secret-needing backends in a thin
lazy-dispatch object (itself trivially constructible with no secret
access) keeps the free backends eager and the paid ones lazy without
fighting that `dict()` call.

Scope decisions this module had to make that providers.yaml's terse shape
(spec 3 section 5) and the two read specs leave implicit -- not spec
violations, just latitude the terse text left to fill in:

- The llm Provider backend (sentence/phrase/entry drafting) has no
  section of its own in providers.yaml -- section 5 lists only "judge
  transport + model" as this project's one configured way to reach an
  LLM. Read as: the llm backend reuses that SAME transport+model (one
  Claude account, one configured transport), registered under three
  backend names (llm-sentence/llm-phrase/llm-entry) rather than one
  "llm" name -- LlmBackend.producer is fixed per instance (provider.py)
  and Provider looks a backend up by name, not by a per-call producer
  argument. A "batch" transport has no single-question `.complete()`
  (assessor.JudgeBackend.fetch's own docstring: "configured for batch
  only -- use Assessor.ask_batch"), so the llm backend is not registered
  at all when judge.transport == "batch" (build_levers mirrors this: no
  "sentence" lever either).
- build_assessor(cfg, db) registers only "judge" -- the one backend
  providers.yaml actually configures (transport + model). "mechanical"
  needs a MediaStore to turn an artifact_sha into a file path (ffprobe
  duration / extension checks) and providers.yaml carries no
  configuration for it at all (MechanicalBackend's key_fn/evaluate are
  injectable CODE, not config); "listener" is unimplemented and
  "learner" is read-side-only -- Assessor.ask() already special-cases
  both of those itself (assessor.py), so neither needs a roster entry.
- load_syllabus's MediaIndex: store.py's `media` table is provenance-only
  (spec 2) -- the word/confusion -> media RELATIONSHIP lives in `cache`
  rows (spec 3's own territory), so `_DbMediaIndex` below derives
  has_picture/recording_speakers/rendition_speakers from
  derivations.current_best over the db, the same source compile.py
  already trusts for "what media does this subject have".
- load_syllabus's sentences: SyllabusDb had a writer (add_sentence) but
  no reader for the `sentences` table at all. Added
  `SyllabusDb.all_sentences()` (store.py) as the minimal read side this
  needed.
- Frequency map: data/frequency_th.txt is project input data living
  outside any one deck's curated/ directory (curated.py's own
  docstring), and load_syllabus(deck_root) has no separate project-root
  parameter. Defaults to `deck_root/data/frequency_th.txt` (absent ->
  empty map, which Syllabus.order() already degrades gracefully to its
  documented float('inf') fallback), with an optional `frequency_path`
  override for a caller keeping the shared corpus elsewhere.
- Tokenizer: pythainlp is imported lazily, and only used when actually
  importable (a plain `try/except ImportError` at call time, guarding
  the only pythainlp import in this module); otherwise load_syllabus
  passes no `tokenizer=` at all, so Syllabus's own private
  whitespace-tokenizer default applies unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .assessor import AssessBackend, Assessor, JudgeBackend
from .curated import ProvidersConfig, load_curated, load_frequency_map, rulebook_file_text
from .derivations import current_best
from .entities import MinimalPair
from .ids import ConfusionId, WordId
from .provider import (
    Backend,
    ForvoBackend,
    ImgfetchBackend,
    LlmBackend,
    Provider,
    Question,
    TtsBackend,
    openverse_backend,
    pexels_backend,
    subprocess_curl_fetcher,
    wikimedia_backend,
)
from .run import FORVO_DEFAULT_DAILY_BUDGET, LEARNER_DEFAULT_SESSION_BUDGET, Budget, Lever
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus
from .transport import ClaudeApiTransport, ClaudeBatchTransport, ClaudeCliTransport
from .tts import pick_voice

__all__ = ["build_provider", "build_assessor", "default_budgets", "build_levers",
           "load_syllabus"]


# --- laziness helpers -------------------------------------------------------

class _LazyBackend:
    """A Backend (cache_key + fetch) that defers building the real backend
    -- and therefore any SecretStore.get() it needs -- until first use.
    """
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._impl: Any = None

    def _resolve(self) -> Any:
        if self._impl is None:
            self._impl = self._factory()
        return self._impl

    def cache_key(self, question: Any) -> str:
        return self._resolve().cache_key(question)

    def fetch(self, question: Any) -> Any:
        return self._resolve().fetch(question)


class _LazyTransport:
    """A `.complete(prompt) -> str`-shaped object that defers building the
    real transport until first call -- used both as LlmBackend.transport
    (an object with .complete) and, via its bound `.complete` method, as
    JudgeBackend.complete (a bare callable) -- see _judge_transport below.
    """
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._impl: Any = None

    def complete(self, prompt: str) -> str:
        if self._impl is None:
            self._impl = self._factory()
        return self._impl.complete(prompt)


class _LazyBatchTransport:
    """Same deferral for the batch transport's three-method shape
    (submit/status/results), so a judge configured for batch never
    resolves its secret until a batch is actually submitted or polled.
    """
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._impl: Any = None

    def _resolve(self) -> Any:
        if self._impl is None:
            self._impl = self._factory()
        return self._impl

    def submit(self, requests: dict[str, str]) -> str:
        return self._resolve().submit(requests)

    def status(self, batch_id: str) -> str:
        return self._resolve().status(batch_id)

    def results(self, batch_id: str) -> dict[str, str | None]:
        return self._resolve().results(batch_id)


def _claude_transport(cfg: ProvidersConfig, secrets) -> _LazyTransport | None:
    """A lazy `.complete(prompt)` transport for the ONE Claude account
    providers.yaml configures (judge.transport/model) -- shared by the
    judge Assessor backend's cli/api transport and the llm Provider
    backend (see module docstring). None for "batch" (no single-question
    transport exists there).
    """
    kind = cfg.judge.transport
    if kind == "cli":
        return _LazyTransport(lambda: ClaudeCliTransport())
    if kind == "api":
        return _LazyTransport(lambda: ClaudeApiTransport(
            api_key=secrets.get("anthropic") or "", model=cfg.judge.model))
    return None


# --- build_provider ----------------------------------------------------

def build_provider(cfg: ProvidersConfig, db: SyllabusDb, media_store: MediaStore,
                   *, secret_store=None) -> Provider:
    """The Provide port's backend roster (spec 3 section 2), wired from
    providers.yaml (spec 3 section 5): search_proxy for the image-search
    backends, imgfetch_path for the curl fetcher, the tts voice pools, and
    the shared judge/llm transport+model for llm-*.
    """
    secrets = secret_store if secret_store is not None else cfg.secret_store()

    backends: dict[str, Backend] = {
        "openverse": openverse_backend(search_proxy=cfg.search_proxy),
        "wikimedia": wikimedia_backend(search_proxy=cfg.search_proxy),
        "imgfetch": ImgfetchBackend(
            media=media_store,
            fetcher=subprocess_curl_fetcher(binary=cfg.imgfetch_path or "curl")),
        "pexels": _LazyBackend(lambda: pexels_backend(
            api_key=secrets.get("pexels") or "", search_proxy=cfg.search_proxy)),
        "forvo": _LazyBackend(lambda: ForvoBackend(api_key=secrets.get("forvo") or "")),
        "tts": _LazyBackend(lambda: TtsBackend(
            tts=_lazy_google_tts(secrets),
            voices=list(cfg.tts_male_voices) + list(cfg.tts_female_voices),
            media=media_store, pick_voice=pick_voice)),
    }

    llm_transport = _claude_transport(cfg, secrets)
    if llm_transport is not None:
        for producer, name in (("sentence-drafter", "llm-sentence"),
                               ("phrase-drafter", "llm-phrase"),
                               ("entry-drafter", "llm-entry")):
            backends[name] = LlmBackend(producer=producer, model=cfg.judge.model,
                                        transport=llm_transport)

    return Provider(record=db, cache=db, backends=backends)


def _lazy_google_tts(secrets):
    """tts.Tts is a Protocol (`.synthesize(text, voice) -> bytes`);
    TtsBackend.fetch calls `self.tts.synthesize(...)` directly, so this
    thin object -- not GoogleTts itself -- is what actually goes into the
    TtsBackend, deferring the google_tts secret until synthesis happens.
    """
    from .tts import GoogleTts

    @dataclass
    class _LazyGoogleTts:
        _impl: Any = None

        def synthesize(self, text: str, voice: str) -> bytes:
            if self._impl is None:
                self._impl = GoogleTts(api_key=secrets.get("google_tts") or "")
            return self._impl.synthesize(text, voice)

    return _LazyGoogleTts()


# --- build_assessor ----------------------------------------------------

def build_assessor(cfg: ProvidersConfig, db: SyllabusDb, *, secret_store=None) -> Assessor:
    """The Assess port's backend roster (spec 3 section 2) providers.yaml
    actually configures: "judge", transport+model. "mechanical" needs a
    MediaStore this signature doesn't take (see module docstring);
    "listener"/"learner" need no roster entry -- Assessor.ask() already
    special-cases both.
    """
    secrets = secret_store if secret_store is not None else cfg.secret_store()
    backends: dict[str, AssessBackend] = {"judge": _build_judge_backend(cfg, secrets)}
    return Assessor(record=db, cache=db, backends=backends)


def _build_judge_backend(cfg: ProvidersConfig, secrets) -> JudgeBackend:
    kind = cfg.judge.transport
    complete = None
    batch_transport = None
    if kind == "batch":
        batch_transport = _LazyBatchTransport(lambda: ClaudeBatchTransport(model=cfg.judge.model))
    else:
        transport = _claude_transport(cfg, secrets)
        complete = transport.complete if transport is not None else None
    return JudgeBackend(model=cfg.judge.model, transport=kind, complete=complete,
                        batch_transport=batch_transport)


# --- budgets -------------------------------------------------------------

def default_budgets(cfg: ProvidersConfig) -> dict[str, Budget]:
    """Budget per backend (spec 3 section 4): the two documented defaults
    (forvo 450/day, learner 20/session) layered under whatever
    providers.yaml's `quotas` section configures -- a configured entry
    overrides the matching default; every other configured backend just
    adds its own Budget.
    """
    budgets: dict[str, Budget] = {
        "forvo": FORVO_DEFAULT_DAILY_BUDGET,
        "learner": LEARNER_DEFAULT_SESSION_BUDGET,
    }
    for backend, quota in cfg.quotas.items():
        budgets[backend] = Budget(max_asks=quota.get("max_asks"),
                                  max_cost=quota.get("max_cost"))
    return budgets


# --- levers: (subject, kind) -> Question, cheapest-first per kind ---------

def build_levers(syllabus: Syllabus, provider: Provider, cfg: ProvidersConfig
                 ) -> dict[str, list[Lever]]:
    """Escalation order per gap kind (spec 3 section 4: "escalate backends
    cheapest-first"), for the kinds derivations._gap_candidates actually
    produces that this deliverable's read specs give enough to wire:
    picture (openverse, wikimedia, pexels -- free-ish HTTP before a paid
    key) and recording (forvo, a native lookup, before tts synthesis).
    "sentence" is wired only when a single-question llm transport exists
    (build_provider's own batch-transport exclusion, mirrored here).
    "rendition" and "grapheme-keyword" are left unlevered: run() already
    tolerates a kind with no registered levers (it just shows up in
    RunReport.available, never attempted -- see run.py's own test
    `test_available_counts_queued_subjects_that_were_never_attempted`),
    and sourcing pair renditions / grapheme keyword data is its own
    design this deliverable's two specs don't fix.
    """
    def picture_question(subject: str, kind: str) -> Question:
        w = syllabus.find_word(WordId(subject))
        query = w.meaning if w and w.meaning else subject
        return Question(subject=subject, provides="picture", params={"query": query})

    def forvo_question(subject: str, kind: str) -> Question:
        w = syllabus.find_word(WordId(subject))
        return Question(subject=subject, provides="recording",
                        params={"word": w.thai if w else subject})

    def tts_question(subject: str, kind: str) -> Question:
        w = syllabus.find_word(WordId(subject))
        return Question(subject=subject, provides="recording",
                        params={"text": w.thai if w else subject})

    def sentence_question(subject: str, kind: str) -> Question:
        w = syllabus.find_word(WordId(subject))
        if w is not None:
            prompt = (f"Write one natural, colloquial Thai sentence using the word "
                     f"{w.thai!r} ({w.meaning}). Output only the Thai sentence.")
        else:
            prompt = f"Write a natural Thai sentence using {subject!r}."
        return Question(subject=subject, provides="sentence", params={"prompt": prompt})

    levers: dict[str, list[Lever]] = {
        "picture": [Lever(backend="openverse", ask=provider.ask, build_question=picture_question),
                   Lever(backend="wikimedia", ask=provider.ask, build_question=picture_question),
                   Lever(backend="pexels", ask=provider.ask, build_question=picture_question)],
        "recording": [Lever(backend="forvo", ask=provider.ask, build_question=forvo_question),
                     Lever(backend="tts", ask=provider.ask, build_question=tts_question)],
    }
    if cfg.judge.transport in ("cli", "api"):
        levers["sentence"] = [Lever(backend="llm-sentence", ask=provider.ask,
                                    build_question=sentence_question)]
    return levers


# --- load_syllabus: curated files + db-backed ports -----------------------

def _pythainlp_tokenizer():
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError:
        return None

    @dataclass
    class _PythainlpTokenizer:
        def tokens(self, text: str) -> list[str]:
            return word_tokenize(text)

    return _PythainlpTokenizer()


@dataclass
class _DbMediaIndex:
    """MediaIndex (ports.py) over SyllabusDb: "has media" is answered from
    derivations.current_best, the same fold compile.py and the review
    server already use to decide what artifact a subject currently has --
    not a separate index store.py doesn't otherwise keep (see module
    docstring).
    """
    db: SyllabusDb
    pairs: tuple[MinimalPair, ...] = ()

    def has_picture(self, word: WordId) -> bool:
        return current_best(self.db, word, "picture").artifact_sha is not None

    def recording_speakers(self, word: WordId) -> frozenset[str]:
        return self._speakers_for(word)

    def rendition_speakers(self, pair_confusion: ConfusionId) -> frozenset[str]:
        speakers: set[str] = set()
        for pair in self.pairs:
            if pair.confusion != pair_confusion:
                continue
            for member in pair.members:
                speakers |= self._speakers_for(member)
        return frozenset(speakers)

    def _speakers_for(self, word: WordId) -> frozenset[str]:
        best = current_best(self.db, word, "recording")
        if best.artifact_sha is None:
            return frozenset()
        prov = self.db.media_provenance(best.artifact_sha)
        speaker = prov.get("speaker_id") if prov else None
        return frozenset({speaker}) if speaker else frozenset()


def load_syllabus(deck_root: str | Path, *,
                  frequency_path: str | Path | None = None) -> Syllabus:
    """Assembles a Syllabus (spec 1 section 3) from a deck directory's
    curated/*.yaml (spec 2 section 1 layout: curated/*.yaml + syllabus.db
    + media/, same layout reviewserver.load_context and migrate.py's
    new_root already use) plus the db-backed ports spec 2 section 3 adds:
    AssessmentReader/RecordWriter (the db itself), a MediaIndex over the
    db, a FrequencyMap, and sentences (spec 2's `sentences` table, absent
    from curated/*.yaml).
    """
    root = Path(deck_root)
    bundle = load_curated(root / "curated")
    db = SyllabusDb(root / "syllabus.db")
    db.set_pair_confusions({p.id: p.confusion for p in bundle.pairs})
    media_index = _DbMediaIndex(db=db, pairs=bundle.pairs)

    freq_file = (Path(frequency_path) if frequency_path is not None
                else root / "data" / "frequency_th.txt")
    freq_map = load_frequency_map(freq_file)
    frequency = {w.id: rank for w in bundle.words
                if (rank := freq_map.rank(w.thai)) is not None}

    rulebook_text = rulebook_file_text(root / "curated" / "rulebook.yaml")
    sentences = tuple(db.all_sentences())

    kwargs: dict[str, Any] = dict(
        words=bundle.words, targets=bundle.targets, pairs=bundle.pairs,
        graphemes=bundle.graphemes, sentences=sentences, confusions=bundle.confusions,
        profile=bundle.profile, frequency=frequency, media=media_index,
        assessments=db, rulebook_text=rulebook_text)

    tokenizer = _pythainlp_tokenizer()
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer

    return Syllabus(**kwargs)
