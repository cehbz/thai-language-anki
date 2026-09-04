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
  LLM. Read as: the llm backend reuses that SAME account, model and
  price, registered under three backend names (llm-sentence/llm-phrase/
  llm-entry) rather than one "llm" name -- LlmBackend.producer is fixed
  per instance (provider.py) and Provider looks a backend up by name,
  not by a per-call producer argument. A "batch" transport has no
  single-question `.complete()` (assessor.JudgeBackend.fetch's own
  docstring: "configured for batch only -- use Assessor.ask_many"), and
  drafting is inherently single-question, so under a batch judge the llm
  backends ride a lazy api transport on the same anthropic secret
  (_llm_transport); they are omitted only when no anthropic secret is
  configured to reach at all.
- build_assessor(cfg, db, media_store) registers "judge" (transport +
  model, resolve_path/price/quota_cost_per_call wired from cfg and the
  db+media_store's _resolver) and "mechanical" (duration check, same
  resolve_path -- MechanicalBackend's key_fn/evaluate are injectable
  CODE, not config, so providers.yaml carries no section for it);
  "listener" is unimplemented and "learner" is read-side-only --
  Assessor.ask() already special-cases both of those itself
  (assessor.py), so neither needs a roster entry.
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
- Tokenizer: pythainlp is imported lazily (a plain `try/except ImportError`
  at call time, guarding the only pythainlp import in this module).
  Syllabus.tokenizer has no default, so load_syllabus refuses with a
  RuntimeError naming pythainlp when it is not installed, rather than
  silently falling back to a tokenizer that mis-splits Thai.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .assessor import AssessBackend, Assessor, JudgeBackend, Price, duration_mechanical_backend
from .attempts import Sourcing
from .curated import (
    CuratedBundle,
    ProvidersConfig,
    load_curated,
    load_frequency_map,
    load_providers_config,
    rulebook_file_text,
)
from .derivations import current_best
from .entities import MinimalPair, Sentence, Word
from .ids import ConfusionId, PairId, WordId
from .media import Speaker
from .provider import (
    Backend,
    FetchBackend,
    ForvoBackend,
    LlmBackend,
    Provider,
    TtsBackend,
    openverse_backend,
    pexels_backend,
    tool_fetcher,
    wikimedia_backend,
)
from .rulebook import RULES, SENTENCE_FOR_TARGET_RUBRIC, apply_overlay, rubrics_for, sentence_note_id
from .run import FORVO_DEFAULT_DAILY_BUDGET, LEARNER_DEFAULT_SESSION_BUDGET, Budget
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus
from .transport import ClaudeApiTransport, ClaudeBatchTransport, ClaudeCliTransport, Completion
from .tts import pick_voice

__all__ = ["build_provider", "build_assessor", "build_sourcing", "default_budgets",
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
    """A `.complete(prompt) -> Completion`-shaped object that defers
    building the real transport until first call -- used both as
    LlmBackend.transport (an object with .complete) and, via its bound
    `.complete` method, as JudgeBackend.complete (a bare callable) -- see
    _judge_transport below.
    """
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._impl: Any = None

    def complete(self, prompt: str, attachments: Sequence[Path] = ()) -> Completion:
        if self._impl is None:
            self._impl = self._factory()
        return self._impl.complete(prompt, attachments)


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

    def submit(self, requests: Mapping[str, tuple[str, Sequence[Path]]]) -> str:
        return self._resolve().submit(requests)

    def status(self, batch_id: str) -> str:
        return self._resolve().status(batch_id)

    def results(self, batch_id: str) -> dict[str, Completion | None]:
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


def _llm_transport(cfg: ProvidersConfig, secrets) -> _LazyTransport | None:
    """The single-question transport llm-sentence/phrase/entry draft on.
    The judge's own cli/api transport where one exists; under a BATCH
    judge -- which has no single-question transport at all -- a lazy api
    transport on the same anthropic secret, so a deck whose verdicts ride
    batches can still draft sentences (an omitted llm backend silently
    left every target unfilled). None only when there is no transport to
    reach at all: a batch judge with no anthropic secret configured.
    """
    transport = _claude_transport(cfg, secrets)
    if transport is not None:
        return transport
    if "anthropic" not in cfg.secrets:
        return None
    return _LazyTransport(lambda: ClaudeApiTransport(
        api_key=secrets.get("anthropic") or "", model=cfg.judge.model))


def _judge_price(cfg: ProvidersConfig) -> Price | None:
    return Price(*cfg.judge.price_per_mtok) if cfg.judge.price_per_mtok else None


def _judge_quota_cost(cfg: ProvidersConfig) -> float:
    """The cli transport spends a flat unit of subscription quota per call
    and reports no token usage; api/batch report usage and are priced."""
    return 1.0 if cfg.judge.transport == "cli" else 0.0


# --- build_provider ----------------------------------------------------

def build_provider(cfg: ProvidersConfig, db: SyllabusDb, media_store: MediaStore,
                   *, secret_store=None) -> Provider:
    """The Provide port's backend roster (spec 3 section 2), wired from
    providers.yaml (spec 3 section 5): search_proxy for the image-search
    backends, imgfetch_path/audiofetch_path for the mediafetch tool
    fetchers (both always registered -- load_providers_config refuses a
    config missing either path), the tts voice pools, and the shared
    judge/llm transport+model for llm-*.
    """
    secrets = secret_store if secret_store is not None else cfg.secret_store()

    backends: dict[str, Backend] = {
        "openverse": openverse_backend(search_proxy=cfg.search_proxy),
        "wikimedia": wikimedia_backend(search_proxy=cfg.search_proxy),
        "pexels": _LazyBackend(lambda: pexels_backend(
            api_key=secrets.get("pexels") or "", search_proxy=cfg.search_proxy)),
        "forvo": _LazyBackend(lambda: ForvoBackend(api_key=secrets.get("forvo") or "")),
        "tts": _LazyBackend(lambda: TtsBackend(
            tts=_lazy_google_tts(secrets),
            voices=list(cfg.tts_male_voices) + list(cfg.tts_female_voices),
            media=media_store, pick_voice=pick_voice,
            cost_per_char=cfg.tts_cost_per_char)),
    }
    # Unconditional: load_providers_config refuses a providers.yaml without
    # both paths (pictures and recordings are always in scope), so there is
    # no "this deck has no imgfetch" case left to skip -- a run that could
    # not fetch what it found must fail at load, not silently source nothing.
    backends["imgfetch"] = FetchBackend(media=media_store,
                                        fetcher=tool_fetcher(cfg.imgfetch_path))
    backends["audiofetch"] = FetchBackend(media=media_store,
                                          fetcher=tool_fetcher(cfg.audiofetch_path))

    llm_transport = _llm_transport(cfg, secrets)
    if llm_transport is not None:
        for producer, name in (("sentence-drafter", "llm-sentence"),
                               ("phrase-drafter", "llm-phrase"),
                               ("entry-drafter", "llm-entry")):
            backends[name] = LlmBackend(producer=producer, model=cfg.judge.model,
                                        transport=llm_transport,
                                        price=_judge_price(cfg),
                                        quota_cost_per_call=_judge_quota_cost(cfg))

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

def _resolver(db: SyllabusDb, media_store: MediaStore):
    """artifact_sha -> the file it resolves to, or None (no provenance row,
    or the object is missing from disk) -- shared by the judge's
    attachments and the mechanical backend's duration check.
    """
    def resolve(sha: str) -> Path | None:
        prov = db.media_provenance(sha)
        if not prov:
            return None
        p = media_store.path_for(sha, prov["ext"])
        return p if p.exists() else None
    return resolve


def build_assessor(cfg: ProvidersConfig, db: SyllabusDb, media_store: MediaStore,
                   *, secret_store=None) -> Assessor:
    """The Assess port's backend roster (spec 3 section 2): "judge"
    (transport+model, resolve_path/price/quota_cost_per_call wired from
    cfg and the db+media_store) and "mechanical" (duration check, same
    resolve_path); "listener"/"learner" need no roster entry -- Assessor.
    ask() already special-cases both.
    """
    secrets = secret_store if secret_store is not None else cfg.secret_store()
    resolve = _resolver(db, media_store)
    judge = _build_judge_backend(cfg, secrets)
    judge.resolve_path = resolve
    judge.price = _judge_price(cfg)
    judge.quota_cost_per_call = _judge_quota_cost(cfg)
    backends: dict[str, AssessBackend] = {
        "judge": judge,
        "mechanical": duration_mechanical_backend(
            resolve_path=lambda sha: str(resolve(sha) or "")),
    }
    return Assessor(record=db, cache=db, backends=backends)


def _build_judge_backend(cfg: ProvidersConfig, secrets) -> JudgeBackend:
    kind = cfg.judge.transport
    complete = None
    batch_transport = None
    if kind == "batch":
        batch_transport = _LazyBatchTransport(lambda: ClaudeBatchTransport(
            api_key=secrets.get("anthropic") or "", model=cfg.judge.model))
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


# --- build_sourcing: the batch run's ctx (spec 3 section 4/5) -------------

def build_sourcing(deck_root: str | Path, cfg: ProvidersConfig | None = None) -> Sourcing:
    """Assembles a Sourcing ctx (attempts.py) for one deck: load_syllabus
    (overlay-applied rules), the db-backed provider/assessor rosters, and
    every value that reaches a cache key -- rubrics, provenance_prior,
    image_candidates, tts_voices, judge_model -- drawn from the deck's own
    curated/providers.yaml + rulebook.yaml, never a bare Sourcing dataclass
    default.

    Opens `db`/`bundle` exactly once and hands them to load_syllabus,
    rather than letting load_syllabus open its own second SyllabusDb/
    CuratedBundle -- so `Sourcing.db` and `syllabus.assessments`/
    `syllabus.media.db` are the SAME connection (one `set_pair_confusions`
    call, one place a run's writes and the Syllabus's reads meet).
    """
    root = Path(deck_root)
    if cfg is None:
        cfg = load_providers_config(root / "curated" / "providers.yaml")
    db = SyllabusDb(root / "syllabus.db")
    media_store = MediaStore(root / "media")
    bundle = load_curated(root / "curated")
    syllabus = load_syllabus(root, db=db, bundle=bundle)
    # rubrics_for covers registered judged Rules only; "sentence-for-target"
    # (attempts.py) is a judge role with no Rule, added here directly.
    rubrics = {**rubrics_for(syllabus.rules), "sentence-for-target": SENTENCE_FOR_TARGET_RUBRIC}
    return Sourcing(syllabus=syllabus, provider=build_provider(cfg, db, media_store),
                    assessor=build_assessor(cfg, db, media_store), db=db, media_store=media_store,
                    rubrics=rubrics,
                    provenance_prior=bundle.rulebook.provenance_prior,
                    image_candidates=cfg.image_candidates, tts_voices=tuple(cfg.tts_male_voices),
                    judge_model=cfg.judge.model)


# --- load_syllabus: curated files + db-backed ports -----------------------

def _pythainlp_tokenizer():
    try:
        from pythainlp.tokenize import word_tokenize
    except ImportError as exc:
        raise RuntimeError(
            "pythainlp is required to tokenize Thai text for load_syllabus; "
            "install it before loading a Syllabus") from exc

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

    rendition_provenance/rendition_speakers are a two-level read, not a
    single current_best lookup: a pair's rendition is current-best under
    its OWN subject (the pair id, role "rendition-for-pair" --
    attempts._record_rendition's mechanical row), and that row's
    `params["members"]` (word id -> sha) names which per-member recording
    actually backs it -- not necessarily each member's own current-best
    recording. Only when NO pair-level rendition row is current-best yet
    does rendition_provenance fall back to each member's own current-best
    recording, so a half-recorded or mixed-speaker pair still surfaces
    provenance for pair/rendition-required and rendition/mixed-speakers to
    warn on; rendition_speakers deliberately skips that fallback so such a
    pair stays in Syllabus.gaps().missing_renditions for the run to source
    a real rendition, even though the deck still compiles (with a warning).
    """
    db: SyllabusDb
    pairs: tuple[MinimalPair, ...] = ()
    words: tuple[Word, ...] = ()
    sentences: tuple[Sentence, ...] = ()
    rubrics: Mapping[str, str] = field(default_factory=dict)
    provenance_prior: Sequence[str] = ()

    def _best(self, subject: str, kind: str):
        """current_best, but with the SAME current_rubric/provenance_prior/
        provenance run.py's queue/attempt loop uses -- so this index and a
        live run never disagree about what's current-best.
        """
        return current_best(self.db, subject, kind, current_rubric=dict(self.rubrics) or None,
                            provenance_prior=self.provenance_prior,
                            provenance=self.db.media_provenance)

    def _deciding_row(self, subject: str, artifact_sha: str):
        """The newest mechanical assess row for `subject` whose
        artifact_sha matches current-best's pick -- the row _best()'s rank
        actually came from, so its own `params` (e.g. rendition's
        `members`) can be read back. Filtered to backend="mechanical" (the
        only backend AUTHORITY_ORDER["rendition-for-pair"] names) so a
        learner row under the same pair subject/artifact_sha can never
        shadow it.
        """
        rows = [r for r in self.db.assessments_of(subject) if r.port == "assess"
               and r.backend == "mechanical" and r.question.get("artifact_sha") == artifact_sha]
        return rows[-1] if rows else None

    def has_picture(self, word: WordId) -> bool:
        return self._best(word, "picture").artifact_sha is not None

    def recording_speakers(self, word: WordId) -> frozenset[str]:
        return self._speakers_for(word)

    def rendition_speakers(self, pair_confusion: ConfusionId) -> frozenset[str]:
        speakers: set[str] = set()
        for pair in self.pairs:
            if pair.confusion != pair_confusion:
                continue
            if self._best(pair.id, "rendition").artifact_sha is None:
                continue  # no pair-level rendition row yet -- skip the
                          # member-recording fallback (see class docstring)
            for prov in self.rendition_provenance(pair.id):
                speaker = prov.get("speaker_id") if prov else None
                if speaker:
                    speakers.add(speaker)
        return frozenset(speakers)

    def recording_provenance(self, word: WordId) -> Mapping[str, Any] | None:
        best = self._best(word, "recording")
        if best.artifact_sha is None:
            return None
        return self.db.media_provenance(best.artifact_sha)

    def rendition_provenance(self, pair_id: PairId) -> tuple[Mapping[str, Any], ...]:
        best = self._best(pair_id, "rendition")
        if best.artifact_sha is not None:
            row = self._deciding_row(pair_id, best.artifact_sha)
            members = row.question.get("params", {}).get("members", {}) if row else {}
            return tuple(prov for sha in members.values()
                        if (prov := self.db.media_provenance(sha)) is not None)

        # No current-best pair-level rendition yet: fall back to each
        # member's own current-best recording, so a partial or
        # mixed-speaker pair still surfaces provenance for the completeness/
        # warning rules (class docstring).
        pair = next((p for p in self.pairs if p.id == pair_id), None)
        if pair is None:
            return ()
        rows: list[Mapping[str, Any]] = []
        for member in pair.members:
            member_best = self._best(member, "recording")
            if member_best.artifact_sha is None:
                continue
            prov = self.db.media_provenance(member_best.artifact_sha)
            if prov is not None:
                rows.append(prov)
        return tuple(rows)

    def picture_sha(self, word: WordId) -> str | None:
        return self._best(word, "picture").artifact_sha

    def _speakers_for(self, word: WordId) -> frozenset[str]:
        best = self._best(word, "recording")
        if best.artifact_sha is None:
            return frozenset()
        prov = self.db.media_provenance(best.artifact_sha)
        speaker = prov.get("speaker_id") if prov else None
        return frozenset({speaker}) if speaker else frozenset()

    def speakers_of(self, corpus: Literal["recording", "rendition", "sentence"]) -> tuple[Speaker, ...]:
        """Distinct speakers behind that corpus's current-best artifacts
        (ports.py's MediaIndex.speakers_of): every word's current-best
        recording for "recording", every pair's current-best rendition
        rows for "rendition", every sentence's current-best recording for
        "sentence".
        """
        if corpus == "recording":
            provenances = (self.recording_provenance(w.id) for w in self.words)
        elif corpus == "sentence":
            provenances = (self.recording_provenance(sentence_note_id(s)) for s in self.sentences)
        elif corpus == "rendition":
            provenances = (prov for pair in self.pairs for prov in self.rendition_provenance(pair.id))
        else:
            raise ValueError(f"speakers_of: unknown corpus {corpus!r}")
        seen: dict[str, Speaker] = {}
        for prov in provenances:
            speaker = prov.get("speaker") if prov else None
            if speaker is not None:
                seen[speaker.id] = speaker
        return tuple(seen.values())


def load_syllabus(deck_root: str | Path, *,
                  frequency_path: str | Path | None = None,
                  db: SyllabusDb | None = None,
                  bundle: CuratedBundle | None = None) -> Syllabus:
    """Assembles a Syllabus (spec 1 section 3) from a deck directory's
    curated/*.yaml (spec 2 section 1 layout: curated/*.yaml + syllabus.db
    + media/, same layout reviewserver.load_context and migrate.py's
    new_root already use) plus the db-backed ports spec 2 section 3 adds:
    AssessmentReader/RecordWriter (the db itself), a MediaIndex over the
    db, a FrequencyMap, and sentences (spec 2's `sentences` table, absent
    from curated/*.yaml).

    `db`/`bundle` are optional injected handles -- opened here (the
    original single-caller shape) when omitted, but build_sourcing passes
    its own so the Syllabus's AssessmentReader/MediaIndex and the
    Sourcing ctx's `db` are the SAME SyllabusDb connection, not two
    separate ones racing to call `set_pair_confusions` on one only.
    """
    root = Path(deck_root)
    if bundle is None:
        bundle = load_curated(root / "curated")
    if db is None:
        db = SyllabusDb(root / "syllabus.db")
    db.set_pair_confusions({p.id: p.confusion for p in bundle.pairs})
    rules = apply_overlay(RULES, bundle.rulebook)
    sentences = tuple(db.all_sentences())
    media_index = _DbMediaIndex(db=db, pairs=bundle.pairs, words=bundle.words, sentences=sentences,
                                rubrics=rubrics_for(rules),
                                provenance_prior=bundle.rulebook.provenance_prior)

    freq_file = (Path(frequency_path) if frequency_path is not None
                else root / "data" / "frequency_th.txt")
    freq_map = load_frequency_map(freq_file)
    frequency = {w.id: rank for w in bundle.words
                if (rank := freq_map.rank(w.thai)) is not None}

    rulebook_text = rulebook_file_text(root / "curated" / "rulebook.yaml")

    kwargs: dict[str, Any] = dict(
        words=bundle.words, targets=bundle.targets, pairs=bundle.pairs,
        graphemes=bundle.graphemes, sentences=sentences, confusions=bundle.confusions,
        profile=bundle.profile, frequency=frequency, categories=bundle.categories,
        media=media_index, assessments=db, rulebook_text=rulebook_text, rules=rules,
        tokenizer=_pythainlp_tokenizer())

    return Syllabus(**kwargs)
