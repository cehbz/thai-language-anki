"""Wiring: the Provide/Assess backend rosters and Budget defaults from
curated/providers.yaml (spec 3 section 5), and a Syllabus assembled from
a deck directory's curated files plus db-backed ports (spec 1/2).

Secrets resolve lazily (spec 3 section 5): each secret-backed backend
(pexels, forvo, tts, the judge's and the drafter's api transports) is wrapped in `_Lazy`,
which builds the real backend -- and so calls `SecretStore.get()` -- at
its first `cache_key`/`fetch`/`complete` call, so a roster entry nobody
asks costs no file or 1Password read.

The llm Provide backends (llm-sentence/llm-phrase/llm-entry/llm-parse) ride
the drafter's transport from drafter.transport (cli by default; api rides
the judge's account, model, price and thinking), one registered name per
producer.

load_syllabus reads the deck's media relationships through `_DbMediaIndex`
(derivations.current_best over the db; the `media` table carries
provenance only), its sentences through `SyllabusDb.all_sentences()`
(unless a caller passes `sentences=` itself), and its frequency map from
`deck_root/curated/frequency_th.txt` (spec 2 section 1); a deck without
that file is refused with a FileNotFoundError naming it. Every sentence is
checked with Syllabus.check_sentence before load_syllabus returns; its
ValueError propagates, refusing a deck whose row is unregistered or
mis-rendered (spec 2 section 2).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Literal

from .assessor import AssessBackend, Assessor, DurationBackend, JudgeBackend, Price, RenditionBackend
from .attempts import Sourcing, provenance_source_for, sources_for
from .curated import (
    CuratedBundle,
    ProvidersConfig,
    load_curated,
    load_frequency_map,
    load_providers_config,
    rulebook_file_text,
)
from .derivations import DEFAULT_SENTENCE_NOTHING_CAP, current_best
from .entities import MinimalPair, Sentence, Word
from .ids import ConfusionId, PairId, WordId
from .media import Provenance, Recording, Speaker
from . import record
from .provider import (
    Backend,
    FetchBackend,
    ForvoBackend,
    LlmBackend,
    Provider,
    TtsBackend,
    OpenverseAuth,
    openverse_backend,
    pexels_backend,
    tool_fetcher,
    wikimedia_backend,
)
from .rulebook import (RULES, PRONUNCIATION_RUBRIC, SENTENCE_FOR_TARGET_RUBRIC, apply_overlay,
                       rubrics_for, sentence_note_id)
from .run import FORVO_DEFAULT_DAILY_BUDGET, LEARNER_DEFAULT_SESSION_BUDGET, Budget
from .store import MediaStore, SyllabusDb
from .syllabus import Syllabus, derive_productive_targets
from .transport import ClaudeApiTransport, ClaudeBatchTransport, ClaudeCliTransport, TransportError
from .tts import pick_voice

__all__ = ["build_provider", "build_assessor", "build_sourcing", "default_budgets",
          "nothing_ttl_for", "pacing_for", "Derivations", "load_derivations",
          "load_syllabus"]

# nothing_ttl_for's own default (spec 3 r19 section 6a/9): a Forvo
# `nothing` outcome stops counting as tried after this many days,
# offering the source again to a corpus that may have grown since --
# every other source never ages unless providers.yaml configures it.
_DEFAULT_NOTHING_TTL: dict[str, int] = {"forvo": 180}

# pacing_for's own defaults (spec 3 r26 sections 6a/8): seconds between
# two requests to a source, and the one wait on a challenge page before
# the single retry. Openverse's registered tier allows 100 requests a
# minute; 1 s keeps a run at 60. Every other source is unpaced and
# treats a challenge page as a plain transport failure unless
# providers.yaml's quotas.<source> configures otherwise.
_DEFAULT_PACING: dict[str, tuple[float, float]] = {"openverse": (1.0, 60.0)}


# --- laziness helpers -------------------------------------------------------

class _Lazy:
    """Builds the real object -- and so calls any SecretStore.get() it
    needs -- at its first attribute access, then forwards everything to
    it: a Backend, a transport, or a `tts.Tts`.
    """
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._impl: Any = None

    def _resolve(self) -> Any:
        if self._impl is None:
            self._impl = self._factory()
        return self._impl

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


def _claude_transport(cfg: ProvidersConfig, secrets) -> _Lazy | None:
    """A lazy `.complete(prompt)` transport for the judge (judge.transport,
    judge.model, judge.thinking, judge.max_tokens). None under a "batch"
    judge.
    """
    kind = cfg.judge.transport
    if kind == "cli":
        return _Lazy(lambda: ClaudeCliTransport())
    if kind == "api":
        return _Lazy(lambda: ClaudeApiTransport(
            api_key=secrets.get("anthropic") or "", model=cfg.judge.model,
            thinking=cfg.judge.thinking, max_tokens=cfg.judge.max_tokens))
    return None


def _drafter_transport(cfg: ProvidersConfig, secrets) -> _Lazy:
    """The single-question transport llm-sentence/phrase/entry draft on
    (spec 3 section 4): drafter.transport, cli or api; api rides the
    judge's account, model, thinking and max_tokens.
    """
    if cfg.drafter.transport == "api":
        return _Lazy(lambda: ClaudeApiTransport(
            api_key=secrets.get("anthropic") or "", model=cfg.judge.model,
            thinking=cfg.judge.thinking, max_tokens=cfg.judge.max_tokens))
    return _Lazy(lambda: ClaudeCliTransport())


def _judge_price(cfg: ProvidersConfig) -> Price | None:
    return Price(*cfg.judge.price_per_mtok) if cfg.judge.price_per_mtok else None


def _judge_quota_cost(cfg: ProvidersConfig) -> float:
    """The cli transport spends a flat unit of subscription quota per call
    and reports no token usage; api/batch report usage and are priced."""
    return 1.0 if cfg.judge.transport == "cli" else 0.0


def _drafter_price(cfg: ProvidersConfig) -> Price | None:
    return _judge_price(cfg) if cfg.drafter.transport == "api" else None


def _drafter_quota_cost(cfg: ProvidersConfig) -> float:
    return 1.0 if cfg.drafter.transport == "cli" else 0.0


# --- build_provider ----------------------------------------------------

def _recognize_drafting_answer(text: str) -> bool:
    """What the sentence drafter counts as having answered: drafts
    `record.drafts_in` can read (spec 3 r10 section 2), or a no-fit answer
    `record.parse_no_fit` can read (spec 3 r19 section 5). Anything else
    is an unusable completion and LlmBackend.fetch raises rather than
    caching it.
    """
    return bool(record.drafts_in(text)) or record.parse_no_fit(text) is not None


def build_provider(cfg: ProvidersConfig, db: SyllabusDb, media_store: MediaStore,
                   *, secret_store=None) -> Provider:
    """The Provide port's backend roster (spec 3 section 2), wired from
    providers.yaml: search_proxy for openverse, image_width for wikimedia,
    imgfetch_path/audiofetch_path for the mediafetch fetchers, the tts
    voice pools, and the drafter transport for llm-*.
    """
    secrets = secret_store if secret_store is not None else cfg.secret_store()

    def paced(source: str) -> dict[str, float]:
        interval, wait = pacing_for(cfg, source)
        return {"min_interval_s": interval, "challenge_wait_s": wait}

    backends: dict[str, Backend] = {
        "openverse": openverse_backend(search_proxy=cfg.search_proxy,
                                       auth=_openverse_auth(cfg, secrets), **paced("openverse")),
        "wikimedia": wikimedia_backend(image_width=cfg.image_width, **paced("wikimedia")),
        "pexels": _Lazy(lambda: pexels_backend(api_key=secrets.get("pexels") or "",
                                               **paced("pexels"))),
        "forvo": _Lazy(lambda: ForvoBackend(api_key=secrets.get("forvo") or "")),
        "tts": _Lazy(lambda: TtsBackend(
            tts=_lazy_google_tts(secrets),
            voices=list(cfg.tts_male_voices) + list(cfg.tts_female_voices),
            media=media_store, pick_voice=pick_voice,
            cost_per_char=cfg.tts_cost_per_char)),
    }
    # Always registered: load_providers_config refuses a providers.yaml
    # without both paths.
    backends["imgfetch"] = FetchBackend(media=media_store,
                                        fetcher=tool_fetcher(cfg.imgfetch_path))
    backends["audiofetch"] = FetchBackend(media=media_store,
                                          fetcher=tool_fetcher(cfg.audiofetch_path))

    drafter_transport = _drafter_transport(cfg, secrets)
    # llm-sentence recognizes a completion drafts_in can read (spec 3 r10
    # section 2) or one that is a no-fit answer (spec 3 r19 section 5's
    # {"sentences": [], "reason": ...}, record.parse_no_fit) -- both are
    # answers the drafting attempt acts on, so both are cached; llm-parse
    # recognizes only one parses_in can read (spec 3 r16 section 5);
    # llm-phrase recognizes only a completion naming at least one asked
    # item's phrase (record.parse_phrases non-empty) -- an answer
    # phrasing none of them (spec 3 section 5, fix round 2 finding 2) is
    # not a cacheable answer, so a stable lacking set never turns into a
    # permanent empty cache hit; llm-entry keeps LlmBackend's default,
    # which recognizes any text.
    recognizers: dict[str, Callable[[str], bool]] = {
        "llm-sentence": _recognize_drafting_answer,
        "llm-parse": lambda text: bool(record.parses_in(text)),
        "llm-phrase": lambda text: bool(record.parse_phrases(text)),
    }
    for producer, name in (("sentence-drafter", "llm-sentence"),
                           ("phrase-drafter", "llm-phrase"),
                           ("entry-drafter", "llm-entry"),
                           ("sentence-parser", "llm-parse")):
        kwargs = {"recognize": recognizers[name]} if name in recognizers else {}
        backends[name] = LlmBackend(producer=producer, model=cfg.judge.model,
                                    transport=drafter_transport,
                                    price=_drafter_price(cfg),
                                    quota_cost_per_call=_drafter_quota_cost(cfg),
                                    **kwargs)

    return Provider(record=db, cache=db, backends=backends)


def _lazy_google_tts(secrets) -> _Lazy:
    """A GoogleTts standing in for tts.Tts, built at the first
    `synthesize` call, so the google_tts secret resolves only then.
    """
    from .tts import GoogleTts

    return _Lazy(lambda: GoogleTts(api_key=secrets.get("google_tts") or ""))


# --- build_assessor ----------------------------------------------------

def _resolver(db: SyllabusDb, media_store: MediaStore):
    """artifact_sha -> its file, or None when there is no provenance row
    or the object is missing from disk.
    """
    def resolve(sha: str) -> Path | None:
        prov = db.media_provenance(sha)
        if not prov:
            return None
        p = media_store.path_for(sha, prov["ext"])
        return p if p.exists() else None
    return resolve


def _speaker_of(db: SyllabusDb) -> Callable[[str], str | None]:
    """artifact sha -> the speaker id its media row names, or None."""
    def speaker_of(sha: str) -> str | None:
        prov = db.media_provenance(sha)
        return prov.get("speaker_id") if prov else None
    return speaker_of


def build_assessor(cfg: ProvidersConfig, db: SyllabusDb, media_store: MediaStore,
                   *, secret_store=None) -> Assessor:
    """The Assess port's backend roster (spec 3 section 2): "judge",
    "mechanical" (the duration check), "rendition". Fills is membership
    (Syllabus.fills), not an Assess backend (spec 1 section 3 r8; spec 3
    r16).
    """
    secrets = secret_store if secret_store is not None else cfg.secret_store()
    resolve = _resolver(db, media_store)
    judge = _build_judge_backend(cfg, secrets)
    judge.resolve_path = resolve
    judge.price = _judge_price(cfg)
    judge.quota_cost_per_call = _judge_quota_cost(cfg)
    backends: dict[str, AssessBackend] = {
        "judge": judge,
        "mechanical": DurationBackend(resolve_path=resolve),
        "rendition": RenditionBackend(speaker_of=_speaker_of(db)),
    }
    return Assessor(record=db, cache=db, backends=backends)


def _build_judge_backend(cfg: ProvidersConfig, secrets) -> JudgeBackend:
    kind = cfg.judge.transport
    complete = None
    batch_transport = None
    if kind == "batch":
        batch_transport = _Lazy(lambda: ClaudeBatchTransport(
            api_key=secrets.get("anthropic") or "", model=cfg.judge.model,
            thinking=cfg.judge.thinking, max_tokens=cfg.judge.max_tokens))
    else:
        transport = _claude_transport(cfg, secrets)
        if transport is not None:
            # Wrapped, not extracted as `transport.complete` directly: the
            # `.complete` lookup on `transport` (a _Lazy) resolves it, so
            # this closure defers that lookup to the moment it is called.
            complete = lambda prompt, attachments=(): transport.complete(prompt, attachments)
    return JudgeBackend(model=cfg.judge.model, transport=kind, complete=complete,
                        batch_transport=batch_transport)


# --- budgets -------------------------------------------------------------

def default_budgets(cfg: ProvidersConfig) -> dict[str, Budget]:
    """Budget per backend (spec 3 section 4): the two documented defaults
    (forvo 450/day from 22:00Z, learner 20/session) layered under
    whatever providers.yaml's `quotas` section configures, field by
    field -- a configured entry that names only `max_asks` still keeps
    the matching default's `day_starts` (and vice versa); every other
    configured backend just adds its own Budget from the fields it names.
    """
    budgets: dict[str, Budget] = {
        "forvo": FORVO_DEFAULT_DAILY_BUDGET,
        "learner": LEARNER_DEFAULT_SESSION_BUDGET,
    }
    for backend, quota in cfg.quotas.items():
        base = budgets.get(backend, Budget())
        budgets[backend] = Budget(
            max_asks=quota.get("max_asks", base.max_asks),
            max_cost=quota.get("max_cost", base.max_cost),
            day_starts=quota.get("day_starts", base.day_starts))
    return budgets


def pacing_for(cfg: ProvidersConfig, source: str) -> tuple[float, float]:
    """(min_interval_s, challenge_wait_s) for `source` (spec 3 r26
    section 8): the default above, layered field by field under
    providers.yaml's `quotas.<source>.{min_interval_seconds,
    challenge_wait_seconds}`.
    """
    interval, wait = _DEFAULT_PACING.get(source, (0.0, 0.0))
    quota = cfg.quotas.get(source, {})
    return (float(quota.get("min_interval_seconds", interval)),
            float(quota.get("challenge_wait_seconds", wait)))


def _openverse_auth(cfg: ProvidersConfig, secrets) -> Callable[[], str | None] | None:
    """The bearer-token callable for Openverse when `secrets.openverse`
    is configured (`client_id:client_secret`, spec 3 r26 section 8),
    else None (anonymous). The secret is read at the first search, the
    token fetched then and reused for the process."""
    if not secrets.configured("openverse"):
        return None
    holder: list[OpenverseAuth] = []

    def token() -> str | None:
        if not holder:
            value = secrets.get("openverse") or ""
            client_id, sep, client_secret = value.strip().partition(":")
            if not sep or not client_id or not client_secret:
                raise TransportError(
                    "secrets.openverse must hold one line client_id:client_secret")
            holder.append(OpenverseAuth(client_id=client_id, client_secret=client_secret,
                                        search_proxy=cfg.search_proxy))
        return holder[0].token()

    return token


def nothing_ttl_for(cfg: ProvidersConfig) -> dict[str, int]:
    """derivations.tried_sources' `nothing_ttl` mapping (spec 3 r19
    section 6a/9): forvo ages a `nothing` outcome out after 180 days by
    default, layered under whatever providers.yaml's own
    `quotas.<source>.nothing_ttl_days` configures, source by source -- a
    configured value overrides the default (or adds ageing for a source
    with none), and a source named in neither never ages.
    """
    ttl = dict(_DEFAULT_NOTHING_TTL)
    for source, quota in cfg.quotas.items():
        if "nothing_ttl_days" in quota:
            ttl[source] = quota["nothing_ttl_days"]
    return ttl


# --- load_derivations: the parameters every fold over the record takes ----

@dataclass(frozen=True)
class Derivations:
    """One deck's record and every parameter derivations.py asks for: the
    Syllabus (media index included), the db the record lives in, the media
    store its artifacts resolve to, and the current_rubric / prior /
    provenance_source / sources_for / attempt_cap / transient_cap /
    sentence_nothing_cap a fold is measured under. build_sourcing wires
    the run's Sourcing from this same bundle, so a surface holding one
    derives exactly what the run derives.
    """
    syllabus: Syllabus
    db: SyllabusDb                     # CacheReader + RecordWriter
    media_store: MediaStore
    current_rubric: Mapping[str, str]  # role -> rubric text
    prior: Sequence[str]               # provenance kinds, most preferred first
    provenance_source: Callable[[str], str | None]
    sources_for: Callable[[str], Sequence[str]]
    attempt_cap: int
    transient_cap: int
    # sentence_exhausted()'s no-fit cap (spec 3 r19 section 5), the same
    # value build_sourcing hands the run's Sourcing.
    sentence_nothing_cap: int = DEFAULT_SENTENCE_NOTHING_CAP
    # tried_sources()'s per-source ageing (spec 3 r19 section 6a/9,
    # nothing_ttl_for), the same mapping build_sourcing hands the run's
    # Sourcing.
    nothing_ttl: Mapping[str, int] = field(default_factory=dict)
    # rulebook.yaml's thresholds overlay (curated.RulebookConfig.thresholds),
    # e.g. "reask/lapses" -- spec 5 section 1 kind 4's own lapse threshold.
    thresholds: Mapping[str, float] = field(default_factory=dict)
    # Per-backend Budget (spec 3 section 7), the same default_budgets(cfg)
    # build_sourcing's own run() takes -- reviewserver's question session
    # reads budgets["learner"].max_asks through here, providers.yaml-
    # configurable through the same "quotas" path as forvo's day budget.
    budgets: Mapping[str, Budget] = field(default_factory=dict)


def load_derivations(deck_root: str | Path, cfg: ProvidersConfig | None = None) -> Derivations:
    """One deck's Derivations from its own curated/*.yaml + syllabus.db +
    media/ (spec 2 section 1 layout), over one db connection.
    """
    root = Path(deck_root)
    if cfg is None:
        cfg = load_providers_config(root / "curated" / "providers.yaml")
    db = SyllabusDb(root / "syllabus.db")
    media_store = MediaStore(root / "media")
    bundle = load_curated(root / "curated")
    syllabus = load_syllabus(root, db=db, bundle=bundle)
    # rubrics_for covers registered judged Rules only; "sentence-for-target"
    # (attempts.py) and "pronunciation-for-word" (the adjudication pass,
    # spec 3 r28) are judge roles with no Rule, added here directly.
    rubrics = {**rubrics_for(syllabus.rules), "sentence-for-target": SENTENCE_FOR_TARGET_RUBRIC,
              "pronunciation-for-word": PRONUNCIATION_RUBRIC}
    return Derivations(syllabus=syllabus, db=db, media_store=media_store,
                       current_rubric=rubrics,
                       prior=bundle.rulebook.provenance_prior,
                       provenance_source=provenance_source_for(db),
                       sources_for=sources_for, attempt_cap=cfg.attempt_cap,
                       transient_cap=cfg.transient_cap,
                       sentence_nothing_cap=cfg.sentence_nothing_cap,
                       nothing_ttl=nothing_ttl_for(cfg),
                       thresholds=dict(bundle.rulebook.thresholds),
                       budgets=default_budgets(cfg))


# --- build_sourcing: the batch run's ctx (spec 3 section 4/5) -------------

def build_sourcing(deck_root: str | Path, cfg: ProvidersConfig | None = None) -> Sourcing:
    """One deck's Sourcing ctx (attempts.py): its Derivations, the
    db-backed provider/assessor rosters, and the values that reach a
    cache key (image_candidates, voices, judge_model), all
    from the deck's own curated/providers.yaml and rulebook.yaml.
    """
    root = Path(deck_root)
    if cfg is None:
        cfg = load_providers_config(root / "curated" / "providers.yaml")
    derivations = load_derivations(root, cfg)
    db, media_store = derivations.db, derivations.media_store
    ctx = Sourcing(
        syllabus=derivations.syllabus, provider=build_provider(cfg, db, media_store),
        assessor=build_assessor(cfg, db, media_store),
        db=db, media_store=media_store, rubrics=derivations.current_rubric,
        provenance_prior=derivations.prior,
        image_candidates=cfg.image_candidates,
        voices={"male": tuple(cfg.tts_male_voices), "female": tuple(cfg.tts_female_voices)},
        judge_model=cfg.judge.model,
        sources_for=derivations.sources_for, attempt_cap=derivations.attempt_cap,
        transient_cap=derivations.transient_cap,
        sentence_nothing_cap=derivations.sentence_nothing_cap,
        sentence_max_clauses=cfg.sentence_max_clauses,
        sentence_introducible_per_ask=cfg.sentence_introducible_per_ask,
        sentence_targets_per_sentence=cfg.sentence_targets_per_sentence,
        nothing_ttl=derivations.nothing_ttl,
        # The deck's own curated store, so run._materialize_adjudications
        # can write words.yaml back (spec 2 r14 section 1; spec 3 r28
        # section 5). `engines` stays None: the run resolves
        # phonology.default_engines() itself, on the first verdict there
        # is to check, so wiring a deck never loads pythainlp/torch.
        curated_dir=root / "curated", engines=None)
    return ctx


# --- load_syllabus: curated files + db-backed ports -----------------------

@dataclass
class _DbMediaIndex:
    """MediaIndex (ports.py) over SyllabusDb: what media a subject has is
    derivations.current_best, the fold compile.py and the review server
    read too.

    rendition_provenance/rendition_speakers are a two-level read: a pair's
    rendition is current-best under the pair id (role
    "rendition-for-pair"), and that row's `params["members"]` (word id ->
    sha) names the per-member recordings backing it. With no pair-level
    rendition row current-best, rendition_provenance falls back to each
    member's own current-best recording, so pair/rendition-required still
    sees provenance to check against; rendition_speakers takes no such
    fallback, so the pair stays in Syllabus.gaps().missing_renditions for
    the run to source. (rendition/mixed-speakers, which once read this
    same provenance to warn, is retired spec 1 section 4 r10: the
    rendition attempt's own one-speaker intersection holds by
    construction.)
    """
    db: SyllabusDb
    pairs: tuple[MinimalPair, ...] = ()
    words: tuple[Word, ...] = ()
    sentences: tuple[Sentence, ...] = ()
    rubrics: Mapping[str, str] = field(default_factory=dict)
    provenance_prior: Sequence[str] = ()

    def _best(self, subject: str, kind: str):
        """current_best under the same current_rubric/prior run.py's
        queue and attempt loop use.
        """
        return current_best(self.db, subject, kind, current_rubric=dict(self.rubrics),
                            prior=self.provenance_prior,
                            provenance_source=provenance_source_for(self.db))

    def _deciding_row(self, subject: str, artifact_sha: str):
        """The newest "rendition" assess row for `subject` at
        current-best's artifact_sha -- the row _best()'s rank came from,
        whose `params["members"]` names the member recordings. Scoped to
        that one backend, the only one
        AUTHORITY_ORDER["rendition-for-pair"] names.
        """
        rows = [r for r in self.db.assessments_of(subject) if r.port == "assess"
               and r.backend == "rendition" and r.question.get("artifact_sha") == artifact_sha]
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
                continue  # no pair-level rendition row yet: no fallback here
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

    def rendition(self, pair_id: PairId) -> tuple[Recording, ...] | None:
        """The pair's current-best rendition (ports.py's MediaIndex.
        rendition): one Recording per member, in member order, built from
        the deciding "rendition" row's params["members"] (word id -> sha)
        and each sha's `media` row. None when the pair has no current-best
        rendition, or when a member's sha has no media provenance row.
        """
        best = self._best(pair_id, "rendition")
        if best.artifact_sha is None:
            return None
        row = self._deciding_row(pair_id, best.artifact_sha)
        members = row.question.get("params", {}).get("members", {}) if row else {}
        pair = next((p for p in self.pairs if p.id == pair_id), None)
        if pair is None:
            return None

        recordings: list[Recording] = []
        for member in pair.members:
            sha = members.get(member)
            prov = self.db.media_provenance(sha) if sha else None
            speaker = prov.get("speaker") if prov else None
            if prov is None or speaker is None:
                return None
            recordings.append(Recording(
                sha=sha,
                provenance=Provenance(source=prov["source"], origin=prov["origin"],
                                      licence=prov["licence"],
                                      acquired=date.fromisoformat(prov["acquired"])),
                speaker=speaker))
        return tuple(recordings)

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
        """Distinct speakers behind that corpus's current-best artifacts:
        word recordings, pair renditions, or sentence recordings.
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
                  db: SyllabusDb | None = None,
                  bundle: CuratedBundle | None = None,
                  sentences: Sequence[Sentence] | None = None) -> Syllabus:
    """A Syllabus (spec 1 section 3) from a deck directory: its
    curated/*.yaml, plus the db-backed ports spec 2 section 3 adds --
    AssessmentReader/RecordWriter (the db itself), a MediaIndex over the
    db, a FrequencyMap, and the `sentences` table. `db`/`bundle` are
    opened here when not passed; a caller passes its own so the Syllabus
    and it share one connection. `sentences`, when given, is used in place
    of `db.all_sentences()` and the sentences table is not read (the
    migration's parse step loads a Syllabus with sentences=() to run
    check_sentence over drafted clauses before they are written). Every
    resulting sentence is checked with Syllabus.check_sentence, whose
    ValueError (an unregistered word id or a text/rendering mismatch)
    propagates, naming the offending row.
    """
    root = Path(deck_root)
    if bundle is None:
        bundle = load_curated(root / "curated")
    if db is None:
        db = SyllabusDb(root / "syllabus.db")
    rules = apply_overlay(RULES, bundle.rulebook)
    resolved_sentences = tuple(sentences) if sentences is not None else tuple(db.all_sentences())
    media_index = _DbMediaIndex(db=db, pairs=bundle.pairs, words=bundle.words,
                                sentences=resolved_sentences, rubrics=rubrics_for(rules),
                                provenance_prior=bundle.rulebook.provenance_prior)

    freq_file = root / "curated" / "frequency_th.txt"
    if not freq_file.exists():
        raise FileNotFoundError(f"no frequency corpus at {freq_file}")
    freq_map = load_frequency_map(freq_file)
    frequency = {w.id: rank for w in bundle.words
                if (rank := freq_map.rank(w.thai)) is not None}
    targets = tuple(bundle.targets) + derive_productive_targets(
        bundle.words, bundle.targets, bundle.categories, frequency,
        bundle.profile.productive_cutoff)

    rulebook_text = rulebook_file_text(root / "curated" / "rulebook.yaml")

    kwargs: dict[str, Any] = dict(
        words=bundle.words, targets=targets, pairs=bundle.pairs,
        graphemes=bundle.graphemes, sentences=resolved_sentences, confusions=bundle.confusions,
        profile=bundle.profile, frequency=frequency, categories=bundle.categories,
        media=media_index, assessments=db, rulebook_text=rulebook_text, rules=rules)

    syllabus = Syllabus(**kwargs)
    for s in syllabus.sentences:
        syllabus.check_sentence(s)
    return syllabus
