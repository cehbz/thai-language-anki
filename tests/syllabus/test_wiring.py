"""Tests for wiring.py: build_provider/build_assessor/build_sourcing/
default_budgets/load_syllabus -- assembling spec 3's backend rosters from
curated/providers.yaml and spec 1/2's Syllabus from a deck directory. No
network, no subprocess, no pythainlp/anthropic import; secret files are
real tmp 0600 files whose reads are tracked to prove lazy resolution.

build_levers/Lever are gone (Task 10 replaced the lever-escalation shape
with attempts.attempt() + attempts.SOURCES); this module is rewired
around build_sourcing (Task 11).
"""
from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from thai_syllabus import secrets as secrets_mod
from thai_syllabus.assessor import Assessor, Price
from thai_syllabus.attempts import DEFAULT_SENTENCE_INTRODUCIBLE_PER_ASK, DEFAULT_SENTENCE_MAX_CLAUSES
from thai_syllabus.cachekeys import JudgeKey, MechanicalKey, ProvideKey, sha
from thai_syllabus.curated import (
    CuratedBundle,
    JudgeConfig,
    ProvidersConfig,
    RulebookConfig,
    load_providers_config,
    save_curated,
)
from thai_syllabus.derivations import DEFAULT_SENTENCE_NOTHING_CAP
from thai_syllabus.entities import Category
from thai_syllabus.media import Speaker
from thai_syllabus.profile import Profile
from thai_syllabus.provider import Provider, Question
from thai_syllabus.rulebook import sentence_note_id
from thai_syllabus.run import Budget
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.wiring import (
    _DbMediaIndex,
    build_assessor,
    build_provider,
    build_sourcing,
    default_budgets,
    load_derivations,
    load_syllabus,
    nothing_ttl_for,
)

from .builders import PROV, sentence, syl, pron, target, thai_of, word


# --- fixtures ----------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "wiring" / "syllabus.db")


@pytest.fixture
def media_store(tmp_path):
    return MediaStore(tmp_path / "wiring" / "media")


def _secret_file(tmp_path, name, value="s3cret\n"):
    path = tmp_path / f"{name}.key"
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return path


@pytest.fixture
def secret_paths(tmp_path):
    return {
        "pexels": _secret_file(tmp_path, "pexels", "pexels-key\n"),
        "forvo": _secret_file(tmp_path, "forvo", "forvo-key\n"),
        "google_tts": _secret_file(tmp_path, "google_tts", "tts-key\n"),
        "anthropic": _secret_file(tmp_path, "anthropic", "anthropic-key\n"),
    }


@pytest.fixture
def cfg(secret_paths):
    # Both fetch paths set: load_providers_config refuses a file without
    # them, so every config build_provider can actually be handed has both.
    return ProvidersConfig(
        secrets={name: str(path) for name, path in secret_paths.items()},
        search_proxy="https://proxy.example",
        imgfetch_path="curl",
        audiofetch_path="curl",
    )


def _track_reads(monkeypatch):
    """Spies on secrets._read_file so tests can assert which named secrets
    were actually resolved, without going through real subprocess/1Password
    or real network -- the file IS real (0600 tmp file) so this proves the
    file itself was never opened, not just that some higher-level cache
    was consulted.
    """
    calls: list[str] = []
    real = secrets_mod._read_file

    def spy(spec, *, name):
        calls.append(name)
        return real(spec, name=name)

    monkeypatch.setattr(secrets_mod, "_read_file", spy)
    return calls


# --- build_provider: roster shape ---------------------------------------

def test_build_provider_returns_a_provider(cfg, db, media_store):
    provider = build_provider(cfg, db, media_store)
    assert isinstance(provider, Provider)


def test_build_provider_registers_the_free_backends(cfg, db, media_store):
    provider = build_provider(cfg, db, media_store)
    for name in ("openverse", "wikimedia", "imgfetch"):
        assert name in provider._backends


def test_build_provider_registers_secret_backed_backends(cfg, db, media_store):
    provider = build_provider(cfg, db, media_store)
    for name in ("pexels", "forvo", "tts"):
        assert name in provider._backends


# --- laziness: constructing the roster must not touch secret files ------

def test_building_the_roster_reads_no_secret_files(cfg, db, media_store, monkeypatch):
    calls = _track_reads(monkeypatch)
    build_provider(cfg, db, media_store)
    assert calls == []


def test_resolving_the_pexels_backend_reads_only_the_pexels_secret(
        cfg, db, media_store, monkeypatch):
    calls = _track_reads(monkeypatch)
    provider = build_provider(cfg, db, media_store)
    resolved = provider._backends["pexels"]._resolve()
    assert calls == ["pexels"]
    assert resolved.cache_key(Question(subject="s", provides="picture",
                                       params={"query": "cat"})).encode() == "pexels::cat"


def test_an_openverse_ask_never_touches_any_secret(cfg, db, media_store, monkeypatch):
    calls = _track_reads(monkeypatch)

    def fake_get(url, params=None, headers=None, timeout=None, proxies=None):
        class _Resp:
            status_code = 200
            def json(self):
                return {"results": []}
        return _Resp()

    provider = build_provider(cfg, db, media_store)
    provider._backends["openverse"].get = fake_get
    provider.ask("openverse", Question(subject="s", provides="picture",
                                       params={"query": "cat"}))
    assert calls == []


# --- build_provider: search_proxy / imgfetch_path threading -------------

def test_search_proxy_reaches_only_openverse(cfg, db, media_store):
    provider = build_provider(cfg, db, media_store)
    assert provider._backends["openverse"].search_proxy == "https://proxy.example"
    assert provider._backends["wikimedia"].search_proxy is None
    assert provider._backends["pexels"]._resolve().search_proxy is None


def test_openverse_is_anonymous_and_paced_by_default(cfg, db, media_store):
    """Spec 3 r26 section 8: no `secrets.openverse` -- no auth callable;
    the default pacing is 1 s between requests and one 60 s wait on a
    challenge page; the other corpora are unpaced."""
    provider = build_provider(cfg, db, media_store)
    openverse = provider._backends["openverse"]
    assert openverse.auth is None
    assert (openverse.min_interval_s, openverse.challenge_wait_s) == (1.0, 60.0)
    wikimedia = provider._backends["wikimedia"]
    assert (wikimedia.min_interval_s, wikimedia.challenge_wait_s) == (0.0, 0.0)


def test_quotas_pacing_fields_override_the_defaults(db, media_store, secret_paths):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          imgfetch_path="curl", audiofetch_path="curl",
                          quotas={"openverse": {"min_interval_seconds": 4},
                                  "wikimedia": {"challenge_wait_seconds": 30}})
    provider = build_provider(cfg, db, media_store)
    openverse, wikimedia = provider._backends["openverse"], provider._backends["wikimedia"]
    assert (openverse.min_interval_s, openverse.challenge_wait_s) == (4.0, 60.0)
    assert (wikimedia.min_interval_s, wikimedia.challenge_wait_s) == (0.0, 30.0)


def test_a_configured_openverse_secret_becomes_a_bearer_token_read_at_first_use(
        db, media_store, secret_paths, tmp_path, monkeypatch):
    """`secrets.openverse` holds client_id:client_secret; the wiring
    reads it only when the first search asks for a token, and the token
    endpoint is posted through the search proxy."""
    key = tmp_path / "openverse.key"
    key.write_text("cid:sec\n")
    key.chmod(0o600)
    cfg = ProvidersConfig(secrets={**{n: str(p) for n, p in secret_paths.items()},
                                   "openverse": str(key)},
                          imgfetch_path="curl", audiofetch_path="curl",
                          search_proxy="https://proxy.example")
    posts = []

    def fake_post(url, data=None, headers=None, timeout=None, proxies=None):
        posts.append((url, data["client_id"], data["client_secret"], proxies["https"]))
        class _Resp:
            status_code = 200
            def json(self):
                return {"access_token": "tok", "expires_in": 3600}
        return _Resp()

    monkeypatch.setattr("thai_syllabus.provider.requests.post", fake_post)
    provider = build_provider(cfg, db, media_store)
    openverse = provider._backends["openverse"]
    assert callable(openverse.auth) and posts == []
    assert openverse.auth() == "tok"
    assert posts == [("https://api.openverse.org/v1/auth_tokens/token/", "cid", "sec",
                      "https://proxy.example")]


def test_wikimedia_image_width_reaches_the_backend(db, media_store, secret_paths):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          imgfetch_path="curl", audiofetch_path="curl",
                          image_width=800)
    provider = build_provider(cfg, db, media_store)
    _, params, _, _ = provider._backends["wikimedia"].build_request("cat")
    assert params["iiurlwidth"] == 800


def test_imgfetch_binary_comes_from_imgfetch_path(db, media_store, secret_paths, monkeypatch):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          imgfetch_path="/opt/bin/imgfetch",
                          audiofetch_path="/opt/bin/audiofetch")
    provider = build_provider(cfg, db, media_store)
    calls = []

    def fake_run(cmd, **kwargs):
        import io
        import subprocess as sp

        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="JPEG")
        Path(cmd[2]).write_bytes(buf.getvalue())
        calls.append(cmd)
        return sp.CompletedProcess(cmd, 0, '{"format":"jpeg"}\n', "")

    import thai_syllabus.provider as provider_mod
    monkeypatch.setattr(provider_mod.subprocess, "run", fake_run)

    answer = provider.ask("imgfetch", Question(subject="s", provides="picture-bytes",
                                               params={"url": "https://x/y.jpg"}))
    assert calls[0][0] == "/opt/bin/imgfetch"
    assert calls[0][1] == "https://x/y.jpg"
    assert calls[0][2] == str(Path(calls[0][2]))  # an out-path was supplied
    assert answer.items[0]["ext"] == "jpg"


def test_audiofetch_binary_comes_from_audiofetch_path(db, media_store, secret_paths):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          imgfetch_path="/opt/bin/imgfetch",
                          audiofetch_path="/opt/bin/audiofetch")
    provider = build_provider(cfg, db, media_store)
    assert "audiofetch" in provider._backends


# --- build_provider: tts voice pools -------------------------------------

def test_tts_backend_voice_pool_combines_male_and_female(cfg, db, media_store):
    cfg = ProvidersConfig(secrets=cfg.secrets, tts_male_voices=("m1", "m2"),
                          tts_female_voices=("f1",))
    provider = build_provider(cfg, db, media_store)
    backend = provider._backends["tts"]._resolve()
    assert set(backend.voices) == {"m1", "m2", "f1"}


# --- build_provider: llm backends follow drafter.transport ----------------

def test_llm_backends_are_always_registered(cfg, db, media_store):
    from thai_syllabus.curated import JudgeConfig
    for judge in (JudgeConfig(transport="cli"),
                  JudgeConfig(transport="batch", model="m", price_per_mtok=(2.0, 10.0))):
        backends = build_provider(ProvidersConfig(secrets=cfg.secrets, judge=judge),
                                  db, media_store)._backends
        assert {"llm-sentence", "llm-phrase", "llm-entry", "llm-parse"} <= set(backends)


def test_llm_sentence_recognizes_only_a_completion_drafts_in_reads(cfg, db, media_store):
    """Spec 3 r10 section 2: llm-sentence's LlmBackend.recognize rejects a
    completion drafts_in cannot read as a draft."""
    backends = build_provider(cfg, db, media_store)._backends
    assert backends["llm-sentence"].recognize("no json here") is False


def test_llm_phrase_recognizes_only_a_completion_naming_at_least_one_phrase(cfg, db, media_store):
    """Fix round 2 finding 2 (spec 3 section 2): an answer phrasing none
    of the asked items -- garbage, or an empty `{"phrases": []}` -- is
    not a recognized answer, so LlmBackend.fetch raises and caches
    nothing; a permanent empty-answer cache hit would otherwise silence
    the drafter forever for a stable lacking set. A partial answer --
    at least one item actually phrased -- is recognized, so its rows are
    still cached (the omitted items simply re-ask next run)."""
    import json

    backends = build_provider(cfg, db, media_store)._backends
    assert backends["llm-phrase"].recognize("no json here") is False
    assert backends["llm-phrase"].recognize(json.dumps({"phrases": []})) is False
    assert backends["llm-phrase"].recognize(
        json.dumps({"phrases": [{"subject": "rice", "phrase": "bowl of rice"}]})) is True


def test_llm_sentence_recognizes_a_no_fit_answer(cfg, db, media_store):
    """Spec 3 r19 section 5: `{"sentences": [], "reason": "..."}` is an
    answer, not an unusable completion -- the provide row is cached and
    sentence_attempt reads the reason off it. An empty listing with no
    reason stays unrecognized."""
    import json

    backends = build_provider(cfg, db, media_store)._backends
    assert backends["llm-sentence"].recognize(
        json.dumps({"sentences": [], "reason": "nothing natural fits"})) is True
    assert backends["llm-sentence"].recognize(json.dumps({"sentences": []})) is False


def test_llm_parse_recognizes_only_a_completion_parses_in_reads(cfg, db, media_store):
    """Spec 3 r16 section 5: llm-parse's LlmBackend.recognize rejects a
    completion parses_in cannot read as a parse."""
    import json

    backends = build_provider(cfg, db, media_store)._backends
    assert backends["llm-parse"].recognize("no json here") is False
    assert backends["llm-parse"].recognize(
        json.dumps({"parses": [{"text": "t", "clauses": [["eat"]]}]})) is True
    assert backends["llm-parse"].producer == "sentence-parser"


def test_the_default_drafter_is_the_cli_transport_whatever_the_judge_is(
        cfg, db, media_store, monkeypatch):
    from thai_syllabus.curated import JudgeConfig
    from thai_syllabus.transport import ClaudeCliTransport
    cfg2 = ProvidersConfig(secrets=cfg.secrets,
                           judge=JudgeConfig(transport="batch", model="m",
                                             price_per_mtok=(2.0, 10.0)))
    calls = _track_reads(monkeypatch)
    backend = build_provider(cfg2, db, media_store)._backends["llm-sentence"]
    assert isinstance(backend.transport._resolve(), ClaudeCliTransport)
    assert calls == []                       # a cli drafter reads no secret
    assert backend.price is None and backend.quota_cost_per_call == 1.0


def test_an_api_drafter_rides_the_judges_account_model_price_and_thinking(
        cfg, db, media_store):
    from thai_syllabus.curated import DrafterConfig, JudgeConfig
    from thai_syllabus.transport import ClaudeApiTransport
    cfg2 = ProvidersConfig(secrets=cfg.secrets,
                           judge=JudgeConfig(transport="batch", model="m", thinking="adaptive",
                                             price_per_mtok=(2.0, 10.0)),
                           drafter=DrafterConfig(transport="api"))
    backend = build_provider(cfg2, db, media_store)._backends["llm-sentence"]
    transport = backend.transport._resolve()
    assert isinstance(transport, ClaudeApiTransport)
    assert transport.model == "m" and transport.thinking == "adaptive"
    assert transport.api_key == "anthropic-key"
    assert backend.price == Price(2.0, 10.0) and backend.quota_cost_per_call == 0.0


def test_the_judge_transports_carry_the_configured_thinking(cfg, db, media_store):
    from thai_syllabus.curated import JudgeConfig
    from thai_syllabus.wiring import _claude_transport
    batch = build_assessor(ProvidersConfig(
        secrets=cfg.secrets,
        judge=JudgeConfig(transport="batch", model="m", price_per_mtok=(2.0, 10.0))),
        db, media_store)
    assert batch._backends["judge"].batch_transport._resolve().thinking == "disabled"
    api = _claude_transport(ProvidersConfig(
        secrets=cfg.secrets,
        judge=JudgeConfig(transport="api", model="m", thinking="adaptive",
                          price_per_mtok=(2.0, 10.0))), cfg.secret_store())._resolve()
    assert api.thinking == "adaptive"


def test_the_judge_and_drafter_transports_carry_the_configured_max_tokens(cfg, db, media_store):
    from thai_syllabus.curated import DrafterConfig, JudgeConfig
    from thai_syllabus.wiring import _claude_transport, _drafter_transport
    batch = build_assessor(ProvidersConfig(
        secrets=cfg.secrets,
        judge=JudgeConfig(transport="batch", model="m", max_tokens=20000,
                          price_per_mtok=(2.0, 10.0))),
        db, media_store)
    assert batch._backends["judge"].batch_transport._resolve().max_tokens == 20000
    api = _claude_transport(ProvidersConfig(
        secrets=cfg.secrets,
        judge=JudgeConfig(transport="api", model="m", max_tokens=20000,
                          price_per_mtok=(2.0, 10.0))), cfg.secret_store())._resolve()
    assert api.max_tokens == 20000
    drafter_api = _drafter_transport(ProvidersConfig(
        secrets=cfg.secrets,
        judge=JudgeConfig(model="m", max_tokens=20000),
        drafter=DrafterConfig(transport="api")), cfg.secret_store())._resolve()
    assert drafter_api.max_tokens == 20000


# --- build_assessor -------------------------------------------------------

def test_build_assessor_registers_judge_and_mechanical(cfg, db, media_store):
    a = build_assessor(cfg, db, media_store)
    assert set(a._backends) >= {"judge", "mechanical"}
    assert isinstance(a, Assessor)


def test_build_assessor_registers_no_fills_backend(cfg, db, media_store):
    """Fills is membership (Syllabus.fills), not an Assess backend (spec 1
    section 3 r8; spec 3 r16): build_assessor's roster carries none."""
    a = build_assessor(cfg, db, media_store)
    assert "fills" not in a._backends


def test_build_assessor_building_the_roster_reads_no_secret_files(cfg, db, media_store, monkeypatch):
    calls = _track_reads(monkeypatch)
    build_assessor(cfg, db, media_store)
    assert calls == []


def test_judge_backend_cli_transport_never_touches_a_secret(cfg, db, media_store, monkeypatch):
    cfg2 = ProvidersConfig(secrets=cfg.secrets, judge=JudgeConfig(transport="cli", model="m"))
    calls = _track_reads(monkeypatch)
    assessor = build_assessor(cfg2, db, media_store)
    judge = assessor._backends["judge"]
    assert judge.transport == "cli"
    assert calls == []


def test_judge_backend_carries_price_and_resolves_media_paths(db, media_store, secret_paths):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          judge=JudgeConfig(transport="api", model="m", price_per_mtok=(2.0, 10.0)))
    a = build_assessor(cfg, db, media_store)
    jb = a._backends["judge"]
    assert jb.price == Price(2.0, 10.0)
    sha = media_store.write(b"img", "jpg")
    db.add_media(sha=sha, kind="picture", ext="jpg", source="t", origin="", licence="?",
                acquired=date(2026, 1, 1))
    assert jb.resolve_path(sha) == media_store.path_for(sha, "jpg")
    assert jb.resolve_path("nope") is None


def test_judge_backend_quota_cost_is_flat_for_cli_zero_otherwise(db, media_store, secret_paths):
    secrets = {n: str(p) for n, p in secret_paths.items()}
    cli_assessor = build_assessor(ProvidersConfig(secrets=secrets,
                                                  judge=JudgeConfig(transport="cli", model="m")),
                                  db, media_store)
    # an api judge is never priceless -- load_providers_config refuses one
    # without a price_per_mtok (spec 3 section 2's cost contract).
    api_assessor = build_assessor(ProvidersConfig(
        secrets=secrets,
        judge=JudgeConfig(transport="api", model="m", price_per_mtok=(2.0, 10.0))),
        db, media_store)
    assert cli_assessor._backends["judge"].quota_cost_per_call == 1.0
    assert api_assessor._backends["judge"].quota_cost_per_call == 0.0
    assert api_assessor._backends["judge"].price == Price(2.0, 10.0)


# --- build_provider: imgfetch/audiofetch as peer fetch backends -----------

def test_build_provider_registers_imgfetch_and_audiofetch_as_peers(db, media_store, secret_paths):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          imgfetch_path="/opt/bin/imgfetch", audiofetch_path="/opt/bin/audiofetch")
    backends = build_provider(cfg, db, media_store)._backends
    assert type(backends["imgfetch"]) is type(backends["audiofetch"])


# --- default_budgets -----------------------------------------------------

def test_default_budgets_includes_forvo_and_learner_defaults(cfg):
    budgets = default_budgets(cfg)
    assert budgets["forvo"].max_asks == 450
    assert budgets["learner"].max_asks == 20


def test_default_budgets_forvo_default_carries_its_22_00z_reset(cfg):
    budgets = default_budgets(cfg)
    assert budgets["forvo"].day_starts == "22:00Z"


def test_an_explicit_null_max_asks_lifts_the_default_cap():
    """`max_asks: null` in providers.yaml is the documented way to run a
    budgeted source uncapped for the day (spec 3 section 9)."""
    cfg = ProvidersConfig(quotas={"forvo": {"max_asks": None}})
    budget = default_budgets(cfg)["forvo"]
    assert budget.max_asks is None
    assert budget.day_starts == "22:00Z"


def test_default_budgets_layers_configured_quotas_over_defaults():
    cfg = ProvidersConfig(quotas={"forvo": {"max_asks": 10},
                                  "judge-api": {"max_cost": 5.0}})
    budgets = default_budgets(cfg)
    assert budgets["forvo"].max_asks == 10  # overridden
    assert budgets["learner"].max_asks == 20  # default still present
    assert budgets["judge-api"].max_cost == 5.0


def test_default_budgets_a_configured_forvo_entry_without_day_starts_keeps_the_default():
    """spec 3 section 7/9: forvo's documented reset time is a property of
    the default, not something every quotas.forvo entry must repeat --
    the layering is field by field, not a whole-Budget replacement."""
    cfg = ProvidersConfig(quotas={"forvo": {"max_asks": 10}})
    budgets = default_budgets(cfg)
    assert budgets["forvo"].day_starts == "22:00Z"


def test_default_budgets_a_configured_forvo_entry_can_override_day_starts():
    cfg = ProvidersConfig(quotas={"forvo": {"day_starts": "18:00+02:00"}})
    budgets = default_budgets(cfg)
    assert budgets["forvo"].day_starts == "18:00+02:00"
    assert budgets["forvo"].max_asks == 450  # untouched by the override


# --- nothing_ttl_for (spec 3 r19 section 6a/9) ----------------------------

def test_nothing_ttl_for_layers_the_forvo_default_of_180_days():
    assert nothing_ttl_for(ProvidersConfig()) == {"forvo": 180}


def test_nothing_ttl_for_a_configured_value_overrides_the_forvo_default():
    cfg = ProvidersConfig(quotas={"forvo": {"nothing_ttl_days": 30}})
    assert nothing_ttl_for(cfg) == {"forvo": 30}


def test_nothing_ttl_for_adds_ageing_for_a_source_with_no_default():
    cfg = ProvidersConfig(quotas={"pexels": {"nothing_ttl_days": 60}})
    ttl = nothing_ttl_for(cfg)
    assert ttl["forvo"] == 180 and ttl["pexels"] == 60


def test_nothing_ttl_for_leaves_an_unconfigured_source_unaged():
    cfg = ProvidersConfig(quotas={"forvo": {"max_asks": 450}})
    assert "wikimedia" not in nothing_ttl_for(cfg)


# --- load_syllabus: round trip over a synthetic curated dir ---------------

def _write_curated_dir(root):
    curated = root / "curated"
    curated.mkdir(parents=True)
    words = [
        {"id": "rice", "thai": "ข้าว", "meaning": "rice", "category": "Food",
         "pron": {"syllables": [{"segments": ["kh", "aa", ""], "vowel_length": "long",
                                 "tone": "low"}], "corroboration": "engines_agree"}},
        {"id": "near", "thai": "ใกล้", "meaning": "near", "category": "Adjectives",
         "pron": {"syllables": [{"segments": ["kl", "ai", ""], "vowel_length": "long",
                                 "tone": "falling"}], "corroboration": "engines_agree"}},
    ]
    (curated / "words.yaml").write_text(yaml.safe_dump(words, allow_unicode=True))
    targets = [{"id": "t-rice", "word": "rice", "skill": "receptive"}]
    (curated / "targets.yaml").write_text(yaml.safe_dump(targets, allow_unicode=True))
    (curated / "profile.yaml").write_text(yaml.safe_dump(
        {"register": "male_colloquial", "emphasis": {}}))
    (curated / "rulebook.yaml").write_text("{}\n", encoding="utf-8")
    (curated / "frequency_th.txt").write_text("", encoding="utf-8")
    return root


def test_load_syllabus_round_trips_words_and_targets(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    syllabus = load_syllabus(root)
    assert isinstance(syllabus, Syllabus)
    assert {w.id for w in syllabus.words} == {"rice", "near"}
    assert {t.id for t in syllabus.targets} == {"t-rice"}


def test_load_syllabus_derives_a_productive_target_at_the_cutoff(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    (root / "curated" / "frequency_th.txt").write_text("ข้าว\n", encoding="utf-8")
    (root / "curated" / "profile.yaml").write_text(yaml.safe_dump(
        {"register": "male_colloquial", "emphasis": {}, "productive_cutoff": 1}))
    syllabus = load_syllabus(root)
    assert "rice/productive" in {t.id for t in syllabus.targets}


def test_load_syllabus_no_productive_suppresses_the_derived_target(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    words = [
        {"id": "rice", "thai": "ข้าว", "meaning": "rice", "category": "Food",
         "no_productive": True,
         "pron": {"syllables": [{"segments": ["kh", "aa", ""], "vowel_length": "long",
                                 "tone": "low"}], "corroboration": "engines_agree"}},
        {"id": "near", "thai": "ใกล้", "meaning": "near", "category": "Adjectives",
         "pron": {"syllables": [{"segments": ["kl", "ai", ""], "vowel_length": "long",
                                 "tone": "falling"}], "corroboration": "engines_agree"}},
    ]
    (root / "curated" / "words.yaml").write_text(yaml.safe_dump(words, allow_unicode=True))
    (root / "curated" / "frequency_th.txt").write_text("ข้าว\n", encoding="utf-8")
    (root / "curated" / "profile.yaml").write_text(yaml.safe_dump(
        {"register": "male_colloquial", "emphasis": {}, "productive_cutoff": 1}))
    syllabus = load_syllabus(root)
    assert "rice/productive" not in {t.id for t in syllabus.targets}


def test_emphasis_from_profile_moves_a_word_earlier_through_load_syllabus(tmp_path):
    """Profile.emphasis reaches Syllabus.order() through load_syllabus's
    categories wiring: a lower-frequency word in an emphasized category
    outranks a higher-frequency word outside it.
    """
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=(word("rice", "ข้าว", "rice"), word("red", "แดง", "red")),  # rice, red
        targets=(target("rice/receptive", "rice"), target("red/receptive", "red")),
        graphemes=(), confusions=(), pairs=(),
        profile=Profile(register="male_colloquial", emphasis={"Food": 3.0}),
        rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset({"rice"})),
                   Category(name="Colors", members=frozenset({"red"})))))
    # red ranks more frequent (1) than rice (2); Food's 3x emphasis must
    # still bring rice's target ahead of red's.
    (root / "curated" / "frequency_th.txt").write_text("แดง\nข้าว\n", encoding="utf-8")

    syllabus = load_syllabus(root)
    ids = [e.id for e in syllabus.order() if e.kind == "word_target"]
    assert ids.index("rice/receptive") < ids.index("red/receptive")


def test_load_syllabus_wires_a_real_assessment_reader(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    syllabus = load_syllabus(root)
    db = SyllabusDb(root / "syllabus.db")
    key = JudgeKey(rubric_sha=sha(""), subject="n1", identity="", role="r1")
    db.append(port="assess", backend="judge", key=key, subject="n1",
              question={"role": "r1", "artifact_sha": None, "rubric": None},
              answer={"value": True})
    # a fresh load_syllabus call re-opens the same db file -- the verdict
    # written above must be visible through Syllabus.assessments.
    syllabus2 = load_syllabus(root)
    answer = syllabus2.assessments.verdict("judge", key)
    assert answer is not None and answer.answer["value"] is True


def test_load_syllabus_sentences_come_from_the_db(tmp_path):
    from datetime import date
    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")
    # _write_curated_dir registers only "rice"/"near" -- clauses must stay
    # inside that vocabulary for check_sentence (load_syllabus's own
    # refusal check) to pass.
    db.add_sentence(text_sha="s1", text="ข้าว", clauses=(("rice",),), gloss="rice",  # rice
                    voice="learner_voice", source="llm", origin="draft", licence="n/a",
                    acquired=date(2026, 1, 1))
    syllabus = load_syllabus(root)
    assert any(s.text == "ข้าว" for s in syllabus.sentences)  # rice


def test_load_syllabus_refuses_a_sentence_naming_an_unregistered_word(tmp_path):
    from datetime import date
    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")
    db.add_sentence(text_sha="s1", text="แมว", clauses=(("cat",),), gloss="cat",  # cat
                    voice="learner_voice", source="llm", origin="draft", licence="n/a",
                    acquired=date(2026, 1, 1))
    with pytest.raises(ValueError, match="cat"):
        load_syllabus(root)


def test_load_syllabus_with_a_given_sentences_sequence_does_not_read_the_db(tmp_path):
    from datetime import date
    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")
    # A row naming an unregistered word: reading it through
    # db.all_sentences() refuses the deck. `sentences=()` bypasses the
    # table read entirely (Task 7's parse step).
    db.add_sentence(text_sha="s1", text="แมว", clauses=(("cat",),), gloss="cat",  # cat
                    voice="learner_voice", source="llm", origin="draft", licence="n/a",
                    acquired=date(2026, 1, 1))
    syllabus = load_syllabus(root, db=db, sentences=())
    assert syllabus.sentences == ()


def test_load_syllabus_media_index_reflects_current_best(tmp_path):
    from thai_syllabus.rulebook import PICTURE_FIT_RUBRIC

    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")
    db.append(port="provide", backend="openverse",
             key=ProvideKey(source="openverse", kind="", query="rice"),
             subject="rice", question={"kind": "picture", "params": {}},
             answer={"items": [{"sha": "abc"}]}, cost=0.0)
    # rubric must match load_syllabus's own rubrics_for(rules) (the default
    # PICTURE_FIT_RUBRIC, no rulebook.yaml overlay here) -- _DbMediaIndex now
    # threads current_rubric through current_best (Task 11), so a verdict
    # under a stale/mismatched rubric would not count.
    db.append(port="assess", backend="judge",
             key=JudgeKey.for_rule(PICTURE_FIT_RUBRIC, "abc", "rice", "picture-for-word"),
             subject="rice",
             question={"role": "picture-for-word", "artifact_sha": "abc",
                      "rubric": PICTURE_FIT_RUBRIC, "kind": "picture"},
             answer={"value": True}, cost=0.0)
    syllabus = load_syllabus(root)
    assert syllabus.media.has_picture("rice") is True
    assert syllabus.media.has_picture("near") is False


# --- _DbMediaIndex: picture_sha / recording_provenance / rendition_provenance

def test_db_media_index_picture_sha_and_recording_provenance_reflect_current_best(db):
    from datetime import date
    db.append(port="provide", backend="openverse",
             key=ProvideKey(source="openverse", kind="", query="rice"),
             subject="rice", question={"kind": "picture", "params": {}},
             answer={"items": [{"sha": "abc"}]}, cost=0.0)
    db.append(port="assess", backend="judge",
             key=JudgeKey.for_rule(None, "abc", "rice", "picture-for-word"),
             subject="rice",
             question={"role": "picture-for-word", "artifact_sha": "abc", "rubric": None,
                      "kind": "picture"},
             answer={"value": True}, cost=0.0)
    db.append(port="provide", backend="forvo",
             key=ProvideKey(source="forvo", kind="", query="rice"),
             subject="rice", question={"kind": "recording", "params": {}},
             answer={"items": [{"sha": "rec1"}]}, cost=0.0)
    # derivations.current_best does not yet rank a bare "mechanical" pass
    # for recordings (Task 5 adds that) -- a judge pass under role
    # "recording-for-word" is what makes a recording candidate current-best
    # today.
    db.append(port="assess", backend="judge",
             key=JudgeKey.for_rule(None, "rec1", "rice", "recording-for-word"),
             subject="rice",
             question={"role": "recording-for-word", "artifact_sha": "rec1", "rubric": None,
                      "kind": "recording"},
             answer={"value": True}, cost=0.0)
    db.add_speaker(Speaker(id="somchai", kind="native"))
    db.add_media(sha="rec1", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id="somchai")

    media = _DbMediaIndex(db=db)
    assert media.picture_sha("rice") == "abc"
    assert media.picture_sha("near") is None

    prov = media.recording_provenance("rice")
    assert prov["source"] == "forvo"
    assert prov["speaker_id"] == "somchai"
    assert prov["speaker"] == Speaker(id="somchai", kind="native")
    assert media.recording_provenance("near") is None


# --- _DbMediaIndex: rendition_provenance/rendition_speakers's two-level read
#
# A pair's rendition is current-best under its OWN subject (the pair id,
# role "rendition-for-pair" -- the "rendition" backend's row);
# that row's params["members"] names which per-member recording actually
# backs it. Only absent a pair-level rendition row does rendition_provenance
# fall back to each member's own current-best recording -- and
# rendition_speakers deliberately skips that fallback (class docstring), so
# a mixed-speaker/partial pair stays in Syllabus.gaps().missing_renditions
# even though the deck still compiles with a warning.

def _tone_pair(db=None):
    from thai_syllabus.entities import MinimalPair, SoundConfusion
    from thai_syllabus.ids import ConfusionId, PairId

    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    near = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    far = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    pair = MinimalPair.create(id=PairId("tone:mid-low/klai"), confusion=confusion,
                              members=(near, far))
    return confusion, pair


def _seed_member_recording(db, subject, sha, speaker_id, sex="unknown",
                           age_band="unknown", region="unknown"):
    db.add_speaker(Speaker(id=speaker_id, kind="native", sex=sex,
                           age_band=age_band, region=region))
    db.append(port="provide", backend="forvo",
             key=ProvideKey(source="forvo", kind="", query=subject),
             subject=subject, question={"kind": "recording", "params": {}},
             answer={"items": [{"sha": sha}]}, cost=0.0)
    db.append(port="assess", backend="judge",
             key=JudgeKey.for_rule(None, sha, subject, "recording-for-word"),
             subject=subject,
             question={"role": "recording-for-word", "artifact_sha": sha, "rubric": None,
                      "kind": "recording"},
             answer={"value": True}, cost=0.0)
    db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id=speaker_id)


def test_db_media_index_rendition_provenance_prefers_the_pair_level_rendition_row(db):
    confusion, pair = _tone_pair(db)
    # Members' own current-best recordings would say "malee"/"somchai" (two
    # different speakers) -- the pair-level rendition row's own members
    # (both "somchai") must win over that when one is current-best.
    _seed_member_recording(db, "near", "sha-near-own", "malee")
    _seed_member_recording(db, "far", "sha-far-own", "somchai")
    db.add_media(sha="sha-near-rendition", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id="somchai")
    db.add_media(sha="sha-far-rendition", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id="somchai")

    # the "rendition" mechanical backend's own row shape.
    db.append(port="assess", backend="rendition",
             key=MechanicalKey(check="rendition", params="v1", subject=str(pair.id),
                               artifact_sha="joined"),
             subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": "joined-sha",
                      "rubric": None, "kind": "rendition",
                      "params": {"members": {"near": "sha-near-rendition",
                                             "far": "sha-far-rendition"}}},
             answer={"value": True}, cost=0.0)

    media = _DbMediaIndex(db=db, pairs=(pair,))
    rows = media.rendition_provenance(pair.id)
    assert {r["speaker_id"] for r in rows} == {"somchai"}
    assert media.rendition_speakers(confusion.id) == frozenset({"somchai"})


def test_db_media_index_rendition_provenance_falls_back_to_member_recordings_without_a_rendition_row(db):
    confusion, pair = _tone_pair(db)
    _seed_member_recording(db, "near", "sha-near", "somchai")
    _seed_member_recording(db, "far", "sha-far", "malee")

    media = _DbMediaIndex(db=db, pairs=(pair,))
    rows = media.rendition_provenance(pair.id)
    assert {r["speaker_id"] for r in rows} == {"somchai", "malee"}
    assert media.rendition_provenance("no-such-pair") == ()
    # rendition_speakers skips the fallback -- a partial/mixed-speaker pair
    # with no pair-level rendition row must not read as "covered".
    assert media.rendition_speakers(confusion.id) == frozenset()


# --- _DbMediaIndex.speakers_of -------------------------------------------

def test_speakers_of_recording_returns_the_shared_speaker_with_its_attributes(db):
    orange = word("orange", "ส้ม")  # orange
    rice = word("rice", "ข้าว")  # rice
    _seed_member_recording(db, "orange", "sha-orange", "somchai", sex="male", region="TH")
    _seed_member_recording(db, "rice", "sha-rice", "somchai", sex="male", region="TH")

    media = _DbMediaIndex(db=db, words=(orange, rice))
    assert media.speakers_of("recording") == (
        Speaker(id="somchai", kind="native", sex="male", region="TH"),)


def test_speakers_of_sentence_returns_the_sentence_recordings_speaker(db):
    rice = word("rice", "ข้าว")  # rice
    s = sentence(((rice.id,),), thai_of(rice))  # rice
    note_id = sentence_note_id(s)
    _seed_member_recording(db, note_id, "sha-sentence", "malee", sex="female")

    media = _DbMediaIndex(db=db, sentences=(s,))
    assert media.speakers_of("sentence") == (Speaker(id="malee", kind="native", sex="female"),)


def test_speakers_of_rendition_returns_the_pairs_rendition_speaker_once(db):
    _, pair = _tone_pair(db)
    _seed_member_recording(db, "near", "sha-near-own", "malee")
    _seed_member_recording(db, "far", "sha-far-own", "somchai")
    db.add_media(sha="sha-near-rendition", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id="somchai")
    db.add_media(sha="sha-far-rendition", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id="somchai")
    # the "rendition" backend's own row shape -- both members' current-
    # best rendition rows resolve to the SAME speaker, so speakers_of must
    # report it once, not twice.
    db.append(port="assess", backend="rendition",
             key=MechanicalKey(check="rendition", params="v1", subject=str(pair.id),
                               artifact_sha="joined"),
             subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": "joined-sha",
                      "rubric": None, "kind": "rendition",
                      "params": {"members": {"near": "sha-near-rendition",
                                             "far": "sha-far-rendition"}}},
             answer={"value": True}, cost=0.0)

    media = _DbMediaIndex(db=db, pairs=(pair,))
    assert media.speakers_of("rendition") == (Speaker(id="somchai", kind="native"),)


def test_speakers_of_an_unknown_corpus_raises_value_error_naming_it(db):
    media = _DbMediaIndex(db=db)
    with pytest.raises(ValueError, match="bogus"):
        media.speakers_of("bogus")


def test_syllabus_gaps_missing_renditions_distinguishes_a_real_rendition_from_the_fallback(db):
    real_confusion, real_pair = _tone_pair(db)
    _seed_member_recording(db, "near", "sha-near-own", "somchai")
    _seed_member_recording(db, "far", "sha-far-own", "somchai")
    db.append(port="assess", backend="rendition",
             key=MechanicalKey(check="rendition", params="v1", subject=str(real_pair.id),
                               artifact_sha="joined"),
             subject=real_pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": "joined-sha",
                      "rubric": None, "kind": "rendition",
                      "params": {"members": {"near": "sha-near-own", "far": "sha-far-own"}}},
             answer={"value": True}, cost=0.0)

    from thai_syllabus.entities import MinimalPair, SoundConfusion
    from thai_syllabus.ids import ConfusionId, PairId
    fallback_confusion = SoundConfusion(id=ConfusionId("vowel:a-aa"), dimension="length",
                                        sounds=("short", "long"))
    short = word("short", "กะ", syllables=(syl(vowel="a", length="short"),))
    long_ = word("long", "กา", syllables=(syl(vowel="a", length="long"),))
    fallback_pair = MinimalPair.create(id=PairId("vowel:a-aa/ka"), confusion=fallback_confusion,
                                       members=(short, long_))
    _seed_member_recording(db, "short", "sha-short", "somchai")
    _seed_member_recording(db, "long", "sha-long", "malee")

    media = _DbMediaIndex(db=db, pairs=(real_pair, fallback_pair))
    syllabus = Syllabus(confusions=(real_confusion, fallback_confusion),
                        pairs=(real_pair, fallback_pair), media=media)
    gaps = syllabus.gaps()
    assert real_confusion.id not in gaps.missing_renditions
    assert fallback_confusion.id in gaps.missing_renditions


def test_load_syllabus_refuses_a_deck_without_a_frequency_corpus(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    (root / "curated" / "frequency_th.txt").unlink()
    with pytest.raises(FileNotFoundError, match="frequency_th.txt"):
        load_syllabus(root)


def test_load_syllabus_reads_a_frequency_file_when_present(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    (root / "curated" / "frequency_th.txt").write_text("ข้าว\nใกล้\n", encoding="utf-8")
    syllabus = load_syllabus(root)
    assert syllabus.frequency["rice"] == 1
    assert syllabus.frequency["near"] == 2


# --- load_syllabus: rulebook overlay + build_sourcing ---------------------

def _minimal_deck(tmp_path):
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=(word("slow", "ช้า", "slow"),), targets=(target("slow/receptive", "slow"),),
        graphemes=(), confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(),
        categories=(Category(name="Adjectives", members=frozenset({"slow"})),)))
    (root / "curated" / "frequency_th.txt").write_text("", encoding="utf-8")
    return root


def test_load_syllabus_applies_severity_overlay(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "rulebook.yaml").write_text(
        "severities: {target/picture-required: warn}\n", encoding="utf-8")
    syl_ = load_syllabus(root)
    assert {r.id: r.severity for r in syl_.rules}["target/picture-required"] == "warn"


def test_build_sourcing_assembles_rubrics_and_prior(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "image_candidates: 2\nimgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n", encoding="utf-8")
    ctx = build_sourcing(root)
    assert ctx.image_candidates == 2 and "picture-for-word" in ctx.rubrics
    assert ctx.provenance_prior == ("commission", "forvo", "tts")


def test_build_sourcing_threads_caps_and_pools(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "attempt_cap: 3\ntransient_cap: 2\nimgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n"
        "tts: {male_voices: [th-TH-Chirp3-HD-Puck], "
        "female_voices: [th-TH-Chirp3-HD-Aoede]}\n", encoding="utf-8")
    ctx = build_sourcing(root)
    assert ctx.attempt_cap == 3    # value written by the fixture
    assert ctx.transient_cap == 2
    assert ctx.voices["male"] and ctx.voices["female"]


def test_load_derivations_carries_the_parameters_build_sourcing_runs_under(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "attempt_cap: 3\ntransient_cap: 2\nimgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n", encoding="utf-8")
    derivations = load_derivations(root)
    ctx = build_sourcing(root)
    assert derivations.current_rubric == ctx.rubrics
    assert derivations.prior == ctx.provenance_prior
    assert derivations.attempt_cap == ctx.attempt_cap == 3
    assert derivations.transient_cap == ctx.transient_cap == 2
    assert derivations.sources_for is ctx.sources_for
    assert derivations.db is derivations.syllabus.assessments


def test_the_sentence_no_fit_cap_reaches_both_derivations_and_sourcing(tmp_path):
    """Spec 3 r19 section 9: providers.yaml's own sentence_nothing_cap is
    the cap the run's attempt and the feedback screen's fold both use."""
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "sentence_nothing_cap: 5\nimgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n", encoding="utf-8")
    derivations = load_derivations(root)
    ctx = build_sourcing(root)
    assert derivations.sentence_nothing_cap == ctx.sentence_nothing_cap == 5


def test_the_sentence_no_fit_cap_defaults_to_three(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n",
        encoding="utf-8")
    assert build_sourcing(root).sentence_nothing_cap == DEFAULT_SENTENCE_NOTHING_CAP == 3


def test_the_sentence_clause_cap_reaches_sourcing(tmp_path):
    """Spec 3 r23 section 5/8: providers.yaml's own sentence_max_clauses
    is the cap the drafting prompt and the acceptance loop both read off
    Sourcing. The feedback screen has no fold over it, so load_derivations
    does not carry it (unlike sentence_nothing_cap)."""
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "sentence_max_clauses: 4\nimgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n", encoding="utf-8")
    assert build_sourcing(root).sentence_max_clauses == 4


def test_the_sentence_clause_cap_defaults_to_two(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n",
        encoding="utf-8")
    assert build_sourcing(root).sentence_max_clauses == DEFAULT_SENTENCE_MAX_CLAUSES == 2


def test_the_sentence_introducible_cap_reaches_sourcing(tmp_path):
    """Spec 3 r24 section 5/8: providers.yaml's own sentence_introducible_per_ask
    is the cap sentence_attempt's own target selection reads off Sourcing."""
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "sentence_introducible_per_ask: 3\nimgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n", encoding="utf-8")
    assert build_sourcing(root).sentence_introducible_per_ask == 3


def test_the_sentence_introducible_cap_defaults_to_five(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n",
        encoding="utf-8")
    assert (build_sourcing(root).sentence_introducible_per_ask
           == DEFAULT_SENTENCE_INTRODUCIBLE_PER_ASK == 5)


def test_nothing_ttl_reaches_both_derivations_and_sourcing(tmp_path):
    """Spec 3 r19 section 9: providers.yaml's own quotas.forvo.nothing_ttl_days
    is the ageing map derivations.tried_sources folds over, in both the
    run's Sourcing and the feedback screen's Derivations."""
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "quotas: {forvo: {nothing_ttl_days: 30}}\nimgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n", encoding="utf-8")
    derivations = load_derivations(root)
    ctx = build_sourcing(root)
    assert derivations.nothing_ttl == ctx.nothing_ttl == {"forvo": 30}


def test_nothing_ttl_defaults_to_the_forvo_180_day_entry(tmp_path):
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n",
        encoding="utf-8")
    assert build_sourcing(root).nothing_ttl == {"forvo": 180}


def test_build_sourcing_shares_one_db_handle_with_the_syllabus(tmp_path):
    # load_syllabus, left to open its own SyllabusDb, would give
    # Sourcing.db and syllabus.assessments/media.db two separate
    # connections whose writes and reads could disagree.
    # build_sourcing must open db/bundle once and inject them.
    root = _minimal_deck(tmp_path)
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\naudiofetch_path: /opt/bin/audiofetch\n",
        encoding="utf-8")
    ctx = build_sourcing(root)
    assert ctx.db is ctx.syllabus.assessments


# --- the batch judge authenticates like the api one (C1) -------------------

def test_batch_judge_transport_carries_the_anthropic_secret(cfg, db, media_store, monkeypatch):
    cfg2 = ProvidersConfig(secrets=cfg.secrets,
                           judge=JudgeConfig(transport="batch", model="m",
                                             price_per_mtok=(2.0, 10.0)))
    calls = _track_reads(monkeypatch)
    judge = build_assessor(cfg2, db, media_store)._backends["judge"]
    assert calls == []                      # still lazy: no secret read to build the roster
    transport = judge.batch_transport._resolve()
    assert transport.api_key == "anthropic-key"
    assert transport.model == "m"
    assert calls == ["anthropic"]
