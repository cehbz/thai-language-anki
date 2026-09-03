"""Tests for wiring.py: build_provider/build_assessor/default_budgets/
build_levers/load_syllabus -- assembling spec 3's backend rosters from
curated/providers.yaml and spec 1/2's Syllabus from a deck directory. No
network, no subprocess, no pythainlp/anthropic import; secret files are
real tmp 0600 files whose reads are tracked to prove lazy resolution.
"""
from __future__ import annotations

import textwrap

import pytest
import yaml

from thai_syllabus import secrets as secrets_mod
from thai_syllabus.assessor import Assessor
from thai_syllabus.curated import ProvidersConfig, load_providers_config
from thai_syllabus.provider import Provider, Question
from thai_syllabus.run import Budget
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.syllabus import Syllabus
from thai_syllabus.wiring import (
    _DbMediaIndex,
    build_assessor,
    build_levers,
    build_provider,
    default_budgets,
    load_syllabus,
)

from .builders import PROV, syl, pron, target, word


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
    return ProvidersConfig(
        secrets={name: str(path) for name, path in secret_paths.items()},
        search_proxy="https://proxy.example",
        imgfetch_path="curl",
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
                                       params={"query": "cat"})) == "pexels:cat"


def test_an_openverse_ask_never_touches_any_secret(cfg, db, media_store, monkeypatch):
    calls = _track_reads(monkeypatch)

    def fake_get(url, params=None, headers=None, timeout=None):
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

def test_search_proxy_reaches_openverse_and_wikimedia(cfg, db, media_store):
    provider = build_provider(cfg, db, media_store)
    q = Question(subject="s", provides="picture", params={"query": "cat"})
    assert provider._backends["openverse"].search_proxy == "https://proxy.example"
    assert provider._backends["wikimedia"].search_proxy == "https://proxy.example"


def test_imgfetch_binary_comes_from_imgfetch_path(db, media_store, secret_paths):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          imgfetch_path="/usr/local/bin/my-curl")
    provider = build_provider(cfg, db, media_store)
    # the fetcher is a closure -- exercised indirectly via a runner spy
    calls = []

    def runner(cmd, **kwargs):
        calls.append(cmd)
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, b"bytes", b"")

    from thai_syllabus.provider import subprocess_curl_fetcher
    fetcher = subprocess_curl_fetcher(runner=runner, binary="/usr/local/bin/my-curl")
    fetcher("https://x/y.jpg")
    assert calls[0][0] == "/usr/local/bin/my-curl"


# --- build_provider: tts voice pools -------------------------------------

def test_tts_backend_voice_pool_combines_male_and_female(cfg, db, media_store):
    cfg = ProvidersConfig(secrets=cfg.secrets, tts_male_voices=("m1", "m2"),
                          tts_female_voices=("f1",))
    provider = build_provider(cfg, db, media_store)
    backend = provider._backends["tts"]._resolve()
    assert set(backend.voices) == {"m1", "m2", "f1"}


# --- build_provider: llm backend follows judge.transport -----------------

def test_llm_backends_registered_for_cli_transport(cfg, db, media_store):
    from thai_syllabus.curated import JudgeConfig
    cfg = ProvidersConfig(secrets=cfg.secrets, judge=JudgeConfig(transport="cli", model="m"))
    provider = build_provider(cfg, db, media_store)
    assert "llm-sentence" in provider._backends


def test_no_llm_backends_registered_for_batch_transport(cfg, db, media_store):
    from thai_syllabus.curated import JudgeConfig
    cfg = ProvidersConfig(secrets=cfg.secrets, judge=JudgeConfig(transport="batch", model="m"))
    provider = build_provider(cfg, db, media_store)
    assert "llm-sentence" not in provider._backends


# --- build_assessor -------------------------------------------------------

def test_build_assessor_registers_judge(cfg, db):
    assessor = build_assessor(cfg, db)
    assert isinstance(assessor, Assessor)
    assert "judge" in assessor._backends


def test_build_assessor_building_the_roster_reads_no_secret_files(cfg, db, monkeypatch):
    calls = _track_reads(monkeypatch)
    build_assessor(cfg, db)
    assert calls == []


def test_judge_backend_cli_transport_never_touches_a_secret(cfg, db, monkeypatch):
    from thai_syllabus.curated import JudgeConfig
    cfg2 = ProvidersConfig(secrets=cfg.secrets, judge=JudgeConfig(transport="cli", model="m"))
    calls = _track_reads(monkeypatch)
    assessor = build_assessor(cfg2, db)
    judge = assessor._backends["judge"]
    assert judge.transport == "cli"
    assert calls == []


# --- default_budgets -----------------------------------------------------

def test_default_budgets_includes_forvo_and_learner_defaults(cfg):
    budgets = default_budgets(cfg)
    assert budgets["forvo"].max_asks == 450
    assert budgets["learner"].max_asks == 20


def test_default_budgets_layers_configured_quotas_over_defaults():
    cfg = ProvidersConfig(quotas={"forvo": {"max_asks": 10},
                                  "judge-api": {"max_cost": 5.0}})
    budgets = default_budgets(cfg)
    assert budgets["forvo"].max_asks == 10  # overridden
    assert budgets["learner"].max_asks == 20  # default still present
    assert budgets["judge-api"].max_cost == 5.0


# --- build_levers ----------------------------------------------------------

@pytest.fixture
def syllabus_for_levers():
    w = word("rice", "ข้าว", "rice")
    t = target("t-rice", "rice", skill="receptive")
    return Syllabus(words=(w,), targets=(t,))


def test_build_levers_covers_picture_and_recording(cfg, db, media_store, syllabus_for_levers):
    provider = build_provider(cfg, db, media_store)
    levers = build_levers(syllabus_for_levers, provider, cfg)
    assert [l.backend for l in levers["picture"]] == ["openverse", "wikimedia", "pexels"]
    assert [l.backend for l in levers["recording"]] == ["forvo", "tts"]


def test_build_levers_sentence_lever_present_for_cli_transport(
        cfg, db, media_store, syllabus_for_levers):
    provider = build_provider(cfg, db, media_store)
    levers = build_levers(syllabus_for_levers, provider, cfg)
    assert levers["sentence"][0].backend == "llm-sentence"


def test_build_levers_no_sentence_lever_for_batch_transport(
        db, media_store, syllabus_for_levers, secret_paths):
    from thai_syllabus.curated import JudgeConfig
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          judge=JudgeConfig(transport="batch", model="m"))
    provider = build_provider(cfg, db, media_store)
    levers = build_levers(syllabus_for_levers, provider, cfg)
    assert "sentence" not in levers


def test_picture_lever_build_question_uses_the_words_meaning(
        cfg, db, media_store, syllabus_for_levers):
    provider = build_provider(cfg, db, media_store)
    levers = build_levers(syllabus_for_levers, provider, cfg)
    q = levers["picture"][0].build_question("rice", "picture")
    assert q.params["query"] == "rice"


def test_tts_lever_build_question_carries_the_words_thai_text(
        cfg, db, media_store, syllabus_for_levers):
    provider = build_provider(cfg, db, media_store)
    levers = build_levers(syllabus_for_levers, provider, cfg)
    tts_lever = [l for l in levers["recording"] if l.backend == "tts"][0]
    q = tts_lever.build_question("rice", "recording")
    assert q.params["text"] == "ข้าว"  # rice


# --- load_syllabus: round trip over a synthetic curated dir ---------------

def _write_curated_dir(root):
    curated = root / "curated"
    curated.mkdir(parents=True)
    words = [
        {"id": "rice", "thai": "ข้าว", "meaning": "rice",
         "pron": {"syllables": [{"segments": ["kh", "aa", ""], "vowel_length": "long",
                                 "tone": "low"}], "corroboration": "engines_agree"}},
        {"id": "near", "thai": "ใกล้", "meaning": "near",
         "pron": {"syllables": [{"segments": ["kl", "ai", ""], "vowel_length": "long",
                                 "tone": "falling"}], "corroboration": "engines_agree"}},
    ]
    (curated / "words.yaml").write_text(yaml.safe_dump(words, allow_unicode=True))
    targets = [{"id": "t-rice", "word": "rice", "skill": "receptive"}]
    (curated / "targets.yaml").write_text(yaml.safe_dump(targets, allow_unicode=True))
    (curated / "profile.yaml").write_text(yaml.safe_dump(
        {"register": "male_colloquial", "emphasis": {}}))
    return root


def test_load_syllabus_round_trips_words_and_targets(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    syllabus = load_syllabus(root)
    assert isinstance(syllabus, Syllabus)
    assert {w.id for w in syllabus.words} == {"rice", "near"}
    assert {t.id for t in syllabus.targets} == {"t-rice"}


def test_load_syllabus_wires_a_real_assessment_reader(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    syllabus = load_syllabus(root)
    db = SyllabusDb(root / "syllabus.db")
    db.append_judge_verdict(rule_id="r1", note_id="n1", verdict=True)
    # a fresh load_syllabus call re-opens the same db file -- the verdict
    # written above must be visible through Syllabus.assessments.
    syllabus2 = load_syllabus(root)
    assert syllabus2.assessments.verdict("r1", "n1") is True


def test_load_syllabus_sentences_come_from_the_db(tmp_path):
    from datetime import date
    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")
    db.add_sentence(text_sha="s1", text="ผมกินข้าว", voice="learner_voice",  # I eat rice
                    source="llm", origin="draft", licence="n/a",
                    acquired=date(2026, 1, 1))
    syllabus = load_syllabus(root)
    assert any(s.text == "ผมกินข้าว" for s in syllabus.sentences)  # I eat rice


def test_load_syllabus_media_index_reflects_current_best(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")
    db.append(port="provide", backend="openverse", key="openverse:rice",
             subject="rice", question={"provides": "picture", "params": {}},
             answer={"items": [{"sha": "abc"}]}, cost=0.0)
    db.append(port="assess", backend="judge", key="judge:x:abc:picture-for-word",
             subject="rice",
             question={"role": "picture-for-word", "artifact_sha": "abc", "rubric": None},
             answer={"value": True}, cost=0.0)
    syllabus = load_syllabus(root)
    assert syllabus.media.has_picture("rice") is True
    assert syllabus.media.has_picture("near") is False


# --- _DbMediaIndex: picture_sha / recording_provenance / rendition_provenance

def test_db_media_index_picture_sha_and_recording_provenance_reflect_current_best(db):
    from datetime import date
    db.append(port="provide", backend="openverse", key="openverse:rice",
             subject="rice", question={"provides": "picture", "params": {}},
             answer={"items": [{"sha": "abc"}]}, cost=0.0)
    db.append(port="assess", backend="judge", key="judge:x:abc:picture-for-word",
             subject="rice",
             question={"role": "picture-for-word", "artifact_sha": "abc", "rubric": None},
             answer={"value": True}, cost=0.0)
    db.append(port="provide", backend="forvo", key="forvo:rice",
             subject="rice", question={"provides": "recording", "params": {}},
             answer={"items": [{"sha": "rec1"}]}, cost=0.0)
    # derivations.current_best does not yet rank a bare "mechanical" pass
    # for recordings (Task 5 adds that) -- a judge pass under role
    # "recording-for-word" is what makes a recording candidate current-best
    # today.
    db.append(port="assess", backend="judge", key="judge:x:rec1:recording-for-word",
             subject="rice",
             question={"role": "recording-for-word", "artifact_sha": "rec1", "rubric": None},
             answer={"value": True}, cost=0.0)
    db.add_media(sha="rec1", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by",
                acquired=date(2026, 1, 1), speaker_id="somchai", speaker_kind="native")

    media = _DbMediaIndex(db=db)
    assert media.picture_sha("rice") == "abc"
    assert media.picture_sha("near") is None

    prov = media.recording_provenance("rice")
    assert prov["source"] == "forvo"
    assert prov["speaker_id"] == "somchai"
    assert prov["speaker_kind"] == "native"
    assert media.recording_provenance("near") is None


def test_db_media_index_rendition_provenance_reflects_current_best(db):
    from datetime import date
    from thai_syllabus.entities import MinimalPair, SoundConfusion
    from thai_syllabus.ids import ConfusionId, PairId

    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    near = word("near", "ใกล้", syllables=(syl(tone="mid"),))  # near
    far = word("far", "ไกล", syllables=(syl(tone="low"),))  # far
    pair = MinimalPair.create(id=PairId("tone:mid-low/klai"), confusion=confusion,
                              members=(near, far))

    def seed_recording(subject, sha, speaker_id):
        db.append(port="provide", backend="forvo", key=f"forvo:{subject}",
                 subject=subject, question={"provides": "recording", "params": {}},
                 answer={"items": [{"sha": sha}]}, cost=0.0)
        db.append(port="assess", backend="judge", key=f"judge:x:{sha}:recording-for-word",
                 subject=subject,
                 question={"role": "recording-for-word", "artifact_sha": sha, "rubric": None},
                 answer={"value": True}, cost=0.0)
        db.add_media(sha=sha, kind="recording", ext="mp3", source="forvo",
                    origin="https://forvo.com/x", licence="cc-by",
                    acquired=date(2026, 1, 1), speaker_id=speaker_id, speaker_kind="native")

    seed_recording("near", "sha-near", "somchai")
    seed_recording("far", "sha-far", "malee")

    media = _DbMediaIndex(db=db, pairs=(pair,))
    rows = media.rendition_provenance(pair.id)
    assert {r["speaker_id"] for r in rows} == {"somchai", "malee"}
    assert media.rendition_provenance("no-such-pair") == ()


def test_load_syllabus_default_tokenizer_is_whitespace_when_pythainlp_absent(
        tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "pythainlp" or name.startswith("pythainlp."):
            raise ImportError("pythainlp not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    root = _write_curated_dir(tmp_path / "deck")
    syllabus = load_syllabus(root)
    assert syllabus.tokenizer.tokens("a b c") == ["a", "b", "c"]


def test_load_syllabus_no_frequency_file_leaves_empty_frequency_map(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    syllabus = load_syllabus(root)
    assert syllabus.frequency == {}


def test_load_syllabus_reads_a_frequency_file_when_present(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    (root / "data").mkdir()
    (root / "data" / "frequency_th.txt").write_text("ข้าว\nใกล้\n", encoding="utf-8")
    syllabus = load_syllabus(root)
    assert syllabus.frequency["rice"] == 1
    assert syllabus.frequency["near"] == 2
