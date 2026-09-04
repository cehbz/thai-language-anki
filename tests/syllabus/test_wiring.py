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
from thai_syllabus.curated import (
    CuratedBundle,
    JudgeConfig,
    ProvidersConfig,
    RulebookConfig,
    load_providers_config,
    save_curated,
)
from thai_syllabus.entities import Category, Target
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
    load_syllabus,
)

from .builders import PROV, sentence, syl, pron, target, word


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


def test_imgfetch_binary_comes_from_imgfetch_path(db, media_store, secret_paths, monkeypatch):
    cfg = ProvidersConfig(secrets={n: str(p) for n, p in secret_paths.items()},
                          imgfetch_path="/opt/bin/imgfetch",
                          audiofetch_path="/opt/bin/audiofetch")
    provider = build_provider(cfg, db, media_store)
    calls = []

    def fake_run(cmd, **kwargs):
        import subprocess as sp
        Path(cmd[2]).write_bytes(b"bytes")
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


# --- build_provider: llm backend follows judge.transport -----------------

def test_llm_backends_registered_for_cli_transport(cfg, db, media_store):
    from thai_syllabus.curated import JudgeConfig
    cfg = ProvidersConfig(secrets=cfg.secrets, judge=JudgeConfig(transport="cli", model="m"))
    provider = build_provider(cfg, db, media_store)
    assert "llm-sentence" in provider._backends


def test_llm_backends_registered_for_a_batch_judge_with_an_anthropic_secret(
        cfg, db, media_store, monkeypatch):
    """A batch judge has no single-question transport, but sentence/phrase/
    entry drafting still needs one -- it rides a lazy api transport on the
    same anthropic secret, so a deck whose verdicts go through batches can
    still draft sentences."""
    cfg2 = ProvidersConfig(secrets=cfg.secrets,
                           judge=JudgeConfig(transport="batch", model="m",
                                             price_per_mtok=(2.0, 10.0)))
    calls = _track_reads(monkeypatch)
    backends = build_provider(cfg2, db, media_store)._backends
    assert {"llm-sentence", "llm-phrase", "llm-entry"} <= set(backends)
    assert calls == []                       # still lazy: no secret read to build the roster
    transport = backends["llm-sentence"].transport
    transport.complete  # a .complete-shaped lazy transport, not a batch one
    assert not hasattr(transport, "submit")


def test_no_llm_backends_registered_for_a_batch_judge_without_an_anthropic_secret(db, media_store):
    cfg = ProvidersConfig(secrets={}, judge=JudgeConfig(transport="batch", model="m"))
    assert "llm-sentence" not in build_provider(cfg, db, media_store)._backends


def test_llm_backends_carry_the_judge_price_and_quota_cost(cfg, db, media_store):
    """Spec 3 section 2's cost contract: llm drafting spends on the same
    account and the same currency the judge does, so it is priced from the
    same providers.yaml judge section."""
    api = build_provider(ProvidersConfig(
        secrets=cfg.secrets,
        judge=JudgeConfig(transport="api", model="m", price_per_mtok=(2.0, 10.0))),
        db, media_store)._backends
    cli = build_provider(ProvidersConfig(
        secrets=cfg.secrets, judge=JudgeConfig(transport="cli", model="m")),
        db, media_store)._backends
    for name in ("llm-sentence", "llm-phrase", "llm-entry"):
        assert api[name].price == Price(2.0, 10.0)
        assert api[name].quota_cost_per_call == 0.0
        assert cli[name].price is None
        assert cli[name].quota_cost_per_call == 1.0


# --- build_assessor -------------------------------------------------------

def test_build_assessor_registers_judge_and_mechanical(cfg, db, media_store):
    a = build_assessor(cfg, db, media_store)
    assert set(a._backends) >= {"judge", "mechanical"}
    assert isinstance(a, Assessor)


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


def test_default_budgets_layers_configured_quotas_over_defaults():
    cfg = ProvidersConfig(quotas={"forvo": {"max_asks": 10},
                                  "judge-api": {"max_cost": 5.0}})
    budgets = default_budgets(cfg)
    assert budgets["forvo"].max_asks == 10  # overridden
    assert budgets["learner"].max_asks == 20  # default still present
    assert budgets["judge-api"].max_cost == 5.0


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
    return root


def test_load_syllabus_round_trips_words_and_targets(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    syllabus = load_syllabus(root)
    assert isinstance(syllabus, Syllabus)
    assert {w.id for w in syllabus.words} == {"rice", "near"}
    assert {t.id for t in syllabus.targets} == {"t-rice"}


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
    (root / "data").mkdir()
    # red ranks more frequent (1) than rice (2); Food's 3x emphasis must
    # still bring rice's target ahead of red's.
    (root / "data" / "frequency_th.txt").write_text("แดง\nข้าว\n", encoding="utf-8")

    syllabus = load_syllabus(root)
    ids = [t.id for t in syllabus.order() if isinstance(t, Target)]
    assert ids.index("rice/receptive") < ids.index("red/receptive")


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
    db.add_sentence(text_sha="s1", text="ผมกินข้าว", gloss="I eat rice",  # I eat rice
                    voice="learner_voice", source="llm", origin="draft", licence="n/a",
                    acquired=date(2026, 1, 1))
    syllabus = load_syllabus(root)
    assert any(s.text == "ผมกินข้าว" for s in syllabus.sentences)  # I eat rice


def test_load_syllabus_media_index_reflects_current_best(tmp_path):
    from thai_syllabus.rulebook import PICTURE_FIT_RUBRIC

    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")
    db.append(port="provide", backend="openverse", key="openverse:rice",
             subject="rice", question={"provides": "picture", "params": {}},
             answer={"items": [{"sha": "abc"}]}, cost=0.0)
    # rubric must match load_syllabus's own rubrics_for(rules) (the default
    # PICTURE_FIT_RUBRIC, no rulebook.yaml overlay here) -- _DbMediaIndex now
    # threads current_rubric through current_best (Task 11), so a verdict
    # under a stale/mismatched rubric would not count.
    db.append(port="assess", backend="judge", key="judge:x:abc:picture-for-word",
             subject="rice",
             question={"role": "picture-for-word", "artifact_sha": "abc",
                      "rubric": PICTURE_FIT_RUBRIC},
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
# role "rendition-for-pair" -- attempts._record_rendition's mechanical row);
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
    db.append(port="provide", backend="forvo", key=f"forvo:{subject}",
             subject=subject, question={"provides": "recording", "params": {}},
             answer={"items": [{"sha": sha}]}, cost=0.0)
    db.append(port="assess", backend="judge", key=f"judge:x:{sha}:recording-for-word",
             subject=subject,
             question={"role": "recording-for-word", "artifact_sha": sha, "rubric": None},
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

    # attempts._record_rendition's own row shape.
    db.append(port="assess", backend="mechanical",
             key=f"mech:rendition:v1:{pair.id}:joined", subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": "joined-sha",
                      "rubric": None,
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
    s = sentence("ข้าว")  # rice
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
    # attempts._record_rendition's own row shape -- both members' current-
    # best rendition rows resolve to the SAME speaker, so speakers_of must
    # report it once, not twice.
    db.append(port="assess", backend="mechanical",
             key=f"mech:rendition:v1:{pair.id}:joined", subject=pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": "joined-sha",
                      "rubric": None,
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
    db.append(port="assess", backend="mechanical",
             key=f"mech:rendition:v1:{real_pair.id}:joined", subject=real_pair.id,
             question={"role": "rendition-for-pair", "artifact_sha": "joined-sha",
                      "rubric": None,
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
    syllabus = Syllabus(confusions=(real_confusion, fallback_confusion), media=media)
    gaps = syllabus.gaps()
    assert real_confusion.id not in gaps.missing_renditions
    assert fallback_confusion.id in gaps.missing_renditions


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


# --- load_syllabus: rulebook overlay + build_sourcing ---------------------

def _minimal_deck(tmp_path):
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=(word("slow", "ช้า", "slow"),), targets=(target("slow/receptive", "slow"),),
        graphemes=(), confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(),
        categories=(Category(name="Adjectives", members=frozenset({"slow"})),)))
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


def test_build_sourcing_shares_one_db_handle_with_the_syllabus(tmp_path):
    # load_syllabus, left to open its own SyllabusDb, would give
    # Sourcing.db and syllabus.assessments/media.db two separate
    # connections -- e.g. set_pair_confusions would land on only one.
    # build_sourcing must open db/bundle once and inject them.
    root = _minimal_deck(tmp_path)
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
