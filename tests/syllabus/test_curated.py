"""Tests for curated.py (spec 2 section 1): loaders/savers for
curated/*.yaml, round-tripped through spec 1's entities. Atomic saves
(temp+os.replace); loading collects every validation error instead of
failing on the first one.
"""
import textwrap

import pytest
import yaml

from thai_syllabus import curated
from thai_syllabus.entities import Category, Grapheme, MinimalPair, SoundConfusion, Target, Word
from thai_syllabus.ids import CategoryName, ConfusionId, PairId, TargetId, WordId
from thai_syllabus.profile import Profile

from .builders import word


def write_words_yaml(path, rows: list[dict]) -> None:
    """Writes raw word rows (dicts), bypassing save_words -- for tests
    that need to control exactly which keys a row carries.
    """
    path.write_text(yaml.safe_dump(rows, allow_unicode=True), encoding="utf-8")


# --- words -------------------------------------------------------------

def test_words_round_trip(tmp_path):
    path = tmp_path / "words.yaml"
    curated.save_words(path, [
        (_word("rice", "ข้าว", "cooked rice", classifier=None), "Food"),
        (_word("plate", "จาน", "plate", classifier=None), "Home"),
    ])
    loaded = curated.load_words(path)
    assert {w.id for w, _ in loaded} == {"rice", "plate"}
    rice, rice_category = next((w, c) for w, c in loaded if w.id == "rice")
    assert rice.thai == "ข้าว"  # rice
    assert rice.meaning == "cooked rice"
    assert rice.pron.syllables[0].tone in ("mid", "low", "falling", "high", "rising")
    assert rice_category == "Food"


def _word(id_, thai, meaning, classifier=None, tone="falling", corroboration="engines_agree"):
    from thai_syllabus.entities import Pronunciation, Syllable
    syl = Syllable(segments=("k", "aː", "w"), vowel_length="long", tone=tone)
    return Word(id=WordId(id_), thai=thai,
               pron=Pronunciation(syllables=(syl,), corroboration=corroboration),
               meaning=meaning, classifier=WordId(classifier) if classifier else None)


def test_words_save_is_atomic_temp_then_replace(tmp_path, monkeypatch):
    path = tmp_path / "words.yaml"
    curated.save_words(path, [(_word("rice", "ข้าว", "cooked rice"), "Food")])
    # no leftover temp files
    assert list(tmp_path.glob("*.tmp*")) == []
    assert path.exists()


def test_load_words_a_row_without_category_loads_with_none(tmp_path):
    """A word need not name a category (spec 1: closure words are in
    none) -- load_words only rejects an unknown name, never an absent one.
    """
    rows = [{"id": "rice", "thai": "ข้าว", "pron": _pron_dict(), "meaning": "rice",
            "category": "Food"},                                       # ข้าว = rice
           {"id": "red", "thai": "แดง", "pron": _pron_dict(), "meaning": "red"}]  # แดง = red
    write_words_yaml(tmp_path / "words.yaml", rows)
    loaded = curated.load_words(tmp_path / "words.yaml")
    by_id = {w.id: c for w, c in loaded}
    assert by_id["rice"] == "Food"
    assert by_id["red"] is None


def test_load_words_refuses_an_unknown_category_name(tmp_path):
    rows = [{"id": "red", "thai": "แดง", "pron": _pron_dict(), "meaning": "red",
            "category": "Kitchen"}]  # แดง = red; "Kitchen" is not one of the 27 FF categories
    write_words_yaml(tmp_path / "words.yaml", rows)
    with pytest.raises(curated.CuratedValidationError, match="red"):
        curated.load_words(tmp_path / "words.yaml")
    with pytest.raises(curated.CuratedValidationError, match="Kitchen"):
        curated.load_words(tmp_path / "words.yaml")


def _pron_dict():
    return {"syllables": [{"segments": ["k", "a", "w"], "vowel_length": "long",
                          "tone": "falling"}], "corroboration": "engines_agree"}


def test_build_categories_groups_members_by_name():
    cats = curated.build_categories([
        (word("rice", "ข้าว"), "Food"), (word("fish", "ปลา"), "Food"),
        (word("red", "แดง"), "Colors")])  # rice, fish, red
    by_name = {c.name: c.members for c in cats}
    assert by_name == {"Food": frozenset({"rice", "fish"}), "Colors": frozenset({"red"})}


def test_category_names_are_loaded_from_the_repo_data_file():
    assert len(curated.CATEGORY_NAMES) == 27
    assert "Food" in curated.CATEGORY_NAMES


# --- targets -----------------------------------------------------------

def test_targets_round_trip(tmp_path):
    path = tmp_path / "targets.yaml"
    targets = [Target(id=TargetId("rice/receptive"), word=WordId("rice"),
                      skill="receptive", introduction="picture_card")]
    curated.save_targets(path, targets)
    loaded = curated.load_targets(path)
    assert loaded == targets


# --- graphemes -----------------------------------------------------------

def test_graphemes_round_trip_with_keyword_resolution(tmp_path):
    words = {"chicken": _word("chicken", "ไก่", "chicken", tone="low")}
    path = tmp_path / "graphemes.yaml"
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                        consonant_class="mid", keyword_word=words["chicken"])
    curated.save_graphemes(path, [g])
    loaded = curated.load_graphemes(path, words_by_id=words)
    assert loaded == [g]


def test_graphemes_round_trip_with_name_word_resolution(tmp_path):
    words = {
        "chicken": _word("chicken", "ไก่", "chicken", tone="low"),
        "letter-name:ko": _word("letter-name:ko", "กอ", "name of the letter ก",
                               tone="mid"),
    }
    path = tmp_path / "graphemes.yaml"
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                        consonant_class="mid", keyword_word=words["chicken"],
                        name_word=words["letter-name:ko"])
    curated.save_graphemes(path, [g])
    loaded = curated.load_graphemes(path, words_by_id=words)
    assert loaded == [g]
    assert loaded[0].name_word == "letter-name:ko"


def test_graphemes_round_trip_without_name_word_stays_none(tmp_path):
    words = {"chicken": _word("chicken", "ไก่", "chicken", tone="low")}
    path = tmp_path / "graphemes.yaml"
    g = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                        consonant_class="mid", keyword_word=words["chicken"])
    curated.save_graphemes(path, [g])
    loaded = curated.load_graphemes(path, words_by_id=words)
    assert loaded[0].name_word is None


def test_graphemes_load_validation_reports_an_unresolved_name_word(tmp_path):
    path = tmp_path / "graphemes.yaml"
    path.write_text(yaml.safe_dump([
        {"symbol": "ก", "kind": "consonant", "sound": "k",
         "consonant_class": "mid", "keyword": "chicken", "name_word": "missing-name"},
    ], allow_unicode=True))
    words = {"chicken": _word("chicken", "ไก่", "chicken", tone="low")}
    with pytest.raises(curated.CuratedValidationError) as exc:
        curated.load_graphemes(path, words_by_id=words)
    assert len(exc.value.errors) == 1


def test_graphemes_load_validation_collects_all_errors(tmp_path):
    path = tmp_path / "graphemes.yaml"
    path.write_text(yaml.safe_dump([
        {"symbol": "ก", "kind": "consonant", "sound": "k",
         "consonant_class": "mid", "keyword": "missing-word"},
        {"symbol": "ข", "kind": "consonant", "sound": "kh",
         "consonant_class": "high", "keyword": "chicken"},  # chicken thai has no ข
    ], allow_unicode=True))
    words = {"chicken": _word("chicken", "ไก่", "chicken", tone="low")}
    with pytest.raises(curated.CuratedValidationError) as exc:
        curated.load_graphemes(path, words_by_id=words)
    assert len(exc.value.errors) == 2


# --- confusions ----------------------------------------------------------

def test_confusions_round_trip(tmp_path):
    path = tmp_path / "confusions.yaml"
    confusions = [SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                                 sounds=("mid", "low"))]
    curated.save_confusions(path, confusions)
    assert curated.load_confusions(path) == confusions


# --- pairs -----------------------------------------------------------------

def test_pairs_round_trip_with_resolution(tmp_path):
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    mid_word = _word("near", "ใกล้", "near", tone="mid")
    low_word = _word("far", "ไกล", "far", tone="low")
    words = {"near": mid_word, "far": low_word}
    confusions = {"tone:mid-low": confusion}
    pair = MinimalPair.create(id=PairId("tone:mid-low/kai"), confusion=confusion,
                              members=(mid_word, low_word))
    path = tmp_path / "pairs.yaml"
    curated.save_pairs(path, [pair])
    loaded = curated.load_pairs(path, words_by_id=words, confusions_by_id=confusions)
    assert loaded == [pair]


def test_pairs_load_validation_collects_all_errors(tmp_path):
    path = tmp_path / "pairs.yaml"
    path.write_text(yaml.safe_dump([
        {"id": "bad-1", "confusion": "missing-confusion", "members": ["a", "b"]},
        {"id": "bad-2", "confusion": "tone:mid-low", "members": ["missing-a", "missing-b"]},
    ], allow_unicode=True))
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    with pytest.raises(curated.CuratedValidationError) as exc:
        curated.load_pairs(path, words_by_id={}, confusions_by_id={"tone:mid-low": confusion})
    assert len(exc.value.errors) == 2


# --- profile -----------------------------------------------------------

def test_profile_round_trip(tmp_path):
    path = tmp_path / "profile.yaml"
    profile = Profile(register="male_colloquial", emphasis={"Animals": 1.5})
    curated.save_profile(path, profile)
    assert curated.load_profile(path) == profile


# --- rulebook config -----------------------------------------------------

def test_rulebook_config_round_trip(tmp_path):
    path = tmp_path / "rulebook.yaml"
    cfg = curated.RulebookConfig(
        severities={"pair/exact-confusion": "error"},
        thresholds={"coverage/categories": 0.1},
        rubrics={"sentence/register-natural": "Is this natural?"})
    curated.save_rulebook_config(path, cfg)
    assert curated.load_rulebook_config(path) == cfg


def test_rulebook_config_validation_rejects_bad_severity(tmp_path):
    path = tmp_path / "rulebook.yaml"
    path.write_text(yaml.safe_dump({
        "severities": {"pair/exact-confusion": "not-a-severity"},
        "thresholds": {}, "rubrics": {},
    }))
    with pytest.raises(curated.CuratedValidationError):
        curated.load_rulebook_config(path)


def test_rulebook_config_defaults_round_trip(tmp_path):
    path = tmp_path / "rulebook.yaml"
    curated.save_rulebook_config(path, curated.RulebookConfig())
    assert curated.load_rulebook_config(path) == curated.RulebookConfig()


def test_rulebook_provenance_prior_round_trip(tmp_path):
    path = tmp_path / "rulebook.yaml"
    path.write_text("provenance_prior: [commission, forvo]\n", encoding="utf-8")
    cfg = curated.load_rulebook_config(path)
    assert cfg.provenance_prior == ("commission", "forvo")
    curated.save_rulebook_config(path, cfg)
    assert curated.load_rulebook_config(path).provenance_prior == ("commission", "forvo")
    assert curated.RulebookConfig().provenance_prior == ("commission", "forvo", "tts")


def test_rulebook_provenance_prior_explicit_empty_list_stays_empty(tmp_path):
    path = tmp_path / "rulebook.yaml"
    path.write_text("provenance_prior: []\n", encoding="utf-8")
    assert curated.load_rulebook_config(path).provenance_prior == ()


def test_rulebook_provenance_prior_rejects_a_bare_string(tmp_path):
    path = tmp_path / "rulebook.yaml"
    path.write_text("provenance_prior: commission\n", encoding="utf-8")
    with pytest.raises(curated.CuratedValidationError, match="provenance_prior"):
        curated.load_rulebook_config(path)


def test_rulebook_provenance_prior_rejects_a_non_string_element(tmp_path):
    path = tmp_path / "rulebook.yaml"
    path.write_text("provenance_prior: [commission, 3]\n", encoding="utf-8")
    with pytest.raises(curated.CuratedValidationError, match="provenance_prior"):
        curated.load_rulebook_config(path)


# --- frequency map -------------------------------------------------------

def test_frequency_map_ranks_by_line_position(tmp_path):
    path = tmp_path / "frequency_th.txt"
    path.write_text("# header\nไม่\nใช่\nแต่\n", encoding="utf-8")
    fm = curated.load_frequency_map(path)
    assert fm.rank("ไม่") == 1
    assert fm.rank("ใช่") == 2
    assert fm.rank("แต่") == 3
    assert fm.rank("ไม่มีทาง") is None


def test_frequency_map_satisfies_the_protocol(tmp_path):
    from thai_syllabus.ports import FrequencyMap
    path = tmp_path / "frequency_th.txt"
    path.write_text("ไม่\n", encoding="utf-8")
    assert isinstance(curated.load_frequency_map(path), FrequencyMap)


def test_frequency_map_missing_file_is_empty(tmp_path):
    fm = curated.load_frequency_map(tmp_path / "does-not-exist.txt")
    assert fm.rank("ไม่") is None


# --- combined load/save --------------------------------------------------

def test_load_curated_bundle_from_a_directory(tmp_path):
    mid_word = _word("near", "ใกล้", "near", tone="mid")
    low_word = _word("far", "ไกล", "far", tone="low")
    confusion = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                               sounds=("mid", "low"))
    pair = MinimalPair.create(id=PairId("tone:mid-low/kai"), confusion=confusion,
                              members=(mid_word, low_word))
    target = Target(id=TargetId("near/receptive"), word=WordId("near"), skill="receptive")
    grapheme = Grapheme.create(symbol="ก", kind="consonant", sound="k",
                               consonant_class="mid", keyword_word=low_word)
    profile = Profile(register="male_colloquial")

    curated.save_words(tmp_path / "words.yaml", [(mid_word, "Adjectives"), (low_word, "Adjectives")])
    curated.save_targets(tmp_path / "targets.yaml", [target])
    curated.save_graphemes(tmp_path / "graphemes.yaml", [grapheme])
    curated.save_confusions(tmp_path / "confusions.yaml", [confusion])
    curated.save_pairs(tmp_path / "pairs.yaml", [pair])
    curated.save_profile(tmp_path / "profile.yaml", profile)
    curated.save_rulebook_config(tmp_path / "rulebook.yaml", curated.RulebookConfig())

    bundle = curated.load_curated(tmp_path)
    assert {w.id for w in bundle.words} == {"near", "far"}
    assert bundle.targets == (target,)
    assert bundle.pairs == (pair,)
    assert bundle.graphemes == (grapheme,)
    assert bundle.confusions == (confusion,)
    assert bundle.profile == profile
    assert {c.name for c in bundle.categories} == {"Adjectives"}


# --- rulebook.yaml raw text (spec 3 section 6) -----------------------------

def test_rulebook_file_text_returns_the_raw_file_contents(tmp_path):
    path = tmp_path / "rulebook.yaml"
    path.write_text("severities:\n  pair/exact-confusion: warn\n", encoding="utf-8")
    assert curated.rulebook_file_text(path) == "severities:\n  pair/exact-confusion: warn\n"


def test_rulebook_file_text_is_empty_when_the_file_is_absent(tmp_path):
    assert curated.rulebook_file_text(tmp_path / "does-not-exist.yaml") == ""


# --- providers.yaml (spec 3 section 5) --------------------------------------
#
# A LOADED providers.yaml must describe a run that can actually happen
# (fail fast and noisy): both mediafetch paths are required -- pictures and
# recordings are always in scope -- and there must be some single-question
# transport for sentence drafting. An ABSENT providers.yaml refuses, naming
# the path (it is a store, spec 2 section 1). The ProvidersConfig dataclass
# itself stays permissive -- a bare ProvidersConfig() is still
# constructible; only the loader refuses.

def _providers(**overrides) -> dict:
    """The minimum providers.yaml the loader accepts, plus overrides --
    so a test about one field doesn't trip over the others."""
    base = {"imgfetch_path": "/opt/bin/imgfetch",
            "audiofetch_path": "/opt/bin/audiofetch",
            "judge": {"transport": "cli", "model": "m"}}
    base.update(overrides)
    return base


def write_providers(tmp_path, **overrides) -> None:
    """Writes the minimum acceptable providers.yaml, plus overrides, to
    tmp_path / "providers.yaml"."""
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(_providers(**overrides)), encoding="utf-8")


# --- missing curated files refuse, naming the path -------------------------

@pytest.mark.parametrize("loader,name", [
    (curated.load_providers_config, "providers.yaml"),
    (curated.load_profile, "profile.yaml"),
    (curated.load_rulebook_config, "rulebook.yaml"),
])
def test_missing_curated_file_refuses(tmp_path, loader, name):
    with pytest.raises(curated.CuratedValidationError, match=name):
        loader(tmp_path / name)


def test_empty_voice_pool_refuses(tmp_path):
    write_providers(tmp_path, tts={"male_voices": [],
                                   "female_voices": ["th-TH-Chirp3-HD-Aoede"]})
    with pytest.raises(curated.CuratedValidationError, match="male_voices"):
        curated.load_providers_config(tmp_path / "providers.yaml")


def test_empty_female_voice_pool_refuses(tmp_path):
    write_providers(tmp_path, tts={"male_voices": ["th-TH-Chirp3-HD-Puck"],
                                   "female_voices": []})
    with pytest.raises(curated.CuratedValidationError, match="female_voices"):
        curated.load_providers_config(tmp_path / "providers.yaml")


def test_providers_config_round_trip(tmp_path):
    path = tmp_path / "providers.yaml"
    config = curated.ProvidersConfig(
        secrets={"forvo": "op://Shared/Forvo/API Key", "google_tts": "~/.secrets/tts",
                 "anthropic": "op://Shared/Anthropic/API Key"},
        search_proxy="https://proxy.example", imgfetch_path="/usr/bin/curl",
        audiofetch_path="/usr/bin/wget",
        tts_male_voices=("v1",), tts_female_voices=("v2",),
        judge=curated.JudgeConfig(transport="batch", model="claude-opus-5",
                                  price_per_mtok=(2.0, 10.0)),
        image_candidates=7,
        batch={"max_requests": 1000}, quotas={"forvo": {"max_asks": 450}},
        k=3, attempt_cap=10)
    curated.save_providers_config(path, config)
    loaded = curated.load_providers_config(path)
    assert loaded == config


def test_providers_judge_price_and_image_candidates_round_trip(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(textwrap.dedent("""
        imgfetch_path: /opt/bin/imgfetch
        audiofetch_path: /opt/bin/audiofetch
        secrets: {anthropic: op://Shared/Anthropic/API Key}
        judge: {transport: batch, model: claude-sonnet-5, price_per_mtok: {input: 2.0, output: 10.0}}
        image_candidates: 3
    """), encoding="utf-8")
    cfg = curated.load_providers_config(path)
    assert cfg.judge.price_per_mtok == (2.0, 10.0) and cfg.image_candidates == 3
    curated.save_providers_config(path, cfg)
    assert curated.load_providers_config(path) == cfg


def test_providers_defaults_when_absent():
    cfg = curated.ProvidersConfig()
    assert cfg.image_candidates == 5 and cfg.judge.price_per_mtok is None


def test_providers_config_defaults_round_trip(tmp_path):
    """Everything except the two required paths left at its default still
    round-trips exactly (the loader refuses a file with no fetch paths, so
    the minimum acceptable config is the default plus those)."""
    path = tmp_path / "providers.yaml"
    minimal = curated.ProvidersConfig(imgfetch_path="/opt/bin/imgfetch",
                                      audiofetch_path="/opt/bin/audiofetch")
    curated.save_providers_config(path, minimal)
    assert curated.load_providers_config(path) == minimal


def test_providers_judge_price_rejects_a_missing_output(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({
        "judge": {"transport": "cli", "price_per_mtok": {"input": 2.0}}}))
    with pytest.raises(curated.CuratedValidationError, match="price_per_mtok"):
        curated.load_providers_config(path)


def test_providers_judge_price_rejects_non_numeric_values(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({
        "judge": {"transport": "cli",
                  "price_per_mtok": {"input": "cheap", "output": 10.0}}}))
    with pytest.raises(curated.CuratedValidationError, match="price_per_mtok"):
        curated.load_providers_config(path)


def test_providers_judge_price_rejects_a_scalar(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({"judge": {"transport": "cli", "price_per_mtok": 2.0}}))
    with pytest.raises(curated.CuratedValidationError, match="price_per_mtok"):
        curated.load_providers_config(path)


def test_providers_judge_price_rejects_a_list(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({
        "judge": {"transport": "cli", "price_per_mtok": [2.0, 10.0]}}))
    with pytest.raises(curated.CuratedValidationError, match="price_per_mtok"):
        curated.load_providers_config(path)


def test_providers_judge_price_rejects_bools(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({
        "judge": {"transport": "cli",
                  "price_per_mtok": {"input": True, "output": 10.0}}}))
    with pytest.raises(curated.CuratedValidationError, match="price_per_mtok"):
        curated.load_providers_config(path)


def test_providers_image_candidates_rejects_zero(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({"image_candidates": 0}))
    with pytest.raises(curated.CuratedValidationError, match="image_candidates"):
        curated.load_providers_config(path)


def test_providers_image_candidates_rejects_a_string(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({"image_candidates": "five"}))
    with pytest.raises(curated.CuratedValidationError, match="image_candidates"):
        curated.load_providers_config(path)


def test_providers_config_secrets_resolve_via_secret_store(tmp_path):
    key_file = tmp_path / "forvo.key"
    key_file.write_text("s3cret\n", encoding="utf-8")
    key_file.chmod(0o600)
    path = tmp_path / "providers.yaml"
    curated.save_providers_config(path, curated.ProvidersConfig(
        secrets={"forvo": str(key_file)}, imgfetch_path="/opt/bin/imgfetch",
        audiofetch_path="/opt/bin/audiofetch"))
    config = curated.load_providers_config(path)
    store = config.secret_store()
    assert store.get("forvo") == "s3cret"


def test_providers_config_rejects_an_unknown_judge_transport(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({"judge": {"transport": "carrier-pigeon"}}))
    with pytest.raises(curated.CuratedValidationError):
        curated.load_providers_config(path)


def test_load_curated_bundle_collects_cross_file_errors(tmp_path):
    curated.save_words(tmp_path / "words.yaml", [(_word("near", "ใกล้", "near"), "Adjectives")])
    curated.save_targets(tmp_path / "targets.yaml", [
        Target(id=TargetId("bad/receptive"), word=WordId("does-not-exist"),
              skill="receptive")])
    curated.save_graphemes(tmp_path / "graphemes.yaml", [])
    curated.save_confusions(tmp_path / "confusions.yaml", [])
    curated.save_pairs(tmp_path / "pairs.yaml", [])
    curated.save_profile(tmp_path / "profile.yaml", Profile(register="male_colloquial"))
    curated.save_rulebook_config(tmp_path / "rulebook.yaml", curated.RulebookConfig())

    with pytest.raises(curated.CuratedValidationError) as exc:
        curated.load_curated(tmp_path)
    assert any("does-not-exist" in e for e in exc.value.errors)


def test_load_curated_refuses_a_targeted_word_without_a_category(tmp_path):
    curated.save_words(tmp_path / "words.yaml", [(_word("near", "ใกล้", "near"), None)])
    curated.save_targets(tmp_path / "targets.yaml", [
        Target(id=TargetId("near/receptive"), word=WordId("near"), skill="receptive")])
    curated.save_graphemes(tmp_path / "graphemes.yaml", [])
    curated.save_confusions(tmp_path / "confusions.yaml", [])
    curated.save_pairs(tmp_path / "pairs.yaml", [])
    curated.save_profile(tmp_path / "profile.yaml", Profile(register="male_colloquial"))
    curated.save_rulebook_config(tmp_path / "rulebook.yaml", curated.RulebookConfig())

    with pytest.raises(curated.CuratedValidationError, match="near"):
        curated.load_curated(tmp_path)


def test_load_curated_allows_an_untargeted_word_with_no_category(tmp_path):
    """A closure word (no Target) needs no category (spec 1)."""
    curated.save_words(tmp_path / "words.yaml", [(_word("near", "ใกล้", "near"), None)])
    curated.save_targets(tmp_path / "targets.yaml", [])
    curated.save_graphemes(tmp_path / "graphemes.yaml", [])
    curated.save_confusions(tmp_path / "confusions.yaml", [])
    curated.save_pairs(tmp_path / "pairs.yaml", [])
    curated.save_profile(tmp_path / "profile.yaml", Profile(register="male_colloquial"))
    curated.save_rulebook_config(tmp_path / "rulebook.yaml", curated.RulebookConfig())

    bundle = curated.load_curated(tmp_path)
    assert bundle.categories == ()


def test_save_curated_refuses_a_targeted_word_without_a_category(tmp_path):
    """Same condition load_curated enforces on read, checked on write too:
    a bundle that would fail to load must not be written."""
    bundle = curated.CuratedBundle(
        words=(_word("near", "ใกล้", "near"),),
        targets=(Target(id=TargetId("near/receptive"), word=WordId("near"),
                        skill="receptive"),),
        graphemes=(), confusions=(), pairs=(),
        profile=Profile(register="male_colloquial"), rulebook=curated.RulebookConfig())
    with pytest.raises(curated.CuratedValidationError, match="near"):
        curated.save_curated(tmp_path, bundle)


def test_save_curated_allows_an_untargeted_word_with_no_category(tmp_path):
    bundle = curated.CuratedBundle(
        words=(_word("near", "ใกล้", "near"),), targets=(),
        graphemes=(), confusions=(), pairs=(),
        profile=Profile(register="male_colloquial"), rulebook=curated.RulebookConfig())
    curated.save_curated(tmp_path, bundle)
    assert curated.load_curated(tmp_path).categories == ()


def test_providers_judge_api_transport_requires_a_price(tmp_path):
    """Spec 3 section 2's cost contract: an api/batch judge spends cash, so
    it cannot be configured without the price that measures it -- a missing
    price silently costed every verdict at zero."""
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({"judge": {"transport": "api", "model": "m"}}))
    with pytest.raises(curated.CuratedValidationError, match="judge.price_per_mtok"):
        curated.load_providers_config(path)


def test_providers_judge_batch_transport_requires_a_price(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump({"judge": {"transport": "batch", "model": "m"}}))
    with pytest.raises(curated.CuratedValidationError, match="judge.price_per_mtok"):
        curated.load_providers_config(path)


def test_providers_judge_cli_transport_needs_no_price(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(_providers(judge={"transport": "cli", "model": "m"})))
    assert curated.load_providers_config(path).judge.price_per_mtok is None


def test_providers_tts_cost_per_char_round_trips(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(textwrap.dedent("""
        imgfetch_path: /opt/bin/imgfetch
        audiofetch_path: /opt/bin/audiofetch
        tts: {male_voices: [m1], female_voices: [f1], cost_per_char: 1.6e-05}
    """), encoding="utf-8")
    cfg = curated.load_providers_config(path)
    assert cfg.tts_cost_per_char == pytest.approx(1.6e-05)
    curated.save_providers_config(path, cfg)
    assert curated.load_providers_config(path) == cfg


def test_providers_tts_cost_per_char_rejects_a_non_number(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(_providers(tts={"cost_per_char": "cheap"})))
    with pytest.raises(curated.CuratedValidationError, match="cost_per_char"):
        curated.load_providers_config(path)


# --- the loader refuses what the run cannot do -----------------------------

def test_providers_requires_an_imgfetch_path(tmp_path):
    """Pictures are always in scope: without imgfetch_path every image the
    search backends find is unfetchable, and the run would quietly source
    no pictures at all."""
    path = tmp_path / "providers.yaml"
    cfg = _providers()
    del cfg["imgfetch_path"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(curated.CuratedValidationError, match="imgfetch_path"):
        curated.load_providers_config(path)


def test_providers_requires_an_audiofetch_path(tmp_path):
    """Recordings are always in scope: without audiofetch_path no Forvo
    recording can ever be downloaded."""
    path = tmp_path / "providers.yaml"
    cfg = _providers()
    del cfg["audiofetch_path"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(curated.CuratedValidationError, match="audiofetch_path"):
        curated.load_providers_config(path)


def test_providers_requires_a_single_question_transport_for_drafting(tmp_path):
    """Sentence drafting is a single-question ask. Under an api/batch judge
    that means the anthropic secret -- without it wiring registers no llm-*
    backend and every target stays silently unfilled."""
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(_providers(
        judge={"transport": "batch", "model": "m",
               "price_per_mtok": {"input": 2.0, "output": 10.0}})))
    with pytest.raises(curated.CuratedValidationError, match="secrets.anthropic"):
        curated.load_providers_config(path)


def test_providers_api_judge_with_an_anthropic_secret_is_accepted(tmp_path):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(_providers(
        secrets={"anthropic": "op://Shared/Anthropic/key"},
        judge={"transport": "api", "model": "m",
               "price_per_mtok": {"input": 2.0, "output": 10.0}})))
    assert curated.load_providers_config(path).judge.transport == "api"


def test_providers_cli_judge_needs_no_anthropic_secret(tmp_path):
    """The cli transport IS the single-question transport -- it shells out
    to the Claude CLI's own session, no api key involved."""
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(_providers()))
    assert curated.load_providers_config(path).judge.transport == "cli"
