"""Loaders/savers for curated/*.yaml (spec 2 section 1): human-owned,
hand-editable data that loads into spec 1's entities.

Saves are temp-file + os.replace (atomic, per the spec's "YAML writes are
temp-file + os.replace" ground rule). Loads validate references (a
grapheme's keyword resolves and contains its symbol, a pair's confusion
and members resolve and satisfy the exact-confusion invariant, a target's
word resolves, a word's classifier resolves) and collect every error
instead of failing on the first one -- `load_curated` on a whole directory
does the same across files, e.g. a target pointing at a nonexistent word.
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .entities import Grapheme, MinimalPair, Pronunciation, SoundConfusion, Syllable, Target, Word
from .ids import ConfusionId, PairId, TargetId, WordId
from .profile import Profile
from .secrets import SecretStore
from .tts import FEMALE_VOICES, MALE_VOICES

_SEVERITIES = {"error", "warn", "info"}


def _is_number(value: Any) -> bool:
    """int/float but not bool (bool is an int subclass in Python)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CuratedValidationError(ValueError):
    """Raised with every collected validation error, not just the first."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _atomic_write_yaml(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def _load_yaml_list(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or []


# --- words -----------------------------------------------------------------

def _syllable_to_dict(s: Syllable) -> dict:
    return {"segments": list(s.segments), "vowel_length": s.vowel_length,
           "tone": s.tone}


def _syllable_from_dict(d: dict) -> Syllable:
    return Syllable(segments=tuple(d["segments"]), vowel_length=d["vowel_length"],
                    tone=d["tone"])


def _pron_to_dict(p: Pronunciation) -> dict:
    return {"syllables": [_syllable_to_dict(s) for s in p.syllables],
           "corroboration": p.corroboration}


def _pron_from_dict(d: dict) -> Pronunciation:
    return Pronunciation(syllables=tuple(_syllable_from_dict(s) for s in d["syllables"]),
                         corroboration=d["corroboration"])


def _word_to_dict(w: Word) -> dict:
    return {"id": w.id, "thai": w.thai, "pron": _pron_to_dict(w.pron),
           "meaning": w.meaning, "classifier": w.classifier}


def _word_from_dict(d: dict) -> Word:
    return Word(id=WordId(d["id"]), thai=d["thai"], pron=_pron_from_dict(d["pron"]),
               meaning=d["meaning"],
               classifier=WordId(d["classifier"]) if d.get("classifier") else None)


def save_words(path: str | Path, words: list[Word]) -> None:
    _atomic_write_yaml(Path(path), [_word_to_dict(w) for w in words])


def load_words(path: str | Path) -> list[Word]:
    rows = _load_yaml_list(Path(path))
    errors: list[str] = []
    words: list[Word] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        try:
            w = _word_from_dict(row)
        except (KeyError, TypeError) as e:
            errors.append(f"words[{i}]: malformed row ({e})")
            continue
        if w.id in seen:
            errors.append(f"words[{i}]: duplicate id {w.id!r}")
            continue
        seen.add(w.id)
        words.append(w)
    if errors:
        raise CuratedValidationError(errors)
    return words


# --- targets -----------------------------------------------------------

def _target_to_dict(t: Target) -> dict:
    return {"id": t.id, "word": t.word, "skill": t.skill,
           "introduction": t.introduction}


def _target_from_dict(d: dict) -> Target:
    return Target(id=TargetId(d["id"]), word=WordId(d["word"]), skill=d["skill"],
                 introduction=d.get("introduction", "picture_card"))


def save_targets(path: str | Path, targets: list[Target]) -> None:
    _atomic_write_yaml(Path(path), [_target_to_dict(t) for t in targets])


def load_targets(path: str | Path, words_by_id: Mapping[str, Word] | None = None
                  ) -> list[Target]:
    rows = _load_yaml_list(Path(path))
    errors: list[str] = []
    targets: list[Target] = []
    for i, row in enumerate(rows):
        try:
            t = _target_from_dict(row)
        except (KeyError, TypeError) as e:
            errors.append(f"targets[{i}]: malformed row ({e})")
            continue
        if words_by_id is not None and t.word not in words_by_id:
            errors.append(f"targets[{i}] ({t.id!r}): word {t.word!r} does not resolve")
            continue
        targets.append(t)
    if errors:
        raise CuratedValidationError(errors)
    return targets


# --- graphemes ---------------------------------------------------------

def _grapheme_to_dict(g: Grapheme) -> dict:
    return {"symbol": g.symbol, "kind": g.kind, "sound": g.sound,
           "consonant_class": g.consonant_class, "keyword": g.keyword,
           "name_word": g.name_word}


def save_graphemes(path: str | Path, graphemes: list[Grapheme]) -> None:
    _atomic_write_yaml(Path(path), [_grapheme_to_dict(g) for g in graphemes])


def load_graphemes(path: str | Path, words_by_id: Mapping[str, Word]) -> list[Grapheme]:
    rows = _load_yaml_list(Path(path))
    errors: list[str] = []
    graphemes: list[Grapheme] = []
    for i, row in enumerate(rows):
        try:
            symbol, kind, sound = row["symbol"], row["kind"], row["sound"]
            consonant_class = row.get("consonant_class")
            keyword_id = row["keyword"]
        except (KeyError, TypeError) as e:
            errors.append(f"graphemes[{i}]: malformed row ({e})")
            continue
        keyword_word = words_by_id.get(keyword_id)
        if keyword_word is None:
            errors.append(f"graphemes[{i}] ({symbol!r}): keyword {keyword_id!r} "
                          f"does not resolve")
            continue
        # name_word (spec 4 section 1) is optional: absent in older/partial
        # curated data, and it carries no containment invariant to enforce
        # (unlike keyword) -- only "if present, must resolve".
        name_word_id = row.get("name_word")
        name_word: Word | None = None
        if name_word_id is not None:
            name_word = words_by_id.get(name_word_id)
            if name_word is None:
                errors.append(f"graphemes[{i}] ({symbol!r}): name_word "
                              f"{name_word_id!r} does not resolve")
                continue
        try:
            g = Grapheme.create(symbol=symbol, kind=kind, sound=sound,
                                consonant_class=consonant_class,
                                keyword_word=keyword_word, name_word=name_word)
        except ValueError as e:
            errors.append(f"graphemes[{i}] ({symbol!r}): {e}")
            continue
        graphemes.append(g)
    if errors:
        raise CuratedValidationError(errors)
    return graphemes


# --- confusions --------------------------------------------------------

def _confusion_to_dict(c: SoundConfusion) -> dict:
    return {"id": c.id, "dimension": c.dimension, "sounds": list(c.sounds)}


def _confusion_from_dict(d: dict) -> SoundConfusion:
    return SoundConfusion(id=ConfusionId(d["id"]), dimension=d["dimension"],
                          sounds=tuple(d["sounds"]))


def save_confusions(path: str | Path, confusions: list[SoundConfusion]) -> None:
    _atomic_write_yaml(Path(path), [_confusion_to_dict(c) for c in confusions])


def load_confusions(path: str | Path) -> list[SoundConfusion]:
    rows = _load_yaml_list(Path(path))
    errors: list[str] = []
    confusions: list[SoundConfusion] = []
    for i, row in enumerate(rows):
        try:
            confusions.append(_confusion_from_dict(row))
        except (KeyError, TypeError) as e:
            errors.append(f"confusions[{i}]: malformed row ({e})")
    if errors:
        raise CuratedValidationError(errors)
    return confusions


# --- pairs -----------------------------------------------------------------

def _pair_to_dict(p: MinimalPair) -> dict:
    return {"id": p.id, "confusion": p.confusion, "members": list(p.members)}


def save_pairs(path: str | Path, pairs: list[MinimalPair]) -> None:
    _atomic_write_yaml(Path(path), [_pair_to_dict(p) for p in pairs])


def load_pairs(path: str | Path, words_by_id: Mapping[str, Word],
               confusions_by_id: Mapping[str, SoundConfusion]) -> list[MinimalPair]:
    rows = _load_yaml_list(Path(path))
    errors: list[str] = []
    pairs: list[MinimalPair] = []
    for i, row in enumerate(rows):
        try:
            pair_id, confusion_id, member_ids = row["id"], row["confusion"], row["members"]
        except (KeyError, TypeError) as e:
            errors.append(f"pairs[{i}]: malformed row ({e})")
            continue
        confusion = confusions_by_id.get(confusion_id)
        if confusion is None:
            errors.append(f"pairs[{i}] ({pair_id!r}): confusion {confusion_id!r} "
                          f"does not resolve")
            continue
        members = [words_by_id.get(m) for m in member_ids]
        if any(m is None for m in members):
            missing = [m for m, w in zip(member_ids, members) if w is None]
            errors.append(f"pairs[{i}] ({pair_id!r}): member(s) {missing!r} "
                          f"do not resolve")
            continue
        try:
            pair = MinimalPair.create(id=PairId(pair_id), confusion=confusion,
                                      members=tuple(members))
        except ValueError as e:
            errors.append(f"pairs[{i}] ({pair_id!r}): {e}")
            continue
        pairs.append(pair)
    if errors:
        raise CuratedValidationError(errors)
    return pairs


# --- profile -------------------------------------------------------------

def save_profile(path: str | Path, profile: Profile) -> None:
    _atomic_write_yaml(Path(path), {"register": profile.register,
                                    "emphasis": dict(profile.emphasis)})


def load_profile(path: str | Path) -> Profile:
    path = Path(path)
    if not path.exists():
        return Profile(register="male_colloquial")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Profile(register=data.get("register", "male_colloquial"),
                   emphasis=dict(data.get("emphasis") or {}))


# --- rulebook config -------------------------------------------------------

@dataclass(frozen=True)
class RulebookConfig:
    """Human-tunable overlay on the code-defined rulebook (spec 1 section
    4's Rule objects): severities, thresholds, judged-rule rubric text, and
    provenance's source-preference order. Not itself the rule registry --
    rulebook.py's RULES list is the code; this is curated data a caller
    applies over it via rulebook.apply_overlay(RULES, config).
    """
    severities: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    rubrics: dict[str, str] = field(default_factory=dict)
    # provenance's preference order (spec 3): earlier sources win ties.
    provenance_prior: tuple[str, ...] = ("commission", "forvo", "tts")


def save_rulebook_config(path: str | Path, config: RulebookConfig) -> None:
    _atomic_write_yaml(Path(path), {
        "severities": dict(config.severities),
        "thresholds": dict(config.thresholds),
        "rubrics": dict(config.rubrics),
        "provenance_prior": list(config.provenance_prior)})


def load_rulebook_config(path: str | Path) -> RulebookConfig:
    path = Path(path)
    if not path.exists():
        return RulebookConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    severities = dict(data.get("severities") or {})
    for rule_id, sev in severities.items():
        if sev not in _SEVERITIES:
            errors.append(f"rulebook.severities[{rule_id!r}]: {sev!r} is not "
                          f"one of {sorted(_SEVERITIES)}")
    thresholds = dict(data.get("thresholds") or {})
    for rule_id, value in thresholds.items():
        if not isinstance(value, (int, float)):
            errors.append(f"rulebook.thresholds[{rule_id!r}]: {value!r} is not numeric")
    rubrics = dict(data.get("rubrics") or {})
    if "provenance_prior" in data:
        raw_prior = data["provenance_prior"]
        if not isinstance(raw_prior, list) or not all(isinstance(x, str) for x in raw_prior):
            errors.append(f"rulebook.provenance_prior: {raw_prior!r} must be a "
                          "list of strings")
            provenance_prior = RulebookConfig().provenance_prior
        else:
            provenance_prior = tuple(raw_prior)
    else:
        provenance_prior = RulebookConfig().provenance_prior
    if errors:
        raise CuratedValidationError(errors)
    return RulebookConfig(severities=severities, thresholds=thresholds, rubrics=rubrics,
                          provenance_prior=provenance_prior)


# --- combined bundle -----------------------------------------------------

@dataclass(frozen=True)
class CuratedBundle:
    words: tuple[Word, ...]
    targets: tuple[Target, ...]
    graphemes: tuple[Grapheme, ...]
    confusions: tuple[SoundConfusion, ...]
    pairs: tuple[MinimalPair, ...]
    profile: Profile
    rulebook: RulebookConfig


def load_curated(root: str | Path) -> CuratedBundle:
    """Load every curated/*.yaml file under `root`, collecting every
    validation error across all of them (not just the first file that
    fails) before raising.
    """
    root = Path(root)
    errors: list[str] = []

    def _collect(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except CuratedValidationError as e:
            errors.extend(e.errors)
            return None

    words = _collect(load_words, root / "words.yaml") or []
    words_by_id = {w.id: w for w in words}
    confusions = _collect(load_confusions, root / "confusions.yaml") or []
    confusions_by_id = {c.id: c for c in confusions}
    targets = _collect(load_targets, root / "targets.yaml", words_by_id) or []
    graphemes = _collect(load_graphemes, root / "graphemes.yaml", words_by_id) or []
    pairs = _collect(load_pairs, root / "pairs.yaml", words_by_id, confusions_by_id) or []
    profile = _collect(load_profile, root / "profile.yaml") or Profile(register="male_colloquial")
    rulebook = _collect(load_rulebook_config, root / "rulebook.yaml") or RulebookConfig()

    # cross-file: a word's classifier must resolve too (targets/graphemes/
    # pairs already validate their own refs against words_by_id above).
    for w in words:
        if w.classifier is not None and w.classifier not in words_by_id:
            errors.append(f"words[{w.id!r}]: classifier {w.classifier!r} "
                          f"does not resolve")

    if errors:
        raise CuratedValidationError(errors)

    return CuratedBundle(words=tuple(words), targets=tuple(targets),
                         graphemes=tuple(graphemes), confusions=tuple(confusions),
                         pairs=tuple(pairs), profile=profile, rulebook=rulebook)


# --- frequency corpus (spec 2 section 3's FrequencyMap) --------------------
#
# Not one of curated/*.yaml (spec 2 section 1 doesn't list a frequency
# file) and not a syllabus.db table either (spec 2 section 2 lists exactly
# four tables). It is project input data that predates and outlives this
# migration -- data/frequency_th.txt, one Thai word per line in rank order,
# a `#`-prefixed header comment block up top. Read-only; nothing ever
# writes it.

@dataclass(frozen=True)
class TextFrequencyMap:
    """FrequencyMap over a flat rank-ordered word list (1-indexed: line 1
    of the data is rank 1, the most frequent word).
    """
    _rank_by_word: Mapping[str, int]

    def rank(self, word_thai: str) -> int | None:
        return self._rank_by_word.get(word_thai)


def load_frequency_map(path: str | Path) -> TextFrequencyMap:
    path = Path(path)
    rank_by_word: dict[str, int] = {}
    if not path.exists():
        return TextFrequencyMap(rank_by_word)
    rank = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rank += 1
        rank_by_word.setdefault(line, rank)
    return TextFrequencyMap(rank_by_word)


# --- rulebook.yaml raw text (spec 3 section 6: Report.rulebook_id) --------
#
# load_rulebook_config above returns the PARSED RulebookConfig; rulebook_id
# (spec 3 section 6) hashes the FILE CONTENTS + the registry's rule ids, so
# it needs the raw text, not the parsed value -- kept as a tiny separate
# reader rather than folded into load_rulebook_config, which has its own
# job (validated config) and no reason to also expose raw bytes.

def rulebook_file_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# --- providers.yaml (spec 3 section 5) --------------------------------------
#
# Per-backend settings: secret references (resolved by SecretStore, ported
# in secrets.py), search_proxy, imgfetch/audiofetch paths, tts voice pools
# (defaulting to tts.py's shipped male/female lists) + cost_per_char, judge
# transport + model + price_per_mtok, image_candidates, batch limits, quotas, k and
# attempt caps. One file; no env vars; no settings in two places (judged-
# rule rubric TEXT stays in rulebook.yaml -- WHAT to ask; this file is HOW
# to reach things).
#
# load_providers_config refuses a file that describes a run the code cannot
# perform: no imgfetch_path/audiofetch_path (pictures and recordings are
# always in scope), an api/batch judge with no price_per_mtok (its spend
# would be silently costed at zero), and an api/batch judge with no
# anthropic secret (nothing left to draft sentences through). The dataclass
# defaults stay permissive -- they are what an ABSENT file yields.

@dataclass(frozen=True)
class JudgeConfig:
    transport: str = "cli"   # "cli" | "api" | "batch"
    model: str = ""
    price_per_mtok: tuple[float, float] | None = None  # (input, output) $/Mtok


@dataclass(frozen=True)
class ProvidersConfig:
    secrets: dict[str, str | None] = field(default_factory=dict)
    search_proxy: str | None = None
    imgfetch_path: str | None = None
    audiofetch_path: str | None = None
    tts_male_voices: tuple[str, ...] = field(default_factory=lambda: tuple(MALE_VOICES))
    tts_female_voices: tuple[str, ...] = field(default_factory=lambda: tuple(FEMALE_VOICES))
    tts_cost_per_char: float = 0.0   # $ per synthesized character (spec 3's cost contract)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    image_candidates: int = 5  # candidate images fetched per target word
    batch: dict[str, Any] = field(default_factory=dict)
    quotas: dict[str, dict[str, Any]] = field(default_factory=dict)
    k: int = 2                 # exhausted()'s "last k provide-attempts" default
    attempt_cap: int = 8       # exhausted()'s per-subject attempt cap default

    def secret_store(self, runner=None) -> SecretStore:
        kwargs: dict[str, Any] = {"specs": self.secrets}
        if runner is not None:
            kwargs["runner"] = runner
        return SecretStore(**kwargs)


def load_providers_config(path: str | Path) -> ProvidersConfig:
    path = Path(path)
    if not path.exists():
        return ProvidersConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []

    secrets_cfg = dict(data.get("secrets") or {})

    tts_cfg = data.get("tts") or {}
    male = tuple(tts_cfg["male_voices"]) if "male_voices" in tts_cfg else tuple(MALE_VOICES)
    female = tuple(tts_cfg["female_voices"]) if "female_voices" in tts_cfg else tuple(FEMALE_VOICES)
    tts_cost_per_char = tts_cfg.get("cost_per_char", 0.0)
    if not _is_number(tts_cost_per_char) or float(tts_cost_per_char) < 0:
        errors.append(f"providers.tts.cost_per_char: {tts_cost_per_char!r} must be "
                      "a non-negative number")
        tts_cost_per_char = 0.0

    judge_cfg = data.get("judge") or {}
    transport = judge_cfg.get("transport", "cli")
    if transport not in ("cli", "api", "batch"):
        errors.append(f"providers.judge.transport: {transport!r} is not one of "
                      "'cli', 'api', 'batch'")
    price_per_mtok = None
    if "price_per_mtok" in judge_cfg:
        price_cfg = judge_cfg["price_per_mtok"]
        if not isinstance(price_cfg, Mapping):
            errors.append(f"providers.judge.price_per_mtok: {price_cfg!r} must be "
                          "a mapping with 'input' and 'output'")
        else:
            input_price = price_cfg.get("input")
            output_price = price_cfg.get("output")
            if not _is_number(input_price) or not _is_number(output_price):
                errors.append(f"providers.judge.price_per_mtok: {price_cfg!r} needs "
                              "numeric 'input' and 'output'")
            else:
                price_per_mtok = (float(input_price), float(output_price))
    if transport in ("api", "batch") and price_per_mtok is None:
        # Spec 3 section 2's cost contract: the api and batch judges spend
        # cash, measured as tokens times this price. Without it every
        # verdict is silently costed at zero and no budget can bind.
        errors.append("providers.judge.price_per_mtok: required for the "
                      f"{transport!r} transport, which spends cash per token")
    judge = JudgeConfig(transport=transport, model=judge_cfg.get("model", ""),
                        price_per_mtok=price_per_mtok)

    # A loaded config must describe a run that can actually happen (fail
    # fast and noisy): both mediafetch paths are required -- pictures and
    # recordings are always in scope, so a run without imgfetch could never
    # fetch a found image and one without audiofetch could never download a
    # Forvo recording; each would look like a run that simply "found
    # nothing". The ProvidersConfig DATACLASS stays permissive (its
    # defaults are what an ABSENT file yields); only this loader refuses.
    imgfetch_path = data.get("imgfetch_path")
    audiofetch_path = data.get("audiofetch_path")
    if not imgfetch_path:
        errors.append("providers.imgfetch_path: required -- pictures are always in "
                      "scope and nothing else can fetch a found image")
    if not audiofetch_path:
        errors.append("providers.audiofetch_path: required -- recordings are always "
                      "in scope and nothing else can download one")

    # Sentence drafting is a single-question ask (wiring._llm_transport):
    # the cli transport IS one; api/batch need the anthropic secret to
    # build one. With neither, wiring registers no llm-* backend at all and
    # every unfilled target stays silently unfilled.
    if transport in ("api", "batch") and "anthropic" not in secrets_cfg:
        errors.append(f"providers.secrets.anthropic: required for the {transport!r} "
                      "judge transport -- sentence drafting needs a single-question "
                      "transport on that account")

    image_candidates = data.get("image_candidates", 5)
    if not isinstance(image_candidates, int) or image_candidates < 1:
        errors.append(f"providers.image_candidates: {image_candidates!r} must be "
                      "a positive integer")

    k = data.get("k", 2)
    attempt_cap = data.get("attempt_cap", 8)
    if not isinstance(k, int) or k < 1:
        errors.append(f"providers.k: {k!r} must be a positive integer")
    if not isinstance(attempt_cap, int) or attempt_cap < 1:
        errors.append(f"providers.attempt_cap: {attempt_cap!r} must be a positive integer")

    if errors:
        raise CuratedValidationError(errors)

    return ProvidersConfig(
        secrets=secrets_cfg, search_proxy=data.get("search_proxy"),
        imgfetch_path=imgfetch_path,
        audiofetch_path=audiofetch_path, tts_male_voices=male,
        tts_female_voices=female, tts_cost_per_char=float(tts_cost_per_char),
        judge=judge, image_candidates=image_candidates,
        batch=dict(data.get("batch") or {}), quotas=dict(data.get("quotas") or {}),
        k=k, attempt_cap=attempt_cap)


def save_providers_config(path: str | Path, config: ProvidersConfig) -> None:
    judge: dict[str, Any] = {"transport": config.judge.transport, "model": config.judge.model}
    if config.judge.price_per_mtok is not None:
        input_price, output_price = config.judge.price_per_mtok
        judge["price_per_mtok"] = {"input": input_price, "output": output_price}
    _atomic_write_yaml(Path(path), {
        "secrets": dict(config.secrets),
        "search_proxy": config.search_proxy,
        "imgfetch_path": config.imgfetch_path,
        "audiofetch_path": config.audiofetch_path,
        "tts": {"male_voices": list(config.tts_male_voices),
               "female_voices": list(config.tts_female_voices),
               "cost_per_char": config.tts_cost_per_char},
        "judge": judge,
        "image_candidates": config.image_candidates,
        "batch": dict(config.batch),
        "quotas": dict(config.quotas),
        "k": config.k,
        "attempt_cap": config.attempt_cap,
    })


def save_curated(root: str | Path, bundle: CuratedBundle) -> None:
    root = Path(root)
    save_words(root / "words.yaml", list(bundle.words))
    save_targets(root / "targets.yaml", list(bundle.targets))
    save_graphemes(root / "graphemes.yaml", list(bundle.graphemes))
    save_confusions(root / "confusions.yaml", list(bundle.confusions))
    save_pairs(root / "pairs.yaml", list(bundle.pairs))
    save_profile(root / "profile.yaml", bundle.profile)
    save_rulebook_config(root / "rulebook.yaml", bundle.rulebook)
