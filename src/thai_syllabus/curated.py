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

_SEVERITIES = {"error", "warn", "info"}


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
           "consonant_class": g.consonant_class, "keyword": g.keyword}


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
        try:
            g = Grapheme.create(symbol=symbol, kind=kind, sound=sound,
                                consonant_class=consonant_class,
                                keyword_word=keyword_word)
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
    4's Rule objects): severities, thresholds, judged-rule rubric text.
    Not itself the rule registry -- rulebook.py's RULES list is the code;
    this is curated data a caller may use to override it (wiring that
    overlay into the registry is out of this deliverable's scope).
    """
    severities: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    rubrics: dict[str, str] = field(default_factory=dict)


def save_rulebook_config(path: str | Path, config: RulebookConfig) -> None:
    _atomic_write_yaml(Path(path), {
        "severities": dict(config.severities),
        "thresholds": dict(config.thresholds),
        "rubrics": dict(config.rubrics)})


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
    if errors:
        raise CuratedValidationError(errors)
    return RulebookConfig(severities=severities, thresholds=thresholds, rubrics=rubrics)


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


def save_curated(root: str | Path, bundle: CuratedBundle) -> None:
    root = Path(root)
    save_words(root / "words.yaml", list(bundle.words))
    save_targets(root / "targets.yaml", list(bundle.targets))
    save_graphemes(root / "graphemes.yaml", list(bundle.graphemes))
    save_confusions(root / "confusions.yaml", list(bundle.confusions))
    save_pairs(root / "pairs.yaml", list(bundle.pairs))
    save_profile(root / "profile.yaml", bundle.profile)
    save_rulebook_config(root / "rulebook.yaml", bundle.rulebook)
