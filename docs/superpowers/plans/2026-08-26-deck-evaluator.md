# Thai Deck Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the evaluator that scores a Fluent Forever–style Thai deck (structured YAML source) with findings + dimension scores, gating a future generation pipeline.

**Architecture:** Staged pipeline (schema → mechanical → linguistic → judge) over a rule registry. Deterministic rules are pure functions on a pydantic deck model; linguistic rules go through G2P/tokenizer ports (pythainlp adapters, fakes in tests); LLM judge is a port with Cli/Api/Fake backends and an SQLite verdict cache.

**Tech Stack:** Python 3.12, uv, pydantic v2, PyYAML, click, pytest, pythainlp (+tltk), anthropic SDK, Claude Code headless (`claude -p`).

**Spec:** `docs/superpowers/specs/2026-08-26-deck-evaluator-design.md`

## Global Constraints

- Package name `thai_deck_eval`, console script `thai-deck-eval`, src layout.
- Rule IDs are namespaced strings: `mech/…`, `lang/…`, `meth/…`, `judge/…`.
- Severities: `error` gates (exit 1), `warn` deducts, `info` reports. Dimensions: `integrity`, `language`, `method`, `content`.
- No pythainlp import at module import time anywhere except `lang/pythainlp_adapter.py` (it is heavyweight); tests use fakes; real-adapter tests are marked `integration`, live-LLM tests marked `live`. Default pytest run excludes both.
- Fixture decks are built programmatically by `tests/helpers.py` (deviation from spec's fixture directories: a builder makes mutation fixtures one-line tweaks and avoids binary media in git; media are stub files written at test time).
- **Commits use commit-gate batch approval.** All task commit messages below are pre-approved via one `approve -F /tmp/cg-batch` before execution. Commit with `git commit -F <msgfile>` using the exact listed message. If a message must change: STOP, get fresh approval.
- Authored IPA format (note `ipa` fields, fake G2P data): segments + length mark `ː` + trailing Chao tone letters, e.g. `kʰaːw˥˩`. Tone letters: mid `˧`, low `˨˩`, falling `˥˩`, high `˦˥`, rising `˨˩˦`.

---

### Task 1: Project scaffold and note models

**Files:**
- Create: `pyproject.toml`, `src/thai_deck_eval/__init__.py`, `src/thai_deck_eval/model/__init__.py`, `src/thai_deck_eval/model/notes.py`
- Test: `tests/test_notes.py`

**Interfaces:**
- Produces: pydantic models `Audio(file, source, speaker)`, `PairMember(thai, ipa, audio, gloss=None)`, `MinimalPairNote(id, contrast, members)`, `SpellingSoundNote(id, pattern, pattern_kind, consonant_class=None, example_word, audio, image)`, `PictureWordNote(id, thai, image, audio, frequency_rank, category, part_of_speech="other", classifier=None, ipa=None, test_spelling=False, personal_connection=None, gloss=None)`, `SentenceNote(id, kind, thai, target, audio, image=None, definition=None, gloss=None, grammar_note=None)`, `DeckMeta(name, version, stage_plan)`, `StagePlan(phases)`. All `model_config = ConfigDict(extra="forbid")`.

- [ ] **Step 1: Scaffold**

```bash
cd /Users/haynes/projects/thai-language-anki
uv init --lib --name thai-deck-eval --package .   # if it refuses non-empty dir, create pyproject by hand as below
mkdir -p src/thai_deck_eval/model tests
```

`pyproject.toml`:

```toml
[project]
name = "thai-deck-eval"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.7", "pyyaml>=6", "click>=8.1"]

[project.optional-dependencies]
nlp = ["pythainlp>=5", "tltk>=1.9"]
llm = ["anthropic>=1"]

[project.scripts]
thai-deck-eval = "thai_deck_eval.cli:main"

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/thai_deck_eval"]

[tool.pytest.ini_options]
markers = ["integration: real pythainlp/tltk", "live: real LLM calls"]
addopts = "-m 'not integration and not live'"
```

- [ ] **Step 2: Write the failing tests**

`tests/test_notes.py`:

```python
import pytest
from pydantic import ValidationError
from thai_deck_eval.model.notes import (
    Audio, MinimalPairNote, PairMember, PictureWordNote, SentenceNote,
)

AUD = {"file": "audio/a.mp3", "source": "native", "speaker": "s1"}

def test_minimal_pair_requires_two_members():
    m = {"thai": "ขาว", "ipa": "kʰaːw˨˩˦", "audio": AUD}
    with pytest.raises(ValidationError):
        MinimalPairNote(id="mp1", contrast="tone", members=[m])
    note = MinimalPairNote(id="mp1", contrast="tone", members=[m, m])
    assert note.contrast == "tone"

def test_audio_source_restricted():
    with pytest.raises(ValidationError):
        Audio(file="a.mp3", source="robot", speaker="s1")

def test_picture_word_defaults():
    w = PictureWordNote(id="w1", thai="หมา", image="images/dog.png",
                        audio=AUD, frequency_rank=120, category="Animals")
    assert w.test_spelling is False and w.classifier is None

def test_sentence_kind_restricted():
    with pytest.raises(ValidationError):
        SentenceNote(id="s1", kind="poem", thai="…", target="…", audio=AUD)

def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        PictureWordNote(id="w1", thai="หมา", image="i.png", audio=AUD,
                        frequency_rank=1, category="Animals", bogus=1)
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_notes.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 4: Implement** `src/thai_deck_eval/model/notes.py`:

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Contrast = Literal["tone", "vowel_length", "aspiration", "vowel_quality", "consonant", "final"]

class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Audio(_Model):
    file: str
    source: Literal["native", "tts"]
    speaker: str

class PairMember(_Model):
    thai: str
    ipa: str
    audio: Audio
    gloss: str | None = None

class MinimalPairNote(_Model):
    id: str
    contrast: Contrast
    members: list[PairMember] = Field(min_length=2, max_length=3)

class SpellingSoundNote(_Model):
    id: str
    pattern: str
    pattern_kind: Literal["consonant", "vowel", "tone_mark"]
    consonant_class: Literal["mid", "high", "low"] | None = None
    example_word: str
    audio: Audio
    image: str

class PictureWordNote(_Model):
    id: str
    thai: str
    image: str
    audio: Audio
    frequency_rank: int
    category: str
    part_of_speech: Literal["noun", "verb", "adjective", "other"] = "other"
    classifier: str | None = None
    ipa: str | None = None
    test_spelling: bool = False
    personal_connection: str | None = None
    gloss: str | None = None

class SentenceNote(_Model):
    id: str
    kind: Literal["new_word", "word_form", "word_order"]
    thai: str
    target: str
    audio: Audio
    image: str | None = None
    definition: str | None = None
    gloss: str | None = None
    grammar_note: str | None = None

class StagePlan(_Model):
    phases: list[Literal["sounds", "words", "sentences"]]

class DeckMeta(_Model):
    name: str
    version: str
    stage_plan: StagePlan
```

Add empty `src/thai_deck_eval/__init__.py` and `src/thai_deck_eval/model/__init__.py`.

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/test_notes.py -v` → all PASS.
- [ ] **Step 6: Commit** — message: `Add project scaffold and deck note models`

---

### Task 2: Deck loader and fixture builder

**Files:**
- Create: `src/thai_deck_eval/model/deck.py`, `tests/helpers.py`
- Test: `tests/test_deck_loader.py`

**Interfaces:**
- Consumes: Task 1 models.
- Produces: `Deck(meta, minimal_pairs, spelling_sound, picture_words, sentences, root: Path)` with `Deck.all_notes() -> list[tuple[str, object]]` (family, note); `load_deck(path: Path) -> Deck`; `DeckSchemaError(issues: list[str])`. `tests/helpers.py: DeckBuilder` — `DeckBuilder(tmp_path).build() -> Path` writes a valid golden mini-deck (YAML + stub media); mutation via keyword hooks shown below.

- [ ] **Step 1: Write the failing tests**

`tests/helpers.py`:

```python
"""Programmatic fixture decks. build() writes a valid golden mini-deck."""
from pathlib import Path
import yaml

def _aud(name, speaker="s1", source="native"):
    return {"file": f"audio/{name}", "source": source, "speaker": speaker}

GOLDEN = {
    "deck": {"name": "golden", "version": "0.1",
             "stage_plan": {"phases": ["sounds", "words", "sentences"]}},
    "minimal_pairs": [
        {"id": "mp-tone-1", "contrast": "tone", "members": [
            {"thai": "ขาว", "ipa": "kʰaːw˨˩˦", "audio": _aud("khao-r.mp3", "s1")},
            {"thai": "ข่าว", "ipa": "kʰaːw˨˩", "audio": _aud("khao-l.mp3", "s2")}]},
        {"id": "mp-asp-1", "contrast": "aspiration", "members": [
            {"thai": "ไก่", "ipa": "kaj˨˩", "audio": _aud("kai.mp3", "s1")},
            {"thai": "ไข่", "ipa": "kʰaj˨˩", "audio": _aud("khai.mp3", "s3")}]},
    ],
    "spelling_sound": [
        {"id": "ss-1", "pattern": "ข", "pattern_kind": "consonant",
         "consonant_class": "high", "example_word": "ไข่",
         "audio": _aud("khai.mp3"), "image": "images/egg.png"},
    ],
    "picture_words": [
        {"id": "w-dog", "thai": "หมา", "image": "images/dog.png",
         "audio": _aud("maa.mp3"), "frequency_rank": 120, "category": "Animals",
         "part_of_speech": "noun", "classifier": "ตัว", "ipa": "maː˨˩˦"},
        {"id": "w-come", "thai": "มา", "image": "images/come.png",
         "audio": _aud("maa2.mp3"), "frequency_rank": 15, "category": "Verbs",
         "part_of_speech": "verb", "ipa": "maː˧"},
        {"id": "w-rice", "thai": "ข้าว", "image": "images/rice.png",
         "audio": _aud("khao-f.mp3"), "frequency_rank": 90, "category": "Food",
         "part_of_speech": "noun", "classifier": "จาน", "ipa": "kʰaːw˥˩"},
    ],
    "sentences": [
        {"id": "s-1", "kind": "new_word", "thai": "หมามากินข้าว",
         "target": "กิน", "audio": _aud("s1.mp3"),
         "image": "images/eat.png", "definition": "เอาอาหารเข้าปาก"},
    ],
}

class DeckBuilder:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "deck"
        import copy
        self.data = copy.deepcopy(GOLDEN)

    def build(self) -> Path:
        notes = self.root / "notes"
        notes.mkdir(parents=True, exist_ok=True)
        (self.root / "deck.yaml").write_text(
            yaml.safe_dump(self.data["deck"], allow_unicode=True))
        for fam in ("minimal_pairs", "spelling_sound", "picture_words", "sentences"):
            (notes / f"{fam}.yaml").write_text(
                yaml.safe_dump(self.data[fam], allow_unicode=True))
        self._write_media()
        return self.root

    def _write_media(self):
        for sub in ("audio", "images"):
            (self.root / "media" / sub).mkdir(parents=True, exist_ok=True)
        for ref in self._media_refs():
            p = self.root / "media" / ref
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00stub")

    def _media_refs(self):
        refs = []
        def walk(o):
            if isinstance(o, dict):
                if "file" in o and "source" in o:
                    refs.append(o["file"])
                for k, v in o.items():
                    if k == "image" and isinstance(v, str):
                        refs.append(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(self.data)
        return refs
```

`tests/test_deck_loader.py`:

```python
import pytest
from thai_deck_eval.model.deck import DeckSchemaError, load_deck
from tests.helpers import DeckBuilder

def test_loads_golden(tmp_path):
    deck = load_deck(DeckBuilder(tmp_path).build())
    assert deck.meta.name == "golden"
    assert len(deck.picture_words) == 3
    assert deck.root.name == "deck"
    fams = {f for f, _ in deck.all_notes()}
    assert fams == {"minimal_pair", "spelling_sound", "picture_word", "sentence"}

def test_schema_error_reports_file_and_note(tmp_path):
    b = DeckBuilder(tmp_path)
    del b.data["picture_words"][0]["category"]
    with pytest.raises(DeckSchemaError) as e:
        load_deck(b.build())
    assert any("picture_words" in i and "w-dog" in i for i in e.value.issues)

def test_missing_notes_file_is_schema_error(tmp_path):
    root = DeckBuilder(tmp_path).build()
    (root / "notes" / "sentences.yaml").unlink()
    with pytest.raises(DeckSchemaError):
        load_deck(root)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_deck_loader.py -v` → FAIL.

- [ ] **Step 3: Implement** `src/thai_deck_eval/model/deck.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from pydantic import ValidationError
from .notes import (DeckMeta, MinimalPairNote, PictureWordNote,
                    SentenceNote, SpellingSoundNote)

_FAMILIES = [
    ("minimal_pairs", "minimal_pair", MinimalPairNote),
    ("spelling_sound", "spelling_sound", SpellingSoundNote),
    ("picture_words", "picture_word", PictureWordNote),
    ("sentences", "sentence", SentenceNote),
]

class DeckSchemaError(Exception):
    def __init__(self, issues: list[str]):
        super().__init__(f"{len(issues)} schema issue(s)")
        self.issues = issues

@dataclass
class Deck:
    meta: DeckMeta
    minimal_pairs: list[MinimalPairNote] = field(default_factory=list)
    spelling_sound: list[SpellingSoundNote] = field(default_factory=list)
    picture_words: list[PictureWordNote] = field(default_factory=list)
    sentences: list[SentenceNote] = field(default_factory=list)
    root: Path = Path(".")

    def all_notes(self) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        for attr, fam, _ in _FAMILIES:
            out += [(fam, n) for n in getattr(self, attr)]
        return out

def load_deck(path: Path) -> Deck:
    issues: list[str] = []
    path = Path(path)
    try:
        meta = DeckMeta.model_validate(
            yaml.safe_load((path / "deck.yaml").read_text()))
    except (OSError, ValidationError, yaml.YAMLError) as e:
        raise DeckSchemaError([f"deck.yaml: {e}"])
    deck = Deck(meta=meta, root=path)
    for attr, _fam, model in _FAMILIES:
        fpath = path / "notes" / f"{attr}.yaml"
        try:
            raw = yaml.safe_load(fpath.read_text()) or []
        except (OSError, yaml.YAMLError) as e:
            issues.append(f"notes/{attr}.yaml: {e}")
            continue
        for entry in raw:
            note_id = entry.get("id", "?") if isinstance(entry, dict) else "?"
            try:
                getattr(deck, attr).append(model.model_validate(entry))
            except ValidationError as e:
                issues.append(f"notes/{attr}.yaml [{note_id}]: {e}")
    if issues:
        raise DeckSchemaError(issues)
    return deck
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_deck_loader.py tests/test_notes.py -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add deck loader and fixture deck builder`

---

### Task 3: Findings, rule registry, and evaluation context

**Files:**
- Create: `src/thai_deck_eval/core/__init__.py`, `src/thai_deck_eval/core/findings.py`, `src/thai_deck_eval/core/registry.py`, `src/thai_deck_eval/core/context.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces: `Severity`/`Dimension`/`Stage` StrEnums (`Stage`: `SCHEMA, MECHANICAL, LINGUISTIC, METHOD, JUDGE`); `Finding(rule, severity, dimension, message, note_id=None, evidence={})`; `Metric(name, value, dimension=Dimension.METHOD, detail={})`; decorator `@rule(id, stage, dimension, default_severity)` registering `fn(ctx) -> Iterable[Finding | Metric]`; `rules_for(stage) -> list[RuleDef]`; `RuleDef(id, stage, dimension, default_severity, fn)` with `RuleDef.finding(message, note_id=None, severity=None, evidence=None) -> Finding`; `EvalContext(deck, config, g2p=None, g2p_second=None, tokenizer=None, freq=None, judge=None)` (plain dataclass; `config` is `dict` until Task 12 replaces it with `RulebookConfig` — rules read config via `ctx.cfg(key, default)`).

- [ ] **Step 1: Write the failing tests**

`tests/test_registry.py`:

```python
from thai_deck_eval.core.findings import Dimension, Finding, Metric, Severity, Stage
from thai_deck_eval.core.registry import _REGISTRY, rule, rules_for

def test_rule_registration_and_finding_defaults():
    @rule("mech/example", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
    def example(ctx):
        yield example.finding("boom", note_id="n1")
    try:
        rd = next(r for r in rules_for(Stage.MECHANICAL) if r.id == "mech/example")
        f = list(rd.fn(None))[0]
        assert isinstance(f, Finding)
        assert (f.rule, f.severity, f.dimension, f.note_id) == (
            "mech/example", Severity.ERROR, Dimension.INTEGRITY, "n1")
    finally:
        _REGISTRY.pop("mech/example")

def test_metric_defaults():
    m = Metric(name="coverage/pairs", value=0.5)
    assert m.dimension == Dimension.METHOD

def test_duplicate_id_rejected():
    @rule("mech/dup", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
    def a(ctx): ...
    try:
        import pytest
        with pytest.raises(ValueError):
            @rule("mech/dup", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
            def b(ctx): ...
    finally:
        _REGISTRY.pop("mech/dup")
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_registry.py -v` → FAIL.

- [ ] **Step 3: Implement**

`src/thai_deck_eval/core/findings.py`:

```python
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field

class Severity(StrEnum):
    ERROR = "error"; WARN = "warn"; INFO = "info"

class Dimension(StrEnum):
    INTEGRITY = "integrity"; LANGUAGE = "language"
    METHOD = "method"; CONTENT = "content"

class Stage(StrEnum):
    SCHEMA = "schema"; MECHANICAL = "mechanical"
    LINGUISTIC = "linguistic"; METHOD = "method"; JUDGE = "judge"

class Finding(BaseModel):
    rule: str
    severity: Severity
    dimension: Dimension
    message: str
    note_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

class Metric(BaseModel):
    name: str
    value: float
    dimension: Dimension = Dimension.METHOD
    detail: dict[str, Any] = Field(default_factory=dict)
```

`src/thai_deck_eval/core/registry.py`:

```python
from dataclasses import dataclass
from typing import Callable, Iterable
from .findings import Dimension, Finding, Metric, Severity, Stage

@dataclass
class RuleDef:
    id: str
    stage: Stage
    dimension: Dimension
    default_severity: Severity
    fn: Callable

    def finding(self, message, note_id=None, severity=None, evidence=None) -> Finding:
        return Finding(rule=self.id, severity=severity or self.default_severity,
                       dimension=self.dimension, message=message,
                       note_id=note_id, evidence=evidence or {})

_REGISTRY: dict[str, RuleDef] = {}

def rule(rule_id: str, stage: Stage, dimension: Dimension, default_severity: Severity):
    def deco(fn):
        if rule_id in _REGISTRY:
            raise ValueError(f"duplicate rule id {rule_id}")
        rd = RuleDef(rule_id, stage, dimension, default_severity, fn)
        _REGISTRY[rule_id] = rd
        fn.finding = rd.finding
        fn.rule_def = rd
        return fn
    return deco

def rules_for(stage: Stage) -> list[RuleDef]:
    return [r for r in _REGISTRY.values() if r.stage == stage]
```

`src/thai_deck_eval/core/context.py`:

```python
from dataclasses import dataclass, field
from typing import Any
from ..model.deck import Deck

@dataclass
class EvalContext:
    deck: Deck
    config: Any = field(default_factory=dict)
    g2p: Any = None
    g2p_second: Any = None
    tokenizer: Any = None
    freq: Any = None
    judge: Any = None

    def cfg(self, key: str, default=None):
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)
```

Empty `src/thai_deck_eval/core/__init__.py`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_registry.py -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add finding model, rule registry, and evaluation context`

---

### Task 4: Staged pipeline runner

**Files:**
- Create: `src/thai_deck_eval/core/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: registry, findings, `load_deck`/`DeckSchemaError`.
- Produces: `EvalResult(findings: list[Finding], metrics: list[Metric], stages_run: list[Stage], stages_skipped: list[Stage])`; `run_pipeline(ctx, stages: list[Stage] | None = None) -> EvalResult` — runs MECHANICAL, LINGUISTIC, METHOD, JUDGE in order (METHOD runs in the linguistic gate group: gate before LINGUISTIC also gates METHOD? No — see gating below); `evaluate_path(path, ctx_factory) -> EvalResult` wraps `load_deck`, converting `DeckSchemaError` issues to `schema/invalid` error findings and skipping everything else. Gating: LINGUISTIC+METHOD run only if no MECHANICAL errors... **exact policy:** stage order `[MECHANICAL, LINGUISTIC, METHOD, JUDGE]`; before each stage, if any `error`-severity finding exists from earlier stages **and** `ctx.cfg("gates", True)` is truthy, skip that stage and all later ones, recording them in `stages_skipped`. `stages` arg filters which stages may run (for `--stages`/`--no-judge`).

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
from thai_deck_eval.core.findings import Dimension, Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.core.registry import _REGISTRY, rule
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import DeckMeta, StagePlan

def _deck():
    return Deck(meta=DeckMeta(name="t", version="0",
                stage_plan=StagePlan(phases=["sounds"])))

def _with_rules(rules, fn):
    try:
        fn()
    finally:
        for rid in rules:
            _REGISTRY.pop(rid, None)

def test_error_gates_later_stages():
    @rule("mech/t-fail", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
    def fail(ctx):
        yield fail.finding("bad")
    ran = []
    @rule("lang/t-probe", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
    def probe(ctx):
        ran.append(1)
        return []
    def go():
        res = run_pipeline(EvalContext(deck=_deck()))
        assert Stage.LINGUISTIC in res.stages_skipped
        assert ran == []
        assert [f.rule for f in res.findings] == ["mech/t-fail"]
    _with_rules(["mech/t-fail", "lang/t-probe"], go)

def test_warn_does_not_gate_and_metrics_collected():
    @rule("mech/t-warn", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
    def w(ctx):
        yield w.finding("meh")
    @rule("meth/t-metric", Stage.METHOD, Dimension.METHOD, Severity.INFO)
    def m(ctx):
        from thai_deck_eval.core.findings import Metric
        yield Metric(name="coverage/x", value=0.5)
    def go():
        res = run_pipeline(EvalContext(deck=_deck()))
        assert res.stages_skipped == []
        assert [m.name for m in res.metrics] == ["coverage/x"]
    _with_rules(["mech/t-warn", "meth/t-metric"], go)

def test_stage_filter():
    ran = []
    @rule("judge/t-probe", Stage.JUDGE, Dimension.CONTENT, Severity.WARN)
    def p(ctx):
        ran.append(1)
        return []
    def go():
        run_pipeline(EvalContext(deck=_deck()), stages=[Stage.MECHANICAL])
        assert ran == []
    _with_rules(["judge/t-probe"], go)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** `src/thai_deck_eval/core/pipeline.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from .context import EvalContext
from .findings import Dimension, Finding, Metric, Severity, Stage
from .registry import rules_for
from ..model.deck import DeckSchemaError, load_deck

ORDER = [Stage.MECHANICAL, Stage.LINGUISTIC, Stage.METHOD, Stage.JUDGE]

@dataclass
class EvalResult:
    findings: list[Finding] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    stages_run: list[Stage] = field(default_factory=list)
    stages_skipped: list[Stage] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

def run_pipeline(ctx: EvalContext, stages: list[Stage] | None = None) -> EvalResult:
    res = EvalResult()
    enabled = stages if stages is not None else ORDER
    gated = False
    for stage in ORDER:
        if stage not in enabled:
            continue
        if gated:
            res.stages_skipped.append(stage)
            continue
        for rd in rules_for(stage):
            for item in rd.fn(ctx) or []:
                (res.metrics if isinstance(item, Metric) else res.findings).append(item)
        res.stages_run.append(stage)
        if ctx.cfg("gates", True) and res.has_errors:
            gated = True
    return res

def evaluate_path(path: Path, ctx_factory, stages=None) -> EvalResult:
    try:
        deck = load_deck(path)
    except DeckSchemaError as e:
        res = EvalResult(stages_skipped=list(ORDER))
        res.findings = [Finding(rule="schema/invalid", severity=Severity.ERROR,
                                dimension=Dimension.INTEGRITY, message=i)
                        for i in e.issues]
        return res
    return run_pipeline(ctx_factory(deck), stages=stages)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add staged pipeline runner with error gating`

---

### Task 5: Mechanical rules

**Files:**
- Create: `src/thai_deck_eval/stages/__init__.py`, `src/thai_deck_eval/stages/mechanical.py`
- Test: `tests/test_mechanical.py`

**Interfaces:**
- Consumes: registry, `EvalContext`, `Deck.all_notes()`, `DeckBuilder`.
- Produces: rules `mech/media-missing` (ERROR), `mech/media-orphan` (INFO), `mech/latin-in-thai` (ERROR), `mech/duplicate-note` (WARN), `mech/target-not-in-sentence` (ERROR), `mech/gloss-on-picture-word` (WARN). Helper `iter_media_refs(deck) -> Iterator[tuple[str, str]]` (note_id, relpath). Importing `thai_deck_eval.stages.mechanical` registers the rules (the CLI and tests import stage modules for side effects).

- [ ] **Step 1: Write the failing tests**

`tests/test_mechanical.py`:

```python
import pytest
import thai_deck_eval.stages.mechanical  # noqa: F401  (registers rules)
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.model.deck import load_deck
from tests.helpers import DeckBuilder

def _run(root):
    return run_pipeline(EvalContext(deck=load_deck(root)),
                        stages=[Stage.MECHANICAL])

def _rules(res):
    return sorted(f.rule for f in res.findings)

def test_golden_is_clean(tmp_path):
    assert _rules(_run(DeckBuilder(tmp_path).build())) == []

def test_media_missing(tmp_path):
    root = DeckBuilder(tmp_path).build()
    (root / "media" / "images" / "dog.png").unlink()
    res = _run(root)
    assert "mech/media-missing" in _rules(res)
    f = next(f for f in res.findings if f.rule == "mech/media-missing")
    assert f.note_id == "w-dog" and f.severity == Severity.ERROR

def test_media_orphan(tmp_path):
    root = DeckBuilder(tmp_path).build()
    (root / "media" / "audio" / "unused.mp3").write_bytes(b"x")
    assert "mech/media-orphan" in _rules(_run(root))

def test_latin_in_thai_field(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["thai"] = "maa หมา"
    assert "mech/latin-in-thai" in _rules(_run(b.build()))

def test_duplicate_picture_word(tmp_path):
    b = DeckBuilder(tmp_path)
    dup = dict(b.data["picture_words"][0]); dup["id"] = "w-dog2"
    b.data["picture_words"].append(dup)
    assert "mech/duplicate-note" in _rules(_run(b.build()))

def test_target_not_in_sentence(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["target"] = "วิ่ง"
    assert "mech/target-not-in-sentence" in _rules(_run(b.build()))

def test_gloss_on_picture_word(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["gloss"] = "dog"
    assert "mech/gloss-on-picture-word" in _rules(_run(b.build()))
```

- [ ] **Step 2: Run to verify failure** → FAIL (module missing).

- [ ] **Step 3: Implement** `src/thai_deck_eval/stages/mechanical.py`:

```python
import re
import unicodedata
from ..core.findings import Dimension, Severity, Stage
from ..core.registry import rule

_LATIN = re.compile(r"[A-Za-z]")

def iter_media_refs(deck):
    for note in deck.minimal_pairs:
        for m in note.members:
            yield note.id, m.audio.file
    for note in deck.spelling_sound:
        yield note.id, note.audio.file
        yield note.id, note.image
    for note in deck.picture_words:
        yield note.id, note.audio.file
        yield note.id, note.image
    for note in deck.sentences:
        yield note.id, note.audio.file
        if note.image:
            yield note.id, note.image

@rule("mech/media-missing", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
def media_missing(ctx):
    for note_id, ref in iter_media_refs(ctx.deck):
        if not (ctx.deck.root / "media" / ref).is_file():
            yield media_missing.finding(f"media file not found: {ref}",
                                        note_id=note_id)

@rule("mech/media-orphan", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.INFO)
def media_orphan(ctx):
    media_dir = ctx.deck.root / "media"
    if not media_dir.is_dir():
        return
    referenced = {ref for _, ref in iter_media_refs(ctx.deck)}
    for p in media_dir.rglob("*"):
        if p.is_file() and str(p.relative_to(media_dir)) not in referenced:
            yield media_orphan.finding(
                f"unreferenced media file: {p.relative_to(media_dir)}")

def _thai_fields(deck):
    for note in deck.minimal_pairs:
        for m in note.members:
            yield note.id, "thai", m.thai
    for note in deck.spelling_sound:
        yield note.id, "example_word", note.example_word
    for note in deck.picture_words:
        yield note.id, "thai", note.thai
    for note in deck.sentences:
        yield note.id, "thai", note.thai
        yield note.id, "target", note.target
        if note.definition:
            yield note.id, "definition", note.definition

@rule("mech/latin-in-thai", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
def latin_in_thai(ctx):
    for note_id, fieldname, text in _thai_fields(ctx.deck):
        if _LATIN.search(unicodedata.normalize("NFC", text)):
            yield latin_in_thai.finding(
                f"Latin characters in Thai field '{fieldname}': {text!r}",
                note_id=note_id)

@rule("mech/duplicate-note", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
def duplicate_note(ctx):
    seen: dict[str, str] = {}
    keys = [(n.id, f"picture:{n.thai}") for n in ctx.deck.picture_words]
    keys += [(n.id, f"sentence:{n.thai}") for n in ctx.deck.sentences]
    for note_id, key in keys:
        if key in seen:
            yield duplicate_note.finding(
                f"duplicate of note {seen[key]}", note_id=note_id)
        else:
            seen[key] = note_id

@rule("mech/target-not-in-sentence", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
def target_not_in_sentence(ctx):
    for note in ctx.deck.sentences:
        if note.target not in note.thai:
            yield target_not_in_sentence.finding(
                f"target {note.target!r} not found in sentence", note_id=note.id)

@rule("mech/gloss-on-picture-word", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
def gloss_on_picture_word(ctx):
    for note in ctx.deck.picture_words:
        if note.gloss:
            yield gloss_on_picture_word.finding(
                "concrete picture words carry no L1 gloss", note_id=note.id)
```

Empty `src/thai_deck_eval/stages/__init__.py`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add mechanical integrity rules`

---

### Task 6: Thai tone engine and syllable analyzer

**Files:**
- Create: `src/thai_deck_eval/lang/__init__.py`, `src/thai_deck_eval/lang/tone.py`
- Test: `tests/test_tone.py`

**Interfaces:**
- Produces: `Tone` StrEnum (`MID, LOW, FALLING, HIGH, RISING`); `ConsClass` StrEnum (`MID, HIGH, LOW`); `CONSONANT_CLASS: dict[str, ConsClass]` (all 44 letters); `tone_of(cls: ConsClass, live: bool, long_vowel: bool, mark: str | None) -> Tone` (mark ∈ {None, "่", "้", "๊", "๋"}); `SyllableAnalysis(initial, cls, vowel, long_vowel, final, live, mark, tone)`; `analyze_syllable(word: str) -> SyllableAnalysis | None` — parses common single-syllable words; returns None when it cannot parse (multi-syllable, rare vowel forms). Handles ห นำ and อ นำ.

- [ ] **Step 1: Write the failing tests**

`tests/test_tone.py`:

```python
import pytest
from thai_deck_eval.lang.tone import (ConsClass, Tone, analyze_syllable, tone_of)

@pytest.mark.parametrize("cls,live,long_v,mark,expected", [
    (ConsClass.MID, True, True, None, Tone.MID),        # กา
    (ConsClass.HIGH, True, True, None, Tone.RISING),    # ขา
    (ConsClass.LOW, True, True, None, Tone.MID),        # คา
    (ConsClass.MID, False, True, None, Tone.LOW),       # บาท
    (ConsClass.HIGH, False, False, None, Tone.LOW),     # ขับ
    (ConsClass.LOW, False, False, None, Tone.HIGH),     # คับ
    (ConsClass.LOW, False, True, None, Tone.FALLING),   # มาก
    (ConsClass.MID, True, True, "่", Tone.LOW),
    (ConsClass.HIGH, True, True, "่", Tone.LOW),
    (ConsClass.LOW, True, True, "่", Tone.FALLING),
    (ConsClass.MID, True, True, "้", Tone.FALLING),
    (ConsClass.HIGH, True, True, "้", Tone.FALLING),
    (ConsClass.LOW, True, True, "้", Tone.HIGH),
    (ConsClass.MID, True, True, "๊", Tone.HIGH),
    (ConsClass.MID, True, True, "๋", Tone.RISING),
])
def test_tone_table(cls, live, long_v, mark, expected):
    assert tone_of(cls, live, long_v, mark) == expected

@pytest.mark.parametrize("word,tone", [
    ("มา", Tone.MID), ("หมา", Tone.RISING),          # ห นำ
    ("ไม่", Tone.FALLING), ("ไม้", Tone.HIGH),
    ("ใหม่", Tone.LOW), ("ไหม", Tone.RISING),
    ("ขาว", Tone.RISING), ("ข่าว", Tone.LOW), ("ข้าว", Tone.FALLING),
    ("ไก่", Tone.LOW), ("ไข่", Tone.LOW),
    ("มาก", Tone.FALLING), ("อยู่", Tone.LOW),        # อ นำ
    ("กิน", Tone.MID),
])
def test_analyze_known_words(word, tone):
    a = analyze_syllable(word)
    assert a is not None, word
    assert a.tone == tone

def test_unparseable_returns_none():
    assert analyze_syllable("โรงเรียน") is None  # multi-syllable
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** `src/thai_deck_eval/lang/tone.py`:

```python
"""Deterministic Thai tone rules: consonant class × live/dead × mark.

Sources: thai-language.com/ref/tone-rules; thaiwithgrace.com/thai-tones
(class split 9 mid / 11 high / 24 low).
"""
from dataclasses import dataclass
from enum import StrEnum

class Tone(StrEnum):
    MID = "mid"; LOW = "low"; FALLING = "falling"; HIGH = "high"; RISING = "rising"

class ConsClass(StrEnum):
    MID = "mid"; HIGH = "high"; LOW = "low"

_MID = "กจฎฏดตบปอ"
_HIGH = "ขฃฉฐถผฝศษสห"
_LOW = "คฅฆงชซฌญฑฒณทธนพฟภมยรลวฬฮ"
CONSONANT_CLASS: dict[str, ConsClass] = (
    {c: ConsClass.MID for c in _MID}
    | {c: ConsClass.HIGH for c in _HIGH}
    | {c: ConsClass.LOW for c in _LOW})

MAI_EK, MAI_THO, MAI_TRI, MAI_CHATTAWA = "่", "้", "๊", "๋"
_MARKS = {MAI_EK, MAI_THO, MAI_TRI, MAI_CHATTAWA}

_SONORANT_FINALS = set("งนมณญยรลฬวว")
_STOP_FINALS = set("กขคฆจชซฌฎฏฐฑฒดตถทธบปพฟภศษส")
_LOW_SONORANTS = set("งญณนมยรลวฬ")

def tone_of(cls: ConsClass, live: bool, long_vowel: bool, mark: str | None) -> Tone:
    if mark == MAI_EK:
        return Tone.FALLING if cls == ConsClass.LOW else Tone.LOW
    if mark == MAI_THO:
        return Tone.HIGH if cls == ConsClass.LOW else Tone.FALLING
    if mark == MAI_TRI:
        return Tone.HIGH
    if mark == MAI_CHATTAWA:
        return Tone.RISING
    if live:
        return Tone.RISING if cls == ConsClass.HIGH else Tone.MID
    if cls == ConsClass.LOW:
        return Tone.FALLING if long_vowel else Tone.HIGH
    return Tone.LOW

@dataclass
class SyllableAnalysis:
    initial: str
    cls: ConsClass
    vowel: str
    long_vowel: bool
    final: str | None
    live: bool
    mark: str | None
    tone: Tone

# (template, vowel name, long?, allows_final?) — pre-vowels use "-" for the
# initial slot; combining vowels/marks are stripped before template matching.
_PRE_VOWELS = {"เ": ("e", True), "แ": ("ɛ", True), "โ": ("o", True),
               "ไ": ("aj", False), "ใ": ("aj", False)}
_POST_LONG = {"า": ("a", True)}
_ABOVE_BELOW = {"ิ": ("i", False), "ี": ("i", True),
                "ึ": ("ɯ", False), "ื": ("ɯ", True),
                "ุ": ("u", False), "ู": ("u", True),
                "ั": ("a", False)}   # ◌ั
_SARA_A = "ะ"

def analyze_syllable(word: str) -> SyllableAnalysis | None:
    chars = list(word)
    mark = next((c for c in chars if c in _MARKS), None)
    chars = [c for c in chars if c not in _MARKS]

    pre = None
    if chars and chars[0] in _PRE_VOWELS:
        pre = chars.pop(0)

    if not chars or chars[0] not in CONSONANT_CLASS:
        return None
    initial = chars.pop(0)
    cls = CONSONANT_CLASS[initial]
    # ห นำ / อ นำ: leading silent ห (or อ) + low sonorant → leader's class
    if chars and chars[0] in CONSONANT_CLASS:
        nxt = chars[0]
        if initial == "ห" and nxt in _LOW_SONORANTS:
            initial, cls = chars.pop(0), ConsClass.HIGH
        elif initial == "อ" and nxt == "ย":
            initial, cls = chars.pop(0), ConsClass.MID
        elif pre is None and nxt not in _SONORANT_FINALS | _STOP_FINALS:
            return None

    vowel = long_v = None
    if pre is not None:
        if any(c in _ABOVE_BELOW or c == _SARA_A or c == "า" for c in chars):
            return None  # complex เ-ือ / เ-าะ / เ-ีย forms: out of scope
        vowel, long_v = _PRE_VOWELS[pre]
    else:
        for c in list(chars):
            if c in _ABOVE_BELOW:
                vowel, long_v = _ABOVE_BELOW[c]
                chars.remove(c)
                break
            if c in _POST_LONG:
                vowel, long_v = _POST_LONG[c]
                chars.remove(c)
                break
            if c == _SARA_A:
                vowel, long_v = "a", False
                chars.remove(c)
                break
    if vowel is None:
        return None

    final = None
    if chars:
        if len(chars) > 1 or chars[0] not in _SONORANT_FINALS | _STOP_FINALS:
            return None
        final = chars[0]

    if pre in ("ไ", "ใ"):
        live = True          # -aj diphthong behaves live
    elif final is None:
        live = long_v
    else:
        live = final in _SONORANT_FINALS
    return SyllableAnalysis(initial, cls, vowel, bool(long_v), final, live,
                            mark, tone_of(cls, live, bool(long_v), mark))
```

Empty `src/thai_deck_eval/lang/__init__.py`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_tone.py -v` → all PASS. If a parser case fails, fix the parser, not the test expectations (expectations are from published tone tables).
- [ ] **Step 5: Commit** — message: `Add Thai tone engine and syllable analyzer`

---

### Task 7: IPA syllable parser and language ports

**Files:**
- Create: `src/thai_deck_eval/lang/ipa.py`, `src/thai_deck_eval/lang/ports.py`, `tests/fakes.py`
- Test: `tests/test_ipa.py`

**Interfaces:**
- Consumes: `Tone` from Task 6.
- Produces: `IpaSyllable(onset: str, vowel: str, long: bool, coda: str | None, tone: Tone)`; `parse_ipa(s: str) -> list[IpaSyllable]` (raises `IpaParseError` on garbage; syllables separated by `.` or space); `diff_features(a: IpaSyllable, b: IpaSyllable) -> set[str]` returning subset of `{"onset","vowel","length","coda","tone","aspiration"}` — aspiration reported (instead of onset) when onsets differ only by `ʰ`; ports in `ports.py`: `class G2P(Protocol): def syllables(self, word: str) -> list[IpaSyllable] | None`, `class Tokenizer(Protocol): def tokens(self, text: str) -> list[str]`, `class FrequencyList(Protocol): def rank(self, word: str) -> int | None`; `tests/fakes.py`: `FakeG2P(dict[str, str])` (values in authored IPA format, parsed via `parse_ipa`; unknown → None), `FakeTokenizer(dict[str, list[str]])` (falls back to `[text]`), `FakeFreq(dict[str, int])`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ipa.py`:

```python
import pytest
from thai_deck_eval.lang.ipa import IpaParseError, diff_features, parse_ipa
from thai_deck_eval.lang.tone import Tone

def test_parse_single_syllable():
    (s,) = parse_ipa("kʰaːw˥˩")
    assert (s.onset, s.vowel, s.long, s.coda, s.tone) == (
        "kʰ", "a", True, "w", Tone.FALLING)

def test_parse_no_coda_short():
    (s,) = parse_ipa("tɕa˨˩")
    assert (s.onset, s.vowel, s.long, s.coda, s.tone) == ("tɕ", "a", False, None, Tone.LOW)

def test_parse_multisyllable():
    syls = parse_ipa("maː˧.kʰaj˨˩")
    assert len(syls) == 2 and syls[1].tone == Tone.LOW

def test_parse_error():
    with pytest.raises(IpaParseError):
        parse_ipa("hello")

def test_diff_tone_only():
    a, b = parse_ipa("kʰaːw˨˩˦")[0], parse_ipa("kʰaːw˨˩")[0]
    assert diff_features(a, b) == {"tone"}

def test_diff_aspiration():
    a, b = parse_ipa("kaj˨˩")[0], parse_ipa("kʰaj˨˩")[0]
    assert diff_features(a, b) == {"aspiration"}

def test_diff_length_and_tone():
    a, b = parse_ipa("kʰaːw˥˩")[0], parse_ipa("kʰaw˨˩")[0]
    assert diff_features(a, b) == {"length", "tone"}
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement**

`src/thai_deck_eval/lang/ipa.py`:

```python
import re
from dataclasses import dataclass
from .tone import Tone

class IpaParseError(ValueError):
    pass

_TONES = {"˧": Tone.MID, "˨˩˦": Tone.RISING, "˨˩": Tone.LOW,
          "˥˩": Tone.FALLING, "˦˥": Tone.HIGH}
_ONSETS = ["tɕʰ", "tɕ", "pʰ", "tʰ", "kʰ", "b", "d", "p", "t", "k", "ʔ",
           "m", "n", "ŋ", "f", "s", "h", "w", "l", "j", "r"]
_VOWELS = ["ɯa", "ia", "ua", "ɯ", "ɤ", "ɛ", "ɔ", "i", "e", "a", "o", "u"]
_CODAS = ["p", "t", "k", "ʔ", "m", "n", "ŋ", "j", "w"]

@dataclass
class IpaSyllable:
    onset: str
    vowel: str
    long: bool
    coda: str | None
    tone: Tone

def _take(s: str, options: list[str]) -> tuple[str | None, str]:
    for o in options:
        if s.startswith(o):
            return o, s[len(o):]
    return None, s

def _parse_one(s: str) -> IpaSyllable:
    tone = None
    for mark in sorted(_TONES, key=len, reverse=True):
        if s.endswith(mark):
            tone, s = _TONES[mark], s.removesuffix(mark)
            break
    if tone is None:
        raise IpaParseError(f"no tone letters in {s!r}")
    onset, s = _take(s, _ONSETS)
    if onset is None:
        raise IpaParseError(f"unknown onset in {s!r}")
    vowel, s = _take(s, _VOWELS)
    if vowel is None:
        raise IpaParseError(f"unknown vowel in {s!r}")
    long = s.startswith("ː")
    s = s.removeprefix("ː")
    coda, s = _take(s, _CODAS)
    if s:
        raise IpaParseError(f"trailing {s!r}")
    return IpaSyllable(onset, vowel, long, coda, tone)

def parse_ipa(s: str) -> list[IpaSyllable]:
    parts = [p for p in re.split(r"[.\s]+", s.strip()) if p]
    if not parts:
        raise IpaParseError("empty")
    return [_parse_one(p) for p in parts]

def diff_features(a: IpaSyllable, b: IpaSyllable) -> set[str]:
    diffs: set[str] = set()
    if a.onset != b.onset:
        bare = {a.onset.replace("ʰ", ""), b.onset.replace("ʰ", "")}
        diffs.add("aspiration" if len(bare) == 1 else "onset")
    if a.vowel != b.vowel:
        diffs.add("vowel")
    if a.long != b.long:
        diffs.add("length")
    if a.coda != b.coda:
        diffs.add("coda")
    if a.tone != b.tone:
        diffs.add("tone")
    return diffs
```

`src/thai_deck_eval/lang/ports.py`:

```python
from typing import Protocol
from .ipa import IpaSyllable

class G2P(Protocol):
    def syllables(self, word: str) -> list[IpaSyllable] | None: ...

class Tokenizer(Protocol):
    def tokens(self, text: str) -> list[str]: ...

class FrequencyList(Protocol):
    def rank(self, word: str) -> int | None: ...
```

`tests/fakes.py`:

```python
from thai_deck_eval.lang.ipa import parse_ipa

class FakeG2P:
    def __init__(self, table: dict[str, str]):
        self.table = table
    def syllables(self, word):
        return parse_ipa(self.table[word]) if word in self.table else None

class FakeTokenizer:
    def __init__(self, table: dict[str, list[str]] | None = None):
        self.table = table or {}
    def tokens(self, text):
        return self.table.get(text, [text])

class FakeFreq:
    def __init__(self, table: dict[str, int]):
        self.table = table
    def rank(self, word):
        return self.table.get(word)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add IPA syllable parser and language ports`

---

### Task 8: Rulebook data files

**Files:**
- Create: `data/contrasts.yaml`, `data/spelling_targets.yaml`, `data/function_words.yaml`, `data/g2p_exceptions.yaml`, `scripts/fetch_frequency.py`, `data/frequency_th.txt` (generated, committed), `src/thai_deck_eval/data_io.py`
- Test: `tests/test_data_io.py`

**Interfaces:**
- Produces: `data_io.load_contrasts(path=None) -> list[ContrastEntry]` where `ContrastEntry(id, kind, weight)`, kind ∈ contrast literals with tone pairs as ids like `tone:mid-low`; `load_spelling_targets(path=None) -> dict[str, list[str]]` (keys `consonants`, `vowels`, `tone_marks`); `load_function_words(path=None) -> set[str]`; `load_g2p_exceptions(path=None) -> dict[str, str]` (word → authored IPA); `FileFrequencyList(path)` implementing the `FrequencyList` port (rank = 1-based line number). Default paths resolve `DATA_DIR = Path(__file__).parent.parent.parent / "data"` — but package installs need the repo checkout; acceptable: evaluator runs from the repo. Note this in a comment.

- [ ] **Step 1: Write data files**

`data/contrasts.yaml` (weights: hardest-for-English-speakers heaviest; from spec §method):

```yaml
# Thai contrast inventory for minimal-pair coverage. weight ~ difficulty for
# English speakers (Wayland & Guion 2004: mid-low hardest).
- {id: "tone:mid-low",      kind: tone,          weight: 5}
- {id: "tone:mid-high",     kind: tone,          weight: 4}
- {id: "tone:low-falling",  kind: tone,          weight: 3}
- {id: "tone:high-rising",  kind: tone,          weight: 3}
- {id: "tone:falling-rising", kind: tone,        weight: 2}
- {id: "tone:mid-falling",  kind: tone,          weight: 2}
- {id: "tone:mid-rising",   kind: tone,          weight: 2}
- {id: "tone:low-high",     kind: tone,          weight: 2}
- {id: "tone:low-rising",   kind: tone,          weight: 3}
- {id: "tone:high-falling", kind: tone,          weight: 2}
- {id: "aspiration:labial", kind: aspiration,    weight: 4}
- {id: "aspiration:alveolar", kind: aspiration,  weight: 4}
- {id: "aspiration:velar",  kind: aspiration,    weight: 4}
- {id: "aspiration:affricate", kind: aspiration, weight: 3}
- {id: "vowel_length",      kind: vowel_length,  weight: 4}
- {id: "consonant:ng-onset", kind: consonant,    weight: 3}
- {id: "vowel_quality:e-ɛ", kind: vowel_quality, weight: 3}
- {id: "vowel_quality:o-ɔ", kind: vowel_quality, weight: 3}
- {id: "vowel_quality:ɯ",   kind: vowel_quality, weight: 3}
- {id: "vowel_quality:ɤ",   kind: vowel_quality, weight: 2}
- {id: "consonant:r-l",     kind: consonant,     weight: 1}
- {id: "final:unreleased",  kind: final,         weight: 2}
```

`data/spelling_targets.yaml`: `consonants:` the 42 modern letters (9 mid `ก จ ฎ ฏ ด ต บ ป อ`, 11 high minus obsolete ฃ → `ข ฉ ฐ ถ ผ ฝ ศ ษ ส ห`, 24 low minus obsolete ฅ → `ค ฆ ง ช ซ ฌ ญ ฑ ฒ ณ ท ธ น พ ฟ ภ ม ย ร ล ว ฬ ฮ`), `vowels:` the forms `["-ะ","-ั-","-า","-ิ","-ี","-ึ","-ื","-ุ","-ู","เ-","เ-ะ","แ-","แ-ะ","โ-","โ-ะ","ไ-","ใ-","เ-า","-อ","เ-อ","เ-ีย","เ-ือ","-ัว","-ำ"]`, `tone_marks: ["่","้","๊","๋"]`.

`data/function_words.yaml`: `["ที่","ของ","และ","ใน","เป็น","ไป","มา","ได้","ให้","ว่า","ก็","จะ","ไม่","นี้","นั้น","ๆ"]` (starter allowlist; tune later).

`data/g2p_exceptions.yaml`: `{"น้ำ": "naːm˦˥"}` (irregular long vowel).

`scripts/fetch_frequency.py`:

```python
"""Fetch hermitdave/FrequencyWords th_50k (CC BY-SA 4.0), keep top 5000 words."""
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
       "master/content/2018/th/th_50k.txt")

def main(out="data/frequency_th.txt", n=5000):
    lines = urllib.request.urlopen(URL).read().decode("utf-8").splitlines()
    words = [ln.split(" ")[0] for ln in lines[:n]]
    header = ["# top {} Thai words from hermitdave/FrequencyWords (OpenSubtitles"
              " 2018), CC BY-SA 4.0".format(n)]
    open(out, "w").write("\n".join(header + words) + "\n")

if __name__ == "__main__":
    main(*sys.argv[1:])
```

Run it: `uv run python scripts/fetch_frequency.py` and commit the output. If the URL is unreachable, create `data/frequency_th.txt` with the header comment plus a placeholder note and file a TODO with the user — do NOT fabricate word lists.

- [ ] **Step 2: Write the failing tests**

`tests/test_data_io.py`:

```python
from thai_deck_eval.data_io import (FileFrequencyList, load_contrasts,
                                    load_function_words, load_g2p_exceptions,
                                    load_spelling_targets)

def test_contrasts_load_and_weights():
    entries = load_contrasts()
    ids = {e.id for e in entries}
    assert "tone:mid-low" in ids
    assert max(entries, key=lambda e: e.weight).id == "tone:mid-low"

def test_spelling_targets_counts():
    t = load_spelling_targets()
    assert len(t["consonants"]) == 42 and len(t["tone_marks"]) == 4

def test_function_words():
    assert "ที่" in load_function_words()

def test_g2p_exceptions():
    assert load_g2p_exceptions()["น้ำ"] == "naːm˦˥"

def test_frequency_list(tmp_path):
    p = tmp_path / "freq.txt"
    p.write_text("# header\nที่\nของ\n")
    fl = FileFrequencyList(p)
    assert fl.rank("ที่") == 1 and fl.rank("ของ") == 2 and fl.rank("x") is None
```

- [ ] **Step 3: Run to verify failure** → FAIL.

- [ ] **Step 4: Implement** `src/thai_deck_eval/data_io.py`:

```python
"""Loaders for rulebook data files. Data lives in the repo's data/ directory
(the evaluator runs from the repo checkout, not an installed wheel)."""
from dataclasses import dataclass
from pathlib import Path
import yaml

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

@dataclass
class ContrastEntry:
    id: str
    kind: str
    weight: float

def load_contrasts(path: Path | None = None) -> list[ContrastEntry]:
    raw = yaml.safe_load((path or DATA_DIR / "contrasts.yaml").read_text())
    return [ContrastEntry(**e) for e in raw]

def load_spelling_targets(path: Path | None = None) -> dict[str, list[str]]:
    return yaml.safe_load((path or DATA_DIR / "spelling_targets.yaml").read_text())

def load_function_words(path: Path | None = None) -> set[str]:
    return set(yaml.safe_load((path or DATA_DIR / "function_words.yaml").read_text()))

def load_g2p_exceptions(path: Path | None = None) -> dict[str, str]:
    return yaml.safe_load((path or DATA_DIR / "g2p_exceptions.yaml").read_text())

class FileFrequencyList:
    def __init__(self, path: Path | None = None):
        lines = (path or DATA_DIR / "frequency_th.txt").read_text().splitlines()
        words = [w for w in lines if w and not w.startswith("#")]
        self._rank = {w: i + 1 for i, w in enumerate(words)}
    def rank(self, word: str) -> int | None:
        return self._rank.get(word)
```

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/test_data_io.py -v` → PASS.
- [ ] **Step 6: Commit** — message: `Add contrast inventory, spelling targets, and frequency data`

---

### Task 9: Linguistic rules

**Files:**
- Create: `src/thai_deck_eval/stages/linguistic.py`
- Test: `tests/test_linguistic.py`

**Interfaces:**
- Consumes: registry, ports, `parse_ipa`/`diff_features`, `analyze_syllable`, `load_g2p_exceptions`, fakes.
- Produces rules:
  - `lang/pair-not-minimal` (ERROR): for each minimal_pair note, g2p every member (exceptions dict first); if any member is multi-syllable or g2p-unknown → `lang/pair-unverifiable` (INFO) instead; else pairwise `diff_features` must equal exactly the declared contrast's feature set (`tone→{tone}`, `vowel_length→{length}`, `aspiration→{aspiration}`, `vowel_quality→{vowel}`, `consonant→{onset}`, `final→{coda}`).
  - `lang/ipa-mismatch` (ERROR): any authored `ipa` (pair members, picture words) must equal the g2p result (compare parsed `IpaSyllable` lists; skip + `lang/ipa-unverifiable` INFO when g2p unknown). When `ctx.g2p_second` is set and disagrees with `ctx.g2p`, demote to WARN with both outputs in evidence (consistency voting).
  - `lang/tone-mismatch` (ERROR): for single-syllable authored-IPA words where `analyze_syllable` parses, the tone-engine tone must equal the authored IPA tone.
  - `lang/dead-syllable-tone-contrast` (ERROR): a `tone` minimal pair whose members are dead syllables claiming tones outside {low, high, falling}.
  - `lang/target-not-token` (WARN): sentence `target` not in `tokenizer.tokens(thai)`.
  - `lang/frequency-rank-wrong` (WARN): `abs(declared - ref) > max(50, 0.2 * ref)` when `ctx.freq.rank(word)` is known; unknown word → `lang/frequency-unknown` (INFO).
  - All rules no-op (return) when the needed port on ctx is None.

- [ ] **Step 1: Write the failing tests**

`tests/test_linguistic.py`:

```python
import thai_deck_eval.stages.linguistic  # noqa: F401
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.model.deck import load_deck
from tests.fakes import FakeFreq, FakeG2P, FakeTokenizer
from tests.helpers import DeckBuilder

G2P = FakeG2P({
    "ขาว": "kʰaːw˨˩˦", "ข่าว": "kʰaːw˨˩", "ข้าว": "kʰaːw˥˩",
    "ไก่": "kaj˨˩", "ไข่": "kʰaj˨˩", "หมา": "maː˨˩˦", "มา": "maː˧",
    "กิน": "kin˧", "ใหม่": "maj˨˩", "ไม้": "maj˦˥",
})
TOK = FakeTokenizer({"หมามากินข้าว": ["หมา", "มา", "กิน", "ข้าว"]})
FREQ = FakeFreq({"หมา": 120, "มา": 15, "ข้าว": 90})

def _run(root, g2p=G2P, second=None):
    ctx = EvalContext(deck=load_deck(root), g2p=g2p, g2p_second=second,
                      tokenizer=TOK, freq=FREQ)
    return run_pipeline(ctx, stages=[Stage.LINGUISTIC])

def _rules(res):
    return sorted(f.rule for f in res.findings)

def test_golden_clean(tmp_path):
    assert _rules(_run(DeckBuilder(tmp_path).build())) == []

def test_two_feature_pair_rejected(tmp_path):
    b = DeckBuilder(tmp_path)
    # ใหม่/ไม้ differ in tone AND (per fake) nothing else here — craft a real
    # two-feature case: ข้าว (long) vs a short-vowel fake entry
    b.data["minimal_pairs"][0]["members"][1] = {
        "thai": "ไม้", "ipa": "maj˦˥",
        "audio": {"file": "audio/khao-l.mp3", "source": "native", "speaker": "s2"}}
    res = _run(b.build())
    assert "lang/pair-not-minimal" in _rules(res)

def test_unknown_word_is_unverifiable(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["minimal_pairs"][0]["members"][0]["thai"] = "เรือ"
    res = _run(b.build())
    assert "lang/pair-unverifiable" in _rules(res)
    assert "lang/pair-not-minimal" not in _rules(res)

def test_ipa_mismatch(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["ipa"] = "maː˧"  # หมา is rising, not mid
    res = _run(b.build())
    assert "lang/ipa-mismatch" in _rules(res)

def test_ipa_mismatch_demoted_when_engines_disagree(tmp_path):
    from thai_deck_eval.core.findings import Severity
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["ipa"] = "maː˧"
    second = FakeG2P({"หมา": "maː˧"})  # second engine agrees with the author
    res = _run(b.build(), second=second)
    f = next(f for f in res.findings if f.rule == "lang/ipa-mismatch")
    assert f.severity == Severity.WARN

def test_tone_mismatch_via_tone_engine(tmp_path):
    b = DeckBuilder(tmp_path)
    g2p = FakeG2P({**G2P.table, "หมา": "maː˧"})  # g2p wrong; tone engine says rising
    b.data["picture_words"][0]["ipa"] = "maː˧"
    res = _run(b.build(), g2p=g2p)
    assert "lang/tone-mismatch" in _rules(res)

def test_target_not_token(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["target"] = "มาก"  # substring of sentence, not a token
    b.data["sentences"][0]["thai"] = "หมามากินข้าว"
    res = _run(b.build())
    assert "lang/target-not-token" in _rules(res)

def test_frequency_rank_wrong(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["frequency_rank"] = 4000
    res = _run(b.build())
    assert "lang/frequency-rank-wrong" in _rules(res)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** `src/thai_deck_eval/stages/linguistic.py`:

```python
from itertools import combinations
from ..core.findings import Dimension, Severity, Stage
from ..core.registry import rule
from ..data_io import load_g2p_exceptions
from ..lang.ipa import IpaParseError, diff_features, parse_ipa
from ..lang.tone import analyze_syllable

_CONTRAST_FEATURE = {"tone": {"tone"}, "vowel_length": {"length"},
                     "aspiration": {"aspiration"}, "vowel_quality": {"vowel"},
                     "consonant": {"onset"}, "final": {"coda"}}

def _g2p(ctx, word):
    exc = load_g2p_exceptions()
    if word in exc:
        return parse_ipa(exc[word])
    return ctx.g2p.syllables(word)

@rule("lang/pair-not-minimal", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def pair_not_minimal(ctx):
    if ctx.g2p is None:
        return
    for note in ctx.deck.minimal_pairs:
        syls = [_g2p(ctx, m.thai) for m in note.members]
        if any(s is None or len(s) != 1 for s in syls):
            yield pair_not_minimal.rule_def.finding(
                "member unknown to g2p or multi-syllable; cannot verify",
                note_id=note.id, severity=Severity.INFO,
                evidence={"rule_override": "lang/pair-unverifiable"})
            continue
        want = _CONTRAST_FEATURE[note.contrast]
        for (i, a), (j, b) in combinations(enumerate(syls), 2):
            got = diff_features(a[0], b[0])
            if got != want:
                yield pair_not_minimal.finding(
                    f"members {note.members[i].thai}/{note.members[j].thai} "
                    f"differ in {sorted(got)}, declared contrast {note.contrast}",
                    note_id=note.id,
                    evidence={"diff": sorted(got), "declared": note.contrast})

def _authored_ipa(deck):
    for note in deck.minimal_pairs:
        for m in note.members:
            yield note.id, m.thai, m.ipa
    for note in deck.picture_words:
        if note.ipa:
            yield note.id, note.thai, note.ipa

@rule("lang/ipa-mismatch", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def ipa_mismatch(ctx):
    if ctx.g2p is None:
        return
    for note_id, word, authored in _authored_ipa(ctx.deck):
        try:
            claimed = parse_ipa(authored)
        except IpaParseError as e:
            yield ipa_mismatch.finding(f"unparseable ipa {authored!r}: {e}",
                                       note_id=note_id)
            continue
        got = _g2p(ctx, word)
        if got is None:
            yield ipa_mismatch.rule_def.finding(
                f"{word}: unknown to g2p", note_id=note_id,
                severity=Severity.INFO,
                evidence={"rule_override": "lang/ipa-unverifiable"})
            continue
        if got != claimed:
            severity = Severity.ERROR
            evidence = {"authored": authored, "g2p": [vars(s) for s in got]}
            if ctx.g2p_second is not None:
                second = ctx.g2p_second.syllables(word)
                if second is not None and second != got:
                    severity = Severity.WARN
                    evidence["g2p_second"] = [vars(s) for s in second]
            yield ipa_mismatch.rule_def.finding(
                f"{word}: authored IPA disagrees with g2p",
                note_id=note_id, severity=severity, evidence=evidence)

@rule("lang/tone-mismatch", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def tone_mismatch(ctx):
    for note_id, word, authored in _authored_ipa(ctx.deck):
        try:
            claimed = parse_ipa(authored)
        except IpaParseError:
            continue  # lang/ipa-mismatch reports it
        if len(claimed) != 1:
            continue
        analysis = analyze_syllable(word)
        if analysis is None:
            continue
        if analysis.tone != claimed[0].tone:
            yield tone_mismatch.finding(
                f"{word}: tone rules give {analysis.tone}, authored {claimed[0].tone}",
                note_id=note_id,
                evidence={"engine": str(analysis.tone), "authored": str(claimed[0].tone)})

@rule("lang/dead-syllable-tone-contrast", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def dead_syllable_tone(ctx):
    allowed = {"low", "high", "falling"}
    for note in ctx.deck.minimal_pairs:
        if note.contrast != "tone":
            continue
        for m in note.members:
            a = analyze_syllable(m.thai)
            if a is not None and not a.live and str(a.tone) not in allowed:
                yield dead_syllable_tone.finding(
                    f"{m.thai}: dead syllable cannot carry {a.tone}",
                    note_id=note.id)

@rule("lang/target-not-token", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
def target_not_token(ctx):
    if ctx.tokenizer is None:
        return
    for note in ctx.deck.sentences:
        toks = ctx.tokenizer.tokens(note.thai)
        if note.target not in toks:
            yield target_not_token.finding(
                f"target {note.target!r} is not a token of the sentence",
                note_id=note.id, evidence={"tokens": toks})

@rule("lang/frequency-rank-wrong", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
def frequency_rank_wrong(ctx):
    if ctx.freq is None:
        return
    for note in ctx.deck.picture_words:
        ref = ctx.freq.rank(note.thai)
        if ref is None:
            yield frequency_rank_wrong.rule_def.finding(
                f"{note.thai}: not in reference frequency list",
                note_id=note.id, severity=Severity.INFO,
                evidence={"rule_override": "lang/frequency-unknown"})
        elif abs(note.frequency_rank - ref) > max(50, 0.2 * ref):
            yield frequency_rank_wrong.finding(
                f"{note.thai}: declared rank {note.frequency_rank}, reference {ref}",
                note_id=note.id, evidence={"reference": ref})
```

Note the `rule_override` evidence convention: a rule that emits a *variant* finding (unverifiable/unknown) records the variant id in evidence rather than registering a separate rule. Report rendering (Task 15) displays `evidence["rule_override"]` as the rule id when present.

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add linguistic correctness rules`

---

### Task 10: pythainlp adapters (integration)

**Files:**
- Create: `src/thai_deck_eval/lang/pythainlp_adapter.py`
- Test: `tests/test_pythainlp_integration.py`

**Interfaces:**
- Consumes: `IpaSyllable`, `parse_ipa` conventions, `Tone`.
- Produces: `PyThaiNLPG2P` (engine `thaig2p`), `TltkG2P`, `PyThaiNLPTokenizer` — all implementing the ports; all imports of pythainlp/tltk inside methods/`__init__`, never module level. `PyThaiNLPG2P.syllables()` must convert the engine's raw output to `list[IpaSyllable]`, returning None when conversion fails (conversion failures must never raise).

- [ ] **Step 1: Discovery (exploratory, not committed as test)**

```bash
uv sync --extra nlp
uv run python -c "
from pythainlp.transliterate import transliterate
for w in ['ขาว','ข่าว','ไก่','ไข่','หมา','น้ำ']:
    print(w, repr(transliterate(w, engine='thaig2p')))
from pythainlp.tokenize import word_tokenize
print(word_tokenize('หมามากินข้าว'))
"
```

Record the actual output format in a comment atop the adapter. thaig2p emits IPA with tone digits or Chao letters and space/`.`-separated phones — the adapter maps whatever it actually emits onto `IpaSyllable`. Write the mapper against the observed output.

- [ ] **Step 2: Write the failing integration tests**

`tests/test_pythainlp_integration.py`:

```python
import pytest
from thai_deck_eval.lang.tone import Tone

pytestmark = pytest.mark.integration

@pytest.fixture(scope="module")
def g2p():
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPG2P
    return PyThaiNLPG2P()

@pytest.mark.parametrize("word,tone", [
    ("ข่าว", Tone.LOW), ("ข้าว", Tone.FALLING), ("ขาว", Tone.RISING),
    ("ไก่", Tone.LOW), ("มา", Tone.MID),
])
def test_g2p_tones(g2p, word, tone):
    syls = g2p.syllables(word)
    assert syls is not None and len(syls) == 1
    assert syls[0].tone == tone

def test_g2p_vowel_length(g2p):
    assert g2p.syllables("ขาว")[0].long is True

def test_g2p_unknown_returns_none_or_parses(g2p):
    assert g2p.syllables("ฟหกด") is None or True  # must not raise

def test_tokenizer():
    from thai_deck_eval.lang.pythainlp_adapter import PyThaiNLPTokenizer
    toks = PyThaiNLPTokenizer().tokens("หมามากินข้าว")
    assert "กิน" in toks
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest -m integration tests/test_pythainlp_integration.py -v` → FAIL.

- [ ] **Step 4: Implement** `src/thai_deck_eval/lang/pythainlp_adapter.py` — shape (mapper details depend on Step 1 observations):

```python
"""pythainlp/tltk adapters. Heavy imports stay inside methods.
Observed thaig2p output format (record here from Task 10 Step 1): …"""
from .ipa import IpaParseError, IpaSyllable
from .tone import Tone

_TONE_DIGITS = {"1": Tone.LOW, "2": Tone.FALLING, "3": Tone.HIGH,
                "4": Tone.RISING, "0": Tone.MID}  # adjust to observed format

class PyThaiNLPG2P:
    def __init__(self):
        from pythainlp.transliterate import transliterate
        self._t = transliterate

    def syllables(self, word: str) -> list[IpaSyllable] | None:
        try:
            raw = self._t(word, engine="thaig2p")
            return self._convert(raw)
        except Exception:
            return None

    def _convert(self, raw: str) -> list[IpaSyllable] | None:
        ...  # written against the observed format; return None if unmappable

class TltkG2P:
    def syllables(self, word: str) -> list[IpaSyllable] | None:
        try:
            from pythainlp.transliterate import transliterate
            return PyThaiNLPG2P._convert(self, transliterate(word, engine="tltk_ipa"))
        except Exception:
            return None

class PyThaiNLPTokenizer:
    def tokens(self, text: str) -> list[str]:
        from pythainlp.tokenize import word_tokenize
        return [t for t in word_tokenize(text) if t.strip()]
```

The `_convert` body is the real work of this task: split raw output into syllables, map segments/tone marks onto `IpaSyllable` fields, normalize aspirates (`tɕʰ` etc.) to the Task 7 inventories. Iterate until the integration tests pass. If thaig2p's tones genuinely disagree with a test word, cross-check against thai-language.com before changing an expectation, and add the word to `data/g2p_exceptions.yaml` if the engine (not the table) is wrong.

- [ ] **Step 5: Run to verify pass** — `uv run pytest -m integration -v` → PASS; `uv run pytest -v` (default) must still pass and must not import pythainlp (verify: `uv run python -c "import sys, thai_deck_eval.stages.linguistic; assert 'pythainlp' not in sys.modules"`).
- [ ] **Step 6: Commit** — message: `Add pythainlp and tltk adapters with integration tests`

---

### Task 11: Method fidelity rules and metrics

**Files:**
- Create: `src/thai_deck_eval/stages/method.py`
- Test: `tests/test_method.py`

**Interfaces:**
- Consumes: registry, data_io loaders, ports, tone engine, `Metric`.
- Produces rules (all Stage.METHOD, Dimension.METHOD):
  - `meth/pair-coverage` — Metric `coverage/minimal_pairs`: map each verified minimal_pair to a `ContrastEntry` id (tone pairs: sorted tone names of the two members from g2p, e.g. `tone:low-rising`; aspiration: place from onset — labial `p`, alveolar `t`, velar `k`, affricate `tɕ`; vowel_length → `vowel_length`; consonant with `ŋ` onset → `consonant:ng-onset`, `r`/`l` → `consonant:r-l`; vowel_quality by the vowel pair → `vowel_quality:e-ɛ` etc., fallback `vowel_quality:ɯ`/`ɤ` by membership; final → `final:unreleased`). value = Σ weight(covered ids)/Σ weight(all). detail lists `covered` and `missing` ids.
  - `meth/spelling-coverage` — Metric `coverage/spelling`: fraction over consonants+vowels+tone_marks targets covered by spelling_sound patterns (equal weight per symbol).
  - `meth/frequency-coverage` — Metric `coverage/frequency`: |{picture words with reference rank ≤ 625}| / 625 (skip when `ctx.freq` is None).
  - `meth/classifier-missing` (WARN): noun picture word with `classifier is None`.
  - `meth/tts-audio` (ERROR on minimal_pair members, WARN elsewhere): any `audio.source == "tts"`.
  - `meth/spelling-taper` (INFO): `test_spelling` true on a note with `frequency_rank > ctx.cfg("taper_rank", 300)`.
  - `meth/sentence-fanout` (WARN): >4 sentence notes sharing one `target`.
  - `meth/premature-sentences` (WARN): stage_plan includes "sentences" and `len(picture_words) < ctx.cfg("sentence_base", 300)` and `len(sentences) > 0`. Suppressed when "words" not in stage_plan phases.
  - `meth/new-elements` (WARN): per sentence in file order — known = all picture-word `thai` + targets of earlier sentences + function words; unknown non-target tokens > 0 → finding listing them. Skip when tokenizer None.
  - `meth/speaker-diversity` — Metric `speakers/minimal_pairs`: `min(1.0, distinct_speakers / ctx.cfg("target_speakers", 3))`.
  - `meth/no-personal-connection` (INFO): picture word with `personal_connection is None` — a user-editable slot generated decks can't fill; info only, never scored deductions (info deduction weight is 0).

- [ ] **Step 1: Write the failing tests**

`tests/test_method.py`:

```python
import thai_deck_eval.stages.method  # noqa: F401
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.model.deck import load_deck
from tests.fakes import FakeFreq, FakeG2P, FakeTokenizer
from tests.helpers import DeckBuilder
from tests.test_linguistic import FREQ, G2P, TOK

def _run(root, **ctx_kw):
    kw = dict(g2p=G2P, tokenizer=TOK, freq=FREQ)
    kw.update(ctx_kw)
    return run_pipeline(EvalContext(deck=load_deck(root), config={"sentence_base": 2},
                                    **kw), stages=[Stage.METHOD])

def _metric(res, name):
    return next(m for m in res.metrics if m.name == name)

def test_pair_coverage_metric(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    m = _metric(res, "coverage/minimal_pairs")
    assert 0 < m.value < 1
    assert "tone:low-rising" in m.detail["covered"]      # ขาว(rising)/ข่าว(low)
    assert "aspiration:velar" in m.detail["covered"]     # ไก่/ไข่
    assert "tone:mid-low" in m.detail["missing"]

def test_spelling_coverage_metric(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    m = _metric(res, "coverage/spelling")
    assert 0 < m.value < 0.1  # 1 of ~69 targets

def test_classifier_missing(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["classifier"] = None
    res = _run(b.build())
    assert any(f.rule == "meth/classifier-missing" for f in res.findings)

def test_tts_on_pair_is_error(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["minimal_pairs"][0]["members"][0]["audio"]["source"] = "tts"
    res = _run(b.build())
    f = next(f for f in res.findings if f.rule == "meth/tts-audio")
    assert f.severity == Severity.ERROR

def test_tts_on_picture_word_is_warn(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][0]["audio"]["source"] = "tts"
    res = _run(b.build())
    f = next(f for f in res.findings if f.rule == "meth/tts-audio")
    assert f.severity == Severity.WARN

def test_spelling_taper(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["picture_words"][2]["test_spelling"] = True   # rank 90 → ok
    b.data["picture_words"][0]["frequency_rank"] = 400
    b.data["picture_words"][0]["test_spelling"] = True   # rank 400 → info
    res = _run(b.build())
    hits = [f for f in res.findings if f.rule == "meth/spelling-taper"]
    assert [f.note_id for f in hits] == ["w-dog"]

def test_premature_sentences(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())   # sentence_base=2, 3 words → ok
    assert not any(f.rule == "meth/premature-sentences" for f in res.findings)
    b = DeckBuilder(tmp_path / "b")
    b.data["picture_words"] = b.data["picture_words"][:1]
    res = _run(b.build())
    assert any(f.rule == "meth/premature-sentences" for f in res.findings)

def test_new_elements(tmp_path):
    b = DeckBuilder(tmp_path)
    b.data["sentences"][0]["thai"] = "หมาวิ่งกิน"
    b.data["sentences"][0]["target"] = "กิน"
    tok = FakeTokenizer({"หมาวิ่งกิน": ["หมา", "วิ่ง", "กิน"]})
    res = _run(b.build(), tokenizer=tok)
    f = next(f for f in res.findings if f.rule == "meth/new-elements")
    assert f.evidence["unknown"] == ["วิ่ง"]

def test_speaker_diversity(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())
    assert _metric(res, "speakers/minimal_pairs").value == 1.0  # s1,s2,s3 / 3

def test_no_personal_connection_is_info(tmp_path):
    res = _run(DeckBuilder(tmp_path).build())  # golden has none filled
    hits = [f for f in res.findings if f.rule == "meth/no-personal-connection"]
    assert len(hits) == 3 and all(f.severity == Severity.INFO for f in hits)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** `src/thai_deck_eval/stages/method.py`:

```python
from collections import Counter
from ..core.findings import Dimension, Metric, Severity, Stage
from ..core.registry import rule
from ..data_io import load_contrasts, load_function_words, load_spelling_targets
from ..stages.linguistic import _g2p

_PLACE = {"p": "labial", "t": "alveolar", "k": "velar", "tɕ": "affricate"}

def _contrast_id(note, syls) -> str | None:
    a, b = syls[0][0], syls[1][0]
    if note.contrast == "tone":
        return "tone:" + "-".join(sorted([str(a.tone), str(b.tone)],
                                         key=["mid","low","falling","high","rising"].index))
    if note.contrast == "aspiration":
        place = _PLACE.get(a.onset.replace("ʰ", ""))
        return f"aspiration:{place}" if place else None
    if note.contrast == "vowel_length":
        return "vowel_length"
    if note.contrast == "consonant":
        if "ŋ" in (a.onset, b.onset):
            return "consonant:ng-onset"
        if {a.onset, b.onset} == {"r", "l"}:
            return "consonant:r-l"
        return None
    if note.contrast == "vowel_quality":
        pair = {a.vowel, b.vowel}
        for cid, vs in [("vowel_quality:e-ɛ", {"e", "ɛ"}),
                        ("vowel_quality:o-ɔ", {"o", "ɔ"})]:
            if pair == vs:
                return cid
        if "ɯ" in pair:
            return "vowel_quality:ɯ"
        if "ɤ" in pair:
            return "vowel_quality:ɤ"
        return None
    if note.contrast == "final":
        return "final:unreleased"
    return None

@rule("meth/pair-coverage", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def pair_coverage(ctx):
    if ctx.g2p is None:
        return
    entries = load_contrasts()
    covered: set[str] = set()
    for note in ctx.deck.minimal_pairs:
        syls = [_g2p(ctx, m.thai) for m in note.members]
        if any(s is None or len(s) != 1 for s in syls):
            continue
        cid = _contrast_id(note, syls)
        if cid:
            covered.add(cid)
    total = sum(e.weight for e in entries)
    got = sum(e.weight for e in entries if e.id in covered)
    yield Metric(name="coverage/minimal_pairs", value=got / total,
                 detail={"covered": sorted(covered),
                         "missing": sorted(e.id for e in entries
                                           if e.id not in covered)})

@rule("meth/spelling-coverage", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def spelling_coverage(ctx):
    targets = load_spelling_targets()
    all_syms = [s for group in targets.values() for s in group]
    covered = {n.pattern for n in ctx.deck.spelling_sound}
    got = sum(1 for s in all_syms if s in covered or s.strip("-") in covered)
    yield Metric(name="coverage/spelling", value=got / len(all_syms),
                 detail={"total": len(all_syms), "covered": got})

@rule("meth/frequency-coverage", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def frequency_coverage(ctx):
    if ctx.freq is None:
        return
    n = sum(1 for w in ctx.deck.picture_words
            if (r := ctx.freq.rank(w.thai)) is not None and r <= 625)
    yield Metric(name="coverage/frequency", value=n / 625)

@rule("meth/classifier-missing", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def classifier_missing(ctx):
    for note in ctx.deck.picture_words:
        if note.part_of_speech == "noun" and note.classifier is None:
            yield classifier_missing.finding("noun without classifier",
                                             note_id=note.id)

@rule("meth/tts-audio", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def tts_audio(ctx):
    for note in ctx.deck.minimal_pairs:
        for m in note.members:
            if m.audio.source == "tts":
                yield tts_audio.rule_def.finding(
                    f"TTS audio on minimal pair member {m.thai}",
                    note_id=note.id, severity=Severity.ERROR)
    others = ([(n, n.audio) for n in ctx.deck.spelling_sound]
              + [(n, n.audio) for n in ctx.deck.picture_words]
              + [(n, n.audio) for n in ctx.deck.sentences])
    for note, audio in others:
        if audio.source == "tts":
            yield tts_audio.finding("TTS audio on tone-bearing card",
                                    note_id=note.id)

@rule("meth/spelling-taper", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def spelling_taper(ctx):
    taper = ctx.cfg("taper_rank", 300)
    for note in ctx.deck.picture_words:
        if note.test_spelling and note.frequency_rank > taper:
            yield spelling_taper.finding(
                f"spelling card beyond taper rank {taper}", note_id=note.id)

@rule("meth/sentence-fanout", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def sentence_fanout(ctx):
    counts = Counter(n.target for n in ctx.deck.sentences)
    for target, c in counts.items():
        if c > 4:
            yield sentence_fanout.finding(
                f"{c} sentence notes for target {target!r} (max 4)")

@rule("meth/premature-sentences", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def premature_sentences(ctx):
    plan = ctx.deck.meta.stage_plan.phases
    if "words" not in plan or "sentences" not in plan:
        return
    base = ctx.cfg("sentence_base", 300)
    if ctx.deck.sentences and len(ctx.deck.picture_words) < base:
        yield premature_sentences.finding(
            f"{len(ctx.deck.sentences)} sentences atop only "
            f"{len(ctx.deck.picture_words)} picture words (base {base})")

@rule("meth/new-elements", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def new_elements(ctx):
    if ctx.tokenizer is None:
        return
    known = {w.thai for w in ctx.deck.picture_words} | load_function_words()
    for note in ctx.deck.sentences:
        toks = ctx.tokenizer.tokens(note.thai)
        unknown = [t for t in toks if t not in known and t != note.target]
        if unknown:
            yield new_elements.finding(
                f"sentence introduces {len(unknown)} unknown non-target tokens",
                note_id=note.id, evidence={"unknown": unknown})
        known.add(note.target)

@rule("meth/no-personal-connection", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def no_personal_connection(ctx):
    for note in ctx.deck.picture_words:
        if note.personal_connection is None:
            yield no_personal_connection.finding(
                "personal-connection slot empty (fill by hand)", note_id=note.id)

@rule("meth/speaker-diversity", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def speaker_diversity(ctx):
    speakers = {m.audio.speaker for n in ctx.deck.minimal_pairs for m in n.members}
    if not ctx.deck.minimal_pairs:
        return
    target = ctx.cfg("target_speakers", 3)
    yield Metric(name="speakers/minimal_pairs",
                 value=min(1.0, len(speakers) / target),
                 detail={"distinct": len(speakers)})
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add method fidelity rules and coverage metrics`

---

### Task 12: Rulebook config and scoring

**Files:**
- Create: `src/thai_deck_eval/config.py`, `src/thai_deck_eval/report/__init__.py`, `src/thai_deck_eval/report/scoring.py`, `rulebook.yaml`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `Finding`, `Metric`, `EvalResult`.
- Produces: `RulebookConfig` (pydantic, `extra="forbid"`): fields `version: str = "1"`, `gates: bool = True`, `taper_rank: int = 300`, `sentence_base: int = 300`, `target_speakers: int = 3`, `deductions: dict[str, float] = {"error": 25, "warn": 2, "info": 0}`, `metric_weights: dict[str, float] = {"coverage/minimal_pairs": 3, "coverage/spelling": 2, "coverage/frequency": 3, "speakers/minimal_pairs": 1}`, `judge: JudgeConfig` where `JudgeConfig(backend: Literal["cli","api","fake"] = "cli", model: str = "claude-opus-5", effort: str = "medium", confidence_floor: float = 0.6, prompt_version: str = "1", cache_path: str = ".thai-deck-eval-cache.sqlite")`. `load_rulebook(path: Path | None) -> RulebookConfig` (None → defaults). `EvalContext.cfg` already reads attributes off it.
- Produces: `Scores` pydantic model `{integrity, language, method, content: float}`; `compute_scores(result: EvalResult, config: RulebookConfig) -> Scores` — integrity/language/content: `max(0, 100 - Σ deductions[f.severity])` over that dimension's findings; method: `100 * (Σ w·metric / Σ w over metrics present) - Σ deductions` clamped to [0, 100]; no metrics present → method = 0.0.

- [ ] **Step 1: Write the failing tests**

`tests/test_scoring.py`:

```python
from thai_deck_eval.config import RulebookConfig, load_rulebook
from thai_deck_eval.core.findings import Dimension, Finding, Metric, Severity
from thai_deck_eval.core.pipeline import EvalResult
from thai_deck_eval.report.scoring import compute_scores

def _f(dim, sev):
    return Finding(rule="x/y", severity=sev, dimension=dim, message="m")

def test_defaults_load():
    cfg = load_rulebook(None)
    assert cfg.judge.backend == "cli" and cfg.gates is True

def test_deductions():
    res = EvalResult(findings=[_f(Dimension.INTEGRITY, Severity.WARN),
                               _f(Dimension.INTEGRITY, Severity.ERROR),
                               _f(Dimension.LANGUAGE, Severity.INFO)])
    s = compute_scores(res, RulebookConfig())
    assert s.integrity == 73 and s.language == 100

def test_method_blend():
    res = EvalResult(metrics=[Metric(name="coverage/minimal_pairs", value=0.5),
                              Metric(name="coverage/frequency", value=1.0)],
                     findings=[_f(Dimension.METHOD, Severity.WARN)])
    s = compute_scores(res, RulebookConfig())
    # (3*0.5 + 3*1.0) / 6 = 0.75 → 75 - 2 = 73
    assert s.method == 73

def test_method_zero_without_metrics():
    assert compute_scores(EvalResult(), RulebookConfig()).method == 0.0

def test_rulebook_file(tmp_path):
    p = tmp_path / "rb.yaml"
    p.write_text("taper_rank: 100\njudge:\n  backend: fake\n")
    cfg = load_rulebook(p)
    assert cfg.taper_rank == 100 and cfg.judge.backend == "fake"
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement**

`src/thai_deck_eval/config.py`:

```python
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field

class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["cli", "api", "fake"] = "cli"
    model: str = "claude-opus-5"
    effort: str = "medium"
    confidence_floor: float = 0.6
    prompt_version: str = "1"
    cache_path: str = ".thai-deck-eval-cache.sqlite"

class RulebookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "1"
    gates: bool = True
    taper_rank: int = 300
    sentence_base: int = 300
    target_speakers: int = 3
    deductions: dict[str, float] = Field(
        default_factory=lambda: {"error": 25.0, "warn": 2.0, "info": 0.0})
    metric_weights: dict[str, float] = Field(
        default_factory=lambda: {"coverage/minimal_pairs": 3.0,
                                 "coverage/spelling": 2.0,
                                 "coverage/frequency": 3.0,
                                 "speakers/minimal_pairs": 1.0})
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

def load_rulebook(path: Path | None) -> RulebookConfig:
    if path is None:
        return RulebookConfig()
    return RulebookConfig.model_validate(yaml.safe_load(Path(path).read_text()) or {})
```

`src/thai_deck_eval/report/scoring.py`:

```python
from pydantic import BaseModel
from ..config import RulebookConfig
from ..core.findings import Dimension
from ..core.pipeline import EvalResult

class Scores(BaseModel):
    integrity: float
    language: float
    method: float
    content: float

def _deducted(result, cfg, dim) -> float:
    return sum(cfg.deductions.get(str(f.severity), 0)
               for f in result.findings if f.dimension == dim)

def compute_scores(result: EvalResult, cfg: RulebookConfig) -> Scores:
    def simple(dim):
        return max(0.0, 100.0 - _deducted(result, cfg, dim))
    weights = {m.name: cfg.metric_weights.get(m.name, 1.0) for m in result.metrics}
    if weights:
        blend = sum(cfg.metric_weights.get(m.name, 1.0) * m.value
                    for m in result.metrics) / sum(weights.values())
        method = min(100.0, max(0.0, 100.0 * blend
                                - _deducted(result, cfg, Dimension.METHOD)))
    else:
        method = 0.0
    return Scores(integrity=simple(Dimension.INTEGRITY),
                  language=simple(Dimension.LANGUAGE),
                  method=method,
                  content=simple(Dimension.CONTENT))
```

Also commit a `rulebook.yaml` at repo root containing only the defaults rendered as documented YAML (every field with its default and a one-line comment) so users have an editable template.

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add rulebook config and dimension scoring`

---

### Task 13: Judge core — port, cache, fake, judge rules

**Files:**
- Create: `src/thai_deck_eval/judge/__init__.py`, `src/thai_deck_eval/judge/core.py`, `src/thai_deck_eval/judge/prompts.py`, `src/thai_deck_eval/stages/judge_rules.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: notes, registry, `JudgeConfig`.
- Produces:
  - `Verdict(rule: str, passed: bool, confidence: float, rationale: str)` (pydantic); `Verdicts(verdicts: list[Verdict])`.
  - `JudgeRequest(note_id: str, rules: list[str], prompt: str, image_path: str | None = None)`.
  - `class Judge(Protocol): def judge(self, req: JudgeRequest) -> list[Verdict]`.
  - `FakeJudge(verdicts: dict[str, list[Verdict]])` keyed by note_id; unknown note → all-pass with confidence 1.0 for `req.rules`.
  - `CachedJudge(inner: Judge, db_path: Path, model: str, prompt_version: str)` — SQLite table `verdicts(key TEXT PRIMARY KEY, payload TEXT)`; key = sha256 of JSON `[sorted(rules), prompt_version, model, prompt, image_sha or None]`; `calls` counter attribute for tests.
  - `prompts.py`: `PROMPT_VERSION = "1"`; `build_sentence_prompt(note) -> str`, `build_picture_prompt(note) -> str` — each instructs: return ONLY JSON matching `{"verdicts": [{"rule": …, "passed": bool, "confidence": 0-1, "rationale": …}]}` and enumerates the rules to judge with one-line rubrics (sentence: `judge/unnatural-sentence` "is this natural, correct Thai a native would produce?", `judge/definition-not-monolingual` "definition, if present, is Thai-only and accurate", `judge/gloss-inaccurate` "gloss, if present, correctly translates the target"; picture: `judge/image-irrelevant` "image plausibly depicts or relates to the word", `judge/image-embedded-text` "image contains no English/romanized text", `judge/classifier-wrong` "classifier, if present, is the conventional one").
  - `stages/judge_rules.py`: one registered rule per judge rule id (Stage.JUDGE, Dimension.CONTENT, WARN default; `judge/unnatural-sentence` ERROR default). A module-level orchestrator function `run_judge(ctx) -> dict[str, list[Verdict]]` memoized on ctx (attribute `_judge_verdicts`) builds one JudgeRequest per sentence note and per picture word, calls `ctx.judge.judge(...)`, and each rule then yields findings for its own id where `passed is False`; confidence `< ctx.config.judge.confidence_floor` demotes to INFO. All judge rules no-op when `ctx.judge` is None.

- [ ] **Step 1: Write the failing tests**

`tests/test_judge.py`:

```python
import thai_deck_eval.stages.judge_rules  # noqa: F401
from thai_deck_eval.config import RulebookConfig
from thai_deck_eval.core.context import EvalContext
from thai_deck_eval.core.findings import Severity, Stage
from thai_deck_eval.core.pipeline import run_pipeline
from thai_deck_eval.judge.core import CachedJudge, FakeJudge, JudgeRequest, Verdict
from thai_deck_eval.model.deck import load_deck
from tests.helpers import DeckBuilder

def _ctx(root, judge):
    return EvalContext(deck=load_deck(root), config=RulebookConfig(), judge=judge)

def test_all_pass_yields_nothing(tmp_path):
    res = run_pipeline(_ctx(DeckBuilder(tmp_path).build(), FakeJudge({})),
                       stages=[Stage.JUDGE])
    assert res.findings == []

def test_failed_verdict_becomes_finding(tmp_path):
    judge = FakeJudge({"s-1": [Verdict(rule="judge/unnatural-sentence",
                                       passed=False, confidence=0.9,
                                       rationale="word order is English-like")]})
    res = run_pipeline(_ctx(DeckBuilder(tmp_path).build(), judge),
                       stages=[Stage.JUDGE])
    f = next(f for f in res.findings if f.rule == "judge/unnatural-sentence")
    assert f.note_id == "s-1" and f.severity == Severity.ERROR
    assert "English-like" in f.message

def test_low_confidence_demoted_to_info(tmp_path):
    judge = FakeJudge({"s-1": [Verdict(rule="judge/unnatural-sentence",
                                       passed=False, confidence=0.3,
                                       rationale="maybe")]})
    res = run_pipeline(_ctx(DeckBuilder(tmp_path).build(), judge),
                       stages=[Stage.JUDGE])
    f = next(f for f in res.findings if f.rule == "judge/unnatural-sentence")
    assert f.severity == Severity.INFO

def test_cache_hits_skip_inner(tmp_path):
    class Counting:
        calls = 0
        def judge(self, req):
            Counting.calls += 1
            return [Verdict(rule=r, passed=True, confidence=1.0, rationale="")
                    for r in req.rules]
    cached = CachedJudge(Counting(), tmp_path / "c.sqlite", "m", "1")
    req = JudgeRequest(note_id="n", rules=["judge/x"], prompt="p")
    a = cached.judge(req)
    b = cached.judge(req)
    assert a == b and Counting.calls == 1
    cached2 = CachedJudge(Counting(), tmp_path / "c.sqlite", "m", "1")
    assert cached2.judge(req) == a and Counting.calls == 1  # persists
    cached3 = CachedJudge(Counting(), tmp_path / "c.sqlite", "m", "2")
    cached3.judge(req)
    assert Counting.calls == 2  # prompt_version bump invalidates
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement**

`src/thai_deck_eval/judge/core.py`:

```python
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from pydantic import BaseModel

class Verdict(BaseModel):
    rule: str
    passed: bool
    confidence: float
    rationale: str

class Verdicts(BaseModel):
    verdicts: list[Verdict]

@dataclass
class JudgeRequest:
    note_id: str
    rules: list[str]
    prompt: str
    image_path: str | None = None

class Judge(Protocol):
    def judge(self, req: JudgeRequest) -> list[Verdict]: ...

class FakeJudge:
    def __init__(self, verdicts: dict[str, list[Verdict]]):
        self._v = verdicts
    def judge(self, req: JudgeRequest) -> list[Verdict]:
        if req.note_id in self._v:
            return self._v[req.note_id]
        return [Verdict(rule=r, passed=True, confidence=1.0, rationale="")
                for r in req.rules]

class CachedJudge:
    def __init__(self, inner: Judge, db_path: Path, model: str, prompt_version: str):
        self.inner, self.model, self.prompt_version = inner, model, prompt_version
        self.calls = 0
        self._db = sqlite3.connect(db_path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, payload TEXT)")

    def _key(self, req: JudgeRequest) -> str:
        image_sha = None
        if req.image_path and Path(req.image_path).is_file():
            image_sha = hashlib.sha256(Path(req.image_path).read_bytes()).hexdigest()
        blob = json.dumps([sorted(req.rules), self.prompt_version, self.model,
                           req.prompt, image_sha], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def judge(self, req: JudgeRequest) -> list[Verdict]:
        key = self._key(req)
        row = self._db.execute("SELECT payload FROM verdicts WHERE key=?",
                               (key,)).fetchone()
        if row:
            return Verdicts.model_validate_json(row[0]).verdicts
        out = self.inner.judge(req)
        self.calls += 1
        self._db.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?)",
                         (key, Verdicts(verdicts=out).model_dump_json()))
        self._db.commit()
        return out
```

`src/thai_deck_eval/judge/prompts.py`:

```python
PROMPT_VERSION = "1"

_SCHEMA = ('Return ONLY JSON: {"verdicts": [{"rule": "<id>", "passed": true|false, '
           '"confidence": 0.0-1.0, "rationale": "<one sentence>"}]} — one entry '
           "per rule listed below.")

SENTENCE_RULES = {
    "judge/unnatural-sentence":
        "Is the Thai sentence natural and grammatically correct — something a "
        "native speaker would actually produce?",
    "judge/definition-not-monolingual":
        "If a definition is given, is it entirely in Thai and accurate for the "
        "target word? Pass if no definition.",
    "judge/gloss-inaccurate":
        "If an English gloss is given, does it correctly translate the target? "
        "Pass if no gloss.",
}

PICTURE_RULES = {
    "judge/image-irrelevant":
        "Does the image plausibly depict or relate to the word?",
    "judge/image-embedded-text":
        "Pass only if the image contains NO English or romanized-Thai text.",
    "judge/classifier-wrong":
        "If a classifier is given, is it the conventional classifier for this "
        "noun? Pass if none given.",
}

def _rules_block(rules: dict[str, str]) -> str:
    return "\n".join(f"- {rid}: {rubric}" for rid, rubric in rules.items())

def build_sentence_prompt(note) -> str:
    return (f"You are evaluating a Thai flashcard for a Fluent Forever deck.\n"
            f"Sentence: {note.thai}\nTarget word: {note.target}\n"
            f"Definition: {note.definition or '(none)'}\n"
            f"Gloss: {note.gloss or '(none)'}\n\nJudge these rules:\n"
            f"{_rules_block(SENTENCE_RULES)}\n\n{_SCHEMA}")

def build_picture_prompt(note) -> str:
    return (f"You are evaluating a Thai picture-word flashcard (image attached "
            f"or at path).\nWord: {note.thai}\nCategory: {note.category}\n"
            f"Part of speech: {note.part_of_speech}\n"
            f"Classifier: {note.classifier or '(none)'}\n\nJudge these rules:\n"
            f"{_rules_block(PICTURE_RULES)}\n\n{_SCHEMA}")
```

`src/thai_deck_eval/stages/judge_rules.py`:

```python
from ..core.findings import Dimension, Severity, Stage
from ..core.registry import rule
from ..judge.core import JudgeRequest
from ..judge.prompts import (PICTURE_RULES, SENTENCE_RULES,
                             build_picture_prompt, build_sentence_prompt)

def _verdicts(ctx):
    if getattr(ctx, "_judge_verdicts", None) is not None:
        return ctx._judge_verdicts
    out: dict[str, list] = {}
    for note in ctx.deck.sentences:
        req = JudgeRequest(note_id=note.id, rules=list(SENTENCE_RULES),
                           prompt=build_sentence_prompt(note))
        out[note.id] = ctx.judge.judge(req)
    for note in ctx.deck.picture_words:
        req = JudgeRequest(note_id=note.id, rules=list(PICTURE_RULES),
                           prompt=build_picture_prompt(note),
                           image_path=str(ctx.deck.root / "media" / note.image))
        out[note.id] = ctx.judge.judge(req)
    ctx._judge_verdicts = out
    return out

def _findings_for(rule_fn, ctx, rule_id):
    if ctx.judge is None:
        return
    floor = ctx.config.judge.confidence_floor
    for note_id, verdicts in _verdicts(ctx).items():
        for v in verdicts:
            if v.rule == rule_id and not v.passed:
                sev = Severity.INFO if v.confidence < floor else None
                yield rule_fn.rule_def.finding(
                    v.rationale or "judge failed this rule", note_id=note_id,
                    severity=sev, evidence={"confidence": v.confidence})

def _make(rule_id, default_severity):
    @rule(rule_id, Stage.JUDGE, Dimension.CONTENT, default_severity)
    def fn(ctx, _rid=rule_id):
        yield from _findings_for(fn, ctx, _rid)
    return fn

unnatural = _make("judge/unnatural-sentence", Severity.ERROR)
definition = _make("judge/definition-not-monolingual", Severity.WARN)
gloss = _make("judge/gloss-inaccurate", Severity.WARN)
image_irrelevant = _make("judge/image-irrelevant", Severity.WARN)
image_text = _make("judge/image-embedded-text", Severity.WARN)
classifier = _make("judge/classifier-wrong", Severity.WARN)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS.
- [ ] **Step 5: Commit** — message: `Add judge port, verdict cache, and judge rules`

---

### Task 14: Cli and Api judge backends

**Files:**
- Create: `src/thai_deck_eval/judge/cli_judge.py`, `src/thai_deck_eval/judge/api_judge.py`
- Test: `tests/test_judge_backends.py` (unit, subprocess faked), `tests/test_judge_live.py` (marked `live`)

**Interfaces:**
- Consumes: `Judge` protocol, `Verdicts`, `JudgeConfig`.
- Produces: `CliJudge(config: JudgeConfig, runner=subprocess.run)` — builds `["claude", "-p", prompt, "--output-format", "json"]`, adding `--allowedTools Read --add-dir <image dir>` and a prompt line `Image file to inspect: <path>` when `image_path` is set; parses stdout JSON envelope (`{"result": "<text>"}`), extracts the first `{...}` block from the result text, validates with `Verdicts`; one retry appending `"\nReturn ONLY the JSON object."` to the prompt on parse failure; raises `JudgeError` after retry. `ApiJudge(config: JudgeConfig)` — anthropic SDK; **before implementing, invoke the `claude-api` skill and read its `python/claude-api/README.md` for current `messages.parse`/structured-output and vision-block syntax — do not write SDK calls from memory**; model from config, `output_config` effort from config, image as base64 vision block; check `stop_reason` and raise `JudgeError` on `refusal`. `JudgeError(Exception)` in `cli_judge.py`, re-exported from `judge/__init__.py`.

- [ ] **Step 1: Write the failing unit tests**

`tests/test_judge_backends.py`:

```python
import json
import pytest
from thai_deck_eval.config import JudgeConfig
from thai_deck_eval.judge.cli_judge import CliJudge, JudgeError
from thai_deck_eval.judge.core import JudgeRequest

GOOD = json.dumps({"result": json.dumps({"verdicts": [
    {"rule": "judge/x", "passed": True, "confidence": 0.9, "rationale": "ok"}]})})

class FakeRun:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.cmds = []
    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        class R:
            returncode = 0
            stdout = self.outputs.pop(0)
            stderr = ""
        return R()

def test_cli_judge_parses():
    runner = FakeRun([GOOD])
    j = CliJudge(JudgeConfig(), runner=runner)
    out = j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))
    assert out[0].passed is True
    assert runner.cmds[0][:2] == ["claude", "-p"]

def test_cli_judge_retries_then_raises():
    runner = FakeRun([json.dumps({"result": "not json"}),
                      json.dumps({"result": "still not"})])
    j = CliJudge(JudgeConfig(), runner=runner)
    with pytest.raises(JudgeError):
        j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p"))
    assert len(runner.cmds) == 2

def test_cli_judge_image_adds_read_tool():
    runner = FakeRun([GOOD])
    j = CliJudge(JudgeConfig(), runner=runner)
    j.judge(JudgeRequest(note_id="n", rules=["judge/x"], prompt="p",
                         image_path="/tmp/x/img.png"))
    cmd = runner.cmds[0]
    assert "--allowedTools" in cmd and "Read" in cmd
    assert any("img.png" in part for part in cmd)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement** `src/thai_deck_eval/judge/cli_judge.py`:

```python
import json
import re
import subprocess
from .core import JudgeRequest, Verdict, Verdicts
from ..config import JudgeConfig

class JudgeError(Exception):
    pass

class CliJudge:
    def __init__(self, config: JudgeConfig, runner=subprocess.run):
        self.config = config
        self.runner = runner

    def judge(self, req: JudgeRequest) -> list[Verdict]:
        prompt = req.prompt
        if req.image_path:
            prompt += f"\nImage file to inspect: {req.image_path}"
        for attempt in range(2):
            cmd = ["claude", "-p", prompt, "--output-format", "json"]
            if req.image_path:
                from pathlib import Path
                cmd += ["--allowedTools", "Read",
                        "--add-dir", str(Path(req.image_path).parent)]
            r = self.runner(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise JudgeError(f"claude -p failed: {r.stderr[:500]}")
            try:
                text = json.loads(r.stdout)["result"]
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    raise ValueError("no JSON object in result")
                return Verdicts.model_validate_json(m.group(0)).verdicts
            except Exception:
                prompt += "\nReturn ONLY the JSON object."
        raise JudgeError(f"unparseable judge output for note {req.note_id}")
```

`api_judge.py`: implement per the claude-api skill's Python README (invoke the skill first). Shape: client from `anthropic`, `messages.parse`-style structured output against `Verdicts`, `model=config.model`, `output_config={"effort": config.effort}`, image as base64 `image` content block when `req.image_path` is set, raise `JudgeError` on `stop_reason == "refusal"` or validation failure.

`tests/test_judge_live.py` (`pytestmark = pytest.mark.live`): one test per backend judging the golden sentence note prompt, asserting a `Verdict` list comes back with all rules present. Run manually: `uv run pytest -m live -v`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → PASS (default excludes `live`).
- [ ] **Step 5: Commit** — message: `Add CLI and API judge backends`

---

### Task 15: CLI entry point and report rendering

**Files:**
- Create: `src/thai_deck_eval/report/model.py`, `src/thai_deck_eval/report/render.py`, `src/thai_deck_eval/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything.
- Produces: `Report` pydantic model `{deck_name, deck_version, rulebook_version, stages_run: list[str], stages_skipped: list[str], findings: list[Finding], metrics: list[Metric], scores: Scores, gate: Literal["pass","fail"]}`; `build_report(deck_meta, result, scores, config) -> Report`; `render_text(report) -> str` (summary header with scores + gate, findings grouped by severity showing `evidence.get("rule_override", finding.rule)`, metric table); `cli.main` (click): argument `deck_dir`; options `--report PATH` (write JSON), `--format [text|json]` default text, `--no-judge`, `--stages TEXT` (comma-separated stage names), `--rulebook PATH`. Context assembly: judge from `config.judge.backend` (`cli` → `CachedJudge(CliJudge(...))`, `api` → `CachedJudge(ApiJudge(...))`, `fake` → `FakeJudge({})`); g2p/tokenizer/freq: real adapters + `FileFrequencyList`, constructed lazily inside `main` guarded by try/ImportError → warn to stderr and pass None (linguistic rules then no-op). Exit codes: 0 pass, 1 gate fail, 2 evaluator exception (click catches and prints).

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
import json
from click.testing import CliRunner
from thai_deck_eval.cli import main
from tests.helpers import DeckBuilder

def _invoke(root, *args):
    return CliRunner().invoke(
        main, [str(root), "--no-judge", "--rulebook", "/dev/null", *args])

def test_golden_passes(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    r = _invoke(DeckBuilder(tmp_path).build(), "--format", "json")
    assert r.exit_code == 0, r.output
    rep = json.loads(r.output)
    assert rep["gate"] == "pass" and "scores" in rep

def test_gate_failure_exit_1(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    root = DeckBuilder(tmp_path).build()
    (root / "media" / "images" / "dog.png").unlink()
    r = _invoke(root)
    assert r.exit_code == 1
    assert "mech/media-missing" in r.output

def test_schema_error_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    root = DeckBuilder(tmp_path).build()
    (root / "deck.yaml").write_text("name: [broken")
    r = _invoke(root)
    assert r.exit_code == 1 and "schema/invalid" in r.output

def test_report_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    out = tmp_path / "rep.json"
    r = _invoke(DeckBuilder(tmp_path).build(), "--report", str(out))
    assert r.exit_code == 0 and json.loads(out.read_text())["gate"] == "pass"
```

Note `--rulebook /dev/null` loads defaults (empty YAML → `{}`); `load_rulebook` must tolerate `/dev/null` (it does: `or {}`).

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement**

`src/thai_deck_eval/report/model.py`:

```python
from typing import Literal
from pydantic import BaseModel
from ..core.findings import Finding, Metric, Severity
from .scoring import Scores

class Report(BaseModel):
    deck_name: str
    deck_version: str
    rulebook_version: str
    stages_run: list[str]
    stages_skipped: list[str]
    findings: list[Finding]
    metrics: list[Metric]
    scores: Scores
    gate: Literal["pass", "fail"]

def build_report(name, version, result, scores, config) -> Report:
    return Report(
        deck_name=name, deck_version=version,
        rulebook_version=config.version,
        stages_run=[str(s) for s in result.stages_run],
        stages_skipped=[str(s) for s in result.stages_skipped],
        findings=result.findings, metrics=result.metrics, scores=scores,
        gate="fail" if result.has_errors else "pass")
```

`src/thai_deck_eval/report/render.py`:

```python
from ..core.findings import Severity
from .model import Report

def render_text(rep: Report) -> str:
    lines = [f"deck: {rep.deck_name} v{rep.deck_version}   "
             f"rulebook v{rep.rulebook_version}   gate: {rep.gate.upper()}",
             f"scores  integrity {rep.scores.integrity:.0f}  "
             f"language {rep.scores.language:.0f}  "
             f"method {rep.scores.method:.0f}  content {rep.scores.content:.0f}",
             f"stages  ran: {', '.join(rep.stages_run) or '-'}"
             + (f"   skipped: {', '.join(rep.stages_skipped)}"
                if rep.stages_skipped else "")]
    for sev in (Severity.ERROR, Severity.WARN, Severity.INFO):
        fs = [f for f in rep.findings if f.severity == sev]
        if fs:
            lines.append(f"\n{str(sev).upper()} ({len(fs)}):")
            for f in fs:
                rid = f.evidence.get("rule_override", f.rule)
                where = f" [{f.note_id}]" if f.note_id else ""
                lines.append(f"  {rid}{where}: {f.message}")
    if rep.metrics:
        lines.append("\nmetrics:")
        for m in rep.metrics:
            lines.append(f"  {m.name}: {m.value:.2f}")
    return "\n".join(lines) + "\n"
```

`src/thai_deck_eval/cli.py`:

```python
import sys
from pathlib import Path
import click
# import stage modules for rule registration side effects
from .stages import judge_rules, linguistic, mechanical, method  # noqa: F401
from .config import load_rulebook
from .core.context import EvalContext
from .core.findings import Stage
from .core.pipeline import evaluate_path
from .judge.core import CachedJudge, FakeJudge
from .report.model import build_report
from .report.render import render_text
from .report.scoring import compute_scores

def _build_language_ports():
    """Return (g2p, g2p_second, tokenizer, freq); None entries disable checks."""
    g2p = second = tok = freq = None
    try:
        from .lang.pythainlp_adapter import (PyThaiNLPG2P, PyThaiNLPTokenizer,
                                             TltkG2P)
        g2p, second, tok = PyThaiNLPG2P(), TltkG2P(), PyThaiNLPTokenizer()
    except ImportError:
        click.echo("warning: pythainlp not installed; linguistic checks skipped",
                   err=True)
    try:
        from .data_io import FileFrequencyList
        freq = FileFrequencyList()
    except OSError:
        click.echo("warning: frequency list missing", err=True)
    return g2p, second, tok, freq

def _build_judge(cfg):
    if cfg.judge.backend == "fake":
        return FakeJudge({})
    if cfg.judge.backend == "api":
        from .judge.api_judge import ApiJudge
        inner = ApiJudge(cfg.judge)
    else:
        from .judge.cli_judge import CliJudge
        inner = CliJudge(cfg.judge)
    return CachedJudge(inner, Path(cfg.judge.cache_path),
                       cfg.judge.model, cfg.judge.prompt_version)

@click.command()
@click.argument("deck_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--report", "report_path", type=click.Path(path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.option("--no-judge", is_flag=True)
@click.option("--stages", "stages_opt")
@click.option("--rulebook", type=click.Path(path_type=Path))
def main(deck_dir, report_path, fmt, no_judge, stages_opt, rulebook):
    cfg = load_rulebook(rulebook)
    stages = None
    if stages_opt:
        stages = [Stage(s.strip()) for s in stages_opt.split(",")]
    elif no_judge:
        stages = [Stage.MECHANICAL, Stage.LINGUISTIC, Stage.METHOD]

    def ctx_factory(deck):
        g2p, second, tok, freq = _build_language_ports()
        judge = None if no_judge else _build_judge(cfg)
        return EvalContext(deck=deck, config=cfg, g2p=g2p, g2p_second=second,
                           tokenizer=tok, freq=freq, judge=judge)

    result = evaluate_path(deck_dir, ctx_factory, stages=stages)
    scores = compute_scores(result, cfg)
    name, version = "?", "?"
    try:
        from .model.deck import load_deck
        meta = load_deck(deck_dir).meta
        name, version = meta.name, meta.version
    except Exception:
        pass
    rep = build_report(name, version, result, scores, cfg)
    out = rep.model_dump_json(indent=2) if fmt == "json" else render_text(rep)
    click.echo(out, nl=False)
    if report_path:
        report_path.write_text(rep.model_dump_json(indent=2))
    sys.exit(1 if rep.gate == "fail" else 0)
```

(`evaluate_path` loads the deck itself; the second `load_deck` for meta is redundant — refactor `evaluate_path` to return `(result, deck | None)` instead if cleaner. Either is acceptable; keep tests green.)

- [ ] **Step 4: Run to verify pass** — `uv run pytest -v` → all PASS. Also run the tool for real: `uv run thai-deck-eval --help`.
- [ ] **Step 5: Full-suite check** — `uv run pytest -v` (default), then `uv run pytest -m integration -v` if the `nlp` extra is installed.
- [ ] **Step 6: Commit** — message: `Add CLI entry point and report rendering`

---

## Commit message batch (for commit-gate `approve -F /tmp/cg-batch`)

1. `Add project scaffold and deck note models`
2. `Add deck loader and fixture deck builder`
3. `Add finding model, rule registry, and evaluation context`
4. `Add staged pipeline runner with error gating`
5. `Add mechanical integrity rules`
6. `Add Thai tone engine and syllable analyzer`
7. `Add IPA syllable parser and language ports`
8. `Add contrast inventory, spelling targets, and frequency data`
9. `Add linguistic correctness rules`
10. `Add pythainlp and tltk adapters with integration tests`
11. `Add method fidelity rules and coverage metrics`
12. `Add rulebook config and dimension scoring`
13. `Add judge port, verdict cache, and judge rules`
14. `Add CLI and API judge backends`
15. `Add CLI entry point and report rendering`
