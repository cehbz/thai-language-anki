# Thai Fluent Forever Deck Evaluator — Design Spec

Date: 2026-08-26
Status: approved design, pre-implementation

## Purpose

An automated evaluator for a Fluent Forever–style Thai Anki deck. It is the
quality gate for a deck **generation pipeline** (evaluator-first): the
generator emits a candidate deck in a structured source format, the evaluator
scores it against a rulebook, and the generator iterates against the findings
and scores. It also works standalone as an audit tool for any deck in the
source format.

The rulebook encodes the **community-refined** Fluent Forever doctrine: the
book's card taxonomy and staging (Wyner 2014 + the Gallery appendix), with
corrections practitioners converged on — no-translation enforced on concrete
picture cards but glosses permitted on abstract words, images required only
where picturable, minimal pairs must be single-contrast, native (non-TTS)
audio required for tone-bearing cards, romanization penalized outside the
earliest script stage.

## Decisions (settled during design)

| Decision | Choice |
|---|---|
| Primary job | Fitness function for generation; standalone audit secondary |
| Input | Source-of-truth deck directory (YAML + media), not .apkg |
| Dimensions | Mechanical integrity, method fidelity, linguistic correctness, content quality (LLM judge) |
| Output | JSON findings + per-dimension scores + raw metrics; text render from same data; error findings gate |
| Stack | Python, uv-managed; pydantic; pythainlp; Anthropic SDK / Claude Agent SDK |
| Doctrine | Community-refined (see above) |
| Architecture | Staged pipeline (cheap→expensive) with short-circuit gates and verdict caching, rule-registry core |

## Deck source format

A deck is a directory:

```
deck/
  deck.yaml              # metadata: name, version, stage plan
  notes/
    minimal_pairs.yaml
    spelling_sound.yaml
    picture_words.yaml
    sentences.yaml       # new_word / word_form / word_order notes
  media/
    audio/...  images/...
```

Validated by pydantic models. One **note** per entry; card fan-out
(comprehension/production/spelling) is derived by the future compiler, so the
evaluator reasons about notes plus declared card intents — no duplicated data
across cards.

`deck.yaml` declares the **stage plan** (which phases the deck claims to
cover) so coverage checks demand only what is claimed — an ear-training-only
deck is not penalized for lacking sentences.

Every media reference is a relative path into `media/`. Every audio entry
carries `source: native | tts` and `speaker` (an opaque id); the rulebook
needs both (TTS on tone-bearing cards is a finding; speaker diversity across
minimal pairs is a metric).

### Note families

- **`minimal_pair`** — 2–3 members, each `{thai, ipa, audio}`; declared
  `contrast: tone | vowel_length | aspiration | vowel_quality | consonant |
  final`; optional English glosses (sanctioned on this family); `speaker` per
  audio file. The evaluator verifies the pair is minimal in *exactly* the
  declared contrast via g2p — never trusted from the generator.
- **`spelling_sound`** — target grapheme/pattern (consonant with class, vowel
  form, or tone-mark rule), concrete example word, audio, picture.
- **`picture_word`** — `thai`, `image`, `audio`, `frequency_rank`, `category`
  (625-list category), optional `classifier` (required for nouns by rule, not
  schema), optional `ipa`, `test_spelling` flag, optional
  `personal_connection` (user-editable slot; generated decks cannot invent
  memories — checked at info level only).
- **`sentence`** (`kind: new_word | word_form | word_order`) — `thai`
  sentence, `target` (the blank/tested element), optional `image`, `audio`,
  optional monolingual `definition`, optional `gloss` (allowed for abstract
  words), `grammar_note`.

Authored format is YAML (generator emits it; humans occasionally hand-edit).

## Pipeline

Four ordered stages by cost; a stage runs only if the previous stage's gate
passes (gates configurable):

1. **Schema** — pydantic validation of the deck directory.
2. **Mechanical** — deterministic checks, no NLP.
3. **Linguistic** — pythainlp + our own tone engine.
4. **Judge** — LLM checks, cached.

Default gates: no schema errors before linguistic; no mechanical errors
before judge. In the generation loop the cheap stages run constantly and the
judge only on candidates that survive them.

## Rules and findings

Every check is a registered rule: namespaced ID (`mech/media-missing`,
`lang/tone-mismatch`, `meth/pair-coverage`, `judge/unnatural-sentence`),
severity (`error` gates, `warn` reduces score, `info` reports), and a
**dimension** it scores against. Dimension ≠ stage: e.g. the one-new-element
rule needs tokenization (linguistic stage) but scores as method fidelity.

A finding = rule ID, severity, note ref, message, evidence (e.g. both g2p
outputs on a mismatch).

### Catalog (representative; the registry is the authority)

**Mechanical** — media refs resolve and no orphans; required fields per
family; Thai fields contain Thai codepoints, no Latin/romanization in
learner-facing fields; no duplicate notes; cloze target string appears in its
sentence; audio declares `source`/`speaker`; gloss present on a
`picture_word` (concrete words are gloss-free under the doctrine).

**Linguistic** — minimal pair differs in exactly the declared contrast and
nothing else (g2p both members; catches ใหม่/ไม้-style two-feature pairs);
claimed IPA/tone matches script, verified two ways: (a) our deterministic
tone algorithm (consonant class + live/dead syllable + tone mark, incl. ห นำ
and อ นำ), (b) `thaig2p`/`tltk` consistency voting with a loanword exceptions
list (e.g. น้ำ); no 5-tone contrast claimed on dead syllables; sentence
`target` is a real token of the sentence; declared `frequency_rank` matches
the reference list.

**Method fidelity** — mostly coverage metrics:
- Minimal-pair coverage of the Thai contrast inventory,
  difficulty-weighted: mid–low tone pair heaviest (hardest for English
  speakers per perception research), then aspiration triplets per place
  (บ/ป/พ, ด/ต/ท, ก/ข-ค, จ/ช), vowel length, /ŋ/-onset, เ-อ-class vowels,
  เ-ือ/เ-า-type diphthongs, unreleased finals.
- Spelling-sound coverage of 42 modern consonants (with class: 9 mid, 11
  high, 24 low), vowel forms, tone marks/rules.
- Frequency-list and 625-category coverage of picture words.
- Nouns carry classifiers.
- Card fan-out within bounds (≤3 per picture word, ≤4 per sentence note).
- `test_spelling` tapers after the first ~300 words.
- Sentences introduce ≈1 unknown token given deck order (tokenized against
  the deck's own vocabulary).
- Speaker diversity across minimal-pair audio (HVPT-informed metric).
- Native (non-TTS) audio on all tone-bearing cards.
- Staging: sentence notes only atop a sufficient picture-word base — the
  threshold is a `rulebook.yaml` value (default: 300 picture words) scoped by
  the stage plan.

**Judge** — sentence naturalness/correctness; image relevance and
no-embedded-English (vision); definition genuinely monolingual and accurate;
classifier correctness; gloss accuracy.

## Scoring

Four dimension scores, 0–100: integrity, language, method, content.
Integrity/language/content start at 100 with weighted per-finding
deductions; method is a weighted blend of the coverage metrics minus
deductions. The report carries the raw metrics (each coverage %, counts)
alongside scores so the generator targets specific gaps.

**Gate:** any `error` finding ⇒ exit 1 regardless of scores. Scores are the
gradient; errors are the gate.

Weights, thresholds, and gate policy live in a versioned `rulebook.yaml`
with baked-in defaults. The rulebook version is stamped into every report;
scores are comparable only within a version.

## LLM judge

Judge rules delegate to a `Judge` port with three implementations:

- **`CliJudge` (default)** — routes calls through Claude Code headless
  (`claude -p`) / the Claude Agent SDK, inheriting the user's Claude
  subscription (Pro/Max) credentials: no per-token API billing. JSON output
  is requested and validated with pydantic, retrying on schema mismatch (no
  `messages.parse()` on this path). Vision checks pass image file paths.
  Constraints: subscription 5-hour usage windows (a cold full-deck pass
  ~1,600 calls may need chunking), higher per-call latency, no Batches.
- **`ApiJudge`** — Anthropic SDK, model `claude-opus-5` (configurable),
  adaptive thinking, structured outputs via `client.messages.parse()`
  against pydantic verdict models; images as vision blocks. For
  unattended/CI runs.
- **`FakeJudge`** — canned verdicts for tests.

Backend, model, and `effort` are `rulebook.yaml` config. Judge calls batch
all applicable rules for one card into a single call.

**Caching:** every verdict cached in local SQLite keyed by `(rule_id,
prompt_version, model, content_hash)`. Only changed cards cost anything;
no-op re-runs cost zero. A `prompt_version`/model bump invalidates by
design.

**Confidence demotion:** verdicts carry confidence; below a configurable
threshold a would-be finding is demoted to `info` rather than gating.

**Cost envelope (ApiJudge, Opus 5):** cold full deck (~625 picture words
with vision + ~1,000 sentences) ≈ 3–4M tokens ≈ $25–35; a 50-card iteration
≈ $1; cached re-run $0. Knobs: judge `effort` (medium suffices for per-card
verdicts), image downscaling before upload (image tokens dominate the vision
slice). `CliJudge` makes cold passes subscription-covered instead.

## CLI

`thai-deck-eval <deck-dir>` (uv-run console script).

Flags: `--report out.json`, `--format text|json`, `--stages` / `--no-judge`,
`--rulebook <path>`.

JSON report: deck metadata, rulebook version, per-note findings, metrics,
dimension scores, gate result. Text format renders from the same structure.

Exit codes: 0 pass, 1 gate failure, 2 evaluator error.

## Testing

TDD throughout; pytest. Backbone: fixture decks —

- `golden/`: a conforming mini-deck that passes everything.
- Mutation fixtures, each seeding exactly one violation; tests assert
  exactly that rule ID fires and nothing else.

Tone engine: pure unit tests against the published tone tables (class ×
live/dead × mark, plus ห นำ/อ นำ). Linguistic checks depend on a `G2P`
port — unit tests use a fake; a separately-marked integration suite
exercises real pythainlp. Judge tested against `FakeJudge`, plus one marked
live smoke test.

## Layout

```
src/thai_deck_eval/
  model/        # pydantic deck schema
  stages/
    mechanical/ linguistic/ method/ judge/   # rule modules + registry
  lang/         # tone engine, G2P ports
  report/
  cli.py
data/           # contrast inventory, consonant-class tables,
                # frequency list — versioned YAML rulebook inputs, not code
tests/
  fixtures/golden/  fixtures/mut_*/
```

## Deferred (explicitly out of scope for v1)

- Message Batches API sweep mode (50% cost for full re-judges after cache
  invalidation).
- Audio↔text verification via STT (Whisper supports Thai).
- .apkg structural validation (belongs with the future compiler).
- Deck generator and compiler themselves — separate projects consuming this
  evaluator's contract.

## Key research references

- FF card gallery: https://blog.fluent-forever.com/gallery/
- FF Thai resource page: https://blog.fluent-forever.com/learn-thai/
- 625 list: https://method.fluent-forever.com/base-vocabulary-list/
- Thai phonology: https://en.wikipedia.org/wiki/Thai_phonology
- Tone rules: http://www.thai-language.com/ref/tone-rules ,
  https://thaiwithgrace.com/thai-tones/
- Minimal pairs: https://preply.com/en/blog/thai-minimal-pairs/
- Mid–low tone difficulty (Wayland & Guion 2004):
  https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9922.2004.00283.x
- pythainlp (tokenize, thaig2p, TNC frequencies):
  https://github.com/PyThaiNLP/pythainlp ; second-opinion g2p: tltk
- Frequency list (machine-readable):
  https://github.com/hermitdave/FrequencyWords (th_50k)
- Community practice synthesis: r/Anki, r/languagelearning, r/learnthai
  threads (see research notes in session history); single-contrast
  minimal-pair rule and native-audio requirement from r/learnthai.
