# thai-deck-eval

Automated evaluator for a [Fluent Forever](https://blog.fluent-forever.com/gallery/)-style
Thai Anki deck. It scores a deck (written in a structured YAML source format)
against a rulebook and reports findings plus per-dimension scores — designed
as the quality gate / fitness function for a deck **generation** pipeline,
and usable standalone to audit any deck in the source format.

The rulebook encodes the community-refined Fluent Forever doctrine:
pronunciation first (minimal pairs, spelling↔sound), a picturable-word base
before sentence cards, no L1 glosses on concrete picture words, native
(non-TTS) audio on tone-bearing cards, and minimal pairs that differ in
exactly one contrast (tone, vowel length, aspiration, …).

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```sh
uv sync                # core evaluator
uv sync --extra nlp    # + pythainlp/tltk/torch: linguistic verification
uv sync --extra llm    # + anthropic SDK: API judge backend
```

## Usage

```sh
uv run thai-deck-eval <deck-dir> [--no-judge] [--format text|json]
                      [--report out.json] [--stages mechanical,linguistic,...]
                      [--rulebook rulebook.yaml]
```

Exit codes: `0` pass, `1` gate failure (any error-severity finding), `2`
evaluator error.

The evaluation runs in stages — schema → mechanical → linguistic → method →
judge — each gated by a configurable dependency DAG (`depends_on` in
`rulebook.yaml`): cheap checks always run on a parseable deck; the expensive
LLM judge only runs on a deck that passed mechanical and linguistic checks.

- **mechanical** — media files resolve, no Latin in Thai fields, cloze
  targets present, no glosses on picture words, …
- **linguistic** — minimal pairs verified minimal via grapheme-to-phoneme
  (with a deterministic tone-rule engine as a second check), authored IPA
  matches the script, frequency ranks match the reference list.
- **method** — Fluent Forever coverage metrics: minimal-pair contrast
  inventory (difficulty-weighted), spelling-target coverage, frequency and
  625-category coverage, speaker diversity, native-audio and classifier
  rules.
- **judge** — LLM checks (sentence naturalness, image relevance, monolingual
  definitions) with a persistent verdict cache, so only changed cards cost
  anything. Backends: `cli` (default; runs `claude -p`, billed to a Claude
  subscription), `api` (Anthropic API), `fake` (tests).

## Deck source format

A deck is a directory:

```
deck/
  deck.yaml              # name, version, stage plan
  notes/
    minimal_pairs.yaml   # 2-3 members, declared contrast, audio per member
    spelling_sound.yaml  # grapheme/pattern -> example word + audio + image
    picture_words.yaml   # word, image, audio, frequency rank, FF category
    sentences.yaml       # cloze-style sentence notes (new word / word form / word order)
  media/
    audio/ ...  images/ ...
```

Audio entries declare `source: native|tts` and a `speaker` id. Authored IPA
uses Chao tone letters (`kʰaːw˥˩`). See
`docs/superpowers/specs/2026-08-26-deck-evaluator-design.md` for the full
contract, and `tests/helpers.py` for a complete minimal example deck.

## Configuration

`rulebook.yaml` holds every knob with its default: deduction weights, metric
weights, stage dependency DAG, judge backend/model/effort, confidence floor,
cache path. The rulebook version is stamped into every report; scores are
comparable only within a version.

## Development

```sh
uv run pytest                  # default suite (no pythainlp/anthropic needed)
uv run pytest -m integration   # real thaig2p g2p (downloads a torch model)
uv run pytest -m live          # real LLM calls (manual, needs credentials)
```

The golden reference deck (built by `tests/helpers.py`) must pass the real
CLI with real NLP ports (`tests/test_cli_integration.py`) — that is the
project's honest end-to-end gate.

## Data

- `data/frequency_th.txt` — 5,000-word list blended 0.7/0.3 from the
  OpenSubtitles 2018 Thai list ([hermitdave/FrequencyWords](https://github.com/hermitdave/FrequencyWords),
  CC BY-SA 4.0) and the Thai National Corpus frequencies shipped with
  pythainlp, filtered to dictionary-attested Thai-script words. The blend
  deliberately favors colloquial spoken Thai. Regenerate with
  `uv run --extra nlp python scripts/fetch_frequency.py`.
- `data/contrasts.yaml` — Thai contrast inventory for minimal-pair coverage,
  difficulty-weighted for English speakers.
- `data/g2p_exceptions.yaml` — curated pronunciations where the g2p engine
  is wrong (e.g. น้ำ, ข้าว).

## Status

The evaluator is complete. Planned companions (see `TODO.md`): the deck
generator, an .apkg compiler, a Message-Batches bulk judge mode, and
STT-based audio verification.
