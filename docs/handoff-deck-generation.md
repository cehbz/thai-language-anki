# Handoff: deck generation session

Context for the session that builds the **deck generator** (and eventually the
.apkg compiler). The evaluator is complete and is your fitness function.

## What exists

- **Evaluator** (`thai_deck_eval`, this repo, public at
  github.com/cehbz/thai-language-anki): scores a deck directory against a
  Fluent Forever rulebook. `uv run thai-deck-eval <deck-dir>` → findings +
  four dimension scores + coverage metrics; exit 0 pass / 1 gate fail / 2
  crash. Stages schema → mechanical → linguistic → method → judge, gated by
  a per-stage `depends_on` DAG (rulebook.yaml).
- **Authority documents**: spec `docs/superpowers/specs/2026-08-26-deck-evaluator-design.md`
  (deck format contract, rulebook doctrine, research references); KB node
  `~/.claude/knowledge/projects/thai-language-anki.md` (measured Thai NLP
  gotchas); `TODO.md` (parked work).
- **Reference deck**: `tests/helpers.py` `DeckBuilder`/`GOLDEN` — a complete
  minimal deck that passes everything; the fastest way to learn the format.

## The generator's job

Emit a deck directory the evaluator scores well, iterating against its
output:

1. Generate/extend deck YAML + media.
2. `uv run thai-deck-eval deck/ --no-judge --format json` — fast loop on the
   free stages. Findings carry rule ids + note ids; metrics
   (`coverage/minimal_pairs`, `coverage/spelling`, `coverage/frequency`,
   `coverage/categories`, `speakers/minimal_pairs`) list `covered`/`missing`
   in their detail — target the gaps, not the aggregate score.
3. Full run with judge on surviving candidates. Judge verdicts are cached in
   SQLite keyed by content — unchanged cards are free; `judge.backend: cli`
   (default) bills the Claude subscription via `claude -p`.

Install: `uv sync --extra nlp` (linguistic stage needs pythainlp+torch;
first thaig2p call downloads a model). `--extra llm` for the API judge.

## Contract facts the generator must respect

- **Format**: deck.yaml + notes/{minimal_pairs,spelling_sound,picture_words,
  sentences}.yaml + media/. Audio declares `source: native|tts` and
  `speaker`; TTS on minimal pairs is a gating ERROR, TTS elsewhere a WARN.
- **Authored IPA**: segments + ː + Chao tone letters (kʰaːw˥˩). It is
  *verified*, not trusted: two-engine g2p voting plus a deterministic
  tone-rule engine. If the engine is wrong for a word (known: น้ำ, ข้าว),
  add curated truth to `data/g2p_exceptions.yaml` — never author wrong IPA
  to appease the engine.
- **Minimal pairs** must differ in exactly the declared contrast (VOT
  triplets บ/ป/พ allowed). Coverage is weighted by difficulty for English
  speakers (`data/contrasts.yaml`; mid-low tone pair weighs most).
- **Picture words**: image + native audio + frequency rank (checked against
  `data/frequency_th.txt`, a colloquial-weighted subtitle/TNC blend) +
  one of the 27 FF categories (`data/categories.yaml`); nouns carry
  classifiers; no English glosses. `personal_connection` stays empty — it
  is the user's slot, not the generator's.
- **Sentences**: cloze target must be a real token (boundary-aligned
  compound membership is accepted — กิน inside กินข้าว is fine); ~1 unknown
  non-target token per sentence given deck order; glosses allowed here.
- **Staging**: sentence notes only atop ≥300 picture words (configurable);
  spelling sub-cards taper after rank ~300.

## Hard problems the generator session must solve (evaluator can't help)

1. **Native audio sourcing** — the doctrine's strongest requirement and the
   biggest open question. Options researched during design: Forvo,
   Rhinospike, commissioned recordings (Fiverr); multi-speaker coverage of
   minimal pairs is a scored metric (`target_speakers: 3`). No pipeline
   exists yet.
2. **Image sourcing** — the judge checks relevance and rejects embedded
   English text (vision), but acquisition/licensing is unsolved.
3. **Word list** — the FF 625 English list needs Thai adaptation (concept
   splits, classifiers); `data/frequency_th.txt` ranks candidates, the
   category metric tracks semantic spread.
4. **.apkg compiler** — separate component (genanki was the assumed route);
   card fan-out per note type is specified in the spec §note families.

## Parked until testable (TODO.md)

Message Batches bulk-judge mode (build when a real deck exists — one live
smoke run needed), STT audio↔text verification (prototype whisper Thai
accuracy first; minimal pairs are ASR's worst case), .apkg validation
(belongs to the compiler).

## Process conventions in this repo

- commit-gate is enabled: agents stage, the human approves each exact
  message (`approve` / `approve-push` from the repo cwd).
- TDD; default pytest suite must not import pythainlp/anthropic; the
  real-ports golden e2e (`uv run pytest -m integration`) is the honest gate.
- Brainstorm → spec → plan → subagent execution worked well for the
  evaluator; specs/plans live under `docs/superpowers/`.
