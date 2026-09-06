# TODO

All items run against src/thai_syllabus (the redesigned pipeline; its
standard is docs/principles.md, docs/architecture.md and docs/specs/).
The old packages (thai_deck_eval, thai_deck_gen) stay; their spec-level
tests in tests/spec/test_deck_doctrine.py and test_generator_contract.py
still run against them.

## Review closure, remaining

- B8: make the run's accounting identity hold on every return path
  (four holes parked at Task B7's cap, 2026-09-06): the resolve-
  unreachable and outstanding-batch branches count exhausted/unserved
  twice; the sentence-attempt-unreachable branch defers none of the
  loop's needs; needs skipped for a Source transport failure land in no
  bucket; available dedups (word, sentence) while the drafted-target
  count is per Target.
- Final whole-branch review of the 2026-09-04 plan, then delete
  .superpowers/sdd/2026-09-04-review-closure.
- KB project node: full rewrite (its architecture/CLI/stores sections
  describe the old pipeline; keep the NLP, judge and media measurements).

## Cutover

- Write `~/decks/thai-ff/curated/providers.yaml`: judge transport batch,
  model, price_per_mtok (required), imgfetch_path and audiofetch_path
  (required; ~/bin/imgfetch, ~/bin/audiofetch), image_candidates,
  attempt_cap, tts male_voices and female_voices (both non-empty),
  secrets forvo / google_tts / anthropic / pexels as 0600 files under
  ~/.config/thai-deck-gen/ (the tts key file is google-tts.key),
  search_proxy. The loader refuses a missing file or field; the review
  screen also needs this file.
- Migrate: `thai-syllabus migrate --old-deck ~/decks/thai-ff.20260903
  --old-data data --new-root ~/decks/thai-ff`. Joins pictures by (thai,
  category) and reports ambiguous forms; carries candidates.yaml
  verdicts under a legacy rubric id that never ranks; idempotent (run it
  twice, read already_present). Every current picture is judged by the
  first run. Provider cache rows are keyed by the current ProvideKey
  encoding (source:kind:query): a syllabus.db written before 2026-09-06
  would miss every provider cache row and a re-run of migrate into it
  would duplicate the forvo rows, so migrate only into a fresh
  syllabus.db.
- First run, batch judge, smoke-capped per source (`--backend-cap
  NAME=N` is a per-day cap read from the record): expect every picture
  question in one batch, nothing improved, pending == pictures. Second
  run resolves it. Read the RunReport line: available == attempted +
  exhausted + pending + unserved + budgeted + deferred (B8 closes the
  known exceptions).
- Compile, delete-and-reimport in Anki, proof pass in `thai-syllabus
  review`, then `import` after a study session; verify study rows (family,
  anchor, card_kind) and flag rows.

## Content decisions (user)

- Productive-Target selection rule: which words get production cards
  ("what I intend to say") — frequency cutoff, category-based, or
  hand-picked.
- Gloss placement (a picture carries meaning, a gloss fixes it): picture-only fronts vs gloss chip — needs the
  gloss-off study impression; unblocks the principles lock.
- ~26 real classifier findings from the judge triage: word-list fixes,
  incl. the 6 time-of-day words wrongly assigned compound parts as
  classifiers (docs/superpowers/review/2026-09-01-judge-triage.md).
- 62 classifier placeholder Words from migration need real facts
  (pronunciation, meaning).

## Content work (machine, once cutover done)

- Grapheme data: 44 consonant name-words (recited names, e.g. กอ ไก่
  "gɔɔ gài") + keywords; vowel/tone-mark keywords chosen (concrete,
  picturable); first Forvo lookups answer whether letter names exist
  there.
- 221 migrated words with placeholder `disputed` pronunciations →
  knowledge-adjudication judge pass (evidence hierarchy in
  docs/superpowers/review/2026-09-01-domain-language.md).
- Sentence corpus: all 766 targets are unfilled after migration; the
  run's sentence attempt (spec 3 section 5: one draft pass per run over
  the open targets, fills() + judge, then adopt) produces them. Sentence
  recordings and scene pictures follow adoption.
- Pair search and grapheme-keyword attempts (spec 3 section 5): shapes
  defined, implemented after the cutover; the old deck's 22 pairs did not
  migrate, so renditions are moot until pairs exist.
- Batch judge granularity: the run submits one Message Batch per need;
  before the whole-syllabus batch pass, gather every judge question of a
  run into one batch and resolve on the next run (the pending derivation
  already supports this).

## Nice to have

- Category framework: the FF 27 categories are English-centric; one
  restructuring seen in the community regroups vocabulary into cognitive
  domains (perception, emotion, cognition, social relationships,
  culture) to avoid imposing one language's logic on another. Categories
  are a coverage measure, so a second grouping would be a second measure
  (research log 2026-09-04).

## Parked

- **Kam Mueang production content** (user goal, 2026-09-02): Northern
  production matters — market/food relationships, perceived effort.
  Parked for cost: Central-only verification tooling, no TTS, thin
  Forvo, เจ้า (jâo) particle gender-marking contradicted across sources
  (needs a local speaker). Sizing open: market phrase set vs systematic.
  Research: docs/superpowers/review/2026-09-02-register-research.md.
- Commission batch (325 native recordings): deferred until deck churn
  settles; work/commission_batch_001.yaml carries the item list. It is
  the native path behind the relaxed `recording/synthetic` and
  `rendition/synthetic` warnings (native-audio principle kept as the
  target).
- Listener backend (audio verification of recordings): calibration
  harness over corroborated words with native recordings; admit at
  measured rank or not at all.
- Meaning vs gloss on Word (a Word's meaning is the sense; the English
  gloss is its L1 rendering).
- Exercise-latency measure; scene-picture prioritization budget.
- Gallery note text: the row keeps answer["note"], but only its card-flag
  label is read (directed(), card_flags); surface the text on the screen.
