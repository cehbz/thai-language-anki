# TODO

All items below run against the redesigned pipeline (src/thai_syllabus).
The old packages (thai_deck_eval, thai_deck_gen) are superseded; their
removal happens once the new pipeline has produced and survived a studied
deck.

## Cutover

Spec 3 was rewritten 2026-09-03 (docs/superpowers/specs/2026-09-03-ports-
backends-design.md): the run had no assess step, picture attempts stopped
at search, recordings never ranked, sentences were never produced, the
gate passed an empty syllabus. The old deck is at ~/decks/thai-ff.20260903;
the new deck root is ~/decks/thai-ff.

- Write `~/decks/thai-ff/curated/providers.yaml` (secret references
  including `anthropic`, proxy, imgfetch_path AND audiofetch_path, judge
  transport batch + model + price, image_candidates -- the loader now
  refuses a file missing any of those) and run the real migration:
  `thai-syllabus migrate --old-deck ~/decks/thai-ff.20260903 --old-data
  data --new-root ~/decks/thai-ff`. Migration carries candidates.yaml's
  recorded pass/fail per candidate over as the judge fit verdict (the old
  judge_cache.sqlite does not migrate): expect 410 of 654 chosen pictures
  to rank on day one (measured 2026-09-03 on ~/decks/thai-ff.20260903;
  scratch migration probe: 356 of 766 words still missing a picture); the
  rest are judged by the first run's assess-first step.
- First sourcing run against the migrated state, smoke-capped per source;
  verify RunReport (attempted / improved / exhausted / pending / excluded
  / unreachable / spend) against expectations; then the whole-syllabus
  batch passes.
- First `thai-syllabus compile`; delete-and-reimport in Anki; a proof
  pass in `thai-syllabus review`. Testing-deck relaxations are severity
  overrides in the deck's rulebook.yaml, recorded there.
- First `thai-syllabus import` after a study session; verify StudyRecords
  and flag/ReviewNote rows.
- Retire the old packages, scripts/proof_gallery.py, and the old deck
  work/ stores after one full loop succeeds.

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

## Follow-ups from the whole-arc review (2026-09-04)

- reviewserver `_tried_summary` lists imgfetch/audiofetch rows and fetched
  urls as "sources/phrases tried" in the direction prompt; filter to the
  Source-ask rows (`provides == kind`) as `_latest_query` does.
- `_picture_attempt`'s pre-search guard returns `attempted=False` even when
  the fit questions were really asked before the judge failed; return
  `_attempted(spend)`.
- A dead judge ends each attempt, not the run: `run()` still escalates every
  source for every need and re-asks the wire each time. Abort the run after
  the first unreachable-judge attempt.
- `load_providers_config` accepts an empty `tts.male_voices`; `pick_voice`
  then divides by zero. Validate non-empty pools.
- `exhausted(pair, "rendition")` is always 0 attempts: rendition asks are
  recorded under the member subjects, never the pair.

## Design follow-ups (round review, 2026-09-04)

- Row conventions are parsed in three places (`derivations._matches_kind`
  and `_machine_ranks`, plus reviewserver's private duplicates of
  `_matches_kind`/`_rows_for`/`_gap_candidates`/`_candidate_shas`). One
  row-reading module both import, and an explicit `kind` on every provide
  row instead of inferring it from `provides`.
- `derivations` imports `AUTHORITY_ORDER`/`ROLE_FOR_KIND` from `assessor`;
  authority and the kind→role map are domain data for a small shared module.
- `Assessor._preparation_failure` re-runs prompt/attachment preparation on
  every inline miss to classify exclusions; replace with a distinct
  `PreparationError` raised by the backend that `ask_many` maps to
  `excluded`.
- Batch resume state is a growing marker convention (`judge-batch-pending`
  rows superseded by abandoned markers); revisit when batch granularity
  moves to one batch per run.
- `rulebook.py` imports rubric texts from `thai_deck_eval.judge.prompts`;
  copy them into the rulebook before retiring the old packages.
- Spec 3 §5 picture query: "gloss head term + category qualifier" is not
  implementable (Word has no category); the query is the whole meaning.
  Amend the spec or add the field.
- Spec 3 §6: `pending` is implemented as "batch marker with an unresolved
  key" (narrower than the spec's wording); a judge-passed, learner-unrated
  picture queues in bucket 1 rather than 3; `RunReport.improved` counts a
  preference bonus as improvement without an artifact change.
- Migration joins picture notes to words by Thai form, first row wins for
  the 39 homographs (45 word rows lose their picture, named in the report);
  a (thai, gloss) key or a curated map would remove the shortcut.

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
