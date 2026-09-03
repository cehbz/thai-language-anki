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

- Implement the spec 3 revision: attempts per need kind (picture,
  recording, rendition, sentence), pending as a derivation, authority-
  driven current-best with the provenance prior, cost on every answer
  (api transport must return usage), artifact paths to the judge with
  per-transport encoding, the fit/preference picture rubrics (old texts
  verbatim), completeness rules at error, the warn rules for synthetic
  and mixed-speaker audio, the spec 2 carry-over amendment (old judge
  cache migrates; learner rows keyed by word id).
- Write `~/decks/thai-ff/curated/providers.yaml` (secret references,
  proxy, imgfetch path, judge transport batch + model + price,
  image_candidates) and run the real migration:
  `thai-syllabus migrate --old-deck ~/decks/thai-ff.20260903 --old-data
  data --new-root ~/decks/thai-ff`. Expect the 718 studied pictures to
  rank on day one.
- First sourcing run against the migrated state, smoke-capped per source;
  verify RunReport (attempted / improved / exhausted / pending / spend)
  against expectations; then the whole-syllabus batch passes.
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
