# TODO

All items below run against the redesigned pipeline (src/thai_syllabus).
The old packages (thai_deck_eval, thai_deck_gen) are superseded; their
removal happens once the new pipeline has produced and survived a studied
deck.

## Cutover

- Write `<new-deck>/curated/providers.yaml` (secret references, proxy,
  imgfetch path, judge transport) and run the real migration:
  `thai-syllabus migrate --old-deck ~/decks/thai-ff --old-data data
  --new-root <new-deck>`.
- First real sourcing run (`thai-syllabus run`) against the migrated
  state; verify RunReport numbers against expectations.
- First `thai-syllabus compile`; delete-and-reimport in Anki; a proof
  pass in `thai-syllabus review`.
- First `thai-syllabus import` after a study session; verify StudyRecords
  and flag/ReviewNote rows.
- Retire the old packages, scripts/proof_gallery.py, and the old deck
  work/ stores after one full loop succeeds.

## Content decisions (user)

- Productive-Target selection rule: which words get production cards
  ("what I intend to say") — frequency cutoff, category-based, or
  hand-picked.
- Gloss placement (F3): picture-only fronts vs gloss chip — needs the
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
- Sentence corpus regeneration under F5 (picture cards introduce,
  sentences exercise, batch set-cover generation is an open design
  choice).

## Parked

- **Kam Mueang production content** (user goal, 2026-09-02): Northern
  production matters — market/food relationships, perceived effort.
  Parked for cost: Central-only verification tooling, no TTS, thin
  Forvo, เจ้า (jâo) particle gender-marking contradicted across sources
  (needs a local speaker). Sizing open: market phrase set vs systematic.
  Research: docs/superpowers/review/2026-09-02-register-research.md.
- Commission batch (325 native recordings): deferred until deck churn
  settles; work/commission_batch_001.yaml carries the item list.
- Listener backend (audio verification of recordings): calibration
  harness over corroborated words with native recordings; admit at
  measured rank or not at all.
- Meaning vs gloss on Word (a Word's meaning is the sense; the English
  gloss is its L1 rendering).
- Exercise-latency measure; scene-picture prioritization budget.
- Rendition and grapheme-keyword sourcing levers are unwired (run()
  tolerates them as unlevered): a rendition needs one speaker recording
  both members — a compound ask no single backend answers; design at
  first sourcing run.
