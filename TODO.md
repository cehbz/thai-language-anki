# TODO

All items run against src/thai_syllabus (the redesigned pipeline; its
standard is docs/principles.md, docs/architecture.md and docs/specs/).
The old packages (thai_deck_eval, thai_deck_gen) stay; their spec-level
tests in tests/spec/test_deck_doctrine.py and test_generator_contract.py
still run against them.

## Priority (user, 2026-09-17): by value to learning

1. Sound stage part 2 (steps 4 to 7 below): graphemes, then pairs; the
   pronunciation-judge experiment first, since pair membership needs
   adjudicated pronunciations (178 words disputed).
2. Cutover to Anki: nothing built since the redesign is in the study
   deck yet.
3. Sentences for the 319 unfilled targets: the drafter's cached answer
   yields the same refused drafts every pass (198 and 126 refusals of
   two drafts in one night); fix the re-ask, then a drafting run.
4. The six open picture needs: learner direction from the review screen
   re-opens them; no engineering.

## Sentences: after the parsimonious-sentences arc

- `Word.components` (curated): a compound's registered parts, so a
  sentence using โรงพยาบาล ("hospital") also mentions โรง ("building")
  where the curator says so; fills clause 1 then reads "in the clauses
  or a component of a word in them" (spec 1 r8 log). Deferred until a
  registered compound and its registered part both matter.
- Fill-set memo keyed by text_sha ignores voice; `met_by` scans every
  adopted sentence per candidate (2 s at 400 sentences); `check_sentence`
  rebuilds the registered-id set per call.

## Sound stage: contrast inventory and pair stimuli (2026-09-18)

DONE 2026-09-18: the inventory revision is applied to both `data/contrasts.yaml`
and the live deck's `curated/confusions.yaml`. 23 confusions -> 27, total weight
70 -> 86, 61 pairs implied. Tone weights now come from Burnham et al. 1992 Table
6(b) (English listeners, all ten pairs, measured) rather than from an estimate;
the aspiration weights are flipped per Nagle et al. 2023; `consonant:r-l` is
dropped; three final-place contrasts added; `vowel_length` split three ways.
Evidence and cautions: docs/references/README.md, research-log 2026-09-18.

Still open:

- **`vowel_length:{low,mid,high}` are indistinguishable to the pair checker.**
  All three carry `sounds: [short, long]`, and `exact_confusion_violation`
  matches on dimension and sound values, not on id. Worse than "3x the pairs
  without guaranteed height coverage": `thai_deck_gen/producers/pairs.py:37`
  branches on the confusion's `kind` alone, ignoring the rest of the id, and
  `find_pair` is deterministic over `sorted(lexicon)` -- so the three axes
  produce three notes containing the identical word pair, not three different
  pairs. Making the split useful needs a `SoundConfusion` field so the pair
  search can restrict each axis to its own vowel group, i.e. a spec 1
  revision. Harmless today (`pairs.yaml` is `[]`); fix before pair search if
  height coverage matters.
- **The aspiration reweight rests on labials only.** Nagle et al. 2023 tested
  /b p pʰ/; the extension to alveolars and velars is theoretically motivated and
  untested. If it is wrong, `aspiration:alveolar-voiced` at w5 is over-weighted.
- **Liu et al. 2022 tone atlas** is in `docs/references/` but its per-pair
  English-listener numbers live in a figure, not the text. If the figure's bars
  can be read they would corroborate or challenge the 1992 table.

## Deferred

- **Hire native recordings for what Forvo lacks.** Pair members (up to 122
  words, one speaker per pair, both members in one session so the pair is
  same-speaker by construction), the 42 recited names (Forvo will not have
  the phrases) and the ~290 corroborated words with no Forvo item. One
  script, each item read in isolation with a pause, one file per speaker,
  split by silence with the existing ffmpeg wrapper, ingested as `learner`-
  supplied recordings with speaker sex/age/region recorded (E7 counts
  known speakers only). Cost, unverified against a live quote: Fiverr
  Thai voice-over gigs list from $35 base (one seller seen), and a rate
  guide (votrainer.com, secondary source) puts Fiverr sellers at $5-25
  per 150 words by level, so a ~450-word script is roughly $35-100 per
  speaker including an isolated-word/split-file extra; three speakers
  for diversity ~$100-300. Get two quotes before committing.

- Forvo growth measurement for the ageing interval: re-look up 30 of the
  333 empty migrated lookups after the 22:00 UTC reset and count how
  many gained a recording since 2026-08-29.
- Escalation anchor tie: `_anchor_ts` takes current-best
  rubric-agnostically; a legacy pass and a fresh pass tie at 50 and the
  tie breaks by sha, so a word whose legacy sha sorts first anchors at
  -1. Prefer the tied sha that has a producing attempt row. Measure
  against the live record first (how many words tie today).
- Multiple classifiers on a Word (banana: ลูก lûuk, หวี wǐi comb, เครือ
  khrʉa bunch): the deck holds one; a primary on the front with the
  alternatives on the back is a spec 1/4 revision if wanted (user,
  2026-09-10).
- Feedback screen: a rendition's accepted members render as "rejected"
  (candidate_shas vs the rendition identity; pre-existing).
- ReviewNote harvests (comments typed in Anki) append `learner-note` rows;
  the comment pass reads `card-flag` rows only, so they are never read.
- `compile` never consumes `record.gloss_on_requested`: a `gloss_on`
  action writes its row, FrontGloss still compiles empty (F3).
- The sentence Listening template now labels the target word; the Anki
  notetype updates on the next import (cutover pending).
- Illustrator: batch submission of generations (Gemini Batch API, half
  price; the run's one-generation-per-query shape fits a batch as the
  judge does); Flickr as a keywords-form source (`record.QUERY_FORMS`,
  spec 3 r36). The pre-r36 phrases: run `work/phrase-redraft.py` on the
  deck (dry run counted 1188 needs; about eight drafter asks).
- Chart cells (spec 3 r41): an automatic redraw when the keyword's
  picture changes (today the cell is redrawn only when the need reopens);
  a chart-cell clause in the picture rubric (a failed cell has no recovery
  but a learner-supplied picture); `chart_cell` computed three times per
  glyph attempt; `Derivations.sources_for_need` bound to the
  construction-time Syllabus; `_DbMediaIndex.words` stale after adoption
  until the next process; ฃ and ฅ by hand with a chosen keyword.
- `exhausted().capped` is query-scoped while its attempt count is per
  need (spec 3 r35): a source at the transient cap under an old query
  counts no attempt; an under-count of at most one per such source.

## Cutover

- Compile, delete-and-reimport in Anki, proof pass in `thai-syllabus
  review`, then `import` after a study session; verify study rows (family,
  anchor, card_kind) and flag rows.

## Content decisions (user)

- Study evidence awaited (principles' provisional marks): F1 minimal-pair
  difficulty; F8 introduction-order feel; F3 gloss placement (below); TTS
  acceptability on receptive sentences (09-02 notes: acceptable on two
  voices).
- Register research, not principles: the casual first-person pronoun's
  neutrality (ฉัน chǎn), particle spelling; Kam Mueang under Parked.
- Gloss placement (F3): picture-only fronts for every word by default;
  month and weekday names are the pre-decided exceptions; picture
  conventions (pointing figures, the two-position clock, object pairs,
  the 10×10 square) belong in the picture query hints; a word whose
  pictures keep failing escalates through the screen, where gloss-on for
  that word is one answer (user, 2026-09-10). Study impression pending.
- 62 classifier placeholder Words from migration need real facts
  (pronunciation, meaning).

## Sound stage (design approved 2026-09-12)

Design: docs/superpowers/specs/2026-09-12-sound-stage-design.md
(gitignored; rulings, sections, sequencing). Steps 1 to 3 are in (spec 5
r7; spec 3 r28 with spec 1 r13 and spec 2 r14; spec 1 r14 with the
inventories). Next act: the part 2 plan for steps 4 to 7.

4. Consonant graphemes: rows, acrophonic keyword Words, name Words with
   both Targets ordered after their grapheme, the `glyph` backend's
   chart cell (spec 1, spec 3).
5. Keyword search for vowel signs and tone marks (spec 3).
6. Pair search (weight-proportional, vocabulary first, recordability
   through Forvo), the retire key, coverage/sound-stage (spec 3, spec 5).
   Ruling 2026-09-18: TTS renditions are allowed for pairs, one voice
   across both members (`_tts_rendition`), `rendition/synthetic` stays
   warn. Measured on the live deck: of the 61 pairs the weights want, the
   vocabulary can fill 3 with one Forvo speaker on both members, 28 with
   any Forvo, 61 with TTS; 453 of the 543 Forvo-covered words have one
   speaker. Improve later: a native same-speaker rendition replaces a
   synthetic one when found, and see the hired-recordings item below.
   Before the search: revert the `vowel_length` three-way split (three
   rows yield the same pair set three times) and retire
   `pair/exact-confusion` (re-checks what `MinimalPair.create` refuses).
7. Principles F6: the recited names are learned as speech.

Adjudication follow-ups (first cycle 2026-09-12: 41 of 246 disputed
words corroborated, 205 stay disputed):
- thaig2p model defects leave 28 deck words with no analysis (loops or
  truncates on long compounds, an `a̯` offglide in coda position, the
  tone letters `˩˩`); those words stay disputed unless a curated
  exception names their pronunciation.
- 29 of 246 judge answers were refused by `parse_pronunciation` (one
  carried `vowel_length: mid`); they re-ask next run. Pull the batch
  results (msgbatch_01Ls9edB6cgm1hVDXp2HS9aM) to see the refused shapes
  before widening the parser.
- Length (32) and segment (29) disagreements between judge and thaig2p
  are unresolved by design (no third oracle). Weekday and month names
  dominate the segment set (วันอังคาร Tuesday: thaig2p assimilates the
  coda; กุมภาพันธ์ February: syllabification). A per-word curated
  exception stays the learner's path.
- `artifactView` renders a rendition as an image (bites in step 6).
  `derivations.confusion_weights(seed)` is a second home for
  `SoundConfusion.weight` and has no caller. `curated.py` imports
  `run.parse_day_starts` (move it out so `run.py` can import
  `save_words` at the top). `pair_count` above weight 5 yields one pair.
  The stats history normalises only `adjudicated`/`stayed_disputed` for
  old rows; a future report field needs the same or a column union.

## Content work (machine)

- Sentence corpus: the run's sentence attempt fills open Targets; since
  spec 3 r27 a sentence fills at most three of them.
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
- Exercise-latency measure; scene-picture prioritization budget; batch
  set-cover sentence generation; grapheme spoken-name recordings.
- Gallery note text: the row keeps answer["note"], but only its card-flag
  label is read (directed(), card_flags); surface the text on the screen.
- The legacy thai_deck_gen GoogleTts (src/thai_deck_gen/media/tts.py) still
  sends its API key as a `?key=` query parameter and interpolates wire
  errors unredacted; the thai_syllabus backend was fixed 2026-09-10.
