# TODO

All items run against src/thai_syllabus (the redesigned pipeline; its
standard is docs/principles.md, docs/architecture.md and docs/specs/).
The old packages (thai_deck_eval, thai_deck_gen) stay; their spec-level
tests in tests/spec/test_deck_doctrine.py and test_generator_contract.py
still run against them.

## Sentences: after the parsimonious-sentences arc

- `Word.components` (curated): a compound's registered parts, so a
  sentence using โรงพยาบาล ("hospital") also mentions โรง ("building")
  where the curator says so; fills clause 1 then reads "in the clauses
  or a component of a word in them" (spec 1 r8 log). Deferred until a
  registered compound and its registered part both matter.
- Fill-set memo keyed by text_sha ignores voice; `met_by` scans every
  adopted sentence per candidate (2 s at 400 sentences); `check_sentence`
  rebuilds the registered-id set per call.

## Deferred

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
(gitignored; rulings, sections, sequencing). Next act: the implementation
plan for step 1, then execution in order.

1. Spec 5 r7, the screen fixes: a rate question requires a candidate
   (else the need waits, or arrives as a direction request once
   exhausted); recordings render as audio players; rejected candidates
   show their deciding verdict; the empty-artifact text names the check.
2. Adjudication pass (spec 3): one judge batch over `disputed` words,
   cross-checked by thaig2p and the tone engine; `adjudicated` joins
   Corroboration; pronunciations written to words.yaml. Unblocks the 246
   disputed vocabulary words (glue words included).
3. Inventories: data/thai_consonants.yaml, data/thai_vowels.yaml;
   confusions.yaml from data/contrasts.yaml with weights (spec 1:
   SoundConfusion.weight; aspiration triples split; final:unreleased
   dropped).
4. Consonant graphemes: rows, acrophonic keyword Words, name Words with
   both Targets ordered after their grapheme, the `glyph` backend's
   chart cell (spec 1, spec 3).
5. Keyword search for vowel signs and tone marks (spec 3).
6. Pair search (weight-proportional, vocabulary first, recordability
   through Forvo), the retire key, coverage/sound-stage (spec 3, spec 5).
7. Principles F6: the recited names are learned as speech.

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
