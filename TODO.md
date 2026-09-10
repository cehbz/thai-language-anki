# TODO

All items run against src/thai_syllabus (the redesigned pipeline; its
standard is docs/principles.md, docs/architecture.md and docs/specs/).
The old packages (thai_deck_eval, thai_deck_gen) stay; their spec-level
tests in tests/spec/test_deck_doctrine.py and test_generator_contract.py
still run against them.

## Spec 3 r19 (proposed 2026-09-10, in the spec file; awaits approval)

- Decides: no re-ask of a live search (§5); the judge's answer is the
  last JSON object and the prompts ask for it alone (§2); `candidates`
  means stored and outcome rows never rank (§6); the queue counts
  attempts as exhausted() does (§6); a text's first gloss stands, only
  differing clauses reject, the drafting prompt lists refused texts, a
  no-fit answer caches and escalates a target to the learner after
  `sentence_nothing_cap` (§5); `nothing` from a growing source ages out
  (`quotas.<source>.nothing_ttl_days`, forvo 180) (§6a, §9). An
  implementation plan follows approval.
- Forvo growth measurement for the ageing interval: re-look up 30 of the
  333 empty migrated lookups after the 22:00 UTC reset and count how
  many gained a recording since 2026-08-29.
## Sentences: after the parsimonious-sentences arc

- Spec 1 §3: a productive target is filled only by a sentence whose last
  used word is the target's word (a productive target filled by a
  sentence clozed on another word yields no card).
- `Word.components` (curated): a compound's registered parts, so a
  sentence using โรงพยาบาล ("hospital") also mentions โรง ("building")
  where the curator says so; fills clause 1 then reads "in the clauses
  or a component of a word in them" (spec 1 r8 log). Deferred until a
  registered compound and its registered part both matter.
- Fill-set memo keyed by text_sha ignores voice; `met_by` scans every
  adopted sentence per candidate (2 s at 400 sentences); `check_sentence`
  rebuilds the registered-id set per call.
- Glue-word pronunciations are disputed placeholders; they join the
  adjudication pass with the 221 migrated words.

## Deferred

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

- Gloss placement (F3): picture-only fronts for every word by default;
  month and weekday names are the pre-decided exceptions; picture
  conventions (pointing figures, the two-position clock, object pairs,
  the 10×10 square) belong in the picture query hints; a word whose
  pictures keep failing escalates through the screen, where gloss-on for
  that word is one answer (user, 2026-09-10). Study impression pending.
- Classifier fix table (2026-09-10, from the judge triage): about 26
  words.yaml edits, unapplied; rainbow keeps ตัว (casual), live music
  keeps วง, menu keeps เล่ม; apply on the user's word.
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
- Sentence corpus: 234 of 822 targets unfilled on 2026-09-10; the
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
