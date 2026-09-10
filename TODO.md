# TODO

All items run against src/thai_syllabus (the redesigned pipeline; its
standard is docs/principles.md, docs/architecture.md and docs/specs/).
The old packages (thai_deck_eval, thai_deck_gen) stay; their spec-level
tests in tests/spec/test_deck_doctrine.py and test_generator_contract.py
still run against them.

## Review closure, remaining

- Rendition escalation anchor: a rendition's outcome row records member
  recording shas while current_best.artifact_sha is the rendition
  identity, so _anchor_ts never anchors a rendition and every rendition
  attempt counts since the beginning. Record the rendition identity in
  the outcome row's candidates (or anchor on the rendition verdict row).

## Spec 3 r11 candidates (user decisions)

- Picture re-ask after a live search: s5 Picture re-asks unconditionally,
  s6a says "from a cached answer"; one wasted search per attempt today.
- A cap on judge re-asks of unparseable answers (a typed `unparseable`
  verdict that never ranks after N), and a cost-only row for the tokens
  such an answer spent (s2 cost contract vs s6a "append nothing").
- s6 `candidates` reads "stored and checked"; the outcome row is written
  before the check on every attempt path, and an excluded check leaves
  `candidates` standing. Reword to "stored", or move the row.
- QueueEntry.attempts excludes sources at the transient cap while
  ExhaustedStatus.attempts includes them (screen order only).
- A drafter answer with zero drafts (`{"sentences": []}`) is refused
  under s2's no-drafts rule and re-asked every run; an honest "no
  sentence fits" answer needs a recognized shape that caches.
- An empty Forvo lookup is `nothing`, definitive for the subject (s6a),
  yet Forvo gains recordings over time: the 333 migrated empty lookups
  from 2026-08-29 never ask again. Whether `nothing` from a growing
  source ages out (a re-ask interval per source) is a spec decision.

## Deck safety (user, 2026-09-09)

- curated/ as a git repo inside the deck: every writing command (migrate,
  run, import, review) commits curated/ before and after; a db snapshot
  through sqlite's backup API to `<deck>/backup/` (one level); per-command
  sanity checks against the snapshot (words, targets, sentences and cache
  rows never fewer unless the command reports the removals by id; the
  run's identity); `thai-syllabus restore --deck D` as a human act. Spec 2
  revision to draft.

## Sentences: after the parsimonious-sentences arc

- Spec 1 §3: a productive target is filled only by a sentence whose last
  used word is the target's word (a productive target filled by a
  sentence clozed on another word yields no card).
- `Word.components` (curated): a compound's registered parts, so a
  sentence using โรงพยาบาล ("hospital") also mentions โรง ("building")
  where the curator says so; fills clause 1 then reads "in the clauses
  or a component of a word in them" (spec 1 r8 log). Deferred until a
  registered compound and its registered part both matter.
- Run-loop termination as a CLI feature (`run --cycles N`,
  `--spend-cap`): resolve, attempt, submit, wait, repeat until a run
  submits nothing or the cap is reached; the scratchpad runner is the
  interim.
- `ask_many`: warn when a key repeats within one call.
- Wire-failure messages interpolate the request URL, which carries the
  Google TTS and Forvo API keys as query parameters (tts.py, provider.py);
  a timeout can put a key in a log. Send keys as headers or redact them
  in the message.
- Fill-set memo keyed by text_sha ignores voice; `met_by` scans every
  adopted sentence per candidate (2 s at 400 sentences); `check_sentence`
  rebuilds the registered-id set per call.
- Glue-word pronunciations are disputed placeholders; they join the
  adjudication pass with the 221 migrated words.

## Deferred from the assess-first and judge-key arcs

- Escalation anchor: `_anchor_ts` takes current-best rubric-agnostically;
  a legacy pass and a fresh pass tie at 50 and the tie now breaks by
  sha, so a word whose legacy sha sorts first anchors at -1. Prefer the
  tied sha that has a producing attempt row.
- Review screen: a need kept queued for an awaiting candidate is listed
  both as a queue item and as a direction question (reviewserver
  ~272-278; pre-existing for directed needs).
- `Assessor.resolve` matches results by rebuilding keys; storing the
  submitted custom ids in the marker makes a key-shape change lossless.
- The inline judge path does not dedupe a repeated question in one
  `ask_many` call (the second overwrites `resolved[key]` as a hit; its
  ask and cost are lost from the spend).
- `assess_first` logs but does not report the excluded items when every
  awaiting candidate is excluded; the run report never sees them.
- Rendition escalation anchor (above) still open.

## Cutover

- migrate's curated-present guard keys off words.yaml alone (a half-written
  curated dir counts as present).
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
