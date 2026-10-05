# TODO

All items run against src/thai_syllabus (the redesigned pipeline; its
standard is docs/principles.md, docs/architecture.md and docs/specs/).
The old packages (thai_deck_eval, thai_deck_gen) stay; their spec-level
tests in tests/spec/test_deck_doctrine.py and test_generator_contract.py
still run against them.

## Priority (user, 2026-09-17): by value to learning

1. Watch the hearing-first deck settle (principles r8, cutover
   2026-10-05: 1,304 notes, 1,790 cards in Anki; 3,339 cards staged
   out until their gates open). The pacer harvests reviews, sources the
   803 gapped clips and the 37 pairs' member pictures, and re-imports;
   check its log after the first studied days: Reading cards appearing
   at D, letters before them, AudioCloze cards once clips exist, and
   that no imported card ever blanks (the "Open after the cutover"
   items below are the known risks).
2. Sound stage part 2 (steps 5 to 7 below): vowel signs and tone marks
   are reading cards and come after the first spoken words under r8;
   3 words stay disputed; a pair member must be corroborated, nothing
   else.

## Open after the hearing-first cutover (arc review 2026-10-04)

- **Block drift.** A sentence adopted mid-order shifts every later block
  by one; cards already in Anki keep their dues while cards entering
  later use the new numbering, so the effective lags P and D grow with
  adoption (the importer never re-dues a card; pre-existing since spec
  1 r24). Fix candidates: re-due through AnkiConnect (`setDueDate`), or
  dues from an index that adoption does not shift.
- **Every re-import rewrites every note**: `CompileId` changes each
  compile. Detect an unchanged build by content hash before importing.
- After a notetype change `importPackage` returns true while skipping
  every note update, and the pacer logs "imported". Compare note
  fields or counts after the import.
- **Harvest cost on Anki's main thread**: `getDecks` with all card ids
  makes about two backend calls per card (7,564 for 3,782 cards) each
  tick, freezing the Anki window (unmeasured). Use `findCards
  "deck:…"` or look up only the reviewed cards.
- Every tick's harvest runs under the writing command: a `curated/`
  commit and the single `backup/syllabus.db` slot overwritten hourly.
- `cardReviews`' `since` cut is exclusive in ms; reviews synced in
  late from another device are missed (latent: no sync in use).
- An unreviewed Reading card is withdrawn when a pair is added to an
  unstable confusion it touches; a sentence-introduced word's
  readability can flip back (no Reading review to latch on).
- Two drafting asks per run can overrun an `llm-sentence` budget by
  one; gapped needs exist for fills beyond `CLOZE_SLOTS` (unreachable
  at `sentence_max_words` 8).
- `pacer.anki_connect_url` accepts any host (card contents and the
  package path go to it); an AnkiConnect `apiKey` is unsupported and
  reads as "collection locked".
- `GAP_BREAK` and the 0.6 s bound are two constants that must agree.
- Content: `season-colloquial`'s Meaning holds หน้าร้อน ("hot season"),
  so Thai leaks onto `face`'s Listening back through OtherSenses;
  สีเหลือง ("yellow") is curated as one syllable and its pair back shows
  [siː˨˩˦].
- The first seed ask's vocabulary is the five seed words themselves
  (no glue word is met within the span, which ends at position 118);
  measure what the drafter makes of it before widening the span.
- The Reddit half of the script-timing research failed on agy's quota;
  rerun `/agy-research` with the saved prompt when it resets.

## Sentence drafting: the judge on the subscription (2026-10-04)

The live judge runs on the cli transport (Sonnet 5.5, `--safe-mode`)
and the quota pacer (spec 3 r64, `tools/quota_pacer.py`, launchd agent
`com.ceh.thai-syllabus.pacer`, hourly, log `~/decks/thai-ff/work/pacer.log`)
releases judge calls against the week's unused allowance. Its first
tick (allowance 8.15 points, 40 calls) adopted 12 sentences. Reports:
.superpowers/sdd/2026-10-03-quota-pacer/task-1..3-report.md.

- **`points_per_call` 0.015 and `session_points_per_call` 0.05 are
  upper bounds**, not measurements: the usage probe reads whole
  percentage points, and 50 calls moved the weekly window by 0 and the
  5-hour window by 1. Tighten them from the pacer log once a few ticks
  have run (compare the probe before and after a 40-call tick).
- **Drafting asks are unpaced.** `--judge-asks` caps judge calls only;
  a drafting ask at the CLI's default model (Opus 5.5 with thinking,
  48 to 152 s, 5k to 17k output tokens) spends about ten safe-mode
  verdicts' worth, so a tick's quota is dominated by its one drafting
  ask. Either amortise it in `points_per_call` (done, roughly) or give
  the drafter its own cap.
- **40 judge rows of the 2026-10-04 12:09 tick carry a notional cost**
  (~$0.10 in all): the cli judge's tokens were priced by
  `judge.price_per_mtok`, meant for the batch judge. The fix is in the
  next commit; the rows stay as written.
- Under cli, `quotas.judge.max_cost` and `quotas.llm-*.max_cost` never
  bind (cost is 0); the loader could refuse `max_cost` on a cli
  backend. The review screen's "spend per newly covered need" reads 0
  for a cli judge and shows nothing about quota. The persisted
  RunReport row keeps `{asks, cost}` per backend; tokens sit only on
  the answer rows.
- `--safe-mode` was measured on picture verdicts only (73% less input,
  verdicts within the judge's noise); its effect on sentence verdicts
  is inferred.
- Sonnet 5.5 against Sonnet 5 on 50 recorded verdicts: pictures 23/25,
  sentences 19/25, every sentence miss Sonnet 5.5 passing what Sonnet 5
  failed, and in 3 of the 6 Sonnet 5's reason was wrong about Thai.
  Sonnet 5.5's own repeat agreement on borderline pictures is 8/10.
- `save_providers_config` does not write the `pacer` block (only tests
  call it). The `drafter.model` value is not stripped of whitespace.
- **Under the batch judge the poll sets the cycle length;
  `--poll-seconds` is the lever.** At 60 s, doubling each poll, 7 of 8
  batches resolved at the third poll, about 190 s after submission
  (308 s at the 300 s default): 60% of a 49-minute run. When a batch
  ends is not recorded, only the poll that saw it.
- The adoption pass logs the same 31 refused drafts on record twice a
  cycle (27 mark both sexes, 3 name unregistered words, 1 does not
  match its clauses): 744 of the log's 924 lines. The mixed-sex ones
  also take up to 8 of the drafting prompt's 20 refused lines (spec 3
  r62), each reason quoting the draft's 64-hex text sha, since
  `check_sentence`'s message is passed through as is.
- imgfetch refuses Pexels originals over its 10 MiB cap (1 in the
  enrichment run at 10.9 MB, 7 in the fill run at 11.7 to 34.8 MB); a
  smaller Pexels rendition would fetch (inferred).
- The ask count restarts at any learner row on the word (spec 3 r19)
  and at the adoption of any sentence using the word, not only one that
  fills its open Target, so a word can be handed more than three times
  in a row. No Target on the live deck is affected: every
  sentence-introduced Target is receptive, and for all 82 the adopted
  sentences using the word are exactly those filling the Target.
- Spec 5 r18 says a classifier's direction question names its nouns "as
  the drafting prompt does", but the screen names them for every
  classifier Word and the prompt only for a sentence-introduced one: a
  picture-introduced classifier at the cap would show its every noun,
  54 for คน (person).
- The run still awaits a vetoed, unjudged candidate's verdict once
  before re-sourcing (spec 3 r55 follow-up).
- The test suite has no network guard (a conftest refusing non-local
  `getaddrinfo`); test_run_e2e was found calling api.pexels.com.

## Sentence cards: after spec 4 r11

- **The per-word cap can push a Cloze card the learner has not reviewed
  out of the build** while it stays in Anki with a blank front (spec 4
  section 1). Between the 2026-10-03 compiles before and after the
  enrichment run, its sentences, placed earlier, pushed 25 pairs on 23
  later sentences out under the cap; 6 of those sentences lost every
  Cloze card. Under r8 a text Cloze card enters only once every word is
  read, so none is in Anki yet.
- **803 text Cloze cards and 803 AudioCloze cards enter as new cards**
  as their gates open; nothing caps productive new cards (F12, under
  Doctrine divergences).
- `sentence/synthetic-productive` warns on 377 sentences.
- 124 sentences carry no Cloze card, and the scene pictures of 94 of
  them show on no card: the sentence Listening card shows none.
- A hand-declared sentence-introduced productive Target capped out of
  every earlier filler stays unfilled (spec 1 r26); none on the live
  deck.
- The import's bury check reads each card's home deck; whether Anki
  applies a parent deck's bury settings to a nested deck is unverified.
- Minor: `SCENE_FIT_RUBRIC` says "the target word" (rewording re-keys
  every verdict); `target_words` is defined only in spec 3's r54 log
  line; `attempts.joined` is a thin public helper; the comment pass does
  not show which Target a commented Cloze card blanked; the scene-picture
  re-ask pools lapses across a sentence's Cloze anchors.

## Corpus after enrichment (user decision)

Wanted counts (spec 1 r30): 10 sentences for the six function words of
frequency rank 20 or better (ไม่ "not", และ "and", ครับ, แล้ว "already",
ได้ "can", ค่ะ; user 2026-10-03), 5 to rank 100, 3 to rank 300, 2
below, classifiers 1. After the pacer's first tick two are short: และ
8 of 10, ค่ะ 9 of 10; the next ticks fill them.

- **The sentences adopted before the fill run never use the function
  words.** The 424 of the corpus's 499 sentences adopted before the
  2026-10-03 fill run use none of the 26 except ที่ ("at", 37 sentences)
  and นี้ ("this", 108). 286 of the 424 use none of the 26.
  - Every use of the other 24 is in the 75 sentences of the two
    2026-10-03 runs. Sentences per word: ไม่ ("not") 11, มี ("have") 14,
    มาก ("very") 9, แล้ว ("already") 8, ครับ (male polite particle) 8,
    ใน ("in") 7, ค่ะ (female polite particle) 6, เป็น ("be") 3, จะ
    ("will") 3, and 2 to 5 for each of the rest.
  - 427 of the 499 use none of the 24.

  Whether to redraft or extend the earlier corpus with these words is
  open (the fill-queue plan's design question).
- **Redundant sentences (measure only).** 20 adopted sentences fill no
  Target that an earlier-placed sentence does not already fill. 8 of
  them are from the enrichment run, and each of those adds a sentence
  toward a function word's wanted count. Counted against wanted counts,
  9 are surplus, all adopted before the fill run, such as กล้องแพง
  ("the camera is expensive").

## Same-spelling words

Each spelling group compiles one Listening, Reading and Spelling set on
its first picture-introduced Word (spec 4 r12); the 2026-10-03 compile
has no `card/unique-front` finding.

- A receptive-only member that does not carry the set compiles no
  note: to-measure (วัด "to measure", beside "temple") and
  banknote-colloquial (แบงก์ "banknote", beside "bank"). Its recording
  is still required, and a rate question on it shows "no card compiles
  for this subject yet".
- The review screen's `shown.picture` takes a card's first picture, so
  a carrier with no picture of its own would report another member's.
- Spec 1 says a Word with no Target is in no spelling group, while
  `spelling_group` returns its form's group for one.

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
  untested. If it is wrong, `consonant:d-t` at w5 is over-weighted.
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
(gitignored; rulings, sections, sequencing). Steps 1 to 4 are in (spec 5
r7; spec 3 r28 with spec 1 r13 and spec 2 r14; spec 1 r14 with the
inventories; step 4: 42 consonant graphemes with illustrator keyword
pictures and judged chart cells, spec 3 r41/r46). Next act: step 5.

5. Keyword search for vowel signs and tone marks (spec 3): 22 rows in
   `data/thai_vowels.yaml`, one LLM ask proposing a picturable word per
   sign, containment checked, then the keyword-picture and chart-cell
   pipeline the consonants went through.
6. Pair search (weight-proportional, vocabulary first, recordability
   through Forvo): in (spec 3 r47). Part 2 in (spec 1 r21, spec 3 r48,
   spec 5 r14).
   Ruling 2026-09-18: TTS renditions are allowed for pairs, one voice
   across both members (`_tts_rendition`), `rendition/synthetic` stays
   warn. Measured on the live deck: of the 61 pairs the weights want, the
   vocabulary can fill 3 with one Forvo speaker on both members, 28 with
   any Forvo, 61 with TTS; 453 of the 543 Forvo-covered words have one
   speaker. Improve later: a native same-speaker rendition replaces a
   synthetic one when found, and see the hired-recordings item below.
   Before the search: done -- the `vowel_length` three-way split is
   reverted to one `vowel_length:short-long` row (Task 1) and
   `pair/exact-confusion` is retired (Task 2, spec 1 r19: re-checked what
   `MinimalPair.create` already refuses).
7. Principles F6 (r8): the recited names are heard on the grapheme
   cards; saying them is a later skill with no card yet.
- Design §5's "stats shows sound-stage coverage" is unimplemented: nothing reads `coverage/sound-stage` (not `compute_stats`, the CLI, or the page).
- `rendition/synthetic` warns about a rendition that does not exist: with no pair-level rendition row, `_DbMediaIndex.rendition_provenance` falls back to the members' own word recordings, so a pair with no rendition and TTS member clips gets a synthetic-rendition finding, and `speakers_of("rendition")` counts speakers who gave none.

Run and screen, found while seating the alphabet (2026-09-19):
- **No resolve-only run mode.** Resolving an outstanding batch takes a
  full cycle; the workaround is every source capped to 0
  (`--backend-cap X=0` for pexels openverse wikimedia brave illustrator
  glyph forvo tts llm-sentence). Assess-first still collects questions on
  unjudged candidates, so a "resolve" cycle can submit a new batch; loop
  until it submits nothing. A `--resolve-only` flag is the fix.
- **A need the learner is meant to decide is still the machine's.** A
  vetoed picture need with a judge-passed candidate is open, so every
  cycle re-sources it and its new questions hide it from the screen
  (`queue()` skips a subject with a question in flight) until the batch
  resolves. F4 says that need is the learner's: stop sourcing a need whose
  newest candidates passed and await the rating.
- **Chart-cell rubric clause** (spec 3 r41 follow-up): the judge failed
  ฌ's cell once claiming the glyph was ถม, and 21 of 48 cells drawn from
  corpus photos; the cell should be judged as a composition (symbol
  legible, keyword picture is the keyword's), not as "a picture of the
  recited name".
- **Doctrine divergences** (tests/spec/test_deck_doctrine.py, DIVERGENCE):
  F7 (TTS on a productive sentence ships, warn), E3 (a female-marked word
  in a male voice ships; marking enforced at sourcing only), F12 (no
  productive-specific new-card cap). Each closes by a rule or by editing
  the principle.
- **A vetoed TTS word recording is re-synthesized in the same voice.**
  `_recording_attempt`'s tts branch picks `pick_voice(need.subject, pool)`
  with no veto filter, so `1` on a word's TTS clip dead-ends the need the
  way a pair's rendition did before spec 3 r48; give it the same
  next-unvetoed-voice rule.
- **First run after spec 3 r49:** on the live deck the pass re-verifies
  every current-best recording (about 1,300: every word and sentence clip
  is under the old duration-only key) and demotes about 51 word recordings
  plus 4 renditions; re-sourcing goes to TTS (every right-word Forvo clip
  was already downloaded as a candidate by the old code, so a demoted word
  with one keeps it and one without has nothing left at Forvo). Then
  re-run the join script (recorded word vs the deck's word over current
  Forvo recordings) and expect 0; `coat` (เสื้อโค้ท) is the one false
  demotion, a doubled tone mark in Forvo's headword.

Adjudication follow-ups (2026-09-21, after the dictionary oracle arc:
54 of 57 disputed words corroborated in one cycle, 3 stay disputed;
Wiktionary was consulted for 5 forms, 1 absent):
- **The adjudication pass never seals a disputed word on the oracles'
  own agreement.** It tests only the judge's verdict against the
  engines and the dictionary; for a newly adopted word two oracles
  agreeing on a whole reading is `engines_agree` without a judge (spec 3
  r45), but an existing disputed word gets no such check. พลาสติก
  (thaig2p and Wiktionary agree, judge's tone wrong) and ฤดูใบไม้ผลิ
  (tltk and Wiktionary agree, judge's tone wrong) stay disputed for
  that reason alone. One spec 3 revision to the Adjudication paragraph:
  materialize `engines_pronunciation` for every disputed word first and
  accept `engines_agree`; ask the judge only for what is left.
- **Rung 2: component reading against the dictionary** for a compound
  Wiktionary lacks (น้ำแข็งเปล่า: the judge reads เปล่า long, correctly;
  both engines read it short; the dictionary has เปล่า but not the
  compound). Segment into dictionary entries (longest match, or the
  space where there is one), concatenate, and let it corroborate the
  judge only when the syllable counts agree. Designed 2026-09-20, not
  built; measured need on the live deck: 1 word.
- **Rung 3: a second LLM family as a corroborating judge** is a
  principles revision (corroboration is the judge plus one engine); the
  residue it would serve is zero today. Not planned.
- The `wiktionary` line in the run's per-backend spend table is minted
  by `default_budgets` from `quotas.wiktionary` and never charged
  (asks=0 while rows are written); cosmetic. Its `judge` line never
  counts batch verdicts, because `_resolve_previous_batch` tallies no
  spend. It read `asks=0 cost=0.0000` in every cycle of both 2026-10-03
  runs, while the enrichment run's batch judging cost USD 0.99; the
  cycle line's `spent=` reads the record and includes it. A deck paced at 0 s never
  backs off from a 429 without a Retry-After header (the wiring default
  is 1 s). An `absent` dictionary row is final; Wiktionary grows, so a
  `nothing_ttl_days` for it may be wanted. Spec 3 §5's grapheme-adoption
  paragraph still describes a single engine.
- The run's batch wait is 21600 s; a 57-request Message Batch took
  6 h 38 min on 2026-09-21 and the run gave up one poll short. The next
  run resolved it. Raise `--max-wait-seconds` or make the resolve free.
- `derivations.confusion_weights(seed)` is a second home for
  `SoundConfusion.weight` and has no caller. `curated.py` imports
  `run.parse_day_starts` (move it out so `run.py` can import
  `save_words` at the top). `pair_count` above weight 5 yields one pair.
  The stats history normalises only `adjudicated`/`stayed_disputed` for
  old rows; a future report field needs the same or a column union.

## Content work (machine)

- Sentence corpus: the run's sentence attempt fills open Targets; a
  word carries at most three production cards (spec 1 r26).
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

## Dependabot cleanup (2026-09-22)

- CI's python job is red. `requests[socks]` is declared only in the `gen`
  optional-dependency extra, and `uv sync --frozen --all-groups` does not
  install extras — `--all-groups` covers dependency-groups only. requests is
  imported by src/thai_syllabus/{provider,tts,dictionary}.py and by four test
  modules, so collection fails with 16 errors. Adding `--extra gen` to
  .github/workflows/ci.yml fixes it (measured: 2286 passed, 5 deselected, 72 s).
- With `--extra gen`, one test still fails:
  tests/syllabus/test_phonology.py::test_default_engines_builds_the_engines_once.
  It fakes Thaig2p through the deferred import, but `default_engines()` also
  builds Tltk, which is not faked, and `Tltk.__init__` does `from tltk import
  nlp` — tltk lives in the `nlp` extra alongside torch and pandas. So the test
  reaches a real tltk despite its docstring's "nothing here loads torch".
  Either fake Tltk too (matches the stated intent) or mark it `integration`
  (the marker is defined as "real pythainlp/tltk", and two other tests in that
  file already carry it). `--all-extras` would work but drags torch into every
  CI run.
- The one open Dependabot alert is nltk <= 3.10.3 (model-artifact APIs bypass
  pathsec and touch files outside the allowed root). No patched version exists
  upstream, so it stays open rather than being dismissed.
