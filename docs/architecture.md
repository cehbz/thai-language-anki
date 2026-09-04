# Architecture

Revision 2, proposed 2026-09-04 (r1 approved 2026-09-04). Written 2026-09-02 from the entity pass
and behavior walk of 2026-09-01/02. The principles (docs/principles.md)
are the companion: every rule traces to a principle, every principle to
one of the three charter meta-rules: is it a well-formed Anki deck / does
it implement re-derived Fluent Forever / does it teach Thai to this
learner.

## Revision process

- A revision is proposed on evidence (a principle revision, a defect the
  shape caused, a measurement), never on speculation. The proposal names
  the sections it changes, the evidence, and every spec the change
  reaches.
- Each revision is approved explicitly. No revision is implied by a
  spec, a plan, a commit, or an approval of anything else.
- The revision number increments; the log records what changed and why.
  Specs cite the revision they were written against and are re-checked
  against a new one.

Revision log:
- r1 2026-09-04: draft of 09-02 approved unchanged.
- r2 2026-09-04: compile moved from the aggregate's operations (§3) to
  the application services (§7); the aggregate keeps no storage
  dependency. Evidence: implementation review 2026-09-04.

## 1. Shape of the system

One aggregate (the Syllabus), two ports (Provide, Assess), an adopted
external domain (Anki), and a small set of durable stores. Everything else
is derived on demand and never persisted. The learner is a backend of both
ports — the most expensive one — not a special case in the model.

Interruption is the normal case: every durable write is an append and
every append is a checkpoint. Regeneration is the normal case: anything
not in §5's durable set is recomputed without loss.

## 2. Domain model

Language model (facts of Thai, learner-independent):

- **Word** — identity = its sense. Written form; spoken form (segments,
  length, tone); meaning; optional classifier (ref to a Word). No
  behavior. Casual/formal counterparts are separate Words.
- **SoundConfusion** — two sounds + the dimension they differ on. The
  aggregation key for pair coverage and lapse reweighting.
- **Grapheme** — symbol, kind (consonant/vowel sign/tone mark), sound
  value, class, keyword (ref to a Word). Reading direction only.

Teaching material (artifacts: produced, assessed, replaceable; learner
input protected):

- **Target** — (word, skill: productive|receptive) plus introduction mode
  (picture-card default; sentence-led allowed). The curated learning list;
  the unit everything orders and measures.
- **MinimalPair** — 2–3 member Words exhibiting exactly one
  SoundConfusion; carries renditions (one speaker across members).
  Members are Words via closure, not vocabulary.
- **Sentence** — Thai text + voice constraint (learner-voice/other-voice)
  + provenance. Fills one or more Targets; identity at the Anki boundary
  is (target, sentence) so replacing a text resets its scheduling.
- **Picture / Recording** — bytes (hash = identity) + Provenance (source,
  origin, licence, date); Recording adds speaker. All learning semantics
  live in the consuming relationships: Word→picture (single, reused
  everywhere), Sentence→scene picture (optional), Word→recordings,
  MinimalPair→renditions, Sentence→recording.

Values: **Provenance**; **Compile** (labeled snapshot of a compilation);
**learner profile** (register + emphasis — the only learner-relative
generation input); **Finding / Metric / Report** (outputs of the rule
runs); rubrics (the judged rules' question text).

- **Rule** — an operationalized principle; immutable value. Mechanical
  (code decides), judged (Assess port decides; its text is the rubric
  parameter), or measure-shaped (aggregates). The rulebook is the
  principles document made executable; traceability is checkable both
  directions.

## 3. The aggregate: Syllabus

The learner's course of study: Targets, Words and their facts,
MinimalPairs, Graphemes, Sentences, current-best artifacts. All
cross-entity behavior lives here:

- `order()` — the introduction order. Constraints as rules (sounds early;
  a sentence after all its words' targets; receptive before productive
  per word); frequency × emphasis breaks ties. Derived, never stored: the
  studied past is fixed by StudyRecords, the unstudied future reorders
  freely, and rules catch what a reorder invalidates.
- `fills(sentence, target)` — the one definition of "this text serves
  that target": word at a token boundary, voice constraint satisfied,
  vocabulary met at entry. Used identically by generation (acceptance)
  and reporting (coverage). Novelty budget: 1 if the sentence fills an
  introducing target, else 0. Strict — no exemption lists; glue words get
  early receptive Targets.
- `report()` — every check on every note plus every measure, for one
  syllabus state; the gate is a property of the report. Reports identify
  the state they judged; a stale report steers nothing.
- `gaps()` — what sourcing should produce next, derived from report
  measures and target needs.
- Compile is not an aggregate operation: the application service (§7)
  reads `report()`, `order()` and the current-best artifacts and
  translates them into Anki's domain (§6).

## 4. Ports and backends

Backends are distinguished by cost, cache key, and authority — authority
is per (backend, role).

**Provide** (media and content): image corpora (Openverse, Wikimedia,
Pexels), Forvo, TTS, commission, the LLM (sentence drafting, word-fact
drafting, query phrasing, keyword proposals), the pair search (dictionary
+ G2P over Thai at large; curated seeds; LLM proposal mechanically
verified), and the learner (supply — costliest). The old producer/filler
split does not exist: all of them answer "provide X for Y" under a query.

**Assess** (fitness of an artifact-in-role, a word fact, a finding, or a
card): the judge (LLM; keyed on rubric + artifact + role — the rubric is
a parameter, so nothing invalidates, a changed rubric misses naturally)
and the learner (keyed on artifact + role only; never evicted;
authoritative where qualified). Roles include: (picture, word) fit;
(scene picture, sentence); (sentence, target) naturalness/register;
(recording, word) QA — mechanical checks now, a listener backend only at
the rank calibration earns it; word facts (classifier, gloss,
pronunciation adjudication — engines as cheap oracles, LLM as better-read
oracle, disputes stay blocked, never guessed); findings (waiver); cards
(flags). The learner is final authority on picture fit and unqualified on
tone correctness — a flag there queues machine re-verification.

**Caching is backend policy, invisible to consumers.** Provider caches
are never evicted and double as the attempt record (entry = source,
query, result including empty, timestamp, subject): they answer
exhausted-for-now, F11 forensics, untried-lever queries, and cross-run
budget accounting. Learner-cache re-asking has exactly three channels:
role change (mechanical miss — the role is in the key; presentation at
card size is part of the question), StudyRecord contradiction (evidence
shown; may reopen "good"), learner re-rating (newest wins). Rubric-only
changes queue a challenger comparison; nothing overrides a learner
answer.

**Budget**: per-backend spend caps per run (requests, dollars, learner
attention). The sourcing loop escalates backends in cost order, learner
last; exhausted-for-now = only the learner remains; any learner input
reopens.

## 5. Durable state

1. **Curated data** — Targets; Word facts (to closure); Graphemes;
   SoundConfusion inventory and profile seed; the rulebook; learner
   profile. Human-owned, machine-drafted.
2. **Artifacts** — picture/recording/sentence content, content-addressed,
   with provenance.
3. **Port caches** — never evicted; the learner's answers and the attempt
   record live here by policy, not by kind.
4. **StudyRecords** — imported Anki reviews: (card content identity,
   compile id, timestamp, grade, time). Kept across regenerations;
   card-level evidence expires with its card, confusion/word-level
   aggregates are about the learner and survive.

Derived, never stored: order, met-at, current-best artifact, exhausted,
the sourcing queue, confusion weights (seed × StudyRecords), reports,
fills edges.

## 6. Boundaries

**Anki** — adopted wholesale (note, card, template, guid, due, tags,
scheduling); zero re-litigation. `compile()` translates: stable model
ids; guid from durable identity ((target, sentence) for sentence cards);
due from order(); sibling separation for renditions; tags carrying what
StudyRecords need to map back (target, confusion, card kind, compile id);
styled cards (A8); refuses on gate failure. Anki adapts scheduling only;
content adaptation is regeneration here — the reason StudyRecords flow
back.

**Feedback screen** — an application surface over both learner backends.
Modes: proof gallery (every card rendered front/back in introduction
order, sequential, no scheduling — the primary review instrument) and
per-subject screens (gloss, current artifact + judge verdict, rejected
candidates at judgeable size, query read-only; ratings
unacceptable-none/unacceptable-use-this/acceptable/good + optional note;
supply). Presents pictures at card size — the presentation is part of the
question.

**Anki imports** — revlog → StudyRecords; flags → learner assessments.

## 7. Application services (thin by design)

- **Batch run**: derive the queue (expected gain per unit of budget),
  iterate: escalate Provide backends, assess results, append. Reports
  attempted / improved / exhausted against available. Any thickness here
  is misplaced domain logic.
- **Feedback session**: serve the screen, append learner answers.
- **Import**: revlog and flags.
- **Compile**: gate on report(), translate order() and the current-best
  artifacts into Anki's domain, label.

## 8. What this architecture deletes from the current code

The five-plus memo stores (port caches with policy replace them); the
evaluator/generator subprocess boundary (rules are one library; Syllabus
is one aggregate); ImageNeed/AudioNeed; two known-vocabulary notions;
three rank-and-walk copies; producer/filler duplication; the exemption
list; separate waiver store; separate LLM/judge cache pairs; the sourcing
record's stored status (derived); Lexicon/Sequence/History as code
concepts.

## 9. Open at time of writing

Study-dependent: gloss placement, TTS acceptability, pair difficulty,
order feel. User decisions: productive-Target selection rule; classifier
findings in the word list. Research: register questions (Kam Mueang, ฉัน,
particle spelling). Parked: meaning-vs-gloss; exercise-latency measure
shape; batch set-cover generation; listener calibration; grapheme
spoken-name recordings; vowel/mark keyword choices.
