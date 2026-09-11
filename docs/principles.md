# Principles

Revision 4, proposed 2026-09-11 (r3 approved 2026-09-11). The architecture
(docs/architecture.md) and the specs (docs/specs/) are the companions.

Three meta-rules from the charter; every principle traces to one; every
rule operationalizes a principle, and the traceability check runs both
directions (a rule without a principle is cruft; a principle without a
rule is unimplemented doctrine).

A principle states what must hold and why, in terms that survive a
reimplementation. Mechanisms (data shapes, keys, screens, functions)
live in the architecture and the specs, which cite the principle.

Status marks. A principle without a mark is locked. **[provisional:
<evidence>]** names the evidence that settles it; the principle binds the
design now and is the first candidate for revision when that evidence
arrives. **[evidence: ...]** cites what established the principle.

## Revision process

This process governs every document under docs/: the principles, the
architecture and each spec, each with its own revision number and log.

- A revision is proposed when evidence exists (study notes, StudyRecords,
  a measurement, a user decision), never on speculation. The proposal
  names what it changes, the evidence, and every rule and spec the
  traceability check finds affected.
- Each revision is approved explicitly. No revision is implied by a
  spec, a plan, a commit, or an approval of anything else.
- The revision number increments; the log records one line per revision:
  what changed. Evidence and measurements live in the knowledge base;
  the log names them. The specs cite the revision they were written
  against and are re-checked against a new one.
- Provisional marks are the first candidates for revision; a locked
  principle can be revised on the same terms.

Revision log:
- r1 2026-09-04: draft of 09-02 locked; E7 added.
- r2 2026-09-04: mechanism sentences moved to the architecture and specs;
  decisions-on-record section retired; F2's no-study-grouping sentence.
- r3 2026-09-11: mechanism sentences r2 left in F1, F5, E7 moved out; A2's
  sentence-card identity is the text (spec 4 r5); F7 absorbs E2's
  content (E2 kept as a pointer for traceability); F12's selection rule
  is decided (spec 1 r9); E3 records the speaker-marking ruling (spec 1
  r10); the `word_form` target kind, which no spec defines, removed from
  F5 and E5; the card taxonomy table moved to spec 4 §1; the open-items
  section moved to TODO.md. No other principle's meaning changed.
- r4 2026-09-11: F13, nothing is grandfathered. Evidence: 15 adopted
  sentences whose every recording exceeded the duration cap sat in the
  deck with no path to a recording; user ruling 2026-09-11.

## Lens 1 — "Is this a well-formed Anki deck?"

- **A1.** Anki's domain is adopted at the boundary, unmodified: note, card,
  template, guid, due, tags, scheduling. A Compile is a pure translation
  of Syllabus state into it; nothing on our side re-invents scheduling.
- **A2.** Card identity is stable across Compiles, so scheduling survives
  regeneration. Identity derives from what the card teaches: a word card
  from its Word, a sentence card from its text, so a replaced sentence is
  a new card (texts are not fungible) while everything else updates in
  place.
- **A3.** No two cards share a front: a front the learner cannot tell
  apart cannot be graded.
- **A4.** Every media reference resolves; filenames are sync-safe.
- **A5.** Introduction order is ours; review scheduling is Anki's. The
  cards of one rendition are never studied back to back (the second would
  give the first away); different pairs of a confusion may interleave.
  [evidence: study 09-02]
- **A6.** Every review maps back to what it taught (Target,
  SoundConfusion, card kind) and to the Compile that produced the card.
- **A7.** Compiles are frequent and cheap; import never duplicates or silently
  drops; the compile refuses on gate failure or ships declared warnings.
- **A8.** The deck styles its own cards: legible Thai, the answer visually
  distinct from distractors, images constrained to the viewport.
  [evidence: study 09-02: unreadable font, indistinguishable pair answer]

## Lens 2 — "Does it implement Fluent Forever (as re-derived)?"

- **F1.** **Sound system first.** [provisional: study, pair difficulty]
  SoundConfusions are trained by MinimalPairs with native renditions, one
  speaker across the members of a rendition so the voice never carries
  the answer. Which confusions are trained, and how heavily, comes from
  the learner profile and, once it exists, the learner's study evidence.
- **F2.** **Concrete vocabulary before grammar.** Word targets ordered by
  colloquial usefulness (frequency blend × emphasis) inside semantic
  spread (category coverage as a measure). A category measures spread
  and weights emphasis; it never groups study, since same-category
  batches impair retention (Tinkham 1993, 1997; Waring 1997; Erten &
  Tekin 2008; Wyner's own guidance; research log 2026-09-04).
- **F3.** **A picture carries meaning; a gloss fixes it.** The picture is the
  prompt on production; a short L1 gloss may appear on any back, and on
  a front only where no picture can fix the sense. Which words carry
  front glosses is recorded, not ad hoc. [provisional: gloss-off study
  pass or Anki study]
- **F4.** **Finding media is the machine's job; the learner rates, guides,
  and curates last.** The learner is the most expensive source and judge,
  asked only after the machine has tried: to rate, to direct, to supply,
  never to search. The learner judges an artifact as the card will show
  it; the presentation is part of the question. Every channel of learner
  feedback lands in one record.
- **F5.** **Picture cards introduce; sentences exercise.** Orthodox FF for
  the base stage, applied to the whole colloquial core; sentence-led
  introduction (Wyner-style) remains available as a Target property. A
  sentence introduces at most one word, and only a word whose Target
  introduces by sentence; every other word it uses has an earlier
  Target, uniformly, no exemptions. Generation and measurement share one
  definition of "this sentence serves that target". A text that serves
  several targets must not drift a word's exercise months late.
- **F6.** **Graphemes teach reading, not writing.** One card per grapheme:
  symbol → sound + keyword, showing the keyword Word's own picture
  (F6a: one picture per word, everywhere). Writing is incidental, never
  a family. Consonant keywords are the acrophonic words; vowels and marks
  take concrete picturable keywords. Tone-rule material is card-back
  reference, never tested; a word's tone is memorized with the word.
- **F6b.** A pair card's back shows both members, marks the stimulus, and
  offers each member's audio individually. [evidence: study 09-02]
- **F7.** **Native audio on anything tone-bearing the learner must
  produce, and production is checked by ear.** Renditions and word
  recordings are native; a Sentence filling a productive Target carries
  native audio in the learner's register, and the learner grades
  production against it; receptive-only Sentences may be TTS.
  [evidence: decided 09-02]
- **F8.** **Order = usefulness in daily speech; then SRS.**
  [provisional: study, intro order feel] Constraints first: sounds early,
  a sentence after its words, receptive before productive per word;
  usefulness (F2) orders within them. Order is derived, never stored: the
  studied past is fixed by the evidence, the unstudied future reorders
  freely, and what a reorder invalidates is caught, not hidden.
- **F9.** **A learner's answer is permanent and final; a machine's answers
  exactly the question asked.** A learner's answer is about an artifact
  in a role and outlives every rubric; it is never discarded and wins on
  conflict. The learner is re-asked only when the question changed (a
  new role), the evidence contradicts (shown), or the learner re-rates
  (newest wins). A machine verdict answers one question, rubric
  included; a changed rubric is a new question. Nothing overrides a
  learner answer; the system queues a question instead.
- **F10.** **Sourcing is periodic batch over the whole Syllabus**, spending
  where the expected gain per unit of budget is highest, cheapest sources
  first, the learner last. A subject is exhausted for now when only the
  learner remains; any learner input reopens it. Every run reports what
  it attempted, improved, and exhausted against what was available: a
  run that did almost nothing must be distinguishable from success.
- **F11.** **No unjudged artifact on a card; no query that cannot describe
  its object.** A missing picture is a gap; a wrong one is a lie
  memorized. Every question asked of a source is on the record, the ones
  that returned nothing included.
- **F12.** **Productive practice for what the learner intends to say.**
  Every Sentence yields receptive cards; productive Targets exist for
  high-usefulness spoken vocabulary, selected by frequency rank against
  a cutoff (spec 1 §2); receptive Target precedes productive per word;
  productive new-card rate capped. [evidence: research 2026-09-02,
  practice pays in its direction and production is the burnout driver;
  user decision 2026-09-10 on the selection rule]
- **F13.** **Nothing is grandfathered.** The deck is what the current
  rulebook says it is. An artifact, sentence or verdict that fails a
  current rule is a gap to re-source or a candidate to retire. Only a
  learner's answer outlives a rule change (F9).

## Lens 3 — "Does it teach Thai, to this learner?"

- **E1.** The learner does not read Thai. No card front requires reading before
  its graphemes are introduced; script-only fronts are staged after the
  spelling-sound material they use.
- **E2.** Production is checked by ear: see F7.
- **E3.** Register: colloquial Central Thai; the profile (male, colloquial)
  shapes generation; other-voice material fills receptive Targets only;
  standard spelling on the page, reduction carried by audio. A word that
  marks its speaker's sex (a particle, a pronoun) is voiced by a speaker
  of that sex on every card, and a sentence the learner produces is one
  the learner would say. [evidence: user rulings 2026-09-10]
- **E4.** A pronunciation is data the learner drills: it must be corroborated
  (engines agree, or curated exception) before a card asserts it.
- **E5.** Classifiers are Words; a noun's unmarked colloquial classifier is a
  Word attribute, taught through counting sentences and displayed as
  reference on backs. Measure words and register variants are
  constructions and register, not noun attributes.
- **E6.** Evidence closes the loop: study records survive regeneration and
  feed confusion reweighting and learner re-asks; until they exist, the
  proxies are the report's measures and the learner's study notes.
- **E7.** **Comprehension needs many voices.** Everything the learner hears is
  reception, productive backs included. Speaker diversity (sex, age band,
  regional accent, over distinct speakers) is a coverage measure on each
  audio corpus; receptive audio spans all of it; productive audio keeps
  the learner's register in the text and a male voice on the recording
  (F7) and varies the speaker within that. An unknown attribute never
  counts as coverage. Diversity never overrides native audio (F7) or one
  speaker per rendition (F1).
