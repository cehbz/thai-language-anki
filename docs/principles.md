# Principles

Revision 1, approved 2026-09-04. Drafted 2026-09-01, rewritten 2026-09-02
in the domain language of the entity pass. The architecture
(docs/architecture.md) is the companion.

Three meta-rules from the charter; every principle traces to one; every
rule operationalizes a principle, and the traceability check runs both
directions (a rule without a principle is cruft; a principle without a
rule is unimplemented doctrine).

Status marks. A principle without a mark is locked. **[provisional:
<evidence>]** names the evidence that settles it; the principle binds the
design now and is the first candidate for revision when that evidence
arrives. **[evidence: study 09-02]** cites the study note that established
the principle.

## Revision process

- A revision is proposed when evidence exists (study notes, StudyRecords,
  a measurement, a user decision), never on speculation. The proposal
  names the principles it changes, the evidence, and every rule the
  traceability check finds affected.
- Each revision is approved explicitly, as revision 1 was. No revision is
  implied by a spec, a plan, a commit, or an approval of anything else.
- The revision number increments; the log below records what changed and
  why. The specs cite the revision they were written against and are
  re-checked against a new one.
- Provisional marks are the first candidates for revision; a locked
  principle can be revised on the same terms.

Revision log:
- r1 2026-09-04: draft of 09-02 locked; open items marked provisional;
  E7 added from the 09-02 study readout on voices.

## Lens 1 — "Is this a well-formed Anki deck?"

- **A1.** Anki's domain is adopted at the boundary, unmodified: note, card,
  template, guid, due, tags, scheduling. A Compile is a pure translation
  of Syllabus state into it; nothing on our side re-invents scheduling.
- **A2.** Identity is stable across Compiles: a Word target's note identity
  derives from the Word; a sentence card's from (Target, Sentence), so a
  replaced sentence resets its scheduling (texts are not fungible) while
  everything else updates in place. Model ids are stable; fields append.
- **A3.** No two cards of a note share a front; a note's first field is its
  identity, unique within its model.
- **A4.** Every media reference resolves; filenames are sync-safe.
- **A5.** New-card order is the Compile's decision (due, from Syllabus.order());
  review scheduling is Anki's. The two cards of one MinimalPair rendition
  are separated (offset due, bury-siblings shipped in the options group);
  different pairs of a confusion may interleave. [evidence: study 09-02]
- **A6.** The Compile carries the tags StudyRecords need to map every review
  back to its Target, SoundConfusion, card kind, and compile id.
- **A7.** Compiles are frequent and cheap; import never duplicates or silently
  drops; the compile refuses on gate failure or ships declared warnings.
- **A8.** The deck styles its own cards: legible Thai, the answer visually
  distinct from distractors, images constrained to the viewport.
  [study 09-02: unreadable font, indistinguishable pair answer]

## Lens 2 — "Does it implement Fluent Forever (as re-derived)?"

- **F1.** **Sound system first.** [provisional: study, pair difficulty] SoundConfusions are trained by MinimalPairs
  with native renditions (one speaker per rendition across members —
  never let the voice carry the answer). Coverage per trained confusion =
  pairs × distinct speakers; targets ≥ N pairs × ≥ M speakers (N, M in
  the rulebook); the trained set and weights come from the learner
  profile seed, re-derived from StudyRecords when they exist. *Current
  deck: one pair per confusion, mostly one speaker — a demonstration.*
- **F2.** **Concrete vocabulary before grammar.** Word targets ordered by
  colloquial usefulness (frequency blend × emphasis) inside semantic
  spread (category coverage as a measure).
- **F3.** **A picture carries meaning; a gloss fixes it.** The picture is the
  prompt on production; a short L1 gloss may appear on any back, and on
  a front only where no picture can fix the sense. Which words carry
  front glosses is recorded, not ad hoc. **[provisional: gloss-off study pass or Anki study]**
- **F4.** **Finding media is the machine's job; the learner rates, guides, and
  curates last.** Feedback screen per subject: gloss, current artifact
  with the judge's verdict, rejected candidates at judgeable size
  (presentation at card size — the presentation is part of the
  question), the query read-only. Ratings: unacceptable-none /
  unacceptable-use-this / acceptable / good, optional note. First mode
  is a proof gallery: every card front/back in introduction order,
  sequential, no scheduling. Anki flags feed the same record.
- **F5.** **Picture cards introduce; sentences exercise** (locked 09-02).
  Orthodox FF for the base stage, applied to the whole colloquial core;
  sentence-led introduction (Wyner-style) remains available as a Target
  property. A Sentence's permitted new-word count: 1 if it fills an
  introducing target, else 0 — every other word has an earlier Target,
  uniformly, no exemption list; glue words get early receptive Targets.
  word_form targets carry their novelty in the construction, words all
  known. One shared definition Syllabus.fills(sentence, target) (word at
  token boundary, voice constraint, met-at-entry) serves generation and
  report alike. Exercise latency per target is a measure (parsimonious
  multi-fill texts must not drift a word's exercise months late).
- **F6.** **Graphemes teach reading, not writing.** One card per grapheme:
  symbol → sound + keyword, showing the keyword Word's own picture
  (F6a: one picture per word, everywhere — structural, since only Word
  has a picture relation). Writing is incidental, never a family.
  Consonant keywords are the acrophonic words; vowels and marks take
  concrete picturable keywords. Tone-rule material is card-back
  reference, never tested; a word's tone is memorized with the word.
- **F6b.** A pair card's back shows both members, marks the stimulus, and offers
  each member's audio individually. [evidence: study 09-02]
- **F7.** **Native audio on anything tone-bearing the learner must produce.**
  Renditions native always; word recordings native; a Sentence filling
  any productive Target carries male native audio (self-grading
  register); receptive-only Sentences may be TTS. [decided 09-02]
- **F8.** **Order = usefulness in daily speech; then SRS.**
  [provisional: study, intro order feel] Syllabus.order() is
  the one implementation; constraints (sounds early, sentence after its
  words, receptive before productive per word) are rules; frequency ×
  emphasis breaks ties. Derived, never stored; the studied past is fixed
  by StudyRecords, the unstudied future reorders freely and rules catch
  what a reorder invalidates.
- **F9.** **Authority and memory are cache policy.** The learner is the most
  expensive backend of both ports: answers keyed on (artifact, role) —
  no rubric in the key, so rubric changes cannot touch them; never
  evicted; authoritative on conflict. Re-asked only by role change
  (mechanical miss), StudyRecord contradiction (evidence shown, may
  reopen "good"), or the learner re-rating (newest wins). Machine
  verdicts are memoization keyed on all inputs including rubric; a
  changed rubric is a new question. Nothing overrides a learner answer;
  things queue questions.
- **F10.** **Sourcing is periodic batch over the whole Syllabus**, ordered by
  expected gain per unit of Budget (per-backend caps), escalating
  backends in cost order, the learner last. Exhausted-for-now = only
  the learner remains; any learner input reopens. The queue is derived
  each run; every run reports attempted / improved / exhausted against
  available.
- **F11.** **No unjudged artifact on a card; no query that cannot describe its
  object.** A missing picture is a gap; a wrong one is a lie memorized.
  Provider caches are never evicted and double as the attempt record
  (the question asked is always on the record).
- **F12.** **Productive practice for what the learner intends to say.**
  [provisional: user decision on the productive-Target selection rule] Every
  Sentence yields receptive cards; productive Targets exist for
  high-usefulness spoken vocabulary; receptive Target precedes
  productive per word; productive new-card rate capped. (Research
  2026-09-02: practice pays in its direction; production is the burnout
  driver; no source states a ratio.)

## Lens 3 — "Does it teach Thai, to this learner?"

- **E1.** The learner does not read Thai. No card front requires reading before
  its graphemes are introduced; script-only fronts are staged after the
  spelling-sound material they use.
- **E2.** Production is checked by ear: productive backs carry native audio in
  the learner's register; the learner grades against it.
- **E3.** Register: colloquial Central Thai; the profile (male, colloquial)
  shapes generation; other-voice material fills receptive Targets only;
  standard spelling on the page, reduction carried by audio.
- **E4.** A pronunciation is data the learner drills: it must be corroborated
  (engines agree, or curated exception) before a card asserts it.
- **E5.** Classifiers are Words; a noun's unmarked colloquial classifier is a
  Word attribute, taught through counting sentences (word_form cloze),
  displayed as reference on backs. Measure words and register variants
  are constructions and register, not noun attributes.
- **E6.** Evidence closes the loop: StudyRecords (kept across regenerations,
  keyed by card content identity + compile id) feed confusion
  reweighting and learner-cache re-asks; until they exist, the proxies
  are the report's measures and the learner's study notes.
- **E7.** **Comprehension needs many voices.** Everything the learner hears is
  reception, productive backs included. Speaker diversity is a coverage
  measure over each audio corpus: sex, age band, and regional accent,
  counted over distinct speakers against rulebook targets. Receptive
  audio spans all of them. Productive audio keeps the learner's register
  in the text and a male voice on the recording (F7, E2) and varies the
  speaker within that where the pool allows. A Speaker carries those
  attributes when known and "unknown" otherwise; unknown never counts
  as coverage. F1's speakers-per-confusion is this measure applied to
  renditions. Diversity never overrides native-audio requirements (F7)
  or one speaker per rendition (F1). Pools are providers config; TTS
  supplies sex and timbre only, Forvo and commissions supply age and
  accent.

## Card taxonomy (as the principles imply)

| skill | front | back | source entity |
|---|---|---|---|
| discriminate | rendition audio (one member) | both members marked, stimulus marked, each audio clickable | MinimalPair |
| hear → meaning | word audio | picture, Thai, IPA | Word target |
| picture → say | picture | Thai, audio, IPA | Word target |
| read → meaning | Thai script | picture, audio | Word target (staged post-graphemes) |
| symbol → sound | grapheme | sound, keyword + its picture, audio | Grapheme |
| produce in context | cloze sentence (+scene picture) | target word, native audio | productive Target × Sentence |
| understand in context | sentence audio | text, target, gloss | receptive Target × Sentence |

## Decisions on record

Study run of 09-01 disposable (scheduling only; StudyRecords kept).
Sentence = artifact; Target = (word, skill); texts fill multiple targets;
(target, text) identity. Grapheme reading-only. Strict i+1. Productive
sentences native male audio. Scene pictures optional per Sentence,
budget-prioritized. Syllabus aggregate root; Compile value; Anki domain
adopted. Learner profile = register + emphasis.

## Provisional, by evidence awaited

Study: F3 gloss placement (gloss-off pass or Anki study); F1 minimal-pair
difficulty; F8 intro order feel; TTS acceptability on receptive sentences
(09-02 notes: acceptable on two voices).
User decision: F12 productive-Target selection rule.
Research, not principles: register questions in TODO (Kam Mueang, the
casual first-person pronoun's neutrality, particle spelling).
Parked: meaning-vs-gloss.
