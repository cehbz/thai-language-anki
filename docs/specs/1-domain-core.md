# Spec 1: Domain core

Revision 1, promoted 2026-09-04 as written on 2026-09-02 against the
principles draft. Re-checked against principles r1 and architecture r1
on 2026-09-04; the revisions that re-check proposed enter as r2 on
approval. Revision process as in docs/architecture.md: proposals on
evidence, explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.

Scope: the entities, values, the Syllabus aggregate and its operations,
and the rule model. Persistence formats are spec 2; port mechanics spec 3;
Anki translation spec 4; UI spec 5. Language: Python 3.12, one package
(the two-package split does not survive; the evaluator/generator boundary
is gone). Names below are the ubiquitous language — code uses them
verbatim.

## 1. Entities

Frozen dataclasses unless noted; identity fields marked. Thai strings in
examples are always accompanied by gloss in comments/docstrings (project
rule).

```
Word                                # language model
  id: WordId                        # identity: the sense, stable slug
  thai: str                         # written form
  pron: Pronunciation               # spoken form
  meaning: str                      # today rendered as the English gloss
  classifier: WordId | None         # nouns: unmarked colloquial classifier

Pronunciation
  syllables: tuple[Syllable, ...]   # segments, vowel length, Chao tone
  corroboration: Corroboration      # see rules R-PRON; uncorroborated
                                    # words exist but block card emission

SoundConfusion                      # language model
  id: ConfusionId                   # e.g. "tone:mid-low"
  dimension: Literal[tone, length, aspiration, vowel_quality, consonant]
  sounds: tuple[str, str]           # the two opposed values

Grapheme                            # language model
  symbol: str                       # identity
  kind: Literal[consonant, vowel_sign, tone_mark]
  sound: str
  consonant_class: Literal[mid, high, low] | None
  keyword: WordId                   # invariant: keyword's thai contains symbol

Target                              # curated learning list; the unit of
  id: TargetId                      # ordering and coverage. identity
  word: WordId
  skill: Literal[receptive, productive]
  introduction: Literal[picture_card, sentence]   # default picture_card

MinimalPair
  id: PairId                        # identity
  confusion: ConfusionId
  members: tuple[WordId, ...]       # 2..3
  # invariant (constructed): members' pronunciations differ in exactly
  # the confusion's dimension and values; loaded data re-checked by rule

Sentence                            # artifact
  text: str                         # identity together with provenance
  voice: Literal[learner_voice, other_voice]
  provenance: Provenance
  # which Targets it fills is DERIVED (Syllabus.fills), never stored
```

Media artifacts (Picture, Recording) are content-addressed values:

```
Picture   { sha: str, provenance: Provenance }
Recording { sha: str, provenance: Provenance, speaker: Speaker }
Speaker   { id: str, kind: Literal[native, synthetic] }
Provenance{ source: str, origin: str, licence: str, acquired: date }
```

Media relationships live on the consuming side and are derived from the
record (spec 2), never fields of the entities above: word→picture (single
current), sentence→scene-picture (optional), word→recordings,
pair→renditions (Rendition { speaker, recordings per member } — one
speaker across members), sentence→recording.

## 2. Learner profile

```
Profile
  register: Literal[male_colloquial]      # shapes generation prompts,
                                          # voice constraints
  emphasis: dict[Category, float]         # order tie-breaking, drafting
```

Confusion training weights are NOT stored here: derived as
seed (curated data) × StudyRecord evidence. L1 is implicit in curated
inputs. (Kam Mueang production parked; when unparked it enters here.)

## 3. The Syllabus aggregate

Owns: all Words, Targets, Graphemes, MinimalPairs, Sentences, the
Profile, and read access to the record/caches (spec 2 interfaces). All
cross-entity behavior:

**order() -> list[TargetLike]** — TargetLike = Target | PairId |
GraphemeId (the umbrella filed in review; a uniform ordering unit).
Constraints, each also stated as a rule: sounds stage (pairs, graphemes)
before words; a sentence's cards after every word it uses; receptive
target before productive target per word. Ties: frequency rank ÷ emphasis
weight. Pure; recomputed each call; the studied past is not consulted
(StudyRecords fix history, rules catch invalidated sentences).

**fills(sentence, target) -> bool** — the single definition:
1. target.word appears in sentence.text at a token boundary
   (tokenizer port; prefix/suffix compound membership counts),
2. sentence.voice satisfies target.skill (other_voice fills receptive
   only),
3. at the sentence's entry position (after its last word's target),
   every word it uses has an earlier Target — except one new word iff
   some filled target has introduction == sentence.
Used by generation as acceptance and by report() as coverage. Novelty
budget is a property of the fill set, not the sentence.

**report() -> Report** — runs every check on every note and every
measure on the aggregate. Report { syllabus_state_id, findings, metrics,
gate }. syllabus_state_id = hash of the aggregate's content; a report
whose state id differs from the live aggregate steers nothing (staleness
is structural, not advisory). gate = no unwaived error findings.

**gaps() -> Gaps** — derived from report metrics + target needs: missing
renditions per confusion (count × distinct speakers vs targets), unfilled
targets, words lacking pictures/recordings, graphemes lacking keyword
data. Input to the batch run (spec 3).

**compile() -> Compile** — spec 4; refuses when gate fails unless
explicitly overridden with declared warnings.

## 4. Rules

```
Rule
  id: str                           # e.g. "pair/exact-confusion"
  principle: str                    # e.g. "F1" — traceability, required
  severity: Literal[error, warn, info]
  shape: check | measure | judged
```

- check(note) -> list[Finding]; Finding { rule, note_id, artifact_sha?,
  evidence } — identity (note, rule, artifact) is what waivers reference.
- measure(syllabus) -> Metric { rule, value, detail }.
- judged rules carry rubric text; execution goes through the Assess port
  (spec 3); their findings derive from cached assessments, so report()
  never blocks on the judge.

Registry is explicit (a module-level list, no import side effects).
Traceability is itself a measure: every rule names a live principle,
every principle with enforcement intent names ≥1 rule; violations are
info findings on the rulebook.

Initial rulebook: the principles draft's lens rules (A2-A8, F1, F5, F6,
F7, F11, E1, E4, E5 as checks/measures; F3 gloss policy parked-variant).
Exact list enumerated at implementation-plan time against the locked
principles.

## 5. Explicitly out

- No Lexicon/Sequence/History classes (review decisions).
- No producer/filler classes: sourcing is spec 3's port + loop.
- No stored fills edges, weights, order, current-best, exhausted.
- Waiver as a store: replaced by learner assessments on finding identity.

## 6. Testing

Spec-level suite mirrors tests/spec today: doctrine tests written against
Syllabus.report()/order()/fills() with fake ports and builder-made
aggregates; entity invariants property-tested (pair construction,
grapheme keyword containment); fills() table-tested against the measured
tokenizer gotchas (compound membership, boundary cases). The existing
tests/spec suite is the behavioral baseline to port, not to import.
