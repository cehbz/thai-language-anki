# Spec 1: Domain core

Revision 4, proposed 2026-09-05 against principles r2 and architecture
r2. Revision process as in docs/architecture.md: proposals on evidence,
explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: Category as a curated collection (F2); Speaker
  attributes (E7); Grapheme.name_word (spec 4); grapheme containment
  re-checked by rule; authority order and role map as domain values;
  initial rulebook enumerated against the locked principles.
- r3 2026-09-04: sentence identity is the text sha; Sentence.gloss;
  order() returns typed entries including sentences; gaps() derives from
  the report and covers sentence recordings and scene pictures; compile
  off the aggregate; frequency resolved by the loader. The r2 log
  overstated grapheme containment: the rule was already registered.
  Evidence: implementation review 2026-09-04.
- r4 2026-09-05: scene/fit joins the F3 row (a scene picture is judged
  against the sentence it illustrates). Evidence: Task B4 review found
  no scene rubric existed outside test fixtures.

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
  keyword: WordId                   # required (F6: the card shows it);
                                    # invariant: keyword's thai contains
                                    # symbol, checked at construction and
                                    # re-checked on loaded data by rule
  name_word: WordId | None          # the recited letter name as a Word
                                    # ("gɔɔ gài" for ก); consonants today

Target                              # curated learning list; the unit of
  id: TargetId                      # ordering and coverage. identity
  word: WordId
  skill: Literal[receptive, productive]
  introduction: Literal[picture_card, sentence]   # default picture_card

Category                            # curated learning list: a theme of
  name: str                         # the FF 625 list. identity
  members: frozenset[WordId]        # invariant: a word is in at most one
                                    # category (rule category/single-
                                    # membership); closure words (pair
                                    # members, keywords) are in none.
                                    # Consumers: coverage/categories (F2),
                                    # emphasis (Profile), the picture query
                                    # qualifier (derived reverse lookup;
                                    # absent for closure words)

MinimalPair
  id: PairId                        # identity
  confusion: ConfusionId
  members: tuple[WordId, ...]       # 2..3
  # invariant (constructed): members' pronunciations differ in exactly
  # the confusion's dimension and values; loaded data re-checked by rule

Sentence                            # artifact
  text: str                         # identity: sha of the text; provenance
                                    # is a fact of the row, not identity
  gloss: str                        # L1 gloss, drafted and judged with
                                    # the text (F3: a gloss on any back)
  voice: Literal[learner_voice, other_voice]
  provenance: Provenance
  # which Targets it fills is DERIVED (Syllabus.fills), never stored
```

Media artifacts (Picture, Recording) are content-addressed values:

```
Picture   { sha: str, provenance: Provenance }
Recording { sha: str, provenance: Provenance, speaker: Speaker }
Speaker   { id: str, kind: Literal[native, synthetic],
            sex: Literal[male, female, unknown],
            age_band: Literal[child, adult, older, unknown],
            region: str | unknown }          # E7; unknown never counts
                                             # as coverage
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
  emphasis: dict[CategoryName, float]     # order tie-breaking, drafting
```

Confusion training weights are NOT stored here: derived as
seed (curated data) × StudyRecord evidence. L1 is implicit in curated
inputs. (Kam Mueang production parked; when unparked it enters here.)

## 3. The Syllabus aggregate

Owns: all Words, Targets, Categories, Graphemes, MinimalPairs,
Sentences, the Profile, and read access to the record/caches (spec 2
interfaces). All
cross-entity behavior:

**order() -> list[OrderEntry]** — OrderEntry { kind: word_target | pair
| grapheme | sentence, id }: the one introduction order of everything the
learner meets. Constraints, each also stated as a rule: sounds stage
(pairs, graphemes) before words; a sentence after every word it uses;
receptive target before productive target per word. Ties: frequency rank
÷ emphasis weight; the loader resolves ranks through the FrequencyMap
port and the aggregate holds the mapping. Pure; recomputed each call;
the studied past is not consulted (StudyRecords fix history, rules catch
invalidated sentences). Consumers (compile, the screen) read positions;
none re-derives placement.

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

**gaps() -> Gaps** — derived from the report's completeness findings
and measures, never recomputed beside them: missing renditions per
confusion (count × distinct speakers vs targets), unfilled targets, words
lacking pictures/recordings, sentences lacking recordings, sentences
lacking an (optional, budget-prioritized) scene picture, graphemes
lacking keyword data. Input to the batch run (spec 3).

Compile is an application service (spec 4; architecture §7) over
report(), order() and the current-best artifacts; the aggregate has no
storage dependency.

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

Authority order per role (ordered backends, most authoritative first)
and the need-kind -> role map are domain values defined beside Rule, in
one module; assessment and the derivations (spec 3) consume them, never
define them.

Registry is explicit (a module-level list, no import side effects).
Traceability is itself a measure: every rule names a live principle,
every principle with enforcement intent names ≥1 rule; violations are
info findings on the rulebook.

Initial rulebook, against principles r2. "compile" = enforced by
Syllabus.compile() (spec 4), not a rule; "structural" = cannot be
violated by construction.

| principle | rules |
|---|---|
| A2, A5, A6, A7, A8 | compile |
| A3 | card/unique-front (check, error) |
| A4 | compile (a missing artifact drops the card, counted) |
| F1 | pair/exact-confusion, pair/rendition-required, rendition/synthetic, rendition/mixed-speakers, coverage/confusions |
| F2 | syllabus/closure, coverage/categories (measure), category/single-membership |
| F3 | picture/fit (judged), picture/preference (judged), scene/fit (judged, role scene-for-sentence), target/picture-required; front-gloss policy provisional |
| F5 | sentence/fills-novelty, target/sentence-required; exercise-latency (measure, parked) |
| F6 | grapheme/keyword-picture-required, grapheme/keyword-contains-symbol |
| F7, E2 | target/recording-required, sentence/recording-required, recording/synthetic, sentence/synthetic-productive |
| F8 | order/sounds-first, order/sentence-after-words, order/receptive-before-productive (checks over order()) |
| F11 | structural: current-best ranks judged candidates only |
| E1 | order/reading-after-graphemes (check over order()) |
| E3 | sentence/register-natural (judged) |
| E4 | word/pronunciation-corroborated (check, error; blocks card emission) |
| E5 | word/classifier-known (check, warn, nouns) |
| E7 | coverage/speakers (measure) |
| F4, F9, F10, F12, E6 | not rule-shaped (architecture and run behavior); F12's rate cap follows the selection decision |
| META-1 | rulebook/traceability |

The set of principles with enforcement intent is every row above with a
rule; the traceability measure reads this table.

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
