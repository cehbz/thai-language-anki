# Spec 1: Domain core

Revision 12, proposed 2026-09-11 against principles r4 and architecture
r3. Revision process: docs/principles.md.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: Category as a curated collection; Speaker attributes;
  Grapheme.name_word; authority order and role map as domain values; the
  initial rulebook.
- r3 2026-09-04: sentence identity is the text sha; Sentence.gloss; typed
  order() entries; gaps() derives from the report; compile off the
  aggregate; frequency resolved by the loader.
- r4 2026-09-05: scene/fit on the F3 row.
- r5 2026-09-08: orthographic marks carry no vocabulary;
  target/picture-required over picture-introduced words.
- r6 2026-09-08: fills' novelty is one unmet sentence-introduced target
  per sentence over the fill set; a word with only sentence-introduced
  targets carries no category.
- r7 2026-09-09: mention by token identity or decomposition (superseded r8).
- r8 2026-09-09: a Sentence is clauses of registered Words, its text the
  rendering, fills membership; the tokenizer leaves the domain.
- r9 2026-09-10: productive Targets derived from Profile.productive_cutoff;
  targets.yaml lists exceptions; Word.no_productive.
- r10 2026-09-10: Word.speaker and the sentence's marking; a productive
  Target filled only by a sentence clozed on its word whose marking admits
  the learner; seven by-construction rules retired.
- r11 2026-09-11: logs reduced to one line each (evidence in the knowledge
  base); entity comments reduced to fields and invariants, behavior in §3
  (`marking` stated there); the rulebook table lists live rules only,
  merging spec 3 §8's rows, with the by-construction principle and the
  severity-override path stated once; transition-era text (§5, §6)
  rewritten in the present. No behavior changed.
- r12 2026-09-11: `coverage/exercise-depth` (measure, F5) replaces the
  parked exercise-latency entry: adopted sentences per word with a filled
  Target. Evidence: 253 sentences use 669 words, 452 of them once.

Scope: the entities, values, the Syllabus aggregate and its operations,
and the rule model. Persistence formats are spec 2; port mechanics spec 3;
Anki translation spec 4; UI spec 5. Python 3.12, one package. Names below
are the ubiquitous language — code uses them verbatim.

## 1. Entities

Frozen dataclasses unless noted; identity fields marked. Thai strings in
examples are always accompanied by gloss in comments/docstrings (project
rule).

```
Word                                # language model
  id: WordId                        # identity: the sense, stable slug
  thai: str                         # written form; not unique across
                                    # Words: a form shared by several
                                    # senses is a homograph, and a
                                    # sentence names the sense
  pron: Pronunciation               # spoken form
  meaning: str                      # today rendered as the English gloss
  classifier: WordId | None         # nouns: unmarked colloquial classifier
  no_productive: bool = False       # withholds the derived productive
                                    # Target
  speaker: Literal[male, female] | None   # the sex a word marks its speaker
                                    # as (ครับ kráp, ผม pǒm male; ค่ะ khâ,
                                    # คะ khá, ดิฉัน dì-chǎn female); None
                                    # otherwise. Curated. Consumed by §3
                                    # marking()

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
                                    # A receptive Target is listed in
                                    # targets.yaml. A productive Target is
                                    # derived: every Word with a Category
                                    # ranked at or above
                                    # Profile.productive_cutoff carries one
                                    # ("<word>/productive", picture_card);
                                    # a targets.yaml row adds one below the
                                    # cutoff, `no_productive: true` withholds
                                    # one. Closure and unranked words carry
                                    # none unless listed. The loader refuses
                                    # a row that duplicates a derived Target.

Category                            # curated learning list: a theme of
  name: str                         # the FF 625 list. identity
  members: frozenset[WordId]        # invariant: a word is in at most one
                                    # category (the loader builds the
                                    # collections from each word row's one
                                    # category field); closure words (pair
                                    # members, keywords) and words whose
                                    # only targets are sentence-introduced
                                    # are in none

MinimalPair
  id: PairId                        # identity
  confusion: ConfusionId
  members: tuple[WordId, ...]       # 2..3
  # invariant (constructed): members' pronunciations differ in exactly
  # the confusion's dimension and values; loaded data re-checked by rule

Sentence                            # artifact
  clauses: tuple[tuple[Element, ...], ...]
                                    # Element = WordId | (WordId, "ๆ"): the
                                    # sentence as its author parsed it,
                                    # clauses of registered Words in order,
                                    # a word carrying the repetition mark
                                    # where it is repeated
  text: str                         # the rendering: a clause is its words'
                                    # thai forms concatenated (ๆ appended
                                    # to a repeated word), clauses join
                                    # with one space; identity: sha of the
                                    # text; provenance is a fact of the
                                    # row, not identity; invariant
                                    # (constructed, re-checked on load):
                                    # every id is registered and
                                    # text == render(clauses)
  gloss: str                        # L1 gloss, drafted and judged with
                                    # the text (F3: a gloss on any back)
  voice: Literal[learner_voice, other_voice]
  provenance: Provenance
  # which Targets it fills is DERIVED (Syllabus.fills), never stored
```

The words a sentence uses are a fact of the artifact, authored with it,
never recovered from its text. A sentence carries no character outside
its words' forms, ๆ after a repeated word, and the clause spaces: a
number is a number word, no punctuation occurs. Spelling is standard
(ครับ "kráp"; the spoken คับ "káp" is pronunciation, not text).

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
  register: Literal[male_colloquial]      # shapes generation prompts and
                                          # voice constraints; names the
                                          # learner's speaker sex (male)
  emphasis: dict[CategoryName, float]     # order tie-breaking, drafting
  productive_cutoff: int = 2000           # the frequency rank at or above
                                          # which a categorized Word carries
                                          # a productive Target
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
receptive target before productive target per word, so productive
Targets enter in frequency order like their words. Ties: frequency rank
÷ emphasis weight; the loader resolves ranks through the FrequencyMap
port and the aggregate holds the mapping. Pure; recomputed each call;
the studied past is not consulted (StudyRecords fix history, rules catch
invalidated sentences). Consumers (compile, the screen) read positions;
none re-derives placement.

**marking(sentence) -> set** — the union of `speaker` over the
sentence's words: empty (any speaker), {male}, or {female}. Both sexes at
once is a defect the Sentence invariant refuses (`check_sentence`). The
marking constrains the recording's speaker (spec 3 §5) and the productive
fill (clause 2 below); a female-marked sentence still fills receptive
Targets (E7).

**fills(sentence, target) -> bool** — the single definition:
1. target.word is in sentence.clauses (a repeated word counts once),
2. sentence.voice satisfies target.skill (other_voice fills receptive
   only); a productive Target is filled only when the sentence's last
   used word is the target's word (the word the Cloze card is on, spec
   4) and the sentence's marking admits the learner's voice, i.e. is
   empty or the Profile's own sex,
3. at the sentence's entry position (after its last word's target):
   every word it uses has a Target, and at most one filled Target is
   sentence-introduced and unmet, no adopted sentence placed at or
   before this one filling it. That every element is a registered word
   holds by construction (§1).
last_used_word and order() read the clauses. No tokenizer port.
Used by generation as acceptance and by report() as coverage. Clause 3
is a rule over the fill set, applied once per sentence, by acceptance
(the attempt), by adoption (the fold) and by the gate.

**report() -> Report** — runs every check on every note and every
measure on the aggregate. Report { syllabus_state_id, rulebook_id,
findings, metrics, gate }. syllabus_state_id = hash of the aggregate's
content, rulebook_id = hash of the rulebook; a report whose ids differ
from the live aggregate and rulebook steers nothing (staleness is
structural, not advisory). gate = no unwaived error findings.

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

No rule for what a constraint already prevents: an invariant a
constructor or the loader enforces (order() constraints, one category
per word, every id registered, one speaker per rendition) gets unit
tests, not a rule. A constructor-enforced invariant is re-checked by a
rule only where the loader does not construct through the checking path
(pairs, grapheme keywords).

Severity: error findings close the gate (compile refuses, spec 4 §2);
warn and info ship as declared warnings. Per-deck severity overrides live
in rulebook.yaml `severities`; that and `compile --force` are the only
relaxation paths. Judged rules' rubric texts live in rulebook.yaml
`rubrics`.

The rulebook. "compile" = enforced by compile (spec 4), not a rule;
"by construction" = cannot be violated on loaded data.

| principle | rules |
|---|---|
| A2, A5, A6, A7, A8 | compile |
| A3 | card/unique-front (check, error) |
| A4 | compile (a missing artifact drops the card, counted) |
| F1 | pair/exact-confusion (check, error); pair/rendition-required (check, error); rendition/synthetic (check, warn); coverage/confusions (measure: pairs and distinct speakers per confusion against targets); one speaker per rendition by construction |
| F2 | coverage/categories (measure); one category per word and closure by construction |
| F3 | picture/fit (judged), picture/preference (judged), scene/fit (judged, role scene-for-sentence), target/picture-required (check, error; words with a picture-introduced target); front-gloss policy provisional |
| F5 | sentence/fills-novelty (check, error), target/sentence-required (check, error: an adopted sentence fills it), coverage/exercise-depth (measure: adopted sentences per word with a filled Target; value = the share used in two or more) |
| F6 | grapheme/keyword-picture-required (check, error), grapheme/keyword-contains-symbol (check, error) |
| F7, E2 | target/recording-required (check, error), sentence/recording-required (check, error), recording/synthetic (check, warn), sentence/synthetic-productive (check, warn) |
| F8 | by construction: order() enforces sounds-first, sentence-after-words and receptive-before-productive |
| F11 | by construction: current-best ranks judged candidates only |
| E1 | by construction: order() places reading after graphemes |
| E3 | sentence/register-natural (judged); the speaker marking holds at sourcing (spec 3 §5) |
| E4 | word/pronunciation-corroborated (check, error; blocks card emission) |
| E5 | word/classifier-known (check, warn, nouns) |
| E7 | coverage/speakers (measure: per audio corpus — word recordings, renditions, sentence recordings — distinct speakers per sex, age band and region against rulebook targets; unknown never counts) |
| F4, F9, F10, F12, F13, E6 | not rule-shaped (architecture and run behavior); F12's rate cap follows the selection rule; F13's retirement is the run's (spec 3 §5) |
| META-1 | rulebook/traceability (measure) |

The set of principles with enforcement intent is every row above with a
rule; the traceability measure reads this table.

## 5. Explicitly out

- No stored fills edges, weights, order, current-best, or exhausted state.
- No waiver store: a waiver is a learner assessment on a finding identity.

## 6. Testing

Doctrine tests against Syllabus.report()/order()/fills() with fake ports
and builder-made aggregates; entity invariants property-tested (pair
construction, grapheme keyword containment); fills() table-tested over
clauses (shared forms, the repetition mark, novelty, the marking).
