# Spec 1: Domain core

Revision 33, proposed 2026-10-07 against principles r8 and architecture
r5. Revision process: docs/principles.md.

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
- r13 2026-09-12: Corroboration gains "adjudicated" (the judge plus one
  engine, spec 3 r28). User approval 2026-09-12.
- r14 2026-09-12: SoundConfusion.weight, the F1 seed; pair_count
  (5 → 4 pairs, 4 → 3, 3 → 2, else 1). Evidence: sound-stage design
  2026-09-12 ruling 2 (about 55 pairs over 24 confusions,
  weight-proportional). User approval 2026-09-12.
- r15 2026-09-13: `coverage/pictures` (measure, F3): needs with an
  current-best picture over picture needs, by subject kind. Evidence: word
  pictures cover 751 of 766 judged subjects (854 have a need), scenes
  220 of 377; the residue (157 scenes, about 100 words) is what the
  illustrator (spec 3 r34) is for, and coverage, not the per-candidate
  pass rate, is the figure that says whether it works. User approval
  2026-09-13.
- r16 2026-09-17: a Grapheme's recited-name Word carries the category
  `Letter names` and both Targets (`<word id>/receptive`,
  `<word id>/productive`, picture_card); order() places those Targets
  directly after that grapheme, inside the sounds stage;
  target/sentence-required exempts them. Evidence: curated's cross-file
  rule refuses a word with a picture-introduced Target and no category,
  so the design's "no category" could not be built; and the rule would
  otherwise hand the drafter 84 Targets no sentence can fill -- a recited
  name is learned from the chart cell and its own recording (F6, design
  2026-09-12 step 7). User approval 2026-09-17.
- r17 2026-09-17: the `grapheme-keyword-for-grapheme` Assess role is
  retired from the role map; a grapheme's keyword picture is judged under
  `picture-for-word`, the keyword Word's own role, because that picture
  is now that word's own need (spec 3 r42). grapheme/keyword-picture-
  required and grapheme/keyword-contains-symbol are unchanged: they still
  name the grapheme, which is what the finding is about. Evidence: no
  backend ever answered the retired role -- it named a need nothing
  served. User approval 2026-09-17.
- r18 2026-09-18: E4's rule is scoped to minimal-pair membership; a `disputed` pronunciation no longer blocks a word's cards. Evidence: TTS synthesizes from the Thai script and Forvo is a native speaker, so the stored IPA never reaches the learner's ear; it renders as reference on a card back that always carries audio (`target/recording-required`, F7, is an error and no targeted word on the live deck lacks a recording). Pair validity is computed from stored pronunciations (`exact_confusion_violation`), so corroboration is load-bearing there and only there. On the live deck this unblocks 140 targeted words, 103 of which have no pronunciation on record at all. User approval 2026-09-18.
- r19 2026-09-19: pair/exact-confusion retired: MinimalPair.create refuses a pair differing in more than its confusion, so the rule could never fire on loaded data (§4's own principle: no rule for what a constraint already prevents). User approval 2026-09-19.
- r20 2026-09-19: Dimension gains `final`: a coda difference is a final difference and its value is the coda; `consonant` is the onset alone. Evidence: the three final-place confusions were declared on `consonant`, whose value is the onset, and the sounds were written with the unreleased diacritic the engines never store (`p̚`/`t̚`/`k̚`; `engines._CODAS` is bare `p`/`t`/`k`), so no pair could ever be exact for them on either count. User approval 2026-09-19.
- r21 2026-09-19: `coverage/sound-stage` (measure, F1): the share of confusions at their weight-proportional pair count, of graphemes whose keyword has a picture, and of recited-name words with a chart cell, value the least of the three (the stage is a gate, F1: its least-built part is how built it is; design 2026-09-12 section 5 said "graphemes with a keyword"; every Grapheme row carries one by construction, so the part that varies is the keyword's picture). Evidence: coverage/confusions reads one pair and one speaker per confusion and cannot say whether the stage is built (25 confusions want 55 pairs; 13 full at the first search). User approval 2026-09-19.
- r22 2026-09-19: `Syllabus.gaps()` lists every pair without a current-best rendition (`pairs_missing_renditions`, from `pair/rendition-required`) in place of the confusions `coverage/confusions` marked uncovered; the measure keeps its reading and no longer gates sourcing. Evidence: spec 3 r48 (a confusion covered by one pair's rendition hid its other pairs; 7 of 30 on the live deck). User approval 2026-09-19.
- r23 2026-09-20: the centering diphthongs ia, ɯa, ua carry `vowel_length: long`, the standard convention; `Ipa` writes them unmarked. Evidence: 68 of 68 corroborated diphthong syllables on the live deck were `short`, an artifact of thaig2p emitting the diphthong with no length mark and the tltk converter overriding tltk's own `iːa`; 37 of the 57 disputed words were judge verdicts that differed from the engines on diphthong length alone. User approval 2026-09-20.
- r24 2026-09-25: order() deals each sentence directly after its last used word's last Target (the entry position §3 clause 3 already names); a sentence whose last used word is a letter-name word follows the sounds block; sentences sharing a last word are ordered by word count then text_sha, the same key the fill-set placement (§3 clause 3) reads; a sentence with no placed word last. Evidence: order() appended every sentence after every word Target, so no sentence card reached the learner until all ~900 word cards had been introduced; the principle (F8) is a sentence after its words. User ruling 2026-09-25.
- r25 2026-10-01: a productive Target is filled by any learner-voice sentence using its word whose marking admits the learner; last_used_word is the placement key only. Evidence: 620 of 650 drafts refused as filling no open Target used an open productive word that was not their last word in order; with spec 1 r24 dealing a sentence after its last word, every word it uses is known when its cards arrive. User ruling 2026-10-01.
- r26 2026-10-01: a productive Target is filled by at most `production_sentences_per_word` (Profile, 3) sentences, the first in placement order, a pair with a study record on its Cloze card kept and counted. Evidence: under r25 the live deck compiled 1,425 Cloze cards, 96 on the male "I" and 33 on "good", median 2 per word; 761 at three per word. User ruling 2026-10-01.
- r27 2026-10-02: a Word whose speaker marking is not the learner's has no productive Target (E3), derived or listed. Evidence: ดิฉัน ("I", female polite) derived one that no sentence can fill, since a productive fill needs a marking that admits the learner. User ruling 2026-10-02.
- r28 2026-10-02: a drafted sentence whose marking does not admit the learner is other_voice. Evidence: every draft was learner_voice, so the three receptive Targets on female-marked words (ดิฉัน, ค่ะ, คะ) had 32 drafts and none could pass. User ruling 2026-10-02.
- r29 2026-10-02: a spelling group is the Words sharing a written form that carry a Target, in introduction order; spec 4 r12 compiles its form-side cards once. Evidence: 12 groups of same-spelling Words compiled 29 identical Reading, Listening and Spelling fronts. User ruling 2026-10-02.
- r30 2026-10-02: a Target may want several sentences (default one) and is open until that many adopted sentences fill it; one with a sentence and short of its count is target/sentences-wanted (warn), which leaves the gate open. Evidence: a sentence-introduced word has no card of its own, and none of the 424 adopted sentences used ไม่ (not), มี (have), เป็น (be), จะ (will), ได้ (can) or แล้ว (already). User ruling 2026-10-02.
- r31 2026-10-03: a sentence is placed where clause 3 first admits it: its entry, or directly after the adopted sentence whose fill leaves at most one of its sentence-introduced words unmet; order() deals it there. Evidence: all 23 draft rows for the last 8 open Targets paired the Target with a word met only by a later-placed sentence and were refused as introducing two words, though the prompt offers every met word. User ruling 2026-10-03.
- r32 2026-10-04: the sounds stage is the pairs; a grapheme is no order() entry, its card placed by compile just before the first Reading card in order of a word containing it and present while any such Reading card is; a cluster onset touches its head consonant's confusions; a recited-name Word carries no Target and keeps its category, its recording need and its chart-cell picture need (it is not a closure word); staging (§3): a word is heard at its position, said P later and read D after it is heard once readable, readable meaning no segmental confusion its pronunciation touches blocks it (spec 2 r21) or its Reading card has a review, so readability latches, its script on its other cards once its Reading card has a review, and a sentence's text and its text Cloze wait until every word it uses is read, its AudioCloze card P after its Listening card; a studied AudioCloze pair counts for clause 4 as a studied Cloze pair; order() reads no study, the staging does; seed sentences: the first picture words' receptive Targets want two sentences (r30), drafted over spec 3 r65's vocabulary; gaps() adds gapped recordings, pair members' pictures and the recited-name Words' needs. Evidence: principles r8 (F1, F6, F8, E1); on the live deck every picture word touches the tone and length confusions (776 of 776) and 100/187/489 touch 0/1/2+ segmental ones; the sounds block put 42 graphemes and their 84 name-word Targets ahead of every vocabulary word; 2 adopted sentences use only the first 50 picture words. User ruling 2026-10-04.
- r33 2026-10-07: a Recording's bytes are conditioned before they are hashed: a clip in which voice activity detection finds no speech (Silero VAD: no window's speech probability reaches 0.25) is refused at ingest; leading and trailing audio more than 40 dB below the clip's own peak is trimmed, 50 ms kept at each end, inner silence untouched; loudness is normalized to −20 LUFS by linear gain, true peak held below −1.5 dBTP. Evidence: two Chirp3-HD pair clips, จะ "will" and ไป "go", were 0.26 s peaking at −39.5/−38.6 dBFS and passed the duration-only check with floor 0.2 s; 15 Forvo clips peak −25 to −33 dB; at peak−40 the shortest trimmed real clip is 0.232 s; under VAD the two silent clips scored 0.046 and 0.059, a flat Forvo clip 0.103, every other of the deck's 2,696 clips 0.31 or more (2,690 at 0.73 or more); a peak floor would have had 3 dB of margin (silent −39.5/−38.6 dBFS, quietest real −33.3) and a floor-relative spread did not separate them; under the −1.5 dBTP ceiling 93% of the trimmed clips reach −20 LUFS by linear gain and 32% reach −16. User ruling 2026-10-07.

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
                                    # sentence names the sense. A form's
                                    # spelling group: its Words that
                                    # carry a Target, in introduction
                                    # order (§3)
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
  syllables: tuple[Syllable, ...]   # segments, vowel length, Chao tone;
                                    # the centering diphthongs ia ɯa ua are
                                    # long (r23), written unmarked by Ipa
  corroboration: Corroboration      # engines_agree | curated_exception |
                                    # adjudicated (r13: the judge plus one
                                    # engine, spec 3 r28) | disputed; see
                                    # rules R-PRON; a disputed word exists
                                    # but blocks card emission

SoundConfusion                      # language model
  id: ConfusionId                   # e.g. "tone:mid-low"
  dimension: Literal[tone, length, aspiration, vowel_quality, consonant, final]
  sounds: tuple[str, str]           # the two opposed values
  weight: int = 1                   # F1 seed; pairs per confusion 5→4,
                                    # 4→3, 3→2, else 1

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
                                    # ("gɔɔ gài" for ก); consonants today.
                                    # That Word carries the category
                                    # `Letter names` (r16) and no Target
                                    # (r32): the learner hears it on the
                                    # grapheme card (spec 4 §1); it keeps
                                    # its recording need and its picture,
                                    # the alphabet-chart cell (spec 3 §3's
                                    # glyph source)

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
                                    # none unless listed. A Word whose
                                    # speaker marking is not the learner's
                                    # carries none, derived or listed (E3).
                                    # The loader refuses a row that
                                    # duplicates a derived Target or lists
                                    # one on such a Word.
  sentences: int = 1                # the adopted sentences it wants: it
                                    # is open until that many fill it
                                    # (§3); a positive integer, one on a
                                    # productive Target (the loader
                                    # refuses more)

Category                            # curated learning list: a theme of
  name: str                         # the FF 625 list. identity
  members: frozenset[WordId]        # invariant: a word is in at most one
                                    # category (the loader builds the
                                    # collections from each word row's one
                                    # category field); closure words (pair
                                    # members, keywords) and words whose
                                    # only targets are sentence-introduced
                                    # are in none; a recited-name Word is
                                    # not a closure word and is in
                                    # `Letter names` (r32)

MinimalPair
  id: PairId                        # identity
  confusion: ConfusionId
  members: tuple[WordId, ...]       # 2..3
  # invariant (constructed): members' pronunciations differ in exactly
  # the confusion's dimension, at one syllable, and both values there are
  # the confusion's; `MinimalPair.create` is the one check and no rule
  # re-checks loaded data for it (r19)

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
                                    # a drafted sentence's voice follows
                                    # its marking (§3): other_voice when
                                    # the marking does not admit the
                                    # learner, else learner_voice
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

A Recording's bytes are conditioned before they are hashed. A clip in
which voice activity detection finds no speech (Silero VAD: no window's
speech probability reaches 0.25) is refused at ingest. Otherwise leading
and trailing audio more than 40 dB below the clip's own peak is trimmed,
50 ms kept at each end; inner silence is untouched (a gapped clip's
breaks survive); loudness is normalized to −20 LUFS (integrated, linear
gain, true peak held below −1.5 dBTP).

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
  production_sentences_per_word: int = 3  # the most sentences filling one
                                          # productive Target (§3, r26)
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
| sentence, id }: the one introduction order of the pairs, Targets and
sentences. Constraints, each also stated as a rule: the sounds stage
(the pairs) before words (r32: a grapheme is not an entry, Staging
below deals its card); a sentence at its placement (clause 3 below; r24, r31); sentences at one place by word count, then text_sha, the same placement clause 3 reads; a sentence with no placed word last;
receptive target before productive target per word, so productive
Targets enter in frequency order like their words. Ties: frequency rank
÷ emphasis weight; the loader resolves ranks through the FrequencyMap
port and the aggregate holds the mapping. Pure; recomputed each call;
the studied past is not consulted (StudyRecords fix history, rules catch
invalidated sentences). Consumers (compile, the screen) read positions;
none re-derives placement.

**Staging (r32)** — when an item's cards are dealt relative to its
order() position, and when its script shows, P and D positions being
spec 4 §2's; compile realizes it as dues, card presence and field gates
(spec 4 §1, §2) and is the one reader of study evidence here. A word is
**readable** while no segmental confusion its pronunciation touches
blocks (spec 2 §2: unstable, with a pair in the deck), or once its
form's Reading card has a review, so a compile never withdraws a
reviewed Reading card; an unreviewed one can be withdrawn when a pair
is added to a touched confusion that is not stable. A confusion is
segmental unless its dimension is tone or length, and a pronunciation
touches it where a syllable's value on its dimension (§1) is one of its
sounds; a cluster onset (kʰr, pl) touches the confusions of its head
consonant (kʰ, p). A word with no pronunciation on record touches none
and is readable. A word is
**read** once its form's Reading card has a review in the study record;
a word whose form has no Reading card (a sentence-introduced word) is
read once readable.
- A word is heard at its position (Listening), said P later
  (Production) and read D after it is heard (Reading, Spelling, present
  once it is readable); its Thai and IPA appear on its other cards once
  it is read.
- A sentence is heard at its placement (Listening), its text on the
  back only once every word it uses is read; it is produced by ear P
  after that (one AudioCloze card per filled productive Target) and
  from its text (Cloze) only once every word it uses is read.
- A grapheme sits just before the first Reading card, in order, of a
  word whose form contains its symbol, and is present while any Reading
  card of such a word is. Only
  consonants have Grapheme rows yet: vowel signs and tone marks have no
  card.

**spelling_group(word) -> tuple[Word]** — the Words sharing the Word's
written form (`thai`) that carry a Target, ordered by the order()
position of each one's first Target, ties by word id. A Word alone in
its form is a group of one; a Word with no Target is in no group.

**marking(sentence) -> set** — the union of `speaker` over the
sentence's words: empty (any speaker), {male}, or {female}. Both sexes at
once is a defect the Sentence invariant refuses (`check_sentence`). The
marking constrains the recording's speaker (spec 3 §5) and the productive
fill (clause 2 below); a female-marked sentence still fills receptive
Targets (E7).

**fills(sentence, target) -> bool** — the single definition:
1. target.word is in sentence.clauses (a repeated word counts once),
2. sentence.voice satisfies target.skill (other_voice fills receptive
   only); a productive Target is filled only when the sentence's marking
   admits the learner's voice, i.e. is empty or the Profile's own sex,
3. at the sentence's placement: every word it uses has a Target, and
   at most one filled Target is sentence-introduced and unmet, no
   adopted sentence placed at or before this one filling it. That every
   element is a registered word holds by construction (§1). A
   sentence's placement (r31) is its entry (after its last used word's
   last Target) unless two or more of its sentence-introduced Targets
   are unmet there; then it is directly after the adopted sentence
   whose fill leaves at most one of them unmet, each placement reading
   only sentences placed before it and fills before the cap (clause 4).
   With no such sentence it stays at its entry and fills nothing.
   Sentences placed directly after one sentence go by word count, then
   text_sha. A draft is placed where it would be once adopted.
4. a productive Target is filled by at most
   `production_sentences_per_word` sentences (r26): among the adopted
   sentences clauses 1-3 admit for it, the first in placement order,
   except that a (sentence, Target) pair with a study record on its Cloze
   or AudioCloze card (r32), clauses 1-3 admitting it, keeps filling it and counts toward the
   cap whatever its position; a studied pair clause 3 refuses holds no
   place. A sentence beyond them does not fill that Target and keeps its
   other fills. A draft (not adopted) fills it only while the studied
   pairs filling it plus the unstudied adopted sentences placed before
   the draft that fill it are fewer than the cap, as the fold would
   place it once adopted. The study records reach the aggregate at
   load; built without them, none is studied.
   sentence/fills-novelty does not flag a Target this clause leaves out.
last_used_word and order() read the clauses. No tokenizer port.
Used by generation as acceptance and by report() as coverage. Clause 3
is a rule over the fill set, applied once per sentence, by acceptance
(the attempt), by adoption (the fold) and by the gate.

**Open and met.** A Target is open while fewer adopted sentences fill it
(their fill sets contain it, clauses 1-4 above) than its `sentences`.
One no adopted sentence fills is target/sentence-required (error: "no
adopted sentence fills it"); one some fill, short of its count, is
target/sentences-wanted (warn: "N of M adopted sentences fill it"), so
the gate does not wait on a wanted count. gaps() lists both among the
unfilled targets, in target order. Seed sentences (r32): the receptive
Targets of the first picture-introduced words want two sentences each,
so simple sentences arrive with the first vocabulary; a seed ask offers
the span's picture words and the glue placed within it, other function
words as introducibles, one per sentence (spec 3 §5). A seed Target's count is met only by the
sentences placed within the seed span, at or before the last seed
Target's word's place; a sentence placed later still fills it (the
gate, met, clause 3) but counts toward no seed Target's `sentences`,
here or in N above. A sentence-introduced
Target is met once an adopted sentence fills it (clause 3), open or not:
from then on its word is vocabulary for the sentences placed after.

**report() -> Report** — runs every check on every note and every
measure on the aggregate. Report { syllabus_state_id, rulebook_id,
findings, metrics, gate }. syllabus_state_id = hash of the aggregate's
content, rulebook_id = hash of the rulebook; a report whose ids differ
from the live aggregate and rulebook steers nothing (staleness is
structural, not advisory). gate = no unwaived error findings.

**gaps() -> Gaps** — derived from the report's completeness findings
and measures, never recomputed beside them: every pair without a
current-best rendition (`pair/rendition-required`), unfilled (open)
targets, words lacking pictures/recordings (recited-name Words among
them, r32), pair members lacking pictures (a pair's card waits on them,
spec 3 r65), sentences lacking recordings,
sentences carrying a Cloze card (a productive fill) that lack a scene
picture (spec 3 r61), filled Cloze slots lacking a gapped recording
(spec 3 r65), graphemes lacking keyword data. Input to the batch run (spec 3).

Compile is an application service (spec 4; architecture §7) over
report(), order(), the current-best artifacts and, for the staging, the
study record; the aggregate has no storage dependency.

## 4. Rules

```
Rule
  id: str                           # e.g. "pair/rendition-required"
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
| F1 | pair/rendition-required (check, error); rendition/synthetic (check, warn); coverage/confusions (measure: pairs and distinct speakers per confusion against targets); coverage/sound-stage (measure: confusions at their weight-proportional pair count, graphemes whose keyword has a picture, recited-name words with a chart cell; value the least of the three); one speaker per rendition by construction; exact confusion by construction (MinimalPair.create; r19) |
| F2 | coverage/categories (measure); one category per word and closure by construction |
| F3 | picture/fit (judged), picture/preference (judged), scene/fit (judged, role scene-for-sentence), target/picture-required (check, error; words with a picture-introduced target); coverage/pictures (measure: needs with a current-best picture over picture needs, by subject kind); front-gloss policy provisional |
| F5 | sentence/fills-novelty (check, error), target/sentence-required (check, error: no adopted sentence fills it), target/sentences-wanted (check, warn: some adopted sentence fills it, fewer than its `sentences`, r30), coverage/exercise-depth (measure: adopted sentences per word with a filled Target; value = the share used in two or more) |
| F6 | grapheme/keyword-picture-required (check, error; the keyword Word's own picture need, spec 3 r42), grapheme/keyword-contains-symbol (check, error) |
| F7, E2 | target/recording-required (check, error), sentence/recording-required (check, error), recording/synthetic (check, warn), sentence/synthetic-productive (check, warn) |
| F8 | by construction: order() enforces sounds-first, sentence-after-words and receptive-before-productive; compile stages hearing before reading (§3 Staging, spec 4 §2) |
| F11 | by construction: current-best ranks judged candidates only |
| E1 | compile: a grapheme card before the first Reading card that needs it; Thai and IPA gated on the item being read (§3 Staging, spec 4 §1, §2) |
| E3 | sentence/register-natural (judged); the speaker marking holds at sourcing (spec 3 §5) |
| E4 | pair/pronunciation-corroborated (check, error; blocks pair membership) |
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
