# Spec 4: The Anki boundary

Revision 15, proposed 2026-10-07 against principles r8 and architecture
r5. Revision process: docs/principles.md.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: fields append; the first field is the note identity.
- r3 2026-09-04: field lists as shipped; pair member notes separated; Gloss
  from the Sentence; flag roles by (family, kind); harvest keyed by anchor.
- r4 2026-09-06: pair notes from the rendition, Choices in member order;
  atomic tags; cumulative due blocks; the typed FlagKey.
- r5 2026-09-08: one note per adopted Sentence, clozed on its last used
  word, a target tag per filled target, guid = text_sha; word notes for
  picture-introduced words only.
- r6 2026-09-09: ThaiCloze is the rendering with the last used word's
  elements blanked; no tokenizer.
- r7 2026-09-11: the card taxonomy (from principles r3) stated in §1; the
  ReviewNote retraction aside reduced to the rule; no behavior changed.
- r8 2026-09-11: the import reads Anki's current collection schema as
  well as the legacy one; a harvested note records the compile id it was
  written against; a Production card needs the picture (§1). Evidence:
  the live collection (schema 18) failed the import before a note was
  read; 23 Production cards compiled with an empty front.
- r9 2026-10-01: the sentence note carries the Listening card; each Target a sentence fills productively has its own Cloze note on that sentence (identity sentence + Target, blanking that Target's word, due with the sentence). Evidence: spec 1 r25 lets one sentence fill several productive Targets. The sentence notetype's fields and Cloze template changed, a break taken at the cutover's delete-and-reimport; the append-only rule holds from r9 on. User ruling 2026-10-01.
- r10 2026-10-01: a sentence Cloze card needs its sentence's scene picture; without one it is dropped from the build (counted, reason "no current-best picture") until the picture exists. Evidence: the Cloze front is the blanked sentence plus the picture, so a pictureless card does not say which word is wanted and two sentences differing only in the blanked word compile to one front. A Cloze card already in the learner's Anki collection stays there when its picture is later rejected, as a Production card does. User ruling 2026-10-01.
- r11 2026-10-02: a sentence's Listening card and its Cloze cards are siblings of one note, a Cloze card in the slot of its word's position in the sentence; a Cloze card needs the sentence's recording as well as its scene picture. Evidence: under r9 each Cloze card was its own note, so Anki's sibling burying no longer separated a sentence's cards: 336 Listening cards were followed at once by a Cloze card on the same sentence and 225 sentences dealt two to seven Cloze cards back to back, each front showing the words the others blank; 4 Cloze cards compiled with no audio on the back. The notetype changes again before any import of r9, and the cutover deletes the stale notetypes with the deck; a notetype change updates a collection in place only under Anki's "Merge note types". Sibling burying is the learner's deck preset, which an import does not set: the import warns when it is off. User ruling 2026-10-02.
- r12 2026-10-02: a spelling group's Listening, Reading and Spelling cards are compiled once, on its first picture-introduced Word's note (gated by the appended field FormSide); the Listening and Reading backs list the group's other members, each with its picture and meaning, after the note's own (the appended field OtherSenses); the Spelling back is unchanged; each member keeps its Production card. A Word alone in its form compiles and renders as under r11. Evidence: 29 of the 33 card/unique-front findings on the live deck were same-spelling Words (12 groups). The notetype change needs "Merge note types" on an import into a collection that holds r11's word notetype, or that notetype is deleted with the deck at a cutover (§5). User ruling 2026-10-02.
- r13 2026-10-04: cards are staged by kind (§2): a word's Listening card at its position, Production P = 5 later, Reading and Spelling D = 50 after Listening and present only once the word is readable (spec 1 r32, the field Readable); ScriptShown, set once the word is read (its Reading card reviewed), gates Thai and IPA on a word's other backs, and a sentence's, set once every word it uses is read, gates the text on its Listening back and the text Cloze fronts; an AudioCloze card per filled slot plays spec 3 r65's gapped recording over the scene picture, P after the sentence's Listening card; a grapheme, no longer an order() entry, sits just before the first Reading card in order of a word whose form contains it, present while any such Reading card is, the recited name playing on its front; a pair's Recognition card is present only once every member has a picture, its front offering them, its script on the back only; the notetype changes and the new dues are taken at a cutover (§5). Evidence: all four word cards were due in one block, so a word's Reading card followed its Listening card; the deck's first cards, the pair fronts, showed Thai script; the sounds block put 42 graphemes and 84 name-word Targets ahead of every vocabulary word; only 18 of the 42 keyword Words carry a Target, so a letter cannot ride its keyword's Reading card; Anki's import skips an existing (guid, ord), so it never re-dues a card. User ruling 2026-10-04.
- r14 2026-10-07: the pair Recognition card takes a tap: a tapped picture is the learner's choice and shows the back, which says whether it was the stimulus; the pictures stack in portrait and sit side by side otherwise, both on one screen; the back plays the stimulus first, then the other members, and no longer repeats the front's pictures. Evidence: on AnkiDroid the two pictures overflowed the screen, any tap revealed the back, and the back autoplayed only the other member beside "you heard" (audio inside FrontSide never autoplays). User ruling 2026-10-07.
- r15 2026-10-07: recordings are conditioned at ingest (spec 1 r33) as images are normalized (§3); the stored, sha'd, checked artifact is the conditioned file. Evidence: two Chirp3-HD pair clips, จะ "will" and ไป "go", were 0.26 s peaking at −39.5/−38.6 dBFS and passed the duration-only check with floor 0.2 s; 15 Forvo clips peak −25 to −33 dB. User ruling 2026-10-07.

Scope: compile — the translation of Syllabus state into Anki's domain —
and the return path: revlog, flags, and ReviewNote harvests. Anki's
domain is adopted unmodified (architecture §6); nothing here re-litigates
it.

## 1. Models and cards

The card taxonomy, by skill: discriminate (minimal_pair Recognition),
hear → meaning (word Listening), picture → say (word Production), read →
meaning (word Reading), symbol → sound (grapheme Reading), produce in
context by ear (sentence AudioCloze), produce in context from the text
(sentence Cloze), understand in context (sentence Listening).

Model ids: sha-derived from model name, stable. A note updates in place
by guid (A2), but a notetype change (fields or templates) updates the
collection's notetype in place only when the package is imported with
Anki's "Merge note types"; without it Anki creates a copy named with a
`+` (`sentence+`) and puts the imported notes under it. The sentence
notetype changed at r9 and again at r11, before any import of r9, a
break taken at the cutover (§5). The word notetype changed at r12 (the
fields OtherSenses and FormSide appended, the Listening, Reading and
Spelling templates changed): an import into a collection holding r11's
`word` notetype under the same model id needs "Merge note types", or
that notetype is deleted with the deck at a cutover (§5). The word and
sentence notetypes change again at r13, taken at a cutover (§5). A note's first
field is its identity, unique within its model (A3). Every model carries the
card CSS (legible Thai, bounded images, answer distinct, night mode — as
shipped in the current compiler) and two service fields rendered by no
template: ReviewNote (the mid-review comment channel) and CompileId.

**word** (from a Word with a picture-introduced Target):
fields Thai, Meaning, Picture, Audio, Ipa, Classifier, FrontGloss,
TestSpelling, ProductiveTarget, ReviewNote, CompileId, OtherSenses,
FormSide, Readable, ScriptShown.
The Listening, Reading and Spelling cards are the form side of a
spelling (spec 1 §3's spelling group: the Words sharing a written form
that carry a Target, in introduction order; a Word alone in its form is
a group of one). They are compiled once per group, on the note of its
first picture-introduced Word (the carrier), and play the carrier's
recording; FormSide (non-empty on the carrier's note only) gates them.
OtherSenses is the group's other members, in introduction order, each
as its current-best picture where it has one and its meaning; a
sentence-introduced member, which has no note, appears there. It is
empty for a Word alone in its form, whose cards render as under r11.
Every other picture-introduced member's note holds its
Production card only, and a member with no productive Target has no
note (no card, none counted as dropped). Identity, tags and due are
each note's own Word's (§2): a review or flag on a form-side card maps
to the carrier. ProductiveTarget gates the Production card (non-empty
iff the word has a productive Target). Ipa renders the Pronunciation
value with tone and length. Readable (r13) is non-empty iff the note's
Word is readable and gates the Reading and Spelling fronts; ScriptShown
(r13) is non-empty iff the Word is read, its form's Reading card having
a review in the study record (spec 1 §3 Staging), and gates the Thai
and IPA on the Listening and Production backs; a non-carrier member's
script therefore follows the carrier's Reading review.
- Listening (receptive; carrier only): front audio; back picture,
  meaning, OtherSenses, Thai and IPA under ScriptShown.
- Production (productive Target and a current-best picture; no picture,
  no card): front picture
  {{#FrontGloss}}gloss chip{{/FrontGloss}}; back native audio, Thai and
  IPA under ScriptShown.
- Reading (carrier only; under Readable): front Thai script; back
  picture, audio, meaning, OtherSenses.
- Spelling (carrier only; under Readable; TestSpelling-gated,
  TestSpelling non-empty iff any member of the group has a productive
  Target): front audio; back Thai.
A form-side card already in the learner's collection on a note that no
longer carries the form side is left with a blank front, its schedule
intact; Anki's Empty Cards deletes it.
FrontGloss is the F3 variant point: empty by default; the compile fills
it per gloss policy (pending study input). Meaning renders on the
Listening and Reading backs, the other members' meanings after it.

**minimal_pair** (one note per rendition member):
fields MemberKey, Choices, Audio, OtherAudio, Stimulus, Speaker, plus
per-member Thai/IPA, ReviewNote, CompileId. Both notes of a pair play the
pair's current-best rendition (one speaker across members); a pair with
no rendition compiles no notes and is counted as dropped. First field =
MemberKey "PAIRID:SPEAKER:INDEX" (unique; the guid source; r14: the
back's script reads the stimulus's index from it, nothing else reads
it back). Back shows both members, marks the stimulus ("played:"), each
member's audio individually playable by its own button (F6b).
- Recognition (r13: present only when every member has a current-best
  picture; otherwise no card, counted with the reason "no current-best
  picture"): front stimulus audio + the Choices field, each member's
  picture, rendered in member order on every note so position never
  marks the stimulus. No Thai or IPA on the front; the back gives both
  members' Thai and IPA. (r14) Front pictures stack in portrait and sit
  side by side otherwise, both on one screen. Tapping a picture records
  it as the learner's choice and shows the back; showing the back
  without a tap records no choice. Back: whether the choice was the
  stimulus (only when one was made); the stimulus's picture, Thai, IPA
  and audio; the other members' Thai, IPA and audio; the audio autoplays
  in that order. The front's pictures are not repeated. The learner
  still grades; the choice is not harvested.

**grapheme**:
fields Symbol, Sound, NameThai, KeywordThai, KeywordGloss,
KeywordPicture, Audio, ReviewNote, CompileId.
- Reading: front symbol and the name word's recording as Audio (r13:
  the recited name is heard here, F6); back the recited letter name
  (NameThai = the name word's own text, e.g. กอ ไก่ "gɔɔ gài", one Word
  whose recording says the whole name), keyword with its own picture
  and gloss. Dealt on demand (§2). One card; no reverse family (F6).
  Name and keyword are Words (Grapheme.name_word, Grapheme.keyword);
  their recordings/pictures come through the normal sourcing path. No
  substitute audio: a grapheme whose name word has no current-best
  recording drops the card, counted.

**sentence** (one note per adopted Sentence, at its order position):
fields Thai, TargetWord, Audio, Gloss, ScenePicture, then per Cloze slot
k = 1..CLOZE_SLOTS ClozeK, ClozeWordK, ClozeTargetK, then ReviewNote,
CompileId, then ScriptShown and per slot ClozeAudioK (r13). Gloss =
Sentence.gloss; TargetWord is the sentence's target
words (spec 3 r54), joined. ScriptShown is non-empty iff every word the
sentence uses is read (spec 1 §3 Staging). ClozeAudioK is the slot's
gapped recording (spec 3 §5), filled only when a Target fills the slot.
Tags: one target::ID per target the sentence fills, one sentence::SHA.
- Listening (receptive), card ord 0: front audio; back gloss, full text
  and target words under ScriptShown.
- Cloze K (productive), card ord K, one per slot: a Target the sentence
  fills productively takes the slot of its word's position among the
  sentence's distinct words (clause order, first occurrence), so a
  (sentence, Target) pair keeps its slot, and its scheduling, however the
  fill set changes. The slot belongs to the sentence's parse: a re-parse
  that changes the distinct-word sequence moves slots, and a card on a
  slot now holding another word is attributed to that word's pair, its
  history included. ClozeTargetK names the Target that fills the slot,
  and in an unfilled slot the lowest-id productive Target of the slot's
  word (empty when the word has none), so a card in the learner's
  collection always maps back to its pair (§4); ClozeK (the cloze on that
  word) and ClozeWordK (the word) are filled only when a Target fills the
  slot. A slot with an empty ClozeK yields no card, not counted as
  dropped. The card needs the sentence's current-best scene picture and
  recording; missing either, no card, counted with a reason naming what
  is missing. Under ScriptShown (r13). Front cloze + scene picture; back
  the word, NATIVE audio (F7), gloss. The sentence stays adopted and its
  Listening card is unaffected.
- AudioCloze K (productive, r13), card ord CLOZE_SLOTS + K, one per
  slot a Target fills: front ClozeAudioK + scene picture; back the
  sentence's recording (NATIVE where F7 asks), gloss, and the word under
  ScriptShown. It needs the scene picture, the recording and the gapped
  recording; missing any, no card, counted with a reason naming what is
  missing.
CLOZE_SLOTS is fixed at 10, covering the live deck's longest adopted
sentence (10 distinct words, of 13 elements); a productive fill on a word
beyond the last slot is dropped, counted ("beyond the last Cloze slot").
Changing the slot count is a notetype change like any other.
A Cloze card already in the learner's collection whose front empties
(the per-word cap pushing out an unstudied pair; a rejected scene
picture, which blanks every Cloze card of that sentence until it is
re-sourced) stays scheduled, and Anki shows "The front of this card is
blank". It refills, its schedule intact, at the next compile and import
that fill the slot. Running Anki's Empty Cards deletes it and its
schedule. A sentence with no current-best recording yields no card, so
its note is absent from the package and its cards in the learner's
collection are left as they were, old fronts and old recording
included, until a recording is current-best again.
ClozeK is the sentence's rendering with every element whose word is the
slot's word blanked (a repeated word keeps its ๆ outside the blank),
never str.replace over the text (the ยา/โรงพยาบาล corruption class:
blanking "medicine" inside "hospital" cannot arise from elements).

## 2. Identity, tags, order

- guid: word = word id; grapheme = symbol; pair = MemberKey;
  sentence = text_sha, a Cloze card being (that note, its slot's ord). A
  replaced sentence resets its scheduling; everything else updates in
  place.
- Tags are atomic, one part per tag, never composed or split: family::,
  kind:: (card kind), word::ID (word notes), pair::ID, confusion::ID,
  member::INDEX and speaker::ID (pair notes), grapheme::SYMBOL,
  target::ID per filled target and sentence::SHA (sentence notes),
  compile::ID,
  src tags for audio/image provenance. The import reads each tag's
  value by prefix and writes the parts as study columns (spec 2); a word
  card's Target is derived from its kind (Listening receptive, Production
  productive).
- due: per card, from the order index (r13). Each order() entry owns a
  block of STRIDE dues (a pair: one per member), cumulative. A word's
  Listening card sits at its first Target's block, its Production card
  P = 5 blocks later, its Reading and Spelling cards D = 50 blocks after
  the Listening card; a sentence's Listening card at its block, its text
  Cloze cards in that block in slot order, its AudioCloze cards P blocks
  later in slot order; a grapheme, which is no order() entry (spec 1
  §3), placed by compile just before the first Reading card in order,
  present or not, of a word whose form contains its symbol, several in
  symbol order, and present while any Reading card of such a word is. No two cards share a due. A card gated out (Staging,
  spec 1 §3) is absent from the build and enters at its own due on a
  later compile. Anki buries siblings only when the deck's preset has its three
  bury settings on (new, review, interday learning siblings); they must
  be on in the learner's preset, set once at cutover; an import leaves
  the learner's deck preset unchanged, whether or not it is run with
  Anki's option to import deck presets. The
  import warns when any is off (§4). The two member notes of a pair are
  not siblings: they sit a stride apart inside the pair's block (A5).
- compile refuses when report().gate fails, unless forced with declared
  warnings; the Compile value records compile id = syllabus state id +
  timestamp, stamped into every note's CompileId field.

## 3. Media

Referenced by content sha basename from media/objects/ (sha256 of the
file bytes: identical content dedupes, collisions negligible); the
package manifest maps them; a missing current-best artifact drops the
dependent card (never an empty front), counted in the compile report.
Images are normalized at ingest — bounded long edge, aspect preserved,
metadata stripped, re-encoded — and the stored, sha'd, judged artifact
is the normalized file: the judge sees the pixels the card shows.
Recordings are conditioned at ingest (spec 1 §1: no-speech refusal,
trimmed ends, normalized loudness) and the stored, sha'd, checked
artifact is the conditioned file: the check measures the clip the card
plays. CSS retains only final fit-to-viewport.

## 4. Return path

- **Revlog import**: read collection.anki2 read-only — the live
  collection (Anki's current schema keeps notetypes in tables) or an
  .apkg's own (the legacy schema keeps them in `col.models`), both read;
  map card -> (card_key, compile id) via tags/CompileId; append study
  rows. Idempotent by (card_key, ts). A sentence Cloze card's Target is
  named by its slot's ClozeTarget field (the card ord is the slot), and
  its anchor is SHA:TARGET_ID, composed from the sentence:: tag and that
  Target as a pair member's MemberKey is from its three tags; its study
  rows and flag rows carry that anchor, and its entity subject stays the
  text_sha. An AudioCloze card maps back as its slot's Cloze card does,
  under its own card kind. A review on a card whose slot is currently unfilled still
  maps to its pair, which becomes studied and fills again (spec 1 §3
  clause 4). A Cloze card whose slot names no Target (its word has no
  productive Target) is skipped with a reason. A Listening card's anchor
  is the text_sha.
- **Flag import**: flags become learner assessments (cache rows) on the
  entity's subject (word id, pair id, grapheme symbol, sentence text_sha)
  with role from (family, card kind): word Listening and sentence
  Listening flag the recording (tone role: a re-verification request,
  never an override); word Production flags the current picture (a
  learner picture-for-word rating on its sha); sentence Cloze flags the
  sentence's current scene picture (a learner scene-for-sentence rating
  on its sha); a card with no artifact role (Reading, Spelling,
  Recognition, grapheme Reading, AudioCloze, a Production card whose word has no
  picture, which only a deck compiled before r8 can hold, or a Cloze
  card with no current-best scene picture to rate, its picture rejected
  since the compile or the deck compiled before r10) is a card-level
  flag, which
  makes the subject directed in the queue (spec 3 §6) and appears on the
  subject screen (spec 5). The idempotence key is the typed
  FlagKey(family, anchor, card_kind, flags); no marker rows.
- **ReviewNote harvest**: read fields directly from the collection
  (read-only); each non-empty note appends a learner row on the
  note's anchor (from its tags) under key learner-note:ANCHOR:sha(TEXT),
  recording the CompileId the note was written against — re-harvesting the same text is an
  exact-key hit (no duplicate), edited text is a new key (reprocessed),
  a cleared field appends nothing and retracts nothing: a newer note on
  the same subject supersedes on read, and a retraction is a superseding
  row. Harvest never writes to Anki.
- Import is one command; it reports rows imported per kind and rows
  skipped with reasons, and warns, without failing, for each deck holding
  compiled cards whose preset has a bury setting off, naming the deck,
  the preset and the settings (§2).

## 5. Explicitly out

- No deck deletion/orphan cleanup in v1 (delete-and-reimport is the
  current practice). AnkiConnect is the pacer's import path (spec 3 §8):
  Anki's note update by guid, which skips every (guid, ord) already in
  the collection, so it never changes an existing card's due and a new
  card takes the package's. The
  r11 cutover deletes the stale notetypes as well as the deck: r11's
  import expects absent a `sentence` notetype with the pre-r11 fields
  under the same model id, a `minimal_pair` notetype with fields other
  than r11's under its model id, and their `+` copies (`sentence+`,
  `minimal_pair+`). The notetypes earlier imports left under other model
  ids (`sentence++`, which holds the sentence notes the learner studies,
  `picture_word`, `picture_word+`, `spelling_sound`, `spelling_sound+`)
  do not collide with r11's; the learner deletes them at the cutover
  along with the deck. r12 changes the `word` notetype under its model
  id: a cutover imported without "Merge note types" also expects absent
  the `word` notetype with r11's fields and its `+` copy (`word+`).
  r13's notetype changes (word: Readable, ScriptShown and its templates; sentence:
  ScriptShown, ClozeAudioK, the Listening and Cloze templates and an
  AudioCloze template per slot) and its dues are taken at a cutover that
  deletes the deck and those two notetypes and imports the package (the
  deck held 12 reviews, user ruling 2026-10-04); a later staging change
  against a studied deck needs another path.
- No native Anki cloze type: the sentence notetype is a plain one with a
  template per Cloze slot; corruption is impossible by construction
  (element replacement).
- Scheduling migration: out of scope, not prohibited. Guid stability
  preserves scheduling across reimports, which covers current needs;
  cross-collection migration (AnkiConnect/colpkg) is possible if ever
  wanted.
