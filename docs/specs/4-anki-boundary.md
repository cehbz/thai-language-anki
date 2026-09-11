# Spec 4: The Anki boundary

Revision 7, proposed 2026-09-11 against principles r3 and architecture
r3. Revision process: docs/principles.md.

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

Scope: compile — the translation of Syllabus state into Anki's domain —
and the return path: revlog, flags, and ReviewNote harvests. Anki's
domain is adopted unmodified (architecture §6); nothing here re-litigates
it.

## 1. Models and cards

The card taxonomy, by skill: discriminate (minimal_pair Recognition),
hear → meaning (word Listening), picture → say (word Production), read →
meaning (word Reading), symbol → sound (grapheme Reading), produce in
context (sentence Cloze), understand in context (sentence Listening).

Model ids: sha-derived from model name, stable; fields only ever
append, so an existing collection updates in place (A2). A note's first
field is its identity, unique within its model (A3). Every model carries the
card CSS (legible Thai, bounded images, answer distinct, night mode — as
shipped in the current compiler) and two service fields rendered by no
template: ReviewNote (the mid-review comment channel) and CompileId.

**word** (from a Word with a picture-introduced Target):
fields Thai, Meaning, Picture, Audio, Ipa, Classifier, FrontGloss,
TestSpelling, ProductiveTarget, ReviewNote, CompileId. ProductiveTarget
gates the Production card (non-empty iff the word has a productive
Target). Ipa renders the Pronunciation value with tone and length.
- Listening (receptive): front audio; back picture, Thai, IPA, meaning.
- Production (productive Target only): front picture
  {{#FrontGloss}}gloss chip{{/FrontGloss}}; back Thai, native audio, IPA.
- Reading (staged: due after the graphemes its spelling uses): front
  Thai script; back picture, audio, meaning.
- Spelling (TestSpelling-gated): front audio; back Thai.
FrontGloss is the F3 variant point: empty by default; the compile fills
it per gloss policy (pending study input). Meaning always renders on
backs.

**minimal_pair** (one note per rendition member):
fields MemberKey, Choices, Audio, OtherAudio, Stimulus, Speaker, plus
per-member Thai/IPA, ReviewNote, CompileId. Both notes of a pair play the
pair's current-best rendition (one speaker across members); a pair with
no rendition compiles no notes and is counted as dropped. First field =
MemberKey "PAIRID:SPEAKER:INDEX" (unique; the guid source; nothing reads
it back). Back shows both members, marks the stimulus ("you heard: ..."),
each member's audio individually playable (F6b).
- Recognition: front stimulus audio + the Choices field, rendered in
  member order on every note so position never marks the stimulus.

**grapheme**:
fields Symbol, Sound, NameThai, KeywordThai, KeywordGloss,
KeywordPicture, Audio, ReviewNote, CompileId.
- Reading: front symbol; back the recited letter name (NameThai = the
  name word's own text, e.g. กอ ไก่ "gɔɔ gài", one Word whose recording
  says the whole name), keyword with its own picture and gloss, and the
  name word's recording as Audio. One card; no reverse family (F6).
  Name and keyword are Words (Grapheme.name_word, Grapheme.keyword);
  their recordings/pictures come through the normal sourcing path. No
  substitute audio: a grapheme whose name word has no current-best
  recording drops the card, counted.

**sentence** (one note per adopted Sentence, at its order position):
fields ThaiCloze, Thai, TargetWord, Audio, ScenePicture, Gloss,
GrammarNote, Productive, ReviewNote, CompileId. Gloss = Sentence.gloss;
TargetWord is the sentence's last used word, the target it enters the
order after; Productive gates the Cloze card (non-empty iff that target
is productive). Tags: one target::ID per target the sentence fills, one
sentence::SHA.
- Cloze (productive last used word only): front cloze on that word +
  optional scene picture; back target word, NATIVE audio (F7), gloss.
- Listening (receptive): front audio; back full text, target, gloss.
ThaiCloze is the sentence's rendering with every element whose word is
the last used word blanked (a repeated word keeps its ๆ outside the
blank), never str.replace over the text (the ยา/โรงพยาบาล corruption
class: blanking "medicine" inside "hospital" cannot arise from
elements).

## 2. Identity, tags, order

- guid: word = word id; grapheme = symbol; pair = MemberKey;
  sentence = text_sha. A replaced sentence resets
  its scheduling; everything else updates in place.
- Tags are atomic, one part per tag, never composed or split: family::,
  kind:: (card kind), word::ID (word notes), pair::ID, confusion::ID,
  member::INDEX and speaker::ID (pair notes), grapheme::SYMBOL,
  target::ID per filled target and sentence::SHA (sentence notes),
  compile::ID,
  src tags for audio/image provenance. The import reads each tag's
  value by prefix and writes the parts as study columns (spec 2); a word
  card's Target is derived from its kind (Listening receptive, Production
  productive).
- due: from Syllabus.order(); each order() entry owns a block of
  (notes it yields) × STRIDE, cumulative, so no two entries' notes share
  a due; sibling cards of one note get separated due values within the
  block, and the shipped deck options group sets bury-siblings. The two
  member notes of a pair are not siblings: they sit a stride apart inside
  the pair's block (A5).
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
is the normalized file: the judge sees the pixels the card shows. CSS
retains only final fit-to-viewport.

## 4. Return path

- **Revlog import**: read collection.anki2 read-only (proven pattern);
  map card -> (card_key, compile id) via tags/CompileId; append study
  rows. Idempotent by (card_key, ts).
- **Flag import**: flags become learner assessments (cache rows) on the
  entity's subject (word id, pair id, grapheme symbol, sentence text_sha)
  with role from (family, card kind): word Listening and sentence
  Listening flag the recording (tone role: a re-verification request,
  never an override); word Production flags the current picture (a
  learner picture-for-word rating on its sha); a card with no artifact
  role (Reading, Spelling, Recognition, Cloze, grapheme Reading, or a
  Production card whose word has no picture) is a card-level flag, which
  makes the subject directed in the queue (spec 3 §6) and appears on the
  subject screen (spec 5). The idempotence key is the typed
  FlagKey(family, anchor, card_kind, flags); no marker rows.
- **ReviewNote harvest**: read fields directly from the collection
  (read-only, proven); each non-empty note appends a learner row on the
  note's anchor (from its tags) under key learner-note:ANCHOR:sha(TEXT) — re-harvesting the same text is an
  exact-key hit (no duplicate), edited text is a new key (reprocessed),
  a cleared field appends nothing and retracts nothing: a newer note on
  the same subject supersedes on read, and a retraction is a superseding
  row. Harvest never writes to Anki.
- Import is one command; it reports rows imported per kind and rows
  skipped with reasons.

## 5. Explicitly out

- No deck deletion/orphan cleanup in v1 (delete-and-reimport is the
  current practice; AnkiConnect-based cleanup is a later addition).
- No native Anki cloze type: the two-template design (cloze + listening)
  stays; corruption is impossible by construction (element replacement).
- Scheduling migration: out of scope, not prohibited. Guid stability
  preserves scheduling across reimports, which covers current needs;
  cross-collection migration (AnkiConnect/colpkg) is possible if ever
  wanted.
