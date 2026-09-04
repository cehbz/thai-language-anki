# Spec 4: The Anki boundary

Revision 3, proposed 2026-09-04 against principles r2 and architecture
r2 (r1 promoted 2026-09-04 as written on 2026-09-03 against the
principles draft). Re-checked against principles r1 and architecture r1
on 2026-09-04; the revisions that re-check proposed enter as r2 on
approval. Revision process as in docs/architecture.md: proposals on
evidence, explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: the two mechanisms principles r2 moved out of A2 and A3
  (fields append; first field is the note identity).
- r3 2026-09-04: field lists as shipped (ProductiveTarget, Thai,
  Productive gate); choice order on pair fronts; pair member notes
  separated; NameThai one reading, no substitute audio; Gloss from the
  Sentence; word:: tags with the Target derived; flag roles by (family,
  kind), card-level flags direct the subject; harvest keyed by anchor.
  Evidence: implementation review 2026-09-04.

Scope: Syllabus.compile() — the translation of Syllabus state into Anki's
domain — and the return path: revlog, flags, and ReviewNote harvests.
Anki's domain is adopted unmodified (architecture §6); nothing here
re-litigates it.

## 1. Models and cards (the card taxonomy, principles draft)

Model ids: sha-derived from model name, stable; fields only ever
append, so an existing collection updates in place (A2). A note's first
field is its identity, unique within its model (A3). Every model carries the
card CSS (legible Thai, bounded images, answer distinct, night mode — as
shipped in the current compiler) and two service fields rendered by no
template: ReviewNote (the mid-review comment channel) and CompileId.

**word** (from a Word with a vocabulary Target):
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
fields as current plus BOTH members' audio on the back — back shows both
members, marks the stimulus ("you heard: ..."), each member's audio
individually playable (F6b). First field = MemberKey
"PAIRID:SPEAKER:INDEX" (unique; fixes the MemberIndex dupe-key defect).
- Recognition: front stimulus audio + both spellings as choices, in
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

**sentence** (one note per (Target, Sentence) with a card-yielding skill):
fields ThaiCloze, Thai, TargetWord, Audio, ScenePicture, Gloss,
GrammarNote, Productive, ReviewNote, CompileId. Gloss = Sentence.gloss;
Productive gates the Cloze card (non-empty iff the Target is productive).
- Cloze (productive Target only): front cloze + optional scene picture;
  back target word, NATIVE audio (F7), gloss.
- Listening (receptive): front audio; back full text, target, gloss.
ThaiCloze is built by token-boundary replacement via the tokenizer port —
never str.replace (the ยา/โรงพยาบาล corruption class: blanking "medicine"
inside "hospital").

## 2. Identity, tags, order

- guid: word = word id; grapheme = symbol; pair = MemberKey;
  sentence = (target id, sentence text_sha). A replaced sentence resets
  its scheduling; everything else updates in place.
- Tags: family::, kind:: (card kind), word::ID (word notes), pair::ID and
  confusion::ID (pair notes), grapheme::SYMBOL, sentence::TARGET:SHA,
  compile::ID, src tags for audio/image provenance. card_key (spec 2) =
  anchor::kind; a word card's Target is derived from its kind (Listening
  receptive, Production productive).
- due: from Syllabus.order(); sibling cards of one note get separated due
  values (offset by a stride), and the shipped deck options group sets
  bury-siblings. The two member notes of a pair are not siblings: they
  are placed a stride apart or interleaved with other pairs of the
  confusion, and never adjacent (A5).
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
  note's subject with role from (family, card kind); a flag on a
  tone-correctness role appends a re-verification request instead of
  overriding (authority per (backend, role)); a flag on a card with no
  artifact role (Reading, Spelling, Recognition, Cloze) is a card-level
  flag, which makes the subject directed in the queue (spec 3 §6) and
  appears on the subject screen (spec 5). The idempotence key is the
  (card, flags) fact itself; no marker rows.
- **ReviewNote harvest**: read fields directly from the collection
  (read-only, proven); each non-empty note appends a learner row on the
  note's anchor (from its tags) under key learner-note:ANCHOR:sha(TEXT) — re-harvesting the same text is an
  exact-key hit (no duplicate), edited text is a new key (reprocessed),
  a cleared field appends nothing and retracts nothing (prior directions
  remain history). No built-in retraction mechanism for a learner note
  exists (YAGNI); a newer note on the same subject supersedes in
  practice (newest-wins reads), and a true retraction is done
  conversationally — the assistant appends a superseding row on request. Clearing the field in Anki requires AnkiConnect and
  is a separate, optional step — harvest never writes to Anki.
- Import is one command; it reports rows imported per kind and rows
  skipped with reasons.

## 5. Explicitly out

- No deck deletion/orphan cleanup in v1 (delete-and-reimport is the
  current practice; AnkiConnect-based cleanup is a later addition).
- No native Anki cloze type: the two-template design (cloze + listening)
  stays; corruption is fixed by tokenized replacement instead.
- Scheduling migration: out of scope, not prohibited. Guid stability
  preserves scheduling across reimports, which covers current needs;
  cross-collection migration (AnkiConnect/colpkg) is possible if ever
  wanted.
