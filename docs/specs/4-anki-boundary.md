# Spec 4: The Anki boundary

Revision 1, promoted 2026-09-04 as written on 2026-09-03 against the
principles draft. Re-checked against principles r1 and architecture r1
on 2026-09-04; the revisions that re-check proposed enter as r2 on
approval. Revision process as in docs/architecture.md: proposals on
evidence, explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.

Scope: Syllabus.compile() — the translation of Syllabus state into Anki's
domain — and the return path: revlog, flags, and ReviewNote harvests.
Anki's domain is adopted unmodified (architecture §6); nothing here
re-litigates it.

## 1. Models and cards (the card taxonomy, principles draft)

Model ids: sha-derived from model name, stable. Every model carries the
card CSS (legible Thai, bounded images, answer distinct, night mode — as
shipped in the current compiler) and two service fields rendered by no
template: ReviewNote (the mid-review comment channel) and CompileId.

**word** (from a Word with a vocabulary Target):
fields Thai, Meaning, Picture, Audio, Ipa, Classifier, FrontGloss,
TestSpelling, ReviewNote, CompileId.
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
- Recognition: front stimulus audio + both spellings as choices.

**grapheme**:
fields Symbol, Sound, NameThai, KeywordThai, KeywordGloss,
KeywordPicture, Audio, ReviewNote, CompileId.
- Reading: front symbol; back the recited letter name (NameThai, e.g.
  กอ ไก่ "gɔɔ gài"), keyword with its own picture and gloss, and the
  name-word's recording as Audio. One card; no reverse family (F6).
  Name and keyword are Words (Grapheme.name_word, Grapheme.keyword);
  their recordings/pictures come through the normal sourcing path.

**sentence** (one note per (Target, Sentence) with a card-yielding skill):
fields ThaiCloze, TargetWord, Audio, ScenePicture, Gloss, GrammarNote,
ReviewNote, CompileId.
- Cloze (productive Target): front cloze + optional scene picture; back
  target word, NATIVE audio (F7), gloss.
- Listening (receptive): front audio; back full text, target, gloss.
ThaiCloze is built by token-boundary replacement via the tokenizer port —
never str.replace (the ยา/โรงพยาบาล corruption class: blanking "medicine"
inside "hospital").

## 2. Identity, tags, order

- guid: word = word id; grapheme = symbol; pair = MemberKey;
  sentence = (target id, sentence text_sha). A replaced sentence resets
  its scheduling; everything else updates in place.
- Tags: family::, kind:: (card kind), target::ID, confusion::ID (pairs),
  compile::ID, src tags for audio/image provenance. Tags carry everything
  StudyRecords need to map a review back (spec 2 card_key = tags-derived).
- due: from Syllabus.order(); sibling cards of one note get separated due
  values (offset by a stride), and the shipped deck options group sets
  bury-siblings — both, per A5.
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
  note's subject with role from the card kind; a flag on a
  tone-correctness role queues machine re-verification instead of
  overriding (authority per (backend, role)).
- **ReviewNote harvest**: read fields directly from the collection
  (read-only, proven); each non-empty note appends a learner row under
  key learner-note:NOTEID:sha(TEXT) — re-harvesting the same text is an
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
