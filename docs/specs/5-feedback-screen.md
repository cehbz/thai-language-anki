# Spec 5: The feedback screen

Revision 10, proposed 2026-09-13 against principles r4 and architecture
r3. Revision process: docs/principles.md.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: excluded and unreachable in stats and on the subject screen.
- r3 2026-09-04: supply by kind through the ingest path; directions recorded
  as directions; two endpoints added; flags on the subject screen; the
  screen consumes the run's derivations.
- r4 2026-09-11: the header's re-check narrative and the learner key
  encoding (spec 3 §4 owns it) removed; no behavior changed.
- r5 2026-09-11: a gallery note records the card as shown and is listed
  under the card, marked stale once the card no longer shows what it
  named; a note that fails to save stays in the box and says so.
  Evidence: notes typed while the server was down vanished silently; a
  note on a card carried no trace of which rendering it was about.
- r6 2026-09-12: a rate question shows every compiled card of its
  subject, front and back, through the model's own template and CSS,
  above the artifact block; with no current artifact it offers only
  "none of these" and "use the picked candidate" and says so. Evidence:
  the first queue question (a scene need, no current picture) showed ten
  rejected thumbnails, a 1 to 4 scale and no Thai anywhere; the learner
  could not tell what was being asked. User ruling 2026-09-12: the
  screen shows at least what the Anki cards show, formatted like them.
- r7 2026-09-12: a rate question requires a candidate (a need with none
  waits for the machine while a source is left, and arrives as a
  direction request once exhausted); a recording renders as a player; a
  rejected candidate shows its deciding verdict (mechanical on a
  recording, judge on a picture); the empty-artifact text names the
  check. Evidence: 2026-09-12, a sentence recording need with one
  Forvo `nothing` and TTS untried reached the learner as a 1 to 4
  question with nothing shown; the learner could not tell what was
  asked. User approval 2026-09-12.
- r8 2026-09-13: the deciding verdict is the highest-authority backend's
  verdict on the artifact, not the role's first backend's; a recording
  only the judge rejected showed "no verdict yet". User approval
  2026-09-13.
- r9 2026-09-13: `n` comments on any card or question (a session
  comment anchors on the subject under card kind "question", recording
  the question kind, the artifact kind, the subject kind and what was
  shown); a comment's identity is derived from its own row; every
  rendered card carries its type label with the one-line meaning of
  that card as a tooltip (the table lives with the models, spec 4 §1);
  the sentence Listening back labels its target word; a question names
  its subject in words (a word's Thai, id and meaning; a sentence's text
  and gloss), never a sha; the subject's comments are listed under a
  question, each "unread" until a run reads it (r10). Evidence: rating a
  sentence's scene picture offered no way to say the sentence itself
  was bad; `n` was inert in a session; a rate question named its subject
  by a sha; a Listening back read as sentence plus a stray word. User
  approval 2026-09-13.
- r10 2026-09-13: under each comment the screen shows its reading once
  a run has read it (spec 3 r30): the reading line, one label per
  action taken ("direction for the picture search: …", "sentence
  retired: …", "replacement drafted: …", "rating N on the …", "gloss
  on: …", "no action: …") and per request the deck could not act on
  ("no action available: …"); a strike control writes a veto row
  against that reading (spec 2 r15), after which every fold ignores the
  rows it produced and the reading shows as struck; a retirement
  stands; the stats history shows comments_read, comment_actions and
  comment_unactionable per run. Evidence: a reading the learner could
  not see or undo would make the comment channel a black box. User
  approval 2026-09-13.

Scope: the learner-backend transport — the local web surface where the
learner answers the system's questions and reviews the deck. Policy lives
in spec 3's derivations; this surface only presents and records.

## 1. Modes

**Proof gallery**: every card rendered front/back in introduction order,
sequential, no scheduling; per-card one-line comments (`n`; r9); pair
drill with per-confusion accuracy logging; gloss overlay; stats. A note
appends as a learner assessment row via RecordWriter, recording the card
as shown: the artifact shas it displayed, the sentence text for a
sentence card, and the syllabus state id. The card lists its notes
thereafter, each marked stale once the card no longer shows what the
note named (F9: an answer is about the thing shown). Every rendered
card, gallery or question, shows its type (family and kind as
`/api/cards` reports them) with the one-line meaning of that card type
as a tooltip (r9). A comment is listed with its reading under it in both
modes — "unread" until a run has read it (r10). Under each comment:
`unread`, or its reading with the actions taken and the unactionable
requests, and a strike control that writes a veto row against that
reading; folds that consume a comment-derived row (directions, ratings)
ignore vetoed ones, and a replacement sentence the struck reading
drafted is no longer adoptable (it drops out of the drafts the run
reads back); a retirement inferred from a comment acts at once and
striking it re-adopts nothing (r10). A note that fails to save stays in
the box with a visible failure and retries on Enter; a strike that fails
to save says so beside the control and the reading is left as it was.
Drill results append as study-adjacent evidence rows.

**Question session**: serves the spec-3 queue, capped by the
learner-attention budget (default 20/session, configurable), highest
expected gain first. Pull-based: the learner answers any number and
stops; unanswered questions stay queued. Question kinds:

1. **Rate a picture** (word or scene role): shows the English gloss (and
   for scenes the sentence gloss), every compiled card of the subject
   front and back through the model's own template and CSS (what
   `/api/cards` serves the gallery; r6), the current artifact WITH the
   judge's verdict line, rejected candidates as thumbnails at judgeable
   size (click to enlarge), each captioned with its deciding verdict
   (the newest fresh verdict by the highest-authority backend that has
   one on the artifact, the same rule ranking uses; r7, r8); a recording
   renders as a player wherever a picture would be a thumbnail, the
   query read-only. Actions: 1
   unacceptable-none / 2 unacceptable-use-this (then pick a thumbnail) /
   3 acceptable / 4 good; optional one-line note (the Direction). With
   no current artifact only 1 (none of these) and 2 (use the picked
   candidate) are offered, and the block says so (r6). Presentation at
   card size for the current artifact — the presentation is part of the
   question (F4, F9). A need with no candidate on record is not a rate
   question: while it has a source left it is the machine's; exhausted,
   it is kind 2 (r7).
2. **Direction request** (exhausted subject): what was tried — phrases,
   sources, best candidates, judge reasons — plus two actions: type a
   direction, or supply an artifact (file path or URL; a URL is fetched
   by kind, imgfetch for pictures and audiofetch for recordings; the
   bytes go through the media ingest path, normalized, with a
   provenance row source=learner, and an implicit use-this). A typed
   direction is recorded as a direction, not as a rating.
3. **Challenger comparison** (rubric change produced a candidate ranked
   above a learner-accepted artifact): side-by-side, keep or switch;
   never auto-switched.
4. **Re-ask with evidence** (StudyRecord contradiction): the original
   answer, the lapse evidence, re-rate.

`n` on any question comments on its subject (r9): the row is the gallery
comment's shape anchored on the subject under card kind `question`,
carrying the question kind, the artifact kind, the subject kind and what
the question showed. Every question names its subject in words: a word's
Thai, id and meaning; a sentence's text and gloss; a pair's members;
never a sha (r9). The subject's comments, any card or question, are
listed under it.

Every answer appends one learner row (spec 3 §4's key; the finding
identity for waivers); the session shows a running count against the
budget and can be closed at any point with nothing lost.

## 2. Server

One process: `thai-syllabus review --deck DIR [--port 8877]`. Reads
Syllabus state, the cache (via AssessmentReader), and media/objects;
writes only via RecordWriter appends. Port 8877 (8765 reserved for
AnkiConnect). Endpoints: / (app), /api/queue, /api/cards, /api/answer
(POST), /api/supply (POST), /api/note (POST), /api/veto (POST),
/api/drill (POST), /media/SHA, /stats. No external resources; inline
CSS/JS; keyboard-first (1-4 rate, n comment, arrows navigate, g gloss, s
stats; the strike is a button, not a key). localStorage for
UI conveniences only (position, mode, gloss toggle); nothing of record
lives in the browser.

## 3. Stats

Per-session: answered/queued, per-confusion drill accuracy, counts of
exhausted subjects remaining. Per-deck: current-best coverage per need,
learner-rated good/acceptable/unacceptable counts, RunReport history
with every field of spec 3 §7 (the comment counts among them, r10). The
per-subject screen lists excluded candidates with the reason and the
subject's card-level flags (spec 4 §4). Every derivation the screen
shows (current-best, exhausted, queue, coverage) comes from the same
wired media index and parameters the run uses; the screen computes none.

## 4. Explicitly out

- No editing of curated data (words, targets) — hand-edit the YAML.
- No judge invocation from the screen (the run pays; the screen reads).
- No auth; localhost only.
- No mobile packaging; the phone surface is Anki itself (ReviewNote,
  flags), imported per spec 4.
- No unstriking a reading (r10): a strike is one-way from the screen —
  it appends a veto row, and nothing on the screen retracts one.
