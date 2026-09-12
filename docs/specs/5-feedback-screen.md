# Spec 5: The feedback screen

Revision 6, proposed 2026-09-12 against principles r4 and architecture
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

Scope: the learner-backend transport — the local web surface where the
learner answers the system's questions and reviews the deck. Policy lives
in spec 3's derivations; this surface only presents and records.

## 1. Modes

**Proof gallery**: every card rendered front/back in introduction order,
sequential, no scheduling; per-card one-line notes; pair drill with
per-confusion accuracy logging; gloss overlay; stats. A note appends as
a learner assessment row via RecordWriter, recording the card as shown:
the artifact shas it displayed, the sentence text for a sentence card,
and the syllabus state id. The card lists its notes thereafter, each
marked stale once the card no longer shows what the note named (F9: an
answer is about the thing shown). A note that fails to save stays in the
box with a visible failure and retries on Enter. Drill results append as
study-adjacent evidence rows.

**Question session**: serves the spec-3 queue, capped by the
learner-attention budget (default 20/session, configurable), highest
expected gain first. Pull-based: the learner answers any number and
stops; unanswered questions stay queued. Question kinds:

1. **Rate a picture** (word or scene role): shows the English gloss (and
   for scenes the sentence gloss), every compiled card of the subject
   front and back through the model's own template and CSS (what
   `/api/cards` serves the gallery; r6), the current artifact WITH the
   judge's verdict line, rejected candidates as thumbnails at judgeable
   size (click to enlarge), the query read-only. Actions: 1
   unacceptable-none / 2 unacceptable-use-this (then pick a thumbnail) /
   3 acceptable / 4 good; optional one-line note (the Direction). With
   no current artifact only 1 (none of these) and 2 (use the picked
   candidate) are offered, and the block says so (r6). Presentation at
   card size for the current artifact — the presentation is part of the
   question (F4, F9).
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

Every answer appends one learner row (spec 3 §4's key; the finding
identity for waivers); the session shows a running count against the
budget and can be closed at any point with nothing lost.

## 2. Server

One process: `thai-syllabus review --deck DIR [--port 8877]`. Reads
Syllabus state, the cache (via AssessmentReader), and media/objects;
writes only via RecordWriter appends. Port 8877 (8765 reserved for
AnkiConnect). Endpoints: / (app), /api/queue, /api/cards, /api/answer
(POST), /api/supply (POST), /api/note (POST), /api/drill (POST),
/media/SHA, /stats. No external resources; inline CSS/JS; keyboard-first
(1-4 rate, n note, arrows navigate, g gloss, s stats). localStorage for
UI conveniences only (position, mode, gloss toggle); nothing of record
lives in the browser.

## 3. Stats

Per-session: answered/queued, per-confusion drill accuracy, counts of
exhausted subjects remaining. Per-deck: current-best coverage per need,
learner-rated good/acceptable/unacceptable counts, RunReport history
with every field of spec 3 §7. The per-subject screen lists excluded
candidates with the reason and the subject's card-level flags (spec 4
§4). Every derivation the screen shows (current-best, exhausted, queue,
coverage) comes from the same wired media index and parameters the run
uses; the screen computes none.

## 4. Explicitly out

- No editing of curated data (words, targets) — hand-edit the YAML.
- No judge invocation from the screen (the run pays; the screen reads).
- No auth; localhost only.
- No mobile packaging; the phone surface is Anki itself (ReviewNote,
  flags), imported per spec 4.
