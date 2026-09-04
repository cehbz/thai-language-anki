# Spec 5: The feedback screen

Revision 1, promoted 2026-09-04 as written on 2026-09-03 against the
principles draft. Re-checked against principles r1 and architecture r1
on 2026-09-04; the revisions that re-check proposed enter as r2 on
approval. Revision process as in docs/architecture.md: proposals on
evidence, explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.

Scope: the learner-backend transport — the local web surface where the
learner answers the system's questions and reviews the deck. Grows out of
scripts/proof_gallery.py (kept patterns: local http.server, keyboard-first,
inline single-page UI, read-only extraction, JSONL/db appends). Policy
lives in spec 3's derivations; this surface only presents and records.

## 1. Modes

**Proof gallery** (exists, kept): every card rendered front/back in
introduction order, sequential, no scheduling; per-card one-line notes;
pair drill with per-confusion accuracy logging; gloss overlay; stats.
Changes: reads the new Compile; notes append as learner assessment rows
via RecordWriter (not proof_notes.jsonl); drill results append as
study-adjacent evidence rows.

**Question session** (new): serves the spec-3 queue, capped by the
learner-attention budget (default 20/session, configurable), highest
expected gain first. Pull-based: the learner answers any number and
stops; unanswered questions stay queued. Question kinds:

1. **Rate a picture** (word or scene role): shows the English gloss (and
   for scenes the sentence gloss), the current artifact WITH the judge's
   verdict line, rejected candidates as thumbnails at judgeable size
   (click to enlarge), the query read-only. Actions: 1 unacceptable-none
   / 2 unacceptable-use-this (then pick a thumbnail) / 3 acceptable /
   4 good; optional one-line note (the Direction). Presentation at card
   size for the current artifact — the presentation is part of the
   question (F9 role key includes it).
2. **Direction request** (exhausted subject): what was tried — phrases,
   sources, best candidates, judge reasons — plus two actions: type a
   direction, or supply an artifact (file path or URL; URL fetched
   through imgfetch; recorded with learner provenance and an implicit
   use-this).
3. **Challenger comparison** (rubric change produced a candidate ranked
   above a learner-accepted artifact): side-by-side, keep or switch;
   never auto-switched.
4. **Re-ask with evidence** (StudyRecord contradiction): the original
   answer, the lapse evidence, re-rate.

Every answer appends one learner cache row keyed learner:sha(ARTIFACT):
ROLE (or the finding identity for waivers); the session shows a running
count against the budget and can be closed at any point with nothing
lost.

## 2. Server

One process: `syllabus review --deck DIR [--port 8877]`. Reads Syllabus
state, the cache (via AssessmentReader), and media/objects; writes only
via RecordWriter appends. Port 8877 (8765 reserved for AnkiConnect).
Endpoints: / (app), /api/queue, /api/cards, /api/answer (POST),
/api/supply (POST), /media/SHA, /stats. No external resources; inline
CSS/JS; keyboard-first (1-4 rate, n note, arrows navigate, g gloss,
s stats). localStorage for position only — all state of record is
server-side.

## 3. Stats

Per-session: answered/queued, per-confusion drill accuracy, counts of
exhausted subjects remaining. Per-deck: current-best coverage per need,
learner-rated good/acceptable/unacceptable counts, RunReport history.

## 4. Explicitly out

- No editing of curated data (words, targets) — hand-edit the YAML.
- No judge invocation from the screen (report/batch pays; the screen
  reads).
- No auth; localhost only.
- No mobile packaging; the phone surface is Anki itself (ReviewNote,
  flags), imported per spec 4.
