# Spec 3: Ports, attempts, and the sourcing run

Revision 1, promoted 2026-09-04 as written on 2026-09-03 against the
principles draft. Re-checked against principles r1 and architecture r1
on 2026-09-04; the revisions that re-check proposed enter as r2 on
approval. Revision process as in docs/architecture.md: proposals on
evidence, explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.

Scope: the Provide and Assess ports, every backend's contract (cost, cache
key, authority), the attempt per need kind, the derivations over the record
(current-best, pending, exhausted, queue), Budget, and the run. Storage
shapes are spec 2; domain consumers are spec 1; UI surfaces are spec 5.

Revision 2026-09-03 (second): adds the attempt, candidates, pending,
authority-driven current-best, cost on every answer, the sentence and
rendition attempts, completeness rules, and the carry-over contract.
Supersedes the first 2026-09-03 draft, whose run had no assess step and
whose picture attempts stopped at search.

## 1. Vocabulary

- **Need**: (subject, kind). Kinds: picture (Word), recording (Word),
  rendition (MinimalPair), sentence (open Targets, per run), grapheme
  keyword (Grapheme). `Syllabus.gaps()` enumerates needs.
- **Source**: a Provide backend. Attempts and budgets are per Source.
- **Attempt**: one Source tried for one need: fetch candidates, obtain a
  verdict on every candidate from the role's deciding authority, re-derive
  current-best. The record of an attempt is its provide rows plus the
  verdict rows it caused; nothing else is stored.
- **Candidate**: an artifact (content-addressed, spec 2) in a role, with
  verdicts and no adoption. Adoption is derived (current-best), never stored.
- **Verdict**: an Assess answer on (artifact, role) under a rubric; carries
  its cost.
- **Current-best / pending / exhausted**: folds over the record (section 6).

## 2. Port contracts

```
Provide.ask(source, question) -> Answer
  question: {subject, provides: picture|picture-bytes|recording|rendition|
             sentence|pair|phrase|entry, params}
  Answer:   {items: [...], cost, ts}      # empty items is an answer; cached

Assess.ask(backend, question) -> Verdict
  question: {subject, role, artifact_sha?, rubric?, params?}
  Verdict:  {value, evidence?, suggestion?, cost, ts}
```

Cache-first; a hit costs nothing and appends nothing; a miss executes and
appends one row. Consumers see ask() only.

**Cost contract.** Every Answer and Verdict carries the cost the backend
incurred, in that backend's currency, measured by the backend: Forvo one
lookup, TTS characters times rate, the api and batch judge tokens times the
model price in providers.yaml, the cli judge one call of quota, the learner
seconds. A transport that receives usage and drops it violates this
contract. Consumers: budget enforcement (section 7), cross-run accounting
from the record, queue order, the run report.

**Artifacts reach assessors as paths.** An Assess question naming an
artifact hands the backend the media-store path. Each transport encodes for
its wire at send time: api as base64 image blocks (the API accepts base64,
an https URL, or a Files-API id; no local-path form exists), cli as a scoped
directory holding a link to the one file, batch as base64 in the request.
No audio reaches any LLM transport (the API has no audio input); recording
roles are assessed by mechanical and, when calibrated, listener.

**Compound question.** A rendition question's subject is the MinimalPair;
its answer is one recording per member, all by one speaker
(`{items: [{member, sha, speaker}, ...]}`). A Source that cannot guarantee
one speaker answers empty.

## 3. Provide backends

| source | provides | key | cost | re-ask |
|---|---|---|---|---|
| openverse, pexels | picture (search hits with url) | source:query | free HTTP | new query = new key |
| wikimedia | picture (search hits with url, via generator=search + prop=imageinfo, iiprop=url) | wikimedia:query | free HTTP | same |
| imgfetch, audiofetch (bytes) | picture-bytes, recording-bytes | url | free | a fetch failure is not cached |
| forvo | recording; rendition (intersection of members' lookups: same username across members) | forvo:WORD (per member) | 1 lookup per ask, 450/day | never re-asked |
| tts | recording; rendition (one voice across members) | tts:VOICE:sha(TEXT) | cash per character | never re-asked |
| commission | recording; rendition | batch item id | money + weeks | out/in via batch files |
| llm | sentence (per run over open targets), phrase, entry | llm:PRODUCER:MODEL:sha(PROMPT) | cash or quota per transport | never re-asked; the prompt text is the contract |
| pair-search | pair | pairs:CONFUSION:DICT_VERSION | free | dictionary bump = new key |
| learner | any (supply) | none; rows are acts | attention | feedback screen only |

Measured 2026-09-03: of 562 word lookups 333 returned nothing; of 40
minimal-pair members 39 are on Forvo and 11 of 22 pairs have a same-speaker
rendition by intersection; Forvo's per-speaker listing returns nothing, so
speaker-directed search does not exist.

## 4. Assess backends and authority

| backend | roles | key | authority |
|---|---|---|---|
| judge (LLM) | picture-for-word (fit, preference), scene-for-sentence, sentence-for-target (naturalness, register), word facts | judge:sha(RUBRIC):ARTIFACT_SHA:ROLE | evidence; below learner where learner is qualified |
| mechanical | recording duration/format; media resolvable; fills(); provenance rules | parameter-explicit, e.g. mech:duration:0.2-5.0:sha | ground truth for what it checks |
| listener | recording-for-word | listener:MODEL:sha:ROLE | absent until calibrated; then above mechanical |
| learner | picture fit, sentence quality, recording flag, waiver, card flag | learner:sha:ROLE (no rubric) | final on fit/quality/waivers; a recording flag queues re-verification, never outranks fact |

**Authority order per role** (data in assessor.py): picture-for-word:
learner > judge. sentence-for-target: learner > judge. recording-for-word:
listener (when calibrated) > mechanical; learner flags queue, never rank.
rendition-for-pair: mechanical (one-speaker check).

**Provenance prior** (rulebook data, an ordered list of provenance kinds,
e.g. commission > forvo > tts): orders eligible candidates only where no
assessor has spoken. It never fails a candidate and any verdict outranks it.

**Judge transports**: cli / api / batch, selected in providers.yaml; the
run does not know which (section 7). Batch resume state is a cache row
keyed on the request set. report() never calls Assess.

## 5. Attempts per need kind

**Picture (Word).** Query = the word's image phrase if a human or judge
drafted one, else gloss head term + category qualifier. Source order:
openverse, wikimedia, pexels. One attempt: search, imgfetch the first N
(providers.yaml `image_candidates`, default 5), judge *fit* on each
(pass/fail, the old rubric texts verbatim), and if more than one passes
judge *preference* once over the passing set; then current-best. A judge
`suggestion` becomes the next attempt's phrase.

**Recording (Word).** Source order: forvo, tts, commission. Forvo attempt:
lookup (cached forever), download each item's mp3, mechanical
duration/format on each; current-best by authority then provenance prior.
TTS attempt: synthesize with a pool voice (production draws male), then
mechanical. Warn `recording/synthetic` when current-best is TTS (the
native-audio principle is kept as the target; commission is tracked in
TODO).

**Rendition (MinimalPair).** Source order: forvo (intersection of members'
lookups by username; one lookup per member, shared with the recording
need), tts (one voice), commission. Mechanical checks one speaker across
members and duration. Findings: none for native one-speaker;
`rendition/synthetic` (warn) for TTS; `rendition/mixed-speakers` (warn)
when the members' current-best recordings differ in speaker and no
rendition exists.

**Sentence (per run over open Targets).** One attempt per run, not per
target: the prompt carries the open targets and, per target, the vocabulary
met at its entry position (Syllabus.order), the profile register, and the
existing sentence openings to avoid. Each drafted text is a candidate:
mechanical `fills()` against the targets it claims, judge
sentence-for-target (naturalness; register), then adopt:
`Syllabus.add_sentence` with provenance. Adoption creates needs: the
sentence's recording (tts allowed for receptive-only; a productive fill
wants native, warn otherwise) and an optional scene picture. A candidate
that fills nothing is a rejected draft in the record.

**Grapheme keyword (Grapheme).** Source: llm proposal (concrete, picturable,
containing the symbol); mechanical `grapheme/keyword-contains-symbol`; the
learner adopts (curated data changes; a machine proposal never adopts
itself). Implemented after cutover.

**Pair (SoundConfusion).** pair-search (dictionary + G2P); mechanical
exact-confusion check; adoption into curated pairs is the learner's act.
Implemented after cutover.

## 6. Derivations (folds; never stored)

- **current_best(subject, kind)**: learner choice wins; else the candidate
  ranked highest by the most authoritative backend that has spoken on it
  for the role, under the current rubric (a stale-rubric verdict does not
  rank); among equals, the provenance prior; never below an artifact the
  learner rated acceptable. A passing mechanical verdict ranks a recording;
  a passing judge fit ranks a picture; preference orders passing pictures.
- **pending(subject, kind)**: some candidate lacks a verdict from the
  deciding authority, or its questions sit in a submitted batch. A pending
  need gets no new attempt and no escalation.
- **exhausted(subject, kind)**: unchanged: the last k attempts produced no
  candidate out-ranking current-best and the attempt cap is reached;
  reopened by learner input, a rubric change, or a new source.
- **queue(syllabus, budgets)**: order per the periodic-batch principle:
  (1) no artifact or learner-unacceptable, directed first; (2) an untried
  option remains (unasked suggestion, rubric changed, unsearched source);
  (3) acceptable/unrated by rank then attempts; excluded: good, exhausted,
  pending (pending is reported, not queued).
- **confusion_weights()**: unchanged.

## 7. Budget and the run

Budget per source in its currency: {max_asks?, max_cost?}; forvo 450/day,
learner 20/session. Spend is summed from the record for per-day budgets.

```
run(syllabus, budgets):
  sentence attempt over the open targets (one ask; its candidates enter
      the queue as sentence needs)
  for need in queue(syllabus, budgets):        # pending excluded
      for source in sources_for(need.kind):     # cheapest first
          if budget spent: continue
          attempt(need, source)                 # provide + assess all
          if pending(need): break               # verdicts outstanding
          if current_best(need) improved: break
  RunReport {attempted, improved, exhausted, pending, available,
             spend per source}
```

Every ask appends; kill-safe anywhere. With an inline transport a pending
need resolves inside the same attempt; with the batch transport it resolves
in a later run. The run is transport-agnostic.

## 8. Rules added

Error (completeness; compile refuses): `target/picture-required`,
`target/recording-required`, `target/sentence-required` (an adopted
sentence fills it), `pair/rendition-required`,
`grapheme/keyword-picture-required`. Warn: `recording/synthetic`,
`rendition/synthetic`, `rendition/mixed-speakers`,
`sentence/synthetic-productive`. Judged: `picture/fit` (old texts),
`picture/preference`, `sentence/register-natural`. Per-deck severity
overrides live in rulebook.yaml severities; that is the only relaxation
path besides compile --force.

## 9. Configuration

providers.yaml adds `judge.price_per_mtok: {input, output}` and
`image_candidates` (5). The provenance prior lives in rulebook.yaml (it is
a judgement, not a route). rulebook.yaml `rubrics` carries the picture/fit,
picture/preference, and sentence texts verbatim.

## 10. Carry-over (amends spec 2 section 4)

- Item 2: current-deck images as before, AND candidates.yaml's recorded
  pass/fail per candidate becomes the judge fit verdict under the picture-fit
  rubric, so studied pictures rank on day one. The old judge_cache.sqlite is
  NOT migrated: its key hashes the full prompt text, which is unrecoverable.
  Item 5 is withdrawn.
- Item 4: learner rows (flags, gallery notes, waivers) are keyed by word id
  via the guid map, never by guid.
- The machine-chosen marker is consumed by the feedback screen only.
- Audio and sentences still regenerate.

## 11. Report identity

Unchanged: Report carries `rulebook_id` alongside `syllabus_state_id`;
staleness = either differs.

## 12. Explicitly out

- No retries; a failed ask is not cached.
- No listener implementation; calibration first.
- No interactive judge.
- No stored need status of any kind.
