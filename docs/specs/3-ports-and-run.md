# Spec 3: Ports, attempts, and the sourcing run

Revision 10, proposed 2026-09-07 against principles r2 and architecture
r2. Revision process as in docs/architecture.md: proposals on evidence,
explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: one Message Batch per run, one source per need per run
  (§7); pending narrowed to an unresolved batch (§6); improved defined,
  excluded and unreachable reported (§7); recording voice constraint and
  speaker attributes, pools (E7, §5); coverage/speakers (§8); an attempt
  appends under the need's own subject (§5); authority order as domain
  data (§4); carry-over rewritten (§10); the 09-03 roster row for
  audiofetch approved.
- r3 2026-09-04: one key function per backend; the rendition answer is
  the artifact set compile resolves; sentence drafts carry a judged
  gloss; flags and re-verification requests make a subject directed.
  Evidence: implementation review 2026-09-04.
- r4 2026-09-05: keys are typed values; the encoding is a storage
  identity, never parsed. User ruling 2026-09-05.
- r5 2026-09-05: a Source transport failure skips the source for the
  run and is counted, only the judge stops the run; RunReport accounts
  for every need gaps() lists. Evidence: Task B3 and B4 reviews.
- r6 2026-09-06: at most one batch outstanding; RunReport gains
  unserved, budgeted, deferred and drafted with the accounting identity;
  day budgets from the record. Evidence: Task B5 and B6 reviews.
- r7 2026-09-06: source selection and exhaustion fold over attempt
  outcomes; a transient failure never advances a need. Evidence:
  audiofetch failures after a successful Forvo lookup advanced needs
  to TTS (final review follow-up).
- r8 2026-09-06: on recording and rendition roles the learner vetoes
  and never ranks upward. Evidence: Task B8 found a learner "good" on
  a rendition with no known speaker ranking above the coverage
  measure; user ruling 2026-09-06 (the learner may be hearing two
  speakers, not the contrast, but can tell an unintelligible
  recording).
- r9 2026-09-06: the drafter's transport is configured on its own
  (`drafter.transport`, cli by default); a drafter transport failure is
  a source failure; the drafting prompt lists the vocabulary once with a
  cutoff per target; the api and batch transports send a thinking
  setting (`judge.thinking`). Evidence: the first cutover run 2026-09-06
  (the api drafter's 4096 output tokens all spent on adaptive thinking,
  no text; 97K input tokens for 40 targets; the run died with no
  report); user ruling 2026-09-06 (subscription quota over cash where
  the CLI can do the job).
- r10 2026-09-07: failure taxonomy (§6a): three states, the retryable
  one bounded by `transient_cap`; a backend appends only an answer it
  recognized; a served refusal of a url from a cached answer re-asks that
  answer once (Forvo urls are time-limited); a missing artifact is
  excluded, never fatal; a batch is outstanding until ended;
  `judge.thinking: adaptive` needs `judge.max_tokens`; `search_proxy` is
  the forward proxy for Openverse search; the fills key names its
  curated and tokenizer version. Evidence: smoke run 2 (every migrated
  Forvo url expired; the proxy answered 400 to a base-url request) and
  the curable-failure audit, both 2026-09-07; user rulings 2026-09-07
  (three states with a retry limit; a served non-audio body is curable).

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
- **Key**: a typed value per backend (a frozen dataclass of its parts),
  defined in this spec's key module and used by every writer and reader.
  Its canonical encoding exists only as the store's identity (key_sha);
  no module parses, splits or prefix-matches an encoded key. A consumer
  that needs a key's parts reads the row's question and subject.
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
appends one row. Consumers see ask() only. reask(backend, question)
executes and appends a row over a hit; the newest row is the answer. Only
the attempt calls it, and only under §6a's re-ask rule. A backend appends
a row only for an answer it positively recognized as the answer to the
question asked: a body of the wrong shape, an unparseable verdict, a
completion with no drafts raise and append nothing.

**Cost contract.** Every Answer and Verdict carries the cost the backend
incurred, in that backend's currency, measured by the backend: Forvo one
lookup, TTS characters times rate, the api and batch judge and the api
drafter tokens times the model price in providers.yaml, the cli judge and
the cli drafter one call of quota, the learner seconds. A transport that receives usage and drops it violates this
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
| openverse, pexels | picture (search hits with url) | source:query | free HTTP | new query = new key; re-asked once per attempt when every hit is refused by its server (§6a) |
| wikimedia | picture (search hits with url, via generator=search + prop=imageinfo, iiprop=url) | wikimedia:query | free HTTP | same |
| imgfetch, audiofetch (bytes) | picture-bytes, recording-bytes | url | free | a refusal is typed (§6a): served or wire; never cached against the url |
| forvo | recording; rendition (intersection of members' lookups: same username across members) | forvo:WORD (per member) | 1 lookup per ask, 450/day | re-asked once per attempt when a url has expired (§6a) |
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
| learner | picture fit, sentence quality, recording veto, waiver, card flag | learner:sha:ROLE (no rubric) | final on fit/quality/waivers; on recording and rendition roles a veto on fitness: unacceptable-none excludes the artifact from current-best and reopens the need, unacceptable-use-this nominates its artifact (it ranks once the machine verdict passes it, like a supplied one), acceptable/good is recorded and shown and never ranks, since correctness of tone and speaker is not the learner's to certify; an Anki flag queues re-verification |

**Authority order per role** (domain data, spec 1 §4): picture-for-word:
learner > judge. sentence-for-target: learner > judge. recording-for-word
and recording-for-sentence: listener (when calibrated) > mechanical;
the learner vetoes, never ranks. rendition-for-pair: the rendition
backend (one-speaker check); the learner vetoes, never ranks. A vetoed
artifact ranks again only when the learner re-rates it.

**Provenance prior** (rulebook data, an ordered list of provenance kinds,
e.g. commission > forvo > tts): orders eligible candidates only where no
assessor has spoken. It never fails a candidate and any verdict outranks it.

**Judge transports**: cli / api / batch, selected in providers.yaml; the
run does not know which (section 7). Batch state is one marker row per
run, keyed on the batch id, released when the batch ends. report() never
calls Assess. A batch is outstanding until its
status is ended; a result of type expired or errored carries no verdict
and its question re-asks.

**Drafter transport**: cli / api, selected in providers.yaml
`drafter.transport` (cli by default); api rides the judge's account,
model and price. `judge.thinking` (disabled by default, or adaptive) is
sent by the api and batch transports on every request; `adaptive`
requires `judge.max_tokens` (at least 16000), which both transports send.

## 5. Attempts per need kind

**Picture (Word).** Query = the word's image phrase if a human or judge
drafted one, else gloss head term + category qualifier. Source order:
openverse, wikimedia, pexels. One attempt: search, imgfetch the first N
(providers.yaml `image_candidates`, default 5; a served refusal of every
hit re-asks the search once and ingests what is new), judge *fit* on each
(pass/fail, the old rubric texts verbatim), and if more than one passes
judge *preference* once over the passing set; then current-best. A judge
`suggestion` becomes the next attempt's phrase.

**Recording (Word).** Source order: forvo, tts, commission. Voice
constraint (E2, E7): male if the word has a productive Target (the
recording plays on the productive back), any sex otherwise; within the
constraint the pick spreads over the pool. Forvo attempt: lookup (cached;
re-asked once within the attempt when a download of one of its urls is
refused by the server, Forvo urls being time-limited, and the item retried
by its Forvo id), download each item's mp3, mechanical duration/format on
each;
the item's sex and country are recorded on the speaker (spec 2);
current-best by authority then provenance prior. TTS attempt: synthesize
with a pool voice (pools per sex in providers.yaml; the roster's sex is
recorded on the speaker), then mechanical. TTS supplies sex and timbre
only; Forvo and commissions supply age and accent. Warn `recording/synthetic` when current-best is TTS (the
native-audio principle is kept as the target; commission is tracked in
TODO).

**Rendition (MinimalPair).** Source order: forvo (intersection of members'
lookups by username; one lookup per member, shared with the recording
need and re-asked per member under the same rule), tts (one voice),
commission. The attempt appends its ask under
the pair, the need's own subject, even though the lookups are cached per
member: exhausted() counts attempts per need. The answer row carries the
per-member shas and the speaker; a rendition is that artifact set, and
compile resolves the pair's current-best rendition from it (a pair with
none does not compile). Mechanical checks one speaker across
members and duration. Findings: none for native one-speaker;
`rendition/synthetic` (warn) for TTS; `rendition/mixed-speakers` (warn)
when the members' current-best recordings differ in speaker and no
rendition exists.

**Sentence (per run over open Targets).** One attempt per run, not per
target: the prompt carries the vocabulary met by the furthest handed target
once, in entry-position order (Syllabus.order), and per target the count of
that list it may use (the per-target vocabularies nest by position), the
profile register, and the existing sentence openings to avoid. Each drafted text is a candidate:
mechanical `fills()` against the targets it claims, judge
sentence-for-target (naturalness; register), then adopt:
`Syllabus.add_sentence` with provenance. Each draft carries its L1
gloss, judged with the text (a gloss that misstates the sentence fails
the candidate). Adoption creates needs: the sentence's recording (tts
allowed for receptive-only; a productive fill wants native, warn
otherwise) and an optional scene picture. A candidate
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
- **pending(subject, kind)**: a question about one of its candidates
  sits in a submitted, unresolved batch. Nothing else is pending: with an
  inline transport a verdict arrives inside the attempt, an unpreparable
  question is excluded, and a judge that cannot be reached stops the
  run. A pending need gets no new attempt.
- **outcome(subject, kind, source)**: what one attempt of a need at a
  source produced: `candidates` (at least one artifact from it was
  stored and checked), `nothing` (the source answered and nothing
  usable came of it), or `transient-failure` (the ask or any fetch it
  needed failed on the wire; retry). The attempt appends one outcome
  row per source it asks (port `attempt`, backend = the source, key
  AttemptOutcomeKey(subject, kind, source)); `candidates` and `nothing`
  count as tried, and so does a source with `transient_cap`
  transient outcomes since the anchor (§6a).
- **next_source(subject, kind)**: the first of the kind's sources,
  cheapest first, not tried since current-best last changed. A
  `transient-failure` outcome advances the need only at the cap (§6a).
- **exhausted(subject, kind)**: over outcomes, not asks: the last k
  attempts produced no candidate out-ranking current-best and the
  attempt cap is reached;
  reopened by learner input, a rubric change, or a new source.
- **queue(syllabus, budgets)**: order per the periodic-batch principle:
  (1) no artifact or learner-unacceptable, directed first (a learner
  direction, an unconsumed re-verification request from a flag on a tone
  role, or a card-level flag all make a subject directed); (2) an untried
  option remains (unasked suggestion, rubric changed, unsearched source);
  (3) acceptable/unrated by rank then attempts; excluded: good, exhausted,
  pending (pending is reported, not queued).
- **confusion_weights()**: unchanged.

## 6a. Failure taxonomy

Every ask and fetch ends in one of three states:

- **Got it.** An artifact stored, or an answer the backend recognized.
  Appended; outcome `candidates`.
- **Failed definitively for the subject.** The source answered and
  what it answered cannot serve this need: an empty lookup, a search with
  no hits, a synthesis the service refuses for this text or voice (a
  4xx other than 429), a downloaded artifact that fails its mechanical
  check. Outcome `nothing`; the need advances to the next source; the
  attempt counts toward the cap.
- **Failed; a retry may succeed.** A wire failure (timeout, connection,
  DNS, a fetcher that cannot run), a 5xx or 429, a served refusal of a
  url (a non-200, a body of the wrong type, undecodable bytes), a batch
  not yet ended. Nothing is appended for the ask; the attempt's outcome
  is `transient-failure`. Bounded: once a source has `transient_cap`
  (providers.yaml, default 3) transient outcomes since the escalation
  anchor, it counts as tried: next_source advances past it and exhausted
  counts it as one attempt. Learner input resets the anchor as for every
  other outcome.

A backend appends a row only for an answer it positively recognized
(§2). A refusal carries a typed reason, never matched as text: the
fetchers report `{"refused": kind, "detail": ...}` on stdout, kind one
of wire | http | content-type | too-large | format | io; wire is not
served, every other kind is.

**Re-ask rule.** A served refusal of a url taken from a cached answer
(a Forvo lookup, an image search) re-asks that answer once within the
attempt (`reask`) and retries the fetch once: a Forvo item by its id, a
search by the hits not yet tried. A second served refusal, or any wire
failure, is transient. Nothing infers durability from a response; the
cap decides it.

**Unreachable versus excluded.** A question the backend cannot prepare
(a missing or unreadable artifact, a member with no verdict) is excluded
for the run and never cached; only a backend that answered none of the
questions it was given on the wire is unreachable. An unrun check is
never a failed check.

**Keys over mutable state name its version.** A verdict computed from
the curated files or a tokenizer (fills) carries a version of that
state in its key, as pair-search carries the dictionary version.

## 7. Budget and the run

Budget per source in its currency: {max_asks?, max_cost?}; forvo 450/day,
learner 20/session. Spend is summed from the record for per-day budgets.

One source per need per run; one judge batch per run; at most one batch
outstanding: when the previous run's batch is still in progress, the run
adopts what has resolved, attempts nothing, submits nothing, and reports
that batch's subjects as pending and the rest as deferred. Escalation to
the next source happens on the next run, for every transport alike, so
the loop has one shape and a run is cheap and repeatable (F10). Per-day
budgets are measured from the record since the local day's start plus
this run's spend.

```
run(syllabus, budgets):
  resolve the previous run's batch, if any: append its verdicts, release
      its marker (a batch ends with expired or errored results too; those
      questions re-ask). Pending clears here.
  sentence attempt over the open targets (one ask; its candidates enter
      the queue as sentence needs)
  questions = []
  for need in queue(syllabus, budgets):        # pending excluded
      source = next_source(need)               # cheapest source not yet
                                               # tried since current-best
                                               # last changed; none ->
                                               # exhausted, skip
      if budget spent: continue
      questions += attempt(need, source)       # provide; assess inline
                                               # where the transport is
                                               # inline, else collect
  submit(questions) as one batch; append its marker  # no-op if empty
  RunReport {attempted, improved, exhausted, pending, excluded,
             unreachable, available, unserved, budgeted, deferred,
             drafted, preferences, source_failures, spend per source}
  # available = every need gaps() lists, and
  # available == attempted + exhausted + pending + unserved + budgeted
  #              + deferred, always:
  #   exhausted  = needs with a Source whose next source is None
  #   pending    = needs (subject, kind) with a question in this run's
  #                batch or the earlier unresolved one; the loop never
  #                attempts a need that already has a question collected
  #                this run, so a need lands in exactly one bucket; a
  #                preference question on a picture that already satisfies
  #                its need is counted under preferences, outside the
  #                identity; questions collected but never submitted
  #                (the judge died) and needs the loop never reached count
  #                under deferred
  #   unserved   = needs whose kind has no Source and no per-run pass yet
  #   budgeted   = needs skipped because their Source's day budget was
  #                spent (every open Target within the drafting cap when
  #                the sentence drafter's budget is spent)
  #   deferred   = needs the run never considered: an earlier batch is
  #                still outstanding, the judge was unreachable at
  #                resolve, or open Targets beyond the per-run drafting cap
  #   attempted  counts needs whose attempt finished this run (inline
  #              verdicts, or no questions raised); a need whose questions
  #              await this run's batch counts as pending, not attempted;
  #              plus the open Targets handed to the drafter (within the
  #              cap) when the sentence attempt runs; drafted = the drafts
  #              it produced
```

improved = the need's current-best artifact sha differs after the
attempt; a re-ranking among unchanged artifacts is not improvement.
excluded = questions that could not be prepared (missing or unreadable
artifact), reported per need and skipped. unreachable = the judge could
not be reached; the run stops at the first such attempt and exits
non-zero (fail fast; nothing after it is attempted). A Source that
cannot be reached does not stop the run: the source is skipped for the
rest of the run, needs whose next source it is stay untouched and
count under deferred, the need whose attempt failed records a
`transient-failure` outcome and counts under deferred too, and the
report counts the failure under source_failures[source]. A drafter
transport failure is a source failure under source_failures["llm-sentence"]:
every open Target's need counts under deferred and the loop runs. Every ask
appends; kill-safe anywhere. The run is transport-agnostic.

## 8. Rules added

Error (completeness; compile refuses): `target/picture-required`,
`target/recording-required`, `target/sentence-required` (an adopted
sentence fills it), `pair/rendition-required`,
`grapheme/keyword-picture-required`. Warn: `recording/synthetic`,
`rendition/synthetic`, `rendition/mixed-speakers`,
`sentence/synthetic-productive`. Judged: `picture/fit` (old texts),
`picture/preference`, `sentence/register-natural`. Measure:
`coverage/speakers` (E7): per audio corpus (word recordings, renditions,
sentence recordings), distinct speakers per sex, age band, and region
against rulebook targets; unknown attributes never count. Per-deck severity
overrides live in rulebook.yaml severities; that is the only relaxation
path besides compile --force.

## 9. Configuration

providers.yaml adds `judge.price_per_mtok: {input, output}`,
`judge.thinking` (disabled | adaptive), `judge.max_tokens` (4096; at least
16000 under `thinking: adaptive`), `drafter.transport` (cli | api),
`image_candidates` (5) and `transient_cap` (3). `search_proxy` is the HTTP
forward proxy Openverse searches go through (media sourcing: Openverse
refuses a Thai egress); no other request uses it. The provenance prior lives in rulebook.yaml (it is
a judgement, not a route). rulebook.yaml `rubrics` carries the picture/fit,
picture/preference, and sentence texts verbatim.

## 10. Carry-over

Spec 2 §4 as revised (r2) is the contract: old candidate verdicts carry
under a legacy rubric id and never rank; every current picture is judged
under the current rubric by the first run's assess-first step; the old
judge_cache.sqlite is retired (its keys are opaque hashes of prompts,
recoverable only by replaying the old package); learner rows are keyed
by word id; no marker of the old deck's choice exists. Audio and
sentences regenerate.

## 11. Report identity

Unchanged: Report carries `rulebook_id` alongside `syllabus_state_id`;
staleness = either differs.

## 12. Explicitly out

- No retries; a failed ask is not cached.
- No listener implementation; calibration first.
- No interactive judge.
- No stored need status of any kind.
