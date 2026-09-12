# Spec 3: Ports, attempts, and the sourcing run

Revision 27, proposed 2026-09-12 against principles r4 and architecture
r3. Revision process: docs/principles.md.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: one Message Batch per run, one source per need per run;
  improved, excluded, unreachable reported; voice constraint and speaker
  pools; coverage/speakers; authority order as domain data.
- r3 2026-09-04: one key function per backend; the rendition answer is the
  artifact set; drafts carry a judged gloss; flags make a subject directed.
- r4 2026-09-05: keys are typed values; the encoding is never parsed.
- r5 2026-09-05: a Source transport failure skips the source for the run;
  only the judge stops the run; RunReport accounts for every need.
- r6 2026-09-06: at most one batch outstanding; the accounting identity;
  day budgets from the record.
- r7 2026-09-06: source selection and exhaustion fold over attempt
  outcomes; a transient failure never advances a need.
- r8 2026-09-06: on recording and rendition roles the learner vetoes and
  never ranks upward.
- r9 2026-09-06: the drafter's transport configured on its own; a drafter
  failure is a source failure; judge thinking setting.
- r10 2026-09-07: failure taxonomy (§6a); a backend appends only an answer
  it recognized; the re-ask rule; search_proxy.
- r11 2026-09-07: assess-first.
- r12 2026-09-07: a judge key names its subject; legacy-current is a
  candidate's provenance; a current-best tie breaks by artifact sha.
- r13 2026-09-07: the newest verdict per backend and artifact ranks.
- r14 2026-09-08: the sentence attempt drafts for coverage, one candidate
  per distinct text.
- r15 2026-09-09: the prompt's vocabulary is met in the fill-set sense;
  introducible targets one per sentence.
- r16 2026-09-09: the drafter answers clauses of word ids; acceptance is
  the Sentence invariant; the fills Assess backend is gone; the parse ask.
- r17 2026-09-09: Wikimedia bitmaps at a bounded width; tried urls on the
  outcome row; a source's quota statement is budget exhaustion; budget
  windows start at a configured time.
- r18 2026-09-10: a Forvo download is a Forvo request; `max_asks: null`.
- r19 2026-09-10: a live search is not re-asked; the verdict is the last
  JSON object; outcome rows never rank; a text's first gloss stands; the
  no-fit answer, its cap and its ageing.
- r20 2026-09-11: §5's no-fit wording aligned with the r19 implementation.
- r21 2026-09-11: the voice constraint derives from the speaker marking;
  the mechanical key carries the subject; rendition/mixed-speakers gone.
- r22 2026-09-11: logs to one line each; the pre-r1 draft note, the
  2026-09-03 Forvo measurement and implementation-status clauses removed;
  §5 Sentence split into labeled parts; §7's buckets stated once as a
  table; §8 merged into spec 1 §4, §10 into spec 2 §4, §11 into spec 1
  §3; §12's "no retries" (contradicted by §6a) dropped. No behavior
  changed.
- r23 2026-09-11: assess-first covers recording needs (a candidate with no
  mechanical verdict under its subject is checked before any source is
  asked); an outcome row's `candidates` are candidates of the need (a
  url-keyed fetch shared by two subjects appends a provide row under the
  first only); Forvo's limit body served at a download url is Quota; the
  drafting prompt asks for at most `sentence_max_clauses` clauses (2) and
  acceptance refuses longer drafts; the run retires an adopted Sentence
  whose recording is exhausted with no passing candidate (F13), leaving
  a retirement row so the text is never re-adopted or re-drafted. Evidence: 2026-09-11 cycles (ผม's male Forvo
  recording unranked under its second subject; 27 download refusals with
  no limit row; 15 three-clause sentences with every recording over the
  5 s cap). User approval 2026-09-11.
- r24 2026-09-11: the drafter is handed at most `sentence_introducible_per_ask`
  (5) sentence-introduced targets per ask, the rest of the 40 being the
  next non-introduced open Targets; every picture need's query is a
  drafted search phrase (one ask per run on the drafter transport, the
  `phrase` provide), the gloss only as a fallback; the drafting prompt
  requires ids exactly as listed and shows a worked example. Evidence:
  2026-09-11 cycles (an ask handing 36 classifier targets of 40 drafted 2
  sentences; scene queries were whole glosses, 336 of 1,405 candidates
  passing; 20 refusals for `ๆ` as an id and a guessed unsuffixed id).
  User approval 2026-09-11.
- r25 2026-09-12: the gloss fallback is gone -- a picture need with no
  query on record (no direction, no fresh suggestion, no drafted phrase)
  is not attempted and counts deferred; the drafter is handed the gloss
  and may answer with it; assess-first's fit question carries no
  phrase. Evidence: 2026-09-11 22:22 resolve (101 fit verdicts on 21
  exhausted words judged against phrases they were not searched for, 1
  pass; all 45 open word-picture needs exhausted under gloss queries
  with an unsearched phrase; the r24 and r24b cycles attempted picture
  needs on the fallback because no phrase ask had answered yet).
  Bringing the record into line (phrases for every need, the pre-phrase
  attempt rows removed) is a one-off outside this spec. User approval
  2026-09-12.
- r26 2026-09-12: a need whose next source is dead for the run takes
  its next live source in the same run (the dead source counts as tried
  for this pass only; deferred only when no live source is left); a
  Cloudflare challenge page is retried once after
  `quotas.<source>.challenge_wait_seconds`; a source's requests are
  spaced `quotas.<source>.min_interval_seconds` apart; Openverse is
  asked as a registered client (`secrets.openverse`, OAuth2 client
  credentials; anonymous 20/minute and 200/day, registered 100/minute
  and 10,000/day per Openverse's throttling documentation) and its
  429 is the Quota state; picture source order pexels, openverse,
  wikimedia. Evidence: 2026-09-12 cycles 2 and 3, one challenge page
  each from Openverse, 495 of 575 needs deferred behind it; 168
  anonymous requests in the 09:00 hour, 28 in the minute before the
  first challenge. User approval 2026-09-12.
- r27 2026-09-12: a drafted sentence fills at most
  `sentence_targets_per_sentence` (3) open Targets, refused at acceptance
  beyond that; met words beyond them are filler; the prompt asks for as
  many natural sentences as it takes, no longer the fewest. A list
  sentence hands each target no context that cues it: it is a word list
  in disguise, the thing sentences exist to avoid. Evidence: adopted
  sentences filled a median of 5 targets, 83 of 318 seven or more ("my
  wife bought a dress, a skirt, a bra, and underwear; I bought a
  shirt" filled six). User ruling and approval 2026-09-12.

Scope: the Provide and Assess ports, every backend's contract (cost, cache
key, authority), the attempt per need kind, the derivations over the record
(current-best, pending, exhausted, queue), Budget, and the run. Storage
shapes are spec 2; domain consumers are spec 1; UI surfaces are spec 5.

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
- **Verdict**: an Assess answer on (subject, artifact, role) under a rubric; carries
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
completion with no drafts raise and append nothing. The judge's answer
is the last JSON object in its completion; prose before it is not a
refusal, since the verdict is positively recognized (r19: with thinking
disabled the model sometimes reasons in text first). The prompts ask
for the JSON object alone. A drafter answer `{"sentences": [],
"reason": "..."}` is recognized (§5).

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
| wikimedia | picture (search hits with url, via generator=search + prop=imageinfo; gsrsearch carries `filetype:bitmap`; imageinfo asks `iiurlwidth` = providers.yaml `image_width`, default 1600, and the hit's url is the scaled `thumburl`, origin the file page) | wikimedia:query | free HTTP | same |
| imgfetch, audiofetch (bytes) | picture-bytes, recording-bytes | url | free | a refusal is typed (§6a): served or wire; never cached against the url |
| forvo | recording; rendition (intersection of members' lookups: same username across members) | forvo:WORD (per member) | 1 request per lookup and per mp3 download (an audiofetch row attributed to forvo counts as one); the day budget is §8's; a `Limit/day reached.` body is Quota (§6a) | re-asked once per attempt when a url has expired (§6a) |
| tts | recording; rendition (one voice across members) | tts:VOICE:sha(TEXT) | cash per character | never re-asked |
| commission | recording; rendition | batch item id | money + weeks | out/in via batch files |
| llm | sentence (per run over open targets), parse (clauses for given texts), phrase, entry | llm:PRODUCER:MODEL:sha(PROMPT) | cash or quota per transport | never re-asked; the prompt text is the contract |
| pair-search | pair | pairs:CONFUSION:DICT_VERSION | free | dictionary bump = new key |
| learner | any (supply) | none; rows are acts | attention | feedback screen only |
| legacy-current | picture (the old deck's current picture, spec 2 §4) | legacy-current:picture:WORD | none; a candidate's provenance, never a Source ask: never tried, budgeted, or listed as asked | never |

## 4. Assess backends and authority

| backend | roles | key | authority |
|---|---|---|---|
| judge (LLM) | picture-for-word (fit, preference), scene-for-sentence, sentence-for-target (naturalness, register), word facts | judge:sha(RUBRIC):SUBJECT:IDENTITY:ROLE (IDENTITY: the artifact sha, the preference set's sha, or empty for a text-only question; a migrated legacy verdict keeps the old shape judge:sha(RUBRIC):ARTIFACT_SHA:ROLE, LegacyVerdictKey, built by migrate alone) | evidence; below learner where learner is qualified |
| mechanical | recording duration/format; media resolvable; provenance rules | parameter-explicit and subject-keyed (one verdict per (subject, artifact), as for the judge), e.g. mech:duration:0.2-5.0:SUBJECT:sha | ground truth for what it checks |
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

**Picture (Word and scene).** Query, in precedence: the latest learner
direction; a judge suggestion newer than the last Source ask on the need
(the search that produced the judged candidate); the drafted phrase —
one ask per run on the drafter transport drafts a short English
image-search phrase for every open picture need, word or scene, that has
none on record and no direction (the `phrase` provide, one row per
subject; the ask is cached by prompt, and an answer phrasing none of the
asked items is not an answer: nothing is appended and the ask is a
source failure, re-asked next run). A need with no query on record is
not attempted this run and counts `deferred` (r25): the gloss is the
drafter's input, never a search (the corpora index English metadata, so
the phrase is English). Source order: pexels, openverse, wikimedia (r26). One attempt: search, imgfetch the first N
(providers.yaml `image_candidates`, default 5) hits no earlier attempt on
the same need and source fetched, fetched meaning ingested or refused by
its server (a wire failure leaves the url untried, §6a) (the outcome row
carries `tried: [url, ...]`; a served refusal of every
hit of a cached answer re-asks the search once within the attempt and
ingests what is new; a search asked live in this attempt is not
re-asked, its hits having just been served (r19)),
judge *fit* on each
(pass/fail, the old rubric texts verbatim), and if more than one passes
judge *preference* once over the passing set; then current-best. A judge
`suggestion` becomes the next attempt's phrase.

Assess-first: when a candidate on record has no fit verdict under the
current rubric, the attempt is the fit questions on those candidates,
asked with no phrase (the search that produced them is not this
attempt's; the rubric's "pass if no phrase is given" applies, r25); no
source is asked and no outcome row is written. A source is asked only
once every candidate is judged. If every such question is excluded
(unpreparable), the source is asked in the same attempt. Scene pictures
follow the same rule. Recording needs follow it too: a candidate on
record with no mechanical verdict under this subject is checked before
any source is asked, at no cost.

**Recording (Word).** Source order: forvo, tts, commission. Voice
constraint (E2, E7; spec 1 §1 r10): derived from the speaker marking. A
word need's marking is its Word's `speaker`; a sentence need's marking is
`Syllabus.marking(sentence)`. Marking female → female; male → male;
empty → male when the recording plays on a productive back (a productive
Target on the word, or a productive Target in the sentence's fill set),
any sex otherwise. Within the constraint the pick spreads over the pool
(TTS pools per sex in providers.yaml; a Forvo item is admitted only when
the sex Forvo states matches). A marking holding both sexes never reaches
sourcing: the Sentence invariant refuses it. No rulebook rule: the
constraint holds at sourcing time; a recording on record that
contradicts it is vetoed once through the learner path (an
`unacceptable-none` rating on that sha, role recording-for-word or
recording-for-sentence) and re-sourced under the constraint. Forvo
attempt: lookup (cached; the §6a re-ask rule on an expired url), download
each item's mp3, mechanical duration/format on each; the item's sex and
country are recorded on the speaker (spec 2); current-best by authority
then provenance prior. TTS attempt: synthesize with a pool voice (the
roster's sex is recorded on the speaker), then mechanical. TTS supplies
sex and timbre only; Forvo and commissions supply age and accent.
`recording/synthetic` warns when current-best is TTS.

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
`rendition/synthetic` (warn) for TTS (one speaker across members holds
by construction of the rendition answer; spec 1 r10 retired the check).

**Sentence (per run over open Targets).** One attempt per run, not per
target. The handed targets are the next open Targets in order, at most
`sentence_targets_per_run` (40), of which at most
`sentence_introducible_per_ask` (§8, default 5) are sentence-introduced
and unmet; the remainder are the next non-introduced open Targets.

*Prompt.* The vocabulary met in the fill-set sense, once, as
`id  thai  (meaning)` lines: the picture-introduced words in
entry-position order up to the furthest handed target, plus every
sentence-introduced word an adopted sentence fills; the handed
sentence-introduced targets not yet met, listed as introducible, at most
one per sentence; the profile register; the existing sentence openings
to avoid; the unadopted texts the judge failed, newest first, at most 20,
each with the verdict's evidence (whitespace-collapsed, 200 characters),
as sentences not to propose. It asks for as many natural sentences as it
takes to cover the handed targets, each filling at most
`sentence_targets_per_sentence` of them (§8, default 3; r27: a sentence
that fills more is a word list in disguise) and free to use any other
listed vocabulary as filler, each of at most `sentence_max_clauses`
clauses (§8, default 2: a longer sentence outruns the 5 s recording
cap), and states the rendering rule (spec 1 §1:
clauses of word ids, ๆ after a repeated word, clauses separated by one
space, standard spelling, numbers as words, no punctuation), requiring
vocabulary ids exactly as listed, suffix included, with one worked
example item showing a suffixed id and a repeated word.

*Answer and acceptance.* `{"sentences": [{"clauses": [["<word id>" |
["<word id>", "ๆ"], ...], ...], "text": "...", "gloss": "..."}]}`.
Acceptance is the Sentence invariant, local and mechanical: an
unregistered id, a rendering that differs from text, more clauses than
the cap, or more open Targets filled than `sentence_targets_per_sentence`
(r27; met words do not count) refuses the draft, logged with the reason,
the provide row keeping it. Each distinct
accepted text is one candidate: a text listed twice is one candidate;
differing clauses reject it; differing glosses keep the first, since the
verdict is keyed by the text and was given on that gloss. A draft
filling no open target is not judged.

*No fit.* `{"sentences": [], "reason": "..."}` is recognized and
cached: one `nothing` outcome row per handed word (the sentence need's
subject; port attempt, backend llm, the handed target ids on the row).
A no-fit served from the cache is re-asked once, so the rows count
refusals, not runs. A word with `sentence_nothing_cap` (§8, default 3)
such rows since its newest learner row is exhausted: its targets are not
handed again, the run counts it exhausted, and the feedback screen asks
the learner a direction question carrying the drafter's reason (supply a
sentence, or retire the target); any learner row on the word reopens it.

*Judging and adoption.* The judge sees each candidate once
(sentence-for-target: naturalness; register; the L1 gloss with the text,
a gloss that misstates the sentence fails the candidate). Adoption
(`Syllabus.add_sentence` with provenance) fills every target `fills()`
says it fills, chosen greedily by targets filled, and creates needs: the
sentence's recording (voice constraint from the marking as for a word,
above; tts allowed for receptive-only, a productive fill wants native,
warn otherwise) and an optional scene picture. A refused draft and a
draft filling nothing are rejected drafts in the record. An adopted
Sentence whose recording need is exhausted with no passing candidate is
retired by the run (F13: nothing is grandfathered): a retirement row is
appended under the sentence (typed key, port attempt, backend run), the
row is deleted and reported by id (spec 2 §6), its drafts stay in the
record, and its Targets reopen. A retired text is never re-adopted and
the drafting prompt lists it among the sentences not to propose. A
learner-supplied or learner-nominated recording, or a learner direction
on the sentence, keeps the sentence (F9); the screen shows it as
exhausted.

**Parse (existing texts).** The same transport, asked once per migration
for the clauses of given texts against the full registered vocabulary
(`id  thai  (meaning)` lines); the answer is
`{"parses": [{"text": "...", "clauses": [...]}]}`, verified by the
Sentence invariant; a text whose parse fails is reported (spec 2 §4).

**Grapheme keyword (Grapheme).** Source: llm proposal (concrete, picturable,
containing the symbol); mechanical `grapheme/keyword-contains-symbol`; the
learner adopts (curated data changes; a machine proposal never adopts
itself).

**Pair (SoundConfusion).** pair-search (dictionary + G2P); mechanical
exact-confusion check; adoption into curated pairs is the learner's act.

## 6. Derivations (folds; never stored)

- **current_best(subject, kind)**: learner choice wins; else the candidate
  ranked highest by the most authoritative backend that has spoken on it
  for the role, under the current rubric (a stale-rubric verdict does not
  rank); of several verdicts by one backend on one artifact, the newest
  ranks; among equals, the provenance prior, then the lower artifact
  sha; never below an artifact the
  learner rated acceptable. A passing mechanical verdict ranks a recording;
  a passing judge fit ranks a picture; preference orders passing pictures.
- **pending(subject, kind)**: a question about one of its candidates
  sits in a submitted, unresolved batch. Nothing else is pending: with an
  inline transport a verdict arrives inside the attempt, an unpreparable
  question is excluded, and a judge that cannot be reached stops the
  run. A pending need gets no new attempt.
- **outcome(subject, kind, source)**: what one attempt of a need at a
  source produced: `candidates` (at least one artifact from it was
  stored; the shas the row names are candidates of the need whether or
  not a provide row under the subject names them — a url-keyed fetch
  shared by two subjects appends a row under the first only — and they
  rank only by their own verdicts: current_best reads assessments only),
  `nothing` (the source answered and nothing
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
  option remains (unasked suggestion, a candidate unjudged under the
  current rubric, unsearched source); (3) acceptable/unrated by rank then
  attempts, counted as exhausted() counts them (a source at the
  transient cap is one attempt, r19); excluded: good, exhausted with no candidate awaiting
  judgement, pending (pending is reported, not queued).
- **confusion_weights()**: unchanged.

## 6a. Failure taxonomy

Every ask and fetch ends in one of four states:

- **Got it.** An artifact stored, or an answer the backend recognized.
  Appended; outcome `candidates`.
- **Failed definitively for the subject.** The source answered and
  what it answered cannot serve this need: an empty lookup, a search with
  no hits, a synthesis the service refuses for this text or voice (a
  4xx other than 429), a downloaded artifact that fails its mechanical
  check. Outcome `nothing`; the need advances to the next source; the
  attempt counts toward the cap.
- **Failed; a retry may succeed.** A wire failure (timeout, connection,
  DNS, a fetcher that cannot run), a 5xx, a served refusal of a
  url (a non-200, a body of the wrong type, undecodable bytes), a batch
  not yet ended. A challenge page in front of a corpus (a 403 that is
  Cloudflare's managed challenge, by its `cf-mitigated: challenge`
  header or its page) is retried once by the backend after
  `quotas.<source>.challenge_wait_seconds`; a second challenge is the
  transport failure (r26). An image corpus's 429 is its own throttle
  statement and is the Quota state above, not this one (r26). Nothing
  is appended for the ask; the attempt's outcome is `transient-failure`. Bounded: once a source has `transient_cap`
  (providers.yaml, default 3) transient outcomes since the escalation
  anchor, it counts as tried: next_source advances past it and exhausted
  counts it as one attempt. Learner input resets the anchor as for every
  other outcome.
- **Ageing.** A `nothing` outcome from a source whose corpus grows
  (Forvo) stops counting as tried once older than
  `quotas.<source>.nothing_ttl_days` (§8; forvo 180, other sources
  never): next_source offers the source again and a fresh lookup
  appends a new row. r19.
- **Quota.** The source itself says its allowance is spent (Forvo: 400
  with body `["Limit/day reached."]` on a lookup, or the same body served
  at one of its download urls, which the fetcher reports and the attempt
  raises typed; never matched downstream). No row is appended, the need
  counts under budgeted, the source is budgeted for the rest of the
  run, and source_failures does not count it.

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

**Keys over mutable state name its version.** An answer computed from
mutable reference data carries a version of that data in its key, as
pair-search carries the dictionary version.

## 7. Budget and the run

Budget per source in its currency: {max_asks?, max_cost?, day_starts?};
forvo 450/day from 22:00 UTC, learner 20/session. Spend is summed from
the record since the most recent `day_starts` instant (HH:MM with a
zone; default local midnight): the source's asks plus the bytes fetches
attributed to it (§4's forvo row).

One source per need per run; one judge batch per run; at most one batch
outstanding: when the previous run's batch is still in progress, the run
adopts what has resolved, attempts nothing, submits nothing, and reports
that batch's subjects as pending and the rest as deferred. Escalation to
the next source happens on the next run, for every transport alike, so
the loop has one shape and a run is cheap and repeatable (F10). Per-day
budgets are measured from the record since the source's day start plus
this run's spend.

```
run(syllabus, budgets):
  resolve the previous run's batch, if any: append its verdicts, release
      its marker (expired or errored results carry no verdict; those
      questions re-ask); pending clears here
  sentence attempt over the open targets (one ask; its candidates enter
      the queue as sentence needs)
  questions = []
  for need in queue(syllabus, budgets):        # pending excluded
      if unjudged(need): questions += assess(need); continue   # §5 assess-first
      source = next_source(need)               # none -> exhausted, skip
      if budget spent: continue                # -> budgeted
      questions += attempt(need, source)       # provide; assess inline or collect
  submit(questions) as one batch; append its marker   # no-op if empty
  RunReport
```

RunReport: `available` is every need gaps() lists, and every need lands
in exactly one of the six buckets below, so
`available == attempted + exhausted + pending + unserved + budgeted + deferred`
always. The remaining fields count events, not needs.

| field | counts |
|---|---|
| attempted | needs whose attempt finished this run (inline verdicts, or no questions raised), plus the open Targets handed to the drafter within the cap |
| exhausted | needs whose next source is None |
| pending | needs with a question in this run's batch or the earlier unresolved one; a need with a question collected this run is never attempted again in it |
| unserved | needs whose kind has no Source and no per-run pass |
| budgeted | needs skipped because their Source's day budget was spent (every open Target within the drafting cap when the drafter's budget is spent) |
| deferred | needs the run never considered: an earlier batch still outstanding, the judge unreachable at resolve, open Targets beyond the per-run drafting cap, needs whose ask failed on the wire or whose every untried source is dead for the run (r26), questions collected but never submitted, a retired sentence's other needs still in this pass's queue, a picture need with no query on record (r25) |
| improved | needs whose current-best artifact sha differs after the attempt (a re-ranking among unchanged artifacts is not improvement) |
| drafted | drafts the sentence attempt produced |
| retired | adopted Sentences the run deleted because their recording need was exhausted with no passing candidate (F13) |
| preferences | preference questions on a picture that already satisfies its need (outside the identity) |
| excluded | questions that could not be prepared (missing or unreadable artifact), per need, skipped |
| unreachable | the judge could not be reached: the run stops at the first such attempt and exits non-zero |
| source_failures[source] | a Source that could not be reached: skipped for the rest of the run; the failing need records a `transient-failure` outcome and counts deferred; a later need whose next source it is takes its next live source in the same run (r26: the dead source counts as tried for this pass only, nothing on the record) and counts deferred only when no live source is left; a drafter transport failure counts under `llm-sentence`, a phrase drafter's under `llm-phrase` |
| spend[source] | the source's asks and cost this run |

Every ask appends; kill-safe anywhere. The run is transport-agnostic.

## 8. Configuration

providers.yaml adds `judge.price_per_mtok: {input, output}`,
`judge.thinking` (disabled | adaptive), `judge.max_tokens` (4096; at least
16000 under `thinking: adaptive`), `drafter.transport` (cli | api),
`image_candidates` (5), `image_width` (1600), `transient_cap` (3) and
`quotas.<source>.{max_asks, max_cost, day_starts}` (forvo 450, `22:00Z`),
layered field by field over the defaults; an explicit `max_asks: null`
lifts a default cap for the day. `quotas.<source>.nothing_ttl_days`
(forvo 180; absent = never), `quotas.<source>.min_interval_seconds`
(seconds between two requests to the source within one process;
openverse 1, others 0) and `quotas.<source>.challenge_wait_seconds`
(the one wait before the single retry of a challenge page, §6a;
openverse 60, others 0: a challenge is a plain transport failure),
`sentence_nothing_cap` (3), `sentence_max_clauses` (2) and
`sentence_introducible_per_ask` (5) and `sentence_targets_per_sentence`
(3). `secrets.openverse` names a
reference to one line `client_id:client_secret` from Openverse's
application registration; when set, the backend fetches an OAuth2
client-credentials access token once per process, through
`search_proxy`, and sends it as a bearer on every search (Openverse's
registered tier: 100 requests a minute, 10,000 a day, against 20 and
200 anonymous, per its throttling documentation); unset, searches are
anonymous. `search_proxy`
is the HTTP forward proxy Openverse searches and its token request go
through (Openverse refuses a Thai egress); no other request uses it. The provenance prior
lives in rulebook.yaml (a judgement, not a route); rulebook.yaml
`rubrics` and `severities` are spec 1 §4's.

## 9. Explicitly out

- No listener implementation; calibration first.
- No interactive judge.
- No stored need status of any kind.
