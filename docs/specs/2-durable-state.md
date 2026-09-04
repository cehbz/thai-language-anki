# Spec 2: Durable state

Revision 3, proposed 2026-09-04 against principles r2 and architecture
r2 (r1 promoted 2026-09-04 as written on 2026-09-02 against the
principles draft). Re-checked against principles r1 and architecture r1
on 2026-09-04; the revisions that re-check proposed enter as r2 on
approval. Revision process as in docs/architecture.md: proposals on
evidence, explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.
- r2 2026-09-04: category on words.yaml rows; speakers table (E7);
  migration joins pictures by (thai, category), carries old verdicts
  under a legacy rubric id that never ranks, normalizes images at
  ingest, and drops the machine-chosen marker; the 09-03 amendment
  folded in.
- r3 2026-09-04: sentences carry gloss; study keyed (card_key, ts) with
  the anchor::kind convention; keys built by spec 3's functions; ranks
  resolved by the loader; no layout version. Evidence: implementation
  review 2026-09-04.

Scope: what persists, where, in what shape; the interfaces the domain core
consumes; migration of the carry-over assets. Port mechanics are spec 3;
this spec only fixes what their caches look like at rest.

Ground rules (from the architecture): four durable stores and nothing
else; derived state is never written; an append is a checkpoint; caches
are never evicted; human-curated data is human-readable and
hand-editable, machine state is not hand-edited.

## 1. Layout

```
<deck>/                        # the Syllabus's home directory
  curated/                     # store 1 — human-owned YAML
    words.yaml                 # Word facts: id, thai, pron (authored IPA),
                               # meaning, classifier; category (the row
                               # is where a human edits it; the loader
                               # builds the Category collections, so
                               # single membership holds by construction)
    targets.yaml               # id, word, skill, introduction
    graphemes.yaml             # symbol, kind, sound, class, keyword
    confusions.yaml            # id, dimension, sounds + profile seed weight
    pairs.yaml                 # adopted MinimalPairs (machine-proposed,
                               # human-kept; small)
    profile.yaml               # register, emphasis
    rulebook.yaml              # rule config: severities, thresholds,
                               # judged-rule rubric text
  media/                       # store 2 — artifact bytes
    objects/<sha>.<ext>        # content-addressed; sha = sha256 of bytes
  syllabus.db                  # stores 2b/3/4 — sqlite, WAL
  work/                        # logs, scratch; disposable
```

YAML writes are temp-file + os.replace (atomic). sqlite in WAL mode; every
write is one transaction (the append-is-checkpoint rule).

## 2. syllabus.db tables

```
sentences(text_sha PK, text, gloss, voice, source, origin, licence,
          acquired)
  -- Sentence artifacts; identity = text_sha, the one sentence id
  -- everywhere (rules, compile, learner rows); fills derived, never here.
media(sha PK, kind, ext, source, origin, licence, acquired, speaker_id)
  -- provenance for media/objects/*; speaker_id null for pictures.
speakers(id PK, kind, sex, age_band, region)
  -- E7. Written at recording ingest from what the source exposes (Forvo
  -- per-item sex and country; the TTS roster's sex; the commission
  -- brief); attributes unknown otherwise. Never hand-edited; a learner
  -- correction is a learner row in cache.
cache(port, backend, key_sha, subject, question, answer, cost, ts)
      -- PK (key_sha, ts); key_sha indexed. 
  -- store 3. port ∈ provide|assess; backend names the concrete one
  -- (openverse, forvo, llm, judge, learner, ...). key_sha = the backend's
  -- cache key (spec 3 defines each key function; the learner's contains
  -- no rubric). question/answer are JSON. NEVER deleted or updated:
  -- a re-ask appends a new row (newest-wins on read for the learner
  -- backend; exact-key hit for memoized backends). subject indexes the
  -- attempt record ("what was tried for X"), including empty answers.
study(card_key, compile_id, ts, grade, time_ms)  -- PK (card_key, ts)
  -- store 4. card_key = "<anchor>::<card kind>", anchor = the note's
  -- guid source (word id, pair MemberKey, grapheme symbol, target id +
  -- text_sha); the Target of a word card is derived from the kind.
  -- Imported from revlog; append-only, insert-or-ignore. Anki flags do
  -- NOT land here: a flag imports as a learner assessment row in cache.
```

Learner authority, regression rules, exhausted, current-best, the queue:
all reads over `cache` (spec 3 owns the fold logic); nothing here stores
them. Confusion weights = confusions.yaml seed × study rows; derived.

## 3. Interfaces consumed by the domain core

```
AssessmentReader   .verdict(backend, key) -> Answer | None
                   .assessments_of(subject) -> list[Answer]   # newest last
                   # keys are built by spec 3's one key function per
                   # backend; the store appends and reads what it is handed
FrequencyMap       .rank(word_thai) -> int | None
                   # consumed by the loader, which hands the aggregate a
                   # word -> rank mapping
RecordWriter       .append(port, backend, key, subject, question, answer,
                           cost)
StudyReader        .records(card_key | confusion) -> list[StudyRecord]
```

All implemented over syllabus.db + curated files; faked in domain tests.

## 4. Migration (one script, run once, idempotent)

Carry-over per the handoff's table; everything else regenerates.

1. **Word list** (data/word_list_th.yaml, 766 rows with ids) →
   curated/words.yaml (id, thai, pron from note ipa where present,
   meaning=gloss, classifier) + curated/targets.yaml (one receptive
   target per row, introduction=picture_card; productive targets NOT
   auto-created — the selection rule is the user's open decision).
   Dropped fields (picturable, emphasis, image_query*, split_of,
   part_of_speech): image_query with source=human migrates as a learner
   direction row in cache; the rest are dropped.
2. **Judged images** (~650 in the deck + work/candidates/*/candidates.yaml
   verdicts) → normalized at ingest like any picture (spec 4 §3) into
   media/objects/ by sha of the normalized bytes, provenance from
   media_manifest.yaml. Old picture notes join words by (thai, category)
   (measured 2026-09-04: unambiguous for 38 of 39 homograph forms); an
   ambiguous form is reported, never guessed. Each candidates.yaml
   verdict → a judge-backend cache row under a legacy rubric id
   ("legacy-picture-rules"), the verdict and failed rule ids as-is: the
   old record does not say which rubric version judged, so under F9 the
   row is evidence of what was seen and rejected, and it never ranks
   (spec 3 §6). Every current picture is judged under the current rubric
   by the first run's assess-first step (one batch, ~654 questions). No
   marker of the old deck's choice is written.
3. **Forvo answers** (work/forvo_lookups.jsonl) → provide/forvo cache
   rows, hit and miss alike.
4. **Proof-gallery notes + ReviewNote harvests + waivers.yaml** → learner
   assessment rows (kind per content: rating, direction, waiver), keyed
   by word id via the guid map, never by Anki guid.
5. **The old judge_cache.sqlite is retired**, not migrated: its keys are
   opaque hashes of rubric texts the redesign replaces; identical
   questions will re-hit via candidates.yaml-derived rows, changed
   questions must re-judge anyway.
6. **StudyRecords**: none migrate (the current run predates ReviewNote
   and is scheduling-disposable; its proof-gallery drill results DO
   migrate as study-adjacent learner evidence rows in cache).

Migration report: counts per store, unmigratable rows listed with
reasons, rows passed over counted (audio, duplicates on re-run), zero
silent drops (the SourcingLog lesson).

No layout version is recorded: the loader validates the shape it expects
and refuses anything else, naming the file and field.

## 5. Explicitly out

- No memo files: forvo_lookups.jsonl, image_review.yaml,
  image_query_proposals.yaml, candidates/, ipa_adjudication.yaml,
  waivers.yaml all end here; their content lives in `cache` or dies.
- No .last-report.json: reports are derived and identified by state hash.
- No media_manifest.yaml: provenance is the media table.
- No per-filler checkpoint code: sqlite transactions are the checkpoint.
