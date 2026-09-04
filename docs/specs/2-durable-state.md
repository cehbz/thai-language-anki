# Spec 2: Durable state

Revision 1, promoted 2026-09-04 as written on 2026-09-02 against the
principles draft. Re-checked against principles r1 and architecture r1
on 2026-09-04; the revisions that re-check proposed enter as r2 on
approval. Revision process as in docs/architecture.md: proposals on
evidence, explicit approval per revision, numbered log.

Revision log:
- r1 2026-09-04: promoted as written.

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
                               # meaning, classifier
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
sentences(text_sha PK, text, voice, source, origin, licence, acquired)
  -- Sentence artifacts; identity = text_sha; fills derived, never here.
media(sha PK, kind, ext, source, origin, licence, acquired, speaker_id,
      speaker_kind)
  -- provenance for media/objects/*; speaker_* null for pictures.
cache(port, backend, key_sha, subject, question, answer, cost, ts)
      -- PK (key_sha, ts); key_sha indexed. 
  -- store 3. port ∈ provide|assess; backend names the concrete one
  -- (openverse, forvo, llm, judge, learner, ...). key_sha = the backend's
  -- cache key (spec 3 defines each key function; the learner's contains
  -- no rubric). question/answer are JSON. NEVER deleted or updated:
  -- a re-ask appends a new row (newest-wins on read for the learner
  -- backend; exact-key hit for memoized backends). subject indexes the
  -- attempt record ("what was tried for X"), including empty answers.
study(card_key, compile_id, ts, grade, time_ms)
  -- store 4. card_key = the compiled card's content identity (from its
  -- tags: target/pair/grapheme id + card kind). Imported from revlog;
  -- append-only. Anki flags do NOT land here: a flag imports as a
  -- learner assessment row in cache.
```

Learner authority, regression rules, exhausted, current-best, the queue:
all reads over `cache` (spec 3 owns the fold logic); nothing here stores
them. Confusion weights = confusions.yaml seed × study rows; derived.

## 3. Interfaces consumed by the domain core

```
AssessmentReader   .verdict(backend, key) -> Answer | None
                   .assessments_of(subject) -> list[Answer]   # newest last
FrequencyMap       .rank(word_thai) -> int | None
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
   verdicts) → bytes into media/objects/ by sha (files already
   sha-identifiable via manifest+disk), provenance from
   media_manifest.yaml; each candidates.yaml verdict → a judge-backend
   cache row keyed under the OLD rubric id recorded as-is. Current deck
   images additionally get a provisional "machine-chosen" marker (an
   answer row), never a learner rating.
3. **Forvo answers** (work/forvo_lookups.jsonl) → provide/forvo cache
   rows, hit and miss alike.
4. **Proof-gallery notes + ReviewNote harvests + waivers.yaml** → learner
   assessment rows (kind per content: rating, direction, waiver).
5. **The old judge_cache.sqlite is retired**, not migrated: its keys are
   opaque hashes of rubric texts the redesign replaces; identical
   questions will re-hit via candidates.yaml-derived rows, changed
   questions must re-judge anyway.
6. **StudyRecords**: none migrate (the current run predates ReviewNote
   and is scheduling-disposable; its proof-gallery drill results DO
   migrate as study-adjacent learner evidence rows in cache).

Migration report: counts per store, unmigratable rows listed with
reasons, zero silent drops (the SourcingLog lesson).

## 5. Explicitly out

- No memo files: forvo_lookups.jsonl, image_review.yaml,
  image_query_proposals.yaml, candidates/, ipa_adjudication.yaml,
  waivers.yaml all end here; their content lives in `cache` or dies.
- No .last-report.json: reports are derived and identified by state hash.
- No media_manifest.yaml: provenance is the media table.
- No per-filler checkpoint code: sqlite transactions are the checkpoint.

## Amendment 2026-09-03 (carry-over, from spec 3 section 10)

- Section 4 item 2: candidates.yaml's recorded pass/fail per candidate
  becomes the judge fit verdict under the picture-fit rubric, so the studied
  deck's pictures rank as current-best on day one. The old
  `work/judge_cache.sqlite` is NOT migrated: its keys are opaque hashes of
  the full prompt text, unrecoverable.
- Section 4 item 4: learner rows (Anki flags, proof-gallery notes, waivers)
  are keyed by word id via the guid map, never by Anki guid.
- Section 4 item 5 stands as written (the old judge_cache.sqlite is retired,
  not migrated).
- The machine-chosen marker has one consumer: the feedback screen.
- Audio and sentences still regenerate.
