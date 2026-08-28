# TODO

- Spec-deferred: Message Batches sweep mode, STT audio↔text verification,
  .apkg structural validation.
- **Learner-adaptive contrast weights (HVPT loop)** — unblocked now that the
  compiler stamps `contrast::<id>` tags on minimal-pair cards; a
  stats script reads Anki revlog (AnkiConnect) and emits per-contrast
  lapse-rate weights overriding data/contrasts.yaml, so coverage gaps —
  and generation — track the learner's actual confusions, with raised
  speaker-diversity targets on confused contrasts.

## Execution-time follow-ups (generator is built; these are real runs, not code)

- Run `thai-deck-gen wordlist` against the real LLM backend to draft
  `data/word_list_th.yaml` (625-word Fluent Forever list), then a human
  pass to review/correct glosses, categories, classifiers, and
  register/split_of calls before it feeds `fill_words`.
- Curate `data/pair_seeds.yaml` for the minimal-pair contrasts the real
  lexicon can't supply on its own — run `fill_pairs` against the curated
  word list once it exists, take its `blocked` contrast-id list, and hand-pick
  a verified real minimal pair (checked against real pythainlp G2P output,
  the way tests/gen/test_e2e_integration.py's two seed pairs were) for each.
- First live Forvo batch (`thai-deck-gen audio fetch-forvo`) against the
  real API key once word/pair/spelling content exists, to validate
  rate-limiting, speaker-diversity behavior, and licensing metadata end to
  end against the real service (not the fakes the e2e test uses).
- First commission batch (`thai-deck-gen audio import-commission`) once a
  batch of native-speaker recordings comes back, to validate the
  recordings↔batch-manifest↔deck plumbing on real audio files.
- First TTS run (`thai-deck-gen audio tts`) against the real Google TTS key
  for sentence audio, to confirm cost/quota and output quality are
  acceptable before running it at full-deck scale.
- Judge pass (drop `--no-judge`) on the first real generated deck, once
  word list + pairs + media are real, to see actual judge findings/cost
  against real content rather than the `--no-judge` mechanical/linguistic/
  method-only path the e2e test exercises.
- Wire `duration_ok` into `fetch_forvo` before the first live Forvo run, so
  clips outside the acceptable duration range are rejected/blocked rather
  than silently accepted.
- Decide whether Forvo multi-speaker variant files (`_s1`, `_s2`, ...)
  should be schema-referenced from the note or dropped -- currently they're
  written to media and recorded in the manifest but nothing in the note
  model points at them.
