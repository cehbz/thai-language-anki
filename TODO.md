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

- Curate `data/pair_seeds.yaml` for `tone:high-falling`, the one contrast
  the real lexicon can't pair on its own — hand-pick a verified minimal
  pair (checked against real pythainlp G2P output, the way
  tests/gen/test_e2e_integration.py's two seed pairs were).
- Adjudicate the 97 words in `~/decks/thai-ff/work/ipa_adjudication.yaml`
  into `data/g2p_exceptions.yaml`; never author unverified IPA.
- Decide what to do about the 146 sentences blocked as "2 unknown
  non-target tokens" (ถูก, สัปดาห์, ร้อน, ...) — a larger known-vocabulary
  base, a relaxed new-elements budget, or hand-authoring. They are genuine
  rejections, not limit halts, so re-running generate will not add them.
- Migrate FORVO_API_KEY / GOOGLE_TTS_API_KEY / OPENAI_API_KEY out of the
  environment into gen.yaml, alongside imgfetch and search_proxy.
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
