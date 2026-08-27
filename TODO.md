# TODO

- Spec-deferred (build with the generator): Message Batches sweep mode, STT
  audio↔text verification, .apkg structural validation.
- **Learner-adaptive contrast weights (HVPT loop)** — after the compiler
  exists: compiler stamps `contrast::<id>` tags on minimal-pair cards; a
  stats script reads Anki revlog (AnkiConnect) and emits per-contrast
  lapse-rate weights overriding data/contrasts.yaml, so coverage gaps —
  and generation — track the learner's actual confusions, with raised
  speaker-diversity targets on confused contrasts.
