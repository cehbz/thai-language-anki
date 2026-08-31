# TODO

- Spec-deferred: Message Batches sweep mode, STT audio↔text verification,
  .apkg structural validation.
- **Learner-adaptive contrast weights (HVPT loop)** — unblocked now that the
  compiler stamps `contrast::<id>` tags on minimal-pair cards; a
  stats script reads Anki revlog (AnkiConnect) and emits per-contrast
  lapse-rate weights overriding data/contrasts.yaml, so coverage gaps —
  and generation — track the learner's actual confusions, with raised
  speaker-diversity targets on confused contrasts.

## Architecture simplification pass

The codebase grew by accretion: every problem this arc hit was answered by
adding a mechanism beside the existing ones. The spec suites
(`tests/spec/`) are implementation-independent, so they are the safety net
for restructuring. Evidence gathered while writing them:

- **Five memoization mechanisms, five formats, five invalidation rules.**
  Forvo lookups (`work/forvo_lookups.jsonl`, append-only, never expires),
  image candidates (`work/candidates/*/candidates.yaml`, invalidated by
  corpus set), exhausted searches (`work/image_review.yaml`, invalidated by
  queries + rubric + corpora), judge verdicts (sqlite, keyed by rules +
  prompt + model + image sha), LLM completions (sqlite, keyed by producer +
  prompt version + model + prompt). They answer the same question — has
  this exact work already been done — and each learned invalidation
  separately and late. One concept with one fingerprint rule would remove
  three of the five bugs this arc produced.
- **`fill_words` and `fill_spelling` ignore the `gaps` they are handed.**
  The loop gates on gaps (`_fillable`) but two of four producers derive
  everything from the word list, so "gap-driven" describes the loop and not
  the producers. Either they should consume gaps or they should not be in
  the gap loop.
- **`media/images.py` is 586 lines** doing search, ordering, download,
  verification, memoization, proposals and the review queue. The largest
  module and the one every bug landed in.
- **Two config surfaces that reference each other.** `gen.yaml` carries a
  `rulebook:` pointer so the generator can borrow the evaluator's judge, and
  secrets live in both. The judge is configured in one file and consumed in
  two.
- **Two notions of known vocabulary**: `known_vocab` (whole deck) and
  `vocabulary_by_position` (progressive). The first is now only used as a
  fallback and is what made `meth/new-elements` vacuous.
- **`ImageNeed` duplicates the note.** It carries category, gloss and
  image_query, all of which now live on `PictureWordNote`; `pending_images`
  needs three lookup maps passed in to rebuild what the deck already knows.
- **Each media filler re-implements the same pattern**: per-item fault
  tolerance, periodic checkpoint, provenance record, blocked list. Forvo,
  TTS, images and thai1000 each have their own copy, and the checkpoint was
  added to two of them only after a killed run lost work.
- **Dead test seam**: `THAI_DECK_GEN_FAKE` and the fake port classes in
  `cli.py` predate the fake-world harness in `tests/spec/world.py` and are
  no longer exercised by anything but themselves.
- **CLI inconsistency**: `init` and `generate` take a positional `dir`;
  every other command takes `--deck`.

## Thai register questions needing expert input (not code)

These were decided provisionally so generation could continue; each needs
deeper analysis by fable, and likely research plus a local speaker.

- **Politeness particle spelling.** The generator wrote คับ at the end of
  152 sentences; those are normalized to ครับ and `lang/nonstandard-particle`
  now enforces it. Provisional reasoning: the deck teaches reading, and TTS
  pronounces ครับ with the natural reduction anyway. Open: whether the káp
  reduction heard around Chiang Mai is general colloquial Thai or a Northern
  feature, and whether a deck aimed at colloquial daily speech should ever
  show the chat spelling. Investigate current Northern Thai pedagogy: what do
  Chiang Mai language schools put on the page?
- **First-person register.** ~204 sentences use ฉัน, which the word list
  glosses "female speaker, or casual general"; production cards are being
  regenerated with ผม and a `usage` field splits production from
  comprehension. Open: whether ฉัน is genuinely register-neutral in casual
  Northern speech, and what proportion of comprehension-only material a deck
  should carry.
- **Female particles.** 33 sentences end in ค่ะ, kept as comprehension
  examples. Open: whether comprehension-only cards should differ in card
  template (recognition only, never a production prompt), and whether other
  gendered forms need the same treatment.
- **Kam Mueang vs Central Thai.** The deck teaches Central Thai to a learner
  living in a Kam Mueang-speaking region. Nothing in the pipeline knows this.
  Worth deciding deliberately rather than by default.

- **new-element tolerance mismatch.** `check_sentence` in the generator
  accepts one unknown non-target token; `meth/new-elements` accepts none.
  26 of 480 sentences sit in that gap. FF's one-new-element principle says
  the rule is right, but tightening the generator raises the block rate --
  decide deliberately rather than leaving the two out of step.

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
