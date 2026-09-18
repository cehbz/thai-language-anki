# References

Source documents the specs and the rulebook cite as evidence. Kept here so a
claim in docs/specs can be checked against the thing it rests on, rather than
against a summary of it. Only redistributable sources belong here -- check the
licence before adding one, and record it below.

## liu-2022-tone-atlas.pdf

Liu, L., Lai, R., Singh, L., Kalashnikova, M., Wong, P.C.M., Kasisopa, B.,
Chen, A., Onsuwan, C. & Burnham, D. (2022). *The tone atlas of perceptual
discriminability and perceptual distance: Four tone languages and five language
groups.* Brain and Language 229, 105106. DOI 10.1016/j.bandl.2022.105106.

Licence: CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/) -- open access,
redistribution permitted with attribution.

Cited by: the `SoundConfusion.weight` values for the ten Thai tone contrasts
(spec 1). Thai has 5 tones and 10 contrasts, and the study tested Thai tone
discrimination by Australian English non-tone listeners (N = 24), which is the
learner this deck is for.

What it is used for: section 3.1 gives mean d' by contrast type across listener
groups -- Df-Dr 3.62, S-Df 3.11, S-Dr 3.07, S-S 2.80, Dr-Dr 2.37. Under the
paper's own classification (static = level or mild slope, "e.g., Thai 45"),
Thai's tones are S = {mid 33, low 21, high 45}, Df = {falling 41},
Dr = {rising 214}, giving 3 S-S, 3 S-Df, 3 S-Dr, 1 Df-Dr and no Dr-Dr -- which
matches the paper's own remark that Thai is the only system tested with no Dr-Dr
contrast.

Two cautions when citing it:
- Those d' values pool all five listener groups. The English-only values are in
  Fig. 2's bars and are not given numerically in the text.
- The paper is internally inconsistent about the ranking: section 3.1's emmeans
  put S-Df and S-Dr above S-S, while section 4.3's prose ranks S-S second
  easiest. The numbers are used here, not the prose.

## Burnham, Kirkwood, Luksaneeyanawin & Pansottee (1992) — NOT included here

Burnham, D., Kirkwood, K., Luksaneeyanawin, S. & Pansottee, S. (1992).
*Perception of Central Thai tones and segments by Thai and Australian adults.*
In Pan-Asiatic Linguistics: Proceedings of the Third International Symposium on
Language and Linguistics, 546-560. No DOI.

**The PDF is deliberately not in this repository.** A scan is reachable at
http://sealang.net/sala/pal/htm/KIRKWOODKathryn.htm but the work carries no
stated licence, so default copyright applies and this repository is public.
Personal use is fine; redistribution is not ours to grant. Fetch it from the
link above if you need the original.

Its findings are recorded here because measurements are facts, not expression.

This is the primary evidence for the ten Thai tone contrast weights, and it
supersedes the contrast-type reasoning derived from the tone atlas: it measured
each of the ten pairs directly, with Australian English listeners, rather than
pooling pairs into static/dynamic classes. Where the two disagree at the level
of an individual pair, this one governs.

Table 6(b), percent correct, English speakers, AX discrimination (chance = 50):

| pair          |  %  | pair          |  %  |
|---------------|-----|---------------|-----|
| low-rising    |  54 | high-rising   |  88 |
| high-falling  |  75 | mid-falling   |  90 |
| mid-low       |  79 | falling-rising|  91 |
| mid-rising    |  79 | high-low      |  94 |
| mid-high      |  81 | low-falling   |  94 |

The paper's own explanation of the ordering is more useful than the table: the
difficulty of a pair tracks how similar the two tones' *nominal starting pitches*
are, not whether they are static or dynamic. Low and rising both begin low, and
English listeners are at chance. Low begins low and falling begins high, and they
are at 94%. Use that mechanism when weighting a contrast the table does not list.

Two cautions:
- The task is AX discrimination ("same or different?"), not identification
  ("which word was it?"), which is what a minimal-pair card drills. The same
  caution applies to the tone atlas.
- Interstimulus interval matters and not in the intuitive direction: English
  accuracy on the ten different pairs FELL from 84.1% at 500 ms to 80.0% at
  1500 ms, while Thai speakers improved. Relevant if a pair card ever spaces its
  two members.
