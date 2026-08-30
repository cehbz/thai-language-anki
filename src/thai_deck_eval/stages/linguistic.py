from itertools import combinations
from ..core.findings import Dimension, Severity, Stage
from ..core.registry import rule
from ..data_io import load_g2p_exceptions
from ..lang.ipa import IpaParseError, diff_features, parse_ipa
from ..lang.tone import analyze_syllable

_CONTRAST_FEATURE = {"tone": {"tone"}, "vowel_length": {"length"},
                     "aspiration": {"aspiration"}, "vowel_quality": {"vowel"},
                     "consonant": {"onset"}, "final": {"coda"}}

# Voice-onset-time (VOT) triplets: same place of articulation, unaspirated
# voiced / unaspirated voiceless / aspirated (e.g. บ/ป/พ, ด/ต/ท). Only
# labial and alveolar stops have a native 3-way voicing contrast in Thai;
# velar (ก vs ข/ค) and affricate (จ vs ช) places only ever have 2 members,
# so an "onset" diff between them is never accepted below.
_ASPIRATION_PLACE = {"b": "bilabial", "p": "bilabial",
                     "d": "alveolar", "t": "alveolar"}

def _bare_onset(onset: str) -> str:
    return onset.replace("ʰ", "")

def _onset_same_place(a_onset: str, b_onset: str) -> bool:
    place = _ASPIRATION_PLACE.get(_bare_onset(a_onset))
    return place is not None and place == _ASPIRATION_PLACE.get(_bare_onset(b_onset))

def _minimal_contrast_ok(contrast: str, want: set[str], got: set[str],
                         a, b) -> bool:
    if contrast == "aspiration" and got and got <= {"aspiration", "onset"}:
        if "onset" not in got:
            return True
        return _onset_same_place(a.onset, b.onset)
    return got == want

def _g2p(ctx, word):
    exc = load_g2p_exceptions()
    if word in exc:
        return parse_ipa(exc[word])
    return ctx.g2p.syllables(word)

@rule("lang/pair-not-minimal", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def pair_not_minimal(ctx):
    if ctx.g2p is None:
        return
    for note in ctx.deck.minimal_pairs:
        syls = [_g2p(ctx, m.thai) for m in note.members]
        if any(s is None or len(s) != 1 for s in syls):
            yield pair_not_minimal.rule_def.finding(
                "member unknown to g2p or multi-syllable; cannot verify",
                note_id=note.id, severity=Severity.INFO,
                evidence={"rule_override": "lang/pair-unverifiable"})
            continue
        want = _CONTRAST_FEATURE[note.contrast]
        for (i, a), (j, b) in combinations(enumerate(syls), 2):
            got = diff_features(a[0], b[0])
            if not _minimal_contrast_ok(note.contrast, want, got, a[0], b[0]):
                yield pair_not_minimal.finding(
                    f"members {note.members[i].thai}/{note.members[j].thai} "
                    f"differ in {sorted(got)}, declared contrast {note.contrast}",
                    note_id=note.id,
                    evidence={"diff": sorted(got), "declared": note.contrast})

def _authored_ipa(deck):
    for note in deck.minimal_pairs:
        for m in note.members:
            yield note.id, m.thai, m.ipa
    for note in deck.picture_words:
        if note.ipa:
            yield note.id, note.thai, note.ipa

@rule("lang/ipa-mismatch", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def ipa_mismatch(ctx):
    if ctx.g2p is None:
        return
    for note_id, word, authored in _authored_ipa(ctx.deck):
        try:
            claimed = parse_ipa(authored)
        except IpaParseError as e:
            yield ipa_mismatch.finding(f"unparseable ipa {authored!r}: {e}",
                                       note_id=note_id)
            continue
        got = _g2p(ctx, word)
        if got is None:
            yield ipa_mismatch.rule_def.finding(
                f"{word}: unknown to g2p", note_id=note_id,
                severity=Severity.INFO,
                evidence={"rule_override": "lang/ipa-unverifiable"})
            continue
        if got != claimed:
            second = ctx.g2p_second.syllables(word) if ctx.g2p_second is not None else None
            evidence = {"authored": authored, "g2p": [vars(s) for s in got]}
            if second is not None:
                evidence["g2p_second"] = [vars(s) for s in second]
            # ERROR only when a second engine corroborates the primary
            # engine's disagreement with the author; otherwise there's no
            # corroborating second opinion (engine unavailable, word unknown
            # to it, or it disagrees with the primary too) so we can't be
            # confident the author is wrong -> WARN.
            severity = Severity.ERROR if second is not None and second == got else Severity.WARN
            yield ipa_mismatch.rule_def.finding(
                f"{word}: authored IPA disagrees with g2p",
                note_id=note_id, severity=severity, evidence=evidence)

@rule("lang/tone-mismatch", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def tone_mismatch(ctx):
    for note_id, word, authored in _authored_ipa(ctx.deck):
        try:
            claimed = parse_ipa(authored)
        except IpaParseError:
            continue  # lang/ipa-mismatch reports it
        if len(claimed) != 1:
            continue
        analysis = analyze_syllable(word)
        if analysis is None:
            continue
        if analysis.tone != claimed[0].tone:
            yield tone_mismatch.finding(
                f"{word}: tone rules give {analysis.tone}, authored {claimed[0].tone}",
                note_id=note_id,
                evidence={"engine": str(analysis.tone), "authored": str(claimed[0].tone)})

@rule("lang/dead-syllable-tone-contrast", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.ERROR)
def dead_syllable_tone(ctx):
    allowed = {"low", "high", "falling"}
    for note in ctx.deck.minimal_pairs:
        if note.contrast != "tone":
            continue
        for m in note.members:
            a = analyze_syllable(m.thai)
            if a is not None and not a.live and str(a.tone) not in allowed:
                yield dead_syllable_tone.finding(
                    f"{m.thai}: dead syllable cannot carry {a.tone}",
                    note_id=note.id)

@rule("lang/target-not-token", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
def target_not_token(ctx):
    if ctx.tokenizer is None:
        return
    for note in ctx.deck.sentences:
        toks = ctx.tokenizer.tokens(note.thai)
        if note.target in toks:
            continue
        # Boundary-aligned compound membership: the target is still a real
        # word if it sits at the start or end of a dictionary-compound
        # token (e.g. target กิน inside token กินข้าว). Only a target
        # embedded strictly mid-token (neither prefix nor suffix) warns.
        if any(tok.startswith(note.target) or tok.endswith(note.target)
              for tok in toks):
            continue
        yield target_not_token.finding(
            f"target {note.target!r} is not a token of the sentence",
            note_id=note.id, evidence={"tokens": toks})

@rule("lang/frequency-rank-wrong", Stage.LINGUISTIC, Dimension.LANGUAGE, Severity.WARN)
def frequency_rank_wrong(ctx):
    if ctx.freq is None:
        return
    for note in ctx.deck.picture_words:
        ref = ctx.freq.rank(note.thai)
        if ref is None:
            yield frequency_rank_wrong.rule_def.finding(
                f"{note.thai}: not in reference frequency list",
                note_id=note.id, severity=Severity.INFO,
                evidence={"rule_override": "lang/frequency-unknown"})
        elif abs(note.frequency_rank - ref) > max(50, 0.2 * ref):
            yield frequency_rank_wrong.finding(
                f"{note.thai}: declared rank {note.frequency_rank}, reference {ref}",
                note_id=note.id, evidence={"reference": ref})


# Chat spellings of the polite particles, in utterance-final position only
# (คับ elsewhere is the ordinary word for "tight").
NONSTANDARD_PARTICLES = {"คับ": "ครับ", "คร้าบ": "ครับ", "ค๊าบ": "ครับ",
                         "ค้าบ": "ครับ", "ค่าา": "ค่ะ"}


@rule("lang/nonstandard-particle", Stage.LINGUISTIC, Dimension.LANGUAGE,
      Severity.WARN)
def nonstandard_particle(ctx):
    """Chat spellings of ครับ/ค่ะ at the end of a sentence.

    A fixed-form defect the LLM judge reported on only a third of the
    sentences that had it; a rule catches all of them for nothing.
    """
    for note in ctx.deck.sentences:
        text = (note.thai or "").rstrip()
        for chat, standard in NONSTANDARD_PARTICLES.items():
            if text.endswith(chat):
                yield nonstandard_particle.finding(
                    f"sentence ends with the chat spelling {chat!r}; "
                    f"the standard written form is {standard!r}",
                    note_id=note.id,
                    evidence={"found": chat, "expected": standard})
                break


# Particles that mark the speaker as female. Understanding them matters;
# rehearsing them does not, when the learner's production register is male.
FEMALE_PARTICLES = ("ค่ะ", "คะ", "ค่า")


@rule("lang/particle-register", Stage.LINGUISTIC, Dimension.LANGUAGE,
      Severity.WARN)
def particle_register(ctx):
    """A production sentence ending in a woman's politeness particle."""
    for note in ctx.deck.sentences:
        if getattr(note, "usage", "production") != "production":
            continue
        text = (note.thai or "").rstrip()
        for particle in FEMALE_PARTICLES:
            if text.endswith(particle):
                yield particle_register.finding(
                    f"production sentence ends with {particle!r}, a female "
                    f"speaker's particle; mark it usage: comprehension or "
                    f"rewrite it with ครับ",
                    note_id=note.id, evidence={"particle": particle})
                break
