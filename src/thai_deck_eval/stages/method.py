from collections import Counter
from ..core.findings import Dimension, Metric, Severity, Stage
from ..core.registry import rule
from ..data_io import (load_categories, load_contrasts, load_function_words,
                       load_spelling_targets)
from ..stages.linguistic import _g2p

_PLACE = {"p": "labial", "t": "alveolar", "k": "velar", "tɕ": "affricate"}

def contrast_id_for(note, ctx) -> str | None:
    """Resolve a minimal-pair note to its specific contrast inventory id
    (e.g. "tone:low-rising", "aspiration:velar") via g2p.

    Public API: the authoritative note->contrast attribution. The .apkg
    compiler stamps `contrast::<id>` card tags from this (via the
    coverage metric's `by_note` detail or by calling it directly), which
    the learner-adaptive weighting loop later joins against Anki revlog
    data. Returns None when the note is unverifiable (g2p-unknown or
    multi-syllable member) or maps to no inventory entry.
    """
    if ctx.g2p is None:
        return None
    syls = [_g2p(ctx, m.thai) for m in note.members]
    if any(s is None or len(s) != 1 for s in syls):
        return None
    return _contrast_id(note, syls)

def _contrast_id(note, syls) -> str | None:
    a, b = syls[0][0], syls[1][0]
    if note.contrast == "tone":
        return "tone:" + "-".join(sorted([str(a.tone), str(b.tone)],
                                         key=["mid","low","falling","high","rising"].index))
    if note.contrast == "aspiration":
        place = _PLACE.get(a.onset.replace("ʰ", ""))
        return f"aspiration:{place}" if place else None
    if note.contrast == "vowel_length":
        return "vowel_length"
    if note.contrast == "consonant":
        if "ŋ" in (a.onset, b.onset):
            return "consonant:ng-onset"
        if {a.onset, b.onset} == {"r", "l"}:
            return "consonant:r-l"
        return None
    if note.contrast == "vowel_quality":
        pair = {a.vowel, b.vowel}
        for cid, vs in [("vowel_quality:e-ɛ", {"e", "ɛ"}),
                        ("vowel_quality:o-ɔ", {"o", "ɔ"})]:
            if pair == vs:
                return cid
        if "ɯ" in pair:
            return "vowel_quality:ɯ"
        if "ɤ" in pair:
            return "vowel_quality:ɤ"
        return None
    if note.contrast == "final":
        return "final:unreleased"
    return None

@rule("meth/pair-coverage", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def pair_coverage(ctx):
    if ctx.g2p is None:
        return
    entries = load_contrasts()
    by_note: dict[str, str] = {}
    for note in ctx.deck.minimal_pairs:
        cid = contrast_id_for(note, ctx)
        if cid:
            by_note[note.id] = cid
    covered = set(by_note.values())
    total = sum(e.weight for e in entries)
    got = sum(e.weight for e in entries if e.id in covered)
    yield Metric(name="coverage/minimal_pairs", value=got / total,
                 detail={"covered": sorted(covered),
                         "missing": sorted(e.id for e in entries
                                           if e.id not in covered),
                         "by_note": by_note})

@rule("meth/spelling-coverage", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def spelling_coverage(ctx):
    targets = load_spelling_targets()
    all_syms = [s for group in targets.values() for s in group]
    covered = {n.pattern for n in ctx.deck.spelling_sound}
    got = sum(1 for s in all_syms if s in covered or s.strip("-") in covered)
    yield Metric(name="coverage/spelling", value=got / len(all_syms),
                 detail={"total": len(all_syms), "covered": got})

@rule("meth/frequency-coverage", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def frequency_coverage(ctx):
    if ctx.freq is None:
        return
    n = sum(1 for w in ctx.deck.picture_words
            if (r := ctx.freq.rank(w.thai)) is not None and r <= 625)
    yield Metric(name="coverage/frequency", value=n / 625)

@rule("meth/classifier-missing", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def classifier_missing(ctx):
    for note in ctx.deck.picture_words:
        if note.part_of_speech == "noun" and note.classifier is None:
            yield classifier_missing.finding("noun without classifier",
                                             note_id=note.id)

@rule("meth/tts-audio", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def tts_audio(ctx):
    for note in ctx.deck.minimal_pairs:
        for m in note.members:
            if m.audio.source == "tts":
                yield tts_audio.rule_def.finding(
                    f"TTS audio on minimal pair member {m.thai}",
                    note_id=note.id, severity=Severity.ERROR)
    others = ([(n, n.audio) for n in ctx.deck.spelling_sound]
              + [(n, n.audio) for n in ctx.deck.picture_words]
              + [(n, n.audio) for n in ctx.deck.sentences])
    for note, audio in others:
        if audio.source == "tts":
            yield tts_audio.finding("TTS audio on tone-bearing card",
                                    note_id=note.id)

@rule("meth/spelling-taper", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def spelling_taper(ctx):
    taper = ctx.cfg("taper_rank", 300)
    for note in ctx.deck.picture_words:
        if note.test_spelling and note.frequency_rank > taper:
            yield spelling_taper.finding(
                f"spelling card beyond taper rank {taper}", note_id=note.id)

@rule("meth/sentence-fanout", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def sentence_fanout(ctx):
    counts = Counter(n.target for n in ctx.deck.sentences)
    for target, c in counts.items():
        if c > 4:
            yield sentence_fanout.finding(
                f"{c} sentence notes for target {target!r} (max 4)")

@rule("meth/premature-sentences", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def premature_sentences(ctx):
    plan = ctx.deck.meta.stage_plan.phases
    if "words" not in plan or "sentences" not in plan:
        return
    base = ctx.cfg("sentence_base", 300)
    if ctx.deck.sentences and len(ctx.deck.picture_words) < base:
        yield premature_sentences.finding(
            f"{len(ctx.deck.sentences)} sentences atop only "
            f"{len(ctx.deck.picture_words)} picture words (base {base})")

def _boundary_known(tok: str, known: set[str]) -> bool:
    return tok in known or any(
        tok.startswith(k) or tok.endswith(k) for k in known)

@rule("meth/new-elements", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def new_elements(ctx):
    if ctx.tokenizer is None:
        return
    known = {w.thai for w in ctx.deck.picture_words} | load_function_words()
    for note in ctx.deck.sentences:
        toks = ctx.tokenizer.tokens(note.thai)
        unknown = [t for t in toks
                  if t != note.target and not _boundary_known(t, known)]
        if unknown:
            yield new_elements.finding(
                f"sentence introduces {len(unknown)} unknown non-target tokens",
                note_id=note.id, evidence={"unknown": unknown})
        known.add(note.target)

@rule("meth/category-coverage", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def category_coverage(ctx):
    categories = load_categories()
    if not categories:
        return
    valid = set(categories)
    covered = {w.category for w in ctx.deck.picture_words if w.category in valid}
    yield Metric(name="coverage/categories", value=len(covered) / len(categories),
                 detail={"covered": sorted(covered),
                         "missing": sorted(valid - covered)})

@rule("meth/unknown-category", Stage.METHOD, Dimension.METHOD, Severity.WARN)
def unknown_category(ctx):
    categories = load_categories()
    if not categories:
        return
    valid = set(categories)
    for note in ctx.deck.picture_words:
        if note.category not in valid:
            yield unknown_category.finding(
                f"{note.category!r} is not a recognized category",
                note_id=note.id)

@rule("meth/no-personal-connection", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def no_personal_connection(ctx):
    for note in ctx.deck.picture_words:
        if note.personal_connection is None:
            yield no_personal_connection.finding(
                "personal-connection slot empty (fill by hand)", note_id=note.id)

@rule("meth/speaker-diversity", Stage.METHOD, Dimension.METHOD, Severity.INFO)
def speaker_diversity(ctx):
    speakers = {m.audio.speaker for n in ctx.deck.minimal_pairs for m in n.members}
    if not ctx.deck.minimal_pairs:
        return
    target = ctx.cfg("target_speakers", 3)
    yield Metric(name="speakers/minimal_pairs",
                 value=min(1.0, len(speakers) / target),
                 detail={"distinct": len(speakers)})
