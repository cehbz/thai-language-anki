import gzip
import random
import urllib.request
from pathlib import Path

from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import Audio, SentenceNote
from thai_deck_gen.llm import LlmError
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps

PROMPT_VERSION = "sn2"

# OPUS OpenSubtitles v2018 Thai monolingual corpus, hosted on CSC's object
# storage (the canonical download path for OPUS mono corpora). Used as a
# source of naturalistic colloquial-register exemplar sentences for the
# sentence producer's LLM prompts.
EXEMPLAR_URL = ("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/"
                "th.txt.gz")


# Chat spellings of the polite particles. Only utterance-final position is
# rewritten: คับ mid-sentence is the ordinary word for "tight".
PARTICLE_FIXES = {"คับ": "ครับ", "คร้าบ": "ครับ", "ค๊าบ": "ครับ", "ค้าบ": "ครับ"}


def normalize_particles(thai: str) -> str:
    """Standard particle spelling, whatever register the model wrote in.

    The deck teaches reading as well as listening, so the written form is
    the standard one even where speech reduces it.
    """
    text = thai.rstrip()
    for chat, standard in PARTICLE_FIXES.items():
        if text.endswith(chat):
            return text[: -len(chat)] + standard
    return thai


def known_vocab(deck: Deck) -> set[str]:
    """Picture-word thais introduced so far."""
    return {n.thai for n in deck.picture_words}


def vocabulary_by_position(picture_words: list, base: int) -> dict[str, set[str]]:
    """For each word, the vocabulary the learner has met when its sentence
    appears.

    Sentences start once `base` words are known and are scheduled after
    their own target is introduced, so the vocabulary available to a
    sentence is the first max(base, position) words of the introduction
    order -- not the whole deck, which is what made every sentence a
    potential wall of unseen words.
    """
    ranked = sorted(picture_words,
                    key=lambda w: (w.frequency_rank is None, w.frequency_rank or 0))
    out: dict[str, set[str]] = {}
    for index, word in enumerate(ranked):
        position = max(base, index + 1)
        out[word.thai] = {w.thai for w in ranked[:position]}
    return out


def _matches_target(tok: str, target: str) -> bool:
    return tok == target or tok.startswith(target) or tok.endswith(target)


def check_sentence(thai: str, target: str, known: set[str], tokenizer) -> str | None:
    """None when the sentence is acceptable; else a human-readable reason.

    The target must appear as a token, or as a boundary-aligned member of a
    compound token (prefix/suffix match) -- this mirrors the evaluator's own
    lang/target-not-token rule. At most one non-target token may fall
    outside `known` (function-word tolerance comes free: the evaluator's own
    new-elements rule tokenizes the same way and folds in function words).
    """
    toks = tokenizer.tokens(thai)
    if not any(_matches_target(t, target) for t in toks):
        return f"target {target!r} is not a token of {toks!r}"
    unknown = [t for t in toks if not _matches_target(t, target) and t not in known]
    if len(unknown) > 1:
        return f"{len(unknown)} unknown non-target tokens: {unknown!r}"
    return None


def _vocab_sample(known: set[str], target: str, size: int = 80) -> list[str]:
    """A per-target slice of the known vocabulary.

    Offering every call the same sorted first-100 words gave the model the
    same raw material each time, and it built the same sentences out of it.
    Seeded by target, so a given word's prompt is stable across runs.
    """
    words = sorted(known)
    rng = random.Random(target)
    if len(words) <= size:
        rng.shuffle(words)
        return words
    sample = rng.sample(words, size)
    rng.shuffle(sample)          # order is a bias too: the top of a long list
    return sample                # is what the model reaches for first


def _pick_exemplars(exemplars: list[str], target: str, count: int = 3) -> list[str]:
    """A per-target draw from the exemplar pool.

    Every prompt seeing the same three reference sentences is uniformity
    built into the corpus before the model writes a word.
    """
    if len(exemplars) <= count:
        return list(exemplars)
    return random.Random(f"ex:{target}").sample(exemplars, count)


def _prompt(target: str, known: set[str], exemplars: list[str],
           feedback: str | None = None, theme: str | None = None,
           avoid: list[str] | None = None) -> str:
    sample = _vocab_sample(known, target)
    lines = [
        f"Target Thai word or grammar marker: {target}",
        f"Known vocabulary (sample): {', '.join(sample)}",
        "Exemplar sentences (register/style reference):",
        *[f"- {e}" for e in _pick_exemplars(exemplars, target)],
        "Write ONE natural sentence in colloquial spoken Thai, "
        "informal-polite register, that uses the target word or marker and "
        "otherwise sticks to the known vocabulary.",
        # Every constraint below is a defect the judge actually found in the
        # first 732 sentences, not a precaution.
        "Spell politeness particles the standard way: ครับ and ค่ะ. Never the "
        "chat spellings คับ, คร้าบ, ค่า, ค๊า -- the deck teaches written "
        "standard forms even in colloquial register.",
        "Use collocations a native actually says: pair each verb with the "
        "noun it normally takes, rather than a literal translation that is "
        "merely grammatical.",
        "The sentence must be a complete clause with a verb; do not juxtapose "
        "a noun and a place adverb with the linking verb left out.",
        "Say something a person would plausibly say. Avoid contrived "
        "statements built only to contain the target word.",
        "The learner speaks as a man: use ผม for \"I\" and ครับ as the "
        "politeness particle, never ฉัน or ค่ะ.",
        "Vary the sentence frame. Do not open with the same word or "
        "construction as the sentences listed under 'Already in the deck'.",
        "Answer with ONLY the Thai sentence.",
    ]
    if avoid:
        lines.append("Already in the deck (do not echo these frames):")
        lines.extend(f"- {a}" for a in avoid[:8])
    if theme:
        lines.append(f"Where it is natural, set the sentence in the context of {theme}.")
    if feedback:
        lines.append(feedback)
    return "\n".join(lines)


def _next_note_id(deck: Deck, target: str) -> str:
    n = sum(1 for note in deck.sentences if note.target == target) + 1
    return f"sn-{target}-{n}"


def recent_frames(deck: Deck, limit: int = 8) -> list[str]:
    """A spread of sentences already in the deck, as frames to avoid.

    Sampled across the deck rather than taken from one end, so the model
    sees the shapes that actually recur.
    """
    sents = [n.thai for n in deck.sentences]
    if len(sents) <= limit:
        return sents
    step = len(sents) // limit
    return [sents[i * step] for i in range(limit)]


def _generate(ctx, known: set[str], target: str,
             feedback: str | None = None,
             theme: str | None = None,
             avoid: list[str] | None = None) -> tuple[str | None, str | None]:
    prompt = _prompt(target, known, ctx.exemplars, feedback, theme, avoid)
    reply = normalize_particles(ctx.llm.complete("sentences", PROMPT_VERSION,
                                                 prompt).strip())
    reason = check_sentence(reply, target, known, ctx.tokenizer)
    if reason is None:
        return reply, None
    prompt = _prompt(target, known, ctx.exemplars,
                     f"Previous attempt was rejected: {reason}", theme, avoid)
    reply = normalize_particles(ctx.llm.complete("sentences", PROMPT_VERSION,
                                                 prompt).strip())
    reason = check_sentence(reply, target, known, ctx.tokenizer)
    if reason is None:
        return reply, None
    return None, reason


def _new_note(deck: Deck, target: str, kind: str, thai: str,
             grammar_note: str | None = None,
             note_id: str | None = None) -> SentenceNote:
    note_id = note_id or _next_note_id(deck, target)
    return SentenceNote(
        id=note_id, kind=kind, thai=thai, target=target,
        audio=Audio(file=f"audio/sentences/{note_id}.mp3",
                    source="tts", speaker="pending"),
        grammar_note=grammar_note)


def fill_sentences(gaps: Gaps, deck: Deck, ctx,
                   checkpoint=None, checkpoint_every: int = 25) -> ProducerResult:
    result = ProducerResult()
    base = ctx.config.sentence_base
    if len(deck.picture_words) < base:
        result.blocked.append(
            f"sentence_base not reached: {len(deck.picture_words)} < {base}")
        return result

    known = known_vocab(deck)
    known_at = vocabulary_by_position(deck.picture_words, base)
    ctx.known_at = known_at
    ctx.checkpoint = checkpoint
    ctx.checkpoint_every = checkpoint_every
    # An LlmError means the backend is unavailable (limit hit, CLI down):
    # keep whatever was generated so far and stop rather than failing
    # every remaining call one subprocess at a time.
    try:
        _add_new_word_sentences(deck, ctx, known, result)
        _add_themed_sentences(deck, ctx, known, result)
        _add_grammar_sentences(deck, ctx, known, result)
        _regenerate_judged(gaps, deck, ctx, known, result)
    except LlmError as exc:
        result.blocked.append(f"llm unavailable, sentence generation halted: {exc}")
    return result


def _add_new_word_sentences(deck: Deck, ctx, known: set[str],
                            result: ProducerResult) -> None:
    have_new_word = {n.target for n in deck.sentences if n.kind == "new_word"}
    for word in deck.picture_words:
        if word.thai in have_new_word:
            continue
        have_new_word.add(word.thai)      # one sentence per thai, even if the word repeats
        at_position = getattr(ctx, "known_at", {}).get(word.thai, known)
        thai, reason = _generate(ctx, at_position, word.thai,
                                 avoid=recent_frames(deck))
        if thai is None:
            result.blocked.append(f"{word.thai}: {reason}")
            continue
        deck.sentences.append(_new_note(deck, word.thai, "new_word", thai))
        result.added += 1
        _checkpoint(ctx, result)


def _checkpoint(ctx, result: ProducerResult) -> None:
    """Flush the deck periodically: these runs take hours and a kill that
    loses every sentence generated so far is the expensive failure."""
    fn = getattr(ctx, "checkpoint", None)
    every = getattr(ctx, "checkpoint_every", 25)
    if fn and result.added and result.added % every == 0:
        fn()


def _is_emphasized(ctx, word) -> bool:
    """Weighted category, or an entry the word list extension pass added."""
    emphasis = getattr(ctx, "emphasis", None)
    if emphasis is None:
        return False
    if emphasis.emphasized(word.category):
        return True
    return any(e.thai == word.thai and e.emphasis
               for e in getattr(ctx, "word_list", []))


def _add_themed_sentences(deck: Deck, ctx, known: set[str],
                          result: ProducerResult) -> None:
    """A second sentence, set in the learner's theme, for emphasized words."""
    emphasis = getattr(ctx, "emphasis", None)
    if emphasis is None:
        return
    have = {n.id for n in deck.sentences}
    for word in deck.picture_words:
        note_id = f"sn-{word.thai}-themed"
        if note_id in have or not _is_emphasized(ctx, word):
            continue
        at_position = getattr(ctx, "known_at", {}).get(word.thai, known)
        thai, reason = _generate(ctx, at_position, word.thai, theme=emphasis.theme,
                                 avoid=recent_frames(deck))
        if thai is None:
            result.blocked.append(f"{word.thai} (themed): {reason}")
            continue
        deck.sentences.append(
            _new_note(deck, word.thai, "new_word", thai, note_id=note_id))
        result.added += 1


def _add_grammar_sentences(deck: Deck, ctx, known: set[str],
                           result: ProducerResult) -> None:
    have_grammar = {(n.target, n.kind) for n in deck.sentences
                    if n.kind in ("word_form", "word_order")}
    for gp in getattr(ctx, "grammar_points", []):
        marker, kind = gp["marker"], gp["kind"]
        if (marker, kind) in have_grammar:
            continue
        thai, reason = _generate(ctx, known, marker,
                                 avoid=recent_frames(deck))
        if thai is None:
            result.blocked.append(f"{marker}: {reason}")
            continue
        deck.sentences.append(
            _new_note(deck, marker, kind, thai, grammar_note=gp.get("description")))
        result.added += 1


def _regenerate_judged(gaps: Gaps, deck: Deck, ctx, known: set[str],
                       result: ProducerResult) -> None:
    judge_messages = {f.note_id: f.message for f in gaps.findings_for("judge/")
                      if f.note_id}
    for note in deck.sentences:
        if note.id not in judge_messages:
            continue
        at_position = getattr(ctx, "known_at", {}).get(note.target, known)
        thai, reason = _generate(ctx, at_position, note.target,
                                 feedback=f"Judge feedback: {judge_messages[note.id]}",
                                 avoid=recent_frames(deck))
        if thai is None:
            result.blocked.append(f"{note.id}: {reason}")
            continue
        old_media = deck.root / "media" / note.audio.file
        if old_media.exists():
            old_media.unlink()
        note.thai = thai
        note.audio.speaker = "pending"
        result.changed += 1


def load_exemplars(path: Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()
           if ln.strip()]


def fetch_exemplars(out_path: Path, url: str = EXEMPLAR_URL,
                    sample_size: int = 500, min_tokens: int = 3,
                    max_tokens: int = 12, rng: random.Random | None = None) -> int:
    """Download the OPUS OpenSubtitles Thai monolingual corpus and sample
    lines of 3-12 whitespace-separated tokens into `out_path`, for use as
    few-shot exemplar sentences in the sentence producer's LLM prompts."""
    rng = rng or random.Random()
    raw = urllib.request.urlopen(url).read()
    text = gzip.decompress(raw).decode("utf-8", errors="ignore")
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        n_tokens = len(line.split())
        if min_tokens <= n_tokens <= max_tokens:
            candidates.append(line)
    sample = rng.sample(candidates, min(sample_size, len(candidates)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sample) + "\n", encoding="utf-8")
    return len(sample)
