import gzip
import random
import urllib.request
from pathlib import Path

from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import Audio, SentenceNote
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps

PROMPT_VERSION = "sn1"

# OPUS OpenSubtitles v2018 Thai monolingual corpus, hosted on CSC's object
# storage (the canonical download path for OPUS mono corpora). Used as a
# source of naturalistic colloquial-register exemplar sentences for the
# sentence producer's LLM prompts.
EXEMPLAR_URL = ("https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/"
                "th.txt.gz")


def known_vocab(deck: Deck) -> set[str]:
    """Picture-word thais introduced so far."""
    return {n.thai for n in deck.picture_words}


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


def _prompt(target: str, known: set[str], exemplars: list[str],
           feedback: str | None = None) -> str:
    sample = sorted(known)[:100]
    lines = [
        f"Target Thai word or grammar marker: {target}",
        f"Known vocabulary (sample): {', '.join(sample)}",
        "Exemplar sentences (register/style reference):",
        *[f"- {e}" for e in exemplars[:3]],
        "Write ONE natural sentence in colloquial spoken Thai, "
        "informal-polite register, that uses the target word or marker and "
        "otherwise sticks to the known vocabulary.",
        "Answer with ONLY the Thai sentence.",
    ]
    if feedback:
        lines.append(feedback)
    return "\n".join(lines)


def _next_note_id(deck: Deck, target: str) -> str:
    n = sum(1 for note in deck.sentences if note.target == target) + 1
    return f"sn-{target}-{n}"


def _generate(ctx, known: set[str], target: str,
             feedback: str | None = None) -> tuple[str | None, str | None]:
    prompt = _prompt(target, known, ctx.exemplars, feedback)
    reply = ctx.llm.complete("sentences", PROMPT_VERSION, prompt).strip()
    reason = check_sentence(reply, target, known, ctx.tokenizer)
    if reason is None:
        return reply, None
    prompt = _prompt(target, known, ctx.exemplars,
                     f"Previous attempt was rejected: {reason}")
    reply = ctx.llm.complete("sentences", PROMPT_VERSION, prompt).strip()
    reason = check_sentence(reply, target, known, ctx.tokenizer)
    if reason is None:
        return reply, None
    return None, reason


def _new_note(deck: Deck, target: str, kind: str, thai: str,
             grammar_note: str | None = None) -> SentenceNote:
    note_id = _next_note_id(deck, target)
    return SentenceNote(
        id=note_id, kind=kind, thai=thai, target=target,
        audio=Audio(file=f"audio/sentences/{note_id}.mp3",
                    source="tts", speaker="pending"),
        grammar_note=grammar_note)


def fill_sentences(gaps: Gaps, deck: Deck, ctx) -> ProducerResult:
    result = ProducerResult()
    base = ctx.config.sentence_base
    if len(deck.picture_words) < base:
        result.blocked.append(
            f"sentence_base not reached: {len(deck.picture_words)} < {base}")
        return result

    known = known_vocab(deck)

    have_new_word = {n.target for n in deck.sentences if n.kind == "new_word"}
    for word in deck.picture_words:
        if word.thai in have_new_word:
            continue
        thai, reason = _generate(ctx, known, word.thai)
        if thai is None:
            result.blocked.append(f"{word.thai}: {reason}")
            continue
        deck.sentences.append(_new_note(deck, word.thai, "new_word", thai))
        result.added += 1

    have_grammar = {(n.target, n.kind) for n in deck.sentences
                    if n.kind in ("word_form", "word_order")}
    for gp in getattr(ctx, "grammar_points", []):
        marker, kind = gp["marker"], gp["kind"]
        if (marker, kind) in have_grammar:
            continue
        thai, reason = _generate(ctx, known, marker)
        if thai is None:
            result.blocked.append(f"{marker}: {reason}")
            continue
        deck.sentences.append(
            _new_note(deck, marker, kind, thai, grammar_note=gp.get("description")))
        result.added += 1

    judge_messages = {f.note_id: f.message for f in gaps.findings_for("judge/")
                      if f.note_id}
    for note in deck.sentences:
        if note.id not in judge_messages:
            continue
        thai, reason = _generate(ctx, known, note.target,
                                 feedback=f"Judge feedback: {judge_messages[note.id]}")
        if thai is None:
            result.blocked.append(f"{note.id}: {reason}")
            continue
        note.thai = thai
        result.changed += 1

    return result


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
