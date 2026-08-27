from typing import Callable, Iterable
from thai_deck_eval.lang.ipa import IpaSyllable, parse_ipa, render_ipa
from thai_deck_eval.lang.ports import G2P
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import Audio, MinimalPairNote, PairMember
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps

_ASPIRATION_ONSETS = {"labial": {"p", "pʰ"}, "alveolar": {"t", "tʰ"},
                       "velar": {"k", "kʰ"}, "affricate": {"tɕ", "tɕʰ"}}

Predicate = Callable[[IpaSyllable, IpaSyllable], bool]

def analyze_lexicon(words: Iterable[str], g2p: G2P,
                    exceptions: dict[str, str]) -> dict[str, IpaSyllable]:
    lexicon: dict[str, IpaSyllable] = {}
    for word in words:
        syls = parse_ipa(exceptions[word]) if word in exceptions else g2p.syllables(word)
        if syls is not None and len(syls) == 1:
            lexicon[word] = syls[0]
    return lexicon

def _predicate(contrast_id: str) -> Predicate | None:
    kind, _, rest = contrast_id.partition(":")
    if kind == "tone":
        tones = set(rest.split("-"))
        return lambda a, b: ((a.onset, a.vowel, a.long, a.coda) ==
                             (b.onset, b.vowel, b.long, b.coda)
                             and {a.tone.value, b.tone.value} == tones)
    if kind == "aspiration":
        onsets = _ASPIRATION_ONSETS.get(rest)
        if onsets is None:
            return None
        return lambda a, b: ((a.vowel, a.long, a.coda, a.tone) ==
                             (b.vowel, b.long, b.coda, b.tone)
                             and {a.onset, b.onset} == onsets)
    if kind == "vowel_length":
        return lambda a, b: ((a.onset, a.vowel, a.coda, a.tone) ==
                             (b.onset, b.vowel, b.coda, b.tone)
                             and a.long != b.long)
    if kind == "consonant" and rest == "ng-onset":
        return lambda a, b: ((a.vowel, a.long, a.coda, a.tone) ==
                             (b.vowel, b.long, b.coda, b.tone)
                             and a.onset != b.onset and "ŋ" in (a.onset, b.onset))
    if kind == "consonant" and rest == "r-l":
        return lambda a, b: ((a.vowel, a.long, a.coda, a.tone) ==
                             (b.vowel, b.long, b.coda, b.tone)
                             and {a.onset, b.onset} == {"r", "l"})
    if kind == "vowel_quality" and rest in ("e-ɛ", "o-ɔ"):
        vowels = set(rest.split("-"))
        return lambda a, b: ((a.onset, a.long, a.coda, a.tone) ==
                             (b.onset, b.long, b.coda, b.tone)
                             and {a.vowel, b.vowel} == vowels)
    if kind == "vowel_quality" and rest in ("ɯ", "ɤ"):
        return lambda a, b: ((a.onset, a.long, a.coda, a.tone) ==
                             (b.onset, b.long, b.coda, b.tone)
                             and a.vowel != b.vowel and rest in (a.vowel, b.vowel))
    if kind == "final" and rest == "unreleased":
        return lambda a, b: ((a.onset, a.vowel, a.long, a.tone) ==
                             (b.onset, b.vowel, b.long, b.tone)
                             and a.coda != b.coda
                             and a.coda in ("p", "t", "k") and b.coda in ("p", "t", "k"))
    return None

def find_pair(contrast_id: str, lexicon: dict[str, IpaSyllable]) -> tuple[str, str] | None:
    pred = _predicate(contrast_id)
    if pred is None:
        return None
    words = sorted(lexicon)
    for i, w1 in enumerate(words):
        for w2 in words[i + 1:]:
            if pred(lexicon[w1], lexicon[w2]):
                return (w1, w2)
    return None

def fill_pairs(gaps: Gaps, deck: Deck, ctx) -> ProducerResult:
    result = ProducerResult()
    existing_ids = {n.id for n in deck.minimal_pairs}
    lexicon = analyze_lexicon(ctx.lexicon_words, ctx.g2p, ctx.exceptions)
    for contrast_id in gaps.missing_contrasts:
        pair = find_pair(contrast_id, lexicon)
        if pair:
            members_data = [(w, lexicon[w]) for w in pair]
        else:
            seed = ctx.pair_seeds.get(contrast_id)
            if not seed:
                result.blocked.append(contrast_id)
                continue
            members_data = [(thai, parse_ipa(ipa)[0]) for thai, ipa in seed]
        note_id = f"mp-{contrast_id.replace(':', '-')}-1"
        if note_id in existing_ids:
            continue
        members = [
            PairMember(thai=thai, ipa=render_ipa([syl]),
                      audio=Audio(file=f"audio/minimal_pairs/{note_id}_{i}.mp3",
                                  source="native", speaker="pending"))
            for i, (thai, syl) in enumerate(members_data)]
        deck.minimal_pairs.append(MinimalPairNote(
            id=note_id, contrast=contrast_id.partition(":")[0], members=members))
        existing_ids.add(note_id)
        result.added += 1
    return result
