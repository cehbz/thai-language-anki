from thai_deck_eval.lang.ports import FrequencyList
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import (MinimalPairNote, PictureWordNote,
                                        SentenceNote, SpellingSoundNote)

UNRANKED = 10 ** 9

def member_rank(note, freq: FrequencyList) -> int:
    """Min rank over member thais / example_word / target / thai (family-dependent)."""
    if isinstance(note, MinimalPairNote):
        ranks = [r for r in (freq.rank(m.thai) for m in note.members) if r is not None]
        return min(ranks) if ranks else UNRANKED
    if isinstance(note, SpellingSoundNote):
        rank = freq.rank(note.example_word)
    elif isinstance(note, SentenceNote):
        rank = freq.rank(note.target)
    elif isinstance(note, PictureWordNote):
        rank = freq.rank(note.thai)
    else:
        return UNRANKED
    return rank if rank is not None else UNRANKED

def intro_order(deck: Deck, freq: FrequencyList,
               base: int = 300) -> list[tuple[str, object]]:
    """(family, note) sequence: minimal_pairs + spelling_sound merged by
    member_rank, then picture_words by frequency_rank, sentences interleaved:
    after word #base is introduced, flush (ordered by target rank) each
    sentence whose target word is already introduced; remaining sentences
    are appended at the end."""
    sounds: list[tuple[str, object]] = (
        [("minimal_pair", n) for n in deck.minimal_pairs] +
        [("spelling_sound", n) for n in deck.spelling_sound])
    sounds.sort(key=lambda fn: member_rank(fn[1], freq))

    words = sorted(deck.picture_words, key=lambda w: w.frequency_rank)
    rank_by_thai = {w.thai: w.frequency_rank for w in words}
    pending = list(deck.sentences)

    result: list[tuple[str, object]] = list(sounds)
    introduced: set[str] = set()
    count = 0
    for word in words:
        result.append(("picture_word", word))
        introduced.add(word.thai)
        count += 1
        if count >= base:
            ready = [s for s in pending if s.target in introduced]
            ready.sort(key=lambda s: rank_by_thai.get(s.target, UNRANKED))
            for s in ready:
                result.append(("sentence", s))
                pending.remove(s)

    for s in pending:
        result.append(("sentence", s))

    return result
