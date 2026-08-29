import yaml
from pathlib import Path
from thai_deck_eval.lang.ipa import parse_ipa, render_ipa
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import Audio, PictureWordNote
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps

UNRANKED_RANK = 10**6   # sentinel frequency_rank for word-list entries absent from the frequency list


def fill_words(gaps: Gaps, deck: Deck, ctx) -> ProducerResult:
    result = ProducerResult()
    existing_thai = {n.thai for n in deck.picture_words}

    ranked_words = []
    seen = set(existing_thai)
    for entry in ctx.word_list:
        if entry.thai in seen:          # listed in two categories, or already in the deck
            continue
        if not entry.picturable:
            result.blocked.append(f"{entry.thai}: not picturable")
            continue
        seen.add(entry.thai)
        # absent from the frequency list is not a reason to drop a curated word:
        # it goes in after every ranked one, keyed by its thai
        ranked_words.append((ctx.freq.rank(entry.thai) or UNRANKED_RANK, entry))

    ranked_words.sort(key=lambda x: x[0])

    adjudication_queue_words = set()
    if ctx.adjudication_queue.exists():
        existing_queue = yaml.safe_load(
            ctx.adjudication_queue.read_text(encoding="utf-8")) or []
        adjudication_queue_words = set(existing_queue)

    for rank, entry in ranked_words:
        ipa = None
        if entry.thai in ctx.exceptions:
            ipa = ctx.exceptions[entry.thai]
        else:
            syls = ctx.g2p.syllables(entry.thai)
            if syls is not None:
                ipa = render_ipa(syls)
            else:
                adjudication_queue_words.add(entry.thai)

        note_id = f"pw-{rank}" if rank != UNRANKED_RANK else f"pw-u-{entry.thai}"
        note = PictureWordNote(
            id=note_id,
            thai=entry.thai,
            image=f"images/{note_id}.jpg",
            audio=Audio(
                file=f"audio/picture_words/{note_id}.mp3",
                source="native",
                speaker="pending"
            ),
            frequency_rank=rank,
            category=entry.category,
            part_of_speech=entry.part_of_speech,
            classifier=entry.classifier,
            ipa=ipa,
            test_spelling=rank <= ctx.config.test_spelling_rank,
            personal_connection=None,
            gloss=None,          # FF doctrine: no L1 gloss on picture words; image search reads the word list
        )
        deck.picture_words.append(note)
        result.added += 1

    if adjudication_queue_words:
        ctx.adjudication_queue.parent.mkdir(parents=True, exist_ok=True)
        ctx.adjudication_queue.write_text(
            yaml.safe_dump(sorted(adjudication_queue_words), allow_unicode=True),
            encoding="utf-8"
        )

    return result
