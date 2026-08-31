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

    # Keep existing notes' search phrase in step with the word list, which
    # is where phrases are curated and where the judge's proposals land.
    phrases = {e.thai: e.image_query for e in ctx.word_list if e.image_query}
    for note in deck.picture_words:
        phrase = phrases.get(note.thai)
        if phrase and note.image_query != phrase:
            note.image_query = phrase
            result.changed += 1

    taken = {n.id for n in deck.picture_words}
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

        # Rank alone is not an identity: a frequency list with ties would
        # give two words one id, one image path and one Anki guid. The thai
        # is what actually distinguishes them.
        note_id = (f"pw-{rank}" if rank != UNRANKED_RANK else f"pw-u-{entry.thai}")
        if note_id in taken:
            note_id = f"pw-{rank}-{entry.thai}"
        taken.add(note_id)
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
            image_query=entry.image_query,
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
