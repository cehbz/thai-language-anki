import yaml
from pathlib import Path
from thai_deck_eval.lang.ipa import parse_ipa, render_ipa
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import Audio, PictureWordNote
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps


def fill_words(gaps: Gaps, deck: Deck, ctx) -> ProducerResult:
    result = ProducerResult()
    existing_thai = {n.thai for n in deck.picture_words}

    ranked_words = []
    for entry in ctx.word_list:
        if entry.thai in existing_thai:
            continue
        if not entry.picturable:
            result.blocked.append(f"{entry.thai}: not picturable")
            continue

        rank = ctx.freq.rank(entry.thai)
        if rank is None:
            result.blocked.append(f"{entry.thai}: unranked")
            continue

        ranked_words.append((rank, entry))

    ranked_words.sort(key=lambda x: x[0])

    adjudication_queue_words = set()
    if ctx.adjudication_queue.exists():
        existing_queue = yaml.safe_load(ctx.adjudication_queue.read_text()) or []
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

        note = PictureWordNote(
            id=f"pw-{rank}",
            thai=entry.thai,
            image=f"images/pw-{rank}.jpg",
            audio=Audio(
                file=f"audio/picture_words/pw-{rank}.mp3",
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
            gloss=entry.gloss
        )
        deck.picture_words.append(note)
        result.added += 1

    if adjudication_queue_words:
        ctx.adjudication_queue.parent.mkdir(parents=True, exist_ok=True)
        ctx.adjudication_queue.write_text(
            yaml.safe_dump(sorted(adjudication_queue_words), allow_unicode=True)
        )

    return result
