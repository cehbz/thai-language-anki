from dataclasses import dataclass
from pathlib import Path
from thai_deck_eval.model.deck import Deck

@dataclass
class AudioNeed:
    family: str
    note_id: str
    text: str
    path: str   # deck-relative
    native_required: bool
    member_index: int | None = None

@dataclass
class ImageNeed:
    family: str
    note_id: str
    term: str
    gloss: str | None
    path: str
    category: str | None = None   # FF category, for query disambiguation
    image_query: str | None = None  # phrase describing what a photo looks like

NATIVE_TIER_FAMILIES = {"minimal_pair", "picture_word", "spelling_sound"}

def pending_audio(deck: Deck) -> list[AudioNeed]:
    """
    Find all Audio refs where the file is missing under deck.root/media
    OR speaker == "pending".

    Audio text: pair member → member thai; picture word → note thai;
    sentence → sentence thai; spelling_sound → example_word.
    native_required based on audio.source == "native" (for minimal_pairs always True)
    """
    needs = []
    media_root = deck.root / "media"

    for family, note in deck.all_notes():
        if family == "minimal_pair":
            # MinimalPairNote has members with audio
            for idx, member in enumerate(note.members):
                audio = member.audio
                native_required = (family == "minimal_pair") or (audio.source == "native")
                file_path = media_root / audio.file
                if not file_path.exists() or audio.speaker == "pending":
                    needs.append(AudioNeed(
                        family=family,
                        note_id=note.id,
                        text=member.thai,
                        path=audio.file,
                        native_required=native_required,
                        member_index=idx
                    ))

        elif family == "picture_word":
            # PictureWordNote has audio
            audio = note.audio
            native_required = audio.source == "native"
            file_path = media_root / audio.file
            if not file_path.exists() or audio.speaker == "pending":
                needs.append(AudioNeed(
                    family=family,
                    note_id=note.id,
                    text=note.thai,
                    path=audio.file,
                    native_required=native_required
                ))

        elif family == "sentence":
            # SentenceNote has audio
            audio = note.audio
            native_required = audio.source == "native"
            file_path = media_root / audio.file
            if not file_path.exists() or audio.speaker == "pending":
                needs.append(AudioNeed(
                    family=family,
                    note_id=note.id,
                    text=note.thai,
                    path=audio.file,
                    native_required=native_required
                ))

        elif family == "spelling_sound":
            # SpellingSoundNote has audio
            audio = note.audio
            native_required = audio.source == "native"
            file_path = media_root / audio.file
            if not file_path.exists() or audio.speaker == "pending":
                needs.append(AudioNeed(
                    family=family,
                    note_id=note.id,
                    text=note.example_word,
                    path=audio.file,
                    native_required=native_required
                ))

    return needs

def pending_images(deck: Deck, flagged: set[str] | None = None,
                   glosses: dict[str, str] | None = None,
                   image_queries: dict[str, str] | None = None,
                   include_present: bool = False) -> list[ImageNeed]:
    """
    Find notes with an image ref whose file is missing, plus any note whose
    id is in `flagged` (judge-rejected images) even when the file exists.
    `include_present` returns every note that should have a picture, whatever
    state it is in: a run that judges for itself needs the pictures the deck
    already has, not the previous report's opinion of them.
    term = note thai (picture words); gloss from `glosses` (thai -> gloss,
    normally the word list) for picture words, note.gloss for sentences.
    """
    flagged = flagged or set()
    glosses = glosses or {}
    image_queries = image_queries or {}
    needs = []
    media_root = deck.root / "media"

    for family, note in deck.all_notes():
        if family == "picture_word":
            # PictureWordNote has image
            image_ref = note.image
            file_path = media_root / image_ref
            if include_present or not file_path.exists() or note.id in flagged:
                needs.append(ImageNeed(
                    family=family,
                    note_id=note.id,
                    term=note.thai,
                    gloss=glosses.get(note.thai),
                    path=image_ref,
                    category=note.category,
                    image_query=image_queries.get(note.thai)
                ))

        elif family == "sentence":
            # SentenceNote has optional image
            if note.image:
                image_ref = note.image
                file_path = media_root / image_ref
                if include_present or not file_path.exists() or note.id in flagged:
                    needs.append(ImageNeed(
                        family=family,
                        note_id=note.id,
                        term=note.thai,
                        gloss=note.gloss,
                        path=image_ref
                    ))

        elif family == "spelling_sound":
            # SpellingSoundNote has image
            image_ref = note.image
            file_path = media_root / image_ref
            if include_present or not file_path.exists() or note.id in flagged:
                needs.append(ImageNeed(
                    family=family,
                    note_id=note.id,
                    term=note.pattern,
                    gloss=None,
                    path=image_ref
                ))

    return needs
