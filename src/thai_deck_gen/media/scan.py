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

def pending_images(deck: Deck) -> list[ImageNeed]:
    """
    Find notes with an image ref whose file is missing.
    term = note thai (picture words), gloss = note.gloss
    """
    needs = []
    media_root = deck.root / "media"

    for family, note in deck.all_notes():
        if family == "picture_word":
            # PictureWordNote has image
            image_ref = note.image
            file_path = media_root / image_ref
            if not file_path.exists():
                needs.append(ImageNeed(
                    family=family,
                    note_id=note.id,
                    term=note.thai,
                    gloss=note.gloss,
                    path=image_ref
                ))

        elif family == "sentence":
            # SentenceNote has optional image
            if note.image:
                image_ref = note.image
                file_path = media_root / image_ref
                if not file_path.exists():
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
            if not file_path.exists():
                needs.append(ImageNeed(
                    family=family,
                    note_id=note.id,
                    term=note.pattern,
                    gloss=None,
                    path=image_ref
                ))

    return needs
