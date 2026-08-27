import re
import unicodedata
from ..core.findings import Dimension, Severity, Stage
from ..core.registry import rule

_LATIN = re.compile(r"[A-Za-z]")

def iter_media_refs(deck):
    for note in deck.minimal_pairs:
        for m in note.members:
            yield note.id, m.audio.file
    for note in deck.spelling_sound:
        yield note.id, note.audio.file
        yield note.id, note.image
    for note in deck.picture_words:
        yield note.id, note.audio.file
        yield note.id, note.image
    for note in deck.sentences:
        yield note.id, note.audio.file
        if note.image:
            yield note.id, note.image

@rule("mech/media-missing", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
def media_missing(ctx):
    for note_id, ref in iter_media_refs(ctx.deck):
        if not (ctx.deck.root / "media" / ref).is_file():
            yield media_missing.finding(f"media file not found: {ref}",
                                        note_id=note_id)

@rule("mech/media-orphan", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.INFO)
def media_orphan(ctx):
    media_dir = ctx.deck.root / "media"
    if not media_dir.is_dir():
        return
    referenced = {ref for _, ref in iter_media_refs(ctx.deck)}
    for p in media_dir.rglob("*"):
        if p.is_file() and str(p.relative_to(media_dir)) not in referenced:
            yield media_orphan.finding(
                f"unreferenced media file: {p.relative_to(media_dir)}")

def _thai_fields(deck):
    for note in deck.minimal_pairs:
        for m in note.members:
            yield note.id, "thai", m.thai
    for note in deck.spelling_sound:
        yield note.id, "example_word", note.example_word
    for note in deck.picture_words:
        yield note.id, "thai", note.thai
    for note in deck.sentences:
        yield note.id, "thai", note.thai
        yield note.id, "target", note.target
        if note.definition:
            yield note.id, "definition", note.definition

@rule("mech/latin-in-thai", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
def latin_in_thai(ctx):
    for note_id, fieldname, text in _thai_fields(ctx.deck):
        if _LATIN.search(unicodedata.normalize("NFC", text)):
            yield latin_in_thai.finding(
                f"Latin characters in Thai field '{fieldname}': {text!r}",
                note_id=note_id)

@rule("mech/duplicate-note", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
def duplicate_note(ctx):
    seen: dict[str, str] = {}
    keys = [(n.id, f"picture:{n.thai}") for n in ctx.deck.picture_words]
    keys += [(n.id, f"sentence:{n.thai}") for n in ctx.deck.sentences]
    for note_id, key in keys:
        if key in seen:
            yield duplicate_note.finding(
                f"duplicate of note {seen[key]}", note_id=note_id)
        else:
            seen[key] = note_id

@rule("mech/target-not-in-sentence", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.ERROR)
def target_not_in_sentence(ctx):
    for note in ctx.deck.sentences:
        if note.target not in note.thai:
            yield target_not_in_sentence.finding(
                f"target {note.target!r} not found in sentence", note_id=note.id)

@rule("mech/gloss-on-picture-word", Stage.MECHANICAL, Dimension.INTEGRITY, Severity.WARN)
def gloss_on_picture_word(ctx):
    for note in ctx.deck.picture_words:
        if note.gloss:
            yield gloss_on_picture_word.finding(
                "concrete picture words carry no L1 gloss", note_id=note.id)
