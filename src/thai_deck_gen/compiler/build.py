import hashlib
import os
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import genanki

from thai_deck_eval.lang.ports import FrequencyList
from thai_deck_eval.model.deck import Deck
from thai_deck_gen.compiler.ordering import intro_order
from thai_deck_gen.media.manifest import Manifest

STAGE_OF = {"minimal_pair": "sounds", "spelling_sound": "sounds",
           "picture_word": "words", "sentence": "sentences"}

# family passed to note_guid, per the concrete example in the brief
# (note_guid("minimal_pairs", ...)) generalized to every family: the
# deck-attribute name, so guids stay stable across recompiles regardless
# of any future rename of the singular "family" tag string.
GUID_FAMILY = {"minimal_pair": "minimal_pairs", "spelling_sound": "spelling_sound",
              "picture_word": "picture_words", "sentence": "sentences"}


class CompileError(Exception):
    def __init__(self, files: list[str]):
        self.files = files
        super().__init__(f"missing media file(s): {', '.join(files)}")


def _model_id(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def note_guid(family: str, note_id: str) -> str:
    return genanki.guid_for(family, note_id)


def _deck_id(name: str) -> int:
    return int(hashlib.sha256(f"thai-deck-gen::{name}".encode()).hexdigest()[:8], 16)


CARD_CSS = """
.card { font-family: sans-serif; font-size: 24px; text-align: center;
        color: #222; background: #fff; }
img { max-width: 100%; max-height: 55vh; width: auto; height: auto; }
.thai, .cloze, .pattern { font-size: 52px; }
.choices { font-size: 44px; }
.ipa { font-size: 22px; color: #666; }
.answer { font-weight: bold; }
.other { color: #888; }
.target { font-size: 44px; }
.gloss, .grammar, .classifier { font-size: 20px; color: #555; }
.nightMode .card { color: #ddd; background: #2f2f31; }
.nightMode .ipa, .nightMode .other { color: #999; }
"""


def _model(name: str, fields: list[str], templates: list[dict]) -> genanki.Model:
    return genanki.Model(_model_id(name), name,
                         fields=[{"name": f} for f in fields],
                         templates=templates, css=CARD_CSS)


def _base_tags(family: str) -> list[str]:
    return [f"family::{family}", f"stage::{STAGE_OF[family]}"]


MODELS: dict[str, genanki.Model] = {
    "minimal_pair": _model(
        "minimal_pair",
        ["MemberIndex", "Thai", "Ipa", "Audio", "OtherThai", "OtherIpa", "ReviewNote"],
        [{
            "name": "Recognition",
            "qfmt": '{{Audio}}<div>Which word did you hear?</div>'
                   '<div class="choices">{{Thai}} / {{OtherThai}}</div>',
            "afmt": '{{FrontSide}}<hr id="answer">'
                   '<div class="answer">{{Thai}} <span class="ipa">[{{Ipa}}]</span></div>'
                   '<div class="other">{{OtherThai}} <span class="ipa">[{{OtherIpa}}]</span></div>',
        }]),
    "spelling_sound": _model(
        "spelling_sound",
        ["Pattern", "ExampleWord", "KeywordGloss", "Audio", "Image", "ReviewNote"],
        [{
            "name": "PatternToSound",
            "qfmt": '<div class="pattern">{{Pattern}}</div>',
            "afmt": '{{FrontSide}}<hr id="answer">{{Audio}}<div>{{ExampleWord}}</div>'
                   '{{#KeywordGloss}}<div class="gloss">{{KeywordGloss}}</div>{{/KeywordGloss}}{{Image}}',
        }, {
            "name": "SoundToPattern",
            "qfmt": '{{Audio}}{{Image}}',
            "afmt": '{{FrontSide}}<hr id="answer"><div class="pattern">{{Pattern}}</div>'
                   '<div>{{ExampleWord}}</div>'
                   '{{#KeywordGloss}}<div class="gloss">{{KeywordGloss}}</div>{{/KeywordGloss}}',
        }]),
    "picture_word": _model(
        "picture_word",
        ["Thai", "Image", "Audio", "Ipa", "Classifier", "TestSpelling", "ReviewNote"],
        [{
            "name": "Comprehension",
            "qfmt": '{{Image}}{{Audio}}',
            "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{Thai}}</div>'
                   '<div class="ipa">{{Ipa}}</div>'
                   '{{#Classifier}}<div class="classifier">{{Classifier}}</div>{{/Classifier}}',
        }, {
            "name": "Production",
            "qfmt": '{{Image}}',
            "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{Thai}}</div>{{Audio}}',
        }, {
            "name": "Spelling",
            "qfmt": '{{#TestSpelling}}{{Audio}}{{Image}}{{/TestSpelling}}',
            "afmt": '{{#TestSpelling}}{{FrontSide}}<hr id="answer">'
                   '<div class="thai">{{Thai}}</div>{{/TestSpelling}}',
        }]),
    "sentence": _model(
        "sentence",
        ["ThaiCloze", "Target", "Audio", "Image", "Gloss", "GrammarNote", "ReviewNote"],
        [{
            "name": "Cloze",
            "qfmt": '<div class="cloze">{{ThaiCloze}}</div>',
            "afmt": '{{FrontSide}}<hr id="answer"><div class="target">{{Target}}</div>{{Image}}'
                   '{{#Gloss}}<div class="gloss">{{Gloss}}</div>{{/Gloss}}'
                   '{{#GrammarNote}}<div class="grammar">{{GrammarNote}}</div>{{/GrammarNote}}',
        }, {
            "name": "Listening",
            "qfmt": '{{#Audio}}{{Audio}}{{/Audio}}',
            "afmt": '{{#Audio}}{{FrontSide}}<hr id="answer">'
                   '<div class="cloze">{{ThaiCloze}}</div><div class="target">{{Target}}</div>{{/Audio}}',
        }]),
}


def _refs(family: str, note) -> list[str]:
    if family == "minimal_pair":
        return [m.audio.file for m in note.members]
    if family in ("spelling_sound", "picture_word"):
        return [note.audio.file, note.image]
    if family == "sentence":
        refs = [note.audio.file]
        if note.image:
            refs.append(note.image)
        return refs
    return []


def _required_refs(family: str, note) -> list[str]:
    """Media the card is *about*: without it there is no card to study.

    A picture word is its image, a spelling note and a minimal pair are
    their audio. Everything else degrades to a working card with an empty
    field, so a sentence still teaches reading before its recording lands.
    """
    if family == "minimal_pair":
        return [m.audio.file for m in note.members]
    if family == "spelling_sound":
        return [note.audio.file]
    if family == "picture_word":
        return [note.image]
    return []


def _resolve_media(deck: Deck, order: list[tuple[str, object]]
                   ) -> tuple[dict[str, str], list[str]]:
    """Deck-relative ref -> basename to use in the package; and missing refs."""
    media_root = deck.root / "media"
    basename_of: dict[str, str] = {}
    source_of_basename: dict[str, str] = {}
    missing: list[str] = []
    for family, note in order:
        for ref in _refs(family, note):
            if ref in basename_of:
                continue
            abs_path = media_root / ref
            if not abs_path.exists():
                missing.append(ref)
                continue
            base = Path(ref).name
            claim = source_of_basename.get(base)
            if claim is not None and claim != ref:
                base = ref.replace("/", "__")  # collision: namespace it
            source_of_basename[base] = ref
            basename_of[ref] = base
    return basename_of, missing


def _stage_media(deck: Deck, basename_of: dict[str, str], tmp_dir: Path) -> list[Path]:
    media_root = deck.root / "media"
    paths = []
    for ref, base in basename_of.items():
        abs_path = media_root / ref
        if abs_path.name == base:
            paths.append(abs_path)
        else:
            staged = tmp_dir / base
            shutil.copy(abs_path, staged)
            paths.append(staged)
    return paths


def _src_tags(manifest: Manifest, kind: str, *refs: str | None) -> list[str]:
    tags = []
    for ref in refs:
        if not ref:
            continue
        channel = manifest.channel_of(f"media/{ref}")
        if channel:
            tags.append(f"{kind}-src::{channel}")
    return tags


def _sound(basename_of, ref) -> str:
    """`[sound:...]` for a staged file; empty when the recording isn't there
    yet, which leaves audio-gated cards ungenerated rather than broken."""
    base = basename_of.get(ref) if ref else None
    return f"[sound:{base}]" if base else ""


def _img(basename_of, ref) -> str:
    base = basename_of.get(ref) if ref else None
    return f'<img src="{base}">' if base else ""


def _pair_notes(note, manifest, pair_by_note, basename_of) -> list[genanki.Note]:
    tags = _base_tags("minimal_pair")
    contrast_id = pair_by_note.get(note.id)
    if contrast_id:
        tags.append(f"contrast::{contrast_id}")
    notes = []
    for k, member in enumerate(note.members):
        others = [m for i, m in enumerate(note.members) if i != k]
        fields = [str(k), member.thai, member.ipa,
                 _sound(basename_of, member.audio.file),
                 " / ".join(m.thai for m in others),
                 " / ".join(m.ipa for m in others), ""]
        member_tags = tags + _src_tags(manifest, "audio", member.audio.file)
        notes.append(genanki.Note(
            model=MODELS["minimal_pair"], fields=fields, tags=member_tags,
            guid=note_guid(GUID_FAMILY["minimal_pair"], f"{note.id}_{k}")))
    return notes


def _spelling_note(note, manifest, basename_of, gloss_of=None) -> genanki.Note:
    tags = (_base_tags("spelling_sound")
           + _src_tags(manifest, "audio", note.audio.file)
           + _src_tags(manifest, "img", note.image))
    fields = [note.pattern, note.example_word,
             (gloss_of or {}).get(note.example_word, ""),
             _sound(basename_of, note.audio.file),
             _img(basename_of, note.image), ""]
    return genanki.Note(model=MODELS["spelling_sound"], fields=fields, tags=tags,
                        guid=note_guid(GUID_FAMILY["spelling_sound"], note.id))


def _word_note(note, manifest, basename_of) -> genanki.Note:
    tags = (_base_tags("picture_word")
           + _src_tags(manifest, "audio", note.audio.file)
           + _src_tags(manifest, "img", note.image))
    fields = [note.thai, _img(basename_of, note.image),
             _sound(basename_of, note.audio.file),
             note.ipa or "", note.classifier or "",
             "1" if note.test_spelling else "", ""]
    return genanki.Note(model=MODELS["picture_word"], fields=fields, tags=tags,
                        guid=note_guid(GUID_FAMILY["picture_word"], note.id))


def _sentence_note(note, manifest, basename_of) -> genanki.Note:
    tags = (_base_tags("sentence")
            + [f"usage::{getattr(note, 'usage', 'production')}"]
            + _src_tags(manifest, "audio", note.audio.file))
    if note.image:
        tags += _src_tags(manifest, "img", note.image)
    fields = [note.thai.replace(note.target, "___"), note.target,
             _sound(basename_of, note.audio.file),
             _img(basename_of, note.image),
             note.gloss or "", note.grammar_note or "", ""]
    return genanki.Note(model=MODELS["sentence"], fields=fields, tags=tags,
                        guid=note_guid(GUID_FAMILY["sentence"], note.id))


def _build_notes(family, note, manifest, pair_by_note, basename_of,
                 gloss_of=None) -> list[genanki.Note]:
    if family == "minimal_pair":
        return _pair_notes(note, manifest, pair_by_note, basename_of)
    if family == "spelling_sound":
        return [_spelling_note(note, manifest, basename_of, gloss_of)]
    if family == "picture_word":
        return [_word_note(note, manifest, basename_of)]
    if family == "sentence":
        return [_sentence_note(note, manifest, basename_of)]
    raise ValueError(f"unknown family: {family}")


def compile_deck(deck: Deck, manifest: Manifest, out: Path, freq: FrequencyList,
                 pair_by_note: dict[str, str], base: int = 300, emphasis=None,
                 skip_incomplete: bool = False,
                 gloss_of: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """Write `out`; return the (family, note_id) pairs left out.

    Missing media is an error by default. `skip_incomplete` instead drops
    the notes whose essential media is absent and empties the optional
    fields of the rest, so a half-recorded deck still compiles to something
    studiable. Guids are stable, so a later full compile updates those same
    notes in place.
    """
    order = intro_order(deck, freq, base, emphasis=emphasis)
    basename_of, missing = _resolve_media(deck, order)
    dropped: list[tuple[str, str]] = []
    if missing and not skip_incomplete:
        raise CompileError(sorted(set(missing)))
    if skip_incomplete and missing:
        gone = set(missing)
        kept = []
        for family, note in order:
            if gone.intersection(_required_refs(family, note)):
                dropped.append((family, note.id))
            else:
                kept.append((family, note))
        order = kept
        basename_of, _ = _resolve_media(deck, order)   # don't bundle orphaned media

    out = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        media_paths = _stage_media(deck, basename_of, Path(tmp))

        genanki_deck = genanki.Deck(_deck_id(deck.meta.name), deck.meta.name)
        guid_order: list[str] = []
        for pos, (family, note) in enumerate(order):
            for gnote in _build_notes(family, note, manifest, pair_by_note,
                                      basename_of, gloss_of):
                gnote.due = pos
                genanki_deck.add_note(gnote)
                guid_order.append(gnote.guid)

        out.parent.mkdir(parents=True, exist_ok=True)
        # fresh file + os.replace: atomic, and the birth date a file picker
        # shows always matches this compile
        tmp_out = out.with_suffix(out.suffix + ".tmp")
        genanki.Package(genanki_deck,
                        media_files=[str(p) for p in media_paths]).write_to_file(str(tmp_out))
        stamp_due(tmp_out, guid_order)
        os.replace(tmp_out, out)

    return dropped
    return dropped


def stamp_due(apkg: Path, guid_order: list[str]) -> None:
    apkg = Path(apkg)
    due_by_guid = {guid: i for i, guid in enumerate(guid_order)}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(apkg) as zf:
            names = zf.namelist()
            zf.extractall(tmp_dir)

        db_path = tmp_dir / "collection.anki2"
        conn = sqlite3.connect(str(db_path))
        try:
            guid_by_nid = dict(conn.execute("select id, guid from notes"))
            for card_id, nid in conn.execute("select id, nid from cards"):
                due = due_by_guid.get(guid_by_nid.get(nid))
                if due is not None:
                    conn.execute("update cards set due = ? where id = ?", (due, card_id))
            conn.commit()
        finally:
            conn.close()

        with zipfile.ZipFile(apkg, "w") as zf:
            for name in names:
                zf.write(db_path if name == "collection.anki2" else tmp_dir / name, name)
