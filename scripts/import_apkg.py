"""Import the "Thai 1000 Common Words" Anki deck (an extracted .apkg) into
this project's deck format.

    uv run python scripts/import_apkg.py <extracted-apkg-dir> <out-deck-dir>

<extracted-apkg-dir> is a directory holding an unzipped .apkg: Anki's
`collection.anki2` SQLite database, a `media` file (a JSON object mapping
each zip-member index, as a string, to the real media filename Anki uses
for it, e.g. `{"0": "661.mp3", ...}`), and the raw numbered media files
themselves (`0`, `1`, `2`, ...).

This script's field mapping is HARDCODED to this specific deck's one note
type (verified against the deck: single model, fields `word_eng,
word_tha, word_phonetic, audio, form, sentence_tha, sentence_phonetic,
sentence_eng`, e.g. `['accident', 'อุบัติเหตุ [ครั้ง]', 'u-bàt hàyt
[kráng]', '[sound:3.mp3]', 'noun', 'หล่อน ประสบ อุบัติเหตุ และ ทำ แขน
หัก', ..., '...']`). It is intentionally YAGNI/not generic -- a deck with
a different note-type shape needs a different script.

Per note, two of this project's note families are produced:

  * picture_words: `thai` is `word_tha` with its trailing bracket
    classifier suffix stripped (e.g. "อุบัติเหตุ [ครั้ง]" ->
    "อุบัติเหตุ", classifier "ครั้ง"; None when no bracket is present).
    `part_of_speech` comes from whether `form` starts with "noun" /
    "verb" / "adj" (else "other"). `audio` is the deck's own native
    recording (source "native", speaker "thai1000"), copied into
    media/audio/ under its real filename. `image` is a single shared
    placeholder (media/images/placeholder.png, a tiny 1x1 PNG generated
    by this script -- this deck has no per-word images, and picture-based
    review isn't this stress test's point; the mechanical/method stages'
    findings on that placeholder are expected noise, not a real defect).
    `frequency_rank` prefers this project's own FileFrequencyList; when a
    word isn't in that reference list, it gets `90000 + ordinal` (ordinal
    = 1-based position among kept notes) as a stable placeholder rank
    that's guaranteed never to collide with a real ranked word.
    `category` is a coarse, deliberately approximate placeholder derived
    from `part_of_speech` (noun->"Miscellaneous Nouns",
    verb->"Verbs", adjective->"Adjectives", other->"Miscellaneous
    Nouns") -- this deck carries no real thematic category, so this is a
    known method-fidelity caveat, not a considered categorization. No
    `gloss` is set (per spec). `ipa` is set only when
    `thai_deck_eval.lang.paiboon.paiboon_to_ipa` converts EVERY syllable
    of `word_phonetic` (bracket-stripped) with confidence; otherwise the
    field is simply omitted, matching that converter's own "never guess"
    contract.

  * sentences: `kind` is always "new_word" (this deck's sentences exist
    to showcase the target word, not word-order or word-form drills).
    `thai` is `sentence_tha` (already space-pre-segmented in the source).
    `target` is the SAME stripped `thai` as this note's own picture word
    (each source note yields exactly one picture_word + one sentence, in
    lockstep). `audio` reuses the picture word's own audio file, because
    the schema requires an `audio` field on every sentence note and this
    deck simply doesn't record separate sentence audio -- this is a
    real, deliberate word/sentence-audio mismatch, accepted and
    documented here (and it will show up as expected noise in the
    evaluator's own findings, not a defect in this script). No
    image/definition/gloss.

Notes with an empty/whitespace-only `word_tha` are skipped entirely (both
families); the count is reported. A note whose audio target filename has
no corresponding raw media file in the extraction (this specific deck has
exactly one such gap, "382.mp3") is still written with that filename --
the resulting dangling reference is a real data-quality issue in the
source deck, and is exactly what the evaluator's own `mech/media-missing`
rule exists to catch, not something for this importer to paper over.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from thai_deck_eval.data_io import FileFrequencyList  # noqa: E402
from thai_deck_eval.lang.paiboon import paiboon_to_ipa  # noqa: E402

# A minimal valid 1x1 transparent PNG (68 bytes: signature + IHDR + IDAT +
# IEND), built and verified with Python's own zlib/struct (not hand-rolled
# deflate bytes) -- see scripts/import_apkg.py's git history for the
# generator, or just re-derive it: signature + IHDR(1x1, RGBA) +
# IDAT(zlib-compressed single transparent pixel) + IEND.
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000"
    "001f15c4890000000b49444154789c6360000200000500017a5eab3f"
    "0000000049454e44ae426082"
)

_BRACKET_SUFFIX = re.compile(r"\s*\[[^\]]*\]\s*$")
_SOUND_TAG = re.compile(r"\[sound:(?P<name>[^\]]+)\]")

_POS_MAP = [("noun", "noun"), ("verb", "verb"), ("adj", "adjective")]
_CATEGORY_MAP = {
    "noun": "Miscellaneous Nouns",
    "verb": "Verbs",
    "adjective": "Adjectives",
}
_DEFAULT_CATEGORY = "Miscellaneous Nouns"


def _part_of_speech(form: str) -> str:
    for prefix, pos in _POS_MAP:
        if form.startswith(prefix):
            return pos
    return "other"


def _category(pos: str) -> str:
    return _CATEGORY_MAP.get(pos, _DEFAULT_CATEGORY)


def _strip_bracket(s: str) -> tuple[str, str | None]:
    """Return (text with any trailing ' [classifier]' suffix removed,
    the bracket's inner content or None)."""
    m = _BRACKET_SUFFIX.search(s)
    if m is None:
        return s.strip(), None
    inner = s[m.start():].strip()[1:-1].strip()
    return s[: m.start()].strip(), (inner or None)


def _audio_filename(field: str) -> str | None:
    m = _SOUND_TAG.search(field)
    return m.group("name") if m else None


def _read_notes(db_path: Path) -> list[list[str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "select flds from notes order by id asc").fetchall()
    finally:
        conn.close()
    return [row[0].split("\x1f") for row in rows]


def _read_version(db_path: Path) -> str:
    import datetime

    conn = sqlite3.connect(str(db_path))
    try:
        (crt,) = conn.execute("select crt from col").fetchone()
    finally:
        conn.close()
    return datetime.datetime.fromtimestamp(
        crt, tz=datetime.timezone.utc).strftime("%Y-%m-%d")


def _copy_audio(src_dir: Path, media_map: dict[str, str], filename: str,
                dest_dir: Path, missing: set[str]) -> None:
    dest = dest_dir / filename
    if dest.exists():
        return
    reverse = _copy_audio._reverse  # type: ignore[attr-defined]
    key = reverse.get(filename)
    src = src_dir / key if key is not None else None
    if src is None or not src.is_file():
        missing.add(filename)
        return
    dest.write_bytes(src.read_bytes())


def import_apkg(src_dir: Path, out_dir: Path) -> dict:
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    media_map: dict[str, str] = json.loads((src_dir / "media").read_text())
    _copy_audio._reverse = {v: k for k, v in media_map.items()}  # type: ignore[attr-defined]

    notes_dir = out_dir / "notes"
    audio_dir = out_dir / "media" / "audio"
    images_dir = out_dir / "media" / "images"
    for d in (notes_dir, audio_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)
    (images_dir / "placeholder.png").write_bytes(_PLACEHOLDER_PNG)

    freq = FileFrequencyList()
    picture_words: list[dict] = []
    sentences: list[dict] = []
    skipped_empty = 0
    skipped_no_audio = 0
    ipa_ok = 0
    missing_audio: set[str] = set()

    raw_notes = _read_notes(src_dir / "collection.anki2")
    ordinal = 0
    for flds in raw_notes:
        (word_eng, word_tha, word_phonetic, audio_field, form,
         sentence_tha, _sentence_phonetic, _sentence_eng) = flds
        if not word_tha.strip():
            skipped_empty += 1
            continue
        audio_name = _audio_filename(audio_field)
        if audio_name is None:
            # audio is a required schema field; this deck has no such
            # notes in practice (verified: all 1000 carry a [sound:...]
            # tag), but a note without one can't become a valid note.
            skipped_no_audio += 1
            continue
        ordinal += 1

        thai, classifier = _strip_bracket(word_tha)
        phon_stripped, _ = _strip_bracket(word_phonetic)
        pos = _part_of_speech(form)
        category = _category(pos)

        _copy_audio(src_dir, media_map, audio_name, audio_dir, missing_audio)

        rank = freq.rank(thai)
        if rank is None:
            rank = 90000 + ordinal

        ipa = paiboon_to_ipa(phon_stripped)
        if ipa is not None:
            ipa_ok += 1

        word_id = f"w{ordinal:04d}"
        picture_words.append({
            "id": word_id,
            "thai": thai,
            "image": "images/placeholder.png",
            "audio": {"file": f"audio/{audio_name}", "source": "native",
                      "speaker": "thai1000"},
            "frequency_rank": rank,
            "category": category,
            "part_of_speech": pos,
            **({"classifier": classifier} if classifier else {}),
            **({"ipa": ipa} if ipa else {}),
        })
        sentences.append({
            "id": f"s{ordinal:04d}",
            "kind": "new_word",
            "thai": sentence_tha.strip(),
            "target": thai,
            "audio": {"file": f"audio/{audio_name}", "source": "native",
                      "speaker": "thai1000"},
        })

    (out_dir / "deck.yaml").write_text(yaml.safe_dump({
        "name": "thai1000-import",
        "version": _read_version(src_dir / "collection.anki2"),
        "stage_plan": {"phases": ["words", "sentences"]},
    }, allow_unicode=True))
    (notes_dir / "picture_words.yaml").write_text(
        yaml.safe_dump(picture_words, allow_unicode=True))
    (notes_dir / "sentences.yaml").write_text(
        yaml.safe_dump(sentences, allow_unicode=True))
    (notes_dir / "minimal_pairs.yaml").write_text(yaml.safe_dump([]))
    (notes_dir / "spelling_sound.yaml").write_text(yaml.safe_dump([]))

    total_kept = len(picture_words)
    return {
        "total_notes": len(raw_notes),
        "skipped_empty_word_tha": skipped_empty,
        "skipped_no_audio": skipped_no_audio,
        "kept": total_kept,
        "ipa_converted": ipa_ok,
        "ipa_coverage_pct": round(100 * ipa_ok / total_kept, 1) if total_kept else 0.0,
        "missing_audio_files": sorted(missing_audio),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        print("usage: import_apkg.py <extracted-apkg-dir> <out-deck-dir>",
              file=sys.stderr)
        return 2
    report = import_apkg(Path(argv[0]), Path(argv[1]))
    print(f"read {report['total_notes']} notes; "
          f"skipped {report['skipped_empty_word_tha']} (empty word_tha); "
          f"kept {report['kept']}")
    print(f"paiboon->ipa conversion: {report['ipa_converted']}/{report['kept']} "
          f"({report['ipa_coverage_pct']}%)")
    if report["missing_audio_files"]:
        print(f"missing source audio for: {report['missing_audio_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
