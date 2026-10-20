"""Strip the stored pronunciations from a deck's words.yaml, once (spec 2
r23: a Word's pronunciation is its form's settlement over the readings
on record, never curated).

    uv run python scripts/strip_pron.py <deck-dir>

Every row is kept; its `pron` (with its corroboration) goes. Prints, per
word whose old reading differs from its form's settled one, the id, the
form and the two readings rendered as IPA ("(none)" where the record
reads nothing), then the counts: same, differs, now none. Runs under the
deck's `strip-pron` writing command (spec 2 section 6).
"""
import sys
from pathlib import Path

import yaml

from thai_syllabus import ipa
from thai_syllabus.curated import load_words, save_words
from thai_syllabus.entities import Pronunciation, Syllable
from thai_syllabus.rulebook import PRONUNCIATION_RUBRICS
from thai_syllabus.safety import writing_command
from thai_syllabus.settlement import settlement_of
from thai_syllabus.store import SyllabusDb


def _old_reading(pron: dict | None) -> str:
    if not pron:
        return ""
    return ipa.render(Pronunciation(
        syllables=tuple(Syllable(segments=tuple(s["segments"]), vowel_length=s["vowel_length"],
                                 tone=s["tone"]) for s in pron["syllables"]),
        posterior=0.0, spoken_dissent=False))


def strip(deck: Path) -> tuple[list[str], dict[str, int]]:
    path = deck / "curated" / "words.yaml"
    old = {row["id"]: row.get("pron")
           for row in yaml.safe_load(path.read_text(encoding="utf-8")) or []}
    rows = load_words(path)
    settled = settlement_of(SyllabusDb(deck / "syllabus.db"), [w for w, _ in rows],
                            current_rubric=PRONUNCIATION_RUBRICS)
    lines: list[str] = []
    counts = dict.fromkeys(("same", "differs", "now none"), 0)
    for w, _category in rows:
        before, after = _old_reading(old.get(w.id)), ipa.render(settled.reading(w.thai))
        if before == after:
            counts["same"] += 1
            continue
        counts["now none" if not after else "differs"] += 1
        lines.append(f"{w.id}\t{w.thai}\t{before or '(none)'}\t->\t{after or '(none)'}")
    save_words(path, rows)
    return lines, counts


def main(deck: str) -> None:
    with writing_command(Path(deck), "strip-pron"):
        lines, counts = strip(Path(deck))
    for line in lines:
        print(line)
    print(f"{counts['same']} same, {counts['differs']} differ, "
          f"{counts['now none']} now none")


if __name__ == "__main__":
    main(*sys.argv[1:])
