"""Import the thai-language.com pages already fetched into the deck's
record, once.

    uv run python scripts/import_paiboon.py <deck-dir> [<html-dir>]

<html-dir> (default data/paiboon/html, gitignored: the site's content is
copyrighted) holds a search page per form, <form>.html, and the entry
pages under id/<N>.html. Each form without a row on record gets the row
`thailanguagecom.ThaiLanguageCom` would have written: the raw IPA strings
and entry ids. A form whose search page is cached but some entry page is
not is left for the live backend. Runs under the deck's `import` writing
command (spec 2 section 6). Reports forms imported (with IPA, with no
entry), skipped (already on record) and left incomplete.
"""
import sys
from pathlib import Path

from thai_syllabus.cachekeys import DictionaryKey
from thai_syllabus.safety import writing_command
from thai_syllabus.store import SyllabusDb
from thai_syllabus.thailanguagecom import SOURCE, entry_row


def import_pages(db: SyllabusDb, html_dir: Path) -> dict[str, int]:
    report = dict.fromkeys(("imported", "with_ipa", "no_entry", "skipped", "incomplete"), 0)

    def entry_page(entry: str) -> str | None:
        file = html_dir / "id" / f"{entry}.html"
        return file.read_text(encoding="utf-8") if file.exists() else None

    for file in sorted(html_dir.glob("*.html")):
        form = file.stem
        key = DictionaryKey(source=SOURCE, form=form)
        if db.latest("provide", SOURCE, key) is not None:
            report["skipped"] += 1
            continue
        answer = entry_row(file.read_text(encoding="utf-8"), form, entry_page)
        if answer is None:
            report["incomplete"] += 1
            continue
        db.append(port="provide", backend=SOURCE, key=key, subject=f"dictionary:{form}",
                  question={"form": form}, answer=answer, cost=0.0)
        report["imported"] += 1
        if not answer["entries"]:
            report["no_entry"] += 1
        elif any(e["ipa"] for e in answer["entries"]):
            report["with_ipa"] += 1
    return report


def main(deck: str, html_dir: str = "data/paiboon/html") -> None:
    with writing_command(Path(deck), "import"):
        report = import_pages(SyllabusDb(Path(deck) / "syllabus.db"), Path(html_dir))
    print(f"{report['imported']} forms imported "
          f"({report['with_ipa']} with IPA, {report['no_entry']} with no entry), "
          f"{report['skipped']} skipped (already on record), "
          f"{report['incomplete']} left for the live backend (entry page not cached)")


if __name__ == "__main__":
    main(*sys.argv[1:])
