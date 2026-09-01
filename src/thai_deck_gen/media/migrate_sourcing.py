"""One-shot fold of the old image stores into sourcing records.

`work/image_review.yaml` holds the failed query history and
`work/candidates/*/candidates.yaml` the verdicts on candidates already
judged. Both are expensive to regenerate -- the second cost 2,178 judgments.
`work/image_query_proposals.yaml` is not migrated: it holds one stale
suggestion per word and the record supersedes it.

Dead code the day after it runs.
"""

from pathlib import Path

import yaml

from thai_deck_gen.media.sourcing import (Attempt, Candidate, Decision,
                                          SourcingLog)


def migrate(deck_root: Path, subjects: dict[str, tuple[str, str]],
            today: str) -> int:
    """Fold the old stores into records. Returns the number of subjects written.

    `subjects` maps note id to (family, subject); the old stores key on note
    id, which outlives the notes, so an entry the caller cannot name is
    skipped rather than inventing a subject for it.

    Safe to re-run: a subject that already has attempts or a decision is left
    alone, since the first run may have been killed part way.
    """
    deck_root = Path(deck_root)
    log = SourcingLog.load(deck_root)
    written = 0

    review = deck_root / "work" / "image_review.yaml"
    items: dict[str, dict] = {}
    if review.exists():
        loaded = yaml.safe_load(review.read_text(encoding="utf-8")) or {}
        items = {it["note_id"]: it for it in loaded.get("items", [])
                 if "note_id" in it}

    for note_id, (family, subject) in subjects.items():
        record = log.get(family, subject)
        if record.attempts or record.decision is not None:
            continue                      # already migrated

        item = items.get(note_id, {})
        queries = item.get("queries") or []
        rubric = item.get("rubric", "")
        corpora = tuple(item.get("tried", ()))
        candidates, accepted = _candidates_for(deck_root, note_id)

        if not queries and not candidates:
            continue                      # nothing was recorded about it

        # The old stores recorded the query set and the candidate pool
        # separately, with no link between them. The pool is attached to the
        # first query rather than duplicated across all of them.
        for index, query in enumerate(queries or [""]):
            log.record_attempt(family, subject, Attempt(
                query=query, query_source="phrase", corpora=corpora,
                rubric=rubric,
                candidates=tuple(candidates) if index == 0 else (),
                dated=today))
        if accepted is not None:
            log.record_decision(family, subject, Decision(
                kind="judge-accepted", file=accepted.file, reason=None,
                dated=today))
        written += 1

    return written


def _candidates_for(deck_root: Path,
                    note_id: str) -> tuple[list[Candidate], Candidate | None]:
    path = deck_root / "work" / "candidates" / note_id / "candidates.yaml"
    if not path.exists():
        return [], None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return [], None
    rows = loaded.get("candidates", []) if isinstance(loaded, dict) else loaded
    candidates: list[Candidate] = []
    accepted: Candidate | None = None
    for row in rows:
        candidate = Candidate(
            url=row.get("url", ""), source=row.get("source", ""),
            license=row.get("license"), file=row.get("file", ""),
            passed=bool(row.get("passed")),
            failed_rules=tuple(row.get("failed_rules", ())))
        candidates.append(candidate)
        if row.get("accepted"):
            accepted = candidate
    return candidates, accepted
