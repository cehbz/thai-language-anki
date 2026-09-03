"""Fakes for testing the Syllabus aggregate: no pythainlp, no anthropic."""
from thai_syllabus.rules import Finding


class FakeTokenizer:
    """Returns pre-declared tokens per exact sentence text; falls back to
    treating the whole text as one token (fine for tests that only care
    about a sentence's word-level structure via explicit token lists).
    """
    def __init__(self, tokens_by_text: dict[str, list[str]] | None = None):
        self._map = dict(tokens_by_text or {})

    def tokens(self, text: str) -> list[str]:
        return self._map.get(text, [text])


class FakeAssessmentReader:
    def __init__(self, verdicts: dict[tuple[str, str, str | None], bool] | None = None,
                waived: set[tuple[str, str, str | None]] | None = None):
        self._verdicts = dict(verdicts or {})
        self._waived = set(waived or set())

    def verdict(self, rule_id: str, note_id: str,
                artifact_sha: str | None = None,
                rubric: str | None = None) -> bool | None:
        # `rubric` is accepted (matching the real AssessmentReader.verdict
        # signature -- spec 4's merged key convention) but not part of this
        # fake's lookup key: FakeAssessmentReader tests report()'s logic in
        # isolation from store.py's actual key mechanics.
        return self._verdicts.get((rule_id, note_id, artifact_sha))

    def is_waived(self, finding: Finding) -> bool:
        return finding.identity() in self._waived


class FakeMediaIndex:
    def __init__(self, pictures: set[str] | None = None,
                recording_speakers: dict[str, frozenset[str]] | None = None,
                rendition_speakers: dict[str, frozenset[str]] | None = None):
        self._pictures = set(pictures or set())
        self._recordings = dict(recording_speakers or {})
        self._renditions = dict(rendition_speakers or {})

    def has_picture(self, word) -> bool:
        return word in self._pictures

    def recording_speakers(self, word) -> frozenset[str]:
        return self._recordings.get(word, frozenset())

    def rendition_speakers(self, pair_confusion) -> frozenset[str]:
        return self._renditions.get(pair_confusion, frozenset())
