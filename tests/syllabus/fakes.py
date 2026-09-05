"""Fakes for testing the Syllabus aggregate: no pythainlp, no anthropic."""
from thai_syllabus.ports import Answer
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
    """`verdicts` is keyed the way tests read (rule_id, note_id,
    artifact_sha) -> bool, not by the cachekeys.JudgeKey report() actually
    passes to verdict() -- this fake tests report()'s logic in isolation
    from store.py's actual key mechanics, matching a JudgeKey by its role
    (rule_id) and identity (artifact_sha, falling back to note_id).
    """
    def __init__(self, verdicts: dict[tuple[str, str, str | None], bool] | None = None,
                waived: set[tuple[str, str, str | None]] | None = None):
        self._verdicts = dict(verdicts or {})
        self._waived = set(waived or set())

    def verdict(self, backend: str, key) -> Answer | None:
        role = getattr(key, "role", None)
        identity = getattr(key, "identity", None)
        for (rule_id, note_id, artifact_sha), value in self._verdicts.items():
            if rule_id == role and identity == (artifact_sha or note_id):
                return Answer(port="assess", backend=backend, key_sha="", key="",
                             subject=note_id, question={}, answer={"value": value},
                             cost=0.0, ts=0)
        return None

    def is_waived(self, finding: Finding) -> bool:
        return finding.identity() in self._waived


class FakeMediaIndex:
    def __init__(self, pictures: set[str] | None = None,
                recording_speakers: dict[str, frozenset[str]] | None = None,
                rendition_speakers: dict[str, frozenset[str]] | None = None,
                recording_provenance: dict[str, dict] | None = None,
                rendition_provenance: dict[str, tuple] | None = None,
                speakers: dict[str, tuple] | None = None):
        self._pictures = set(pictures or set())
        self._recordings = dict(recording_speakers or {})
        self._renditions = dict(rendition_speakers or {})
        self._recording_provenance = dict(recording_provenance or {})
        self._rendition_provenance = dict(rendition_provenance or {})
        self._speakers = dict(speakers or {})

    def has_picture(self, word) -> bool:
        return word in self._pictures

    def recording_speakers(self, word) -> frozenset[str]:
        return self._recordings.get(word, frozenset())

    def rendition_speakers(self, pair_confusion) -> frozenset[str]:
        return self._renditions.get(pair_confusion, frozenset())

    def recording_provenance(self, word) -> dict | None:
        return self._recording_provenance.get(word)

    def rendition_provenance(self, pair_id) -> tuple:
        return self._rendition_provenance.get(pair_id, ())

    def picture_sha(self, word) -> str | None:
        return f"sha-{word}" if word in self._pictures else None

    def speakers_of(self, corpus) -> tuple:
        return self._speakers.get(corpus, ())
