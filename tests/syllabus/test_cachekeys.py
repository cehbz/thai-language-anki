"""Tests for cachekeys.py (spec 3 section 1 "Key"): one key dataclass per
Assess backend, encode()'s canonical string, sha()'s 16-hex truncation,
and preference_identity()'s order-independence.
"""
from thai_syllabus.cachekeys import (
    BatchMarkerKey,
    CacheKey,
    DrillKey,
    JudgeKey,
    LearnerKey,
    LearnerNoteKey,
    MechanicalKey,
    ReverifyKey,
    WaiverKey,
    preference_identity,
    sha,
)


def test_sha_truncates_to_16_hex_by_default():
    digest = sha("some rubric text")
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha_is_deterministic():
    assert sha("x") == sha("x")
    assert sha("x") != sha("y")


def test_preference_identity_is_order_independent():
    assert preference_identity(["b", "a"]) == preference_identity(["a", "b"])
    assert preference_identity(["a", "b"]) != preference_identity(["a", "c"])


def test_judge_key_encodes_rubric_identity_role():
    key = JudgeKey(rubric_sha="abc123", identity="deadbeef", role="picture-for-word")
    assert key.encode() == "judge:abc123:deadbeef:picture-for-word"
    assert key.kind == "JudgeKey"
    assert isinstance(key, CacheKey)


def test_learner_key_falls_back_to_dash_with_no_artifact():
    assert LearnerKey(artifact_sha=None, role="card-flag").encode() == "learner:-:card-flag"
    assert LearnerKey(artifact_sha="sha1", role="card-flag").encode() == "learner:sha1:card-flag"


def test_learner_note_key_encodes_anchor_and_text_sha():
    key = LearnerNoteKey(anchor="n1", text_sha=sha("hello"))
    assert key.encode() == f"learner-note:n1:{sha('hello')}"


def test_drill_key_encodes_pair_and_confusion():
    assert DrillKey(pair_id="p1", confusion="c1").encode() == "learner:drill:p1:c1"


def test_reverify_key_falls_back_to_anchor_with_no_artifact():
    key = ReverifyKey(artifact_sha=None, anchor="w1", role="recording-for-word")
    assert key.encode() == "learner:reverify:w1:recording-for-word"
    key_with_sha = ReverifyKey(artifact_sha="s1", anchor="w1", role="recording-for-word")
    assert key_with_sha.encode() == "learner:reverify:s1:recording-for-word"


def test_waiver_key_falls_back_to_dash_with_no_artifact():
    assert (WaiverKey(rule_id="r", note_id="n", artifact_sha=None).encode()
           == "waiver:r:n:-")
    assert (WaiverKey(rule_id="r", note_id="n", artifact_sha="s").encode()
           == "waiver:r:n:s")


def test_mechanical_key_is_parameter_explicit():
    key = MechanicalKey(check="duration", params="0.2-5.0", artifact_sha="deadbeef")
    assert key.encode() == "mech:duration:0.2-5.0:deadbeef"


def test_batch_marker_key_encodes_the_batch_id():
    assert BatchMarkerKey(batch_id="batch-1").encode() == "batch-marker:batch-1"


def test_keys_are_hashable_and_equal_by_value():
    a = JudgeKey(rubric_sha="x", identity="y", role="z")
    b = JudgeKey(rubric_sha="x", identity="y", role="z")
    assert a == b and hash(a) == hash(b)
    assert {a, b} == {a}
