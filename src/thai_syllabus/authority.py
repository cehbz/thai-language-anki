"""Authority domain data (spec 1 section 4): the per-role authority order
and the (need kind, subject kind) -> role map. Assessment (assessor.py) and
the derivations (spec 3) consume these values; they do not define them.
"""
from __future__ import annotations

__all__ = ["AUTHORITY_ORDER", "ROLE_FOR_KIND", "ROLE_FOR_SENTENCE_SUBJECT", "role_for"]


# Per role, backends ordered most- to least-authoritative. Not a single
# global ranking -- authority is per (backend, role): the learner is final
# on fit/quality/waivers but unqualified on tone correctness, where
# mechanical is ground truth and listener ranks only once calibrated
# (absent from this table until a deployment's providers.yaml supplies a
# measured rank, which is why "listener" does not appear in the
# recording-for-word row below: an uncalibrated listener contributes
# nothing to current_best).
AUTHORITY_ORDER: dict[str, tuple[str, ...]] = {
    "picture-for-word": ("learner", "judge"),
    "scene-for-sentence": ("learner", "judge"),
    "sentence-for-target": ("learner", "judge"),
    "finding-waiver": ("learner",),
    "card-flag": ("learner",),
    "recording-for-word": ("mechanical", "judge"),  # learner may flag, never outrank
    "recording-for-sentence": ("mechanical", "judge"),
    "rendition-for-pair": ("rendition",),   # the one-speaker check
}


# Need kind -> the judged Assess role that kind's fit verdict is asked
# under, for the kind's own natural subject (a word's picture, a word's
# recording, a pair's rendition, a grapheme's keyword).
ROLE_FOR_KIND: dict[str, str] = {
    "picture": "picture-for-word",
    "recording": "recording-for-word",
    "rendition": "rendition-for-pair",
    "sentence": "sentence-for-target",
    "grapheme-keyword": "grapheme-keyword-for-grapheme",
}


# The same artifact kinds asked about a SENTENCE: a sentence's picture is
# a scene, a sentence's recording is judged as a sentence reading. The
# artifact kinds themselves are unchanged ("picture", "recording") -- the
# subject is what differs, and it is the subject that changes the role.
ROLE_FOR_SENTENCE_SUBJECT: dict[str, str] = {
    "picture": "scene-for-sentence",
    "recording": "recording-for-sentence",
}


def role_for(kind: str, subject_kind: str = "word") -> str:
    """The Assess role a need's fit verdict is asked under. Raises KeyError
    naming `kind` when no role is mapped.
    """
    if subject_kind == "sentence" and kind in ROLE_FOR_SENTENCE_SUBJECT:
        return ROLE_FOR_SENTENCE_SUBJECT[kind]
    return ROLE_FOR_KIND[kind]
