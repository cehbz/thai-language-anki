"""sha(): the one hashing primitive spec 3's readable cache keys use.

Spec 3 section 2: "Keys are canonical readable strings; sha() wraps only
components too large or binary to inspect (prompts, rubric texts, artifact
bytes)." Every other component of a key (a word, a backend name, a role, a
voice id) goes into the key verbatim -- only these get sha()'d. Truncated
to 16 hex chars (64 bits): collision risk is irrelevant here (a false cache
hit needs both a truncation collision AND the same port/backend/subject,
astronomically unlikely for the corpus sizes this project has).
"""
import hashlib


def sha(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
