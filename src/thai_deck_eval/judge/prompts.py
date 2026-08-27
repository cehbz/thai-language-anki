PROMPT_VERSION = "1"

_SCHEMA = ('Return ONLY JSON: {"verdicts": [{"rule": "<id>", "passed": true|false, '
           '"confidence": 0.0-1.0, "rationale": "<one sentence>"}]} — one entry '
           "per rule listed below.")

SENTENCE_RULES = {
    "judge/unnatural-sentence":
        "Is the Thai sentence natural and grammatically correct — something a "
        "native speaker would actually produce?",
    "judge/definition-not-monolingual":
        "If a definition is given, is it entirely in Thai and accurate for the "
        "target word? Pass if no definition.",
    "judge/gloss-inaccurate":
        "If an English gloss is given, does it correctly translate the target? "
        "Pass if no gloss.",
}

PICTURE_RULES = {
    "judge/image-irrelevant":
        "Does the image plausibly depict or relate to the word?",
    "judge/image-embedded-text":
        "Pass only if the image contains NO English or romanized-Thai text.",
    "judge/classifier-wrong":
        "If a classifier is given, is it the conventional classifier for this "
        "noun? Pass if none given.",
}

def _rules_block(rules: dict[str, str]) -> str:
    return "\n".join(f"- {rid}: {rubric}" for rid, rubric in rules.items())

def build_sentence_prompt(note) -> str:
    return (f"You are evaluating a Thai flashcard for a Fluent Forever deck.\n"
            f"Sentence: {note.thai}\nTarget word: {note.target}\n"
            f"Definition: {note.definition or '(none)'}\n"
            f"Gloss: {note.gloss or '(none)'}\n\nJudge these rules:\n"
            f"{_rules_block(SENTENCE_RULES)}\n\n{_SCHEMA}")

def build_picture_prompt(note) -> str:
    return (f"You are evaluating a Thai picture-word flashcard (image attached "
            f"or at path).\nWord: {note.thai}\nCategory: {note.category}\n"
            f"Part of speech: {note.part_of_speech}\n"
            f"Classifier: {note.classifier or '(none)'}\n\nJudge these rules:\n"
            f"{_rules_block(PICTURE_RULES)}\n\n{_SCHEMA}")
