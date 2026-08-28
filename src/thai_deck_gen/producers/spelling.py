import re
import yaml
from pathlib import Path
from thai_deck_eval.lang.tone import CONSONANT_CLASS, ConsClass
from thai_deck_eval.model.deck import Deck
from thai_deck_eval.model.notes import Audio, SpellingSoundNote
from thai_deck_gen.producers import ProducerResult
from thai_deck_gen.report import Gaps


def missing_patterns(deck: Deck, targets_path: Path) -> list[str]:
    targets = yaml.safe_load(targets_path.read_text())
    all_patterns = []
    for section in ("consonants", "vowels", "tone_marks"):
        all_patterns.extend(targets.get(section, []))
    existing = {n.pattern for n in deck.spelling_sound}
    return [p for p in all_patterns if p not in existing]


def _pattern_matches(pattern: str, word: str) -> bool:
    """Check if a pattern matches a word.

    Patterns use '-' as a consonant placeholder (e.g., '-ะ', 'เ-', 'เ-อ').
    Replace each '-' with [ก-ฮ] (Thai consonant range) and escape other chars.
    """
    # Build regex by replacing - with Thai consonant class, escaping others
    parts = []
    for char in pattern:
        if char == '-':
            parts.append('[ก-ฮ]')
        else:
            parts.append(re.escape(char))
    regex = ''.join(parts)
    return re.search(regex, word) is not None


def fill_spelling(gaps: Gaps, deck: Deck, ctx) -> ProducerResult:
    result = ProducerResult()
    targets = yaml.safe_load(ctx.targets_path.read_text())
    existing_ids = {n.id for n in deck.spelling_sound}
    existing_patterns = {n.pattern for n in deck.spelling_sound}

    sections = {
        "consonants": "consonant",
        "vowels": "vowel",
        "tone_marks": "tone_mark"
    }

    for section, pattern_kind in sections.items():
        for pattern in targets.get(section, []):
            if pattern in existing_patterns:
                continue

            note_id = f"sp-{pattern}"
            if note_id in existing_ids:
                continue

            consonant_class = None
            if pattern_kind == "consonant":
                consonant_class = CONSONANT_CLASS.get(pattern)
                if consonant_class:
                    consonant_class = consonant_class.value

            example_word = None
            for word_entry in ctx.word_list:
                if _pattern_matches(pattern, word_entry.thai):
                    example_word = word_entry.thai
                    break

            if not example_word:
                result.blocked.append(pattern)
                continue

            note = SpellingSoundNote(
                id=note_id,
                pattern=pattern,
                pattern_kind=pattern_kind,
                consonant_class=consonant_class,
                example_word=example_word,
                audio=Audio(
                    file=f"audio/spelling_sound/sp-{pattern}.mp3",
                    source="native",
                    speaker="pending"
                ),
                image=f"images/sp-{pattern}.jpg"
            )
            deck.spelling_sound.append(note)
            existing_ids.add(note_id)
            result.added += 1

    return result
