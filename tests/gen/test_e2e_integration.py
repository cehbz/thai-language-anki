"""End-to-end integration test for the full generator pipeline.

Real pythainlp/tone/tokenizer ports (via build_context(nlp=True)) and the
real evaluator subprocess (thai-deck-eval, via orchestrator.run_eval) drive
the whole loop; only the LLM and the media channels (Forvo/TTS/thai1000/
image search) are faked, since those need network access or paid services.

The LLM stub always echoes back the sentence producer's own target word or
grammar marker as the "sentence" -- a real, single-token Thai word that
trivially survives the real tokenizer's one-unknown-token check because it
*is* the known/target token, with zero risk of the real segmenter splitting
a synthesized multi-word sentence in a way that trips the checker.
"""
import io
import json
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from thai_deck_eval.data_io import FileFrequencyList
from thai_deck_eval.model.deck import load_deck
from thai_deck_gen.compiler.build import GUID_FAMILY, compile_deck, note_guid
from thai_deck_gen.compiler.ordering import intro_order
from thai_deck_gen.config import GenConfig
from thai_deck_gen.context import build_context
from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import pending_audio, pending_images
from thai_deck_gen.orchestrator import generate, run_eval
from thai_deck_gen.report import parse_report

from tests.gen.helpers_apkg import read_apkg

pytestmark = pytest.mark.integration

REPO_DATA = Path(__file__).parents[2] / "data"

# 10 real, high-frequency Thai words (all present in data/frequency_th.txt,
# so fill_words can rank them) spanning 9 of data/categories.yaml's
# categories, with classifiers on every noun.
WORD_LIST_YAML = """\
- {thai: "ดี", gloss: "good", category: "Adjectives", part_of_speech: "adjective"}
- {thai: "เพื่อน", gloss: "friend", category: "People", part_of_speech: "noun", classifier: "คน"}
- {thai: "บ้าน", gloss: "house", category: "Home", part_of_speech: "noun", classifier: "หลัง"}
- {thai: "น้ำ", gloss: "water", category: "Beverages", part_of_speech: "noun", classifier: "แก้ว"}
- {thai: "รถ", gloss: "car", category: "Transportation", part_of_speech: "noun", classifier: "คัน"}
- {thai: "กิน", gloss: "eat", category: "Verbs", part_of_speech: "verb"}
- {thai: "หนังสือ", gloss: "book", category: "Miscellaneous Nouns", part_of_speech: "noun", classifier: "เล่ม"}
- {thai: "หมา", gloss: "dog", category: "Animals", part_of_speech: "noun", classifier: "ตัว"}
- {thai: "แมว", gloss: "cat", category: "Animals", part_of_speech: "noun", classifier: "ตัว"}
- {thai: "ข้าว", gloss: "rice", category: "Food", part_of_speech: "noun", classifier: "จาน"}
"""

# Curated fallback pairs (real minimal pairs, IPA verified against real
# pythainlp thaig2p output) for two contrasts our tiny 10-word lexicon can't
# supply on its own -- exactly pair_seeds.yaml's documented purpose. Same
# pair as tests/helpers.py's golden fixture (already proven to pass the
# real-ports gate in tests/test_cli_integration.py).
PAIR_SEEDS_YAML = """\
"tone:low-rising":
  - ["ขาว", "kʰaːw˨˩˦"]
  - ["ข่าว", "kʰaːw˨˩"]
"aspiration:velar":
  - ["ไก่", "kaj˨˩"]
  - ["ไข่", "kʰaj˨˩"]
"""


class StubSentenceLlm:
    """Echoes the target word/marker straight out of the producer's own
    prompt template, so every completion is real Thai vocabulary the real
    tokenizer already knows as a single token."""

    _TARGET_RE = re.compile(r"Target Thai word or grammar marker: (\S+)")

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, producer: str, prompt_version: str, prompt: str) -> str:
        self.prompts.append(prompt)
        m = self._TARGET_RE.search(prompt)
        assert m, f"no target line found in prompt: {prompt!r}"
        return m.group(1)


def _tiny_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(210, 210, 210)).save(buf, format="JPEG")
    return buf.getvalue()


def _tiny_mp3_bytes() -> bytes:
    # A single MPEG-1 Layer III frame header (0xFFFB90) followed by zeroed
    # payload: a structurally valid (silent) mp3 frame.
    return b"\xff\xfb\x90\x00" + b"\x00" * 96


def _audio_index(deck):
    """(note_id, audio path) -> the live Audio object, so pending_audio's
    reports (which carry only the path, not the object) can be mutated."""
    index = {}
    for family, note in deck.all_notes():
        if family == "minimal_pair":
            for member in note.members:
                index[(note.id, member.audio.file)] = member.audio
        else:
            index[(note.id, note.audio.file)] = note.audio
    return index


def _fill_pending_media(deck) -> None:
    """Stand-in for Forvo/TTS/thai1000/image-search: write tiny valid media
    files for every still-pending need and clear the 'pending' speaker
    placeholder, so the mechanical stage's media-missing check is satisfied."""
    jpeg, mp3 = _tiny_jpeg_bytes(), _tiny_mp3_bytes()
    audio_by_key = _audio_index(deck)

    pair_member_k = 0
    for need in pending_audio(deck):
        path = deck.root / "media" / need.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(mp3)
        audio = audio_by_key[(need.note_id, need.path)]
        if need.family == "minimal_pair":
            audio.speaker = f"fake:{pair_member_k % 3}"
            pair_member_k += 1
        else:
            audio.speaker = "fake:native"

    for need in pending_images(deck):
        path = deck.root / "media" / need.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)


@pytest.fixture
def _clean_media_env(monkeypatch):
    monkeypatch.delenv("THAI_DECK_GEN_FAKE", raising=False)


def test_end_to_end_generation(tmp_path, _clean_media_env):
    deck_root = tmp_path / "deck"
    deck = new_deck(deck_root, "e2e-test", ["sounds", "words", "sentences"])
    write_deck(deck)

    data_dir = tmp_path / "data"
    shutil.copytree(REPO_DATA, data_dir)
    (data_dir / "word_list_th.yaml").write_text(WORD_LIST_YAML, encoding="utf-8")
    (data_dir / "pair_seeds.yaml").write_text(PAIR_SEEDS_YAML, encoding="utf-8")

    config = GenConfig(lexicon_top_n=50, sentence_base=5, max_iterations=5,
                       images=False)
    llm = StubSentenceLlm()
    ctx = build_context(deck_root, data_dir, llm, nlp=True, config=config)

    # Round 1: content producers converge (pairs/spelling/words/sentences);
    # media stays pending, so the report keeps failing on media-missing.
    generate(deck, ctx)

    assert len(deck.picture_words) == 10
    # >= 2: the two curated pair_seeds contrasts are guaranteed; the real
    # lexicon (word list + top-50 frequency words) may supply a few more
    # real minimal pairs on its own (e.g. tone/vowel_quality contrasts
    # incidentally satisfied by common frequency words).
    assert len(deck.minimal_pairs) >= 2
    assert len(deck.sentences) == 16          # 10 new_word + 6 grammar points
    assert llm.prompts, "sentence producer never called the stub llm"

    # Fake media fillers stand in for Forvo/TTS/thai1000/image search.
    _fill_pending_media(deck)
    write_deck(deck)

    # Round 2: with media satisfied, generate() should settle cleanly (no
    # further content to add, no progress -> stop) with a passing report.
    generate(deck, ctx)

    final_report = json.loads((deck_root / ".last-report.json").read_text())
    judge_only = all(f["rule"].startswith("judge/") for f in final_report["findings"])
    assert final_report["gate"] == "pass" or judge_only, final_report["findings"]

    metrics = {m["name"]: m for m in final_report["metrics"]}
    assert metrics["coverage/minimal_pairs"]["value"] > 0

    # The deck loads cleanly through the real schema loader.
    loaded = load_deck(deck_root)
    assert len(loaded.picture_words) == 10
    assert len(loaded.minimal_pairs) == len(deck.minimal_pairs)

    # Compile to .apkg and check it's non-empty and due-ordered.
    gaps = parse_report(final_report, data_dir / "contrasts.yaml")
    freq = FileFrequencyList(data_dir / "frequency_th.txt")
    manifest = Manifest.load(deck_root)
    out = tmp_path / "e2e-test.apkg"
    compile_deck(loaded, manifest, out, freq, gaps.pair_by_note,
                base=config.sentence_base)

    packaged = read_apkg(out)
    assert packaged["notes"]
    assert packaged["cards"]

    due_by_nid = {}
    for card in packaged["cards"]:
        due_by_nid.setdefault(card["nid"], set()).add(card["due"])
    for nid, dues in due_by_nid.items():
        assert len(dues) == 1, "all cards of one note should share a due"
    due_by_guid = {n["guid"]: next(iter(due_by_nid[n["id"]]))
                  for n in packaged["notes"]}

    order = intro_order(loaded, freq, base=config.sentence_base)
    expected_guids: list[str] = []
    for family, note in order:
        if family == "minimal_pair":
            expected_guids += [note_guid(GUID_FAMILY[family], f"{note.id}_{k}")
                               for k in range(len(note.members))]
        else:
            expected_guids.append(note_guid(GUID_FAMILY[family], note.id))

    assert set(expected_guids) == set(due_by_guid)
    dues_in_intro_order = [due_by_guid[g] for g in expected_guids]
    assert dues_in_intro_order == sorted(dues_in_intro_order)
