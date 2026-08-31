"""A fake world for the generator: everything outside the process, faked.

Search corpora, image downloads, the LLM, the judge, Forvo, TTS. The deck
directory and every file the generator writes are real, so a test can assert
on the artifact rather than on calls.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from thai_deck_eval.judge.core import Verdict
from thai_deck_eval.model.notes import Audio, PictureWordNote
from thai_deck_gen.config import GenConfig
from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.wordlist import WordEntry


@dataclass
class Spend:
    """What the run consumed from the outside world."""
    llm_calls: int = 0
    judgments: int = 0
    searches: list[str] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    forvo_lookups: list[str] = field(default_factory=list)
    tts_calls: list[str] = field(default_factory=list)


class World:
    def __init__(self, tmp_path, words=None, sentence="ผมกินข้าวครับ",
                 images_pass=True, forvo_has=(), **config):
        self.root = Path(tmp_path) / "deck"
        self.spend = Spend()
        self.images_pass = images_pass
        self.forvo_has = set(forvo_has)
        self.sentence = sentence
        self.config = GenConfig(**{"sentence_base": 0, "max_iterations": 3,
                                   **config})
        self.words = words or [
            WordEntry(thai="ข้าว", gloss="rice", category="Food",
                      part_of_speech="noun", classifier="จาน",
                      image_query="a plate of rice"),
            WordEntry(thai="หมา", gloss="dog", category="Animals",
                      part_of_speech="noun", classifier="ตัว",
                      image_query="a dog on a lead"),
        ]
        deck = new_deck(self.root, "spec", ["sounds", "words", "sentences"])
        write_deck(deck)

    # --- ports the generator consumes ---

    def complete(self, *args):
        self.spend.llm_calls += 1
        return self.sentence

    def judge_many(self, reqs):
        self.spend.judgments += len(reqs)
        return {r.note_id: [Verdict(rule=rule, passed=self.images_pass,
                                    confidence=0.9, rationale="")
                            for rule in r.rules] for r in reqs}

    def http_get(self, url, timeout=30, headers=None):
        self.spend.searches.append(url)

        class R:
            status_code = 200
            @staticmethod
            def json():
                return {"results": [{"url": "http://img/a.jpg", "license": "cc0"}],
                        "photos": [], "query": {"pages": {}}}
        return R()

    class _Fetch:
        def __init__(self, world): self.world = world
        def fetch(self, url):
            self.world.spend.downloads.append(url)
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (640, 480), (30, 90, 160)).save(buf, format="JPEG")
            return buf.getvalue()

    class _Forvo:
        def __init__(self, world): self.world = world
        def pronunciations(self, word):
            self.world.spend.forvo_lookups.append(word)
            return ([{"username": "native", "pathmp3": f"http://forvo/{word}.mp3"}]
                    if word in self.world.forvo_has else [])
        def download(self, url): return b"mp3"

    class _Tts:
        voice = "th-TH-Neural2-C"
        def __init__(self, world): self.world = world
        def synthesize(self, text, voice=None):
            self.world.spend.tts_calls.append(text)
            return b"mp3"

    # --- context the producers receive ---

    def context(self, **overrides):
        from thai_deck_gen.context import GenContext

        class _G:
            # No engine agrees on these fixtures, so words route to
            # adjudication rather than getting invented pronunciations.
            def syllables(self, w): return None
        class _T:
            def tokens(self, t): return [t]
        class _F:
            def rank(self, w): return 1

        ctx = GenContext(
            g2p=_G(), tokenizer=_T(), freq=_F(), llm=self,
            word_list=self.words, lexicon_words=[w.thai for w in self.words],
            exceptions={}, pair_seeds={}, grammar_points=[], exemplars=["ตัวอย่าง"],
            config=self.config, data_dir=Path("data"),
            adjudication_queue=self.root / "work" / "ipa_adjudication.yaml",
            targets_path=Path("data") / "spelling_targets.yaml",
            http_get=self.http_get, imgfetch=self._Fetch(self),
            image_candidates=2)
        for k, v in overrides.items():
            setattr(ctx, k, v)
        return ctx

    # --- reading the artifact back ---

    def deck(self):
        from thai_deck_eval.model.deck import load_deck
        return load_deck(self.root)

    def manifest(self):
        path = self.root / "media_manifest.yaml"
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        return {e["file"]: e for e in (data or {}).get("entries", [])}

    def work_file(self, name):
        path = self.root / "work" / name
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return ([json.loads(l) for l in text.splitlines() if l.strip()]
                if name.endswith(".jsonl") else yaml.safe_load(text))


def report(missing_contrasts=(), missing_categories=(), findings=()):
    """An evaluator report shaped as the generator consumes it."""
    return {
        "gate": "fail",
        "findings": list(findings),
        "metrics": [
            {"name": "coverage/minimal_pairs", "value": 0.0,
             "detail": {"missing": list(missing_contrasts), "by_note": {}}},
            {"name": "coverage/categories", "value": 0.0,
             "detail": {"missing": list(missing_categories)}},
            {"name": "coverage/frequency", "value": 0.0, "detail": {}},
        ],
        "scores": {"integrity": 0, "language": 0, "method": 0, "content": 0},
    }
