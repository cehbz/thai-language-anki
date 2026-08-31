"""What an image run must do, stated without reference to how it does it.

These describe observable behaviour of a run: which corpora get consulted,
which words get retried, what the operator is told. Nothing here names a
fingerprint, a cache file, or a channel constant -- if the implementation
changes shape, these should still hold.
"""
import yaml

from thai_deck_eval.judge.core import Verdict
from thai_deck_gen.deckio import write_deck
from thai_deck_gen.media.images import fill_images
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import ImageNeed
from tests.gen.test_images import R, FakeFetch, _deck_with_pw, _jpeg_bytes
from tests.gen.test_pairs import _gaps


class RecordingWorld:
    """A pretend internet: records which corpora were asked, serves images."""

    def __init__(self, serving: set[str]):
        self.serving = serving          # corpora that return a usable result
        self.asked: list[str] = []
        self.imgfetch = FakeFetch({f"http://{s}/img.jpg": _jpeg_bytes()
                                   for s in ("openverse", "wikimedia", "pexels")})
        self.imagegen = None
        self.image_query_hints = {}
        self.image_candidates = 5
        self.pexels_key = "KEY" if "pexels" in serving else None

    def http_get(self, url, timeout=30, headers=None):
        corpus = ("pexels" if "pexels.com" in url
                  else "openverse" if "openverse" in url else "wikimedia")
        self.asked.append(corpus)
        if corpus not in self.serving:
            return R(payload={"results": [], "photos": [], "query": {"pages": {}}})
        if corpus == "pexels":
            return R(payload={"photos": [{"alt": "a", "src": {"large": "http://pexels/img.jpg"}}]})
        if corpus == "openverse":
            return R(payload={"results": [{"url": "http://openverse/img.jpg", "license": "cc0"}]})
        return R(payload={"query": {"pages": {"1": {"imageinfo": [{"url": "http://wikimedia/img.jpg"}]}}}})


class Judge:
    def __init__(self, accept: bool):
        self.accept = accept

    def judge_many(self, reqs):
        return {r.note_id: [Verdict(rule="judge/image-irrelevant", passed=self.accept,
                                    confidence=0.9, rationale="")] for r in reqs}


def _need(category="Pronouns"):
    return ImageNeed(family="picture_word", note_id="pw-0", term="ฉัน", gloss="I",
                     category=category, image_query="person pointing at themselves",
                     path="images/pw-0.jpg")


def _deck(tmp_path, category="Pronouns"):
    deck = _deck_with_pw(tmp_path, term="ฉัน", gloss="I")
    deck.picture_words[0].category = category
    write_deck(deck)
    return deck


def test_a_word_written_off_is_retried_when_a_new_corpus_becomes_available(tmp_path):
    """The point of adding a library is that words nothing could serve get
    another chance. A run that skips them has wasted the addition."""
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                "2026-08-31", judge=Judge(accept=False))

    better = RecordingWorld(serving={"pexels"})
    res = fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), better,
                      "2026-09-01", judge=Judge(accept=True))

    assert "pexels" in better.asked, "the new corpus was never consulted"
    assert res.changed == 1, "the word was not recovered by the new corpus"


def test_abstract_words_ask_the_concept_library_first(tmp_path):
    deck = _deck(tmp_path, category="Pronouns")
    world = RecordingWorld(serving={"pexels", "openverse"})
    fill_images([_need("Pronouns")], _gaps([]), deck, Manifest.load(deck.root),
                world, "2026-08-31", judge=Judge(accept=True))
    assert world.asked[0] == "pexels"


def test_concrete_words_ask_the_amateur_library_first(tmp_path):
    deck = _deck(tmp_path, category="Food")
    world = RecordingWorld(serving={"pexels", "openverse"})
    fill_images([_need("Food")], _gaps([]), deck, Manifest.load(deck.root),
                world, "2026-08-31", judge=Judge(accept=True))
    assert world.asked[0] == "openverse"


def test_a_source_that_cannot_run_is_reported(tmp_path, capsys):
    """Silent degradation is the failure mode that cost three smoke runs."""
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})      # no pexels key
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                "2026-08-31", judge=Judge(accept=True))
    assert "pexels" in capsys.readouterr().out.lower()


def test_a_word_is_not_re_searched_when_nothing_has_changed(tmp_path):
    """The memo still has to work: same corpora, same queries, no repeat."""
    deck = _deck(tmp_path)
    first = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), first,
                "2026-08-31", judge=Judge(accept=False))

    again = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), again,
                "2026-09-01", judge=Judge(accept=False))
    assert again.asked == [], "a settled word was searched again for nothing"


def test_a_changed_rubric_re_scores_without_re_downloading(tmp_path, monkeypatch):
    """Keeping rejected candidates exists so a relaxed rule can reconsider
    them; re-searching is the slow half of a run."""
    import thai_deck_eval.judge.prompts as prompts
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                "2026-08-31", judge=Judge(accept=False))
    downloads_before = len(world.imgfetch.urls)
    assert downloads_before > 0

    monkeypatch.setitem(prompts.PICTURE_RULES, "judge/image-embedded-text",
                        "a materially more permissive rubric")
    again = RecordingWorld(serving={"openverse"})
    res = fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), again,
                      "2026-09-01", judge=Judge(accept=True))

    assert res.changed == 1, "the relaxed rubric did not reconsider the word"
    assert again.imgfetch.urls == [], "candidates were downloaded a second time"
