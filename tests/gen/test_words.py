import yaml
from pathlib import Path
from thai_deck_gen.deckio import new_deck
from thai_deck_gen.producers.words import fill_words
from thai_deck_gen.wordlist import WordEntry
from tests.gen.test_pairs import FakeG2P, _gaps

class FakeFreq:
    def __init__(self, table): self.table = table
    def rank(self, w): return self.table.get(w)

def _ctx(tmp_path):
    class Cfg: test_spelling_rank = 300
    class Ctx:
        word_list = [
            WordEntry(thai="น้ำ", gloss="water", category="Beverages",
                      part_of_speech="noun", classifier="แก้ว"),
            WordEntry(thai="กิน", gloss="eat", category="Food",
                      part_of_speech="verb"),
            WordEntry(thai="ๆๆ", gloss="mystery", category="Food",
                      part_of_speech="other")]
        freq = FakeFreq({"น้ำ": 5, "กิน": 2})
        g2p = FakeG2P({"กิน": "kin˧"})
        exceptions = {"น้ำ": "naːm˦˥"}
        adjudication_queue = tmp_path / "work" / "ipa_adjudication.yaml"
        config = Cfg()
    return Ctx()

def test_fill_words_orders_by_rank_and_uses_exceptions(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    res = fill_words(_gaps([]), deck, _ctx(tmp_path))
    assert [n.thai for n in deck.picture_words] == ["กิน", "น้ำ"]
    assert deck.picture_words[1].ipa == "naːm˦˥"     # exception wins
    assert deck.picture_words[0].test_spelling is True
    assert res.added == 2

def test_fill_words_queues_unrankable(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    ctx = _ctx(tmp_path)
    res = fill_words(_gaps([]), deck, ctx)
    assert any("ๆๆ" in b for b in res.blocked)

def test_fill_words_queues_g2p_unknown_for_adjudication(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    ctx = _ctx(tmp_path)
    ctx.exceptions = {}                                # น้ำ now unknown
    fill_words(_gaps([]), deck, ctx)
    queued = yaml.safe_load(ctx.adjudication_queue.read_text())
    assert "น้ำ" in queued
    note = next(n for n in deck.picture_words if n.thai == "น้ำ")
    assert note.ipa is None

def test_fill_words_idempotent(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    ctx = _ctx(tmp_path)
    fill_words(_gaps([]), deck, ctx)
    res2 = fill_words(_gaps([]), deck, ctx)
    assert res2.added == 0
