import yaml
from pathlib import Path
from thai_deck_gen.deckio import new_deck
from thai_deck_gen.producers.words import UNRANKED_RANK, fill_words
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
    assert [n.thai for n in deck.picture_words] == ["กิน", "น้ำ", "ๆๆ"]   # unranked last
    assert deck.picture_words[1].ipa == "naːm˦˥"     # exception wins
    assert deck.picture_words[0].test_spelling is True
    assert res.added == 3

def test_fill_words_queues_unrankable(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    ctx = _ctx(tmp_path)
    res = fill_words(_gaps([]), deck, ctx)
    note = next(n for n in deck.picture_words if n.thai == "ๆๆ")
    assert note.id == "pw-u-ๆๆ"                      # unranked: stable thai-keyed id
    assert note.frequency_rank == UNRANKED_RANK
    assert note.test_spelling is False
    assert [n.thai for n in deck.picture_words][-1] == "ๆๆ"   # after every ranked word
    assert not any("ๆๆ" in b for b in res.blocked)

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


def test_fill_words_adds_each_thai_once_even_if_listed_in_two_categories(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    ctx = _ctx(tmp_path)
    ctx.word_list = ctx.word_list + [
        WordEntry(thai="กิน", gloss="eat (Verbs)", category="Verbs",
                  part_of_speech="verb")]
    res = fill_words(_gaps([]), deck, ctx)
    assert [n.thai for n in deck.picture_words].count("กิน") == 1
    assert len({n.id for n in deck.picture_words}) == len(deck.picture_words)


def test_fill_words_keeps_gloss_off_the_note(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    fill_words(_gaps([]), deck, _ctx(tmp_path))
    assert all(n.gloss is None for n in deck.picture_words)


def test_fill_words_honours_the_picturable_flag(tmp_path):
    """A month name has no photograph, but "durian season" in the same
    category does: picturable is per word, never per category."""
    deck = new_deck(tmp_path / "d", "t", ["words"])
    ctx = _ctx(tmp_path)
    ctx.word_list = ctx.word_list + [
        WordEntry(thai="มกรา", gloss="January", category="Months",
                  part_of_speech="noun", classifier="เดือน", picturable=False)]
    ctx.word_list = ctx.word_list + [
        WordEntry(thai="หน้าทุเรียน", gloss="durian season", category="Months",
                  part_of_speech="noun", classifier="หน้า")]
    res = fill_words(_gaps([]), deck, ctx)
    thai = [n.thai for n in deck.picture_words]
    assert "มกรา" not in thai            # January: nothing to photograph
    assert "หน้าทุเรียน" in thai          # durian season: photograph the durians
    assert res.added == 4
