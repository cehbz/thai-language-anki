from thai_deck_gen.deckio import new_deck
from thai_deck_gen.producers.sentences import check_sentence, fill_sentences
from tests.gen.test_pairs import _gaps
from tests.gen.fakes import FakeLlm

class FakeTok:
    def __init__(self, table): self.table = table
    def tokens(self, text): return self.table[text]

def test_check_sentence_accepts_one_unknown():
    tok = FakeTok({"ฉันกินข้าว": ["ฉัน", "กิน", "ข้าว"]})
    assert check_sentence("ฉันกินข้าว", "กิน", {"ฉัน", "กิน"}, tok) is None

def test_check_sentence_rejects_two_unknowns():
    tok = FakeTok({"ฉันกินข้าวเย็น": ["ฉัน", "กิน", "ข้าว", "เย็น"]})
    assert check_sentence("ฉันกินข้าวเย็น", "กิน", {"กิน"}, tok) is not None

def test_check_sentence_accepts_compound_member_target():
    tok = FakeTok({"ฉันกินข้าว": ["ฉัน", "กินข้าว"]})
    assert check_sentence("ฉันกินข้าว", "กิน", {"ฉัน", "กิน", "ข้าว"}, tok) is None

def test_check_sentence_rejects_absent_target():
    tok = FakeTok({"ฉันนอน": ["ฉัน", "นอน"]})
    assert check_sentence("ฉันนอน", "กิน", {"ฉัน", "นอน"}, tok) is not None

def _deck_with_words(tmp_path, n):
    deck = new_deck(tmp_path / "d", "t", ["words", "sentences"])
    # cheap synthetic base: n picture words known to the tokenizer
    from thai_deck_eval.model.notes import Audio, PictureWordNote
    for i in range(n):
        deck.picture_words.append(PictureWordNote(
            id=f"pw-{i}", thai=f"w{i}", image=f"images/pw-{i}.jpg",
            audio=Audio(file=f"audio/picture_words/pw-{i}.mp3",
                        source="native", speaker="pending"),
            frequency_rank=i + 1, category="Food"))
    return deck

class Cfg: sentence_base = 3

def test_fill_sentences_blocked_below_base(tmp_path):
    deck = _deck_with_words(tmp_path, 2)
    class Ctx:
        llm = None; tokenizer = None; exemplars = []; config = Cfg()
    res = fill_sentences(_gaps([]), deck, Ctx())
    assert res.added == 0 and res.blocked

def test_fill_sentences_generates_and_checks(tmp_path):
    deck = _deck_with_words(tmp_path, 3)
    class CachedFake:
        def __init__(self, resp): self.resp = list(resp)
        def complete(self, producer, version, prompt): return self.resp.pop(0)
    tok = FakeTok({"w0w1": ["w0", "w1"], "w1w2": ["w1", "w2"],
                   "w0w2": ["w0", "w2"]})
    class Ctx:
        llm = CachedFake(["w0w1", "w1w2", "w0w2"])
        tokenizer = tok; exemplars = ["ตัวอย่าง"]; config = Cfg()
    # grammar points not exercised: patch load to empty via ctx
    Ctx.grammar_points = []
    res = fill_sentences(_gaps([]), deck, Ctx())
    assert res.added == 3
    assert deck.sentences[0].kind == "new_word"
    assert deck.sentences[0].target == "w0"
