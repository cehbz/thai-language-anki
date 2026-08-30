from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.producers.sentences import check_sentence, fill_sentences
from thai_deck_gen.report import GapFinding
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

def test_fill_sentences_judge_regen_resets_audio_and_deletes_media(tmp_path):
    from thai_deck_eval.model.notes import Audio, SentenceNote
    deck = _deck_with_words(tmp_path, 3)
    for i in range(3):
        deck.sentences.append(SentenceNote(
            id=f"sn-w{i}-1", kind="new_word", thai=f"w{i}filler", target=f"w{i}",
            audio=Audio(file=f"audio/sentences/sn-w{i}-1.mp3",
                        source="tts", speaker="tts:voice")))
    write_deck(deck)
    media_path = deck.root / "media" / "audio" / "sentences" / "sn-w0-1.mp3"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(b"OLD")

    class CachedFake:
        def __init__(self, resp): self.resp = list(resp)
        def complete(self, producer, version, prompt): return self.resp.pop(0)
    tok = FakeTok({"w0w2": ["w0", "w2"]})
    gaps = _gaps([])
    gaps.findings.append(GapFinding(rule="judge/sentence_mismatch", severity="warn",
                                    note_id="sn-w0-1", message="doesn't fit"))

    class Ctx:
        llm = CachedFake(["w0w2"])
        tokenizer = tok; exemplars = []; config = Cfg()
    Ctx.grammar_points = []

    res = fill_sentences(gaps, deck, Ctx())

    assert res.changed == 1
    note = deck.sentences[0]
    assert note.thai == "w0w2"
    assert note.audio.speaker == "pending"
    assert note.audio.source == "tts"
    assert not media_path.exists()

    from thai_deck_gen.media.scan import pending_audio
    needs = pending_audio(deck)
    assert any(n.note_id == "sn-w0-1" for n in needs)

def test_fill_sentences_halts_on_llm_error_keeping_progress(tmp_path):
    from thai_deck_gen.llm import LlmError
    deck = _deck_with_words(tmp_path, 3)
    class FlakyLlm:
        def __init__(self): self.calls = 0
        def complete(self, producer, version, prompt):
            self.calls += 1
            if self.calls == 1:
                return "w0w1"
            raise LlmError("usage limit reached")
    tok = FakeTok({"w0w1": ["w0", "w1"]})
    class Ctx:
        llm = FlakyLlm(); tokenizer = tok; exemplars = []; config = Cfg()
        grammar_points = []
    res = fill_sentences(_gaps([]), deck, Ctx())
    assert res.added == 1                      # first sentence kept
    assert len(deck.sentences) == 1
    assert any("usage limit reached" in b for b in res.blocked)
    assert Ctx.llm.calls == 2                  # halted, no further attempts

def test_fill_sentences_adds_themed_sentence_for_emphasized_words(tmp_path):
    from thai_deck_gen.emphasis import Emphasis
    deck = _deck_with_words(tmp_path, 3)          # all category "Food"
    class RecordingLlm:
        def __init__(self, resp): self.resp = list(resp); self.prompts = []
        def complete(self, producer, version, prompt):
            self.prompts.append(prompt); return self.resp.pop(0)
    tok = FakeTok({"w0w1": ["w0", "w1"], "w1w2": ["w1", "w2"],
                   "w0w2": ["w0", "w2"]})
    class Ctx:
        llm = RecordingLlm(["w0w1", "w1w2", "w0w2"] * 2)
        tokenizer = tok; exemplars = []; config = Cfg(); grammar_points = []
        emphasis = Emphasis(theme="food and cooking",
                            category_weights={"Food": 2})
        word_list = []
    res = fill_sentences(_gaps([]), deck, Ctx())
    assert res.added == 6                          # plain + themed per word
    themed = [n for n in deck.sentences if n.id.endswith("-themed")]
    assert [n.target for n in themed] == ["w0", "w1", "w2"]
    assert all(n.kind == "new_word" for n in themed)
    themed_prompts = [p for p in Ctx.llm.prompts if "food and cooking" in p]
    assert len(themed_prompts) == 3
    assert not any("food and cooking" in p for p in Ctx.llm.prompts[:3])  # plain prompt unchanged

def test_fill_sentences_themed_pass_skips_unweighted_and_is_idempotent(tmp_path):
    from thai_deck_gen.emphasis import Emphasis
    deck = _deck_with_words(tmp_path, 3)
    deck.picture_words[1].category = "Animals"      # weight 1 -> no themed sentence
    class CachedFake:
        def __init__(self, resp): self.resp = list(resp)
        def complete(self, producer, version, prompt): return self.resp.pop(0)
    tok = FakeTok({"w0w1": ["w0", "w1"], "w1w2": ["w1", "w2"],
                   "w0w2": ["w0", "w2"]})
    class Ctx:
        llm = CachedFake(["w0w1", "w1w2", "w0w2", "w0w1", "w0w2"])
        tokenizer = tok; exemplars = []; config = Cfg(); grammar_points = []
        emphasis = Emphasis(theme="t", category_weights={"Food": 2})
        word_list = []
    res = fill_sentences(_gaps([]), deck, Ctx())
    assert res.added == 5                          # 3 plain + 2 themed
    Ctx.llm = CachedFake([])
    res2 = fill_sentences(_gaps([]), deck, Ctx())
    assert res2.added == 0                         # nothing regenerated


def test_fill_sentences_themed_pass_ignores_default_weighted_categories(tmp_path):
    from thai_deck_gen.emphasis import Emphasis
    deck = _deck_with_words(tmp_path, 2)
    deck.picture_words[1].category = "Colors"
    class CachedFake:
        def __init__(self, resp): self.resp = list(resp)
        def complete(self, producer, version, prompt): return self.resp.pop(0)
    tok = FakeTok({"w0w1": ["w0", "w1"], "w1w0": ["w1", "w0"]})
    class Ctx:
        llm = CachedFake(["w0w1", "w1w0", "w0w1"])
        tokenizer = tok; exemplars = []; config = type("C", (), {"sentence_base": 2})()
        grammar_points = []; word_list = []
        emphasis = Emphasis(theme="t", category_weights={"default": 1.2, "Food": 3})
    res = fill_sentences(_gaps([]), deck, Ctx())
    assert res.added == 3                          # 2 plain + themed for the Food word only
    assert [n.target for n in deck.sentences if n.id.endswith("-themed")] == ["w0"]


def test_fill_sentences_one_plain_sentence_per_thai_even_with_duplicate_words(tmp_path):
    deck = _deck_with_words(tmp_path, 3)
    deck.picture_words.append(deck.picture_words[0].model_copy(update={"id": "pw-dup"}))
    class CachedFake:
        def __init__(self, resp): self.resp = list(resp)
        def complete(self, producer, version, prompt): return self.resp.pop(0)
    tok = FakeTok({"w0w1": ["w0", "w1"], "w1w2": ["w1", "w2"], "w0w2": ["w0", "w2"]})
    class Ctx:
        llm = CachedFake(["w0w1", "w1w2", "w0w2"])
        tokenizer = tok; exemplars = []; config = Cfg(); grammar_points = []
    res = fill_sentences(_gaps([]), deck, Ctx())
    assert res.added == 3
    assert [n.target for n in deck.sentences] == ["w0", "w1", "w2"]


def test_prompt_pins_standard_particle_spelling_and_collocation():
    """54 of 140 judge rejections were the chat spelling คับ for ครับ, and
    most of the rest were unnatural collocations or verbless clauses."""
    from thai_deck_gen.producers.sentences import _prompt
    prompt = _prompt("กิน", {"ข้าว"}, ["ตัวอย่าง"])
    assert "ครับ" in prompt and "คับ" in prompt      # names the defect explicitly
    assert "collocation" in prompt.lower()
    assert "verb" in prompt.lower()


def test_normalizes_the_chat_particle_to_the_standard_spelling():
    """คับ at the end of an utterance is the chat spelling of ครับ. Elsewhere
    it is the ordinary word for "tight", which must survive untouched."""
    from thai_deck_gen.producers.sentences import normalize_particles
    assert normalize_particles("ผมกินข้าวคับ") == "ผมกินข้าวครับ"
    assert normalize_particles("ผมกินข้าวครับ") == "ผมกินข้าวครับ"
    assert normalize_particles("รองเท้าของฉันคับค่ะ") == "รองเท้าของฉันคับค่ะ"
    assert normalize_particles("เสื้อตัวนี้คับ") == "เสื้อตัวนี้ครับ"   # ambiguous: particle wins


def test_prompt_varies_its_vocabulary_sample_per_target():
    """Handing every call the same sorted first-100 known words is why 43%
    of the deck opened with one of two frames."""
    from thai_deck_gen.producers.sentences import _prompt
    known = {f"w{i}" for i in range(200)}
    a = _prompt("กิน", known, ["ตัวอย่าง"])
    b = _prompt("ซื้อ", known, ["ตัวอย่าง"])
    assert a != b
    sample_a = a.split("Known vocabulary (sample): ")[1].split("\n")[0]
    sample_b = b.split("Known vocabulary (sample): ")[1].split("\n")[0]
    assert sample_a != sample_b                    # different words offered
    assert _prompt("กิน", known, ["ตัวอย่าง"]) == a   # stable for one target


def test_prompt_states_the_production_register():
    from thai_deck_gen.producers.sentences import _prompt
    prompt = _prompt("กิน", {"ข้าว"}, ["ตัวอย่าง"])
    assert "ผม" in prompt and "ครับ" in prompt


def test_prompt_lists_frames_to_avoid():
    from thai_deck_gen.producers.sentences import _prompt
    prompt = _prompt("กิน", {"ข้าว"}, ["ตัวอย่าง"], avoid=["คืนนี้ผมกินข้าว"])
    assert "คืนนี้ผมกินข้าว" in prompt
    assert "avoid" in prompt.lower() or "vary" in prompt.lower()


def test_generation_offers_only_vocabulary_known_at_that_position(tmp_path):
    """The generator must build from what the learner has met by the time
    the sentence appears, or every sentence it writes fails the rule."""
    from thai_deck_gen.producers.sentences import vocabulary_by_position
    from thai_deck_eval.model.notes import Audio, PictureWordNote

    def _w(thai, rank):
        return PictureWordNote(
            id=f"pw-{rank}", thai=thai, image="i.jpg",
            audio=Audio(file="a.mp3", source="native", speaker="pending"),
            frequency_rank=rank, category="Food")

    words = [_w("ก", 3), _w("ข", 1), _w("ค", 2), _w("ง", 4)]
    known_for = vocabulary_by_position(words, base=2)
    assert known_for["ข"] == {"ข", "ค"}            # base of 2: the first two
    assert known_for["ก"] == {"ข", "ค", "ก"}       # third word: itself included
    assert known_for["ง"] == {"ข", "ค", "ก", "ง"}


def test_vocabulary_sample_order_varies_per_target():
    """Position in a long list is a bias: the same words at the top produce
    the same sentences. Order is shuffled per target, not just membership."""
    from thai_deck_gen.producers.sentences import _vocab_sample
    known = {f"w{i:03d}" for i in range(300)}
    a = _vocab_sample(known, "กิน")
    b = _vocab_sample(known, "ซื้อ")
    assert a != sorted(a)                       # not alphabetical
    assert a != b
    assert _vocab_sample(known, "กิน") == a     # deterministic per target


def test_exemplars_vary_per_target():
    """Showing every prompt the same three style references is uniformity
    by construction."""
    from thai_deck_gen.producers.sentences import _pick_exemplars
    pool = [f"ประโยค{i}" for i in range(40)]
    assert _pick_exemplars(pool, "กิน") != _pick_exemplars(pool, "ซื้อ")
    assert _pick_exemplars(pool, "กิน") == _pick_exemplars(pool, "กิน")
    assert len(_pick_exemplars(pool, "กิน")) == 3
