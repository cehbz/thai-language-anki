from thai_deck_eval.model.notes import (Audio, MinimalPairNote, PairMember,
                                        PictureWordNote, SentenceNote,
                                        SpellingSoundNote)
from thai_deck_gen.compiler.ordering import intro_order, member_rank
from thai_deck_gen.deckio import new_deck
from tests.gen.test_words import FakeFreq

def _audio(name):
    return Audio(file=f"audio/{name}.mp3", source="native", speaker="pending")

def _member(thai):
    return PairMember(thai=thai, ipa="x", audio=_audio(thai))

def _pair(id_, contrast, thais):
    return MinimalPairNote(id=id_, contrast=contrast,
                           members=[_member(t) for t in thais])

def _spelling(id_, example_word):
    return SpellingSoundNote(id=id_, pattern="-ะ", pattern_kind="vowel",
                             example_word=example_word, audio=_audio(id_),
                             image="images/x.jpg")

def _word(id_, thai, rank):
    return PictureWordNote(id=id_, thai=thai, image="images/x.jpg",
                           audio=_audio(id_), frequency_rank=rank,
                           category="Food")

def _sentence(id_, target):
    return SentenceNote(id=id_, kind="new_word", thai=f"sentence-{target}",
                        target=target, audio=_audio(id_))

FREQ = FakeFreq({"c1": 1, "c2": 2, "d": 3, "a1": 5, "a2": 9})

def _build_deck(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds", "words", "sentences"])
    deck.minimal_pairs = [_pair("mp-1", "tone", ["a1", "a2"]),
                          _pair("mp-2", "tone", ["c1", "c2"])]
    deck.spelling_sound = [_spelling("sp-1", "d")]
    deck.picture_words = [_word("pw-1", "one", 1), _word("pw-2", "two", 2),
                          _word("pw-3", "three", 3), _word("pw-4", "four", 4)]
    deck.sentences = [_sentence("sn-one", "one"), _sentence("sn-three", "three")]
    return deck

def test_member_rank_min_over_members_and_unranked_sentinel():
    pair = _pair("mp-1", "tone", ["a1", "a2"])
    assert member_rank(pair, FREQ) == 5
    unranked = _pair("mp-x", "tone", ["zz", "yy"])
    assert member_rank(unranked, FREQ) > 10**6

def test_member_rank_spelling_and_sentence():
    assert member_rank(_spelling("sp-1", "d"), FREQ) == 3
    assert member_rank(_sentence("sn-one", "one"), FakeFreq({"one": 7})) == 7

def test_intro_order_sequence(tmp_path):
    deck = _build_deck(tmp_path)
    order = intro_order(deck, FREQ, base=3)
    ids = [(fam, n.id) for fam, n in order]
    assert ids == [
        ("minimal_pair", "mp-2"),
        ("spelling_sound", "sp-1"),
        ("minimal_pair", "mp-1"),
        ("picture_word", "pw-1"),
        ("picture_word", "pw-2"),
        ("picture_word", "pw-3"),
        ("sentence", "sn-one"),
        ("sentence", "sn-three"),
        ("picture_word", "pw-4"),
    ]

def test_intro_order_appends_unmatched_sentences_at_end(tmp_path):
    deck = _build_deck(tmp_path)
    deck.sentences.append(_sentence("sn-never", "unknown-word"))
    order = intro_order(deck, FREQ, base=3)
    ids = [(fam, n.id) for fam, n in order]
    assert ids[-1] == ("sentence", "sn-never")

def test_intro_order_divides_rank_by_emphasis_weight(tmp_path):
    from thai_deck_gen.emphasis import Emphasis
    deck = new_deck(tmp_path / "d", "t", ["words"])
    food = PictureWordNote(id="pw-10", thai="food", image="images/x.jpg",
                           audio=_audio("f"), frequency_rank=10, category="Food")
    animal = PictureWordNote(id="pw-4", thai="animal", image="images/x.jpg",
                             audio=_audio("a"), frequency_rank=4, category="Animals")
    deck.picture_words = [animal, food]
    plain = [n.thai for _, n in intro_order(deck, FREQ)]
    assert plain == ["animal", "food"]
    weighted = [n.thai for _, n in intro_order(
        deck, FREQ, emphasis=Emphasis(theme="t", category_weights={"Food": 3}))]
    assert weighted == ["food", "animal"]          # 10/3 < 4
