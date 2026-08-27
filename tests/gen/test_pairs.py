from pathlib import Path
from thai_deck_eval.lang.ipa import parse_ipa
from thai_deck_gen.deckio import new_deck
from thai_deck_gen.producers.pairs import analyze_lexicon, fill_pairs, find_pair
from thai_deck_gen.report import Gaps

class FakeG2P:
    # maps thai word -> ipa string; None when unknown
    def __init__(self, table): self.table = table
    def syllables(self, word):
        ipa = self.table.get(word)
        return parse_ipa(ipa) if ipa else None

TABLE = {"คา": "kʰaː˧", "ค่า": "kʰaː˥˩", "ขา": "kʰaː˨˩˦",
         "ปา": "paː˧", "พา": "pʰaː˧", "มานาว": "maː˧.naːw˧"}

def _lex():
    return analyze_lexicon(TABLE, FakeG2P(TABLE), exceptions={})

def test_analyze_lexicon_skips_multisyllable():
    assert "มานาว" not in _lex()

def test_find_tone_pair():
    assert set(find_pair("tone:mid-falling", _lex())) == {"คา", "ค่า"}

def test_find_aspiration_pair():
    assert set(find_pair("aspiration:labial", _lex())) == {"ปา", "พา"}

def test_find_pair_none_when_absent():
    assert find_pair("vowel_length", _lex()) is None

def _gaps(missing):
    return Gaps(gate="fail", missing_contrasts=missing, pair_by_note={},
                missing_categories=[], frequency_covered=0, speaker_value=0,
                findings=[])

class Ctx:
    def __init__(self):
        self.g2p = FakeG2P(TABLE)
        self.lexicon_words = list(TABLE)
        self.exceptions = {}
        self.pair_seeds = {"vowel_length": [["เขา", "kʰaw˨˩˦"], ["ขาว", "kʰaːw˨˩˦"]]}

def test_fill_pairs_adds_note_with_rendered_ipa(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    res = fill_pairs(_gaps(["tone:mid-falling"]), deck, Ctx())
    assert res.added == 1
    note = deck.minimal_pairs[0]
    assert note.contrast == "tone"
    assert {m.ipa for m in note.members} == {"kʰaː˧", "kʰaː˥˩"}
    assert note.members[0].audio.speaker == "pending"

def test_fill_pairs_uses_seeds_then_blocks(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["sounds"])
    res = fill_pairs(_gaps(["vowel_length", "consonant:r-l"]), deck, Ctx())
    assert res.added == 1                      # from seeds
    assert res.blocked == ["consonant:r-l"]    # nowhere to get it
