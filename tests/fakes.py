from thai_deck_eval.lang.ipa import parse_ipa

class FakeG2P:
    def __init__(self, table: dict[str, str]):
        self.table = table
    def syllables(self, word):
        return parse_ipa(self.table[word]) if word in self.table else None

class FakeTokenizer:
    def __init__(self, table: dict[str, list[str]] | None = None):
        self.table = table or {}
    def tokens(self, text):
        return self.table.get(text, [text])

class FakeFreq:
    def __init__(self, table: dict[str, int]):
        self.table = table
    def rank(self, word):
        return self.table.get(word)
