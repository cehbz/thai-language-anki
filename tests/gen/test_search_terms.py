import yaml
from pathlib import Path

from thai_deck_gen.media.scan import ImageNeed
from thai_deck_gen.media.images import _queries
from thai_deck_gen.wordlist import WordEntry, draft_image_queries, load_word_list

DATA = Path(__file__).parents[2] / "data"


class FakeLlm:
    """Returns a canned YAML mapping per call, recording prompts."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _list(tmp_path, rows):
    p = tmp_path / "word_list_th.yaml"
    p.write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


def test_image_query_is_preferred_over_the_gloss():
    """A photo of 'I' does not exist; a photo of someone pointing at
    themselves does."""
    need = ImageNeed(family="picture_word", note_id="pw-0", term="ฉัน",
                     gloss="I (female speaker)", category="Pronouns",
                     image_query="person pointing at themselves",
                     path="images/pw-0.jpg")
    assert _queries(need, {"Pronouns": "person"})[0] == "person pointing at themselves"


def test_draft_fills_missing_queries_and_keeps_existing(tmp_path):
    rows = [
        {"thai": "ส้ม", "gloss": "orange", "category": "Food",
         "part_of_speech": "noun", "classifier": "ลูก"},
        {"thai": "กิน", "gloss": "eat", "category": "Food",
         "part_of_speech": "verb", "image_query": "already set"},
    ]
    path = _list(tmp_path, rows)
    llm = FakeLlm(["ส้ม: ripe orange fruit on a market stall\n"])

    n = draft_image_queries(llm, path)

    out = {r["thai"]: r.get("image_query")
           for r in yaml.safe_load(path.read_text(encoding="utf-8"))}
    assert n == 1
    assert out["ส้ม"] == "ripe orange fruit on a market stall"
    assert out["กิน"] == "already set"          # never redrafted
    assert "กิน" not in llm.prompts[0]          # nor sent to the model


def test_draft_skips_words_that_are_not_picturable(tmp_path):
    rows = [{"thai": "มกรา", "gloss": "January", "category": "Months",
             "part_of_speech": "noun", "classifier": "เดือน", "picturable": False}]
    path = _list(tmp_path, rows)
    llm = FakeLlm([])
    assert draft_image_queries(llm, path) == 0
    assert llm.prompts == []


def test_draft_writes_after_each_category(tmp_path):
    """Session limits kill long runs: a killed draft keeps what it drafted."""
    rows = [{"thai": "ส้ม", "gloss": "orange", "category": "Food",
             "part_of_speech": "noun", "classifier": "ลูก"},
            {"thai": "หมา", "gloss": "dog", "category": "Animals",
             "part_of_speech": "noun", "classifier": "ตัว"}]
    path = _list(tmp_path, rows)

    class Dies(FakeLlm):
        def complete(self, prompt):
            if self.prompts:
                raise RuntimeError("session limit")
            return super().complete(prompt)

    llm = Dies(["ส้ม: orange fruit\n", "หมา: a dog"])
    try:
        draft_image_queries(llm, path)
    except RuntimeError:
        pass
    out = {r["thai"]: r.get("image_query")
           for r in yaml.safe_load(path.read_text(encoding="utf-8"))}
    assert out["ส้ม"] == "orange fruit"        # the first category survived


def test_apply_proposals_marks_them_as_judge_written(tmp_path):
    """A phrase a judge wrote must be distinguishable from one you wrote."""
    from thai_deck_gen.wordlist import apply_query_proposals
    rows = [{"thai": "ปี", "gloss": "year", "category": "Time",
             "part_of_speech": "noun", "classifier": "ปี",
             "image_query": "same tree in four seasons"}]
    path = _list(tmp_path, rows)
    proposals = tmp_path / "image_query_proposals.yaml"
    proposals.write_text(yaml.safe_dump(
        {"ปี": {"suggestion": "birthday cake with candles",
                "previous": "same tree in four seasons"}},
        allow_unicode=True), encoding="utf-8")

    n = apply_query_proposals(path, proposals)

    out = yaml.safe_load(path.read_text(encoding="utf-8"))[0]
    assert n == 1
    assert out["image_query"] == "birthday cake with candles"
    assert out["image_query_source"] == "judge"


def test_apply_proposals_leaves_a_hand_written_phrase_alone(tmp_path):
    """The word list is curated: a machine does not overwrite your wording."""
    from thai_deck_gen.wordlist import apply_query_proposals
    rows = [{"thai": "ปี", "gloss": "year", "category": "Time",
             "part_of_speech": "noun", "classifier": "ปี",
             "image_query": "rings in a cut tree stump",
             "image_query_source": "human"}]
    path = _list(tmp_path, rows)
    proposals = tmp_path / "p.yaml"
    proposals.write_text(yaml.safe_dump({"ปี": {"suggestion": "a calendar"}},
                                        allow_unicode=True), encoding="utf-8")
    assert apply_query_proposals(path, proposals) == 0
    assert yaml.safe_load(path.read_text(encoding="utf-8"))[0]["image_query"] \
        == "rings in a cut tree stump"
