"""recorded.py: a local engine's answer for a form is a record row under
`EngineKey(engine, version, form)`, read before the engine runs and
appended on a miss -- "no reading" included -- so one form costs one
computation per engine version ever. A reading is stored as
`phonology.read_form` presents it. Engines are injected fakes.
"""
import pytest

from thai_syllabus.cachekeys import EngineKey
from thai_syllabus.entities import Syllable
from thai_syllabus.phonology import Engines, read_form
from thai_syllabus.recorded import record_readings, recorded_reading, recorded_tone
from thai_syllabus.store import SyllabusDb


def S(on, v, co, ln="long", t="mid"):
    return Syllable(segments=(on, v, co), vowel_length=ln, tone=t)


PA = (S("p", "a", ""),)
DEAD_PA = (S("p", "a", "ʔ", "short", "high"),)


@pytest.fixture
def db(tmp_path):
    return SyllabusDb(tmp_path / "syllabus.db")


class Counting:
    """A fake engine over a dict that counts its calls."""
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __call__(self, form):
        self.calls.append(form)
        return self.answers.get(form)


def test_a_form_is_computed_once_and_the_row_answers_every_later_ask(db):
    eng = Counting({"ปา": PA})
    read = recorded_reading(db, "thaig2p", "1", eng)
    assert read("ปา") == PA and read("ปา") == PA
    assert eng.calls == ["ปา"]
    row = db.latest("provide", "thaig2p", EngineKey(engine="thaig2p", version="1", form="ปา"))
    assert row is not None and row.subject == "engine:ปา" and row.cost == 0


def test_no_reading_is_recorded_so_an_unreadable_form_is_not_recomputed(db):
    eng = Counting({})
    read = recorded_reading(db, "tltk", "1", eng)
    assert read("ปา") is None and read("ปา") is None
    assert eng.calls == ["ปา"]


def test_a_new_version_is_a_new_key_and_asks_the_engine_again(db):
    first, second = Counting({"ปา": PA}), Counting({"ปา": PA})
    recorded_reading(db, "tltk", "1", first)("ปา")
    recorded_reading(db, "tltk", "2", second)("ปา")
    assert first.calls == ["ปา"] and second.calls == ["ปา"]


def test_engines_are_recorded_apart(db):
    a, b = Counting({"ปา": PA}), Counting({"ปา": PA})
    recorded_reading(db, "thaig2p", "1", a)("ปา")
    recorded_reading(db, "tltk", "1", b)("ปา")
    assert a.calls == ["ปา"] and b.calls == ["ปา"]


def test_the_recorded_reading_is_the_one_read_form_presents(db):
    """ʔ-coda convention applied, a phrase read token by token, a
    degenerate reading dropped: the same thing from the record as from the
    live engine."""
    ba = (S("b", "a", ""),)
    loop = tuple(S("w", "a", "") for _ in range(11))
    answers = {"ปะ": DEAD_PA, "ปา": PA, "บา": ba, "ภาพวาด": loop}
    for form in ("ปะ", "ปา บา", "ภาพวาด", "ซ"):
        live = read_form(Counting(answers), form)
        assert recorded_reading(db, "e", "1", Counting(answers))(form) == live
    assert recorded_reading(db, "e", "1", Counting({}))("ปา บา") == PA + ba


def test_a_phrase_with_an_unreadable_token_is_recorded_as_no_reading(db):
    eng = Counting({"ปา": PA})
    read = recorded_reading(db, "e", "1", eng)
    assert read("ปา ซ") is None and read("ปา ซ") is None
    assert eng.calls == ["ปา", "ซ"]


def test_the_rule_tone_answer_is_recorded_including_none(db):
    eng = Counting({"ปา": "mid"})
    tone = recorded_tone(db, "rule-tone", "1", eng)
    assert tone("ปา") == "mid" and tone("ปา") == "mid"
    assert tone("ซซ") is None and tone("ซซ") is None
    assert eng.calls == ["ปา", "ซซ"]


def test_every_local_engine_records_its_reading_of_every_form(db):
    g2p, tltk, tone = Counting({"ปา": PA}), Counting({"ปา": PA}), Counting({"ปา": "mid"})
    resolved = []

    def engines():
        resolved.append(1)
        return Engines(g2p={"thaig2p": g2p, "tltk": tltk}, tone=tone)

    versions = {"thaig2p": "1", "tltk": "2", "rule-tone": "3"}
    record_readings(db, engines, ["ปา", "ซ"], versions=versions)
    assert db.latest("provide", "tltk", EngineKey("tltk", "2", "ปา")).answer["value"] is not None
    assert db.latest("provide", "rule-tone", EngineKey("rule-tone", "3", "ซ")).answer == {
        "value": None}
    assert resolved == [1]
    record_readings(db, engines, ["ปา", "ซ"], versions=versions)
    assert resolved == [1]                  # all on record: the engines are never resolved
    assert (g2p.calls, tone.calls) == (["ปา", "ซ"], ["ปา", "ซ"])
