import pytest
from thai_deck_eval.lang.tone import (ConsClass, Tone, analyze_syllable, tone_of)

@pytest.mark.parametrize("cls,live,long_v,mark,expected", [
    (ConsClass.MID, True, True, None, Tone.MID),        # กา
    (ConsClass.HIGH, True, True, None, Tone.RISING),    # ขา
    (ConsClass.LOW, True, True, None, Tone.MID),        # คา
    (ConsClass.MID, False, True, None, Tone.LOW),       # บาท
    (ConsClass.HIGH, False, False, None, Tone.LOW),     # ขับ
    (ConsClass.LOW, False, False, None, Tone.HIGH),     # คับ
    (ConsClass.LOW, False, True, None, Tone.FALLING),   # มาก
    (ConsClass.MID, True, True, "่", Tone.LOW),
    (ConsClass.HIGH, True, True, "่", Tone.LOW),
    (ConsClass.LOW, True, True, "่", Tone.FALLING),
    (ConsClass.MID, True, True, "้", Tone.FALLING),
    (ConsClass.HIGH, True, True, "้", Tone.FALLING),
    (ConsClass.LOW, True, True, "้", Tone.HIGH),
    (ConsClass.MID, True, True, "๊", Tone.HIGH),
    (ConsClass.MID, True, True, "๋", Tone.RISING),
])
def test_tone_table(cls, live, long_v, mark, expected):
    assert tone_of(cls, live, long_v, mark) == expected

@pytest.mark.parametrize("word,tone", [
    ("มา", Tone.MID), ("หมา", Tone.RISING),          # ห นำ
    ("ไม่", Tone.FALLING), ("ไม้", Tone.HIGH),
    ("ใหม่", Tone.LOW), ("ไหม", Tone.RISING),
    ("ขาว", Tone.RISING), ("ข่าว", Tone.LOW), ("ข้าว", Tone.FALLING),
    ("ไก่", Tone.LOW), ("ไข่", Tone.LOW),
    ("มาก", Tone.FALLING), ("อยู่", Tone.LOW),        # อ นำ
    ("กิน", Tone.MID),
])
def test_analyze_known_words(word, tone):
    a = analyze_syllable(word)
    assert a is not None, word
    assert a.tone == tone

@pytest.mark.parametrize("word", ["โรงเรียน", "ธนา", "คนา", "ชนา"])
def test_unparseable_returns_none(word):
    # โรงเรียน: multi-syllable. ธนา/คนา/ชนา: CCV two-syllable words whose
    # middle consonant is the *second* syllable's initial, not a final —
    # must not be mis-analyzed as a single syllable with a fabricated tone.
    assert analyze_syllable(word) is None
