"""paiboon: a reading rendered in Paiboon romanization (spec 3 r70's
pronunciation ruling question), the system thai-language.com's own
Paiboon strings use: unaspirated stops g/bp/dt, aspirated k/p/t, long
vowels doubled, tone marked on the vowel (mid none, low grave, falling
circumflex, high acute, rising caron), syllables joined by hyphens."""
import pytest

from thai_syllabus.entities import Syllable
from thai_syllabus.paiboon import render


def S(onset, vowel, coda="", length="short", tone="mid"):
    return Syllable(segments=(onset, vowel, coda), vowel_length=length, tone=tone)


@pytest.mark.parametrize("syllables, expected", [
    ((S("kʰ", "a", "w", "long", "falling"),), "kâao"),        # ข้าว rice
    ((S("kʰr", "a", "p", "short", "high"),), "kráp"),         # ครับ
    ((S("k", "i", "n"),), "gin"),                             # กิน eat
    ((S("pl", "a", "", "long"),), "bplaa"),                   # ปลา fish
    ((S("t", "a", "", "short", "low"),), "dtà"),              # ตะ
    ((S("tʰ", "a", "", "long", "falling"),), "tâa"),          # ถ้า if
    ((S("tɕ", "a", "ʔ", "short", "low"),), "jà"),             # จะ, the glottal coda dropped
    ((S("tɕʰ", "ɔ", "p", "long", "falling"),), "chɔ̂ɔp"),      # ชอบ like
    ((S("m", "a", "j", "short", "falling"),), "mâi"),         # ไม่ not
    ((S("h", "i", "w", "short", "rising"),), "hǐu"),          # หิว hungry
    ((S("m", "ɯa", "ŋ", "long"),), "mʉang"),                  # เมือง city
    ((S("ŋ", "ɤ", "n", "long"),), "ngəən"),                   # เงิน money
    ((S("j", "u", "", "long", "low"),), "yùu"),               # อยู่ be at
    ((S("ʔ", "a", "j", "long"),), "aai"),                     # อาย shy
    ((S("s", "a", "", "short", "low"), S("w", "a", "t", "short", "low"),
      S("d", "i", "", "long")), "sà-wàt-dii"),                # สวัสดี hello
])
def test_a_reading_renders_in_paiboon(syllables, expected):
    assert render(syllables) == expected
