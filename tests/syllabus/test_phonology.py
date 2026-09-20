"""Tests for phonology.py (spec 3 r28 section 5; the 2026-09-01
domain-language ruling): the corroboration rule that cross-checks a
judge's pronunciation verdict against thaig2p and, on a monosyllabic
tone disagreement, the deterministic tone engine. Engines are injected
fakes here -- thaig2p (torch) loads only inside default_engines(), never
in these tests.
"""
import pytest

from thai_syllabus.entities import Pronunciation, Syllable
from thai_syllabus.phonology import (Engines, corroborates, engines_pronunciation, is_degenerate,
                                     syllables_from_verdict)


def S(on, v, co, ln, t):
    return Syllable(segments=(on, v, co), vowel_length=ln, tone=t)


J = (S("kʰ", "a", "w", "long", "falling"),)


def test_agreement_with_thaig2p_corroborates():
    eng = Engines(g2p=(lambda w: J,), tone=lambda w: None)
    assert corroborates(J, "ข้าว", eng) is True  # ข้าว: rice


def test_tone_disagreement_is_settled_by_the_tone_engine_on_a_monosyllable():
    g2p = (S("kʰ", "a", "w", "long", "low"),)
    eng = Engines(g2p=(lambda w: g2p,), tone=lambda w: "falling")
    assert corroborates(J, "ข้าว", eng) is True
    eng2 = Engines(g2p=(lambda w: g2p,), tone=lambda w: "low")
    assert corroborates(J, "ข้าว", eng2) is False


def test_segment_disagreement_or_no_g2p_answer_leaves_it_disputed():
    other = (S("k", "a", "w", "long", "falling"),)
    assert corroborates(J, "ข้าว", Engines(g2p=(lambda w: other,), tone=lambda w: "falling")) is False
    assert corroborates(J, "ข้าว", Engines(g2p=(lambda w: None,), tone=lambda w: "falling")) is False


def test_syllables_from_verdict_builds_entities():
    v = {"syllables": [{"segments": ["kʰ", "a", "w"], "vowel_length": "long", "tone": "falling"}]}
    assert syllables_from_verdict(v) == J


@pytest.mark.integration
def test_default_engines_g2p_returns_a_syllable_for_a_real_word():
    """Documents thaig2p's actual output for ข้าว (rice) via the real
    pythainlp adapter -- skipped when the model isn't installed. The KB
    records thaig2p gets this word's vowel length wrong (short, not
    long); this test asserts whatever it actually returns rather than
    the "correct" answer, so a model upgrade that fixes it is visible as
    a test change, not a silent pass/fail flip.
    """
    from thai_syllabus.phonology import default_engines
    try:
        engines = default_engines()
        syls = engines.g2p[0]("ข้าว")  # ข้าว: rice
    except Exception as e:  # pragma: no cover - model/deps absent in CI
        pytest.skip(f"thaig2p unavailable: {e}")
    if syls is None:
        pytest.skip("thaig2p returned no analysis for ข้าว")
    assert len(syls) == 1
    assert syls[0].segments[1] == "a"


def test_default_engines_builds_the_engines_once(monkeypatch, real_default_engines):
    """M7: every caller reads `ctx.engines` first and falls back to this
    function, so a run with no injected engines builds Thaig2p -- and with
    it pythainlp and torch -- once per call site instead of once. The
    fallback is memoised: same Engines object, one construction. The
    injection seam is untouched; this is only about the default.

    The fake stands in for Thaig2p through the function's own deferred
    import, so nothing here loads torch. `real_default_engines` is the
    guard fixture's handle on the unpatched function (conftest.py), and
    the memo is cleared either side so no other test inherits the fake.
    """
    from thai_syllabus import engines as engines_module

    built = []

    class _FakeThaig2p:
        def __init__(self):
            built.append(1)

        def __call__(self, thai):
            return None

    monkeypatch.setattr(engines_module, "Thaig2p", _FakeThaig2p)
    real_default_engines.cache_clear()
    try:
        first = real_default_engines()
        second = real_default_engines()
        assert first is second
        assert built == [1]
    finally:
        real_default_engines.cache_clear()


# --- engines_pronunciation: the adoption pass's seed (spec 3 r40 §5) -------

def _engines(g2p_result, tone_result="mid"):
    return Engines(g2p=(lambda thai: g2p_result,), tone=lambda thai: tone_result)


def test_a_monosyllable_the_tone_engine_agrees_with_is_engines_agree():
    one = Syllable(segments=("k", "a", ""), vowel_length="short", tone="mid")
    got = engines_pronunciation("ไก่", _engines((one,), "mid"))   # ไก่: chicken
    assert got == Pronunciation(syllables=(one,), corroboration="engines_agree")


def test_a_monosyllable_the_tone_engine_disagrees_with_stays_disputed():
    one = Syllable(segments=("k", "a", ""), vowel_length="short", tone="mid")
    got = engines_pronunciation("ไก่", _engines((one,), "low"))   # ไก่: chicken
    assert got == Pronunciation(syllables=(one,), corroboration="disputed")


def test_nothing_read_is_no_pronunciation_at_all():
    assert engines_pronunciation("ๆ", _engines(None)) is None
    assert engines_pronunciation("ๆ", _engines(())) is None


# --- a phrase is its tokens (design 2026-09-20 §3): thaig2p reads "ปอ" and
# "ปลา" but not "ปอ ปลา", and reads "งอ งู" as ŋɔŋ.ŋu while "งอ" alone is ŋɔ.

def _token_engines(readings: dict, tone_result="mid"):
    """g2p keyed by the exact string asked: `readings[thai]`, None for any
    thai not in the map."""
    return Engines(g2p=(lambda thai: readings.get(thai),), tone=lambda thai: tone_result)


PO = (Syllable(segments=("p", "ɔ", ""), vowel_length="long", tone="mid"),)
PLA = (Syllable(segments=("pl", "a", ""), vowel_length="long", tone="mid"),)
WRONG = (Syllable(segments=("p", "ɔ", "ŋ"), vowel_length="long", tone="mid"),) + PLA


def test_a_phrase_is_read_token_by_token_never_whole():
    eng = _token_engines({"ปอ ปลา": WRONG, "ปอ": PO, "ปลา": PLA})
    assert eng.readings("ปอ ปลา") == (PO + PLA,)
    assert engines_pronunciation("ปอ ปลา", eng) == Pronunciation(
        syllables=PO + PLA, corroboration="disputed")


def test_a_phrase_with_one_unreadable_token_is_no_reading_for_that_engine():
    eng = _token_engines({"ปอ": PO})   # "ปลา" has no entry -> None
    assert eng.readings("ปอ ปลา") == ()
    assert engines_pronunciation("ปอ ปลา", eng) is None


def test_two_engines_agreeing_token_wise_is_engines_agree():
    eng = Engines(g2p=(lambda t: {"ปอ": PO, "ปลา": PLA}.get(t),
                       lambda t: {"ปอ": PO, "ปลา": PLA}.get(t)),
                  tone=lambda t: None)
    assert engines_pronunciation("ปอ ปลา", eng) == Pronunciation(
        syllables=PO + PLA, corroboration="engines_agree")


def test_a_verdict_matching_the_token_wise_reading_corroborates():
    eng = _token_engines({"ปอ ปลา": WRONG, "ปอ": PO, "ปลา": PLA})
    assert corroborates(PO + PLA, "ปอ ปลา", eng) is True
    assert corroborates(WRONG, "ปอ ปลา", eng) is False


def test_a_single_token_form_the_engine_cannot_read_is_still_no_pronunciation():
    assert engines_pronunciation("ๆ", _token_engines({})) is None


def test_a_whitespace_only_form_is_no_reading_even_when_engines_would_agree():
    """`thai.split()` on an all-whitespace form is `[]` -- no tokens, not
    one empty token -- so this must not read as an empty reading both
    engines "agree" on: a Word is never written with an empty syllable
    tuple."""
    eng = Engines(g2p=(lambda t: PO, lambda t: PO), tone=lambda t: None)
    assert eng.readings("  ") == ()
    assert engines_pronunciation("  ", eng) is None


# --- agreement is pairwise (design 2026-09-20 §3) ---------------------------

def test_agreement_between_the_second_and_third_engines_is_engines_agree():
    a = (S("k", "a", "", "short", "mid"),)
    b = (S("k", "a", "", "long", "mid"),)
    eng = Engines(g2p=(lambda t: a, lambda t: b, lambda t: b), tone=lambda t: None)
    assert engines_pronunciation("กะ", eng) == Pronunciation(
        syllables=b, corroboration="engines_agree")


def test_three_engines_all_differing_write_the_first_as_disputed():
    a = (S("k", "a", "", "short", "mid"),)
    b = (S("k", "a", "", "long", "mid"),)
    c = (S("k", "a", "", "long", "low"),)
    eng = Engines(g2p=(lambda t: a, lambda t: b, lambda t: c), tone=lambda t: "mid")
    assert engines_pronunciation("กะ", eng) == Pronunciation(
        syllables=a, corroboration="disputed")


# --- degenerate readings: the neural g2p's decoder loop (2026-09-18) --------
# Evidence from the live deck: thaig2p reads ภาพวาด as eleven syllables
# ("pʰa pʰa wa wa wa wa wa wa wa wa wa") and ษอ ฤๅษี as eleven, and
# migrate wrote eleven such readings into words.yaml as curated_exception,
# the one label the adjudication pass never revisits.

def _syl(segments, length="short", tone="mid"):
    return Syllable(segments=segments, vowel_length=length, tone=tone)


def test_three_identical_syllables_in_a_row_is_a_decoder_loop():
    wa = _syl(("w", "a", ""), "long")
    assert is_degenerate((wa, wa, wa), "ภาพวาด") is True


def test_reduplication_of_two_syllables_is_not_degenerate():
    """ๆ repeats a word exactly once (ข้างๆ kʰâːŋ.kʰâːŋ), and ตุ๊กตุ๊ก
    túk.túk is spelt with the repeat: two identical syllables are Thai,
    not a loop."""
    tuk = _syl(("t", "u", "k"), "short", "high")
    assert is_degenerate((tuk, tuk), "ตุ๊กตุ๊ก") is False


def test_more_syllables_than_the_spelling_has_consonants_is_degenerate():
    """ภาพวาด is four consonant letters; no reading of it has eleven
    syllables, since every Thai syllable needs at least one."""
    wa = _syl(("w", "a", ""), "long")
    assert is_degenerate(tuple(wa for _ in range(11)), "ภาพวาด") is True


def test_a_long_honest_reading_is_not_degenerate():
    """ออกกำลังกาย: four syllables, eight consonant letters, none repeated
    three times over."""
    reading = (_syl(("ʔ", "ɔ", "k"), "long", "low"), _syl(("k", "a", "m")),
               _syl(("l", "a", "ŋ")), _syl(("k", "a", "j"), "long"))
    assert is_degenerate(reading, "ออกกำลังกาย") is False


def test_a_degenerate_whole_form_reading_is_no_pronunciation_at_all():
    """The caller then reports the row it could not adopt, exactly as when
    the engine read nothing (spec 3 r43): a looped reading is not a
    reading."""
    wa = _syl(("w", "a", ""), "long")
    assert engines_pronunciation("ภาพวาด", _engines(tuple(wa for _ in range(11)))) is None


def test_a_degenerate_token_wise_reading_is_no_pronunciation_at_all():
    """ษอ ฤๅษี, adopted 2026-09-17 with eleven syllables: the token-wise
    fallback concatenates per-token readings, so the loop has to be caught
    on the combined result too."""
    si = _syl(("s", "i", ""), "long")
    eng = _token_engines({"ษอ": (_syl(("s", "ɔ", ""), "long", "rising"),),
                          "ฤๅษี": tuple(si for _ in range(10))})
    assert engines_pronunciation("ษอ ฤๅษี", eng) is None



def test_a_judge_verdict_is_normalized_on_the_way_in():
    """syllables_from_verdict is the one funnel a verdict passes through,
    for comparison in corroborates and for what the adjudication pass
    writes, so normalizing here covers both (design §5)."""
    got = syllables_from_verdict(
        {"syllables": [{"segments": ["pʰ", "a", "ʔ"], "vowel_length": "short",
                        "tone": "high"}]})
    assert got == (Syllable(segments=("pʰ", "a", ""), vowel_length="short",
                            tone="high"),)


# --- two segmental engines (design 2026-09-18 §1-§3) ----------------------
# Evidence: on the live deck 78 of 140 stuck judge verdicts corroborate
# against tltk and not thaig2p, and the run logged "141 of 141 verdicts not
# corroborated by an engine" on five consecutive cycles before this.

def _two(first, second, tone_result="mid"):
    return Engines(g2p=(lambda thai: first, lambda thai: second),
                   tone=lambda thai: tone_result)


def test_a_verdict_the_second_engine_agrees_with_is_corroborated():
    judge = (Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),)
    wrong = (Syllable(segments=("pʰ", "a", ""), vowel_length="short", tone="high"),)
    assert corroborates(judge, "ภาพ", _two(wrong, judge)) is True


def test_a_verdict_no_engine_agrees_with_is_not_corroborated():
    """Both engines must differ from the verdict AND from each other --
    two identical engines would also pass against a broken implementation
    that only ever consults the first one."""
    judge = (Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),)
    wrong = (Syllable(segments=("pʰ", "a", ""), vowel_length="short", tone="high"),)
    other_wrong = (Syllable(segments=("w", "a", "t"), vowel_length="long", tone="rising"),)
    assert corroborates(judge, "ภาพ", _two(wrong, other_wrong)) is False


def test_two_engines_agreeing_is_engines_agree_without_the_judge():
    """A multi-syllable form could never be engines_agree before: the rule
    tone engine settles one syllable only."""
    two = (Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),
           Syllable(segments=("w", "a", "t"), vowel_length="long", tone="falling"))
    got = engines_pronunciation("ภาพวาด", _two(two, two))
    assert got == Pronunciation(syllables=two, corroboration="engines_agree")


def test_engines_disagreeing_stays_disputed_for_the_judge():
    a = (Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),)
    b = (Syllable(segments=("pʰ", "a", ""), vowel_length="short", tone="high"),)
    got = engines_pronunciation("ภาพ", _two(a, b, tone_result="rising"))
    assert got == Pronunciation(syllables=a, corroboration="disputed")


def test_the_first_engines_reading_is_the_one_written_on_disagreement():
    a = (Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),)
    b = (Syllable(segments=("w", "a", "t"), vowel_length="long", tone="falling"),)
    assert engines_pronunciation("ภาพ", _two(a, b, "rising")).syllables == a


def test_a_degenerate_reading_does_not_count_as_agreement():
    """Both engines looping the same way must not become engines_agree."""
    loop = tuple(Syllable(segments=("w", "a", ""), vowel_length="long", tone="mid")
                 for _ in range(11))
    assert engines_pronunciation("ภาพวาด", _two(loop, loop)) is None


def test_a_lone_monosyllable_the_tone_engine_confirms_does_not_win_over_a_contradicting_second_engine():
    """FIX 2 (design 2026-09-18): the rule-tone fallback is single-engine
    behaviour -- it may only confirm a reading no other engine was there to
    contradict. Shape: tltk's real truncation can collapse a two-syllable
    word to a wrong single syllable (สิบเอ็ด: 'si2.', sìp.ʔèt truncated to
    'si'); if the rule-tone engine happens to agree with that wrong
    syllable's tone, the pre-fix code granted `engines_agree` even though
    the other, correct two-syllable reading never confirmed it -- only the
    tone rule did, unopposed by nothing since the second reading was
    ignored entirely once it failed the whole-match check."""
    truncated = (Syllable(segments=("s", "i", ""), vowel_length="short", tone="low"),)
    correct = (Syllable(segments=("s", "i", "p"), vowel_length="short", tone="low"),
              Syllable(segments=("ʔ", "e", "t"), vowel_length="short", tone="low"))
    eng = _two(truncated, correct, tone_result="low")
    got = engines_pronunciation("สิบเอ็ด", eng)
    assert got == Pronunciation(syllables=truncated, corroboration="disputed")


def test_the_second_engine_alone_still_reads_a_word_the_first_cannot():
    one = (Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),)
    got = engines_pronunciation("ภาพ", _two(None, one, tone_result="rising"))
    assert got == Pronunciation(syllables=one, corroboration="disputed")


def test_a_single_engines_degenerate_reading_does_not_corroborate_even_a_matching_verdict():
    """`readings()` drops a degenerate reading before `corroborates` ever
    compares it -- a behaviour change from before this task, when
    `corroborates` called `engines.g2p(thai)` directly and never consulted
    `is_degenerate`: a judge verdict that happened to match thaig2p's own
    decoder loop used to corroborate. It no longer does, for a
    single-engine `Engines` exactly as for a multi-engine one -- the
    change can only leave a word `disputed`, never wrongly corroborate
    one, but it IS a change, and this pins it."""
    loop = tuple(Syllable(segments=("w", "a", ""), vowel_length="long", tone="mid")
                 for _ in range(3))
    single = Engines(g2p=(lambda thai: loop,), tone=lambda thai: "mid")
    assert corroborates(loop, "ภาพวาด", single) is False


def test_a_degenerate_first_engine_does_not_force_engines_agree_with_the_second():
    """The dangerous mixed case: engine A's reading is degenerate and
    dropped, leaving engine B as the sole survivor. That must not read as
    "the surviving engine agreeing with itself" -- `readings[1:]` is empty,
    so `any(...)` is False and the result falls through to the ordinary
    single-reading rule, `disputed` here since it is two syllables. This
    is the exact failure mode ("one engine's say-so treated as
    corroboration") the whole plan exists to prevent."""
    loop = tuple(Syllable(segments=("w", "a", ""), vowel_length="long", tone="mid")
                 for _ in range(11))
    sound = (Syllable(segments=("pʰ", "a", "p"), vowel_length="long", tone="falling"),
             Syllable(segments=("w", "a", "t"), vowel_length="long", tone="falling"))
    got = engines_pronunciation("ภาพวาด", _two(loop, sound, tone_result="rising"))
    assert got == Pronunciation(syllables=sound, corroboration="disputed")


@pytest.mark.integration
def test_default_engines_wires_both_segmental_oracles():
    from thai_syllabus.phonology import default_engines as real
    real.cache_clear()
    try:
        eng = real()
        assert len(eng.g2p) == 2
        # the case the second engine exists for
        assert len(eng.readings("ภาพวาด")) >= 1
        assert all(len(r) == 2 for r in eng.readings("ภาพวาด"))
    finally:
        real.cache_clear()


# --- the dictionary is consulted lazily (design 2026-09-20 §2) -------------

def _counting_dictionary(readings):
    calls = []
    def dictionary(thai):
        calls.append(thai)
        return readings
    return dictionary, calls


A = (S("k", "a", "", "short", "mid"),)
B = (S("k", "a", "", "long", "mid"),)


def test_the_dictionary_is_not_consulted_when_the_local_engines_agree():
    d, calls = _counting_dictionary((B,))
    eng = Engines(g2p=(lambda t: A, lambda t: A), tone=lambda t: None, dictionary=d)
    assert engines_pronunciation("กะ", eng) == Pronunciation(syllables=A, corroboration="engines_agree")
    assert calls == []


def test_the_dictionary_settles_a_local_disagreement_by_agreeing_with_one_engine():
    d, calls = _counting_dictionary((B,))
    eng = Engines(g2p=(lambda t: A, lambda t: B), tone=lambda t: None, dictionary=d)
    assert engines_pronunciation("กะ", eng) == Pronunciation(syllables=B, corroboration="engines_agree")
    assert calls == ["กะ"]


def test_a_dictionary_reading_alone_is_written_disputed_when_no_engine_reads():
    d, calls = _counting_dictionary((B,))
    eng = Engines(g2p=(lambda t: None,), tone=lambda t: None, dictionary=d)
    assert engines_pronunciation("ชานมไข่มุก", eng) == Pronunciation(syllables=B, corroboration="disputed")


def test_the_dictionary_is_not_consulted_when_a_local_reading_corroborates():
    d, calls = _counting_dictionary((B,))
    eng = Engines(g2p=(lambda t: A,), tone=lambda t: None, dictionary=d)
    assert corroborates(A, "กะ", eng) is True
    assert calls == []


def test_the_dictionary_corroborates_a_verdict_no_local_engine_matches():
    d, calls = _counting_dictionary((A, B))
    eng = Engines(g2p=(lambda t: A,), tone=lambda t: None, dictionary=d)
    assert corroborates(B, "กะ", eng) is True
    assert calls == ["กะ"]
    assert corroborates((S("k", "o", "", "short", "mid"),), "กะ", eng) is False


def test_the_dictionary_reading_is_normalized_and_degeneracy_checked():
    glottal = (S("tɕ", "a", "ʔ", "short", "low"),)
    d, _ = _counting_dictionary((glottal,))
    eng = Engines(g2p=(lambda t: None,), tone=lambda t: None, dictionary=d)
    assert eng.dictionary_readings("จะ") == ((S("tɕ", "a", "", "short", "low"),),)
    loop = (S("w", "a", "", "short", "mid"),) * 4
    d2, _ = _counting_dictionary((loop,))
    eng2 = Engines(g2p=(lambda t: None,), tone=lambda t: None, dictionary=d2)
    assert eng2.dictionary_readings("วา") == ()


def test_a_phrase_never_reaches_the_dictionary():
    d, calls = _counting_dictionary((B,))
    eng = Engines(g2p=(lambda t: None,), tone=lambda t: None, dictionary=d)
    assert eng.dictionary_readings("งอ งู") == ()
    assert calls == []


def test_engines_without_a_dictionary_behave_as_before():
    eng = Engines(g2p=(lambda t: A, lambda t: B), tone=lambda t: None)
    assert eng.dictionary is None
    assert engines_pronunciation("กะ", eng) == Pronunciation(syllables=A, corroboration="disputed")


def test_two_dictionary_spans_that_collapse_to_one_reading_do_not_agree_with_each_other():
    """The dictionary's own variants are ONE opinion. English Wiktionary
    lists several IPA spans per entry, and two of them can normalize to
    the same reading (a /tɕaʔ/ and a /tɕa/ span collapse under the ʔ-coda
    convention). `_agreed` returns any reading two entries of the tuple
    share, so an undeduplicated pair would seal `engines_agree` -- which
    is terminal (`is_corroborated`) -- on the dictionary's say-so alone,
    with no local engine involved. It stays `disputed`, for the judge."""
    d, _ = _counting_dictionary(((S("tɕ", "a", "ʔ", "short", "low"),),
                                 (S("tɕ", "a", "", "short", "low"),)))
    eng = Engines(g2p=(lambda t: None,), tone=lambda t: None, dictionary=d)
    one = (S("tɕ", "a", "", "short", "low"),)
    assert eng.dictionary_readings("จะ") == (one,)
    assert engines_pronunciation("จะ", eng) == Pronunciation(syllables=one,
                                                             corroboration="disputed")
