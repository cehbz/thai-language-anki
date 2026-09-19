"""The pure pair search (design 2026-09-12 section 2; spec 3 r47)."""
from thai_syllabus.entities import MinimalPair, Pronunciation, SoundConfusion, Syllable
from thai_syllabus.ids import ConfusionId, PairId, WordId
from thai_syllabus.pairsearch import Candidate, pair_id_for, select_pairs, wanted

TONE = SoundConfusion(id=ConfusionId("tone:mid-low"), dimension="tone",
                      sounds=("mid", "low"), weight=5)
LENGTH = SoundConfusion(id=ConfusionId("vowel_length:short-long"), dimension="length",
                        sounds=("short", "long"), weight=4)
# The codas the engines store are bare (engines._CODAS is p/t/k, no
# unreleased diacritic), so a final:place-* confusion's sounds are bare
# too -- C1: the live deck declared p̚/t̚/k̚ and could match nothing.
FINAL = SoundConfusion(id=ConfusionId("final:place-p-t"), dimension="final",
                       sounds=("p", "t"), weight=4)
CONSONANT = SoundConfusion(id=ConfusionId("consonant:place-p-t"), dimension="consonant",
                           sounds=("p", "t"), weight=4)


def pron(tone="mid", length="short", onset="k", vowel="a", coda="", corroboration="engines_agree"):
    return Pronunciation(syllables=(Syllable(segments=(onset, vowel, coda), vowel_length=length,
                                             tone=tone),), corroboration=corroboration)


def cand(thai, pron_, word_id=None, rank=None):
    return Candidate(thai=thai, pron=pron_, word_id=WordId(word_id) if word_id else None,
                     rank=rank)


def test_wanted_is_pair_count_minus_the_pairs_already_adopted():
    have = [MinimalPair(id=PairId("tone:mid-low/a-b"), confusion=TONE.id, members=("a", "b"))]
    assert wanted([TONE, LENGTH], have) == {TONE.id: 3, LENGTH.id: 3}


def test_select_pairs_forms_exact_pairs_only():
    cs = [cand("กา", pron("mid"), "crow"), cand("ก่า", pron("low")),
          cand("ข่า", pron("low", onset="kʰ"))]            # differs in onset too
    got = select_pairs(TONE, cs, wanted=4, taken=())
    assert [(a.thai, b.thai) for a, b in got] == [("กา", "ก่า")]


def test_select_pairs_prefers_vocabulary_members_then_shared_speaker_then_rank():
    cs = [cand("กา", pron("mid"), "crow", rank=900),
          cand("ก่า", pron("low"), rank=50),
          cand("ตา", pron("mid"), "eye", rank=10),
          cand("ต่า", pron("low"), "ta2", rank=4000),
          cand("มา", pron("mid"), "ma", rank=5000),
          cand("ม่า", pron("low"), "ma2", rank=6000)]
    shares = lambda a, b: {a.thai, b.thai} == {"มา", "ม่า"}
    got = select_pairs(TONE, cs, wanted=4, taken=(), shares=shares)
    # มา/ม่า: both Words and a shared speaker, ahead of ตา/ต่า (both Words,
    # better rank, no shared speaker), ahead of กา/ก่า (one Word).
    assert [(a.thai, b.thai) for a, b in got] == [("มา", "ม่า"), ("ตา", "ต่า"), ("กา", "ก่า")]


def test_select_pairs_uses_no_member_twice_and_skips_taken_forms():
    cs = [cand("กา", pron("mid"), "crow"), cand("ก่า", pron("low")), cand("ก้า", pron("low"))]
    assert len(select_pairs(TONE, cs, wanted=4, taken=())) == 1
    assert select_pairs(TONE, cs, wanted=4, taken=("กา",)) == []


def test_select_pairs_ignores_an_uncorroborated_candidate():
    cs = [cand("กา", pron("mid", corroboration="disputed"), "crow"), cand("ก่า", pron("low"))]
    assert select_pairs(TONE, cs, wanted=1, taken=()) == []


def test_select_pairs_stops_at_wanted():
    cs = [cand("กา", pron("mid")), cand("ก่า", pron("low")),
          cand("ตา", pron("mid")), cand("ต่า", pron("low"))]
    assert len(select_pairs(TONE, cs, wanted=1, taken=())) == 1


def test_pair_id_names_word_ids_when_known_else_the_forms():
    a, b = cand("ไกล", pron("low"), "far"), cand("ใกล้", pron("mid"))
    assert pair_id_for(TONE, a, b) == PairId("tone:mid-low/far-ใกล้")


def test_select_pairs_ties_break_by_the_pairs_thai_forms():
    cs = [cand("กา", pron("mid")), cand("ก่า", pron("low")),
          cand("ขา", pron("mid", onset="kʰ")), cand("ข่า", pron("low", onset="kʰ"))]
    got = select_pairs(TONE, cs, wanted=4, taken=())
    # Equal on words (none), shares (none) and rank sum (all unranked): the
    # tiebreak is the pair's Thai forms.
    assert [(a.thai, b.thai) for a, b in got] == [("กา", "ก่า"), ("ขา", "ข่า")]


def test_select_pairs_buckets_instead_of_scanning_every_pair():
    # 3000 candidates with distinct vowels: none of them can form an exact
    # pair with each other or with the real pair, so bucketing by the
    # masked pronunciation must keep this linear, not quadratic, and must
    # still find the one real pair.
    noise = [cand(f"noise{i}", pron("mid", vowel=f"v{i}")) for i in range(3000)]
    real = [cand("กา", pron("mid"), "crow"), cand("ก่า", pron("low"))]
    got = select_pairs(TONE, noise + real, wanted=10, taken=())
    assert [(a.thai, b.thai) for a, b in got] == [("กา", "ก่า")]


def _two_syllables(first_tone):
    """A two-syllable pronunciation whose second syllable reads `falling`
    -- not one of tone:mid-low's sounds -- whatever the first reads."""
    return Pronunciation(
        syllables=(Syllable(segments=("m", "a", ""), vowel_length="short", tone=first_tone),
                   Syllable(segments=("n", "a", ""), vowel_length="short", tone="falling")),
        corroboration="engines_agree")


def test_select_pairs_reads_the_syllable_that_differs_in_a_polysyllabic_pair():
    # I2: the confusion's sounds are read where the members differ, not at
    # the last syllable (which reads `falling` for both here). มานะ:
    # perseverance; ม่านะ: a nonce partner, its first syllable low-toned.
    cs = [cand("มานะ", _two_syllables("mid")),
          cand("ม่านะ", _two_syllables("low"))]
    assert [(a.thai, b.thai) for a, b in select_pairs(TONE, cs, wanted=4, taken=())] == [
        ("มานะ", "ม่านะ")]
    # The same two forms under a confusion whose sounds they do not carry:
    # no pair, though the shared last syllable reads the same for both.
    other = SoundConfusion(id=ConfusionId("tone:high-rising"), dimension="tone",
                           sounds=("high", "rising"), weight=4)
    assert select_pairs(other, cs, wanted=4, taken=()) == []


def test_select_pairs_finds_a_coda_only_pair_under_final():
    cs = [cand("กาบ", pron(coda="p")), cand("กาด", pron(coda="t"))]
    got = select_pairs(FINAL, cs, wanted=4, taken=())
    assert [(a.thai, b.thai) for a, b in got] == [("กาบ", "กาด")]


def test_select_pairs_ignores_the_same_coda_only_pair_under_consonant():
    # Onset-identical, coda-different: a "final" difference, not
    # "consonant" (spec 1 r20) -- exact_confusion_violation rejects it
    # under "consonant" (the value it checks is the onset), and
    # select_pairs must agree, not just its bucketing shortcut.
    cs = [cand("กาบ", pron(coda="p")), cand("กาด", pron(coda="t"))]
    assert select_pairs(CONSONANT, cs, wanted=4, taken=()) == []
