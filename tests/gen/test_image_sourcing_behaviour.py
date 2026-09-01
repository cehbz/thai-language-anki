"""What an image run must do, stated without reference to how it does it.

These describe observable behaviour of a run: which corpora get consulted,
which words get retried, what the operator is told. Nothing here names a
fingerprint, a cache file, or a channel constant -- if the implementation
changes shape, these should still hold.
"""
import yaml

from thai_deck_eval.judge.core import Verdict
from thai_deck_gen.deckio import write_deck
from thai_deck_gen.media.images import fill_images
from thai_deck_gen.media.manifest import Manifest
from thai_deck_gen.media.scan import ImageNeed
from tests.gen.test_images import R, FakeFetch, _deck_with_pw, _jpeg_bytes
from tests.gen.test_pairs import _gaps


class RecordingWorld:
    """A pretend internet: records which corpora were asked, serves images."""

    def __init__(self, serving: set[str]):
        self.serving = serving          # corpora that return a usable result
        self.asked: list[str] = []
        self.imgfetch = FakeFetch({f"http://{s}/img.jpg": _jpeg_bytes()
                                   for s in ("openverse", "wikimedia", "pexels")})
        self.imagegen = None
        self.image_query_hints = {}
        self.image_candidates = 5
        self.pexels_key = "KEY" if "pexels" in serving else None

    def http_get(self, url, timeout=30, headers=None):
        corpus = ("pexels" if "pexels.com" in url
                  else "openverse" if "openverse" in url else "wikimedia")
        self.asked.append(corpus)
        if corpus not in self.serving:
            return R(payload={"results": [], "photos": [], "query": {"pages": {}}})
        if corpus == "pexels":
            return R(payload={"photos": [{"alt": "a", "src": {"large": "http://pexels/img.jpg"}}]})
        if corpus == "openverse":
            return R(payload={"results": [{"url": "http://openverse/img.jpg", "license": "cc0"}]})
        return R(payload={"query": {"pages": {"1": {"imageinfo": [{"url": "http://wikimedia/img.jpg"}]}}}})


class Judge:
    def __init__(self, accept: bool, reject: set[str] | None = None):
        self.accept = accept
        self.reject = reject or set()      # request ids this judge turns down
        self.prompts: dict[str, str] = {}

    def judge_many(self, reqs):
        self.prompts.update({r.note_id: r.prompt for r in reqs})
        return {r.note_id: [Verdict(rule="judge/image-irrelevant",
                                    passed=self.accept and r.note_id not in self.reject,
                                    confidence=0.9, rationale="")] for r in reqs}


def _need(category="Pronouns"):
    return ImageNeed(family="picture_word", note_id="pw-0", term="ฉัน", gloss="I",
                     category=category, image_query="person pointing at themselves",
                     path="images/pw-0.jpg")


def _deck(tmp_path, category="Pronouns"):
    deck = _deck_with_pw(tmp_path, term="ฉัน", gloss="I")
    deck.picture_words[0].category = category
    write_deck(deck)
    return deck


def test_a_word_written_off_is_retried_when_a_new_corpus_becomes_available(tmp_path):
    """The point of adding a library is that words nothing could serve get
    another chance. A run that skips them has wasted the addition."""
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                "2026-08-31", judge=Judge(accept=False))

    better = RecordingWorld(serving={"pexels"})
    res = fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), better,
                      "2026-09-01", judge=Judge(accept=True))

    assert "pexels" in better.asked, "the new corpus was never consulted"
    assert res.changed == 1, "the word was not recovered by the new corpus"


def test_abstract_words_ask_the_concept_library_first(tmp_path):
    deck = _deck(tmp_path, category="Pronouns")
    world = RecordingWorld(serving={"pexels", "openverse"})
    fill_images([_need("Pronouns")], _gaps([]), deck, Manifest.load(deck.root),
                world, "2026-08-31", judge=Judge(accept=True))
    assert world.asked[0] == "pexels"


def test_concrete_words_ask_the_amateur_library_first(tmp_path):
    deck = _deck(tmp_path, category="Food")
    world = RecordingWorld(serving={"pexels", "openverse"})
    fill_images([_need("Food")], _gaps([]), deck, Manifest.load(deck.root),
                world, "2026-08-31", judge=Judge(accept=True))
    assert world.asked[0] == "openverse"


def test_a_source_that_cannot_run_is_reported(tmp_path, capsys):
    """Silent degradation is the failure mode that cost three smoke runs."""
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})      # no pexels key
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                "2026-08-31", judge=Judge(accept=True))
    assert "pexels" in capsys.readouterr().out.lower()


def test_a_word_is_not_re_searched_when_nothing_has_changed(tmp_path):
    """The memo still has to work: same corpora, same queries, no repeat."""
    deck = _deck(tmp_path)
    first = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), first,
                "2026-08-31", judge=Judge(accept=False))

    again = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), again,
                "2026-09-01", judge=Judge(accept=False))
    assert again.asked == [], "a settled word was searched again for nothing"


def test_a_changed_rubric_re_scores_without_re_downloading(tmp_path, monkeypatch):
    """Keeping rejected candidates exists so a relaxed rule can reconsider
    them; re-searching is the slow half of a run."""
    import thai_deck_eval.judge.prompts as prompts
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})
    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                "2026-08-31", judge=Judge(accept=False))
    downloads_before = len(world.imgfetch.urls)
    assert downloads_before > 0

    monkeypatch.setitem(prompts.PICTURE_RULES, "judge/image-embedded-text",
                        "a materially more permissive rubric")
    again = RecordingWorld(serving={"openverse"})
    res = fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), again,
                      "2026-09-01", judge=Judge(accept=True))

    assert res.changed == 1, "the relaxed rubric did not reconsider the word"
    assert again.imgfetch.urls == [], "candidates were downloaded a second time"


def test_a_run_without_a_limit_attempts_every_word(tmp_path):
    """A full run means all of them: silently doing five is indistinguishable
    from success in the summary line."""
    deck = _deck(tmp_path)
    from thai_deck_eval.model.notes import Audio, PictureWordNote
    for i in range(8):
        deck.picture_words.append(PictureWordNote(
            id=f"pw-1{i}", thai=f"w{i}", image=f"images/pw-1{i}.jpg",
            audio=Audio(file=f"audio/pw-1{i}.mp3", source="native",
                        speaker="pending"),
            frequency_rank=i + 2, category="Food"))
    write_deck(deck)

    needs = [ImageNeed(family="picture_word", note_id=n.id, term=n.thai,
                       gloss="g", category=n.category, path=n.image)
             for n in deck.picture_words]
    judge = Judge(accept=True)
    world = RecordingWorld(serving={"openverse"})
    res = fill_images(needs, _gaps([]), deck, Manifest.load(deck.root), world,
                      "2026-08-31", judge=judge)
    assert res.changed == len(needs), f"only {res.changed} of {len(needs)} attempted"


def test_a_limit_caps_words_not_candidates(tmp_path):
    deck = _deck(tmp_path)
    from thai_deck_eval.model.notes import Audio, PictureWordNote
    for i in range(4):
        deck.picture_words.append(PictureWordNote(
            id=f"pw-1{i}", thai=f"w{i}", image=f"images/pw-1{i}.jpg",
            audio=Audio(file=f"audio/pw-1{i}.mp3", source="native",
                        speaker="pending"),
            frequency_rank=i + 2, category="Food"))
    write_deck(deck)
    needs = [ImageNeed(family="picture_word", note_id=n.id, term=n.thai,
                       gloss="g", category=n.category, path=n.image)
             for n in deck.picture_words]
    world = RecordingWorld(serving={"openverse"})
    res = fill_images(needs, _gaps([]), deck, Manifest.load(deck.root), world,
                      "2026-08-31", judge=Judge(accept=True), limit=2)
    assert res.changed == 2


def _plant_picture(deck, path="images/pw-0.jpg"):
    """The picture the deck already carries, as a run would find it."""
    dest = deck.root / "media" / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_jpeg_bytes())
    return dest


def test_a_word_whose_picture_still_passes_is_not_searched_again(tmp_path):
    """No report flagged it and it is still good, so the run owes it nothing.
    Searching anyway spends the network on settled words."""
    deck = _deck(tmp_path)
    _plant_picture(deck)
    world = RecordingWorld(serving={"openverse"})

    res = fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                      "2026-08-31", judge=Judge(accept=True))

    assert world.asked == [], "a picture that still passes was searched again"
    assert res.changed == 0 and res.blocked == []


def test_a_picture_the_judge_now_rejects_is_replaced_in_the_same_run(tmp_path):
    """Changing what the judge accepts must reach the deck immediately.
    Waiting for the next report to flag the word is the second cycle."""
    deck = _deck(tmp_path)
    _plant_picture(deck)
    world = RecordingWorld(serving={"openverse"})
    judge = Judge(accept=True, reject={"pw-0#current"})

    res = fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                      "2026-08-31", judge=judge)

    assert "pw-0#current" in judge.prompts, "the deck's own picture was never judged"
    assert world.asked, "the rejected picture was never re-searched"
    assert res.changed == 1


def test_the_picture_in_the_deck_is_judged_against_the_current_phrase(tmp_path):
    """The verdict cache keys on the prompt, so carrying the phrase into the
    incumbent's judgment is what makes a new phrase cost a fresh verdict."""
    deck = _deck(tmp_path)
    _plant_picture(deck)
    judge = Judge(accept=True)

    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root),
                RecordingWorld(serving={"openverse"}), "2026-08-31", judge=judge)

    assert "person pointing at themselves" in judge.prompts["pw-0#current"]


def test_an_approved_picture_survives_a_run_that_judges_it_again(tmp_path):
    """Approval is of one artifact by a person who looked at it. A run that
    scores the deck's own pictures must not overturn that and re-fetch."""
    import hashlib
    from thai_deck_eval.waivers import Waiver, save_waivers

    deck = _deck(tmp_path)
    approved = _plant_picture(deck).read_bytes()
    save_waivers(deck.root, [Waiver(
        note_id="pw-0", rule="judge/image-irrelevant",
        reason="the only photograph of this that exists", date="2026-08-31",
        sha=hashlib.sha256(approved).hexdigest())])
    world = RecordingWorld(serving={"openverse"})

    res = fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                      "2026-08-31", judge=Judge(accept=False))

    assert world.asked == [], "an approved picture was re-searched"
    assert res.changed == 0
    assert (deck.root / "media" / "images" / "pw-0.jpg").read_bytes() == approved


def test_an_approval_does_not_transfer_to_a_different_picture(tmp_path):
    """The waiver names a sha: swap the file and the approval stops applying,
    or one review would license every later image on that word."""
    from thai_deck_eval.waivers import Waiver, save_waivers

    deck = _deck(tmp_path)
    _plant_picture(deck)
    save_waivers(deck.root, [Waiver(
        note_id="pw-0", rule="judge/image-irrelevant", reason="reviewed",
        date="2026-08-31", sha="0" * 64)])
    world = RecordingWorld(serving={"openverse"})

    fill_images([_need()], _gaps([]), deck, Manifest.load(deck.root), world,
                "2026-08-31", judge=Judge(accept=True, reject={"pw-0#current"}))

    assert world.asked, "a stale approval suppressed the re-search"


# --- the audit: "no picture can represent this" put to the corpora ---

def _written_off(thai="ฤดู", gloss="season", category="Time"):
    from thai_deck_gen.wordlist import WordEntry
    import re
    return WordEntry(id=re.sub(r"[^a-z0-9]+", "-", gloss.lower()).strip("-"),
                     thai=thai, gloss=gloss, category=category,
                     part_of_speech="noun", classifier="ฤดู", picturable=False)


def test_the_audit_names_the_picture_that_serves_a_written_off_word(tmp_path):
    """'No picture can represent this' was decided by a model that searched
    for nothing. The audit searches, judges what comes back, and reports the
    picture, so the claim is answered with an artifact rather than a count."""
    from thai_deck_gen.media.images import audit_picturable
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})

    found = audit_picturable([_written_off()], deck, world, Judge(accept=True), {})

    assert set(found) == {"ฤดู"}
    assert found["ฤดู"].source == "openverse"
    assert found["ฤดู"].url


def test_the_audit_reports_nothing_for_a_word_the_judge_turns_down(tmp_path):
    """Results existing is not a picture. A count says 'Monday' is findable;
    only the judge says whether what came back depicts Monday."""
    from thai_deck_gen.media.images import audit_picturable
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})

    assert audit_picturable([_written_off()], deck, world,
                            Judge(accept=False), {}) == {}


def test_the_audit_leaves_alone_the_words_that_already_have_cards(tmp_path):
    """A word carrying a card is not the audit's subject, and searching for
    it spends the network on a settled question."""
    from thai_deck_gen.media.images import audit_picturable
    from thai_deck_gen.wordlist import WordEntry
    deck = _deck(tmp_path)
    world = RecordingWorld(serving={"openverse"})

    audit_picturable([WordEntry(id="dog", thai="หมา", gloss="dog", category="Animals",
                                part_of_speech="noun", classifier="ตัว")],
                     deck, world, Judge(accept=True), {})

    assert world.asked == []
