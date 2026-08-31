import io
from urllib.parse import quote
from PIL import Image
import yaml
from thai_deck_gen.media.images import (
    ImageCandidate, ImageError, OpenAiImageGen, downscale, fill_images,
    openverse_search, wikimedia_search,
)
from thai_deck_gen.media.manifest import Manifest, MediaEntry
from thai_deck_gen.media.scan import ImageNeed
from thai_deck_gen.deckio import new_deck, write_deck
from thai_deck_gen.report import GapFinding
from tests.gen.test_pairs import _gaps
from thai_deck_eval.model.notes import Audio, PictureWordNote
import pytest

class FakeFetch:
    """Stands in for ImgFetch: serves canned bytes per URL, records requests."""
    def __init__(self, by_url):
        self.by_url, self.urls = by_url, []
    def fetch(self, url):
        self.urls.append(url)
        return self.by_url.get(url)



def _jpeg_bytes(size=(1200, 800)) -> bytes:
    img = Image.new("RGB", size, color=(200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _deck_with_pw(tmp_path, term="คำ", gloss="word"):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    deck.picture_words.append(PictureWordNote(
        id="pw-0", thai=term, image="images/pw-0.jpg", gloss=gloss,
        audio=Audio(file="audio/picture_words/pw-0.mp3",
                    source="native", speaker="pending"),
        frequency_rank=1, category="Food"))
    write_deck(deck)
    return deck


class R:
    def __init__(self, status_code=200, payload=None, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


def test_downscale_shrinks_both_axes():
    raw = _jpeg_bytes((1200, 800))
    out = downscale(raw, max_px=600)
    img = Image.open(io.BytesIO(out))
    assert img.width <= 600 and img.height <= 600
    assert img.format == "JPEG"


def test_openverse_search_parses_results():
    def http_get(url, timeout=30, headers=None):
        assert "api.openverse.org" in url
        return R(payload={"results": [{"url": "http://img/a.jpg", "license": "cc0"}]})
    cands = openverse_search("คำ", http_get)
    assert cands == [ImageCandidate(url="http://img/a.jpg", source="openverse", license="cc0")]


def test_wikimedia_search_parses_results():
    def http_get(url, timeout=30, headers=None):
        assert "commons.wikimedia.org" in url
        return R(payload={"query": {"pages": {
            "1": {"imageinfo": [{"url": "http://img/b.jpg"}]}}}})
    cands = wikimedia_search("คำ", http_get)
    assert cands == [ImageCandidate(url="http://img/b.jpg", source="wikimedia", license=None)]


def test_fill_images_queries_the_gloss_before_the_thai_term(tmp_path):
    """Openverse indexes English metadata: a Thai query matches only the few
    Thai-captioned items it holds, which are posters and book covers."""
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    thai_q = quote("คำ")
    jpeg = _jpeg_bytes()
    calls = []

    def http_get(url, timeout=30, headers=None):
        calls.append(url)
        if "api.openverse.org" in url and "word" in url:
            return R(payload={"results": [{"url": "http://img/x.jpg", "license": "cc0"}]})
        return R(payload={"results": []})

    class Ctx:
        imagegen = None
        image_query_hints = {}
        imgfetch = FakeFetch({"http://img/x.jpg": jpeg})
    Ctx.http_get = staticmethod(http_get)

    manifest = Manifest.load(deck.root)
    res = fill_images([need], _gaps([]), deck, manifest, Ctx(), "2026-08-27")

    assert res.changed == 1
    assert manifest.channel_of("media/images/pw-0.jpg") == "openverse"
    assert "word" in calls[0] and "api.openverse.org" in calls[0]
    assert thai_q not in calls[0]        # the gloss goes first now
    assert len(calls) == 1


def test_category_qualifier_disambiguates_the_gloss(tmp_path):
    """'orange' alone returns an orange tabby cat; 'orange food' does not."""
    deck = _deck_with_pw(tmp_path, term="ส้ม", gloss="orange")
    need = ImageNeed(family="picture_word", note_id="pw-0", term="ส้ม",
                      gloss="orange", category="Food", path="images/pw-0.jpg")
    calls = []

    def http_get(url, timeout=30, headers=None):
        calls.append(url)
        return R(payload={"results": []})

    class Ctx:
        imagegen = None
        image_query_hints = {"Food": "food"}
        imgfetch = FakeFetch({})
    Ctx.http_get = staticmethod(http_get)

    fill_images([need], _gaps([]), deck, Manifest.load(deck.root), Ctx(), "2026-08-27")

    openverse = [c for c in calls if "api.openverse.org" in c]
    assert quote("orange food") in openverse[0]
    assert "orange" in openverse[1] and quote("orange food") not in openverse[1]
    assert quote("ส้ม") in openverse[2]      # Thai last, for culture-specific terms


def test_fill_images_blocks_when_nothing_found(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")

    def http_get(url, timeout=30, headers=None):
        if "api.openverse.org" in url:
            return R(payload={"results": []})
        return R(payload={"query": {"pages": {}}})

    class Ctx:
        imagegen = None
        imgfetch = FakeFetch({})
    Ctx.http_get = staticmethod(http_get)

    manifest = Manifest.load(deck.root)
    res = fill_images([need], _gaps([]), deck, manifest, Ctx(), "2026-08-27")
    assert res.blocked == ["pw-0"]
    assert manifest.channel_of("media/images/pw-0.jpg") is None


def _judge_gaps(note_id, rule="judge/image_mismatch"):
    g = _gaps([])
    g.findings.append(GapFinding(rule=rule, severity="warn", note_id=note_id,
                                  message="image does not match term"))
    return g


def test_escalation_to_ai_overwrites_and_flips_channel(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    dst = deck.root / "media" / "images" / "pw-0.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"OLD-BYTES")

    manifest = Manifest.load(deck.root)
    manifest.record(MediaEntry(file="media/images/pw-0.jpg", channel="openverse",
                                origin="http://old/img.jpg", fetched="2026-08-01"))

    gaps = _judge_gaps("pw-0")

    class FakeImageGen:
        def __init__(self):
            self.prompts = []
        def generate(self, prompt):
            self.prompts.append(prompt)
            return _jpeg_bytes((900, 900))

    def boom(url, timeout=30):
        raise AssertionError("search should not run during escalation")

    fake_gen = FakeImageGen()
    class Ctx:
        http_get = staticmethod(boom)
        imagegen = fake_gen

    res = fill_images([need], gaps, deck, manifest, Ctx(), "2026-08-27")

    assert res.changed == 1
    assert dst.read_bytes() != b"OLD-BYTES"
    assert manifest.channel_of("media/images/pw-0.jpg") == "ai"
    assert manifest.entries["media/images/pw-0.jpg"].origin == "gpt-image-1"
    assert "word" in fake_gen.prompts[0]
    img = Image.open(io.BytesIO(dst.read_bytes()))
    assert img.width <= 600 and img.height <= 600


def test_escalation_falls_back_to_term_when_gloss_missing(tmp_path):
    deck = _deck_with_pw(tmp_path, term="คำ", gloss=None)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss=None, path="images/pw-0.jpg")
    dst = deck.root / "media" / "images" / "pw-0.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"OLD-BYTES")

    manifest = Manifest.load(deck.root)
    manifest.record(MediaEntry(file="media/images/pw-0.jpg", channel="openverse",
                                origin="http://old/img.jpg", fetched="2026-08-01"))
    gaps = _judge_gaps("pw-0")

    class FakeImageGen:
        def __init__(self):
            self.prompts = []
        def generate(self, prompt):
            self.prompts.append(prompt)
            return _jpeg_bytes((900, 900))

    fake_gen = FakeImageGen()
    class Ctx:
        http_get = staticmethod(lambda url, timeout=30, headers=None: (_ for _ in ()).throw(
            AssertionError("no search expected")))
        imagegen = fake_gen

    res = fill_images([need], gaps, deck, manifest, Ctx(), "2026-08-27")

    assert res.changed == 1
    assert "None" not in fake_gen.prompts[0]
    assert "คำ" in fake_gen.prompts[0]


def test_escalation_imagegen_failure_queues_review_and_continues(tmp_path):
    deck = new_deck(tmp_path / "d", "t", ["words"])
    deck.picture_words.append(PictureWordNote(
        id="pw-0", thai="คำ", image="images/pw-0.jpg", gloss="word",
        audio=Audio(file="audio/picture_words/pw-0.mp3",
                    source="native", speaker="pending"),
        frequency_rank=1, category="Food"))
    deck.picture_words.append(PictureWordNote(
        id="pw-1", thai="บ้าน", image="images/pw-1.jpg", gloss="house",
        audio=Audio(file="audio/picture_words/pw-1.mp3",
                    source="native", speaker="pending"),
        frequency_rank=2, category="Places"))
    write_deck(deck)

    need0 = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                       gloss="word", path="images/pw-0.jpg")
    need1 = ImageNeed(family="picture_word", note_id="pw-1", term="บ้าน",
                       gloss="house", path="images/pw-1.jpg")
    dst0 = deck.root / "media" / "images" / "pw-0.jpg"
    dst0.parent.mkdir(parents=True, exist_ok=True)
    dst0.write_bytes(b"OLD-BYTES-0")

    manifest = Manifest.load(deck.root)
    manifest.record(MediaEntry(file="media/images/pw-0.jpg", channel="openverse",
                                origin="http://old/img.jpg", fetched="2026-08-01"))
    gaps = _judge_gaps("pw-0")

    class FailingImageGen:
        def generate(self, prompt):
            raise ImageError("quota exceeded")

    def http_get(url, timeout=30, headers=None):
        # only need1's plain search fill should reach here
        if "api.openverse.org" in url:
            return R(payload={"results": [{"url": "http://img/y.jpg", "license": "cc0"}]})
        return R(payload={"query": {"pages": {}}})

    class Ctx:
        imgfetch = FakeFetch({"http://img/y.jpg": _jpeg_bytes()})
        imagegen = FailingImageGen()
    Ctx.http_get = staticmethod(http_get)

    res = fill_images([need0, need1], gaps, deck, manifest, Ctx(), "2026-08-27")

    assert res.blocked == ["pw-0"]
    assert res.changed == 1
    assert dst0.read_bytes() == b"OLD-BYTES-0"
    assert manifest.channel_of("media/images/pw-0.jpg") == "openverse"
    assert manifest.channel_of("media/images/pw-1.jpg") == "openverse"

    review = yaml.safe_load((deck.root / "work" / "image_review.yaml").read_text())
    assert review["items"] == [{"note_id": "pw-0", "term": "คำ", "tried": ["ai"]}]


def test_escalation_queues_review_when_imagegen_missing(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    dst = deck.root / "media" / "images" / "pw-0.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"OLD-BYTES")

    manifest = Manifest.load(deck.root)
    manifest.record(MediaEntry(file="media/images/pw-0.jpg", channel="openverse",
                                origin="http://old/img.jpg", fetched="2026-08-01"))
    gaps = _judge_gaps("pw-0")

    class Ctx:
        http_get = staticmethod(lambda url, timeout=30, headers=None: (_ for _ in ()).throw(
            AssertionError("no search expected")))
        imagegen = None

    res = fill_images([need], gaps, deck, manifest, Ctx(), "2026-08-27")

    assert res.blocked == ["pw-0"]
    assert dst.read_bytes() == b"OLD-BYTES"
    assert manifest.channel_of("media/images/pw-0.jpg") == "openverse"

    review = yaml.safe_load((deck.root / "work" / "image_review.yaml").read_text())
    assert review["items"] == [{"note_id": "pw-0", "term": "คำ", "tried": ["openverse"]}]


def test_escalation_already_ai_rejects_to_review(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    dst = deck.root / "media" / "images" / "pw-0.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"AI-BYTES")

    manifest = Manifest.load(deck.root)
    manifest.record(MediaEntry(file="media/images/pw-0.jpg", channel="ai",
                                origin="gpt-image-1", fetched="2026-08-01"))
    gaps = _judge_gaps("pw-0")

    class FakeImageGen:
        def generate(self, prompt):
            raise AssertionError("should not be called for an already-ai image")

    class Ctx:
        http_get = staticmethod(lambda url, timeout=30, headers=None: (_ for _ in ()).throw(
            AssertionError("no search expected")))
        imagegen = FakeImageGen()

    res = fill_images([need], gaps, deck, manifest, Ctx(), "2026-08-27")
    assert res.blocked == ["pw-0"]
    assert dst.read_bytes() == b"AI-BYTES"


def test_openai_image_gen_parses_b64_response():
    import base64
    payload = base64.b64encode(b"png-bytes").decode()

    def http_post(url, json, headers=None, timeout=60):
        assert json["model"] == "gpt-image-1"
        return R(payload={"data": [{"b64_json": payload}]})

    gen = OpenAiImageGen("KEY", http_post=http_post)
    assert gen.generate("a cat") == b"png-bytes"


def test_openai_image_gen_raises_on_error():
    from thai_deck_gen.media.images import ImageError

    def http_post(url, json, headers=None, timeout=60):
        return R(status_code=400, content=b"", payload=None)

    gen = OpenAiImageGen("KEY", http_post=http_post)
    with pytest.raises(ImageError):
        gen.generate("a cat")


def test_fill_images_blocks_on_network_error_and_continues(tmp_path):
    import requests
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")

    def http_get(url, timeout=30, headers=None):
        raise requests.ConnectTimeout("api.openverse.org timed out")

    class Ctx:
        imagegen = None
        imgfetch = FakeFetch({})
    Ctx.http_get = staticmethod(http_get)

    manifest = Manifest.load(deck.root)
    res = fill_images([need], _gaps([]), deck, manifest, Ctx(), "2026-08-27")
    assert res.blocked == ["pw-0"]              # per-item: blocked, not raised
    assert res.changed == 0


def test_fill_images_downloads_candidates_through_imgfetch_not_http_get(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    jpeg = _jpeg_bytes()

    def http_get(url, timeout=30, headers=None):
        if "api.openverse.org" in url:
            return R(payload={"results": [{"url": "http://img/x.jpg", "license": "cc0"}]})
        raise AssertionError(f"http_get must not download images: {url}")

    class Ctx:
        imagegen = None
        imgfetch = FakeFetch({"http://img/x.jpg": jpeg})
    Ctx.http_get = staticmethod(http_get)
    manifest = Manifest.load(deck.root)
    res = fill_images([need], _gaps([]), deck, manifest, Ctx(), "2026-08-27")
    assert res.changed == 1
    assert Ctx.imgfetch.urls == ["http://img/x.jpg"]
    assert (deck.root / "media" / "images" / "pw-0.jpg").exists()


def test_fill_images_blocks_everything_when_imgfetch_is_missing(tmp_path, capsys):
    from thai_deck_gen.media.imgfetch import ImgFetchUnavailable
    deck = _deck_with_pw(tmp_path)
    needs = [ImageNeed(family="picture_word", note_id=f"pw-{i}", term="คำ",
                       gloss="word", path=f"images/pw-{i}.jpg") for i in range(2)]

    def http_get(url, timeout=30, headers=None):
        return R(payload={"results": [{"url": "http://img/x.jpg", "license": "cc0"}]})

    class Missing:
        def fetch(self, url):
            raise ImgFetchUnavailable("imgfetch not found at /nope/imgfetch")

    class Ctx:
        imagegen = None
        imgfetch = Missing()
    Ctx.http_get = staticmethod(http_get)
    res = fill_images(needs, _gaps([]), deck, Manifest.load(deck.root), Ctx(), "2026-08-27")
    assert res.blocked == ["pw-0", "pw-1"]
    assert "imgfetch not found" in capsys.readouterr().out


def test_search_requests_identify_the_tool_with_a_user_agent():
    # Wikimedia returns 403 to anonymous default agents (robot policy)
    seen = {}
    def http_get(url, timeout=30, headers=None):
        seen["headers"] = headers or {}
        return R(payload={"query": {"pages": {}}})
    wikimedia_search("คำ", http_get)
    assert seen["headers"].get("User-Agent", "").startswith("thai-deck-gen/")
    assert "github.com/cehbz" in seen["headers"]["User-Agent"]
    def http_get2(url, timeout=30, headers=None):
        seen["headers2"] = headers or {}
        return R(payload={"results": []})
    openverse_search("คำ", http_get2)
    assert seen["headers2"].get("User-Agent", "").startswith("thai-deck-gen/")


def test_fill_images_prefers_a_later_query_on_the_better_source(tmp_path):
    # Wikimedia's Thai-term hits are often transliteration junk (จิบ "to sip"
    # -> Gibberellin diagrams); Openverse must get every query before
    # Wikimedia is consulted at all.
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    order = []

    def http_get(url, timeout=30, headers=None):
        openverse = "api.openverse.org" in url
        is_gloss = "word" in url
        order.append(("openverse" if openverse else "wikimedia", is_gloss))
        if openverse and not is_gloss:           # only the Thai query hits here
            return R(payload={"results": [{"url": "http://img/good.jpg", "license": "cc0"}]})
        if openverse:
            return R(payload={"results": []})
        return R(payload={"query": {"pages": {
            "1": {"imageinfo": [{"url": "http://img/junk.jpg"}]}}}})

    class Ctx:
        imagegen = None
        image_query_hints = {}
        imgfetch = FakeFetch({"http://img/good.jpg": _jpeg_bytes(),
                              "http://img/junk.jpg": _jpeg_bytes()})
    Ctx.http_get = staticmethod(http_get)
    manifest = Manifest.load(deck.root)
    res = fill_images([need], _gaps([]), deck, manifest, Ctx(), "2026-08-29")
    assert res.changed == 1
    assert Ctx.imgfetch.urls == ["http://img/good.jpg"]
    assert manifest.channel_of("media/images/pw-0.jpg") == "openverse"
    # every Openverse query is spent before Wikimedia is consulted at all
    assert order == [("openverse", True), ("openverse", False)]


# --- candidate verification: judge several, keep the one that passes ---

from thai_deck_eval.judge.core import Verdict


class FakeBatchJudge:
    """Judge port stand-in: verdicts keyed by candidate request id."""

    def __init__(self, passing: set[str]):
        self.passing, self.seen = passing, []

    def judge_many(self, reqs):
        self.seen = [r.note_id for r in reqs]
        out = {}
        for r in reqs:
            ok = r.note_id in self.passing
            out[r.note_id] = [
                Verdict(rule=rule, passed=ok, confidence=0.9,
                        rationale="" if ok else "unrelated image")
                for rule in r.rules]
        return out


def _five_results(prefix="http://img"):
    return {"results": [{"url": f"{prefix}/{i}.jpg", "license": "cc0"} for i in range(5)]}


def _verify_ctx(passing, urls=5):
    jpeg = _jpeg_bytes()

    class Ctx:
        imagegen = None
        image_query_hints = {}
        image_candidates = 5
        imgfetch = FakeFetch({f"http://img/{i}.jpg": jpeg for i in range(urls)})
    Ctx.http_get = staticmethod(
        lambda url, timeout=30, headers=None:
        R(payload=_five_results()) if "openverse" in url else R(payload={"results": []}))
    return Ctx()


def test_every_candidate_is_judged_and_the_passing_one_is_kept(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    judge = FakeBatchJudge({"pw-0#2"})          # only the third candidate is relevant
    manifest = Manifest.load(deck.root)

    res = fill_images([need], _gaps([]), deck, manifest, _verify_ctx(judge),
                      "2026-08-30", judge=judge)

    assert res.changed == 1
    assert judge.seen == [f"pw-0#{i}" for i in range(5)]
    assert manifest.entries["media/images/pw-0.jpg"].origin == "http://img/2.jpg"
    assert (deck.root / "media" / "images" / "pw-0.jpg").exists()


def test_all_candidates_rejected_queues_review(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    judge = FakeBatchJudge(set())
    res = fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                      _verify_ctx(judge), "2026-08-30", judge=judge)

    assert res.blocked == ["pw-0"]
    assert not (deck.root / "media" / "images" / "pw-0.jpg").exists()
    review = yaml.safe_load((deck.root / "work" / "image_review.yaml").read_text())
    assert review["items"][0]["note_id"] == "pw-0"


def test_gloss_is_reduced_to_a_searchable_head_term():
    """Word-list glosses are learner definitions ('I (female speaker, or
    casual general)'); the parenthetical returns nothing from an image search."""
    from thai_deck_gen.media.images import _queries
    need = ImageNeed(family="picture_word", note_id="pw-0", term="ฉัน",
                     gloss="I (female speaker, or casual general)",
                     category="Pronouns", path="images/pw-0.jpg")
    assert _queries(need, {}) == ["I", "ฉัน"]

    need = ImageNeed(family="picture_word", note_id="pw-1", term="ส้ม",
                     gloss="orange, mandarin", category="Food",
                     path="images/pw-1.jpg")
    assert _queries(need, {"Food": "food"}) == ["orange food", "orange", "ส้ม"]


# --- attempt memo: a note given up on is not re-searched from scratch ---

def test_a_note_is_not_retried_with_the_same_queries(tmp_path):
    """Re-running must not re-spend on a note whose search found nothing
    acceptable, unless the queries themselves changed."""
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food", path="images/pw-0.jpg")
    judge = FakeBatchJudge(set())
    ctx = _verify_ctx(judge)

    first = fill_images([need], _gaps([]), deck, Manifest.load(deck.root), ctx,
                        "2026-08-30", judge=judge)
    assert first.blocked == ["pw-0"]
    assert judge.seen                                  # candidates were judged

    second_judge = FakeBatchJudge(set())
    ctx2 = _verify_ctx(second_judge)
    second = fill_images([need], _gaps([]), deck, Manifest.load(deck.root), ctx2,
                         "2026-08-31", judge=second_judge)
    assert second.blocked == ["pw-0"]
    assert second_judge.seen == []                     # nothing re-judged
    assert ctx2.imgfetch.urls == []                    # nothing re-downloaded


def test_a_changed_query_retries_the_note(tmp_path):
    """A new search phrase is new information: try again."""
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food", path="images/pw-0.jpg")
    judge = FakeBatchJudge(set())
    fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                _verify_ctx(judge), "2026-08-30", judge=judge)

    retried = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                        gloss="word", category="Food",
                        image_query="a written word on paper",
                        path="images/pw-0.jpg")
    judge2 = FakeBatchJudge({"pw-0#0"})
    res = fill_images([retried], _gaps([]), deck, Manifest.load(deck.root),
                      _verify_ctx(judge2), "2026-08-31", judge=judge2)
    assert res.changed == 1
    assert judge2.seen                                 # the new phrase was tried


def test_candidate_downloads_resume_after_a_stopped_run(tmp_path):
    """Collection is the slow half of an image run; a run stopped partway
    must reuse what it already fetched, provenance included."""
    from thai_deck_gen.media.images import _collect_candidates
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food", path="images/pw-0.jpg")

    work = deck.root / "work" / "candidates" / "pw-0"
    work.mkdir(parents=True)
    for i in range(2):
        (work / f"{i}.jpg").write_bytes(_jpeg_bytes())
    (work / "candidates.yaml").write_text(yaml.safe_dump([
        {"file": "0.jpg", "url": "http://img/0.jpg", "source": "openverse",
         "license": "cc0"},
        {"file": "1.jpg", "url": "http://img/1.jpg", "source": "wikimedia",
         "license": None}], allow_unicode=True), encoding="utf-8")

    def no_search(*a, **k):
        raise AssertionError("must not search again")

    class NoFetch:
        def fetch(self, url):
            raise AssertionError("must not download again")

    cands = _collect_candidates(need, deck, no_search, NoFetch(), {}, 5)
    assert [c.url for c in cands] == ["http://img/0.jpg", "http://img/1.jpg"]
    assert cands[0].source == "openverse" and cands[1].source == "wikimedia"


def test_partial_candidate_dir_is_refetched(tmp_path):
    """A manifest row whose file is missing means the run died mid-write."""
    from thai_deck_gen.media.images import _cached_candidates
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", path="images/pw-0.jpg")
    work = tmp_path / "candidates" / "pw-0"
    work.mkdir(parents=True)
    (work / "candidates.yaml").write_text(yaml.safe_dump([
        {"file": "0.jpg", "url": "u", "source": "openverse"}]), encoding="utf-8")
    assert _cached_candidates(work, need, 5) == []


def test_a_note_with_no_candidates_is_not_marked_exhausted(tmp_path):
    """Zero candidates means the search or the network failed, not that the
    queries were tried and found wanting. A sleeping laptop must not
    blacklist a note forever."""
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food", path="images/pw-0.jpg")
    judge = FakeBatchJudge(set())

    class DeadNetwork:
        imagegen = None
        image_query_hints = {}
        image_candidates = 5
        imgfetch = FakeFetch({})
        http_get = staticmethod(
            lambda url, timeout=30, headers=None: R(status_code=503, payload={}))

    res = fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                      DeadNetwork(), "2026-08-30", judge=judge)
    assert res.blocked == ["pw-0"]
    review = deck.root / "work" / "image_review.yaml"
    if review.exists():
        items = yaml.safe_load(review.read_text())["items"]
        assert all("queries" not in it for it in items)   # nothing memoized

    # a later run with a working network must try again
    good = _verify_ctx(FakeBatchJudge({"pw-0#0"}))
    judge2 = FakeBatchJudge({"pw-0#0"})
    res2 = fill_images([need], _gaps([]), deck, Manifest.load(deck.root), good,
                       "2026-08-31", judge=judge2)
    assert res2.changed == 1


def test_search_preflight_detects_a_dead_proxy():
    """A run that walks 500 notes against a dead tunnel wastes an hour and
    teaches nothing; check once, up front."""
    from thai_deck_gen.media.images import search_reachable
    ok = search_reachable(lambda url, timeout=30, headers=None:
                          R(payload={"results": [{"url": "u"}]}))
    assert ok is None

    import requests as _rq
    def dead(url, timeout=30, headers=None):
        raise _rq.RequestException("tunnel down")
    assert "unreachable" in search_reachable(dead).lower()


def test_exhaustion_expires_when_the_rubric_changes(tmp_path, monkeypatch):
    """The memo records that these queries found nothing acceptable -- which
    is a statement about the rubric as much as the queries."""
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food", path="images/pw-0.jpg")
    judge = FakeBatchJudge(set())
    fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                _verify_ctx(judge), "2026-08-31", judge=judge)

    import thai_deck_eval.judge.prompts as prompts
    monkeypatch.setitem(prompts.PICTURE_RULES, "judge/image-embedded-text",
                        "a materially different rubric")
    judge2 = FakeBatchJudge({"pw-0#0"})
    res = fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                      _verify_ctx(judge2), "2026-09-01", judge=judge2)
    assert res.changed == 1                  # reconsidered under the new rubric


class SuggestingJudge:
    """Rejects everything and names a better phrase, as the triple asks."""

    def __init__(self, suggestion):
        self.suggestion, self.prompts = suggestion, []

    def judge_many(self, reqs):
        self.prompts = [r.prompt for r in reqs]
        return {r.note_id: [
            Verdict(rule="judge/image-irrelevant", passed=False, confidence=0.9,
                    rationale="does not evoke the word",
                    suggestion=self.suggestion)] for r in reqs}


def test_the_intended_phrase_reaches_the_judge(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food",
                     image_query="a written word on paper", path="images/pw-0.jpg")
    judge = SuggestingJudge("a dictionary page")
    fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                _verify_ctx(judge), "2026-08-31", judge=judge)
    assert "a written word on paper" in judge.prompts[0]


def test_a_rejected_word_records_the_suggested_phrase(tmp_path):
    """The judge naming a better phrase is the whole retry mechanism: a new
    phrase changes the query set, which expires the exhaustion memo."""
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food",
                     image_query="a written word on paper", path="images/pw-0.jpg")
    judge = SuggestingJudge("a dictionary page")
    fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                _verify_ctx(judge), "2026-08-31", judge=judge)

    proposals = yaml.safe_load(
        (deck.root / "work" / "image_query_proposals.yaml").read_text())
    assert proposals["คำ"]["suggestion"] == "a dictionary page"
    assert proposals["คำ"]["previous"] == "a written word on paper"


def test_a_phrase_that_finds_nothing_is_recorded_before_any_download(tmp_path):
    """Whether a phrase is searchable is knowable from the result count, for
    free, before spending a download or a judgment on it."""
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food",
                     image_query="a phrase nothing matches", path="images/pw-0.jpg")
    jpeg = _jpeg_bytes()

    def http_get(url, timeout=30, headers=None):
        if quote("a phrase nothing matches") in url:
            return R(payload={"results": []})          # phrase finds nothing
        return R(payload={"results": [{"url": "http://img/0.jpg", "license": "cc0"}]})

    class Ctx:
        imagegen = None
        image_query_hints = {}
        image_candidates = 5
        imgfetch = FakeFetch({"http://img/0.jpg": jpeg})
    Ctx.http_get = staticmethod(http_get)

    judge = FakeBatchJudge({"pw-0#0"})
    res = fill_images([need], _gaps([]), deck, Manifest.load(deck.root), Ctx(),
                      "2026-08-31", judge=judge)

    assert res.changed == 1                      # the gloss carried it
    unsearchable = yaml.safe_load(
        (deck.root / "work" / "image_query_proposals.yaml").read_text())
    assert unsearchable["คำ"]["reason"] == "phrase returned no results"
    assert unsearchable["คำ"]["previous"] == "a phrase nothing matches"


def test_a_searchable_phrase_is_not_recorded(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                     gloss="word", category="Food",
                     image_query="something findable", path="images/pw-0.jpg")
    judge = FakeBatchJudge({"pw-0#0"})
    fill_images([need], _gaps([]), deck, Manifest.load(deck.root),
                _verify_ctx(judge), "2026-08-31", judge=judge)
    assert not (deck.root / "work" / "image_query_proposals.yaml").exists()


def test_pexels_search_parses_results():
    def http_get(url, timeout=30, headers=None):
        assert headers and headers.get("Authorization") == "KEY"
        return R(payload={"photos": [
            {"alt": "man pointing at himself",
             "src": {"large": "http://p/1.jpg"},
             "photographer": "Someone"}]})
    from thai_deck_gen.media.images import pexels_search
    cands = pexels_search("person pointing at themselves", http_get, api_key="KEY")
    assert cands[0].url == "http://p/1.jpg"
    assert cands[0].source == "pexels"


def test_pexels_without_a_key_yields_nothing():
    from thai_deck_gen.media.images import pexels_search
    def http_get(url, timeout=30, headers=None):
        raise AssertionError("must not call without a key")
    assert pexels_search("x", http_get, api_key=None) == []


def test_source_order_puts_concept_photography_first_for_abstract_words():
    """A studio shot of someone pointing at themselves is what an abstract
    word needs; a real Chiang Mai street is what ส้มตำ needs."""
    from thai_deck_gen.media.images import source_order

    class Abstract:
        category, part_of_speech = "Pronouns", "other"

    class Concrete:
        category, part_of_speech = "Food", "noun"

    assert source_order(Abstract())[0].__name__.startswith("pexels")
    assert source_order(Concrete())[0].__name__.startswith("openverse")
    # every source is still tried, only the order differs
    assert {f.__name__ for f in source_order(Abstract())} == \
           {f.__name__ for f in source_order(Concrete())}


def test_limit_caps_the_words_attempted(tmp_path):
    """A short run is how you sanity-check a new source before spending on
    the whole deck."""
    deck = _deck_with_pw(tmp_path)
    needs = [ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                       gloss="word", category="Food", path="images/pw-0.jpg"),
             ImageNeed(family="picture_word", note_id="pw-1", term="ข", gloss="b",
                       category="Food", path="images/pw-1.jpg")]
    judge = FakeBatchJudge({"pw-0#0"})
    res = fill_images(needs[:1], _gaps([]), deck, Manifest.load(deck.root),
                      _verify_ctx(judge), "2026-08-31", judge=judge, limit=1)
    assert res.changed == 1
    assert {r.split("#")[0] for r in judge.seen} == {"pw-0"}


def test_limit_counts_words_the_verified_path_can_serve(tmp_path):
    """Sentence and spelling images are not handled here; counting them
    against the limit makes a smoke run silently do nothing."""
    deck = _deck_with_pw(tmp_path)
    other = ImageNeed(family="spelling_sound", note_id="sp-1", term="-ะ",
                      gloss=None, path="images/sp-1.jpg")
    pw = ImageNeed(family="picture_word", note_id="pw-0", term="คำ", gloss="word",
                   category="Food", path="images/pw-0.jpg")
    judge = FakeBatchJudge({"pw-0#0"})
    res = fill_images([other, pw], _gaps([]), deck, Manifest.load(deck.root),
                      _verify_ctx(judge), "2026-08-31", judge=judge, limit=1)
    assert res.changed == 1                 # the picture word was reached


def test_a_configured_source_without_a_key_warns_once(tmp_path, capsys):
    """Pexels silently returning nothing for want of a key cost a whole
    smoke run; a missing key must be visible."""
    deck = _deck_with_pw(tmp_path, term="ฉัน", gloss="I")
    deck.picture_words[0].category = "Pronouns"
    need = ImageNeed(family="picture_word", note_id="pw-0", term="ฉัน", gloss="I",
                     category="Pronouns", path="images/pw-0.jpg")
    judge = FakeBatchJudge(set())
    ctx = _verify_ctx(judge)
    ctx.pexels_key = None
    fill_images([need], _gaps([]), deck, Manifest.load(deck.root), ctx,
                "2026-08-31", judge=judge)
    assert "pexels" in capsys.readouterr().out.lower()


def test_exhaustion_fingerprint_tracks_usable_sources_not_declared_ones(tmp_path):
    """A source declared but unusable for want of a key must not count as
    searched: otherwise adding the key changes nothing and the words stay
    written off."""
    from thai_deck_gen.media.images import rubric_fingerprint
    assert rubric_fingerprint(pexels=False) != rubric_fingerprint(pexels=True)
