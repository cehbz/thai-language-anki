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
    def http_get(url, timeout=30):
        assert "api.openverse.org" in url
        return R(payload={"results": [{"url": "http://img/a.jpg", "license": "cc0"}]})
    cands = openverse_search("คำ", http_get)
    assert cands == [ImageCandidate(url="http://img/a.jpg", source="openverse", license="cc0")]


def test_wikimedia_search_parses_results():
    def http_get(url, timeout=30):
        assert "commons.wikimedia.org" in url
        return R(payload={"query": {"pages": {
            "1": {"imageinfo": [{"url": "http://img/b.jpg"}]}}}})
    cands = wikimedia_search("คำ", http_get)
    assert cands == [ImageCandidate(url="http://img/b.jpg", source="wikimedia", license=None)]


def test_fill_images_queries_thai_before_gloss(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")
    thai_q = quote("คำ")
    jpeg = _jpeg_bytes()
    calls = []

    def http_get(url, timeout=30):
        calls.append(url)
        if "api.openverse.org" in url and thai_q in url:
            return R(payload={"results": []})
        if "commons.wikimedia.org" in url and thai_q in url:
            return R(payload={"query": {"pages": {}}})
        if "api.openverse.org" in url and "word" in url:
            return R(payload={"results": [{"url": "http://img/x.jpg", "license": "cc0"}]})
        if url == "http://img/x.jpg":
            return R(content=jpeg)
        raise AssertionError(f"unexpected url {url}")

    class Ctx:
        imagegen = None
    Ctx.http_get = staticmethod(http_get)

    manifest = Manifest.load(deck.root)
    res = fill_images([need], _gaps([]), deck, manifest, Ctx(), "2026-08-27")

    assert res.changed == 1
    assert (deck.root / "media" / "images" / "pw-0.jpg").exists()
    assert manifest.channel_of("media/images/pw-0.jpg") == "openverse"
    # Thai term queried (both sources) before the gloss fallback
    assert thai_q in calls[0]
    assert thai_q in calls[1]
    assert "word" in calls[2]


def test_fill_images_blocks_when_nothing_found(tmp_path):
    deck = _deck_with_pw(tmp_path)
    need = ImageNeed(family="picture_word", note_id="pw-0", term="คำ",
                      gloss="word", path="images/pw-0.jpg")

    def http_get(url, timeout=30):
        if "api.openverse.org" in url:
            return R(payload={"results": []})
        return R(payload={"query": {"pages": {}}})

    class Ctx:
        imagegen = None
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
        http_get = staticmethod(lambda url, timeout=30: (_ for _ in ()).throw(
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

    def http_get(url, timeout=30):
        # only need1's plain search fill should reach here
        if "api.openverse.org" in url:
            return R(payload={"results": [{"url": "http://img/y.jpg", "license": "cc0"}]})
        if url == "http://img/y.jpg":
            return R(content=_jpeg_bytes())
        return R(payload={"query": {"pages": {}}})

    class Ctx:
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
        http_get = staticmethod(lambda url, timeout=30: (_ for _ in ()).throw(
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
        http_get = staticmethod(lambda url, timeout=30: (_ for _ in ()).throw(
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
