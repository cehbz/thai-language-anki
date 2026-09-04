"""MediaStore.add_image's ingest normalization (spec 4 section 3): bounded
long edge (800px, aspect preserved), metadata stripped, re-encoded -- the
stored, sha'd bytes are the normalized file, so a judge/learner always
sees the pixels the card shows.

Pillow is a hard dependency of add_image; this module needs a real Pillow
to build fixture images, so it skips cleanly when Pillow isn't installed,
per the task's "no Pillow requirement for the default suite" instruction.
"""
import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from thai_syllabus.store import MediaStore  # noqa: E402


def _png_bytes(size, mode="RGB", color=(200, 50, 50)) -> bytes:
    img = Image.new(mode, size, color)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _jpeg_with_exif_bytes(size=(200, 100)) -> bytes:
    img = Image.new("RGB", size, (10, 200, 10))
    out = io.BytesIO()
    # A minimal EXIF blob (Pillow accepts raw bytes for the `exif` kwarg);
    # its content doesn't matter, only that saving with it embeds APP1/Exif
    # data we then expect normalization to strip.
    exif = Image.Exif()
    exif[271] = "Test Camera Make"  # Make tag
    img.save(out, format="JPEG", exif=exif)
    return out.getvalue()


def test_add_image_bounds_the_long_edge_to_800px_preserving_aspect(tmp_path):
    store = MediaStore(tmp_path)
    data = _png_bytes((1600, 800))  # 2:1 aspect, long edge 1600
    result = store.add_image(data, ext="png")
    stored = Image.open(store.path_for(result.sha, result.ext))
    assert max(stored.size) <= 800
    assert stored.size[0] / stored.size[1] == pytest.approx(1600 / 800, rel=0.01)


def test_add_image_leaves_a_small_image_undownscaled(tmp_path):
    store = MediaStore(tmp_path)
    data = _png_bytes((300, 200))
    result = store.add_image(data, ext="png")
    stored = Image.open(store.path_for(result.sha, result.ext))
    assert stored.size == (300, 200)


def test_add_image_strips_exif_metadata(tmp_path):
    store = MediaStore(tmp_path)
    data = _jpeg_with_exif_bytes()
    assert Image.open(io.BytesIO(data)).getexif()  # fixture sanity check
    result = store.add_image(data, ext="jpg")
    stored = Image.open(store.path_for(result.sha, result.ext))
    assert not stored.getexif()


def test_add_image_is_content_addressed_on_the_normalized_bytes(tmp_path):
    store = MediaStore(tmp_path)
    data = _png_bytes((300, 200))
    r1 = store.add_image(data, ext="png")
    r2 = store.add_image(data, ext="png")
    assert r1.sha == r2.sha
    import hashlib
    stored_bytes = store.path_for(r1.sha, r1.ext).read_bytes()
    assert r1.sha == hashlib.sha256(stored_bytes).hexdigest()


@pytest.fixture
def media_store(tmp_path):
    return MediaStore(tmp_path)


def test_undecodable_image_refuses(media_store):
    with pytest.raises(ValueError, match="decode"):
        media_store.add_image(b"not an image", "jpg")
