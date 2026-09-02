"""Unit tests for scripts/proof_gallery.py's pure logic: template
rendering, apkg extraction / media resolution / card ordering, and
note-append. No pythainlp/anthropic imports -- this must stay in the
default (non-integration, non-live) suite.
"""
import json
import sqlite3
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import proof_gallery as pg  # noqa: E402


# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------

def test_render_template_plain_field():
    assert pg.render_template("{{Thai}}", {"Thai": "ก่อน"}) == "ก่อน"


def test_render_template_conditional_section_truthy():
    # copied from MODELS["picture_word"] Comprehension afmt (build.py)
    fmt = ('{{#Classifier}}<div class="classifier">{{Classifier}}</div>'
           '{{/Classifier}}')
    assert (pg.render_template(fmt, {"Classifier": "ตัว"})
            == '<div class="classifier">ตัว</div>')


def test_render_template_conditional_section_falsy():
    fmt = ('{{#Classifier}}<div class="classifier">{{Classifier}}</div>'
           '{{/Classifier}}')
    assert pg.render_template(fmt, {"Classifier": ""}) == ""


def test_render_template_inverted_section():
    fmt = "{{^Gloss}}<div class=\"no-gloss\">none</div>{{/Gloss}}"
    assert (pg.render_template(fmt, {"Gloss": ""})
            == '<div class="no-gloss">none</div>')
    assert pg.render_template(fmt, {"Gloss": "eat"}) == ""


def test_render_template_front_side():
    # copied from MODELS["spelling_sound"] PatternToSound afmt (build.py)
    fmt = ('{{FrontSide}}<hr id="answer">{{Audio}}<div>{{ExampleWord}}</div>'
           '{{Image}}')
    out = pg.render_template(
        fmt,
        {"Audio": "", "ExampleWord": "กา", "Image": ""},
        front_side='<div class="pattern">ก</div>',
    )
    assert out == '<div class="pattern">ก</div><hr id="answer"><div>กา</div>'


def test_render_template_sound_to_button():
    assert (pg.render_template("{{Audio}}", {"Audio": "[sound:foo.mp3]"})
            == '<button class="snd" data-src="/media/foo.mp3">▶</button>')


def test_render_template_sound_empty_field_is_empty():
    assert pg.render_template("{{Audio}}", {"Audio": ""}) == ""


def test_render_template_img_src_rewrite():
    assert (pg.render_template("{{Image}}", {"Image": '<img src="foo.jpg">'})
            == '<img src="/media/foo.jpg">')


def test_render_template_combined_real_minimal_pair_afmt():
    # copied from MODELS["minimal_pair"] Recognition afmt (build.py)
    fmt = ('{{FrontSide}}<hr id="answer">'
           '<div class="answer">{{Thai}} <span class="ipa">[{{Ipa}}]</span></div>'
           '<div class="other">{{OtherThai}} <span class="ipa">[{{OtherIpa}}]</span></div>')
    fields = {
        "Thai": "กี", "Ipa": "kiː˧", "OtherThai": "กี่", "OtherIpa": "kiː˨˩",
    }
    out = pg.render_template(fmt, fields, front_side='<button class="snd" data-src="/media/a.mp3">▶</button>')
    assert out == (
        '<button class="snd" data-src="/media/a.mp3">▶</button>'
        '<hr id="answer">'
        '<div class="answer">กี <span class="ipa">[kiː˧]</span></div>'
        '<div class="other">กี่ <span class="ipa">[kiː˨˩]</span></div>'
    )


# ---------------------------------------------------------------------------
# apkg extraction / media map / card ordering
# ---------------------------------------------------------------------------

MODEL_A = {
    "id": 1001,
    "name": "wordish",
    "flds": [{"name": "Thai"}, {"name": "Audio"}, {"name": "Image"}],
    "tmpls": [
        {
            "name": "Recall",
            "qfmt": '<div class="thai">{{Image}}{{Audio}}</div>',
            "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{Thai}}</div>',
        },
        {
            "name": "Reverse",
            "qfmt": '<div class="thai">{{Thai}}</div>',
            "afmt": '{{FrontSide}}<hr id="answer">{{Audio}}',
        },
    ],
}

MODEL_B = {
    "id": 1002,
    "name": "sentish",
    "flds": [{"name": "ThaiCloze"}, {"name": "Target"}, {"name": "Gloss"}],
    "tmpls": [
        {
            "name": "Cloze",
            "qfmt": '<div class="cloze">{{ThaiCloze}}</div>',
            "afmt": ('{{FrontSide}}<hr id="answer"><div class="target">{{Target}}</div>'
                     '{{#Gloss}}<div class="gloss">{{Gloss}}</div>{{/Gloss}}'),
        },
    ],
}


def _build_fixture_apkg(path: Path) -> None:
    """A tiny synthetic .apkg: 2 models, 3 notes, 4 cards with distinct
    due, one real media file (a.mp3) and one dangling reference
    (ghost.jpg, never present) so missing-media handling is exercised.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        db_path = tmp / "collection.anki2"
        conn = sqlite3.connect(str(db_path))
        models = {
            str(MODEL_A["id"]): {
                "name": MODEL_A["name"],
                "flds": MODEL_A["flds"],
                "tmpls": MODEL_A["tmpls"],
            },
            str(MODEL_B["id"]): {
                "name": MODEL_B["name"],
                "flds": MODEL_B["flds"],
                "tmpls": MODEL_B["tmpls"],
            },
        }
        conn.execute("create table col (id integer primary key, models text, decks text)")
        conn.execute("insert into col (id, models, decks) values (1, ?, '{}')",
                    (json.dumps(models),))
        conn.execute("create table notes (id integer primary key, guid text, mid integer, "
                    "flds text, tags text)")
        conn.execute("create table cards (id integer primary key, nid integer, ord integer, "
                    "due integer)")

        def add_note(nid, mid, flds, tags=""):
            conn.execute(
                "insert into notes (id, guid, mid, flds, tags) values (?, ?, ?, ?, ?)",
                (nid, f"guid-{nid}", mid, "\x1f".join(flds), tags),
            )

        add_note(1, MODEL_A["id"], ["ก", "[sound:a.mp3]", ""])
        add_note(2, MODEL_A["id"], ["ข", "", '<img src="ghost.jpg">'])
        add_note(3, MODEL_B["id"], ["___กิน", "กิน", "eat"])

        def add_card(cid, nid, ord_, due):
            conn.execute("insert into cards (id, nid, ord, due) values (?, ?, ?, ?)",
                        (cid, nid, ord_, due))

        add_card(10, 1, 0, 3)  # Recall, nid1
        add_card(11, 1, 1, 1)  # Reverse, nid1
        add_card(12, 2, 0, 0)  # Recall, nid2 (missing image, empty audio)
        add_card(13, 3, 0, 2)  # Cloze, nid3

        conn.commit()
        conn.close()

        media_dir = tmp
        (media_dir / "0").write_bytes(b"FAKE-MP3-BYTES")
        (media_dir / "media").write_text(json.dumps({"0": "a.mp3"}), encoding="utf-8")

        with zipfile.ZipFile(path, "w") as zf:
            zf.write(db_path, "collection.anki2")
            zf.write(media_dir / "0", "0")
            zf.write(media_dir / "media", "media")


def test_extract_apkg_populates_cache_dir(tmp_path):
    apkg = tmp_path / "deck.apkg"
    _build_fixture_apkg(apkg)
    cache_dir = tmp_path / "work" / "proof_cache"

    pg.extract_apkg(apkg, cache_dir)

    assert (cache_dir / "collection.anki2").exists()
    assert (cache_dir / "media").exists()
    assert (cache_dir / "0").read_bytes() == b"FAKE-MP3-BYTES"


def test_extract_apkg_reextracts_when_apkg_newer(tmp_path):
    import os
    import time

    apkg = tmp_path / "deck.apkg"
    _build_fixture_apkg(apkg)
    cache_dir = tmp_path / "work" / "proof_cache"

    pg.extract_apkg(apkg, cache_dir)
    sentinel = cache_dir / "stale_marker.txt"
    sentinel.write_text("stale")

    # not stale: re-running should not touch the cache
    pg.extract_apkg(apkg, cache_dir)
    assert sentinel.exists()

    # touch the apkg into the future -> cache must be rebuilt
    future = time.time() + 10
    os.utime(apkg, (future, future))
    pg.extract_apkg(apkg, cache_dir)
    assert not sentinel.exists()


def test_load_media_map(tmp_path):
    apkg = tmp_path / "deck.apkg"
    _build_fixture_apkg(apkg)
    cache_dir = tmp_path / "work" / "proof_cache"
    pg.extract_apkg(apkg, cache_dir)

    media_map = pg.load_media_map(cache_dir)

    assert media_map == {"0": "a.mp3"}


def test_load_cards_orders_by_due(tmp_path):
    apkg = tmp_path / "deck.apkg"
    _build_fixture_apkg(apkg)
    cache_dir = tmp_path / "work" / "proof_cache"
    pg.extract_apkg(apkg, cache_dir)

    cards = pg.load_cards(cache_dir)

    assert [c["card_id"] for c in cards] == [12, 11, 13, 10]
    assert [c["model"] for c in cards] == ["wordish", "wordish", "sentish", "wordish"]
    assert [c["template"] for c in cards] == ["Recall", "Reverse", "Cloze", "Recall"]
    assert [c["index"] for c in cards] == [0, 1, 2, 3]


def test_load_cards_resolves_available_media(tmp_path):
    apkg = tmp_path / "deck.apkg"
    _build_fixture_apkg(apkg)
    cache_dir = tmp_path / "work" / "proof_cache"
    pg.extract_apkg(apkg, cache_dir)

    cards = pg.load_cards(cache_dir)
    by_card_id = {c["card_id"]: c for c in cards}

    # card 10: nid1, Recall -> Audio present in media map -> real button
    assert 'data-src="/media/a.mp3"' in by_card_id[10]["front"]
    assert "missing" not in by_card_id[10]["front"]

    # card 12: nid2, Recall -> Image "ghost.jpg" never in media map -> chip
    assert "missing: ghost.jpg" in by_card_id[12]["front"]
    # empty Audio field -> no sound button at all
    assert "snd" not in by_card_id[12]["front"]

    # card 11: nid1, Reverse -> back plays Audio via a real button
    assert 'data-src="/media/a.mp3"' in by_card_id[11]["back"]

    # card 13: nid3, Cloze -> back shows Target + Gloss section
    assert '<div class="target">กิน</div>' in by_card_id[13]["back"]
    assert '<div class="gloss">eat</div>' in by_card_id[13]["back"]


# ---------------------------------------------------------------------------
# append_note
# ---------------------------------------------------------------------------

def test_append_note_writes_one_json_line_with_timestamp(tmp_path):
    path = tmp_path / "work" / "proof_notes.jsonl"
    payload = {"index": 3, "note_id": 42, "guid": "abc", "model": "picture_word",
              "kind": "note", "text": "looks off"}

    record = pg.append_note(path, payload)

    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["index"] == 3
    assert stored["text"] == "looks off"
    assert "ts" in stored
    assert record == stored


def test_append_note_appends_without_clobbering(tmp_path):
    path = tmp_path / "proof_notes.jsonl"
    pg.append_note(path, {"index": 0, "kind": "note", "text": "a"})
    pg.append_note(path, {"index": 1, "kind": "drill", "contrast": "tone:mid-low",
                          "correct": True})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "a"
    assert json.loads(lines[1])["contrast"] == "tone:mid-low"
