"""Proof gallery: a sequential (no-SRS) review tool for a compiled Anki
.apkg, served over http.server for local review.

    uv run python scripts/proof_gallery.py --apkg <deck>/<deck>.apkg \\
        --deck <deck-dir> [--port 8765]

Only reads the .apkg (extracted into a cache dir under <deck-dir>/work/)
and the deck source directory's notes/sentences.yaml + this repo's
data/word_list_th.yaml (for glosses/voices); the only thing it writes is
<deck-dir>/work/proof_notes.jsonl (append-only review notes/drill results)
and the proof_cache/ extraction itself.
"""
from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import re
import shutil
import sqlite3
import urllib.parse
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Template rendering (pure)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"\{\{#(\w+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)
_INVERTED_SECTION_RE = re.compile(r"\{\{\^(\w+)\}\}(.*?)\{\{/\1\}\}", re.DOTALL)
_FIELD_RE = re.compile(r"\{\{(\w+)\}\}")
_SOUND_RE = re.compile(r"\[sound:([^\]]+)\]")
_IMG_SRC_RE = re.compile(r'(<img\s+[^>]*?src=")([^"]+)(")')


def render_template(fmt: str, fields: dict[str, str], front_side: str = "") -> str:
    """Render one Anki-style qfmt/afmt string against note field values.

    Supports {{Field}}, {{#Field}}...{{/Field}} (kept iff the field is
    non-empty), {{^Field}}...{{/Field}} (kept iff empty), and
    {{FrontSide}}. Field values themselves may carry `[sound:NAME]` and
    `<img src="NAME">` markup (as genanki's compiler emits) -- these are
    converted to a clickable audio button and a /media/-rooted src.
    """
    fields = fields or {}

    def section(match: re.Match) -> str:
        name, inner = match.group(1), match.group(2)
        return inner if fields.get(name, "") else ""

    def inverted_section(match: re.Match) -> str:
        name, inner = match.group(1), match.group(2)
        return inner if not fields.get(name, "") else ""

    html = _SECTION_RE.sub(section, fmt)
    html = _INVERTED_SECTION_RE.sub(inverted_section, html)

    def field(match: re.Match) -> str:
        name = match.group(1)
        if name == "FrontSide":
            return front_side
        return fields.get(name, "")

    html = _FIELD_RE.sub(field, html)

    html = _SOUND_RE.sub(
        lambda m: f'<button class="snd" data-src="/media/{m.group(1)}">▶</button>',
        html,
    )

    def img(match: re.Match) -> str:
        prefix, src, suffix = match.group(1), match.group(2), match.group(3)
        if src.startswith("/media/") or src.startswith("http"):
            return match.group(0)
        return f"{prefix}/media/{src}{suffix}"

    html = _IMG_SRC_RE.sub(img, html)
    return html


_MISSING_BTN_RE = re.compile(r'<button class="snd" data-src="/media/([^"]+)">▶</button>')
_MISSING_IMG_RE = re.compile(r'<img src="/media/([^"]+)">')


def flag_missing_media(html: str, available: set[str]) -> str:
    """Replace a /media/NAME reference with a visible chip when NAME
    isn't actually present, instead of leaving a dead link/button.
    """

    def btn(match: re.Match) -> str:
        name = match.group(1)
        if name in available:
            return match.group(0)
        return f'<span class="missing">missing: {name}</span>'

    def img(match: re.Match) -> str:
        name = match.group(1)
        if name in available:
            return match.group(0)
        return f'<span class="missing">missing: {name}</span>'

    html = _MISSING_BTN_RE.sub(btn, html)
    html = _MISSING_IMG_RE.sub(img, html)
    return html


def extract_sound_name(field_value: str) -> str | None:
    match = _SOUND_RE.search(field_value or "")
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# apkg extraction / media map / card ordering
# ---------------------------------------------------------------------------

def extract_apkg(apkg_path: Path, cache_dir: Path) -> Path:
    """Extract `apkg_path` (a zipfile) into `cache_dir`, re-extracting
    only when `apkg_path`'s mtime has changed since the last extraction.
    """
    apkg_path = Path(apkg_path)
    cache_dir = Path(cache_dir)
    src_mtime = apkg_path.stat().st_mtime
    stamp = cache_dir / ".source_mtime"

    if stamp.exists() and (cache_dir / "collection.anki2").exists():
        try:
            cached_mtime = float(stamp.read_text().strip())
        except ValueError:
            cached_mtime = None
        if cached_mtime == src_mtime:
            return cache_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(apkg_path) as zf:
        zf.extractall(cache_dir)
    stamp.write_text(repr(src_mtime))
    return cache_dir


def load_media_map(cache_dir: Path) -> dict[str, str]:
    """The apkg's `media` member: zip-member index (as a string) -> the
    real filename Anki uses for it.
    """
    media_file = Path(cache_dir) / "media"
    if not media_file.exists():
        return {}
    return json.loads(media_file.read_text(encoding="utf-8"))


def load_cards(cache_dir: Path) -> list[dict[str, Any]]:
    """The full card list, ordered by (due, card id) -- the deck's
    introduction order (see compiler/build.py:stamp_due). Each entry
    carries the rendered front/back HTML (media resolved, missing media
    flagged) plus enough raw data (fields, tags) for the gallery UI.
    """
    cache_dir = Path(cache_dir)
    media_map = load_media_map(cache_dir)
    available = set(media_map.values())

    conn = sqlite3.connect(str(cache_dir / "collection.anki2"))
    try:
        (models_json,) = conn.execute("select models from col").fetchone()
        models = json.loads(models_json)

        notes: dict[int, dict[str, Any]] = {}
        for nid, mid, flds, tags, guid in conn.execute(
            "select id, mid, flds, tags, guid from notes"
        ):
            notes[nid] = {
                "mid": mid,
                "flds": flds.split("\x1f"),
                "tags": [t for t in tags.split(" ") if t],
                "guid": guid,
            }

        card_rows = conn.execute(
            "select id, nid, ord, due from cards order by due, id"
        ).fetchall()
    finally:
        conn.close()

    cards: list[dict[str, Any]] = []
    for index, (card_id, nid, ord_, due) in enumerate(card_rows):
        note = notes[nid]
        model = models[str(note["mid"])]
        field_names = [f["name"] for f in model["flds"]]
        fields = dict(zip(field_names, note["flds"]))
        tmpl = model["tmpls"][ord_]

        front = flag_missing_media(render_template(tmpl["qfmt"], fields), available)
        back = flag_missing_media(
            render_template(tmpl["afmt"], fields, front_side=front), available
        )

        cards.append({
            "index": index,
            "card_id": card_id,
            "note_id": nid,
            "guid": note["guid"],
            "model": model["name"],
            "template": tmpl["name"],
            "tags": note["tags"],
            "fields": fields,
            "due": due,
            "front": front,
            "back": back,
        })
    return cards


# ---------------------------------------------------------------------------
# gloss / voice / drill enrichment
# ---------------------------------------------------------------------------

def load_gloss_map(word_list_path: Path) -> dict[str, str]:
    """thai -> gloss, first row wins (data/word_list_th.yaml)."""
    word_list_path = Path(word_list_path)
    if not word_list_path.exists():
        return {}
    rows = yaml.safe_load(word_list_path.read_text(encoding="utf-8")) or []
    glosses: dict[str, str] = {}
    for row in rows:
        thai = row.get("thai")
        gloss = row.get("gloss")
        if thai and gloss and thai not in glosses:
            glosses[thai] = gloss
    return glosses


def load_speaker_map(sentences_path: Path) -> dict[str, str]:
    """sentence-note-id -> audio.speaker (<deck>/notes/sentences.yaml)."""
    sentences_path = Path(sentences_path)
    if not sentences_path.exists():
        return {}
    rows = yaml.safe_load(sentences_path.read_text(encoding="utf-8")) or []
    speakers: dict[str, str] = {}
    for row in rows:
        nid = row.get("id")
        speaker = (row.get("audio") or {}).get("speaker")
        if nid and speaker:
            speakers[nid] = speaker
    return speakers


def enrich_cards(cards: list[dict[str, Any]], gloss_map: dict[str, str],
                 speaker_map: dict[str, str]) -> None:
    """Add gloss/voice/drill data in place, computed once at load time so
    the client stays a dumb renderer.
    """
    for card in cards:
        fields = card["fields"]
        model = card["model"]

        gloss = None
        if model == "picture_word":
            gloss = gloss_map.get(fields.get("Thai", ""))
        elif model == "sentence":
            gloss = gloss_map.get(fields.get("Target", ""))
        elif model == "spelling_sound":
            word = fields.get("ExampleWord", "")
            word_gloss = gloss_map.get(word)
            if word_gloss:
                gloss = f"exemplar: {word_gloss}"
        elif model == "minimal_pair":
            parts = [
                f"{fields.get(f, '')}: {gloss_map[fields[f]]}"
                for f in ("Thai", "OtherThai")
                if fields.get(f) and fields.get(f) in gloss_map
            ]
            if parts:
                gloss = " / ".join(parts)
        card["gloss"] = gloss

        voice = None
        if model == "sentence":
            sound_name = extract_sound_name(fields.get("Audio", ""))
            if sound_name:
                note_id_guess = Path(sound_name).stem
                voice = speaker_map.get(note_id_guess)
        card["voice"] = voice

        drill = None
        if model == "minimal_pair" and card["template"] == "Recognition":
            contrast = next(
                (t.split("::", 1)[1] for t in card["tags"] if t.startswith("contrast::")),
                None,
            )
            sound_name = extract_sound_name(fields.get("Audio", ""))
            drill = {
                "audio": f"/media/{sound_name}" if sound_name else None,
                "thai": fields.get("Thai", ""),
                "other_thai": fields.get("OtherThai", ""),
                "contrast": contrast,
            }
        card["drill"] = drill


# ---------------------------------------------------------------------------
# note appending
# ---------------------------------------------------------------------------

def append_note(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one timestamped JSON line to `path`; return the record
    written (payload + "ts").
    """
    path = Path(path)
    record = dict(payload)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def compute_stats(notes_path: Path) -> dict[str, Any]:
    """Drill accuracy per contrast + a count of free-text notes taken."""
    notes_path = Path(notes_path)
    drills: dict[str, dict[str, int]] = {}
    note_count = 0
    if notes_path.exists():
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "drill":
                contrast = record.get("contrast") or "unknown"
                bucket = drills.setdefault(contrast, {"correct": 0, "total": 0})
                bucket["total"] += 1
                if record.get("correct"):
                    bucket["correct"] += 1
            elif kind == "note":
                note_count += 1
    return {"drills": drills, "note_count": note_count}


# ---------------------------------------------------------------------------
# gallery context + server
# ---------------------------------------------------------------------------

@dataclass
class GalleryContext:
    cards: list[dict[str, Any]]
    media_paths: dict[str, Path]
    notes_path: Path


def build_gallery(apkg_path: Path, deck_dir: Path) -> GalleryContext:
    apkg_path = Path(apkg_path)
    deck_dir = Path(deck_dir)

    cache_dir = deck_dir / "work" / "proof_cache"
    extract_apkg(apkg_path, cache_dir)

    media_map = load_media_map(cache_dir)
    media_paths = {name: cache_dir / idx for idx, name in media_map.items()}

    cards = load_cards(cache_dir)
    gloss_map = load_gloss_map(REPO_ROOT / "data" / "word_list_th.yaml")
    speaker_map = load_speaker_map(deck_dir / "notes" / "sentences.yaml")
    enrich_cards(cards, gloss_map, speaker_map)

    notes_path = deck_dir / "work" / "proof_notes.jsonl"
    return GalleryContext(cards=cards, media_paths=media_paths, notes_path=notes_path)


_PUBLIC_CARD_KEYS = (
    "index", "card_id", "note_id", "guid", "model", "template", "tags",
    "front", "back", "gloss", "voice", "drill",
)


def public_card(card: dict[str, Any]) -> dict[str, Any]:
    return {k: card[k] for k in _PUBLIC_CARD_KEYS}


def make_handler(ctx: GalleryContext) -> type[http.server.BaseHTTPRequestHandler]:
    cards_json = json.dumps([public_card(c) for c in ctx.cards], ensure_ascii=False).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "ProofGallery/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj: Any, status: int = 200) -> None:
            self._send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8", status)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/cards":
                self._send_bytes(cards_json, "application/json; charset=utf-8")
            elif parsed.path == "/stats":
                self._send_json(compute_stats(ctx.notes_path))
            elif parsed.path.startswith("/media/"):
                self._serve_media(urllib.parse.unquote(parsed.path[len("/media/"):]))
            else:
                self.send_error(404, "not found")

        def _serve_media(self, name: str) -> None:
            path = ctx.media_paths.get(name)
            if path is None or not path.exists():
                self.send_error(404, f"missing media: {name}")
                return
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            try:
                self._send_bytes(path.read_bytes(), ctype)
            except OSError:
                self.send_error(404, f"missing media: {name}")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/note":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            append_note(ctx.notes_path, payload)
            self._send_json({"ok": True})

    return Handler


def serve(ctx: GalleryContext, port: int) -> None:
    handler = make_handler(ctx)
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    print(f"proof gallery: http://127.0.0.1:{port}/  ({len(ctx.cards)} cards)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# gallery HTML page (self-contained: inline CSS + JS, no external resources)
# ---------------------------------------------------------------------------

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Proof Gallery</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; display: flex; flex-direction: column;
    background: #111417; color: #e8e8e8;
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  #bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #1b1f24; border-bottom: 1px solid #2a2f36;
    font-size: 13px; gap: 12px; flex-wrap: wrap;
  }
  #bar .left { display: flex; align-items: center; gap: 10px; }
  #progress { font-weight: 600; }
  #meta { color: #9aa4b1; }
  #saved-marker { color: #e6c200; display: none; }
  #gotoInput {
    width: 64px; background: #0e1114; color: #e8e8e8; border: 1px solid #333;
    border-radius: 4px; padding: 2px 6px;
  }
  #help { color: #6b7480; }
  #card {
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 24px; overflow: auto; text-align: center;
    gap: 10px;
  }
  .thai, .cloze, .pattern, .choices { font-size: 52px; line-height: 1.3; }
  .ipa { font-size: 22px; color: #9aa4b1; }
  .target { font-size: 40px; }
  .classifier, .grammar { font-size: 18px; color: #9aa4b1; }
  img { max-width: min(90vw, 640px); max-height: 55vh; width: auto; height: auto; border-radius: 6px; }
  hr#answer { width: 60%; border: none; border-top: 1px solid #333; margin: 14px 0; }
  .answer { color: #7fdc7f; font-weight: 600; font-size: 40px; }
  .other { color: #8b93a0; font-size: 30px; }
  button.snd {
    font-size: 28px; background: #262b31; color: #fff; border: 1px solid #3a4048;
    border-radius: 50%; width: 56px; height: 56px; cursor: pointer; margin: 6px;
  }
  button.snd:hover { background: #323942; }
  .missing {
    display: inline-block; background: #3a1f1f; color: #ff9d9d;
    border: 1px dashed #a55; padding: 4px 10px; border-radius: 6px; font-size: 16px;
  }
  .gloss-chip {
    display: inline-block; border: 2px dashed #4fb3bf; color: #7fe0ea;
    padding: 6px 14px; border-radius: 10px; font-size: 22px; background: #10262a;
  }
  .voice { color: #8ea0e0; font-size: 14px; }
  .choices { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
  .choice {
    font-size: 44px; padding: 18px 34px; background: #262b31; color: #fff;
    border: 1px solid #3a4048; border-radius: 10px; cursor: pointer;
  }
  .choice.correct { background: #1f5c2e; border-color: #2f8a44; }
  .choice.wrong { background: #5c1f1f; border-color: #8a2f2f; }
  #noteInput {
    position: fixed; left: 50%; bottom: 60px; transform: translateX(-50%);
    background: #1b1f24; border: 1px solid #3a4048; border-radius: 8px;
    padding: 10px 14px; display: flex; gap: 8px; align-items: center;
  }
  #noteInput input {
    width: 420px; background: #0e1114; color: #e8e8e8; border: 1px solid #333;
    border-radius: 4px; padding: 6px 10px; font-size: 15px;
  }
  #statsOverlay {
    position: fixed; inset: 0; background: rgba(10,12,14,0.92);
    display: flex; align-items: center; justify-content: center;
  }
  #statsOverlay .panel {
    background: #1b1f24; border: 1px solid #3a4048; border-radius: 10px;
    padding: 24px 32px; min-width: 320px;
  }
  #statsOverlay table { border-collapse: collapse; width: 100%; margin-top: 10px; }
  #statsOverlay td { padding: 4px 10px; border-bottom: 1px solid #2a2f36; font-size: 14px; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
  <div id="bar">
    <div class="left">
      <span id="progress">- / -</span>
      <span id="saved-marker">★ saved</span>
      <span id="meta"></span>
    </div>
    <div class="left">
      <span id="help">space reveal &middot; j/k next/prev &middot; f note &middot; g gloss &middot; s stats</span>
      <input id="gotoInput" type="text" placeholder="#">
    </div>
  </div>
  <div id="card"></div>
  <div id="noteInput" hidden>
    <input id="noteText" type="text" placeholder="note (Enter to save, Esc to cancel)">
  </div>
  <div id="statsOverlay" hidden><div class="panel" id="statsPanel"></div></div>

<script>
(function () {
  "use strict";

  var STORAGE_KEY = "proofGalleryIndex";
  var cards = [];
  var idx = 0;
  var revealed = false;
  var glossOn = (localStorage.getItem("pg_gloss") ?? "1") === "1";
  var savedIndices = new Set();
  var noteBoxOpen = false;

  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) { for (var k in attrs) { e.setAttribute(k, attrs[k]); } }
    if (html !== undefined) { e.innerHTML = html; }
    return e;
  }

  function wireSoundButtons(container) {
    var buttons = container.querySelectorAll("button.snd");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var audio = new Audio(btn.dataset.src);
        audio.play().catch(function () {});
      });
    });
  }

  function autoplayFirst(container) {
    var btn = container.querySelector("button.snd");
    if (btn) { btn.click(); }
  }

  function saveProgress() {
    try { localStorage.setItem(STORAGE_KEY, String(idx)); } catch (e) {}
  }

  function updateBar(card) {
    document.getElementById("progress").textContent = (idx + 1) + " / " + cards.length;
    document.getElementById("meta").textContent = card.model + " · " + card.template +
      (card.tags.length ? " · " + card.tags.join(" ") : "");
    document.getElementById("saved-marker").style.display =
      savedIndices.has(card.index) ? "inline" : "none";
  }

  function render() {
    revealed = false;
    saveProgress();
    var card = cards[idx];
    var cardEl = document.getElementById("card");
    cardEl.innerHTML = "";
    updateBar(card);

    if (card.drill) {
      renderDrill(card, cardEl);
      return;
    }

    var front = el("div", { "class": "front" }, card.front);
    cardEl.appendChild(front);

    if (glossOn && card.gloss) {
      cardEl.appendChild(el("div", { "class": "gloss-chip" }, card.gloss));
    }
    if (card.voice) {
      cardEl.appendChild(el("div", { "class": "voice" }, "voice: " + card.voice));
    }

    wireSoundButtons(front);
    autoplayFirst(front);
  }

  function reveal() {
    if (revealed) { return; }
    var card = cards[idx];
    if (card.drill) { return; }  // drill reveals on choice
    revealed = true;
    var cardEl = document.getElementById("card");
    var back = el("div", { "class": "back" }, card.back);
    cardEl.appendChild(back);
    wireSoundButtons(back);  // never autoplay on reveal
  }

  function renderDrill(card, cardEl) {
    var choicesDiv = el("div", { "class": "choices" });
    var options = [card.drill.thai, card.drill.other_thai];
    if (Math.random() < 0.5) { options.reverse(); }
    options.forEach(function (text) {
      var btn = el("button", { "class": "choice" }, text);
      btn.dataset.correct = (text === card.drill.thai) ? "1" : "0";
      btn.addEventListener("click", function () { selectDrill(btn, card, choicesDiv); });
      choicesDiv.appendChild(btn);
    });
    cardEl.appendChild(choicesDiv);

    if (card.drill.audio) {
      var audio = new Audio(card.drill.audio);
      audio.play().catch(function () {});
    }
  }

  function selectDrill(btn, card, choicesDiv) {
    if (revealed) { return; }
    revealed = true;
    var correct = btn.dataset.correct === "1";
    Array.prototype.forEach.call(choicesDiv.children, function (b) {
      b.classList.add(b.dataset.correct === "1" ? "correct" : "wrong");
    });
    var cardEl = document.getElementById("card");
    var back = el("div", { "class": "back" }, card.back);
    cardEl.appendChild(back);
    wireSoundButtons(back);
    postEvent(Object.assign({ kind: "drill", contrast: card.drill.contrast, correct: correct }, card));
  }

  function next() { if (idx < cards.length - 1) { idx++; render(); } }
  function prev() { if (idx > 0) { idx--; render(); } }
  function goto(n) { if (n >= 0 && n < cards.length) { idx = n; render(); } }

  function postEvent(extra) {
    var card = cards[idx];
    var payload = {
      index: card.index, card: card.index + 1, note_id: card.note_id, guid: card.guid,
      model: card.model, tags: card.tags,
    };
    for (var k in extra) {
      if (k !== "index" && k !== "note_id" && k !== "guid" && k !== "model" &&
          k !== "tags" && k !== "front" && k !== "back" && k !== "gloss" &&
          k !== "voice" && k !== "drill" && k !== "template" && k !== "card_id") {
        payload[k] = extra[k];
      }
    }
    return fetch("/api/note", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).catch(function () {});
  }

  function openNoteBox() {
    noteBoxOpen = true;
    var box = document.getElementById("noteInput");
    var input = document.getElementById("noteText");
    box.hidden = false;
    input.value = "";
    input.focus();
  }

  function closeNoteBox() {
    noteBoxOpen = false;
    document.getElementById("noteInput").hidden = true;
    document.getElementById("card").focus && document.body.focus();
  }

  function toggleStats() {
    var overlay = document.getElementById("statsOverlay");
    if (!overlay.hidden) { overlay.hidden = true; return; }
    fetch("/stats").then(function (r) { return r.json(); }).then(function (stats) {
      var panel = document.getElementById("statsPanel");
      var html = "<h2>Stats</h2><div>Notes taken: " + stats.note_count + "</div>";
      html += "<table>";
      var contrasts = Object.keys(stats.drills);
      if (!contrasts.length) {
        html += "<tr><td>no drill results yet</td></tr>";
      } else {
        contrasts.forEach(function (c) {
          var b = stats.drills[c];
          var pct = b.total ? Math.round(100 * b.correct / b.total) : 0;
          html += "<tr><td>" + c + "</td><td>" + b.correct + "/" + b.total + " (" + pct + "%)</td></tr>";
        });
      }
      html += "</table><div style='margin-top:10px;color:#9aa4b1;'>press s to close</div>";
      panel.innerHTML = html;
      overlay.hidden = false;
    }).catch(function () {});
  }

  document.getElementById("noteText").addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      var text = document.getElementById("noteText").value;
      closeNoteBox();
      savedIndices.add(cards[idx].index);
      postEvent({ kind: "note", text: text }).then(function () { updateBar(cards[idx]); });
      updateBar(cards[idx]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      closeNoteBox();
    }
  });

  document.getElementById("gotoInput").addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      var n = parseInt(this.value, 10);
      if (!isNaN(n)) { goto(n - 1); }
      this.value = "";
      this.blur();
    }
  });

  document.addEventListener("keydown", function (e) {
    var active = document.activeElement;
    if (active && active.tagName === "INPUT") { return; }
    if (e.key === " ") { e.preventDefault(); reveal(); return; }
    if (e.key === "j" || e.key === "ArrowRight") { next(); return; }
    if (e.key === "k" || e.key === "ArrowLeft") { prev(); return; }
    if (e.key === "f") { openNoteBox(); return; }
    if (e.key === "g") { glossOn = !glossOn; localStorage.setItem("pg_gloss", glossOn ? "1" : "0"); render(); return; }
    if (e.key === "s") { toggleStats(); return; }
    if (e.key === "Escape") {
      var overlay = document.getElementById("statsOverlay");
      if (!overlay.hidden) { overlay.hidden = true; }
      return;
    }
    var card = cards[idx];
    if (card && card.drill && !revealed && (e.key === "1" || e.key === "2")) {
      var choicesDiv = document.querySelector("#card .choices");
      if (choicesDiv) { selectDrill(choicesDiv.children[parseInt(e.key, 10) - 1], card, choicesDiv); }
    }
  });

  fetch("/api/cards").then(function (r) { return r.json(); }).then(function (data) {
    cards = data;
    var saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch (e) {}
    idx = saved !== null ? Math.max(0, Math.min(parseInt(saved, 10) || 0, cards.length - 1)) : 0;
    if (cards.length) { render(); }
    else { document.getElementById("card").textContent = "no cards"; }
  }).catch(function (e) {
    document.getElementById("card").textContent = "failed to load cards: " + e;
  });
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apkg", required=True, type=Path, help="path to the compiled .apkg")
    parser.add_argument("--deck", required=True, type=Path, help="deck source directory")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    ctx = build_gallery(args.apkg, args.deck)
    serve(ctx, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
