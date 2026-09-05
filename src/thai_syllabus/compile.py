"""compile_syllabus (spec 4): translate a Syllabus into an Anki .apkg.

`compile_syllabus(syllabus, db, media_store, out_path, *, force=False)` is
an application service, a free function rather than a Syllabus method,
because it needs two dependencies (a CacheReader for
`derivations.current_best`, and a MediaStore to stage bytes into the
package) that spec 1's frozen Syllabus dataclass has no field for, and
because "the translation of Syllabus state into Anki's domain" (spec 4's
own framing) is a distinct concern from the aggregate's pure,
storage-agnostic domain logic in syllabus.py. `db` is typed concretely as
store.SyllabusDb (not just the narrower CacheReader Protocol) because
compile also needs its non-Protocol `media_provenance` (extension/source/
speaker lookups) -- consistent with how migrate.py already depends on
SyllabusDb directly.

Ambiguities the terse spec text left implicit, resolved here (not
redesigns -- nothing here contradicts spec 4, it fills gaps the prose
didn't spell out):

- **Production's "(productive Target only)" gate**: a word compiles to
  ONE note (guid = word id, spec 4 section 2) carrying every template;
  genanki only drops a per-note CARD automatically based on whether
  fields its qfmt structurally needs are empty (verified against
  genanki's Model._req / Note._front_back_cards). Since "does this word
  have a productive Target" is a fact about the SYLLABUS, not something
  expressible as emptiness of the fields spec 4 §1 lists for `word`, an
  extra boolean field, `ProductiveTarget`, gates it -- exactly the same
  pattern `TestSpelling` already uses to gate `Spelling`. `TestSpelling`
  itself has no curated source yet either (no Target/Word field says
  "test this word's spelling"); until one exists, it mirrors
  `ProductiveTarget`'s value (spelling is tested for words the learner is
  expected to produce). Both are parked-variant placeholders, same
  footing as FrontGloss's "pending study input" (spec 4 §1).
- **Reading's "staged after its graphemes"**: falls out for free from
  Syllabus.order()'s own structure (`[*sounds, *words]`, spec 1 section
  3) -- EVERY grapheme precedes EVERY word target unconditionally, so the
  due-stride block for any word (see below) already sits after every
  grapheme's block. No extra logic needed here beyond using order()'s
  existing sequence.
- **Sentence Listening's "back full text"**: `ThaiCloze` has the target
  permanently blanked (spec 4 section 5: no native Anki cloze type, so
  this field is literal pre-blanked text, not a togglable reveal) -- the
  unblanked sentence isn't recoverable from it. An extra `Thai` field
  (the sentence's full, unmodified text) carries it, alongside the spec's
  literal `ThaiCloze`/`TargetWord`/... list.
- **minimal_pair's ">2 members' audio individually playable"**: a pair
  has 2-3 members (spec 1 section 1); `OtherAudio` concatenates one
  `[sound:...]` tag per other member -- Anki renders each as its own
  playable control, so this covers 3-member pairs without extra fields.
- **card_key's word/pair anchor** (ports.py's StudyRecord docstring says
  "target/pair/grapheme id"): for a WORD's cards this is the WORD id, not
  a specific Target id -- a word note aggregates every Target the word
  has into one note (guid = word id too, spec 4 section 2), so no single
  Target id anchors it uniquely; ports.py's phrasing is read as shorthand
  for "the order()-entry identity", which bottoms out at the word for word
  cards. For pairs it IS the pair id (not MemberKey) -- Syllabus.
  study_by_confusion strips the card_key's trailing "::<kind>" and
  matches what remains against a known pair id (exact, or the longest
  pair id it starts with, since a pair id may itself contain ":") to
  group pair StudyRecords by confusion, so this compiles cards under
  exactly the "<pair_id>::<kind>" shape.
- **Bury-siblings options group** (spec 4 section 2's "the shipped deck
  options group sets bury-siblings"): genanki's own default dconf (the
  "Default" preset every genanki.Deck uses) already ships
  `new.bury=true` and `rev.bury=true` (verified against
  genanki.package.APKG_COL) -- nothing here needs to edit it; a compile
  test asserts this rather than re-implementing it.

Media (spec 4 section 3): every artifact reference is resolved via
`derivations.current_best(db, subject, kind)` -- never a raw "does the
word have a picture" boolean (spec 1's narrower MediaIndex protocol
can't tell compile() the SHA-, so compile() reads the CacheReader
surface directly, same as spec 3's own derivations do) -- then staged
into the package under its content-sha basename (`media/objects/<sha>.
<ext>`, already collision-free, so no basename renaming is needed unlike
thai_deck_gen/compiler/build.py's `_resolve_media`, which had to
namespace collisions among non-content-addressed deck-relative refs). A
missing current-best artifact leaves the dependent field empty; genanki's
own required-field computation (see above) then drops exactly the
card(s) whose front needed it, and CompileReport.dropped is compiled by
comparing each note's requested templates against the cards genanki
actually generated for it.

Due stamping (spec 4 section 2, "sibling cards ... offset by a stride"):
genanki writes ONE `due` value per NOTE onto every one of its sibling
cards (verified against genanki.Note.write_to_db) -- there is no
per-template due at Note-construction time. So, like
thai_deck_gen/compiler/build.py's `stamp_due`, this reopens the written
.apkg's collection.anki2 and overwrites `cards.due` directly, but keyed
by (note guid, card ord) rather than guid alone, so sibling cards land at
distinct, stride-separated due values instead of collapsing onto one.
"""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import genanki

from .derivations import current_best
from .entities import Grapheme, MinimalPair, Sentence, Target, Word
from .rulebook import sentence_note_id
from .rules import Compile, CompileReport, DroppedCard, Finding, Report

if TYPE_CHECKING:
    from .store import MediaStore, SyllabusDb
    from .syllabus import Syllabus

__all__ = ["compile_syllabus", "GateRefusal", "thai_cloze"]


class GateRefusal(Exception):
    """compile() refuses: report().gate is False and force was not set
    (spec 4 section 2). Carries the report so callers can inspect why.
    """
    def __init__(self, report: Report):
        blocking = [f for f in report.findings]
        super().__init__(
            f"compile refused: gate is closed ({len(blocking)} finding(s)); "
            f"pass force=True to compile anyway")
        self.report = report


# --- CSS (ported out of thai_deck_gen's compiler/build.py CARD_CSS; spec 4
# section 3: "CSS retains only final fit-to-viewport" for images) --------

CARD_CSS = """
.card { font-family: sans-serif; font-size: 24px; text-align: center;
        color: #222; background: #fff; }
img { max-width: 100%; height: auto; }
.thai, .cloze, .choices { font-size: 48px; }
.ipa { font-size: 20px; color: #666; }
.answer { font-weight: bold; }
.other { color: #888; }
.target { font-size: 40px; }
.gloss, .grammar, .classifier { font-size: 20px; color: #555; }
.nightMode .card { color: #ddd; background: #2f2f31; }
.nightMode .ipa, .nightMode .other { color: #999; }
"""


def _model_id(name: str) -> int:
    import hashlib
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def _deck_id(name: str) -> int:
    import hashlib
    return int(hashlib.sha256(f"thai-syllabus::{name}".encode()).hexdigest()[:8], 16)


def _model(name: str, fields: list[str], templates: list[dict]) -> genanki.Model:
    all_fields = [*fields, "ReviewNote", "CompileId"]
    return genanki.Model(_model_id(name), name,
                         fields=[{"name": f} for f in all_fields],
                         templates=templates, css=CARD_CSS)


WORD_MODEL = _model(
    "word",
    ["Thai", "Meaning", "Picture", "Audio", "Ipa", "Classifier", "FrontGloss",
     "TestSpelling", "ProductiveTarget"],
    [{
        "name": "Listening",
        "qfmt": "{{Audio}}",
        "afmt": '{{FrontSide}}<hr id="answer">{{Picture}}'
               '<div class="thai">{{Thai}}</div><div class="ipa">{{Ipa}}</div>'
               '<div class="gloss">{{Meaning}}</div>',
    }, {
        "name": "Production",
        "qfmt": '{{#ProductiveTarget}}{{Picture}}'
               '{{#FrontGloss}}<div class="gloss">{{FrontGloss}}</div>{{/FrontGloss}}'
               '{{/ProductiveTarget}}',
        "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{Thai}}</div>'
               '{{Audio}}<div class="ipa">{{Ipa}}</div>',
    }, {
        "name": "Reading",
        "qfmt": '<div class="thai">{{Thai}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer">{{Picture}}{{Audio}}'
               '<div class="gloss">{{Meaning}}</div>',
    }, {
        "name": "Spelling",
        "qfmt": "{{#TestSpelling}}{{Audio}}{{/TestSpelling}}",
        "afmt": '{{#TestSpelling}}{{FrontSide}}<hr id="answer">'
               '<div class="thai">{{Thai}}</div>{{/TestSpelling}}',
    }])

MINIMAL_PAIR_MODEL = _model(
    "minimal_pair",
    ["MemberKey", "Thai", "Ipa", "Audio", "OtherThai", "OtherIpa", "OtherAudio"],
    [{
        "name": "Recognition",
        "qfmt": '{{Audio}}<div>Which word did you hear?</div>'
               '<div class="choices">{{Thai}} / {{OtherThai}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer">'
               '<div class="answer">you heard: {{Thai}} '
               '<span class="ipa">[{{Ipa}}]</span> {{Audio}}</div>'
               '<div class="other">{{OtherThai}} '
               '<span class="ipa">[{{OtherIpa}}]</span> {{OtherAudio}}</div>',
    }])

# Audio is a field but not referenced by qfmt (a grapheme's front is
# always its Symbol -- media never gates this card, spec 4 section 1:
# "one card; no reverse family"); it belongs on the back, so it's added
# after `_model`'s ReviewNote/CompileId append point by building the
# genanki.Model directly rather than through the `_model` helper.
GRAPHEME_MODEL = genanki.Model(
    _model_id("grapheme"), "grapheme",
    fields=[{"name": f} for f in
           ["Symbol", "Sound", "NameThai", "KeywordThai", "KeywordGloss",
            "KeywordPicture", "Audio", "ReviewNote", "CompileId"]],
    templates=[{
        "name": "Reading",
        "qfmt": '<div class="thai">{{Symbol}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{NameThai}}</div>'
               '{{KeywordPicture}}<div class="thai">{{KeywordThai}}</div>'
               '{{#KeywordGloss}}<div class="gloss">{{KeywordGloss}}</div>{{/KeywordGloss}}'
               '{{Audio}}<div class="ipa">{{Sound}}</div>',
    }],
    css=CARD_CSS)

SENTENCE_MODEL = _model(
    "sentence",
    ["ThaiCloze", "Thai", "TargetWord", "Audio", "ScenePicture", "Gloss",
     "GrammarNote"],
    [{
        "name": "Cloze",
        "qfmt": '<div class="cloze">{{ThaiCloze}}</div>{{ScenePicture}}',
        "afmt": '{{FrontSide}}<hr id="answer"><div class="target">{{TargetWord}}</div>'
               '{{Audio}}{{#Gloss}}<div class="gloss">{{Gloss}}</div>{{/Gloss}}'
               '{{#GrammarNote}}<div class="grammar">{{GrammarNote}}</div>{{/GrammarNote}}',
    }, {
        "name": "Listening",
        "qfmt": "{{Audio}}",
        "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{Thai}}</div>'
               '<div class="target">{{TargetWord}}</div>'
               '{{#Gloss}}<div class="gloss">{{Gloss}}</div>{{/Gloss}}',
    }])

STRIDE = 100  # due-per-order-position block size; comfortably above the
             # largest sibling count any family below uses (word: 4).


def _guid(family: str, *parts: str) -> str:
    return genanki.guid_for(family, *parts)


# --- ThaiCloze: tokenized boundary replacement, never str.replace --------

def thai_cloze(tokens: list[str], target_thai: str, blank: str = "___") -> str:
    """Blank every token that boundary-matches `target_thai` (Syllabus.
    _boundary_match's own rule: exact match, or a compound token starting/
    ending with it) and rejoin. A whole matching token is blanked, never a
    substring within it -- this is what actually prevents the "ยา"/
    "โรงพยาบาล" corruption (str.replace would blind-match the substring
    anywhere; token-boundary matching only ever considers whole tokens, and
    "โรงพยาบาล" does not start or end with "ยา", so it is never touched).
    """
    def matches(tok: str) -> bool:
        return tok == target_thai or tok.startswith(target_thai) or tok.endswith(target_thai)

    return "".join(blank if matches(tok) else tok for tok in tokens)


# --- media resolution ------------------------------------------------------

@dataclass
class _Resolver:
    """Resolves (subject, kind) to the CURRENT-BEST artifact (spec 3
    derivations.current_best over `db` as a CacheReader), staged into the
    package under its content-sha basename. `used` collects every
    basename actually referenced, for _stage_media; `warnings` collects
    non-fatal data-integrity notes (e.g. a resolved sha with no
    media-table provenance row) that land in CompileReport.warnings.
    """
    db: "SyllabusDb"
    media_store: "MediaStore"
    used: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def artifact(self, subject: str, kind: str) -> tuple[str, str] | None:
        best = current_best(self.db, subject, kind)
        if best.artifact_sha is None:
            return None
        prov = self.db.media_provenance(best.artifact_sha)
        if prov is None:
            self.warnings.append(
                f"current-best {kind} {best.artifact_sha!r} for {subject!r} "
                f"has no media provenance row -- skipped")
            return None
        ext = prov["ext"]
        path = self.media_store.path_for(best.artifact_sha, ext)
        if not path.exists():
            self.warnings.append(
                f"media object missing on disk: {best.artifact_sha}.{ext} "
                f"(subject={subject!r} kind={kind!r})")
            return None
        basename = f"{best.artifact_sha}.{ext}"
        self.used[basename] = path
        return best.artifact_sha, ext

    def sound(self, subject: str, kind: str) -> str:
        got = self.artifact(subject, kind)
        return f"[sound:{got[0]}.{got[1]}]" if got else ""

    def img(self, subject: str, kind: str) -> str:
        got = self.artifact(subject, kind)
        return f'<img src="{got[0]}.{got[1]}">' if got else ""

    def provenance(self, sha: str) -> dict[str, Any] | None:
        return self.db.media_provenance(sha)

    def src_tag(self, prefix: str, subject: str, kind: str) -> list[str]:
        got = self.artifact(subject, kind)
        if not got:
            return []
        prov = self.provenance(got[0])
        source = prov.get("source") if prov else None
        return [f"{prefix}-src::{source}"] if source else []

    def speaker(self, subject: str, kind: str) -> str:
        got = self.artifact(subject, kind)
        if not got:
            return "unknown"
        prov = self.provenance(got[0])
        return (prov or {}).get("speaker_id") or "unknown"


# --- order positions ---------------------------------------------------

@dataclass
class _Positions:
    """Where each order()-entry sits, plus one due block per (sentence,
    target) fill (spec 4 section 2): a sentence's own position comes
    straight from its order() entry -- never recomputed here.
    """
    entry_index: dict[str, int]           # grapheme symbol / pair id -> position
    target_index: dict[str, int]          # target id -> position
    word_index: dict[str, int]            # word id -> min position of its targets
    sentence_entries: list[tuple[Sentence, Target, int]]  # (sentence, target, due block index)
    order_length: int


def _positions(syllabus: "Syllabus") -> _Positions:
    order_list = syllabus.order()
    entry_index: dict[str, int] = {}
    target_index: dict[str, int] = {}
    word_index: dict[str, int] = {}
    sentence_position: dict[str, int] = {}
    target_word = {t.id: t.word for t in syllabus.targets}
    for i, entry in enumerate(order_list):
        if entry.kind == "word_target":
            target_index[entry.id] = i
            word = target_word[entry.id]
            word_index[word] = min(word_index.get(word, i), i)
        elif entry.kind == "sentence":
            sentence_position[entry.id] = i
        else:
            entry_index[entry.id] = i

    fills_entries: list[tuple[Sentence, Target, int]] = []
    for s in syllabus.sentences:
        for t in syllabus.targets:
            if syllabus.fills(s, t):
                position = sentence_position.get(sentence_note_id(s), len(order_list))
                fills_entries.append((s, t, position))
    fills_entries.sort(key=lambda e: (e[2], sentence_note_id(e[0]), e[1].id))
    sentence_entries = [(s, t, len(order_list) + i)
                        for i, (s, t, _) in enumerate(fills_entries)]

    return _Positions(entry_index=entry_index, target_index=target_index,
                      word_index=word_index, sentence_entries=sentence_entries,
                      order_length=len(order_list))


# --- note builders -----------------------------------------------------
# Each returns (genanki.Note, due, family, subject) or None when the
# family has nothing to compile for this item; the caller detects
# per-template drops by comparing note.cards against the model's
# templates after construction.

def _word_note(syllabus: "Syllabus", word: Word, resolver: _Resolver,
               compile_id: str, positions: _Positions) -> tuple[genanki.Note, int] | None:
    if word.id not in positions.word_index:
        return None  # no Target at all -- not a compiled word (spec 4 section 1)

    productive = any(t.word == word.id and t.skill == "productive"
                     for t in syllabus.targets)
    classifier_word = syllabus.find_word(word.classifier) if word.classifier else None

    tags = [f"family::word", f"target::{word.id}", f"compile::{compile_id}"]
    tags += [f"kind::{tpl['name'].lower()}" for tpl in WORD_MODEL.templates]
    tags += resolver.src_tag("img", word.id, "picture")
    tags += resolver.src_tag("audio", word.id, "recording")

    fields = [
        word.thai,
        word.meaning,
        resolver.img(word.id, "picture"),
        resolver.sound(word.id, "recording"),
        _ipa(word),
        classifier_word.thai if classifier_word else "",
        "",  # FrontGloss: F3 variant point, empty by default (spec 4 section 1)
        "1" if productive else "",   # TestSpelling: parked, mirrors ProductiveTarget for now
        "1" if productive else "",   # ProductiveTarget
        "",  # ReviewNote: mid-review comment channel, rendered by no template
        compile_id,
    ]
    note = genanki.Note(model=WORD_MODEL, fields=fields, tags=tags,
                        guid=_guid("word", word.id))
    due = positions.word_index[word.id] * STRIDE
    return note, due


def _ipa(word: Word) -> str:
    """A readable IPA-ish rendering of `word.pron` (spec 1's Pronunciation
    doesn't carry a pre-rendered string -- entities.py's Syllable exposes
    onset/vowel/coda). Not a phonetically rigorous IPA transcription
    (tone marking, length diacritics), just a stable per-syllable join
    good enough for a card's back; refining it is out of this spec's
    scope.
    """
    return ".".join(f"{s.onset}{s.vowel}{s.coda}" for s in word.pron.syllables)


def _pair_notes(pair: MinimalPair, syllabus: "Syllabus", resolver: _Resolver,
                compile_id: str, positions: _Positions) -> list[tuple[genanki.Note, int]]:
    if pair.id not in positions.entry_index:
        return []
    base_due = positions.entry_index[pair.id] * STRIDE
    members = [syllabus.find_word(m) for m in pair.members]
    if any(m is None for m in members):
        return []  # syllabus/closure already flags this; compile just skips it

    notes = []
    for i, member in enumerate(members):
        others = [m for j, m in enumerate(members) if j != i]
        speaker = resolver.speaker(member.id, "recording")
        member_key = f"{pair.id}:{speaker}:{i}"
        tags = [f"family::minimal_pair", f"pair::{pair.id}",
               f"confusion::{pair.confusion}", f"compile::{compile_id}",
               "kind::recognition"]
        tags += resolver.src_tag("audio", member.id, "recording")
        fields = [
            member_key,
            member.thai,
            _ipa(member),
            resolver.sound(member.id, "recording"),
            " / ".join(o.thai for o in others),
            " / ".join(_ipa(o) for o in others),
            "".join(resolver.sound(o.id, "recording") for o in others),
            "",
            compile_id,
        ]
        note = genanki.Note(model=MINIMAL_PAIR_MODEL, fields=fields, tags=tags,
                            guid=_guid("minimal_pair", member_key))
        notes.append((note, base_due + i))
    return notes


def _grapheme_note(grapheme: Grapheme, syllabus: "Syllabus", resolver: _Resolver,
                   compile_id: str, positions: _Positions) -> tuple[genanki.Note, int] | None:
    if grapheme.symbol not in positions.entry_index:
        return None
    keyword = syllabus.find_word(grapheme.keyword)
    if keyword is None:
        return None  # syllabus/closure already flags this
    name_word = syllabus.find_word(grapheme.name_word) if grapheme.name_word else None

    # name_word.thai IS the full recited name (e.g. กอ ไก่ "gɔɔ gài", the
    # letter ก) -- one Word, never composed with the keyword.
    name_thai = name_word.thai if name_word else f"{grapheme.symbol} {keyword.thai}"

    audio = ""
    audio_subject = None
    if name_word is not None:
        audio = resolver.sound(name_word.id, "recording")
        audio_subject = name_word.id if audio else None
    if not audio:
        audio = resolver.sound(keyword.id, "recording")
        audio_subject = keyword.id if audio else audio_subject

    tags = ["family::grapheme", f"grapheme::{grapheme.symbol}",
           f"compile::{compile_id}", "kind::reading"]
    tags += resolver.src_tag("img", keyword.id, "picture")
    if audio_subject:
        tags += resolver.src_tag("audio", audio_subject, "recording")

    fields = [
        grapheme.symbol,
        grapheme.sound,
        name_thai,
        keyword.thai,
        keyword.meaning,
        resolver.img(keyword.id, "picture"),
        audio,
        "",
        compile_id,
    ]
    note = genanki.Note(model=GRAPHEME_MODEL, fields=fields, tags=tags,
                        guid=_guid("grapheme", grapheme.symbol))
    due = positions.entry_index[grapheme.symbol] * STRIDE
    return note, due


def _sentence_note(sentence: Sentence, target: Target, due_block: int,
                   syllabus: "Syllabus", resolver: _Resolver,
                   compile_id: str) -> tuple[genanki.Note, int] | None:
    target_word = syllabus.find_word(target.word)
    if target_word is None:
        return None
    tokens = syllabus.tokenizer.tokens(sentence.text)
    cloze = thai_cloze(tokens, target_word.thai)
    text_sha = sentence_note_id(sentence)

    # kind = "recording"/"picture" (not "sentence-recording"/"scene-
    # picture"): derivations.py's gap/kind vocabulary is just
    # "picture"/"recording" throughout (see _gap_candidates); text_sha as
    # the subject already disambiguates a sentence's own audio/picture
    # from any word's, so no separate kind is needed here.
    tags = ["family::sentence", f"target::{target.id}", f"sentence::{text_sha}",
           f"compile::{compile_id}", "kind::cloze", "kind::listening"]
    tags += resolver.src_tag("audio", text_sha, "recording")
    tags += resolver.src_tag("img", text_sha, "picture")

    fields = [
        cloze,
        sentence.text,
        target_word.thai,
        resolver.sound(text_sha, "recording"),
        resolver.img(text_sha, "picture"),
        sentence.gloss,
        "",  # GrammarNote: no curated source yet
        "",
        compile_id,
    ]
    note = genanki.Note(model=SENTENCE_MODEL, fields=fields, tags=tags,
                        guid=_guid("sentence", target.id, text_sha))
    due = due_block * STRIDE
    return note, due


# --- card/unique-front (A3) -------------------------------------------------
# A minimal mustache-section renderer, just enough to compare rendered card
# fronts -- not a full Anki template engine.

_HASH_SECTION_RE = re.compile(r"{{#([A-Za-z0-9_]+)}}(.*?){{/\1}}", re.DOTALL)
_CARET_SECTION_RE = re.compile(r"{{\^([A-Za-z0-9_]+)}}(.*?){{/\1}}", re.DOTALL)
_FIELD_RE = re.compile(r"{{([A-Za-z0-9_]+)}}")


def _render_qfmt(qfmt: str, values: Mapping[str, str]) -> str:
    def hashed(m: re.Match) -> str:
        return _render_qfmt(m.group(2), values) if values.get(m.group(1)) else ""

    def caret(m: re.Match) -> str:
        return _render_qfmt(m.group(2), values) if not values.get(m.group(1)) else ""

    text = _HASH_SECTION_RE.sub(hashed, qfmt)
    text = _CARET_SECTION_RE.sub(caret, text)
    return _FIELD_RE.sub(lambda m: values.get(m.group(1), ""), text)


def _record_fronts(entries: list[tuple[str, str, str]], model: genanki.Model,
                   subject: str, note: genanki.Note) -> None:
    """Appends (model:ord, subject, rendered front) for every card the note
    actually generated -- card/unique-front compares these within a
    (model, ord) group.
    """
    values = dict(zip((f["name"] for f in model.fields), note.fields))
    for card in note.cards:
        front = _render_qfmt(model.templates[card.ord]["qfmt"], values)
        entries.append((f"{model.name}:{card.ord}", subject, front))


def _duplicate_front_findings(entries: list[tuple[str, str, str]]) -> list[Finding]:
    by_front: dict[tuple[str, str], list[str]] = {}
    for group, subject, front in entries:
        by_front.setdefault((group, front), []).append(subject)
    findings = []
    for (group, _front), subjects in by_front.items():
        if len(subjects) < 2:
            continue
        for subject in subjects:
            others = sorted(s for s in subjects if s != subject)
            findings.append(Finding(rule="card/unique-front", note_id=subject,
                                    evidence=f"front matches {others} ({group})"))
    return findings


# --- assembly ------------------------------------------------------------

def _dropped_for(note: genanki.Note, model: genanki.Model, family: str,
                 subject: str, reason: str) -> list[DroppedCard]:
    present_ords = {c.ord for c in note.cards}
    return [DroppedCard(family=family, kind=tpl["name"], subject=subject, reason=reason)
           for ord_, tpl in enumerate(model.templates) if ord_ not in present_ords]


def _stage_media(used: dict[str, Path], tmp_dir: Path) -> list[Path]:
    # Content-sha basenames are already globally unique -- unlike
    # thai_deck_gen/compiler/build.py's _resolve_media (deck-relative
    # refs, needs collision namespacing), objects can be handed to
    # genanki straight from media/objects/ with no staging copy needed.
    return list(used.values())


def _stamp_due(apkg_path: Path, due_by_guid_ord: dict[tuple[str, int], int]) -> None:
    """Post-process pass (ported out of thai_deck_gen's compiler/build.py
    stamp_due): genanki writes one `due` per NOTE onto every sibling card
    (verified against genanki.Note.write_to_db), so per-template stride
    separation needs this direct sqlite rewrite, keyed by (guid, ord)
    instead of guid alone.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(apkg_path) as zf:
            names = zf.namelist()
            zf.extractall(tmp_dir)

        db_path = tmp_dir / "collection.anki2"
        conn = sqlite3.connect(str(db_path))
        try:
            guid_by_nid = dict(conn.execute("select id, guid from notes"))
            for card_id, nid, ord_ in conn.execute("select id, nid, ord from cards"):
                due = due_by_guid_ord.get((guid_by_nid.get(nid), ord_))
                if due is not None:
                    conn.execute("update cards set due = ? where id = ?", (due, card_id))
            conn.commit()
        finally:
            conn.close()

        with zipfile.ZipFile(apkg_path, "w") as zf:
            for name in names:
                zf.write(db_path if name == "collection.anki2" else tmp_dir / name, name)


def compile_syllabus(syllabus: "Syllabus", db: "SyllabusDb", media_store: "MediaStore",
                     out_path: str | Path, *, force: bool = False,
                     now: Callable[[], float] = time.time) -> Compile:
    report = syllabus.report()
    warnings: list[str] = []
    if not report.gate:
        if not force:
            raise GateRefusal(report)
        for f in report.findings:
            if syllabus._severity(f.rule) == "error" and not syllabus.assessments.is_waived(f):
                warnings.append(f"{f.rule}: {f.evidence} (note {f.note_id})")

    out_path = Path(out_path)
    state_id = syllabus.state_id()
    ts = int(now() * 1000)
    compile_id = f"{state_id}:{ts}"

    resolver = _Resolver(db=db, media_store=media_store)
    positions = _positions(syllabus)
    deck_name = out_path.stem
    deck = genanki.Deck(_deck_id(deck_name), deck_name)

    dropped: list[DroppedCard] = []
    notes_written = 0
    cards_written = 0
    due_by_guid_ord: dict[tuple[str, int], int] = {}
    front_entries: list[tuple[str, str, str]] = []

    targeted_word_ids = {t.word for t in syllabus.targets}
    for word in syllabus.words:
        if word.id not in targeted_word_ids:
            continue
        built = _word_note(syllabus, word, resolver, compile_id, positions)
        if built is None:
            continue
        note, base_due = built
        if not note.cards:
            dropped.extend(_dropped_for(note, WORD_MODEL, "word", word.id,
                                        "no current-best artifact resolves for any template"))
            continue
        dropped.extend(_dropped_for(note, WORD_MODEL, "word", word.id,
                                    "no current-best artifact resolves"))
        deck.add_note(note)
        _record_fronts(front_entries, WORD_MODEL, word.id, note)
        for c in note.cards:
            due_by_guid_ord[(note.guid, c.ord)] = base_due + c.ord
        notes_written += 1
        cards_written += len(note.cards)

    for pair in syllabus.pairs:
        for note, base_due in _pair_notes(pair, syllabus, resolver, compile_id, positions):
            if not note.cards:
                dropped.append(DroppedCard(family="minimal_pair", kind="Recognition",
                                           subject=note.fields[0],
                                           reason="no current-best recording resolves"))
                continue
            deck.add_note(note)
            _record_fronts(front_entries, MINIMAL_PAIR_MODEL, note.fields[0], note)
            due_by_guid_ord[(note.guid, 0)] = base_due
            notes_written += 1
            cards_written += len(note.cards)

    for grapheme in syllabus.graphemes:
        built = _grapheme_note(grapheme, syllabus, resolver, compile_id, positions)
        if built is None:
            continue
        note, base_due = built
        if not note.cards:
            dropped.append(DroppedCard(family="grapheme", kind="Reading",
                                       subject=grapheme.symbol,
                                       reason="Symbol field unexpectedly empty"))
            continue
        deck.add_note(note)
        _record_fronts(front_entries, GRAPHEME_MODEL, grapheme.symbol, note)
        due_by_guid_ord[(note.guid, 0)] = base_due
        notes_written += 1
        cards_written += len(note.cards)

    for sentence, target, due_block in positions.sentence_entries:
        built = _sentence_note(sentence, target, due_block, syllabus, resolver, compile_id)
        if built is None:
            continue
        note, base_due = built
        if not note.cards:
            dropped.extend(_dropped_for(note, SENTENCE_MODEL, "sentence",
                                        f"{target.id}:{sentence_note_id(sentence)}",
                                        "ThaiCloze field unexpectedly empty"))
            continue
        dropped.extend(_dropped_for(note, SENTENCE_MODEL, "sentence",
                                    f"{target.id}:{sentence_note_id(sentence)}",
                                    "no current-best artifact resolves"))
        deck.add_note(note)
        _record_fronts(front_entries, SENTENCE_MODEL,
                       f"{target.id}:{sentence_note_id(sentence)}", note)
        for c in note.cards:
            due_by_guid_ord[(note.guid, c.ord)] = base_due + c.ord
        notes_written += 1
        cards_written += len(note.cards)

    unique_front_rule = next((r for r in syllabus.rules if r.id == "card/unique-front"), None)
    front_findings = _duplicate_front_findings(front_entries) if unique_front_rule else []
    blocking_front_findings = [f for f in front_findings
                               if unique_front_rule.severity == "error"
                               and not syllabus.assessments.is_waived(f)]
    if blocking_front_findings and not force:
        raise GateRefusal(replace(report, gate=False,
                                  findings=report.findings + tuple(front_findings)))
    for f in blocking_front_findings:
        warnings.append(f"{f.rule}: {f.evidence} (note {f.note_id})")
    gate = report.gate and not blocking_front_findings

    warnings.extend(resolver.warnings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with tempfile.TemporaryDirectory() as tmp:
        media_paths = _stage_media(resolver.used, Path(tmp))
        genanki.Package(deck, media_files=[str(p) for p in media_paths]).write_to_file(
            str(tmp_out), timestamp=now())
    _stamp_due(tmp_out, due_by_guid_ord)
    os.replace(tmp_out, out_path)  # atomic (thai_deck_gen/compiler/build.py's pattern)

    compile_report = CompileReport(
        compile_id=compile_id, gate=gate, forced=force,
        warnings=tuple(warnings), notes_written=notes_written,
        cards_written=cards_written, dropped=tuple(dropped),
        out_path=str(out_path), findings=tuple(front_findings))
    return Compile(label=deck_name, syllabus_state_id=state_id,
                   compile_id=compile_id, report=compile_report)
