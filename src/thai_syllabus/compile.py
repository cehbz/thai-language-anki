"""compile_syllabus (spec 4): translate a Syllabus into an Anki .apkg.

`compile_syllabus(syllabus, db, media_store, out_path, *, force=False,
now=time.time)` takes a Syllabus, a SyllabusDb (current-best artifact
lookups and media provenance), a MediaStore (staged media bytes on disk)
and an output path. It writes one .apkg: one note per word, grapheme and
(sentence, target) pair with a card-yielding skill, one note per
minimal-pair member; every note tagged, due-stamped from
Syllabus.order(), and stamped with this compile's CompileId.

It refuses (raises GateRefusal) when Syllabus.report().gate is False, or
when the compiled notes produce duplicate card fronts under rule
card/unique-front, unless `force=True` -- a forced compile stamps the
blocking findings into CompileReport.warnings instead of raising.

It counts every card a template did not produce, with a reason
(CompileReport.dropped), and returns a Compile carrying the compile id,
gate/forced status, and notes/cards written.
"""
from __future__ import annotations

import os
import re
import sqlite3
import tempfile
import time
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import genanki

from . import ipa
from .derivations import current_best
from .entities import Grapheme, MinimalPair, Sentence, Target, Word
from .rulebook import sentence_note_id
from .rules import Compile, CompileReport, DroppedCard, Finding, OrderEntry, Report

if TYPE_CHECKING:
    from .store import MediaStore, SyllabusDb
    from .syllabus import Syllabus

__all__ = ["BuiltDeck", "build_deck", "compile_syllabus", "GateRefusal", "render_card",
          "thai_cloze"]


class GateRefusal(Exception):
    """Raised when compile refuses: report().gate is False and force was
    not set. Carries the report; `blocking` is the count of unwaived
    error-severity findings that closed the gate.
    """
    def __init__(self, report: Report, blocking: Sequence[Finding]):
        self.blocking = len(blocking)
        super().__init__(
            f"compile refused: gate is closed ({self.blocking} finding(s)); "
            f"pass force=True to compile anyway")
        self.report = report


# --- CSS (spec 4 section 3: "CSS retains only final fit-to-viewport") ---

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
    ["MemberKey", "Speaker", "Choices", "Audio", "Stimulus", "Ipa", "OtherIpa", "OtherAudio"],
    [{
        "name": "Recognition",
        "qfmt": '{{Audio}}<div>Which word did you hear?</div>'
               '<div class="choices">{{Choices}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer">'
               '<div class="answer">you heard: {{Stimulus}} '
               '<span class="ipa">[{{Ipa}}]</span></div>'
               '<div class="other"><span class="ipa">[{{OtherIpa}}]</span> {{OtherAudio}}</div>',
    }])

# Audio is a field but not referenced by qfmt: a grapheme's front is
# always its Symbol -- media never gates this card (spec 4 section 1:
# "one card; no reverse family"); Audio renders on the back only.
GRAPHEME_MODEL = _model(
    "grapheme",
    ["Symbol", "Sound", "NameThai", "KeywordThai", "KeywordGloss",
     "KeywordPicture", "Audio"],
    [{
        "name": "Reading",
        "qfmt": '<div class="thai">{{Symbol}}</div>',
        "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{NameThai}}</div>'
               '{{KeywordPicture}}<div class="thai">{{KeywordThai}}</div>'
               '{{#KeywordGloss}}<div class="gloss">{{KeywordGloss}}</div>{{/KeywordGloss}}'
               '{{Audio}}<div class="ipa">{{Sound}}</div>',
    }])

SENTENCE_MODEL = _model(
    "sentence",
    ["ThaiCloze", "Thai", "TargetWord", "Audio", "ScenePicture", "Gloss",
     "GrammarNote", "Productive"],
    [{
        "name": "Cloze",
        "qfmt": '{{#Productive}}<div class="cloze">{{ThaiCloze}}</div>'
               '{{ScenePicture}}{{/Productive}}',
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


def thai_cloze(tokens: list[str], target_thai: str, blank: str = "___") -> str:
    """Blanks every token that boundary-matches `target_thai` (exact
    match, or a compound token starting/ending with it) and rejoins.
    Blanks a whole token, never a substring, so "โรงพยาบาล" ("hospital")
    is untouched when blanking "ยา" ("medicine").
    """
    def matches(tok: str) -> bool:
        return tok == target_thai or tok.startswith(target_thai) or tok.endswith(target_thai)

    return "".join(blank if matches(tok) else tok for tok in tokens)


# --- media resolution ------------------------------------------------------

@dataclass
class _Resolver:
    """Resolves (subject, kind) to the current-best artifact, staged
    under its content-sha basename. `used` maps each referenced basename
    to its on-disk path (handed to genanki.Package as media_files);
    `warnings` collects non-fatal data-integrity notes for
    CompileReport.warnings.
    """
    db: "SyllabusDb"
    media_store: "MediaStore"
    used: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def artifact(self, subject: str, kind: str) -> tuple[str, str] | None:
        best = current_best(self.db, subject, kind, current_rubric={}, prior=(),
                            provenance_source=lambda s: None)
        if best.artifact_sha is None:
            return None
        prov = self.db.media_provenance(best.artifact_sha)
        if prov is None:
            self.warnings.append(
                f"current-best {kind} {best.artifact_sha!r} for {subject!r} "
                f"has no media provenance row -- skipped")
            return None
        return self._stage(best.artifact_sha, prov["ext"], f"subject={subject!r} kind={kind!r}")

    def _stage(self, sha: str, ext: str, context: str) -> tuple[str, str] | None:
        path = self.media_store.path_for(sha, ext)
        if not path.exists():
            self.warnings.append(f"media object missing on disk: {sha}.{ext} ({context})")
            return None
        basename = f"{sha}.{ext}"
        self.used[basename] = path
        return sha, ext

    def sound(self, subject: str, kind: str) -> str:
        got = self.artifact(subject, kind)
        return f"[sound:{got[0]}.{got[1]}]" if got else ""

    def img(self, subject: str, kind: str) -> str:
        got = self.artifact(subject, kind)
        return f'<img src="{got[0]}.{got[1]}">' if got else ""

    def rendition_sound(self, pair_id: str, sha: str) -> str:
        """[sound:sha.ext] for one member sha of `pair_id`'s rendition;
        a missing provenance row or on-disk object warns and yields "".
        """
        prov = self.provenance(sha)
        if prov is None:
            self.warnings.append(
                f"rendition member sha {sha!r} for pair {pair_id!r} has no "
                "media provenance row -- skipped")
            return ""
        got = self._stage(sha, prov["ext"], f"pair={pair_id!r}")
        return f"[sound:{got[0]}.{got[1]}]" if got else ""

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
    """Where each order()-entry's due block starts, in STRIDE units, plus
    one due block per (sentence, target) fill. An entry's block is
    `width` units wide (a pair: len(members); everything else: 1), so
    blocks are cumulative and never overlap.
    """
    entry_index: dict[str, int]           # grapheme symbol / pair id -> block start
    target_index: dict[str, int]          # target id -> block start
    word_index: dict[str, int]            # word id -> min block start of its targets
    sentence_entries: list[tuple[Sentence, Target, int]]  # (sentence, target, due block index)
    order_length: int


def _order_entry_width(entry: OrderEntry, pairs_by_id: Mapping[str, MinimalPair]) -> int:
    """Due-block width of one order() entry, in STRIDE units: a pair
    needs one unit per member; every other kind needs exactly one.
    """
    if entry.kind == "pair":
        pair = pairs_by_id.get(entry.id)
        return len(pair.members) if pair is not None else 1
    return 1


def _positions(syllabus: "Syllabus") -> _Positions:
    order_list = syllabus.order()
    pairs_by_id = {p.id: p for p in syllabus.pairs}
    entry_index: dict[str, int] = {}
    target_index: dict[str, int] = {}
    word_index: dict[str, int] = {}
    sentence_position: dict[str, int] = {}
    target_word = {t.id: t.word for t in syllabus.targets}
    block = 0
    for entry in order_list:
        if entry.kind == "word_target":
            target_index[entry.id] = block
            word = target_word[entry.id]
            word_index[word] = min(word_index.get(word, block), block)
        elif entry.kind == "sentence":
            sentence_position[entry.id] = block
        else:
            entry_index[entry.id] = block
        block += _order_entry_width(entry, pairs_by_id)
    total_blocks = block

    fills_entries: list[tuple[Sentence, Target, int]] = []
    for s in syllabus.sentences:
        for t in syllabus.targets:
            if syllabus.fills(s, t):
                position = sentence_position.get(sentence_note_id(s), total_blocks)
                fills_entries.append((s, t, position))
    fills_entries.sort(key=lambda e: (e[2], sentence_note_id(e[0]), e[1].id))
    sentence_entries = [(s, t, total_blocks + i)
                        for i, (s, t, _) in enumerate(fills_entries)]

    return _Positions(entry_index=entry_index, target_index=target_index,
                      word_index=word_index, sentence_entries=sentence_entries,
                      order_length=len(order_list))


# --- note builders -----------------------------------------------------
# One builder per family, called from that family's *_items generator
# below; each returns (genanki.Note, due) or None for an item with
# nothing to compile.

def _word_note(syllabus: "Syllabus", word: Word, resolver: _Resolver,
               compile_id: str, positions: _Positions) -> tuple[genanki.Note, int] | None:
    if word.id not in positions.word_index:
        return None  # no Target at all -- not a compiled word (spec 4 section 1)

    productive = any(t.word == word.id and t.skill == "productive"
                     for t in syllabus.targets)
    classifier_word = syllabus.find_word(word.classifier) if word.classifier else None

    tags = [f"family::word", f"word::{word.id}", f"compile::{compile_id}"]
    tags += [f"kind::{tpl['name'].lower()}" for tpl in WORD_MODEL.templates]
    tags += resolver.src_tag("img", word.id, "picture")
    tags += resolver.src_tag("audio", word.id, "recording")

    fields = [
        word.thai,
        word.meaning,
        resolver.img(word.id, "picture"),
        resolver.sound(word.id, "recording"),
        ipa.render(word.pron),
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


def _pair_notes(pair: MinimalPair, syllabus: "Syllabus", recordings: tuple,
                resolver: _Resolver, compile_id: str,
                positions: _Positions) -> list[tuple[genanki.Note, int]]:
    """One note per member of `pair`, all playing `recordings` (the
    pair's current-best rendition, one per member, in member order).
    `Choices` lists every member in that same fixed order on every note,
    so which member is this note's own stimulus never shows through
    choice position. Member notes sit one STRIDE apart.
    """
    base_due = positions.entry_index[pair.id] * STRIDE
    members = [syllabus.find_word(m) for m in pair.members]
    if any(m is None for m in members):
        return []  # syllabus/closure already flags this; compile just skips it

    choices = " / ".join(m.thai for m in members)
    notes = []
    for i, member in enumerate(members):
        other_indices = [j for j in range(len(members)) if j != i]
        speaker = recordings[i].speaker.id
        member_key = f"{pair.id}:{speaker}:{i}"
        tags = ["family::minimal_pair", f"pair::{pair.id}",
               f"confusion::{pair.confusion}", f"member::{i}", f"speaker::{speaker}",
               f"compile::{compile_id}",
               "kind::recognition", f"audio-src::{recordings[i].provenance.source}"]
        fields = [
            member_key,
            speaker,
            choices,
            resolver.rendition_sound(pair.id, recordings[i].sha),
            member.thai,
            ipa.render(member.pron),
            " / ".join(ipa.render(members[j].pron) for j in other_indices),
            "".join(resolver.rendition_sound(pair.id, recordings[j].sha) for j in other_indices),
            "",
            compile_id,
        ]
        note = genanki.Note(model=MINIMAL_PAIR_MODEL, fields=fields, tags=tags,
                            guid=_guid("minimal_pair", member_key))
        notes.append((note, base_due + i * STRIDE))
    return notes


@dataclass(frozen=True)
class _GraphemeBuild:
    """Either a built note or the reason its card was dropped -- exactly
    one of `note`/`due` and `dropped_reason` is set. `dropped_reason` is
    also None when the grapheme isn't compiled at all and isn't counted
    (not in order(), or its keyword is unresolved).
    """
    note: genanki.Note | None
    due: int | None
    dropped_reason: str | None


def _grapheme_note(grapheme: Grapheme, syllabus: "Syllabus", resolver: _Resolver,
                   compile_id: str, positions: _Positions) -> _GraphemeBuild:
    if grapheme.symbol not in positions.entry_index:
        return _GraphemeBuild(None, None, None)
    keyword = syllabus.find_word(grapheme.keyword)
    if keyword is None:
        return _GraphemeBuild(None, None, None)  # syllabus/closure already flags this
    name_word = syllabus.find_word(grapheme.name_word) if grapheme.name_word else None
    if name_word is None:
        return _GraphemeBuild(None, None, "no name word")

    # NameThai is the name word's own text (e.g. กอ ไก่ "gɔɔ gài", the
    # recited name of the letter ก) -- one Word whose recording says the
    # whole name; no substitute audio (spec 4 section 1).
    audio = resolver.sound(name_word.id, "recording")
    if not audio:
        return _GraphemeBuild(None, None, "no name recording")

    tags = ["family::grapheme", f"grapheme::{grapheme.symbol}",
           f"compile::{compile_id}", "kind::reading"]
    tags += resolver.src_tag("img", keyword.id, "picture")
    tags += resolver.src_tag("audio", name_word.id, "recording")

    fields = [
        grapheme.symbol,
        grapheme.sound,
        name_word.thai,
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
    return _GraphemeBuild(note, due, None)


def _sentence_note(sentence: Sentence, target: Target, due_block: int,
                   syllabus: "Syllabus", resolver: _Resolver,
                   compile_id: str) -> tuple[genanki.Note, int] | None:
    target_word = syllabus.find_word(target.word)
    if target_word is None:
        return None
    tokens = syllabus.tokenizer.tokens(sentence.text)
    cloze = thai_cloze(tokens, target_word.thai)
    text_sha = sentence_note_id(sentence)
    productive = target.skill == "productive"

    # A sentence's audio/picture are resolved by (text_sha, kind), the
    # same artifact kinds a word's audio and picture carry.
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
        "1" if productive else "",   # Productive
        "",  # ReviewNote
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


def render_card(model: genanki.Model, note: genanki.Note, ord_: int) -> tuple[str, str]:
    """Front and back HTML for one card (`ord_` into `model.templates`) of
    `note`, substituted through the same mustache subset that computes
    card/unique-front's fronts -- extended to afmt and the Anki
    {{FrontSide}} convention (the rendered front, injected into the
    back). This is what the review screen renders (spec 5 section 1):
    the model's own qfmt/afmt, nothing recomposed, so the learner judges
    the card Anki will actually show (principles F4).
    """
    values = dict(zip((f["name"] for f in model.fields), note.fields))
    template = model.templates[ord_]
    front = _render_qfmt(template["qfmt"], values)
    back = _render_qfmt(template["afmt"], {**values, "FrontSide": front})
    return front, back


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

@dataclass(frozen=True)
class _DropCause:
    """What a (model, template) pair's card-generation depends on: either
    a `gate_field`, whose emptiness means the card wasn't asked for
    (reason `gate_reason`), or the `artifact_kind` whose current-best
    absence is why the card has no front ("no current-best
    <artifact_kind>").
    """
    gate_field: str | None
    gate_reason: str | None
    artifact_kind: str


# One entry per (model name, template name) for word and sentence, whose
# card presence is resolved through genanki's own required-field
# computation rather than decided before the note is built (grapheme and
# minimal_pair decide their one drop reason before building the note).
_TEMPLATE_DROP_CAUSES: dict[tuple[str, str], _DropCause] = {
    ("word", "Listening"): _DropCause(None, None, "recording"),
    ("word", "Production"): _DropCause("ProductiveTarget", "gated: no productive Target", "recording"),
    ("word", "Reading"): _DropCause(None, None, "recording"),
    ("word", "Spelling"): _DropCause("TestSpelling", "gated: spelling not tested", "recording"),
    ("sentence", "Cloze"): _DropCause("Productive", "gated: no productive Target", "recording"),
    ("sentence", "Listening"): _DropCause(None, None, "recording"),
}


def _template_drop_reason(model_name: str, template_name: str,
                          fields_by_name: Mapping[str, str]) -> str:
    """Looks up `(model_name, template_name)` in _TEMPLATE_DROP_CAUSES --
    KeyError (not a generic reason) when a template has no registered
    cause, so a renamed template fails loudly instead of being
    misreported as a missing artifact.
    """
    cause = _TEMPLATE_DROP_CAUSES[(model_name, template_name)]
    if cause.gate_field is not None and not fields_by_name[cause.gate_field]:
        return cause.gate_reason
    return f"no current-best {cause.artifact_kind}"


def _dropped_for(note: genanki.Note, model: genanki.Model, family: str,
                 subject: str) -> list[DroppedCard]:
    """One DroppedCard per template the note didn't generate a card for,
    reason from _template_drop_reason.
    """
    present_ords = {c.ord for c in note.cards}
    fields_by_name = dict(zip((f["name"] for f in model.fields), note.fields))
    return [DroppedCard(family=family, kind=tpl["name"], subject=subject,
                        reason=_template_drop_reason(model.name, tpl["name"], fields_by_name))
           for ord_, tpl in enumerate(model.templates) if ord_ not in present_ords]


def _stamp_due(apkg_path: Path, due_by_guid_ord: dict[tuple[str, int], int]) -> None:
    """genanki writes one `due` per note onto every one of its sibling
    cards; this reopens the written .apkg's collection.anki2 and
    overwrites `cards.due` directly, keyed by (note guid, card ord), so
    sibling cards land at distinct, stride-separated due values.
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


def _blocking_findings(findings: tuple[Finding, ...], syllabus: "Syllabus") -> list[Finding]:
    """Unwaived error-severity findings among `findings` -- the ones that
    close the gate (Syllabus.report()'s own filter, reused here so
    GateRefusal counts exactly what refused the compile).
    """
    return [f for f in findings if syllabus._severity(f.rule) == "error"
           and not syllabus.assessments.is_waived(f)]


@dataclass(frozen=True)
class Built:
    """One compiled note ready for the deck. `base_due` is the due value
    for card ord 0; sibling cards land at base_due + card.ord. `model`
    and `subject` are used to record this note's card fronts for
    card/unique-front.
    """
    note: genanki.Note
    base_due: int
    family: str
    subject: str
    model: genanki.Model


def _gated_items(built: tuple[genanki.Note, int] | None, model: genanki.Model,
                 family: str, subject: str) -> Iterator[Built | DroppedCard]:
    """DroppedCards for `built`'s un-produced templates, then its Built
    record if any card survived; nothing when `built` is None.
    """
    if built is None:
        return
    note, base_due = built
    yield from _dropped_for(note, model, family, subject)
    if note.cards:
        yield Built(note, base_due, family, subject, model)


def _word_items(syllabus: "Syllabus", resolver: _Resolver, compile_id: str,
                positions: _Positions) -> Iterator[Built | DroppedCard]:
    targeted_word_ids = {t.word for t in syllabus.targets}
    for word in syllabus.words:
        if word.id not in targeted_word_ids:
            continue
        built = _word_note(syllabus, word, resolver, compile_id, positions)
        yield from _gated_items(built, WORD_MODEL, "word", word.id)


def _pair_items(syllabus: "Syllabus", resolver: _Resolver, compile_id: str,
                positions: _Positions) -> Iterator[Built | DroppedCard]:
    for pair in syllabus.pairs:
        if pair.id not in positions.entry_index:
            continue
        recordings = syllabus.media.rendition(pair.id)
        if recordings is None:
            yield DroppedCard(family="minimal_pair", kind="Recognition",
                              subject=pair.id, reason="no rendition")
            continue
        for note, base_due in _pair_notes(pair, syllabus, recordings, resolver,
                                          compile_id, positions):
            yield Built(note, base_due, "minimal_pair", note.fields[0], MINIMAL_PAIR_MODEL)


def _grapheme_items(syllabus: "Syllabus", resolver: _Resolver, compile_id: str,
                    positions: _Positions) -> Iterator[Built | DroppedCard]:
    for grapheme in syllabus.graphemes:
        built = _grapheme_note(grapheme, syllabus, resolver, compile_id, positions)
        if built.dropped_reason is not None:
            yield DroppedCard(family="grapheme", kind="Reading",
                              subject=grapheme.symbol, reason=built.dropped_reason)
            continue
        if built.note is None:
            continue
        note, base_due = built.note, built.due
        if not note.cards:
            yield DroppedCard(family="grapheme", kind="Reading", subject=grapheme.symbol,
                              reason="Symbol field unexpectedly empty")
            continue
        yield Built(note, base_due, "grapheme", grapheme.symbol, GRAPHEME_MODEL)


def _sentence_items(syllabus: "Syllabus", resolver: _Resolver,
                    compile_id: str, positions: _Positions) -> Iterator[Built | DroppedCard]:
    for sentence, target, due_block in positions.sentence_entries:
        built = _sentence_note(sentence, target, due_block, syllabus, resolver, compile_id)
        subject = f"{target.id}:{sentence_note_id(sentence)}"
        yield from _gated_items(built, SENTENCE_MODEL, "sentence", subject)


@dataclass(frozen=True)
class BuiltDeck:
    """compile_syllabus's pre-write stage: every Built note (family/pair/
    grapheme/sentence, chained), the drop list, the media files their
    fronts/backs reference (basename -> on-disk path, genanki.Package's
    media_files shape), and the card/unique-front findings computed over
    the compiled notes themselves. compile_syllabus writes this to an
    .apkg; the review screen (spec 5 section 1) renders it directly, one
    front/back per note.cards entry, so the gallery shows exactly the
    notes a real compile would write.
    """
    built: tuple[Built, ...]
    dropped: tuple[DroppedCard, ...]
    front_findings: tuple[Finding, ...]
    media_files: Mapping[str, Path]
    warnings: tuple[str, ...]


def build_deck(syllabus: "Syllabus", db: "SyllabusDb", media_store: "MediaStore", *,
               compile_id: str | None = None) -> BuiltDeck:
    """Resolves media, positions due blocks, and builds one Built record
    per note that produced at least one card, in the family order
    compile_syllabus writes them (word, pair, grapheme, sentence).
    `compile_id` stamps every note's CompileId field (spec 4 section 2);
    omitted (the review screen's use, spec 5 section 1, which never
    writes an .apkg), it is the syllabus state id alone -- CompileId is
    a service field rendered by no template, so its exact value never
    reaches a rendered card.
    """
    compile_id = compile_id if compile_id is not None else syllabus.state_id()
    resolver = _Resolver(db=db, media_store=media_store)
    positions = _positions(syllabus)

    dropped: list[DroppedCard] = []
    built: list[Built] = []
    front_entries: list[tuple[str, str, str]] = []

    family_items = chain(
        _word_items(syllabus, resolver, compile_id, positions),
        _pair_items(syllabus, resolver, compile_id, positions),
        _grapheme_items(syllabus, resolver, compile_id, positions),
        _sentence_items(syllabus, resolver, compile_id, positions))
    for item in family_items:
        if isinstance(item, DroppedCard):
            dropped.append(item)
            continue
        _record_fronts(front_entries, item.model, item.subject, item.note)
        built.append(item)

    unique_front_rule = next((r for r in syllabus.rules if r.id == "card/unique-front"), None)
    front_findings = tuple(_duplicate_front_findings(front_entries)) if unique_front_rule else ()

    return BuiltDeck(built=tuple(built), dropped=tuple(dropped), front_findings=front_findings,
                     media_files=dict(resolver.used), warnings=tuple(resolver.warnings))


def compile_syllabus(syllabus: "Syllabus", db: "SyllabusDb", media_store: "MediaStore",
                     out_path: str | Path, *, force: bool = False,
                     now: Callable[[], float] = time.time) -> Compile:
    report = syllabus.report()
    warnings: list[str] = []
    if not report.gate:
        blocking = _blocking_findings(report.findings, syllabus)
        if not force:
            raise GateRefusal(report, blocking)
        for f in blocking:
            warnings.append(f"{f.rule}: {f.evidence} (note {f.note_id})")

    out_path = Path(out_path)
    state_id = syllabus.state_id()
    ts = int(now() * 1000)
    compile_id = f"{state_id}:{ts}"

    built_deck = build_deck(syllabus, db, media_store, compile_id=compile_id)

    blocking_front_findings = _blocking_findings(built_deck.front_findings, syllabus)
    if blocking_front_findings and not force:
        raise GateRefusal(replace(report, gate=False,
                                  findings=report.findings + built_deck.front_findings),
                          blocking_front_findings)
    for f in blocking_front_findings:
        warnings.append(f"{f.rule}: {f.evidence} (note {f.note_id})")
    gate = report.gate and not blocking_front_findings

    warnings.extend(built_deck.warnings)

    deck_name = out_path.stem
    deck = genanki.Deck(_deck_id(deck_name), deck_name)
    due_by_guid_ord: dict[tuple[str, int], int] = {}
    notes_written = 0
    cards_written = 0
    for item in built_deck.built:
        deck.add_note(item.note)
        for c in item.note.cards:
            due_by_guid_ord[(item.note.guid, c.ord)] = item.base_due + c.ord
        notes_written += 1
        cards_written += len(item.note.cards)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    media_paths = [str(p) for p in built_deck.media_files.values()]
    genanki.Package(deck, media_files=media_paths).write_to_file(
        str(tmp_out), timestamp=now())
    _stamp_due(tmp_out, due_by_guid_ord)
    os.replace(tmp_out, out_path)  # atomic

    compile_report = CompileReport(
        compile_id=compile_id, gate=gate, forced=force,
        warnings=tuple(warnings), notes_written=notes_written,
        cards_written=cards_written, dropped=built_deck.dropped,
        out_path=str(out_path), findings=built_deck.front_findings)
    return Compile(label=deck_name, syllabus_state_id=state_id,
                   compile_id=compile_id, report=compile_report)
