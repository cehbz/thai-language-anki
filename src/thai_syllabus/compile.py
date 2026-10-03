"""compile_syllabus (spec 4): a Syllabus, a SyllabusDb (current-best
artifacts and media provenance) and a MediaStore into one Anki .apkg.

One note per picture-introduced word (a spelling group's later members
only when productive: the group's first such Word carries its
Listening, Reading and Spelling cards), grapheme, and adopted sentence
that fills a target (its Listening card and a Cloze card per productive
Target it fills), one per minimal-pair member; every note tagged,
due-stamped from Syllabus.order(), and stamped with this compile's
CompileId.

It raises GateRefusal when Syllabus.report().gate is False, or when the
compiled notes duplicate a card front (rule card/unique-front), unless
`force=True`, which stamps those findings into CompileReport.warnings.
Every card a template did not produce is counted with a reason
(CompileReport.dropped); the returned Compile carries the compile id,
gate/forced status and the note/card counts.
"""
from __future__ import annotations

import hashlib
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
from .entities import Grapheme, MinimalPair, Sentence, Target, Word, render
from .ids import WordId, sentence_cloze_key
from .rulebook import _picture_introduced_words, sentence_note_id
from .rules import Compile, CompileReport, DroppedCard, Finding, OrderEntry, Report
from .syllabus import Syllabus

if TYPE_CHECKING:
    from .store import MediaStore, SyllabusDb

__all__ = ["BuiltDeck", "build_deck", "card_kind_of", "CARD_CSS", "CARD_MEANINGS",
          "card_meaning", "CLOZE_SLOTS", "cloze_target_field", "compile_syllabus",
          "field_values", "GateRefusal", "render_card", "tag_value", "template_kind",
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
.target .label { font-size: 0.6em; color: #888; margin-right: 0.4em; text-transform: uppercase; }
.gloss, .grammar, .classifier { font-size: 20px; color: #555; }
.nightMode .card { color: #ddd; background: #2f2f31; }
.nightMode .ipa, .nightMode .other { color: #999; }
"""


def _stable_id(*parts: str) -> int:
    """The genanki id for a model or deck: the first 8 hex of sha256 over
    the parts, joined "::". The same name always yields the same id.
    """
    return int(hashlib.sha256("::".join(parts).encode()).hexdigest()[:8], 16)


def _model(name: str, fields: list[str], templates: list[dict],
           appended: tuple[str, ...] = ()) -> genanki.Model:
    """`appended` are fields a later revision added, after the service
    fields, so every earlier field keeps its ord."""
    all_fields = [*fields, "ReviewNote", "CompileId", *appended]
    return genanki.Model(_stable_id(name), name,
                         fields=[{"name": f} for f in all_fields],
                         templates=templates, css=CARD_CSS)


# The word note (spec 4 r12). Listening, Reading and Spelling are the
# spelling's form-side cards: their fronts nest in FormSide, set on the
# note of the spelling group's first picture-introduced Word only. The
# Listening and Reading backs follow the note's own picture and meaning
# with OtherSenses, the group's other members' (empty for a Word alone
# in its form).
WORD_MODEL = _model(
    "word",
    ["Thai", "Meaning", "Picture", "Audio", "Ipa", "Classifier", "FrontGloss",
     "TestSpelling", "ProductiveTarget"],
    [{
        "name": "Listening",
        "qfmt": "{{#FormSide}}{{Audio}}{{/FormSide}}",
        "afmt": '{{FrontSide}}<hr id="answer">{{Picture}}'
               '<div class="thai">{{Thai}}</div><div class="ipa">{{Ipa}}</div>'
               '<div class="gloss">{{Meaning}}</div>{{OtherSenses}}',
    }, {
        # Picture nests inside its own section, not just ProductiveTarget's
        # (spec 4 section 1: "productive Target and a current-best
        # picture; no picture, no card"): genanki's required-field
        # computation (genanki.Model._req) treats a field as required when
        # blanking it alone empties the rendered qfmt, so nesting Picture
        # makes genanki compute this template's own requirement as "all
        # of ProductiveTarget, Picture" -- the card-presence rule is
        # explicit in the template, not left to genanki merely noticing
        # ProductiveTarget on its own (which is all the un-nested section
        # gave it, and how a Production card with an empty Picture front
        # got compiled).
        "name": "Production",
        "qfmt": '{{#ProductiveTarget}}{{#Picture}}{{Picture}}'
               '{{#FrontGloss}}<div class="gloss">{{FrontGloss}}</div>{{/FrontGloss}}'
               '{{/Picture}}{{/ProductiveTarget}}',
        "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{Thai}}</div>'
               '{{Audio}}<div class="ipa">{{Ipa}}</div>',
    }, {
        "name": "Reading",
        "qfmt": '{{#FormSide}}<div class="thai">{{Thai}}</div>{{/FormSide}}',
        "afmt": '{{FrontSide}}<hr id="answer">{{Picture}}{{Audio}}'
               '<div class="gloss">{{Meaning}}</div>{{OtherSenses}}',
    }, {
        "name": "Spelling",
        "qfmt": "{{#FormSide}}{{#TestSpelling}}{{Audio}}{{/TestSpelling}}{{/FormSide}}",
        "afmt": '{{#TestSpelling}}{{FrontSide}}<hr id="answer">'
               '<div class="thai">{{Thai}}</div>{{/TestSpelling}}',
    }],
    appended=("OtherSenses", "FormSide"))

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

# A sentence note's Cloze card slots (spec 4 r11): slot k, card ord k,
# holds the Cloze card of the productive Target on the sentence's k-th
# distinct word (clause order, first occurrence). The live deck's longest
# adopted sentence uses 10 distinct words.
CLOZE_SLOTS = 10


def _cloze_fields(slot: int) -> tuple[str, str, str]:
    """Slot `slot`'s fields: the sentence with its word blanked and that
    word (both empty when the sentence does not fill the slot's Target),
    and the id of the slot word's productive Target, filled or not."""
    return f"Cloze{slot}", f"ClozeWord{slot}", f"ClozeTarget{slot}"


def cloze_target_field(ord_: int) -> str:
    """The sentence-note field naming the Target of the Cloze card at
    card ord `ord_` (its slot)."""
    return _cloze_fields(ord_)[2]


def _cloze_template(slot: int) -> dict[str, str]:
    """Slot `slot`'s Cloze card. The front nests in the sections of its
    Cloze field, ScenePicture and Audio, so genanki computes all three as
    required: no Target in the slot, no picture or no recording, no card
    (spec 4 r10, r11)."""
    cloze, word, _target = _cloze_fields(slot)
    return {
        "name": f"Cloze {slot}",
        "qfmt": f'{{{{#{cloze}}}}}{{{{#ScenePicture}}}}{{{{#Audio}}}}'
                f'<div class="cloze">{{{{{cloze}}}}}</div>{{{{ScenePicture}}}}'
                f'{{{{/Audio}}}}{{{{/ScenePicture}}}}{{{{/{cloze}}}}}',
        "afmt": f'{{{{FrontSide}}}}<hr id="answer"><div class="target">{{{{{word}}}}}</div>'
                '{{Audio}}{{#Gloss}}<div class="gloss">{{Gloss}}</div>{{/Gloss}}',
    }


# The sentence note: its Listening card, then one Cloze card per slot.
# TargetWord is the sentence's target words (Syllabus.target_words),
# joined.
SENTENCE_MODEL = _model(
    "sentence",
    ["Thai", "TargetWord", "Audio", "Gloss", "ScenePicture",
     *(f for slot in range(1, CLOZE_SLOTS + 1) for f in _cloze_fields(slot))],
    [{
        "name": "Listening",
        "qfmt": "{{Audio}}",
        "afmt": '{{FrontSide}}<hr id="answer"><div class="thai">{{Thai}}</div>'
               '<div class="target"><span class="label">target words</span> {{TargetWord}}</div>'
               '{{#Gloss}}<div class="gloss">{{Gloss}}</div>{{/Gloss}}',
    }, *(_cloze_template(slot) for slot in range(1, CLOZE_SLOTS + 1))])

# Spec 5 r9 (design ruling 4): one line per card type -- what the front
# asks, what the back shows -- keyed by the family and kind /api/cards
# reports (card_kind_of over the template name). The page shows it as
# the card type's tooltip and the comment pass hands it to the reader
# with the comment; both read this one table.
CARD_MEANINGS: dict[tuple[str, str], str] = {
    ("word", "listening"): "Front plays the word; back shows its picture, Thai, IPA and meaning, then any other meaning of that spelling with its picture.",
    ("word", "production"): "Front shows the picture (and a gloss when set); back shows the Thai, plays it and gives the IPA.",
    ("word", "reading"): "Front shows the Thai; back shows the picture, plays the word and gives the meaning, then any other meaning of that spelling with its picture.",
    ("word", "spelling"): "Front plays the word; back shows the Thai spelling.",
    ("minimal_pair", "recognition"): "Front plays one member of a minimal pair and offers both; back names the one heard, with IPA, and plays the other.",
    ("grapheme", "reading"): "Front shows the letter; back shows its recited name, the keyword picture, the keyword's Thai and gloss, plays it and gives the sound.",
    ("sentence", "cloze"): "Front shows the sentence with the target word blanked, plus the scene picture; back shows the target word, plays the sentence and gives the gloss.",
    ("sentence", "listening"): "Front plays the sentence; back shows its Thai, names its target words and gives the gloss.",
}


def card_meaning(family: str, kind: str) -> str | None:
    return CARD_MEANINGS.get((family, kind))


STRIDE = 100  # due-per-order-position block size, in which a note's
             # cards are due at base + ord; above the most templates any
             # one note has (word: 4; sentence: 1 + CLOZE_SLOTS).


def _guid(family: str, *parts: str) -> str:
    return genanki.guid_for(family, *parts)


def thai_cloze(sentence: Sentence, target: WordId, thai_of: Callable[[WordId], str],
               blank: str = "___") -> str:
    """`sentence`'s rendering (entities.render) with every element whose
    word is `target` shown as `blank` -- a repeated element's mark stays
    outside the blank, render's own doing. Blanking reads element word
    identity alone, never a substring test over the rendered text: the
    ยา/โรงพยาบาล corruption class (blanking "medicine" inside
    "hospital") cannot arise from elements (spec 4 section 1).
    """
    def blank_or_thai(word: WordId) -> str:
        return blank if word == target else thai_of(word)

    return render(sentence.clauses, blank_or_thai)


# --- media resolution ------------------------------------------------------

# How a staged artifact is referenced from a note field, by media kind.
_SOUND_TAG = "[sound:{sha}.{ext}]"
_IMG_TAG = '<img src="{sha}.{ext}">'


@dataclass
class _Resolver:
    """Resolves (subject, kind) to the current-best artifact, staged
    under its content-sha basename. `current_rubric`/`prior`/
    `provenance_source` are the same parameters derivations.current_best
    ranks under everywhere else (wiring.Derivations); a stale-rubric
    verdict is not current-best for compile, matching the rulebook.
    `used` maps each referenced basename to its on-disk path
    (genanki.Package's media_files); `warnings` collects the non-fatal
    notes CompileReport.warnings carries.
    """
    db: "SyllabusDb"
    media_store: "MediaStore"
    current_rubric: Mapping[str, str]
    prior: Sequence[str]
    provenance_source: Callable[[str], str | None]
    used: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def artifact(self, subject: str, kind: str) -> tuple[str, str] | None:
        best = current_best(self.db, subject, kind, current_rubric=self.current_rubric,
                            prior=self.prior, provenance_source=self.provenance_source)
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

    @staticmethod
    def _tag(staged: tuple[str, str] | None, template: str) -> str:
        """`template` filled with a staged artifact's sha and ext; "" when
        nothing was staged.
        """
        return template.format(sha=staged[0], ext=staged[1]) if staged else ""

    def sound(self, subject: str, kind: str) -> str:
        return self._tag(self.artifact(subject, kind), _SOUND_TAG)

    def img(self, subject: str, kind: str) -> str:
        return self._tag(self.artifact(subject, kind), _IMG_TAG)

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
        return self._tag(self._stage(sha, prov["ext"], f"pair={pair_id!r}"), _SOUND_TAG)

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
    """Where each order() entry's due block starts, in STRIDE units --
    an adopted sentence's own block included: order() deals it among
    the word targets (r24), so it shares their one block sequence and
    is due at its own order() position. A block is `width` units wide
    (a pair: len(members); everything else: 1); blocks never overlap.
    """
    entry_index: dict[str, int]           # grapheme symbol / pair id -> block start
    target_index: dict[str, int]          # target id -> block start
    word_index: dict[str, int]            # word id -> min block start of its targets
    # one entry per adopted sentence: (sentence, filled targets in target-id
    # order, due block index)
    sentence_entries: list[tuple[Sentence, tuple[Target, ...], int]]
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

    # One entry per adopted sentence with at least one filled target (the
    # note it compiles into). Its due block is its own order() position
    # (r24: order() already deals it directly after its last used word's
    # last Target, its own block of width 1) -- not a block after every
    # word. A sentence with no order() position (should not arise: order()
    # covers every syllabus.sentences entry, but this stays a defensive
    # fallback) is due after every other block, such sentences sorted by
    # id for a deterministic (if arbitrary) relative order among them.
    positioned: list[tuple[Sentence, tuple[Target, ...], int]] = []
    unpositioned: list[tuple[Sentence, tuple[Target, ...]]] = []
    for s in syllabus.sentences:
        filled = syllabus.fill_set(s)   # already sorted by target id
        if not filled:
            continue
        position = sentence_position.get(sentence_note_id(s))
        if position is None:
            unpositioned.append((s, filled))
        else:
            positioned.append((s, filled, position))
    unpositioned.sort(key=lambda sf: sentence_note_id(sf[0]))
    sentence_entries = positioned + [
        (s, filled, total_blocks + i) for i, (s, filled) in enumerate(unpositioned)]

    return _Positions(entry_index=entry_index, target_index=target_index,
                      word_index=word_index, sentence_entries=sentence_entries,
                      order_length=len(order_list))


# --- note builders -----------------------------------------------------
# One builder per family, called from that family's *_items generator
# below; each returns (genanki.Note, due) or None for an item with
# nothing to compile.

def _other_senses(word: Word, group: Sequence[Word], resolver: _Resolver) -> str:
    """The OtherSenses field (spec 4 r12): each other member of `word`'s
    spelling group, in introduction order, as its current-best picture
    (none when it has none) and its meaning; empty for a group of one."""
    return "".join(f'{resolver.img(w.id, "picture")}<div class="gloss">{w.meaning}</div>'
                   for w in group if w.id != word.id)


def _word_note(syllabus: "Syllabus", word: Word, carries_form_side: bool,
               productive_words: frozenset[WordId], resolver: _Resolver,
               compile_id: str, positions: _Positions) -> tuple[genanki.Note, int] | None:
    """`word`'s note; `carries_form_side` when it is its spelling group's
    first picture-introduced Word, whose note holds the group's
    Listening, Reading and Spelling cards (spec 4 r12)."""
    if word.id not in positions.word_index:
        return None  # no Target at all -- not a compiled word (spec 4 section 1)

    group = syllabus.spelling_group(word.id)
    productive = word.id in productive_words
    spelling_tested = any(w.id in productive_words for w in group)
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
        "1" if spelling_tested else "",   # TestSpelling: any member productive
        "1" if productive else "",   # ProductiveTarget
        "",  # ReviewNote: mid-review comment channel, rendered by no template
        compile_id,
        _other_senses(word, group, resolver),
        "1" if carries_form_side else "",   # FormSide
    ]
    note = genanki.Note(model=WORD_MODEL, fields=fields, tags=tags,
                        guid=_guid("word", word.id))
    due = positions.word_index[word.id] * STRIDE
    return note, due


def _pair_notes(pair: MinimalPair, syllabus: "Syllabus", recordings: tuple,
                resolver: _Resolver, compile_id: str,
                positions: _Positions) -> list[tuple[genanki.Note, int]]:
    """One note per member of `pair`, all playing `recordings` (the
    pair's current-best rendition, in member order). `Choices` lists
    every member in that same order on every note, so choice position
    never gives away the stimulus. Member notes sit one STRIDE apart.
    """
    base_due = positions.entry_index[pair.id] * STRIDE
    members = [syllabus.find_word(m) for m in pair.members]
    if any(m is None for m in members):
        return []  # the loader's registration check already enforces this (spec 1 section 4 r10)

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
    """Either a built note or the reason its card was dropped: exactly
    one of `note`/`due` and `dropped_reason` is set. Both are None for a
    grapheme that is not compiled and not counted.
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
        # the loader's registration check already enforces this (spec 1 section 4 r10)
        return _GraphemeBuild(None, None, None)
    name_word = syllabus.find_word(grapheme.name_word) if grapheme.name_word else None
    if name_word is None:
        return _GraphemeBuild(None, None, "no name word")

    # NameThai is the name word's own text (กอ ไก่ "gɔɔ gài", the recited
    # name of the letter ก, "k"): one Word whose recording says the whole
    # name, with no substitute audio (spec 4 section 1).
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


def _cloze_slots(sentence: Sentence, productive: tuple[Target, ...]
                 ) -> tuple[dict[int, Target], list[tuple[Target, str]]]:
    """(slot -> Target, dropped (Target, reason)) for `sentence`'s
    productive fills (spec 4 r11): a Target takes the slot of its word's
    position among the sentence's distinct words; one beyond the last
    slot, or on a slot another Target already holds, is dropped.
    """
    slots: dict[int, Target] = {}
    dropped: list[tuple[Target, str]] = []
    for target in productive:
        slot = sentence.words.index(target.word) + 1
        if slot > CLOZE_SLOTS:
            dropped.append((target, f"beyond the last Cloze slot ({CLOZE_SLOTS})"))
        elif slot in slots:
            dropped.append((target, f"Cloze slot {slot} held by {slots[slot].id}"))
        else:
            slots[slot] = target
    return slots, dropped


def _sentence_note(sentence: Sentence, targets: tuple[Target, ...], slots: Mapping[int, Target],
                   productive_of: Mapping[WordId, Target], due_block: int,
                   syllabus: "Syllabus", resolver: _Resolver,
                   compile_id: str) -> tuple[genanki.Note, int]:
    """The sentence note, `targets` the ones it fills (target-id order),
    `slots` its filled Cloze slots' Targets, `productive_of` each word's
    productive Target (named in its slot whether filled or not), due at
    the start of its order() block.
    """
    text_sha = sentence_note_id(sentence)
    target_words = ", ".join(syllabus.word(w).thai for w in syllabus.target_words(sentence))

    # A sentence's audio/picture are resolved by (text_sha, kind), the
    # same artifact kinds a word's audio and picture carry.
    tags = ["family::sentence"]
    tags += [f"target::{t.id}" for t in targets]
    tags += [f"sentence::{text_sha}", f"compile::{compile_id}", "kind::listening", "kind::cloze"]
    tags += resolver.src_tag("audio", text_sha, "recording")
    tags += resolver.src_tag("img", text_sha, "picture")

    words = sentence.words
    slot_fields: list[str] = []
    for slot in range(1, CLOZE_SLOTS + 1):
        target = slots.get(slot)
        named = productive_of.get(words[slot - 1]) if slot <= len(words) else None
        if target is not None:
            slot_fields += [thai_cloze(sentence, target.word, lambda w: syllabus.word(w).thai),
                            syllabus.word(target.word).thai, target.id]
        else:
            slot_fields += ["", "", named.id if named is not None else ""]
    fields = [
        sentence.text,
        target_words,
        resolver.sound(text_sha, "recording"),
        sentence.gloss,
        resolver.img(text_sha, "picture"),
        *slot_fields,
        "",  # ReviewNote
        compile_id,
    ]
    note = genanki.Note(model=SENTENCE_MODEL, fields=fields, tags=tags,
                        guid=_guid("sentence", text_sha))
    return note, due_block * STRIDE


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


def field_values(model: genanki.Model, note: genanki.Note) -> dict[str, str]:
    """note.fields as a name -> value mapping, in `model`'s own field
    order -- the lookup render_card, card/unique-front's front recording,
    and dropped-card reasoning all key their template substitution on.
    """
    return dict(zip((f["name"] for f in model.fields), note.fields))


_SLOT_SUFFIX_RE = re.compile(r" \d+$")


def template_kind(template_name: str) -> str:
    """A template's card kind as named: its name without a Cloze slot's
    number ("Cloze 3" -> "Cloze")."""
    return _SLOT_SUFFIX_RE.sub("", template_name)


def card_kind_of(template_name: str) -> str:
    """study.card_kind for a card (spec 4 section 2): its template's
    kind, lowered. The one place this conversion happens -- anki_import.py's
    revlog/flag import and reviewserver.py's gallery both read a card's
    kind through this function.
    """
    return template_kind(template_name).lower()


def tag_value(note: genanki.Note, prefix: str) -> str | None:
    """The value of the one atomic tag on `note` reading "prefix::value"
    (spec 4 section 2's tag convention), or None when it carries none --
    the same convention anki_import.py's return path reads tags by.
    """
    needle = f"{prefix}::"
    for t in note.tags:
        if t.startswith(needle):
            return t[len(needle):]
    return None


def _record_fronts(entries: list[tuple[str, str, str]], built: "Built") -> None:
    """Appends (model:kind, card subject, rendered front) for every card
    the note actually generated -- card/unique-front compares these
    within a (model, card kind) group, every Cloze slot in one.
    """
    model = built.model
    values = field_values(model, built.note)
    for card in built.note.cards:
        template = model.templates[card.ord]
        front = _render_qfmt(template["qfmt"], values)
        entries.append((f"{model.name}:{card_kind_of(template['name'])}",
                        built.subject_of(card.ord), front))


def render_card(model: genanki.Model, note: genanki.Note, ord_: int) -> tuple[str, str]:
    """Front and back HTML for one card (`ord_` into `model.templates`)
    of `note`, through the same mustache subset card/unique-front uses,
    extended to afmt and Anki's {{FrontSide}}. This is what the review
    screen renders (spec 5 section 1): the model's own qfmt/afmt.
    """
    values = field_values(model, note)
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
    """What a (model, template) pair's card generation depends on: its
    `gates`, (field, reason) in order, the first with an empty field
    meaning the card was not asked for (with that reason; None counts no
    drop at all), else the artifacts, (field, artifact kind), whose
    missing current-best leaves the card no front.
    """
    gates: tuple[tuple[str, str | None], ...]
    artifacts: tuple[tuple[str, str], ...]


# A form-side card on a note that does not carry its spelling's form side
# is no card and no drop: the group's first Word carries it (spec 4 r12).
_FORM_SIDE_GATE = ("FormSide", None)

# One entry per (model name, template name) for word and sentence, whose
# card presence genanki's own required-field computation decides once the
# note is built; grapheme and minimal_pair decide their drop reason
# before building theirs.
_TEMPLATE_DROP_CAUSES: dict[tuple[str, str], _DropCause] = {
    ("word", "Listening"): _DropCause((_FORM_SIDE_GATE,), (("Audio", "recording"),)),
    # gated on ProductiveTarget (dropped for "gated: ..." when the word
    # isn't productive); when it IS productive but the card still didn't
    # generate, the front's other requirement -- a current-best picture
    # (spec 4 section 1/3) -- is what's missing.
    ("word", "Production"): _DropCause((("ProductiveTarget", "gated: no productive Target"),),
                                       (("Picture", "picture"),)),
    ("word", "Reading"): _DropCause((_FORM_SIDE_GATE,), (("Audio", "recording"),)),
    ("word", "Spelling"): _DropCause(
        (_FORM_SIDE_GATE, ("TestSpelling", "gated: spelling not tested")),
        (("Audio", "recording"),)),
    ("sentence", "Listening"): _DropCause((), (("Audio", "recording"),)),
    # A slot holding no Target is no card and no drop (spec 4 r11).
    **{("sentence", f"Cloze {slot}"): _DropCause(
        ((_cloze_fields(slot)[0], None),), (("ScenePicture", "picture"), ("Audio", "recording")))
       for slot in range(1, CLOZE_SLOTS + 1)},
}


def _template_drop_reason(model_name: str, template_name: str,
                          fields_by_name: Mapping[str, str]) -> str | None:
    """The drop reason `(model_name, template_name)` registers in
    _TEMPLATE_DROP_CAUSES, naming the missing artifacts (every listed one
    when none is empty); None when its gate counts no drop. A template
    with no entry raises KeyError.
    """
    cause = _TEMPLATE_DROP_CAUSES[(model_name, template_name)]
    for gate_field, gate_reason in cause.gates:
        if not fields_by_name[gate_field]:
            return gate_reason
    missing = [kind for name, kind in cause.artifacts if not fields_by_name[name]]
    return "no current-best " + " and ".join(missing or [kind for _, kind in cause.artifacts])


def _dropped_for(note: genanki.Note, model: genanki.Model, family: str,
                 subject_of: Callable[[int], str]) -> list[DroppedCard]:
    """One DroppedCard per template the note didn't generate a card for
    whose drop is counted, reason from _template_drop_reason, subject the
    card's own.
    """
    present_ords = {c.ord for c in note.cards}
    fields_by_name = field_values(model, note)
    dropped = []
    for ord_, tpl in enumerate(model.templates):
        if ord_ in present_ords:
            continue
        reason = _template_drop_reason(model.name, tpl["name"], fields_by_name)
        if reason is not None:
            dropped.append(DroppedCard(family=family, kind=template_kind(tpl["name"]),
                                       subject=subject_of(ord_), reason=reason))
    return dropped


def _stamp_due(apkg_path: Path, due_by_guid_ord: dict[tuple[str, int], int]) -> None:
    """Reopens the written .apkg's collection.anki2 and sets `cards.due`
    per (note guid, card ord), so sibling cards land at distinct,
    stride-separated due values.
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
    """One compiled note ready for the deck. `base_due` is card ord 0's
    due value; siblings land at base_due + card.ord. `subject` is the
    note's anchor, `card_subjects` a card's own where it differs (a
    sentence's Cloze card: its (sentence, Target)); `model` and the card
    subjects record its fronts for card/unique-front.
    """
    note: genanki.Note
    base_due: int
    family: str
    subject: str
    model: genanki.Model
    card_subjects: Mapping[int, str] = field(default_factory=dict)

    def subject_of(self, ord_: int) -> str:
        return self.card_subjects.get(ord_, self.subject)


def _gated_items(built: tuple[genanki.Note, int] | None, model: genanki.Model,
                 family: str, subject: str,
                 card_subjects: Mapping[int, str] | None = None) -> Iterator[Built | DroppedCard]:
    """DroppedCards for `built`'s un-produced templates, then its Built
    record if any card survived; nothing when `built` is None.
    """
    if built is None:
        return
    note, base_due = built
    item = Built(note, base_due, family, subject, model, dict(card_subjects or {}))
    yield from _dropped_for(note, model, family, item.subject_of)
    if note.cards:
        yield item


def _word_items(syllabus: "Syllabus", resolver: _Resolver, compile_id: str,
                positions: _Positions) -> Iterator[Built | DroppedCard]:
    # A word note is compiled only for a word with a picture-introduced
    # Target (spec 4 section 1); a sentence-introduced word compiles no
    # word note, it is carried by its sentence note. The first
    # picture-introduced Word of a spelling group carries the group's
    # form-side cards; every other member's note holds its Production
    # card only, so a member with no productive Target has no note
    # (spec 4 r12).
    picture_introduced_word_ids = set(_picture_introduced_words(syllabus))
    productive_words = frozenset(t.word for t in syllabus.targets if t.skill == "productive")
    for word in syllabus.words:
        if word.id not in picture_introduced_word_ids:
            continue
        carrier = next(w for w in syllabus.spelling_group(word.id)
                       if w.id in picture_introduced_word_ids)
        carries_form_side = carrier.id == word.id
        if not carries_form_side and word.id not in productive_words:
            continue
        built = _word_note(syllabus, word, carries_form_side, productive_words, resolver,
                           compile_id, positions)
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
    """Per adopted sentence: its note at the start of its block, the
    Listening card and each Cloze card siblings due at base + ord; a
    Cloze card is subject (sentence, Target), and a productive fill with
    no slot, or a slotted card without the picture or recording, yields a
    DroppedCard.
    """
    productive_of: dict[WordId, Target] = {}
    for t in sorted(syllabus.targets, key=lambda t: t.id):
        if t.skill == "productive":
            productive_of.setdefault(t.word, t)
    for sentence, targets, due_block in positions.sentence_entries:
        text_sha = sentence_note_id(sentence)
        slots, unslotted = _cloze_slots(sentence, syllabus.productive_fills(sentence))
        for target, reason in unslotted:
            yield DroppedCard(family="sentence", kind="Cloze",
                              subject=sentence_cloze_key(text_sha, target.id), reason=reason)
        built = _sentence_note(sentence, targets, slots, productive_of, due_block, syllabus,
                               resolver, compile_id)
        yield from _gated_items(built, SENTENCE_MODEL, "sentence", text_sha,
                                {slot: sentence_cloze_key(text_sha, t.id)
                                 for slot, t in slots.items()})


@dataclass(frozen=True)
class BuiltDeck:
    """compile_syllabus's pre-write stage: every Built note, the drop
    list, the media files their fronts and backs reference (basename ->
    on-disk path, genanki.Package's media_files shape), and the
    card/unique-front findings over the compiled notes. compile_syllabus
    writes it to an .apkg; the review screen renders it directly.
    """
    built: tuple[Built, ...]
    dropped: tuple[DroppedCard, ...]
    front_findings: tuple[Finding, ...]
    media_files: Mapping[str, Path]
    warnings: tuple[str, ...]


def build_deck(syllabus: "Syllabus", db: "SyllabusDb", media_store: "MediaStore", *,
               current_rubric: Mapping[str, str], prior: Sequence[str],
               provenance_source: Callable[[str], str | None],
               compile_id: str | None = None) -> BuiltDeck:
    """Resolves media, positions due blocks, and builds one Built record
    per note that produced at least one card, in the family order
    compile_syllabus writes them (word, pair, grapheme, sentence).
    `compile_id` stamps every note's CompileId field (spec 4 section 2);
    omitted, it is the syllabus state id alone. `current_rubric`/`prior`/
    `provenance_source` are current_best's own parameters (wiring.
    Derivations carries the deck's real ones); build_deck resolves media
    exactly as the run and the review screen do.
    """
    compile_id = compile_id if compile_id is not None else syllabus.state_id()
    resolver = _Resolver(db=db, media_store=media_store, current_rubric=current_rubric,
                         prior=prior, provenance_source=provenance_source)
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
        _record_fronts(front_entries, item)
        built.append(item)

    unique_front_rule = next((r for r in syllabus.rules if r.id == "card/unique-front"), None)
    front_findings = tuple(_duplicate_front_findings(front_entries)) if unique_front_rule else ()

    return BuiltDeck(built=tuple(built), dropped=tuple(dropped), front_findings=front_findings,
                     media_files=dict(resolver.used), warnings=tuple(resolver.warnings))


def compile_syllabus(syllabus: "Syllabus", db: "SyllabusDb", media_store: "MediaStore",
                     out_path: str | Path, *, current_rubric: Mapping[str, str],
                     prior: Sequence[str], provenance_source: Callable[[str], str | None],
                     force: bool = False,
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

    built_deck = build_deck(syllabus, db, media_store, compile_id=compile_id,
                            current_rubric=current_rubric, prior=prior,
                            provenance_source=provenance_source)

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
    deck = genanki.Deck(_stable_id("thai-syllabus", deck_name), deck_name)
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
