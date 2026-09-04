"""End to end over a fixture deck: build_sourcing + run with fake search, Forvo, LLM, judge and
mechanical backends; picture, recording and sentence needs close; the gate opens."""
import dataclasses
import json
from pathlib import Path

from thai_syllabus.assessor import RawVerdict
from thai_syllabus.attempts import current_best_of
from thai_syllabus.curated import CuratedBundle, RulebookConfig, save_curated
from thai_syllabus.entities import Category
from thai_syllabus.provider import FetchBackend, RawAnswer
from thai_syllabus.profile import Profile
from thai_syllabus.run import run
from thai_syllabus.store import MediaStore, SyllabusDb
from thai_syllabus.transport import Completion
from thai_syllabus.wiring import build_sourcing, load_syllabus

from .builders import target, word
from .fakes import FakeTokenizer


def _deck(tmp_path):
    root = tmp_path / "deck"
    save_curated(root / "curated", CuratedBundle(
        words=(word("orange", "ส้ม", "orange"), word("eat", "กิน", "eat")),
        targets=(target("eat/receptive", "eat"), target("orange/receptive", "orange")),
        graphemes=(), confusions=(), pairs=(), profile=Profile(register="male_colloquial"),
        rulebook=RulebookConfig(),
        categories=(Category(name="Food", members=frozenset({"orange"})),
                   Category(name="Verbs", members=frozenset({"eat"})))))
    # imgfetch/audiofetch paths and the anthropic secret reference are what
    # load_providers_config now requires of any real providers.yaml; the
    # test replaces both fetch backends (and the judge's transport) with
    # fakes below, so neither binary nor secret is ever touched.
    (root / "curated" / "providers.yaml").write_text(
        "imgfetch_path: /opt/bin/imgfetch\n"
        "audiofetch_path: /opt/bin/audiofetch\n"
        "secrets: {anthropic: op://Shared/Anthropic/API Key}\n"
        "judge: {transport: api, model: m, price_per_mtok: {input: 2.0, output: 10.0}}\n",
        encoding="utf-8")
    SyllabusDb(root / "syllabus.db").close()
    MediaStore(root / "media")
    return root


class _Search:
    """Two hits per query -- one image alone can never exercise the
    preference branch (_assess_all_candidates only asks a preference
    question over 2+ passing candidates)."""
    def cache_key(self, q):
        return "s:" + q.params["query"]

    def fetch(self, q):
        return RawAnswer(items=(
            {"url": f"https://x/{q.subject}-good.jpg", "source": "openverse", "licence": "by"},
            {"url": f"https://x/{q.subject}-good2.jpg", "source": "openverse", "licence": "by"}))


class _Forvo:
    def cache_key(self, q):
        return "forvo:" + q.params["word"]

    def fetch(self, q):
        return RawAnswer(items=({"pathmp3": f"https://f/{q.params['word']}.mp3", "username": "kris"},), cost=1.0)


class _Llm:
    def cache_key(self, q):
        return "llm:sentence-drafter:m:x"

    def fetch(self, q):
        return RawAnswer(items=('{"sentences": [{"text": "กินส้ม", "targets": ["orange/receptive", "eat/receptive"]}]}',))


def _judge_complete(prompt, attachments=()):
    """The judge's default prompt builders dispatch per role -- fit
    questions attach at most one artifact, a picture-preference question
    attaches every candidate -- so this fakes the same distinction by
    attachment count rather than sniffing prompt text.
    """
    if len(attachments) > 1:
        shas = [Path(p).stem for p in attachments]
        return Completion(text=json.dumps({"ranking": shas}))
    return Completion(text='{"value": true, "evidence": "ok"}')


def test_run_closes_picture_recording_and_sentence_needs(tmp_path):
    root = _deck(tmp_path)
    ctx = build_sourcing(root)
    ctx.provider._backends.update({
        "openverse": _Search(), "forvo": _Forvo(), "llm-sentence": _Llm(),
        "imgfetch": FetchBackend(media=ctx.media_store, fetcher=lambda url: (url.encode(), "jpg")),
        "audiofetch": FetchBackend(media=ctx.media_store, fetcher=lambda url: (url.encode(), "mp3"))})
    ctx.assessor._backends["judge"].complete = _judge_complete
    ctx.assessor._backends["mechanical"].evaluate = lambda q: RawVerdict(value=True, evidence="1.0s")
    ctx.syllabus = dataclasses.replace(ctx.syllabus, tokenizer=FakeTokenizer({"กินส้ม": ["กิน", "ส้ม"]}))

    before = load_syllabus(root).gaps()
    assert before.unfilled_targets and before.words_missing_pictures and before.words_missing_recordings

    report = run(ctx, {})
    assert report.improved >= 4 and report.pending == 0 and report.sentences_adopted == 1

    after = dataclasses.replace(load_syllabus(root), tokenizer=FakeTokenizer({"กินส้ม": ["กิน", "ส้ม"]}))
    g = after.gaps()
    assert g.words_missing_pictures == () and g.words_missing_recordings == () and g.unfilled_targets == ()
    rep = after.report()
    assert "recording/synthetic" not in {f.rule for f in rep.findings}
    assert rep.gate is True

    # two candidate pictures per word -> a preference question ran and its
    # positional bonus lifted the winner above a bare judge pass (50.0).
    assert current_best_of(ctx, "orange", "picture").rank > 50
