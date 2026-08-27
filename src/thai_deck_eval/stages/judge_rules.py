from ..core.findings import Dimension, Severity, Stage
from ..core.registry import rule
from ..judge.core import JudgeRequest
from ..judge.prompts import (PICTURE_RULES, SENTENCE_RULES,
                             build_picture_prompt, build_sentence_prompt)

def _verdicts(ctx):
    if getattr(ctx, "_judge_verdicts", None) is not None:
        return ctx._judge_verdicts
    out: dict[str, list] = {}
    for note in ctx.deck.sentences:
        req = JudgeRequest(note_id=note.id, rules=list(SENTENCE_RULES),
                           prompt=build_sentence_prompt(note))
        out[note.id] = ctx.judge.judge(req)
    for note in ctx.deck.picture_words:
        req = JudgeRequest(note_id=note.id, rules=list(PICTURE_RULES),
                           prompt=build_picture_prompt(note),
                           image_path=str(ctx.deck.root / "media" / note.image))
        out[note.id] = ctx.judge.judge(req)
    ctx._judge_verdicts = out
    return out

def _findings_for(rule_fn, ctx, rule_id):
    if ctx.judge is None:
        return
    floor = ctx.config.judge.confidence_floor
    for note_id, verdicts in _verdicts(ctx).items():
        for v in verdicts:
            if v.rule == rule_id and not v.passed:
                sev = Severity.INFO if v.confidence < floor else None
                yield rule_fn.rule_def.finding(
                    v.rationale or "judge failed this rule", note_id=note_id,
                    severity=sev, evidence={"confidence": v.confidence})

def _make(rule_id, default_severity):
    @rule(rule_id, Stage.JUDGE, Dimension.CONTENT, default_severity)
    def fn(ctx, _rid=rule_id):
        yield from _findings_for(fn, ctx, _rid)
    return fn

unnatural = _make("judge/unnatural-sentence", Severity.ERROR)
definition = _make("judge/definition-not-monolingual", Severity.WARN)
gloss = _make("judge/gloss-inaccurate", Severity.WARN)
image_irrelevant = _make("judge/image-irrelevant", Severity.WARN)
image_text = _make("judge/image-embedded-text", Severity.WARN)
classifier = _make("judge/classifier-wrong", Severity.WARN)
