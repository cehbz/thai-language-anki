"""Manual, credential-requiring tests against the real judge backends.

Excluded from the default suite by `addopts = "-m 'not integration and not live'"`
in pyproject.toml. Run explicitly with: `uv run pytest -m live -v`.
"""
import pytest
from thai_deck_eval.config import JudgeConfig
from thai_deck_eval.judge.api_judge import ApiJudge
from thai_deck_eval.judge.cli_judge import CliJudge
from thai_deck_eval.judge.core import JudgeRequest
from thai_deck_eval.judge.prompts import SENTENCE_RULES, build_sentence_prompt
from thai_deck_eval.model.notes import Audio, SentenceNote

pytestmark = pytest.mark.live

GOLDEN_SENTENCE = SentenceNote(
    id="s-1", kind="new_word", thai="หมามากินข้าว", target="กิน",
    audio=Audio(file="audio/s1.mp3", source="native", speaker="s1"),
    definition="เอาอาหารเข้าปาก",
)

def _assert_all_rules_present(verdicts):
    assert {v.rule for v in verdicts} == set(SENTENCE_RULES)

def test_cli_judge_live():
    judge = CliJudge(JudgeConfig(backend="cli"))
    req = JudgeRequest(note_id=GOLDEN_SENTENCE.id, rules=list(SENTENCE_RULES),
                       prompt=build_sentence_prompt(GOLDEN_SENTENCE))
    _assert_all_rules_present(judge.judge(req))

def test_api_judge_live():
    judge = ApiJudge(JudgeConfig(backend="api"))
    req = JudgeRequest(note_id=GOLDEN_SENTENCE.id, rules=list(SENTENCE_RULES),
                       prompt=build_sentence_prompt(GOLDEN_SENTENCE))
    _assert_all_rules_present(judge.judge(req))


# --- the incidental-text rule, put to the real judge ---
#
# The rule is that text fails only when it reveals the answer. Read as a
# blanket text ban it discarded 590 relevant candidates against 570 accepted,
# so the discrimination below is the behaviour worth pinning, not the wording.

def _image_with_text(path, text):
    """A picture whose only text is `text`. Pillow-drawn, so no binary
    fixture and no licence question."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 480), (200, 170, 140))
    draw = ImageDraw.Draw(img)
    draw.ellipse((200, 140, 440, 340), fill=(120, 90, 60))
    draw.text((20, 450), text, fill=(255, 255, 255))
    img.save(path, "JPEG")
    return str(path)


def _text_verdict(image_path):
    from thai_deck_eval.judge.prompts import PICTURE_RULES, build_picture_prompt

    class N:
        thai, category, part_of_speech, classifier = "หมา", "Animals", "noun", "ตัว"
        gloss = None

    # CliJudge: subscription tokens, and no key wiring. ApiJudge(config)
    # without an injected client ignores config.api_key and falls back to
    # ANTHROPIC_API_KEY -- production passes a client from _api_client().
    judge = CliJudge(JudgeConfig(backend="cli"))
    req = JudgeRequest(note_id="pw-0", rules=list(PICTURE_RULES),
                       prompt=build_picture_prompt(N(), phrase="a dog"),
                       image_path=image_path)
    by_rule = {v.rule: v for v in judge.judge(req)}
    return by_rule["judge/image-embedded-text"]


def test_a_copyright_notice_does_not_fail_the_text_rule(tmp_path):
    """A watermark gives nothing away. Rejecting it costs a usable picture
    and teaches the learner nothing."""
    path = _image_with_text(tmp_path / "c.jpg", "(c) Dave Pearce 2015")
    assert _text_verdict(path).passed


def test_text_naming_the_answer_does_fail_the_text_rule(tmp_path):
    """The other half: the rule exists so the picture cannot give away the
    word. Without this the first test passes under a rule that permits
    everything."""
    path = _image_with_text(tmp_path / "a.jpg", "dog")
    assert not _text_verdict(path).passed
