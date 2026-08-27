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
