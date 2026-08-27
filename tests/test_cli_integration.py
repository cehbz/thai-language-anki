"""Real-ports CLI regression net for C1 (see fix-wave brief): the golden
deck must pass the gate with real pythainlp ports wired up (i.e. without
monkeypatching `_build_language_ports` away like tests/test_cli.py does).
"""
import pytest
from click.testing import CliRunner

from thai_deck_eval.cli import main
from tests.helpers import DeckBuilder

pytestmark = pytest.mark.integration


def test_golden_deck_passes_gate_with_real_ports(tmp_path):
    root = DeckBuilder(tmp_path).build()
    r = CliRunner().invoke(main, [str(root), "--no-judge"])
    assert r.exit_code == 0, r.output
    assert "gate: PASS" in r.output
    assert "lang/target-not-token" not in r.output
