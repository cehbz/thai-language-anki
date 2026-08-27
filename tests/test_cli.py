import json
from click.testing import CliRunner
from thai_deck_eval.cli import main
from tests.helpers import DeckBuilder

def _invoke(root, *args):
    return CliRunner().invoke(
        main, [str(root), "--no-judge", "--rulebook", "/dev/null", *args])

def test_golden_passes(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    r = _invoke(DeckBuilder(tmp_path).build(), "--format", "json")
    assert r.exit_code == 0, r.output
    rep = json.loads(r.output)
    assert rep["gate"] == "pass" and "scores" in rep

def test_gate_failure_exit_1(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    root = DeckBuilder(tmp_path).build()
    (root / "media" / "images" / "dog.png").unlink()
    r = _invoke(root)
    assert r.exit_code == 1
    assert "mech/media-missing" in r.output

def test_schema_error_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    root = DeckBuilder(tmp_path).build()
    (root / "deck.yaml").write_text("name: [broken")
    r = _invoke(root)
    assert r.exit_code == 1 and "schema/invalid" in r.output

def test_report_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    out = tmp_path / "rep.json"
    r = _invoke(DeckBuilder(tmp_path).build(), "--report", str(out))
    assert r.exit_code == 0 and json.loads(out.read_text())["gate"] == "pass"

def test_malformed_rulebook_exit_2(tmp_path, monkeypatch):
    monkeypatch.setattr("thai_deck_eval.cli._build_language_ports",
                        lambda: (None, None, None, None))
    rb = tmp_path / "bad.yaml"
    rb.write_text("taper_rank: [broken")
    runner = CliRunner()
    r = runner.invoke(main, [str(DeckBuilder(tmp_path).build()),
                              "--no-judge", "--rulebook", str(rb)])
    assert r.exit_code == 2
    assert "error:" in r.output
    assert "Traceback" not in r.output
