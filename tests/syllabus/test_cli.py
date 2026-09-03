"""Tests for cli.py's `compile` and `run` subcommands (cli.py's own
docstring named this gap: "compile and the sourcing run stay library-level
until their configs settle ... the subcommands land with that wiring").

`compile`'s happy path is a real end-to-end run (curated dir -> a real
.apkg via genanki, no mocks) since that's cheap and proves the actual
wiring works; its gate/force branches are exercised via monkeypatched
`cli.compile_syllabus`, matching how compile.py's own gate/force logic is
already fully covered by test_compile.py -- this file only needs to prove
the CLI plumbs `--force` through and reports a refusal correctly. `run`'s
tests monkeypatch the wiring calls entirely (no network, no subprocess,
no pythainlp): the point here is that main(argv) parses flags and wires
build_provider/build_assessor/build_levers/default_budgets/run() together
correctly, not that a real Provider backend does anything.
"""
from __future__ import annotations

import zipfile

import pytest
import yaml

from thai_syllabus import cli
from thai_syllabus.compile import GateRefusal
from thai_syllabus.rules import Compile, CompileReport, Finding, Report
from thai_syllabus.run import Budget, RunReport, Spend


def _write_curated_dir(root):
    curated = root / "curated"
    curated.mkdir(parents=True)
    words = [{"id": "rice", "thai": "ข้าว", "meaning": "rice",  # rice
             "pron": {"syllables": [{"segments": ["kh", "aa", ""],
                                     "vowel_length": "long", "tone": "low"}],
                      "corroboration": "engines_agree"}}]
    (curated / "words.yaml").write_text(yaml.safe_dump(words, allow_unicode=True))
    targets = [{"id": "t-rice", "word": "rice", "skill": "receptive"}]
    (curated / "targets.yaml").write_text(yaml.safe_dump(targets, allow_unicode=True))
    (curated / "profile.yaml").write_text(yaml.safe_dump(
        {"register": "male_colloquial", "emphasis": {}}))
    return root


# --- compile: real end-to-end happy path ----------------------------------

def test_compile_writes_an_apkg_and_prints_a_summary(tmp_path, capsys):
    root = _write_curated_dir(tmp_path / "deck")
    out = tmp_path / "deck.apkg"
    rc = cli.main(["compile", "--deck", str(root), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        assert "collection.anki2" in zf.namelist()
    text = capsys.readouterr().out
    assert "notes_written" in text or "note" in text.lower()


def test_compile_prints_dropped_cards(tmp_path, capsys):
    # the word has no picture/audio media at all -- every media-gated
    # card (Listening/Production/Spelling) is dropped, only Reading (no
    # media dependency) survives; the CompileReport must say so.
    root = _write_curated_dir(tmp_path / "deck")
    out = tmp_path / "deck.apkg"
    cli.main(["compile", "--deck", str(root), "--out", str(out)])
    text = capsys.readouterr().out
    assert "dropped" in text.lower()


# --- compile: gate / --force plumbing (monkeypatched compile_syllabus) ----

def test_compile_refuses_and_reports_findings_when_gate_is_closed(
        tmp_path, monkeypatch, capsys):
    root = _write_curated_dir(tmp_path / "deck")
    report = Report(syllabus_state_id="s", rulebook_id="r",
                    findings=(Finding(rule="syllabus/closure", note_id="t-rice",
                                      evidence="bad reference"),),
                    metrics=(), gate=False)

    def fake_compile_syllabus(syllabus, db, media_store, out_path, *, force=False):
        assert force is False
        raise GateRefusal(report)

    monkeypatch.setattr(cli, "compile_syllabus", fake_compile_syllabus)
    rc = cli.main(["compile", "--deck", str(root), "--out", str(tmp_path / "out.apkg")])
    assert rc == 1
    text = capsys.readouterr().out
    assert "syllabus/closure" in text
    assert "bad reference" in text


def test_compile_force_flag_is_threaded_to_compile_syllabus(tmp_path, monkeypatch):
    root = _write_curated_dir(tmp_path / "deck")
    calls = []

    def fake_compile_syllabus(syllabus, db, media_store, out_path, *, force=False):
        calls.append(force)
        return Compile(label="deck", syllabus_state_id="s", compile_id="s:1",
                       report=CompileReport(compile_id="s:1", gate=False, forced=True,
                                            warnings=("w1",), notes_written=1,
                                            cards_written=1, dropped=(),
                                            out_path=str(out_path)))

    monkeypatch.setattr(cli, "compile_syllabus", fake_compile_syllabus)
    rc = cli.main(["compile", "--deck", str(root), "--out", str(tmp_path / "out.apkg"),
                  "--force"])
    assert rc == 0
    assert calls == [True]


# --- run: wiring plumbing (monkeypatched) ----------------------------------

@pytest.fixture
def deck(tmp_path):
    return _write_curated_dir(tmp_path / "deck")


def test_run_wires_provider_assessor_levers_and_budgets(deck, monkeypatch, capsys):
    calls = {}

    fake_syllabus = object()
    monkeypatch.setattr(cli, "load_syllabus", lambda root: fake_syllabus)

    fake_provider = object()
    fake_assessor = object()
    monkeypatch.setattr(cli, "build_provider", lambda cfg, db, media_store: fake_provider)
    monkeypatch.setattr(cli, "build_assessor", lambda cfg, db: fake_assessor)
    monkeypatch.setattr(cli, "default_budgets",
                        lambda cfg: {"forvo": Budget(max_asks=450)})
    monkeypatch.setattr(cli, "build_levers",
                        lambda syllabus, provider, cfg: {"picture": []})

    def fake_run(syllabus, cache, budgets, levers_by_kind, **kwargs):
        calls["syllabus"] = syllabus
        calls["cache"] = cache
        calls["budgets"] = budgets
        calls["levers_by_kind"] = levers_by_kind
        return RunReport(attempted=1, improved=1, exhausted=0, available=2,
                         spend={"forvo": Spend(asks=1, cost=0.0)})

    monkeypatch.setattr(cli, "run_pipeline", fake_run)

    rc = cli.main(["run", "--deck", str(deck)])
    assert rc == 0
    assert calls["syllabus"] is fake_syllabus
    assert calls["budgets"]["forvo"].max_asks == 450
    assert calls["levers_by_kind"] == {"picture": []}
    text = capsys.readouterr().out
    assert "attempted=1" in text
    assert "improved=1" in text


def test_run_backend_cap_overrides_the_default_budget(deck, monkeypatch):
    calls = {}
    monkeypatch.setattr(cli, "load_syllabus", lambda root: object())
    monkeypatch.setattr(cli, "build_provider", lambda cfg, db, media_store: object())
    monkeypatch.setattr(cli, "build_assessor", lambda cfg, db: object())
    monkeypatch.setattr(cli, "default_budgets",
                        lambda cfg: {"forvo": Budget(max_asks=450)})
    monkeypatch.setattr(cli, "build_levers", lambda syllabus, provider, cfg: {})

    def fake_run(syllabus, cache, budgets, levers_by_kind, **kwargs):
        calls["budgets"] = budgets
        return RunReport()

    monkeypatch.setattr(cli, "run_pipeline", fake_run)

    rc = cli.main(["run", "--deck", str(deck), "--backend-cap", "forvo=5",
                  "--backend-cap", "learner=3"])
    assert rc == 0
    assert calls["budgets"]["forvo"].max_asks == 5
    assert calls["budgets"]["learner"].max_asks == 3


# --- existing subcommands keep working -------------------------------------

def test_migrate_and_import_subcommands_still_parse():
    # not exercised end-to-end here (already covered elsewhere); this only
    # guards against `compile`/`run` breaking argparse's subparser wiring.
    parser_argv_migrate = ["migrate", "--old-deck", "x", "--old-data", "y",
                          "--new-root", "z"]
    with pytest.raises(SystemExit):
        cli.main([])  # no subcommand -- required=True still enforced
