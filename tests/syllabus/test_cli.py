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
build_sourcing/default_budgets/run_pipeline together into a Sourcing ctx
+ budgets correctly, not that a real Provider backend does anything.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import zipfile

import pytest
import yaml

from thai_syllabus import cli
from thai_syllabus.attempts import Sourcing
from thai_syllabus.compile import GateRefusal
from thai_syllabus.rules import Compile, CompileReport, Finding, Report
from thai_syllabus.run import RunReport, Spend


def _write_curated_dir(root):
    curated = root / "curated"
    curated.mkdir(parents=True)
    words = [{"id": "rice", "thai": "ข้าว", "meaning": "rice", "category": "Food",  # rice
             "pron": {"syllables": [{"segments": ["kh", "aa", ""],
                                     "vowel_length": "long", "tone": "low"}],
                      "corroboration": "engines_agree"}}]
    (curated / "words.yaml").write_text(yaml.safe_dump(words, allow_unicode=True))
    targets = [{"id": "t-rice", "word": "rice", "skill": "receptive"}]
    (curated / "targets.yaml").write_text(yaml.safe_dump(targets, allow_unicode=True))
    (curated / "profile.yaml").write_text(yaml.safe_dump(
        {"register": "male_colloquial", "emphasis": {}}))
    (curated / "rulebook.yaml").write_text("{}\n", encoding="utf-8")
    return root


# --- compile: real end-to-end happy path ----------------------------------

def test_compile_writes_an_apkg_and_prints_a_summary(tmp_path, capsys):
    # A real, non-forced compile: seed the curated deck's db with a
    # current-best picture, a current-best recording, and a sentence that
    # fills the target -- the spec 4 completeness rules are then all
    # satisfied and the gate opens on its own, with no --force needed.
    from datetime import date

    from thai_syllabus.entities import text_sha
    from thai_syllabus.media import Speaker
    from thai_syllabus.rulebook import PICTURE_FIT_RUBRIC
    from thai_syllabus.store import SyllabusDb

    root = _write_curated_dir(tmp_path / "deck")
    db = SyllabusDb(root / "syllabus.db")

    db.append(port="provide", backend="openverse", key="openverse:rice",
             subject="rice", question={"provides": "picture", "params": {}},
             answer={"items": [{"sha": "pic1"}]}, cost=0.0)
    # rubric must match load_syllabus's own rubrics_for(rules) (the default
    # PICTURE_FIT_RUBRIC, no rulebook.yaml overlay here) -- _DbMediaIndex now
    # threads current_rubric through current_best (Task 11).
    db.append(port="assess", backend="judge", key="judge:x:pic1:picture-for-word",
             subject="rice",
             question={"role": "picture-for-word", "artifact_sha": "pic1",
                      "rubric": PICTURE_FIT_RUBRIC},
             answer={"value": True}, cost=0.0)
    db.add_media(sha="pic1", kind="picture", ext="jpg", source="openverse",
                origin="https://example.com/x.jpg", licence="cc0", acquired=date(2026, 1, 1))

    db.append(port="provide", backend="forvo", key="forvo:rice",
             subject="rice", question={"provides": "recording", "params": {}},
             answer={"items": [{"sha": "rec1"}]}, cost=0.0)
    # derivations.current_best does not yet rank a bare mechanical pass for
    # recordings (Task 5 adds that) -- a judge pass under role
    # "recording-for-word" is what makes a recording candidate current-best
    # today.
    db.append(port="assess", backend="judge", key="judge:x:rec1:recording-for-word",
             subject="rice",
             question={"role": "recording-for-word", "artifact_sha": "rec1", "rubric": None},
             answer={"value": True}, cost=0.0)
    db.add_speaker(Speaker(id="somchai", kind="native"))
    db.add_media(sha="rec1", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by", acquired=date(2026, 1, 1),
                speaker_id="somchai")

    # The default tokenizer falls back to whitespace when pythainlp is
    # absent (as here); a bare single-word sentence puts "rice" at a
    # boundary with no companion token that would need its own curated
    # Word+Target to satisfy fills()'s strict novelty budget (spec 1 §3).
    db.add_sentence(text_sha="s1", text="ข้าว", gloss="rice", voice="learner_voice",  # rice
                    source="llm", origin="draft", licence="n/a", acquired=date(2026, 1, 1))

    # sentence/recording-required (F7): a current-best recording under the
    # sentence's OWN text_sha (Sentence.text_sha derives from the text, not
    # the sentences-table row's stored key above).
    sentence_sha = text_sha("ข้าว")
    db.append(port="provide", backend="forvo", key=f"forvo:{sentence_sha}",
             subject=sentence_sha, question={"provides": "recording", "params": {}},
             answer={"items": [{"sha": "rec-sentence"}]}, cost=0.0)
    db.append(port="assess", backend="judge", key=f"judge:x:rec-sentence:recording-for-word",
             subject=sentence_sha,
             question={"role": "recording-for-word", "artifact_sha": "rec-sentence", "rubric": None},
             answer={"value": True}, cost=0.0)
    db.add_media(sha="rec-sentence", kind="recording", ext="mp3", source="forvo",
                origin="https://forvo.com/x", licence="cc-by", acquired=date(2026, 1, 1),
                speaker_id="somchai")

    out = tmp_path / "deck.apkg"
    rc = cli.main(["compile", "--deck", str(root), "--out", str(out)])
    assert rc == 0  # no --force: this only succeeds if the gate opened on its own
    assert out.exists()
    with zipfile.ZipFile(out) as zf:
        assert "collection.anki2" in zf.namelist()
    text = capsys.readouterr().out
    assert "notes_written" in text or "note" in text.lower()


def test_compile_prints_dropped_cards(tmp_path, capsys):
    # the word has no picture/audio media at all -- every media-gated
    # card (Listening/Production/Spelling) is dropped, only Reading (no
    # media dependency) survives; the CompileReport must say so. Also
    # forced past the closed gate, same as the summary test above.
    root = _write_curated_dir(tmp_path / "deck")
    out = tmp_path / "deck.apkg"
    cli.main(["compile", "--deck", str(root), "--out", str(out), "--force"])
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


# --- run: wiring plumbing (monkeypatched run_pipeline only) ----------------
#
# build_levers/Lever are gone (Task 10); cli._cmd_run now wires its Sourcing
# ctx via wiring.build_sourcing (Task 11) rather than constructing one
# inline. This test lets build_sourcing run for real against the fixture
# deck (no network/subprocess -- it only builds lazy backend rosters) and
# only monkeypatches run_pipeline, proving main(argv) parses flags, builds
# a real Sourcing ctx whose judge_model/image_candidates come from the
# deck's own providers.yaml, and layers --backend-cap onto default_budgets.

@pytest.fixture
def deck(tmp_path):
    root = _write_curated_dir(tmp_path / "deck")
    # non-default values, so the cli-run test can prove Sourcing.image_candidates/
    # judge_model come from the deck's own providers.yaml, not a bare default.
    (root / "curated" / "providers.yaml").write_text(yaml.safe_dump(
        {"image_candidates": 9, "imgfetch_path": "/opt/bin/imgfetch",
         "audiofetch_path": "/opt/bin/audiofetch",
         "judge": {"transport": "cli", "model": "claude-run-test"}}))
    return root


def test_run_wires_a_sourcing_ctx_and_budgets_into_run_pipeline(deck, monkeypatch, capsys):
    calls = {}

    def fake_run(ctx, budgets, **kwargs):
        calls["ctx"] = ctx
        calls["budgets"] = budgets
        return RunReport(attempted=1, improved=1, exhausted=0, available=2, pending=1,
                         sentences_adopted=3, spend={"forvo": Spend(asks=1, cost=0.0)})

    monkeypatch.setattr(cli, "run_pipeline", fake_run)

    rc = cli.main(["run", "--deck", str(deck), "--backend-cap", "forvo=5",
                  "--backend-cap", "learner=3"])
    assert rc == 0
    ctx = calls["ctx"]
    assert isinstance(ctx, Sourcing)
    # drawn from the deck's own (non-default) providers.yaml, not a bare
    # Sourcing dataclass default.
    assert ctx.judge_model == "claude-run-test"
    assert ctx.image_candidates == 9
    assert calls["budgets"]["forvo"].max_asks == 5
    assert calls["budgets"]["learner"].max_asks == 3
    text = capsys.readouterr().out
    assert "attempted=1" in text
    assert "improved=1" in text
    assert "pending=1" in text
    assert "sentences_adopted=3" in text


def test_run_prints_excluded_and_unreachable(deck, monkeypatch, capsys):
    """The two "what went wrong" fields have to reach the operator's
    terminal, not just the persisted row."""
    monkeypatch.setattr(cli, "run_pipeline",
                        lambda ctx, budgets, **kw: RunReport(attempted=1, excluded=2))
    rc = cli.main(["run", "--deck", str(deck)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "excluded=2" in text
    assert "unreachable=False" in text


def test_run_exits_1_when_the_judge_was_unreachable(deck, monkeypatch, capsys):
    """A run that could not reach the judge did not do its job: exit
    non-zero so a script or a cron job notices."""
    monkeypatch.setattr(cli, "run_pipeline",
                        lambda ctx, budgets, **kw: RunReport(attempted=0, unreachable=True))
    rc = cli.main(["run", "--deck", str(deck)])
    assert rc == 1
    assert "unreachable=True" in capsys.readouterr().out


# --- logging: warnings must reach the terminal (item 3) --------------------

def test_main_configures_logging_so_module_warnings_reach_stderr(deck):
    """attempts.py logs "judge unreachable" at WARNING; without a logging
    configuration at the entry point those warnings never reach a terminal.

    Run in a real subprocess, not in-process: pytest installs its own root
    handler, which makes logging.basicConfig a silent no-op and would let
    this pass whether or not cli.main configures anything. A fresh
    interpreter is the only honest witness. run_pipeline is replaced there
    so the subprocess makes no network/subprocess call of its own.
    """
    program = textwrap.dedent("""
        import logging, sys
        from thai_syllabus import cli
        from thai_syllabus.run import RunReport

        def fake_run(ctx, budgets, **kw):
            logging.getLogger("thai_syllabus.attempts").warning("judge unreachable (401)")
            return RunReport()

        cli.run_pipeline = fake_run
        sys.exit(cli.main(["run", "--deck", sys.argv[1]]))
    """)
    proc = subprocess.run([sys.executable, "-c", program, str(deck)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "judge unreachable (401)" in proc.stderr
    assert "WARNING" in proc.stderr and "thai_syllabus.attempts" in proc.stderr


# --- existing subcommands keep working -------------------------------------

def test_migrate_and_import_subcommands_still_parse():
    # not exercised end-to-end here (already covered elsewhere); this only
    # guards against `compile`/`run` breaking argparse's subparser wiring.
    parser_argv_migrate = ["migrate", "--old-deck", "x", "--old-data", "y",
                          "--new-root", "z"]
    with pytest.raises(SystemExit):
        cli.main([])  # no subcommand -- required=True still enforced
