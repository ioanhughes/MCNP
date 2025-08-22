import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cli
import run_packages


def test_cli_non_interactive_single_file(monkeypatch, tmp_path):
    # Create a dummy input file
    inp = tmp_path / "test.inp"
    inp.write_text("ctme 10")

    # Ensure no interactive prompts are attempted
    def fail_input(*args, **kwargs):
        raise AssertionError("input() should not be called in non-interactive mode")

    monkeypatch.setattr("builtins.input", fail_input)
    monkeypatch.setattr(run_packages, "extract_ctme_minutes", lambda p: 1.0)
    monkeypatch.setattr(run_packages, "check_existing_outputs", lambda files, folder: [])

    called = {}

    def fake_run_simulations(files, jobs):
        called["files"] = list(files)
        called["jobs"] = jobs

    monkeypatch.setattr(run_packages, "run_simulations", fake_run_simulations)

    monkeypatch.setattr(sys, "argv", ["prog", "--mode", "single", "--file", str(inp), "--jobs", "2"])

    cli.main()

    assert called["files"] == [str(inp)]
    assert called["jobs"] == 2

