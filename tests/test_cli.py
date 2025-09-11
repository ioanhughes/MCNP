import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcnp import cli, run_packages


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


@pytest.fixture
def cli_setup(tmp_path):
    """Return environment and input file for running the CLI via subprocess."""
    mcnp_root = tmp_path / "fake_mcnp"
    bin_dir = mcnp_root / "MCNP_CODE" / "bin"
    bin_dir.mkdir(parents=True)
    exec_path = bin_dir / "mcnp6"
    exec_path.write_text("#!/bin/sh\nexit 0\n")
    exec_path.chmod(0o755)

    inp = tmp_path / "test.inp"
    inp.write_text("ctme 10\n")

    env = os.environ.copy()
    env["MY_MCNP"] = str(mcnp_root)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    return env, inp


def test_cli_subprocess_success_single_file(cli_setup):
    env, inp = cli_setup
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "mcnp.cli",
            "--mode",
            "single",
            "--file",
            str(inp),
            "--jobs",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
    )
    assert result.returncode == 0
    assert "Found 1 input files" in result.stderr
    assert "Running up to 1 jobs in parallel" in result.stderr


def test_cli_missing_file_non_interactive_no_prompt(cli_setup):
    env, _ = cli_setup
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "mcnp.cli", "--mode", "single"],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
    )
    assert result.returncode == 0
    assert (
        "--file is required for single mode when not running interactively" in result.stderr
    )
    assert "Enter number of concurrent jobs" not in result.stdout
    assert "Enter 'a' to run all files in a folder" not in result.stdout


def test_cli_invalid_mode_exit_code(cli_setup):
    env, _ = cli_setup
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "mcnp.cli", "--mode", "invalid"],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_cli_interactive_jobs_prompt(cli_setup):
    env, inp = cli_setup
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "mcnp.cli",
            "--interactive",
            "--mode",
            "single",
            "--file",
            str(inp),
        ],
        input="1\n",
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
    )
    assert result.returncode == 0
    assert "Enter number of concurrent jobs" in result.stdout
    assert "Found 1 input files. Running up to 1 jobs in parallel." in result.stderr

