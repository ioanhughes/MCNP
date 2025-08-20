import sys
from pathlib import Path
import json
import logging

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import run_packages


def test_extract_ctme_minutes_returns_last_value(tmp_path):
    file = tmp_path / "sample.inp"
    file.write_text("ctme 5\nother\nctme 7.5\n")
    assert run_packages.extract_ctme_minutes(file) == 7.5


def test_extract_ctme_minutes_missing_returns_zero(tmp_path):
    file = tmp_path / "no_ctme.inp"
    file.write_text("nothing here\n")
    assert run_packages.extract_ctme_minutes(file) == 0.0


def test_validate_input_folder(tmp_path):
    assert run_packages.validate_input_folder(tmp_path)
    assert not run_packages.validate_input_folder(tmp_path / "missing")


def test_gather_input_files_filters_correctly(tmp_path):
    # Create various files
    inp1 = tmp_path / "a.inp"
    inp1.write_text("")
    inp2 = tmp_path / "b"
    inp2.write_text("")
    keep = tmp_path / "keep"
    keep.write_text("")
    # Files that should be ignored
    (tmp_path / "outputo").write_text("")  # extensionless ending with o
    (tmp_path / "resultr").write_text("")  # extensionless ending with r
    (tmp_path / "ignored.txt").write_text("")  # has extension
    (tmp_path / "ignored.o").write_text("")  # known output extension

    files = run_packages.gather_input_files(tmp_path, "multi")
    assert set(files) == {str(inp1), str(inp2), str(keep)}
    # single mode should ignore folder
    assert run_packages.gather_input_files(tmp_path, "single") == []


def test_run_mcnp_missing_executable_logs_error(monkeypatch, tmp_path, caplog):
    dummy_inp = tmp_path / "case.inp"
    dummy_inp.write_text("")
    monkeypatch.setattr(run_packages, "MCNP_EXECUTABLE", tmp_path / "missing" / "mcnp6")
    with caplog.at_level(logging.ERROR):
        run_packages.run_mcnp(dummy_inp)
    assert "MCNP executable not found" in caplog.text
