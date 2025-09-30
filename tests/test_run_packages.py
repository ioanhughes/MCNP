import sys
from pathlib import Path
import json
import logging

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mcnp import run_packages


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
    (tmp_path / "meshmsht").write_text("")  # extensionless ending with msht
    (tmp_path / "ignored.txt").write_text("")  # has extension
    (tmp_path / "ignored.o").write_text("")  # known output extension
    (tmp_path / ".DS_Store").write_text("")  # hidden file

    files = run_packages.gather_input_files(tmp_path, "multi")
    assert set(files) == {str(inp1), str(inp2), str(keep)}
    # single mode should ignore folder
    assert run_packages.gather_input_files(tmp_path, "single") == []


def test_check_existing_outputs_includes_msht(tmp_path):
    inp = tmp_path / "case"
    inp.write_text("")
    msht = tmp_path / "casemsht"
    msht.write_text("")
    outputs = run_packages.check_existing_outputs([str(inp)], tmp_path)
    assert str(msht) in outputs


def test_run_mcnp_missing_executable_logs_error(monkeypatch, tmp_path, caplog):
    dummy_inp = tmp_path / "case.inp"
    dummy_inp.write_text("")
    monkeypatch.setattr(run_packages, "get_mcnp_executable", lambda: tmp_path / "missing" / "mcnp6")
    with caplog.at_level(logging.ERROR):
        run_packages.run_mcnp(dummy_inp)
    assert "MCNP executable not found" in caplog.text


def test_extract_ctme_minutes_handles_binary_file(tmp_path, caplog):
    binary = tmp_path / "binaryfile"
    binary.write_bytes(b"\xb8\x00\x00")
    with caplog.at_level(logging.ERROR):
        value = run_packages.extract_ctme_minutes(binary)
    assert value == 0.0
    assert "Error reading ctme" not in caplog.text


def test_run_mesh_tally_invalid_suffix_logs_error(tmp_path, caplog):
    runtpe = tmp_path / "pi_2cm"
    runtpe.write_text("")
    with caplog.at_level(logging.ERROR):
        run_packages.run_mesh_tally(runtpe)
    assert "must end with 'r'" in caplog.text


def test_run_mesh_tally_missing_executable_logs_error(monkeypatch, tmp_path, caplog):
    runtpe = tmp_path / "pi_2cmr"
    runtpe.write_text("")
    monkeypatch.setattr(run_packages, "get_mcnp_executable", lambda: tmp_path / "missing" / "mcnp6")
    with caplog.at_level(logging.ERROR):
        run_packages.run_mesh_tally(runtpe)
    assert "MCNP executable not found" in caplog.text


def test_get_mcnp_executable_prefers_base_dir_env(monkeypatch, tmp_path):
    env_base = tmp_path / "env"
    monkeypatch.setenv("MCNP_BASE_DIR", str(env_base))
    monkeypatch.setenv("MY_MCNP", str(tmp_path / "my_mcnp"))
    monkeypatch.setattr(run_packages, "load_settings", lambda: {"MY_MCNP_PATH": str(tmp_path / "settings")})

    executable = run_packages.get_mcnp_executable()

    expected = env_base / "MCNP_CODE" / "bin" / "mcnp6"
    assert executable == expected
