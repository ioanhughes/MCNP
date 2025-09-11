#!/usr/bin/env python3

import concurrent.futures
import datetime
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .utils.config_utils import load_settings

logger = logging.getLogger(__name__)

"""
Path resolution and configuration
---------------------------------

This module previously read the MCNP base path from a dotfile in the user's
home directory ("~/.mcnp_tools_settings.json"). The GUI, however, persists the
selected path to a project-local "config.json" file. As a result, the runner
would ignore the GUI selection and fall back to the home directory, leading to
errors like:

    ERROR:run_packages MCNP executable not found at /Users/<user>/MCNP_CODE/bin/mcnp6

To unify configuration, we now read settings from both locations with the
following precedence:

1) Environment variables: ``MCNP_BASE_DIR`` or ``MY_MCNP``
2) Project config: ``<repo>/config.json`` (key: "MY_MCNP_PATH")
3) Legacy home config: ``~/.mcnp_tools_settings.json`` (key: "MY_MCNP_PATH")
4) Fallback: ``Path.home()``

This keeps backward compatibility while ensuring the GUI selection is honored.
"""

settings = load_settings()

# Centralised base directory used for all path construction. Priority is given
# to environment variables so installations can be relocated without editing
# configuration files. ``MY_MCNP`` mirrors the variable commonly set by the
# MCNP installation scripts.
env_base_dir = os.getenv("MCNP_BASE_DIR") or os.getenv("MY_MCNP")
settings_base_dir = settings.get("MY_MCNP_PATH")
BASE_DIR = Path(env_base_dir or settings_base_dir or Path.home()).expanduser()


def resolve_path(path: str | Path) -> Path:
    """Return an absolute path, resolving relative paths against ``BASE_DIR``."""
    p = Path(path)
    return p if p.is_absolute() else BASE_DIR / p


def get_mcnp_executable() -> Path:
    """Return the path to the MCNP executable.

    This is resolved dynamically so changes to environment variables or
    configuration files after import time are respected.
    """
    settings = load_settings()
    root = os.getenv("MY_MCNP") or settings.get("MY_MCNP_PATH")
    if root:
        return Path(root).expanduser() / "MCNP_CODE" / "bin" / "mcnp6"
    return Path("/nonexistent/MCNP_CODE/bin/mcnp6")


def calculate_estimated_time(ctme_minutes: float, num_files: int, jobs: int) -> float:
    """Estimate total runtime in minutes for running ``num_files`` with ``jobs`` workers."""

    num_batches = (num_files + jobs - 1) // jobs
    return ctme_minutes * num_batches


def is_valid_input_file(filename: str) -> bool:
    """Return ``True`` if ``filename`` does not end with known output suffixes."""

    invalid_suffixes = {"o", "r", "l", "m", "c", "msht"}
    return not any(filename.endswith(s) for s in invalid_suffixes)


def validate_input_folder(folder: str | Path) -> bool:
    """Check if the provided folder exists.

    Parameters
    ----------
    folder : str
        Path to the folder to validate.

    Returns
    -------
    bool
        ``True`` if the folder exists, ``False`` otherwise.
    """
    return bool(folder and Path(folder).is_dir())


def gather_input_files(folder: str | Path, mode: str) -> List[str]:
    """Return a list of MCNP input files for the given folder and mode."""

    known_output_suffixes = {"o", "r", "l", "c", "msht"}
    if mode == "single":
        return []  # single file handled via GUI
    folder = resolve_path(folder)
    inp_files = list(folder.glob("*.inp"))
    inp_files += [
        f
        for f in folder.glob("*")
        if f.is_file()
        and f.suffix == ""
        and not f.name.startswith(".")
        and not any(f.name.endswith(suffix) for suffix in known_output_suffixes)
    ]
    return [str(f) for f in inp_files]


def check_existing_outputs(inp_files: Iterable[str], folder: str | Path) -> List[str]:
    """Return a list of existing output files corresponding to ``inp_files``."""

    folder = resolve_path(folder)
    existing_outputs: List[str] = []
    for inp in inp_files:
        base = Path(inp).name
        for suffix in ("o", "r", "c", "msht"):
            out_name = folder / f"{base}{suffix}"
            if out_name.exists():
                existing_outputs.append(str(out_name))
    return existing_outputs


def delete_or_backup_outputs(
    existing_outputs: Iterable[str], folder: str | Path, action: str
) -> None:
    """Delete or move existing output files based on ``action``."""

    folder = resolve_path(folder)
    if action == "delete":
        for f in existing_outputs:
            try:
                Path(f).unlink()
                logger.info(f"Deleted {f}")
            except Exception as e:
                logger.error(f"Could not delete {f}: {e}")
    elif action == "backup":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = folder / f"backup_outputs_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for f in existing_outputs:
            try:
                shutil.move(f, backup_dir)
                logger.info(f"Moved {f} to {backup_dir}")
            except Exception as e:
                logger.error(f"Could not move {f}: {e}")


def extract_ctme_minutes(file_path: str | Path) -> float:
    """Return the last ``ctme`` value (minutes) found in ``file_path``."""

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            for line in reversed(lines):
                match = re.search(r"\bctme\s+(\d+(\.\d+)?)", line, re.IGNORECASE)
                if match:
                    return float(match.group(1))
    except Exception as e:
        logger.error(f"Error reading ctme from {file_path}: {e}")
    return 0.0  # Default if not found


def run_mcnp(inp_file: str | Path, process_list: Optional[List[Any]] = None) -> None:
    """Run a single MCNP simulation for ``inp_file``.

    Parameters
    ----------
    inp_file : str | Path
        Input file to run.
    process_list : list[Any], optional
        List in which the spawned process will be stored for later management.
    """

    inp_path = Path(inp_file)
    file_name = inp_path.name
    file_dir = inp_path.parent
    exec_path = get_mcnp_executable()
    if not exec_path.is_file():
        logger.error(f"MCNP executable not found at {exec_path}")
        return
    cmd = [str(exec_path), "ixr", f"name={file_name}"]
    try:
        proc = subprocess.Popen(cmd, cwd=str(file_dir))
        if process_list is not None:
            process_list.append(proc)
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        logger.info(f"Completed: {inp_file}")
    except Exception as e:
        logger.error(f"Error running {inp_file}: {e}")


def run_geometry_plotter(inp_file: str | Path, process_list: Optional[List[Any]] = None) -> None:
    """Launch the MCNP geometry plotter for ``inp_file`` without blocking."""

    inp_path = Path(inp_file)
    file_name = inp_path.name
    file_dir = inp_path.parent
    cmd = [str(get_mcnp_executable()), "ip", f"name={file_name}"]
    try:
        proc = subprocess.Popen(cmd, cwd=str(file_dir))
        if process_list is not None:
            process_list.append(proc)
        logger.info(f"Geometry plotter launched for: {inp_file}")
    except Exception as e:
        logger.error(f"Error launching geometry plotter for {inp_file}: {e}")


def run_mesh_tally(runtpe_file: str | Path, process_list: Optional[List[Any]] = None) -> None:
    """Run the MCNP mesh tally post-processing for ``runtpe_file``."""

    runtpe_path = Path(runtpe_file)
    if not runtpe_path.name.endswith("r"):
        logger.error(f"Mesh tally runtpe file must end with 'r': {runtpe_file}")
        return
    if not runtpe_path.is_file():
        logger.error(f"Runtpe file not found: {runtpe_file}")
        return
    exec_path = get_mcnp_executable()
    if not exec_path.is_file():
        logger.error(f"MCNP executable not found at {exec_path}")
        return
    file_dir = runtpe_path.parent
    cmd = [str(exec_path), "z", f"runtpe={runtpe_path.name}"]
    try:
        proc = subprocess.Popen(cmd, cwd=str(file_dir))
        if process_list is not None:
            process_list.append(proc)
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        logger.info(f"Mesh tally completed: {runtpe_file}")
    except Exception as e:
        logger.error(f"Error running mesh tally for {runtpe_file}: {e}")

def run_simulations(input_files: Iterable[str], jobs: int) -> None:
    """Run MCNP simulations for ``input_files`` using up to ``jobs`` workers."""

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_mcnp, inp): inp for inp in input_files}
        for future in concurrent.futures.as_completed(futures):
            inp = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error running {inp}: {e}")


if __name__ == "__main__":
    from cli import main as cli_main

    cli_main()
