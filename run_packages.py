#!/usr/bin/env python3

import concurrent.futures
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Dynamically load MCNP6 base path from user settings
SETTINGS_PATH = Path.home() / ".mcnp_tools_settings.json"

try:
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
except Exception:
    settings = {}

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


# Allow the MCNP executable to be located via the ``MY_MCNP`` environment
# variable, falling back to ``BASE_DIR`` if it is not set.
MCNP_EXECUTABLE = Path(os.getenv("MY_MCNP") or BASE_DIR) / "MCNP_CODE" / "bin" / "mcnp6"


def calculate_estimated_time(ctme_minutes: float, num_files: int, jobs: int) -> float:
    """Estimate total runtime in minutes for running ``num_files`` with ``jobs`` workers."""

    num_batches = (num_files + jobs - 1) // jobs
    return ctme_minutes * num_batches


def is_valid_input_file(filename: str) -> bool:
    """Return ``True`` if ``filename`` does not end with known output suffixes."""

    invalid_suffixes = {"o", "r", "l", "m", "c"}
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

    known_output_suffixes = {"o", "r", "l", "c"}
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
        for suffix in ("o", "r", "c"):
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
    if not Path(MCNP_EXECUTABLE).is_file():
        logger.error(f"MCNP executable not found at {MCNP_EXECUTABLE}")
        return
    cmd = [str(MCNP_EXECUTABLE), "ixr", f"name={file_name}"]
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
    cmd = [str(MCNP_EXECUTABLE), "ip", f"name={file_name}"]
    try:
        proc = subprocess.Popen(cmd, cwd=str(file_dir))
        if process_list is not None:
            process_list.append(proc)
        logger.info(f"Geometry plotter launched for: {inp_file}")
    except Exception as e:
        logger.error(f"Error launching geometry plotter for {inp_file}: {e}")

def run_simulations(input_files: Iterable[str], jobs: int) -> None:
    """Run MCNP simulations for ``input_files`` using up to ``jobs`` workers."""

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        for _ in executor.map(run_mcnp, input_files):
            pass


if __name__ == "__main__":
    from cli import main as cli_main

    cli_main()

