#!/usr/bin/env python3

import argparse
import concurrent.futures
import datetime
import glob
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


def run_simulations_concurrently(
    inp_files: List[str],
    jobs: int,
    running_processes: List[Any],
    run_mcnp_fn: Callable[[str, List[Any]], Any],
) -> Tuple[concurrent.futures.ProcessPoolExecutor, Dict[concurrent.futures.Future[Any], str]]:
    """Run MCNP simulations concurrently using a process pool."""

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)
    futures = {executor.submit(run_mcnp_fn, f, running_processes): f for f in inp_files}
    return executor, futures


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
        with open(file_path, "r") as f:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCNP simulations in parallel.")
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=3,
        help="Number of concurrent MCNP jobs (default: 3)",
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=None,
        help="Directory containing MCNP input files (default: prompt via GUI)",
    )
    args = parser.parse_args()

    # Prompt for number of concurrent jobs if desired
    try:
        jobs_input = input(f"Enter number of concurrent jobs [default {args.jobs}]: ")
        if jobs_input.strip():
            args.jobs = int(jobs_input.strip())
    except ValueError:
        logger.warning(f"Invalid input for jobs; using default = {args.jobs}")

    # Ask whether to run all files in a folder or a single file
    run_choice = input("Enter 'a' to run all files in a folder, or 's' to run a single file: ")
    if run_choice.lower() == "s":
        # Single-file run: ask for input file
        root = Tk()
        root.withdraw()
        selected_file = askopenfilename(title="Select MCNP input file to run")
        root.destroy()
        if not selected_file:
            logger.info("No file selected; exiting.")
            return
        args.directory = Path(selected_file).parent
        mcnp_dir = resolve_path(args.directory)
        os.chdir(str(mcnp_dir))
        input_files = [Path(selected_file).name]
    elif run_choice.lower() == "a":
        # Multi-file run: ask for directory if not provided
        if args.directory is None:
            root = Tk()
            root.withdraw()
            selected = askdirectory(title="Select folder with MCNP input files")
            root.destroy()
            if not selected:
                logger.info("No folder selected; exiting.")
                return
            args.directory = selected
        mcnp_dir = resolve_path(args.directory)
        os.chdir(str(mcnp_dir))
        # Find all MCNP input files: .inp files and files without an extension
        inp_files = list(mcnp_dir.glob("*.inp"))
        noext_files = [f for f in mcnp_dir.glob("*") if f.is_file() and f.suffix == ""]
        input_files = sorted({f.name for f in inp_files + noext_files})
    else:
        logger.warning("Invalid choice; exiting.")
        return

    if input_files:
        ctme_value = extract_ctme_minutes(mcnp_dir / input_files[0])
        num_files = len(input_files)
        num_batches = (num_files + args.jobs - 1) // args.jobs
        estimated_parallel_time = ctme_value * num_batches
        total_ctme = ctme_value * num_files
        logger.info(f"Estimated total run time based on ctme values: {total_ctme:.1f} minutes")
        logger.info(
            f"Estimated actual runtime with {args.jobs} parallel jobs: {estimated_parallel_time:.1f} minutes ({estimated_parallel_time / 60:.2f} hours)"
        )
        estimated_completion_time = datetime.datetime.now() + datetime.timedelta(minutes=estimated_parallel_time)
        logger.info(f"Estimated completion time: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for existing MCNP output files and prompt user
    existing_outputs = check_existing_outputs(input_files, mcnp_dir)
    if existing_outputs:
        logger.info("Detected existing output files:")
        for f in existing_outputs:
            logger.info(f"  {f}")
        choice = input("Enter 'd' to delete, 'm' to move to 'backup_outputs', or any other key to cancel: ")
        if choice.lower() == "d":
            delete_or_backup_outputs(existing_outputs, mcnp_dir, "delete")
        elif choice.lower() == "m":
            delete_or_backup_outputs(existing_outputs, mcnp_dir, "backup")
        else:
            logger.info("Aborting run.")
            return

    if not input_files:
        logger.warning("No input files found in directory (checked for .inp and files without extension).")
        return

    logger.info(f"Found {len(input_files)} input files. Running up to {args.jobs} jobs in parallel.")

    # Run with a process pool to limit concurrency
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        # Schedule and run all simulations, collecting results to force execution
        for _ in executor.map(run_mcnp, input_files):
            pass


if __name__ == "__main__":
    main()

