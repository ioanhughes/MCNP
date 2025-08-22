#!/usr/bin/env python3

"""Command line interface for running MCNP simulations.

This module provides an interactive command line interface as well as
argument parsing so that simulations can be launched non-interactively.
All user interaction previously embedded in :mod:`run_packages` has been
moved here to keep :mod:`run_packages` a pure library.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
from typing import List

import run_packages


logger = logging.getLogger(__name__)


def _select_file_via_dialog() -> str | None:
    """Return a file path selected via a Tk file dialog or ``None``."""

    root = Tk()
    root.withdraw()
    selected = askopenfilename(title="Select MCNP input file to run")
    root.destroy()
    return selected or None


def _select_directory_via_dialog() -> str | None:
    """Return a directory path selected via a Tk directory dialog or ``None``."""

    root = Tk()
    root.withdraw()
    selected = askdirectory(title="Select folder with MCNP input files")
    root.destroy()
    return selected or None


def main() -> None:
    """Entry point for the MCNP runner CLI."""

    parser = argparse.ArgumentParser(
        description="Run MCNP simulations in parallel.",
        epilog="Use --interactive to supply missing options via prompts.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=3,
        help="Number of concurrent MCNP jobs (default: 3)",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Directory containing MCNP input files",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["all", "single"],
        help="Run all files in a directory or a single input file",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to MCNP input file when running a single file",
    )
    parser.add_argument(
        "-a",
        "--action",
        choices=["delete", "backup", "abort"],
        help="Action if existing output files are detected",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt for missing values interactively",
    )
    args = parser.parse_args()

    # Optional interactive override for the number of jobs
    if args.interactive:
        try:
            jobs_input = input(
                f"Enter number of concurrent jobs [default {args.jobs}]: "
            )
            if jobs_input.strip():
                args.jobs = int(jobs_input.strip())
        except Exception:
            logger.warning(f"Invalid input for jobs; using default = {args.jobs}")

    # Determine run mode
    if args.mode is None:
        if args.interactive:
            run_choice = input(
                "Enter 'a' to run all files in a folder, or 's' to run a single file: "
            )
            if run_choice.lower() == "s":
                args.mode = "single"
            elif run_choice.lower() == "a":
                args.mode = "all"
            else:
                logger.warning("Invalid choice; exiting.")
                return
        else:
            logger.error("--mode is required when not running interactively")
            return

    input_files: List[str]
    if args.mode == "single":
        file_path = args.file
        if file_path is None:
            if args.interactive:
                file_path = _select_file_via_dialog()
                if file_path is None:
                    logger.info("No file selected; exiting.")
                    return
            else:
                logger.error("--file is required for single mode when not running interactively")
                return
        inp_path = Path(file_path)
        if not inp_path.is_absolute():
            inp_path = run_packages.resolve_path(inp_path)
        mcnp_dir = inp_path.parent
        input_files = [str(inp_path)]
    else:  # args.mode == "all"
        directory = args.directory
        if directory is None:
            if args.interactive:
                directory = _select_directory_via_dialog()
                if directory is None:
                    logger.info("No folder selected; exiting.")
                    return
            else:
                logger.error("--directory is required for all mode when not running interactively")
                return
        mcnp_dir = run_packages.resolve_path(directory)
        input_files = run_packages.gather_input_files(mcnp_dir, "multi")

    if input_files:
        ctme_value = run_packages.extract_ctme_minutes(Path(input_files[0]))
        num_files = len(input_files)
        estimated_parallel_time = run_packages.calculate_estimated_time(
            ctme_value, num_files, args.jobs
        )
        total_ctme = ctme_value * num_files
        logger.info(
            f"Estimated total run time based on ctme values: {total_ctme:.1f} minutes"
        )
        logger.info(
            f"Estimated actual runtime with {args.jobs} parallel jobs: {estimated_parallel_time:.1f} minutes "
            f"({estimated_parallel_time / 60:.2f} hours)"
        )
        estimated_completion_time = datetime.datetime.now() + datetime.timedelta(
            minutes=estimated_parallel_time
        )
        logger.info(
            f"Estimated completion time: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # Check for existing outputs
    existing_outputs = run_packages.check_existing_outputs(input_files, mcnp_dir)
    if existing_outputs:
        logger.info("Detected existing output files:")
        for f in existing_outputs:
            logger.info(f"  {f}")
        action = args.action
        if action is None:
            if args.interactive:
                choice = input(
                    "Enter 'd' to delete, 'm' to move to 'backup_outputs', or any other key to cancel: "
                )
                if choice.lower() == "d":
                    action = "delete"
                elif choice.lower() == "m":
                    action = "backup"
                else:
                    action = "abort"
            else:
                logger.warning(
                    "Existing output files detected and no --action specified; aborting."
                )
                return
        if action in {"delete", "backup"}:
            run_packages.delete_or_backup_outputs(existing_outputs, mcnp_dir, action)
        else:
            logger.info("Aborting run.")
            return

    if not input_files:
        logger.warning(
            "No input files found in directory (checked for .inp and files without extension)."
        )
        return

    logger.info(
        f"Found {len(input_files)} input files. Running up to {args.jobs} jobs in parallel."
    )

    run_packages.run_simulations(input_files, args.jobs)


if __name__ == "__main__":
    main()

