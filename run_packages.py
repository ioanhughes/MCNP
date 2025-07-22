#!/usr/bin/env python3

import subprocess
import glob
import os
import argparse
import concurrent.futures
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Path to your MCNP6 executable
MCNP_EXECUTABLE = "/Users/ioanhughes/Documents/PhD/MCNP/MY_MCNP/MCNP_CODE/bin/mcnp6"

def run_mcnp(inp_file):
    """
    Run a single MCNP simulation for the given input file.
    """
    base = os.path.splitext(inp_file)[0]
    cmd = [MCNP_EXECUTABLE, "ixr", f"n={base}"]
    try:
        subprocess.run(cmd, check=True)
        print(f"Completed: {inp_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {inp_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run MCNP simulations in parallel.")
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=3,
        help="Number of concurrent MCNP jobs (default: 3)"
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=None,
        help="Directory containing MCNP input files (default: prompt via GUI)"
    )
    args = parser.parse_args()

    # Prompt user with a folder selector if no directory is provided
    if args.directory is None:
        root = Tk()
        root.withdraw()
        selected = askdirectory(title="Select folder with MCNP input files")
        root.destroy()
        if not selected:
            print("No folder selected; exiting.")
            return
        args.directory = selected

    # Change to the specified directory
    os.chdir(args.directory)

    # Find all MCNP input files: .inp files and files without an extension
    inp_files = glob.glob("*.inp")
    noext_files = [
        f for f in glob.glob("*")
        if os.path.isfile(f) and os.path.splitext(f)[1] == ""
    ]
    input_files = sorted(set(inp_files + noext_files))
    if not input_files:
        print("No input files found in directory (checked for .inp and files without extension).")
        return

    print(f"Found {len(input_files)} input files. Running up to {args.jobs} jobs in parallel.")

    # Run with a process pool to limit concurrency
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        # Schedule and run all simulations, collecting results to force execution
        for _ in executor.map(run_mcnp, input_files):
            pass

if __name__ == "__main__":
    main()