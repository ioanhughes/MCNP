def calculate_estimated_time(ctme_minutes, num_files, jobs):
    num_batches = (num_files + jobs - 1) // jobs
    return ctme_minutes * num_batches

def run_simulations_concurrently(inp_files, jobs, running_processes, run_mcnp_fn):
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)
    futures = {executor.submit(run_mcnp_fn, f, running_processes): f for f in inp_files}
    return executor, futures

def is_valid_input_file(filename):
    invalid_suffixes = {"o", "r", "l", "m"}
    return not any(filename.endswith(s) for s in invalid_suffixes)
def validate_input_folder(folder):
    import os
    if not folder or not os.path.isdir(folder):
        return False
    os.chdir(folder)
    return True

def gather_input_files(folder, mode):
    import glob, os
    known_output_suffixes = {"o", "r", "l"}
    if mode == "single":
        return []  # single file handled via GUI
    else:
        # Always resolve from BASE_DIR if not already absolute
        if not os.path.isabs(folder):
            folder = os.path.expanduser(os.path.join("~/Documents/PhD/MCNP/MY_MCNP", folder))
        inp_files = glob.glob(os.path.join(folder, "*.inp"))
        inp_files += [
            os.path.basename(f) for f in glob.glob(os.path.join(folder, "*"))
            if os.path.isfile(f)
            and os.path.splitext(f)[1] == ""
            and not any(f.endswith(suffix) for suffix in known_output_suffixes)
        ]
        return inp_files

def check_existing_outputs(inp_files, folder):
    import os, datetime, shutil
    # Always resolve from BASE_DIR if not already absolute
    if not os.path.isabs(folder):
        folder = os.path.expanduser(os.path.join("~/Documents/PhD/MCNP/MY_MCNP", folder))
    existing_outputs = []
    for inp in inp_files:
        base = os.path.splitext(inp)[0]
        for suffix in ("o", "r"):
            out_name = os.path.join(folder, f"{base}{suffix}")
            if os.path.exists(out_name):
                existing_outputs.append(out_name)
    return existing_outputs

def delete_or_backup_outputs(existing_outputs, folder, action):
    import shutil, os, datetime
    # Always resolve from BASE_DIR if not already absolute
    if not os.path.isabs(folder):
        folder = os.path.expanduser(os.path.join("~/Documents/PhD/MCNP/MY_MCNP", folder))
    if action == "delete":
        for f in existing_outputs:
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")
    elif action == "backup":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(folder, f"backup_outputs_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        for f in existing_outputs:
            try:
                shutil.move(f, backup_dir)
                print(f"Moved {f} to {backup_dir}")
            except Exception as e:
                print(f"Could not move {f}: {e}")
#!/usr/bin/env python3

import subprocess
import glob
import os
import argparse
import concurrent.futures
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
import shutil
import datetime
import re

# Dynamically load MCNP6 executable path from user settings
import os
import json

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".mcnp_tools_settings.json")
default_path = "/Users/ioanhughes/Documents/PhD/MCNP/MY_MCNP"

try:
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
        base_path = settings.get("MY_MCNP_PATH", default_path)
except Exception:
    base_path = default_path

MCNP_EXECUTABLE = os.path.join(base_path, "MCNP_CODE", "bin", "mcnp6")

def extract_ctme_minutes(file_path):
    """
    Extract the 'ctme' value (in minutes) from the last line of the file.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                match = re.search(r"\bctme\s+(\d+(\.\d+)?)", line, re.IGNORECASE)
                if match:
                    return float(match.group(1))
    except Exception as e:
        print(f"Error reading ctme from {file_path}: {e}")
    return 0.0  # Default if not found

def run_mcnp(inp_file, process_list=None):
    """
    Run a single MCNP simulation for the given input file.
    Optionally, register the process in process_list for later termination.
    """
    import os
    file_name = os.path.basename(inp_file)
    file_dir = os.path.dirname(inp_file)
    cmd = [MCNP_EXECUTABLE, "ixr", f"name={file_name}"]
    try:
        proc = subprocess.Popen(cmd, cwd=file_dir)
        if process_list is not None:
            process_list.append(proc)
        return_code = proc.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        print(f"Completed: {inp_file}")
    except Exception as e:
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

    # Prompt for number of concurrent jobs if desired
    try:
        jobs_input = input(f"Enter number of concurrent jobs [default {args.jobs}]: ")
        if jobs_input.strip():
            args.jobs = int(jobs_input.strip())
    except ValueError:
        print(f"Invalid input for jobs; using default = {args.jobs}")

    # Ask whether to run all files in a folder or a single file
    run_choice = input("Enter 'a' to run all files in a folder, or 's' to run a single file: ")
    if run_choice.lower() == 's':
        # Single-file run: ask for input file
        root = Tk(); root.withdraw()
        selected_file = askopenfilename(title="Select MCNP input file to run")
        root.destroy()
        if not selected_file:
            print("No file selected; exiting.")
            return
        args.directory = os.path.dirname(selected_file)
        input_files = [os.path.basename(selected_file)]
        # Always resolve directory from BASE_DIR
        os.chdir(os.path.join(os.path.expanduser("~/Documents/PhD/MCNP/MY_MCNP"), os.path.relpath(args.directory, "/")))
    elif run_choice.lower() == 'a':
        # Multi-file run: ask for directory if not provided
        if args.directory is None:
            root = Tk(); root.withdraw()
            selected = askdirectory(title="Select folder with MCNP input files")
            root.destroy()
            if not selected:
                print("No folder selected; exiting.")
                return
            args.directory = selected
        # Always resolve directory from BASE_DIR
        mcnp_dir = os.path.join(os.path.expanduser("~/Documents/PhD/MCNP/MY_MCNP"), os.path.relpath(args.directory, "/")) if not os.path.isabs(args.directory) else args.directory
        os.chdir(mcnp_dir)
        # Find all MCNP input files: .inp files and files without an extension
        inp_files = glob.glob(os.path.join(mcnp_dir, "*.inp"))
        noext_files = [
            f for f in glob.glob(os.path.join(mcnp_dir, "*"))
            if os.path.isfile(f) and os.path.splitext(f)[1] == ""
        ]
        input_files = sorted(set([os.path.basename(f) for f in inp_files + noext_files]))
    else:
        print("Invalid choice; exiting.")
        return

    if input_files:
        # Always resolve directory from BASE_DIR
        mcnp_dir = os.path.join(os.path.expanduser("~/Documents/PhD/MCNP/MY_MCNP"), os.path.relpath(args.directory, "/")) if not os.path.isabs(args.directory) else args.directory
        ctme_value = extract_ctme_minutes(os.path.join(mcnp_dir, input_files[0]))
        num_files = len(input_files)
        num_batches = (num_files + args.jobs - 1) // args.jobs
        estimated_parallel_time = ctme_value * num_batches
        total_ctme = ctme_value * num_files
        print(f"Estimated total run time based on ctme values: {total_ctme:.1f} minutes")
        print(f"Estimated actual runtime with {args.jobs} parallel jobs: {estimated_parallel_time:.1f} minutes ({estimated_parallel_time / 60:.2f} hours)")
        estimated_completion_time = datetime.datetime.now() + datetime.timedelta(minutes=estimated_parallel_time)
        print(f"Estimated completion time: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check for existing MCNP output files and prompt user
    existing_outputs = []
    # Always resolve directory from BASE_DIR
    mcnp_dir = os.path.join(os.path.expanduser("~/Documents/PhD/MCNP/MY_MCNP"), os.path.relpath(args.directory, "/")) if not os.path.isabs(args.directory) else args.directory
    for inp in input_files:
        base = os.path.splitext(inp)[0]
        for suffix in ("o", "r"):
            out_name = os.path.join(mcnp_dir, f"{base}{suffix}")
            if os.path.exists(out_name):
                existing_outputs.append(out_name)
    if existing_outputs:
        print("Detected existing output files:")
        for f in existing_outputs:
            print(f"  {f}")
        choice = input("Enter 'd' to delete, 'm' to move to 'backup_outputs', or any other key to cancel: ")
        if choice.lower() == "d":
            for f in existing_outputs:
                try:
                    os.remove(f)
                    print(f"Deleted {f}")
                except Exception as e:
                    print(f"Could not delete {f}: {e}")
        elif choice.lower() == "m":
            # Create a timestamped backup folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(mcnp_dir, f"backup_outputs_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)
            for f in existing_outputs:
                try:
                    shutil.move(f, backup_dir)
                    print(f"Moved {f} to {backup_dir}")
                except Exception as e:
                    print(f"Could not move {f}: {e}")
        else:
            print("Aborting run.")
            return

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