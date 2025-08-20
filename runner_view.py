import concurrent.futures
import datetime
import os
import threading
import logging
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Any, List, Optional

import ttkbootstrap as ttk

from he3_plotter.io_utils import select_file, select_folder
from run_packages import (
    calculate_estimated_time,
    check_existing_outputs,
    delete_or_backup_outputs,
    extract_ctme_minutes,
    gather_input_files,
    run_geometry_plotter,
    run_mcnp,
    validate_input_folder,
)


@dataclass
class SimulationJob:
    """Representation of a single MCNP simulation input file."""

    filepath: str
    name: str = field(init=False)
    base: str = field(init=False)
    status: str = "Pending"

    def __post_init__(self) -> None:
        """Populate derived filename attributes after initialisation."""

        self.name = os.path.basename(self.filepath)
        self.base = os.path.splitext(self.name)[0]


class RunnerView:
    """Logic for running MCNP simulations."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        """Create the simulation runner view.

        Parameters
        ----------
        app : Any
            Main application instance providing shared variables.
        parent : tk.Widget
            Parent widget that will contain the runner view.
        """

        self.app = app
        self.frame = parent

        # runtime attributes
        self.executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self.future_map: dict[concurrent.futures.Future[Any], SimulationJob] = {}
        self.run_in_progress: bool = False
        self.update_countdown: bool = False
        self.running_processes: List[Any] = []

        self.build()

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct all runner widgets."""

        runner_frame = ttk.LabelFrame(self.frame, text="MCNP Simulation Runner")
        runner_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(runner_frame, text="MCNP Input Folder:").pack(anchor="w")
        self.app.mcnp_folder_var.set("")
        folder_entry = tk.Entry(runner_frame, textvariable=self.app.mcnp_folder_var, width=50)
        folder_entry.pack(fill="x", pady=2)
        if getattr(self.app, "tkdnd", None):
            folder_entry.drop_target_register(self.app.tkdnd.DND_FILES)  # type: ignore[attr-defined]
            folder_entry.dnd_bind(  # type: ignore[attr-defined]
                "<<Drop>>", lambda e: self.app.mcnp_folder_var.set(e.data.strip("{}"))
            )
        ttk.Button(runner_frame, text="Browse", command=self.browse_mcnp_folder).pack(pady=2)

        ttk.Label(runner_frame, text="Number of Parallel Jobs:").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(runner_frame, from_=1, to=16, textvariable=self.app.mcnp_jobs_var).pack()

        ttk.Button(runner_frame, text="Run Simulations", command=self.run_mcnp_jobs_threaded).pack(pady=10)
        ttk.Button(runner_frame, text="Open Geometry Plotter (single file)", command=self.open_geometry_plotter).pack(pady=2)
        ttk.Button(runner_frame, text="Run Single File (ixr)", command=self.run_single_file_ixr).pack(pady=2)

        self.app.runtime_summary_label = ttk.Label(self.frame, text="")
        self.app.runtime_summary_label.pack(pady=(0, 5))

        paned = ttk.Panedwindow(self.frame, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        output_frame = ttk.LabelFrame(paned, text="Output Console")
        self.runner_output_console = ScrolledText(output_frame, wrap=tk.WORD, height=8)
        self.runner_output_console.pack(fill="both", expand=True)
        paned.add(output_frame, weight=3)

        self.progress_frame = ttk.Frame(paned)
        self.progress_var = tk.DoubleVar()
        self.runner_progress = ttk.Progressbar(
            self.progress_frame, variable=self.progress_var, maximum=100
        )
        self.runner_progress.pack(fill="x", pady=(5, 5))
        self.app.countdown_label = ttk.Label(self.progress_frame, text="")
        self.app.countdown_label.pack(pady=(0, 10))
        paned.add(self.progress_frame, weight=1)

        self.queue_table = ttk.Treeview(
            self.frame, columns=("file", "status"), show="headings", height=6
        )
        self.queue_table.heading("file", text="Input File")
        self.queue_table.heading("status", text="Status")
        self.queue_table.pack(fill="both", expand=False, padx=10, pady=(0, 10))

    # ------------------------------------------------------------------
    def browse_mcnp_folder(self) -> None:
        """Prompt the user for an MCNP input folder and update the setting."""

        path = select_folder("Select Folder with MCNP Input Files")
        if path:
            self.app.mcnp_folder_var.set(path)

    def _set_runner_enabled(self, enabled: bool) -> None:
        """Enable or disable all widgets in the runner tab."""

        state = "normal" if enabled else "disabled"
        for child in self.frame.winfo_children():
            try:
                child.configure(state=state)  # type: ignore[call-arg]
            except Exception:
                for sub in getattr(child, "winfo_children", lambda: [])():
                    try:
                        sub.configure(state=state)  # type: ignore[call-arg]
                    except Exception:
                        pass

    def _reset_after_abort(self) -> None:
        """Re-enable runner controls after an aborted run."""

        self.run_in_progress = False
        # Ensure UI updates occur on the main thread
        self.app.root.after(0, lambda: self._set_runner_enabled(True))

    def _handle_existing_outputs(self, files: List[str], folder: str) -> bool:
        """Check for existing output files and prompt the user for action.

        Parameters
        ----------
        files : list[str]
            Input file names to check.
        folder : str
            Directory where the input files reside.

        Returns
        -------
        bool
            ``True`` if processing should continue, ``False`` otherwise.
        """

        existing_outputs = check_existing_outputs(files, folder)
        if existing_outputs:
            self.app.log("Detected existing output files:")
            for f in existing_outputs:
                self.app.log(f"  {f}")
            response = messagebox.askyesnocancel(
                title="Existing Output Files Found",
                message=(
                    "Output files already exist (suffix o/r/c).\n\n"
                    "Yes = Delete them\nNo = Move them to backup\nCancel = Abort"
                ),
            )
            if response is True:
                delete_or_backup_outputs(existing_outputs, folder, "delete")
            elif response is False:
                delete_or_backup_outputs(existing_outputs, folder, "backup")
            else:
                return False
        return True

    def open_geometry_plotter(self) -> None:
        """Launch the MCNP geometry plotter for a single input file."""

        try:
            file_path = select_file("Select MCNP input file for geometry plotter")
            if not file_path:
                self.app.log("Geometry plotter cancelled.")
                return
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.app.base_dir, file_path)
            if not os.path.exists(file_path):
                self.app.log(f"Selected file does not exist: {file_path}")
                return
            folder = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            if not self._handle_existing_outputs([base_name], folder):
                self.app.log("Aborting geometry plotter launch.")
                return
            run_geometry_plotter(file_path, self.running_processes)
            self.app.log(f"Launching geometry plotter (ip) for: {file_path}")
            self.app.root.lift()
        except Exception as e:
            self.app.log(f"Failed to launch geometry plotter: {e}", logging.ERROR)

    def run_single_file_ixr(self) -> None:
        """Run a single MCNP input file using the ``ixr`` mode."""

        if self.run_in_progress:
            messagebox.showinfo(
                "Run in progress", "A run is already in progress. Please wait before starting another."
            )
            return
        self.run_in_progress = True
        self._set_runner_enabled(False)
        try:
            file_path = select_file("Select MCNP input file to run (ixr)")
            if not file_path:
                self.app.log("Single-file run cancelled.")
                self.run_in_progress = False
                self._set_runner_enabled(True)
                return
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.app.base_dir, file_path)
            if not os.path.exists(file_path):
                self.app.log(f"Selected file does not exist: {file_path}")
                self.run_in_progress = False
                self._set_runner_enabled(True)
                return
            folder = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            if not self._handle_existing_outputs([base_name], folder):
                self.app.log("Aborting single-file run.")
                self.run_in_progress = False
                self._set_runner_enabled(True)
                return
            job = SimulationJob(file_path)
            ctme_value = extract_ctme_minutes(file_path) or 0.0
            if ctme_value <= 0:
                ctme_value = 1.0
            self.app.start_time = datetime.datetime.now()
            self.app.estimated_completion = self.app.start_time + datetime.timedelta(minutes=ctme_value)
            self.app.runtime_summary_label.config(
                text=(
                    f"Estimated completion: {self.app.estimated_completion.strftime('%Y-%m-%d %H:%M:%S')} — "
                    f"{ctme_value:.1f} min ({ctme_value/60:.2f} hr)"
                )
            )
            self.progress_var.set(0)
            self.runner_progress.update_idletasks()
            self.update_countdown = True
            self.app.root.after(1000, self.update_countdown_timer)

            def _runner():
                try:
                    run_mcnp(file_path, self.running_processes)
                    self.app.root.after(0, lambda: self.mark_job_completed(job))
                except Exception as e:
                    self.app.root.after(
                        0, lambda: self.app.log(f"Failed single-file run: {e}", logging.ERROR)
                    )
                    self.app.root.after(0, self.on_run_complete)

            threading.Thread(target=_runner, daemon=True).start()
            self.app.log(f"Launching single-file run (ixr) for: {file_path}")
            self.app.root.lift()
        except Exception as e:
            self.app.log(f"Failed to start single-file run: {e}", logging.ERROR)
            self.run_in_progress = False
            self._set_runner_enabled(True)

    def run_mcnp_jobs_threaded(self) -> None:
        """Launch ``run_mcnp_jobs`` in a background thread."""

        if self.run_in_progress:
            messagebox.showinfo(
                "Run in progress", "A run is already in progress. Please wait before starting another."
            )
            return
        self.run_in_progress = True
        self._set_runner_enabled(False)
        t = threading.Thread(target=self.run_mcnp_jobs)
        t.daemon = True
        t.start()

    def initialize_queue_table(self, jobs: List[SimulationJob]) -> None:
        """Populate the queue table with a list of jobs."""

        self.queue_table.delete(*self.queue_table.get_children())
        for job in jobs:
            self.queue_table.insert("", "end", iid=job.base, values=(job.name, job.status))

    def execute_mcnp_runs(self, inp_files: List[str], jobs: int) -> None:
        """Run MCNP simulations either serially or in parallel.

        Parameters
        ----------
        inp_files : list[str]
            Input file paths to simulate. Currently unused but retained for
            interface compatibility.
        jobs : int
            Number of parallel jobs requested.
        """

        self.running_processes = []
        if len(self.jobs) == 1:
            def run_in_thread(job: SimulationJob) -> None:
                try:
                    run_mcnp(job.filepath, self.running_processes)
                    self.app.root.after(0, lambda: self.mark_job_completed(job))
                except Exception as e:
                    self.app.root.after(0, lambda: self.app.log(f"Run interrupted: {e}"))
                    self.app.root.after(0, self.on_run_complete)

            threading.Thread(
                target=run_in_thread, args=(self.jobs[0],), daemon=True
            ).start()
            return
        else:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=int(self.app.mcnp_jobs_var.get())
            )

        assert self.executor is not None
        future_map = {
            self.executor.submit(run_mcnp, job.filepath, self.running_processes): job
            for job in self.jobs
        }
        self.future_map = future_map
        completed = 0
        total = len(self.future_map)
        try:
            for future in concurrent.futures.as_completed(self.future_map):
                try:
                    future.result()
                    completed += 1
                    percentage = (completed / total) * 100
                    self.progress_var.set(percentage)
                    self.runner_progress.update_idletasks()
                    job = self.future_map[future]
                    job.status = "Completed"
                    self.queue_table.item(job.base, values=(job.name, job.status))
                except Exception:
                    self.app.log("Run interrupted.")
                    self.update_countdown = False
                    self.progress_var.set(0)
                    self.app.countdown_label.config(text="Run interrupted.")
                    return
        finally:
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
            self.future_map = {}
        self.app.root.after(0, self.on_run_complete)

    def mark_job_completed(self, job: SimulationJob) -> None:
        """Update the UI to reflect a completed job."""

        self.progress_var.set(100)
        self.runner_progress.update_idletasks()
        job.status = "Completed"
        self.queue_table.item(job.base, values=(job.name, job.status))
        self.on_run_complete()

    def on_run_complete(self) -> None:
        """Handle completion of all MCNP simulations."""

        self.app.log("All MCNP simulations completed.")
        self.update_countdown = False
        self.progress_var.set(100)
        self.runner_progress.update_idletasks()
        self.run_in_progress = False
        self._set_runner_enabled(True)

    def run_mcnp_jobs(self) -> None:
        """Run MCNP simulations for all input files in a folder."""

        folder = self.app.mcnp_folder_var.get()
        if not folder:
            self.app.log("No folder selected.")
            self._reset_after_abort()
            return
        folder_resolved = folder
        if not os.path.isabs(folder):
            folder_resolved = os.path.join(self.app.base_dir, folder)
        if not validate_input_folder(folder_resolved):
            self.app.log("Invalid or no folder selected.")
            self._reset_after_abort()
            return
        os.chdir(folder_resolved)
        inp_files = gather_input_files(folder_resolved, "folder")
        if not inp_files:
            self.app.log("No MCNP input files found.")
            self._reset_after_abort()
            return
        effective_folder = folder_resolved
        existing_outputs = check_existing_outputs(inp_files, effective_folder)
        if existing_outputs:
            self.app.log("Detected existing output files:")
            for f in existing_outputs:
                self.app.log(f"  {f}")
            response = messagebox.askyesnocancel(
                title="Existing Output Files Found",
                message="Output files already exist.\n\nYes = Delete them\nNo = Move them to backup\nCancel = Abort",
            )
            if response is True:
                delete_or_backup_outputs(existing_outputs, effective_folder, "delete")
            elif response is False:
                delete_or_backup_outputs(existing_outputs, effective_folder, "backup")
            else:
                self.app.log("Aborting run.")
                self._reset_after_abort()
                return
        jobs = int(self.app.mcnp_jobs_var.get())
        ctme_value = extract_ctme_minutes(
            os.path.join(effective_folder, os.path.basename(inp_files[0]))
        )
        estimated_parallel_time = calculate_estimated_time(ctme_value, len(inp_files), jobs)
        completion_time = datetime.datetime.now() + datetime.timedelta(
            minutes=estimated_parallel_time
        )
        self.app.estimated_completion = completion_time
        self.app.start_time = datetime.datetime.now()
        self.app.runtime_summary_label.config(
            text=(
                f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')} — "
                f"{estimated_parallel_time:.1f} min ({estimated_parallel_time / 60:.2f} hr)"
            )
        )
        self.update_countdown = True
        self.app.root.after(1000, self.update_countdown_timer)
        self.progress_var.set(0)
        self.runner_progress.update_idletasks()
        self.jobs = [SimulationJob(os.path.join(effective_folder, os.path.basename(f))) for f in inp_files]
        self.initialize_queue_table(self.jobs)
        self.app.log(f"Running {len(inp_files)} simulations with {jobs} parallel jobs...")
        self.execute_mcnp_runs(inp_files, jobs)

    def update_countdown_timer(self) -> None:
        """Update the progress bar and remaining-time label."""

        if not self.update_countdown:
            return
        now = datetime.datetime.now()
        remaining = self.app.estimated_completion - now
        total_duration = (self.app.estimated_completion - self.app.start_time).total_seconds()
        elapsed = (now - self.app.start_time).total_seconds()
        progress = min(max((elapsed / total_duration) * 100, 0), 100)
        self.progress_var.set(progress)
        self.runner_progress.update_idletasks()
        if remaining.total_seconds() <= 0:
            self.app.countdown_label.config(text="Estimated completion: Done")
            self.update_countdown = False
        else:
            hours, remainder = divmod(int(remaining.total_seconds()), 3600)
            minutes = remainder // 60
            seconds = remainder % 60
            self.app.countdown_label.config(
                text=f"Time remaining: {hours}h {minutes}m {seconds}s"
            )
            self.app.root.after(1000, self.update_countdown_timer)
