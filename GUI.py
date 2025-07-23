import os
import json
from tkinter import filedialog
import concurrent.futures

# SimulationJob class for job management
class SimulationJob:
    def __init__(self, filepath):
        import os
        self.filepath = filepath
        self.name = os.path.basename(filepath)
        self.base = os.path.splitext(self.name)[0]
        self.status = "Pending"


import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
try:
    import tkinterdnd2 as tkdnd
except ImportError:
    tkdnd = None
from PIL import Image, ImageTk
import threading
import json
import os
import sys
import subprocess
from run_packages import (
    validate_input_folder, gather_input_files, check_existing_outputs, delete_or_backup_outputs,
    calculate_estimated_time, run_simulations_concurrently, is_valid_input_file
)
from He3_Plotter import (
    SINGLE_SOURCE_YIELD,
    THREE_SOURCE_YIELD,
    AREA,
    VOLUME,
    run_analysis_type_1,
    run_analysis_type_2,
    run_analysis_type_3,
    run_analysis_type_4,
    select_file,
    select_folder
)

CONFIG_FILE = "config.json"

# Replacement TextRedirector class
class TextRedirector:
    def __init__(self, main_widget, listbox, secondary_widget=None):
        self.main_widget = main_widget
        self.secondary_widget = secondary_widget
        self.listbox = listbox

    def write(self, string):
        if self.main_widget:
            self.main_widget.insert("end", string)
            self.main_widget.see("end")
        if self.secondary_widget:
            self.secondary_widget.insert("end", string)
            self.secondary_widget.see("end")
        if "Saved:" in string and self.listbox:
            path = string.split("Saved:", 1)[1].strip()
            self.listbox.insert("end", path)

    def flush(self):
        pass

class He3PlotterApp:
    def log(self, message):
        print(message)  # Redirected to output console
    def __init__(self, root):
        self.root = root
        self.root.title("MCNP Tools")
        self.root.geometry("900x650")

        # self.neutron_yield will be set after loading settings below
        self.analysis_type = tk.StringVar(value="1")
        self.plot_listbox = None

        self.runner_output_console = None  # Will be set in build_runner_tab

        # Executor for MCNP jobs
        self.executor = None
        self.future_map = {}

        # --- Dynamic MY_MCNP directory selection ---
        self.settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.base_dir = self.load_mcnp_path()
        if not self.base_dir:
            self.base_dir = filedialog.askdirectory(
                title="Select your MY_MCNP directory",
                initialdir=os.path.expanduser("~/Documents"),
                mustexist=True
            )
            if not self.base_dir:
                Messagebox.showerror(
                    "Missing MY_MCNP Directory",
                    "You must select the folder that contains the MCNP_CODE directory and your simulation folders.\n\n"
                    "This is typically the 'MY_MCNP' folder inside your MCNP installation."
                )
                sys.exit(1)
            else:
                try:
                    with open(self.settings_path, "w") as f:
                        json.dump({"MY_MCNP_PATH": self.base_dir}, f)
                except Exception as e:
                    print(f"Failed to save MY_MCNP path: {e}")

        # Load default_jobs, dark_mode, save_csv, neutron_yield, and theme from saved settings
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r") as f:
                    settings = json.load(f)
                    default_jobs = settings.get("default_jobs", 3)
                    self.default_jobs_var = tk.IntVar(value=default_jobs)
                    self.mcnp_jobs_var = tk.IntVar(value=default_jobs)
                    self.dark_mode_var = tk.BooleanVar(value=settings.get("dark_mode", False))
                    self.save_csv_var = tk.BooleanVar(value=settings.get("save_csv", True))
                    # Set neutron_yield variable using saved value or default to "single"
                    self.neutron_yield = tk.StringVar(value=settings.get("neutron_yield", "single"))
                    # Set theme variable using saved value or default to "flatly"
                    self.theme_var = tk.StringVar(value=settings.get("theme", "flatly"))
            except Exception as e:
                print(f"Failed to load default job settings: {e}")
                self.default_jobs_var = tk.IntVar(value=3)
                self.mcnp_jobs_var = tk.IntVar(value=3)
                self.dark_mode_var = tk.BooleanVar(value=False)
                self.save_csv_var = tk.BooleanVar(value=True)
                self.neutron_yield = tk.StringVar(value="single")
                self.theme_var = tk.StringVar(value="flatly")
        else:
            self.default_jobs_var = tk.IntVar(value=3)
            self.mcnp_jobs_var = tk.IntVar(value=3)
            self.dark_mode_var = tk.BooleanVar(value=False)
            self.save_csv_var = tk.BooleanVar(value=True)
            self.neutron_yield = tk.StringVar(value="single")
            self.theme_var = tk.StringVar(value="flatly")

        # Apply theme on startup
        self.toggle_theme()

        self.build_interface()
        self.load_config()

        # Redirect stdout to output console, plot listbox, and runner output console
        sys.stdout = TextRedirector(self.output_console, self.plot_listbox, self.runner_output_console)

    def load_mcnp_path(self):
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    return json.load(f).get("MY_MCNP_PATH")
        except Exception:
            pass
        # Default fallback
        fallback = os.path.expanduser("~/Documents/PhD/MCNP/MY_MCNP")
        if os.path.exists(fallback):
            return fallback
        return None

    def build_interface(self):
        # Create tabs
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        # Create tab frames
        self.runner_tab = ttk.Frame(self.tabs)
        self.analysis_tab = ttk.Frame(self.tabs)
        self.help_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.runner_tab, text="Run MCNP")
        self.tabs.add(self.analysis_tab, text="Analysis")
        self.tabs.add(self.help_tab, text="How to Use")
        # Add settings tab after help tab
        self.settings_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.settings_tab, text="Settings")
        self.build_runner_tab()
        self.build_settings_tab()

        # Yield frame
        yield_frame = ttk.LabelFrame(self.analysis_tab, text="Neutron Source Selection")
        yield_frame.pack(fill="x", padx=10, pady=5)
        # Replace radiobuttons with checkboxes for neutron sources
        self.source_vars = {
            "Small tank (1.25e6)": tk.BooleanVar(),
            "Big tank (2.5e6)": tk.BooleanVar(),
            "Graphite stack (7.5e6)": tk.BooleanVar()
        }
        for label, var in self.source_vars.items():
            ttk.Checkbutton(yield_frame, text=label, variable=var).pack(anchor="w", padx=10)

        # Analysis type frame
        analysis_frame = ttk.LabelFrame(self.analysis_tab, text="Analysis Type")
        analysis_frame.pack(fill="x", padx=10, pady=5)
        self.analysis_type_map = {
            "Efficiency & Neutron Rates": "1",
            "Thickness Comparison": "2",
            "Source Position Alignment": "3",
            "Photon Tally Plot": "4"
        }
        self.analysis_combobox = ttk.Combobox(
            analysis_frame,
            values=list(self.analysis_type_map.keys()),
            state="readonly"
        )
        self.analysis_combobox.set("Efficiency & Neutron Rates")
        self.analysis_combobox.pack(padx=10, pady=5)
        self.analysis_combobox.bind("<<ComboboxSelected>>", self.update_analysis_type)

        # Button frame
        button_frame = ttk.Frame(self.analysis_tab)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis_threaded).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Saved Plots", command=self.clear_saved_plots).pack(side="left", padx=5)
        # CSV export toggle
        ttk.Checkbutton(button_frame, text="Save CSVs", variable=self.save_csv_var).pack(side="left", padx=5)

        # Output console frame
        output_frame = ttk.LabelFrame(self.analysis_tab, text="Output Console")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_console = ScrolledText(output_frame, wrap=tk.WORD, height=8)
        self.output_console.pack(fill="both", expand=True)

        # File output list frame
        file_frame = ttk.LabelFrame(self.analysis_tab, text="Saved Plots")
        file_frame.pack(fill="both", expand=False, padx=10, pady=5)
        self.plot_listbox = tk.Listbox(file_frame, height=4)
        self.plot_listbox.pack(fill="both", expand=True)
        self.plot_listbox.bind("<Double-Button-1>", self.open_selected_plot)

        # Populate help tab with instructions
        help_label = tk.Label(self.help_tab, text="How to Use MCNP Tools", font=("Arial", 14, "bold"))
        help_label.pack(pady=10)

        help_text = (
            "Welcome to MCNP Tools — your unified environment for running and analysing MCNP simulations.\n\n"
            "========================\n"
            "Running Simulations\n"
            "========================\n"
            "1. Select Folder or Single File\n"
            "• Use the 'Browse' button to choose your MCNP input folder.\n"
            "• Choose between running all files in the folder or a single file.\n\n"
            "2. Configure Settings\n"
            "• Set the number of parallel jobs (simulations that can run at once).\n"
            "• MCNP input files must end in .inp or have no extension.\n\n"
            "3. Run Simulations\n"
            "• Click 'Run Simulations' to start.\n"
            "• The interface will show estimated completion time, progress, countdown, and live queue status.\n"
            "• If output files already exist (e.g. o or r), you’ll be prompted to delete, back up, or cancel.\n\n"
            "========================\n"
            "Analysing Results\n"
            "========================\n"
            "1. Select Neutron Sources\n"
            "• Tick boxes for neutron sources in your setup:\n"
            "  – Small tank (1.25e6 n/s)\n"
            "  – Big tank (2.5e6 n/s)\n"
            "  – Graphite stack (7.5e6 n/s)\n\n"
            "2. Choose Analysis Type\n"
            "• Options:\n"
            "  – Efficiency & Neutron Rates (single output file)\n"
            "  – Thickness Comparison (simulation vs experiment)\n"
            "  – Source Position Alignment (displacement scan)\n"
            "  – Photon Tally Plot (gamma tally from single file)\n\n"
            "3. Run the Analysis\n"
            "• Click 'Run Analysis' to begin.\n"
            "• Saved plots will appear below and in the 'plots' folder.\n\n"
            "File Naming Tips:\n"
            "• Thickness Comparison: use endings like '_10cmo', '_5cmo'.\n"
            "• Source Alignment: use displacement-style names like '5_0cm'.\n\n"
            "========================\n"
            "CSV Output (if enabled)\n"
            "========================\n"
            "• CSVs will save in a separate 'csvs' folder.\n"
            "• Type 1: neutron & photon tallies + summary.\n"
            "• Type 2: simulated vs experimental data.\n"
            "• Type 3: displacement vs detected rate.\n"
            "• Type 4: photon tally energy spectrum.\n"
        )

        help_box = ScrolledText(self.help_tab, wrap=tk.WORD, height=25)
        help_box.insert("1.0", help_text)
        help_box.configure(state="disabled")
        help_box.pack(fill="both", expand=True, padx=10, pady=10)

    def update_analysis_type(self, event=None):
        selected_description = self.analysis_combobox.get()
        self.analysis_type.set(self.analysis_type_map[selected_description])

    def clear_output(self):
        self.output_console.delete("1.0", tk.END)

    def clear_saved_plots(self):
        self.plot_listbox.delete(0, tk.END)

    def open_selected_plot(self, event):
        selection = self.plot_listbox.curselection()
        if selection:
            file_path = self.plot_listbox.get(selection[0])
            if os.path.exists(file_path):
                try:
                    if sys.platform.startswith("darwin"):
                        subprocess.run(["open", file_path])
                    elif sys.platform.startswith("linux"):
                        subprocess.run(["xdg-open", file_path])
                    elif sys.platform.startswith("win"):
                        os.startfile(file_path)
                except Exception as e:
                    self.log(f"Failed to open file: {e}")

    # Removed preview_selected_plot method (inline plot preview functionality)

    def save_config(self):
        config = {
            "neutron_yield": self.neutron_yield.get(),
            "analysis_type": self.analysis_type.get(),
            "sources": {label: var.get() for label, var in self.source_vars.items()}
        }
        # Save run profile (without run_mode)
        config["run_profile"] = {
            "jobs": self.mcnp_jobs_var.get(),
            "folder": self.mcnp_folder_var.get()
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
        except Exception as e:
            self.log(f"Failed to save config: {e}")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.neutron_yield.set(config.get("neutron_yield", "single"))
                    self.analysis_type.set(config.get("analysis_type", "1"))
                    # Update the combobox to reflect the loaded analysis_type
                    for desc, val in self.analysis_type_map.items():
                        if val == self.analysis_type.get():
                            self.analysis_combobox.set(desc)
                            break
                    # Restore neutron sources selection
                    sources = config.get("sources", {})
                    for label, var in self.source_vars.items():
                        var.set(sources.get(label, False))
                    # Restore run profile if present
                    run_profile = config.get("run_profile", {})
                    self.mcnp_jobs_var.set(run_profile.get("jobs", 3))
                    self.mcnp_folder_var.set(run_profile.get("folder", ""))
            except Exception as e:
                self.log(f"Failed to load config: {e}")

    def run_analysis_threaded(self):
        # Neutron source selection logic
        selected_sources = {
            "Small tank (1.25e6)": 1.25e6,
            "Big tank (2.5e6)": 2.5e6,
            "Graphite stack (7.5e6)": 7.5e6
        }
        yield_value = sum(val for label, val in selected_sources.items() if self.source_vars[label].get())
        if yield_value == 0:
            self.log("No neutron sources selected. Please select at least one.")
            return
        analysis = self.analysis_type.get()

        # Gather inputs for each analysis
        if analysis == "1":
            file_path = select_file("Select MCNP Output File")
            if not file_path:
                self.log("Analysis cancelled.")
                return
            args = (1, file_path, yield_value)

        elif analysis == "2":
            folder_path = select_folder("Select Folder with Simulated Data")
            if not folder_path:
                self.log("Analysis cancelled.")
                return
            lab_data_path = select_file("Select Experimental Lab Data CSV")
            if not lab_data_path:
                self.log("Analysis cancelled.")
                return
            args = (2, folder_path, lab_data_path, yield_value)

        elif analysis == "3":
            folder_path = select_folder("Select Folder with Simulated Source Position CSVs")
            if not folder_path:
                self.log("Analysis cancelled.")
                return
            args = (3, folder_path, yield_value)

        elif analysis == "4":
            file_path = select_file("Select MCNP Output File for Gamma Analysis")
            if not file_path:
                self.log("Analysis cancelled.")
                return
            args = (4, file_path)

        else:
            messagebox.showerror("Error", "Invalid analysis type selected.")
            return

        # Start background processing
        t = threading.Thread(target=self.process_analysis, args=(args,))
        t.daemon = True
        t.start()

    def process_analysis(self, args):
        self.save_config()
        # Apply CSV export preference
        import He3_Plotter
        He3_Plotter.EXPORT_CSV = self.save_csv_var.get()
        try:
            if args[0] == 1:
                _, file_path, yield_value = args
                run_analysis_type_1(file_path, AREA, VOLUME, yield_value)
            elif args[0] == 2:
                _, folder_path, lab_data_path, yield_value = args
                run_analysis_type_2(folder_path, lab_data_path, AREA, VOLUME, yield_value)
            elif args[0] == 3:
                _, folder_path, yield_value = args
                run_analysis_type_3(folder_path, AREA, VOLUME, yield_value)
            elif args[0] == 4:
                _, file_path = args
                run_analysis_type_4(file_path)
        except Exception as e:
            self.log(f"Error during analysis: {e}")

    def build_runner_tab(self):
        runner_frame = ttk.LabelFrame(self.runner_tab, text="MCNP Simulation Runner")
        runner_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(runner_frame, text="MCNP Input Folder:").pack(anchor="w")
        self.mcnp_folder_var = tk.StringVar()
        # Use tk.Entry for drag-and-drop support if possible
        folder_entry = tk.Entry(runner_frame, textvariable=self.mcnp_folder_var, width=50)
        folder_entry.pack(fill="x", pady=2)
        if tkdnd:
            folder_entry.drop_target_register(tkdnd.DND_FILES)
            folder_entry.dnd_bind("<<Drop>>", lambda e: self.mcnp_folder_var.set(e.data.strip("{}")))
        ttk.Button(runner_frame, text="Browse", command=self.browse_mcnp_folder).pack(pady=2)

        ttk.Label(runner_frame, text="Number of Parallel Jobs:").pack(anchor="w", pady=(10, 0))
        self.mcnp_jobs_var = tk.IntVar(value=3)
        ttk.Spinbox(runner_frame, from_=1, to=16, textvariable=self.mcnp_jobs_var).pack()

        ttk.Button(runner_frame, text="Run Simulations", command=self.run_mcnp_jobs_threaded).pack(pady=10)

        # Estimated runtime summary label above progress bar
        self.runtime_summary_label = ttk.Label(self.runner_tab, text="")
        self.runtime_summary_label.pack(pady=(0, 5))

        # Resizable output console and progress bar in a paned window
        paned = ttk.Panedwindow(self.runner_tab, orient=tk.VERTICAL)
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        output_frame = ttk.LabelFrame(paned, text="Output Console")
        self.runner_output_console = ScrolledText(output_frame, wrap=tk.WORD, height=8)
        self.runner_output_console.pack(fill="both", expand=True)
        paned.add(output_frame, weight=3)

        self.progress_frame = ttk.Frame(paned)
        self.progress_var = tk.DoubleVar()
        self.runner_progress = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.runner_progress.pack(fill="x", pady=(5, 5))
        self.countdown_label = ttk.Label(self.progress_frame, text="")
        self.countdown_label.pack(pady=(0, 10))
        paned.add(self.progress_frame, weight=1)

        # Job queue viewer (live table)
        self.queue_table = ttk.Treeview(self.runner_tab, columns=("file", "status"), show="headings", height=6)
        self.queue_table.heading("file", text="Input File")
        self.queue_table.heading("status", text="Status")
        self.queue_table.pack(fill="both", expand=False, padx=10, pady=(0, 10))

    def browse_mcnp_folder(self):
        path = select_folder("Select Folder with MCNP Input Files")
        if path:
            self.mcnp_folder_var.set(path)

    def run_mcnp_jobs_threaded(self):
        t = threading.Thread(target=self.run_mcnp_jobs)
        t.daemon = True
        t.start()


    # Removed: calculate_estimated_time method; now handled via run_packages helper

    def initialize_queue_table(self, jobs):
        self.queue_table.delete(*self.queue_table.get_children())
        for job in jobs:
            self.queue_table.insert("", "end", iid=job.base, values=(job.name, job.status))

    def execute_mcnp_runs(self, inp_files, jobs):
        import concurrent.futures
        from run_packages import run_mcnp
        import threading
        self.running_processes = []
        # Use threading for single job to avoid UI freezing; else ProcessPoolExecutor
        if len(self.jobs) == 1:
            def run_in_thread(job):
                try:
                    run_mcnp(job.filepath, self.running_processes)
                    self.root.after(0, lambda: self.mark_job_completed(job))
                except Exception as e:
                    self.root.after(0, lambda: self.log(f"Run interrupted: {e}"))
                    self.root.after(0, self.on_run_complete)

            threading.Thread(target=run_in_thread, args=(self.jobs[0],), daemon=True).start()
            return
        else:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)

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
                    self.log("Run interrupted.")
                    self.update_countdown = False
                    self.progress_var.set(0)
                    self.countdown_label.config(text="Run interrupted.")
                    return
        finally:
            if self.executor:
                # cancel_futures only supported for ProcessPoolExecutor, but safe to pass
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
            self.future_map = {}

        # All MCNP simulations are complete; finalize in main thread
        self.root.after(0, self.on_run_complete)


    def mark_job_completed(self, job):
        self.progress_var.set(100)
        self.runner_progress.update_idletasks()
        job.status = "Completed"
        self.queue_table.item(job.base, values=(job.name, job.status))
        self.on_run_complete()


    def on_run_complete(self):
        self.log("All MCNP simulations completed.")
        self.update_countdown = False
        self.progress_var.set(100)
        self.runner_progress.update_idletasks()
        self.countdown_label.config(text="Completed")
        Messagebox.showinfo("Run Complete", "All MCNP simulations completed successfully.")

    def run_mcnp_jobs(self):
        import os
        import datetime
        from run_packages import extract_ctme_minutes
        folder = self.mcnp_folder_var.get()
        # Always resolve the folder as a subdirectory of self.base_dir unless it's already absolute
        folder_resolved = folder
        if not os.path.isabs(folder):
            folder_resolved = os.path.join(self.base_dir, folder)
        if not validate_input_folder(folder_resolved):
            self.log("Invalid or no folder selected.")
            return

        # Only folder mode is supported now
        mode = "folder"
        inp_files = gather_input_files(folder_resolved, mode)
        if not inp_files:
            self.log("No MCNP input files found.")
            return

        effective_folder = folder_resolved
        existing_outputs = check_existing_outputs(inp_files, effective_folder)
        if existing_outputs:
            self.log("Detected existing output files:")
            for f in existing_outputs:
                self.log(f"  {f}")
            response = Messagebox.askyesnocancel(
                "Existing Output Files Found",
                "Output files already exist.\n\nYes = Delete them\nNo = Move them to backup\nCancel = Abort"
            )
            if response is True:
                delete_or_backup_outputs(existing_outputs, effective_folder, "delete")
            elif response is False:
                delete_or_backup_outputs(existing_outputs, effective_folder, "backup")
            else:
                self.log("Aborting run.")
                return

        jobs = int(self.mcnp_jobs_var.get())
        ctme_value = extract_ctme_minutes(os.path.join(effective_folder, os.path.basename(inp_files[0])))
        estimated_parallel_time = calculate_estimated_time(ctme_value, len(inp_files), jobs)
        import datetime
        completion_time = datetime.datetime.now() + datetime.timedelta(minutes=estimated_parallel_time)
        self.estimated_completion = completion_time
        self.start_time = datetime.datetime.now()
        self.runtime_summary_label.config(
            text=f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')} — {estimated_parallel_time:.1f} min ({estimated_parallel_time / 60:.2f} hr)"
        )

        self.update_countdown = True
        self.root.after(1000, self.update_countdown_timer)

        self.progress_var.set(0)
        self.runner_progress.update_idletasks()
        # Create SimulationJob objects and initialize table
        self.jobs = [SimulationJob(os.path.join(effective_folder, os.path.basename(f))) for f in inp_files]
        self.initialize_queue_table(self.jobs)
        self.log(f"Running {len(inp_files)} simulations with {jobs} parallel jobs...")
        self.execute_mcnp_runs(inp_files, jobs)




    def update_countdown_timer(self):
        import datetime
        if not getattr(self, 'update_countdown', False):
            return
        now = datetime.datetime.now()
        remaining = self.estimated_completion - now
        total_duration = (self.estimated_completion - self.start_time).total_seconds()
        elapsed = (now - self.start_time).total_seconds()
        progress = min(max((elapsed / total_duration) * 100, 0), 100)
        self.progress_var.set(progress)
        self.runner_progress.update_idletasks()

        if remaining.total_seconds() <= 0:
            self.countdown_label.config(text="Estimated completion: Done")
            self.update_countdown = False
        else:
            hours, remainder = divmod(int(remaining.total_seconds()), 3600)
            minutes = (remainder // 60)
            seconds = remainder % 60
            self.countdown_label.config(text=f"Time remaining: {hours}h {minutes}m {seconds}s")
            self.root.after(1000, self.update_countdown_timer)

    def toggle_theme(self):
        style = ttk.Style()
        try:
            selected_theme = self.theme_var.get()
            style.theme_use(selected_theme)
        except Exception:
            pass
        self.root.update_idletasks()

    def build_settings_tab(self):
        frame = ttk.LabelFrame(self.settings_tab, text="User Preferences")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # MY_MCNP path
        ttk.Label(frame, text="MY_MCNP Path:").pack(anchor="w")
        self.mcnp_path_var = tk.StringVar(value=self.base_dir)
        path_entry = ttk.Entry(frame, textvariable=self.mcnp_path_var, state="readonly", width=60)
        path_entry.pack(fill="x", pady=5)
        ttk.Button(frame, text="Change Path", command=self.change_mcnp_path).pack()

        # Default parallel jobs
        ttk.Label(frame, text="Default Parallel Jobs:").pack(anchor="w", pady=(10, 0))
        self.default_jobs_var = tk.IntVar(value=self.mcnp_jobs_var.get())
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.default_jobs_var).pack()

        # Save CSVs by default
        ttk.Checkbutton(frame, text="Save analysis CSVs by default", variable=self.save_csv_var).pack(anchor="w", pady=10)

        # Theme selection dropdown
        ttk.Label(frame, text="Select Theme:").pack(anchor="w", pady=(10, 0))
        self.theme_var = getattr(self, "theme_var", tk.StringVar(value="flatly"))
        self.theme_combobox = ttk.Combobox(frame, textvariable=self.theme_var, state="readonly")
        self.theme_combobox['values'] = ['flatly', 'darkly', 'superhero', 'cyborg', 'solar', 'vapor']
        self.theme_combobox.pack(fill="x", pady=5)
        self.theme_combobox.bind("<<ComboboxSelected>>", lambda e: self.toggle_theme())

        # Save button
        ttk.Button(frame, text="Save Settings", command=self.save_settings).pack(pady=10)

        # Reset Settings button
        ttk.Button(frame, text="Reset Settings", command=self.reset_settings).pack(pady=10)


    def change_mcnp_path(self):
        new_path = filedialog.askdirectory(title="Select your MY_MCNP directory")
        if new_path:
            self.base_dir = new_path
            self.mcnp_path_var.set(new_path)
            try:
                with open(self.settings_path, "w") as f:
                    json.dump({"MY_MCNP_PATH": new_path}, f)
                self.log("MY_MCNP path updated.")
            except Exception as e:
                self.log(f"Failed to update MY_MCNP path: {e}")

    def save_settings(self):
        # Update main job variable
        self.mcnp_jobs_var.set(self.default_jobs_var.get())
        self.toggle_theme()
        self.save_config()
        try:
            settings = {
                "MY_MCNP_PATH": self.base_dir,
                "default_jobs": self.default_jobs_var.get(),
                "dark_mode": self.dark_mode_var.get(),
                "save_csv": self.save_csv_var.get(),
                "neutron_yield": self.neutron_yield.get(),
                "theme": self.theme_var.get()
            }
            with open(self.settings_path, "w") as f:
                json.dump(settings, f)
            self.log("Settings saved.")
        except Exception as e:
            self.log(f"Failed to save settings: {e}")

    def reset_settings(self):
        if Messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to default?"):
            try:
                if os.path.exists(self.settings_path):
                    os.remove(self.settings_path)
                Messagebox.showinfo("Reset Complete", "Settings reset to default. Please restart the application.")
                self.root.quit()
            except Exception as e:
                Messagebox.showerror("Error", f"Failed to reset settings: {e}")

if __name__ == "__main__":
    if tkdnd:
        root = tkdnd.TkinterDnD.Tk()
    else:
        root = ttk.Window(themename="flatly")  # Modern look
    app = He3PlotterApp(root)
    # Force the window to the front on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.mainloop()