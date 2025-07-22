

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import json
import os
import sys
import subprocess
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
    def __init__(self, root):
        self.root = root
        self.root.title("MCNP Tools")
        self.root.iconbitmap("/Users/ioanhughes/Documents/PhD/MCNP/Code/icon.icns")
        self.root.geometry("900x650")

        self.neutron_yield = tk.StringVar(value="single")
        self.analysis_type = tk.StringVar(value="1")
        self.plot_listbox = None

        self.runner_output_console = None  # Will be set in build_runner_tab

        # Executor for MCNP jobs
        self.executor = None
        self.future_map = {}

        # Dark mode toggle variable and checkbutton
        self.dark_mode_var = tk.BooleanVar()
        ttk.Checkbutton(self.root, text="Dark Mode", variable=self.dark_mode_var, command=self.toggle_dark_mode).pack(anchor="ne", padx=5, pady=5)

        self.build_interface()
        self.load_config()

        # Redirect stdout to output console, plot listbox, and runner output console
        sys.stdout = TextRedirector(self.output_console, self.plot_listbox, self.runner_output_console)

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
        self.build_runner_tab()

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
        self.save_csv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Save CSVs", variable=self.save_csv_var).pack(side="left", padx=5)

        # Output console frame
        output_frame = ttk.LabelFrame(self.analysis_tab, text="Output Console")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_console = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=8)
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
            "• If output files already exist (e.g. .o or .r), you’ll be prompted to delete, back up, or cancel.\n\n"
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

        help_box = scrolledtext.ScrolledText(self.help_tab, wrap=tk.WORD, height=25)
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
                    print(f"Failed to open file: {e}")

    # Removed preview_selected_plot method (inline plot preview functionality)

    def save_config(self):
        config = {
            "neutron_yield": self.neutron_yield.get(),
            "analysis_type": self.analysis_type.get(),
            "sources": {label: var.get() for label, var in self.source_vars.items()}
        }
        # Save run profile
        config["run_profile"] = {
            "run_mode": self.run_mode_var.get(),
            "jobs": self.mcnp_jobs_var.get(),
            "folder": self.mcnp_folder_var.get()
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Failed to save config: {e}")

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
                    self.run_mode_var.set(run_profile.get("run_mode", "folder"))
                    self.mcnp_jobs_var.set(run_profile.get("jobs", 3))
                    self.mcnp_folder_var.set(run_profile.get("folder", ""))
            except Exception as e:
                print(f"Failed to load config: {e}")

    def run_analysis_threaded(self):
        # Neutron source selection logic
        selected_sources = {
            "Small tank (1.25e6)": 1.25e6,
            "Big tank (2.5e6)": 2.5e6,
            "Graphite stack (7.5e6)": 7.5e6
        }
        yield_value = sum(val for label, val in selected_sources.items() if self.source_vars[label].get())
        if yield_value == 0:
            print("No neutron sources selected. Please select at least one.")
            return
        analysis = self.analysis_type.get()

        # Gather inputs for each analysis
        if analysis == "1":
            file_path = select_file("Select MCNP Output File")
            if not file_path:
                print("Analysis cancelled.")
                return
            args = (1, file_path, yield_value)

        elif analysis == "2":
            folder_path = select_folder("Select Folder with Simulated Data")
            if not folder_path:
                print("Analysis cancelled.")
                return
            lab_data_path = select_file("Select Experimental Lab Data CSV")
            if not lab_data_path:
                print("Analysis cancelled.")
                return
            args = (2, folder_path, lab_data_path, yield_value)

        elif analysis == "3":
            folder_path = select_folder("Select Folder with Simulated Source Position CSVs")
            if not folder_path:
                print("Analysis cancelled.")
                return
            args = (3, folder_path, yield_value)

        elif analysis == "4":
            file_path = select_file("Select MCNP Output File for Gamma Analysis")
            if not file_path:
                print("Analysis cancelled.")
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
            print(f"Error during analysis: {e}")

    def build_runner_tab(self):
        runner_frame = ttk.LabelFrame(self.runner_tab, text="MCNP Simulation Runner")
        runner_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(runner_frame, text="MCNP Input Folder:").pack(anchor="w")
        self.mcnp_folder_var = tk.StringVar()
        folder_entry = ttk.Entry(runner_frame, textvariable=self.mcnp_folder_var, width=50)
        folder_entry.pack(fill="x", pady=2)
        ttk.Button(runner_frame, text="Browse", command=self.browse_mcnp_folder).pack(pady=2)

        # Run mode radio buttons
        self.run_mode_var = tk.StringVar(value="folder")
        run_mode_frame = ttk.LabelFrame(runner_frame, text="Run Mode")
        run_mode_frame.pack(fill="x", pady=(5, 5))
        ttk.Radiobutton(run_mode_frame, text="Folder (all input files)", variable=self.run_mode_var, value="folder").pack(anchor="w", padx=5)
        ttk.Radiobutton(run_mode_frame, text="Single File", variable=self.run_mode_var, value="single").pack(anchor="w", padx=5)

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
        self.runner_output_console = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=8)
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

    def run_mcnp_jobs(self):
        from run_packages import run_mcnp, extract_ctme_minutes
        import glob
        import datetime
        import shutil
        import concurrent.futures
        import os

        folder = self.mcnp_folder_var.get()
        if not folder or not os.path.isdir(folder):
            print("Invalid or no folder selected.")
            return
        os.chdir(folder)

        known_output_suffixes = {"o", "r", "l"}
        inp_files = []
        mode = self.run_mode_var.get()
        if mode == "single":
            single_file = select_file("Select MCNP input file")
            if single_file:
                inp_files = [os.path.basename(single_file)]
                folder = os.path.dirname(single_file)
                self.mcnp_folder_var.set(folder)
                os.chdir(folder)
            else:
                print("No file selected.")
                return
        else:
            inp_files = glob.glob("*.inp")
            inp_files += [
                f for f in glob.glob("*")
                if os.path.isfile(f)
                and os.path.splitext(f)[1] == ""
                and not any(f.endswith(suffix) for suffix in known_output_suffixes)
            ]
        if not inp_files:
            print("No MCNP input files found.")
            return

        ctme_value = extract_ctme_minutes(os.path.join(folder, inp_files[0]))
        num_files = len(inp_files)
        jobs = self.mcnp_jobs_var.get()
        num_batches = (num_files + jobs - 1) // jobs
        estimated_parallel_time = ctme_value * num_batches
        total_ctme = ctme_value * num_files
        print(f"Estimated total run time: {total_ctme:.1f} minutes")
        print(f"Estimated runtime with {jobs} jobs: {estimated_parallel_time:.1f} minutes ({estimated_parallel_time / 60:.2f} hours)")
        completion_time = datetime.datetime.now() + datetime.timedelta(minutes=estimated_parallel_time)
        print(f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Update runtime summary label above progress bar
        self.runtime_summary_label.config(text=f"Estimated completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')} — {estimated_parallel_time:.1f} min ({estimated_parallel_time / 60:.2f} hr)")

        # Check for existing output files and prompt user with messagebox
        existing_outputs = []
        for inp in inp_files:
            base = os.path.splitext(inp)[0]
            for suffix in ("o", "r"):
                out_name = f"{base}{suffix}"
                if os.path.exists(out_name):
                    existing_outputs.append(out_name)
        if existing_outputs:
            print("Detected existing output files:")
            for f in existing_outputs:
                print(f"  {f}")
            response = messagebox.askyesnocancel(
                "Existing Output Files Found",
                "Output files already exist.\n\nYes = Delete them\nNo = Move them to backup\nCancel = Abort"
            )
            if response is True:  # Yes = Delete
                for f in existing_outputs:
                    try:
                        os.remove(f)
                        print(f"Deleted {f}")
                    except Exception as e:
                        print(f"Could not delete {f}: {e}")
            elif response is False:  # No = Move
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(folder, f"backup_outputs_{timestamp}")
                os.makedirs(backup_dir, exist_ok=True)
                for f in existing_outputs:
                    try:
                        shutil.move(f, backup_dir)
                        print(f"Moved {f} to {backup_dir}")
                    except Exception as e:
                        print(f"Could not move {f}: {e}")
            else:  # Cancel = Abort
                print("Aborting run.")
                self.update_countdown = False
                return

        # Start countdown label update (after handling existing outputs)
        self.estimated_completion = datetime.datetime.now() + datetime.timedelta(minutes=estimated_parallel_time)
        self.update_countdown = True
        self.root.after(1000, self.update_countdown_timer)

        # Reset progress bar
        self.progress_var.set(0)
        self.root.update_idletasks()

        # Populate job queue viewer (live table)
        self.queue_table.delete(*self.queue_table.get_children())
        for f in inp_files:
            self.queue_table.insert("", "end", iid=f, values=(f, "Pending"))

        print(f"Running {num_files} simulations with {jobs} parallel jobs...")
        import concurrent.futures
        self.running_processes = []
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=jobs)
        self.future_map = {self.executor.submit(run_mcnp, f, self.running_processes): f for f in inp_files}
        completed = 0
        total = len(self.future_map)
        try:
            for future in concurrent.futures.as_completed(self.future_map):
                try:
                    future.result()
                    completed += 1
                    self.progress_var.set((completed / total) * 100)
                    self.root.update_idletasks()
                    # Update queue table to "Completed"
                    self.queue_table.item(self.future_map[future], values=(self.future_map[future], "Completed"))
                except Exception:
                    print("Run interrupted.")
                    self.update_countdown = False
                    self.progress_var.set(0)
                    self.countdown_label.config(text="Run interrupted.")
                    return
        finally:
            if self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
            self.future_map = {}
        print("All MCNP simulations completed.")
        self.update_countdown = False
        # Completion popup
        messagebox.showinfo("Run Complete", "All MCNP simulations completed successfully.")




    def update_countdown_timer(self):
        import datetime
        if not getattr(self, 'update_countdown', False):
            return
        remaining = self.estimated_completion - datetime.datetime.now()
        if remaining.total_seconds() <= 0:
            self.countdown_label.config(text="Estimated completion: Done")
            self.update_countdown = False
        else:
            hours, remainder = divmod(int(remaining.total_seconds()), 3600)
            minutes = remainder // 60
            self.countdown_label.config(text=f"Time remaining: {hours}h {minutes}m")
            self.root.after(60000, self.update_countdown_timer)

    def toggle_dark_mode(self):
        style = ttk.Style()
        if self.dark_mode_var.get():
            style.theme_use('clam')
            style.configure('.', background='#2e2e2e', foreground='white')
            style.configure('TLabel', background='#2e2e2e', foreground='white')
            style.configure('TFrame', background='#2e2e2e')
            style.configure('TButton', background='#444', foreground='white')
        else:
            style.theme_use('default')

if __name__ == "__main__":
    root = tk.Tk()
    app = He3PlotterApp(root)
    # Force the window to the front on macOS
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.mainloop()