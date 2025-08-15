import os
import subprocess
import threading
import logging
import json
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from enum import Enum

import ttkbootstrap as ttk

import He3_Plotter

CONFIG_FILE = "config.json"


class AnalysisType(Enum):
    """Enumeration of supported analysis types."""

    EFFICIENCY_NEUTRON_RATES = 1
    THICKNESS_COMPARISON = 2
    SOURCE_POSITION_ALIGNMENT = 3
    PHOTON_TALLY_PLOT = 4


class AnalysisView:
    """UI and logic for the analysis tab."""

    def __init__(self, app, parent):
        self.app = app
        self.frame = parent

        self.analysis_type = tk.IntVar(
            value=AnalysisType.EFFICIENCY_NEUTRON_RATES.value
        )
        self._analysis_arg_collectors = {
            AnalysisType.EFFICIENCY_NEUTRON_RATES: self._collect_args_type1,
            AnalysisType.THICKNESS_COMPARISON: self._collect_args_type2,
            AnalysisType.SOURCE_POSITION_ALIGNMENT: self._collect_args_type3,
            AnalysisType.PHOTON_TALLY_PLOT: self._collect_args_type4,
        }

        self.build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build(self):
        yield_frame = ttk.LabelFrame(self.frame, text="Neutron Source Selection")
        yield_frame.pack(fill="x", padx=10, pady=5)
        self.source_vars = {
            "Small tank (1.25e6)": tk.BooleanVar(),
            "Big tank (2.5e6)": tk.BooleanVar(),
            "Graphite stack (7.5e6)": tk.BooleanVar(),
        }
        for label, var in self.source_vars.items():
            ttk.Checkbutton(yield_frame, text=label, variable=var).pack(anchor="w", padx=10)

        analysis_frame = ttk.LabelFrame(self.frame, text="Analysis Type")
        analysis_frame.pack(fill="x", padx=10, pady=5)
        self.analysis_type_map = {
            AnalysisType.EFFICIENCY_NEUTRON_RATES: "Efficiency & Neutron Rates",
            AnalysisType.THICKNESS_COMPARISON: "Thickness Comparison",
            AnalysisType.SOURCE_POSITION_ALIGNMENT: "Source Position Alignment",
            AnalysisType.PHOTON_TALLY_PLOT: "Photon Tally Plot",
        }
        self.analysis_type_reverse_map = {
            v: k for k, v in self.analysis_type_map.items()
        }
        self.analysis_combobox = ttk.Combobox(
            analysis_frame,
            values=list(self.analysis_type_map.values()),
            state="readonly",
        )
        self.analysis_combobox.set(
            self.analysis_type_map[AnalysisType.EFFICIENCY_NEUTRON_RATES]
        )
        self.analysis_combobox.pack(padx=10, pady=5)
        self.analysis_combobox.bind("<<ComboboxSelected>>", self.update_analysis_type)

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis_threaded).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Clear Saved Plots", command=self.clear_saved_plots).pack(
            side="left", padx=5
        )
        ttk.Checkbutton(button_frame, text="Save CSVs", variable=self.app.save_csv_var).pack(
            side="left", padx=5
        )

        output_frame = ttk.LabelFrame(self.frame, text="Output Console")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_console = ScrolledText(output_frame, wrap=tk.WORD, height=8)
        self.output_console.pack(fill="both", expand=True)

        file_frame = ttk.LabelFrame(self.frame, text="Saved Plots")
        file_frame.pack(fill="both", expand=False, padx=10, pady=5)
        self.plot_listbox = tk.Listbox(file_frame, height=4)
        self.plot_listbox.pack(fill="both", expand=True)
        self.plot_listbox.bind("<Double-Button-1>", self.open_selected_plot)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def save_config(self):
        config = {
            "neutron_yield": self.app.neutron_yield.get(),
            "analysis_type": self.analysis_type.get(),
            "sources": {label: var.get() for label, var in self.source_vars.items()},
            "run_profile": {
                "jobs": self.app.mcnp_jobs_var.get(),
                "folder": self.app.mcnp_folder_var.get(),
            },
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
        except Exception as e:
            self.app.log(f"Failed to save config: {e}", logging.ERROR)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.app.neutron_yield.set(config.get("neutron_yield", "single"))
                    self.analysis_type.set(
                        config.get(
                            "analysis_type",
                            AnalysisType.EFFICIENCY_NEUTRON_RATES.value,
                        )
                    )
                    for atype, desc in self.analysis_type_map.items():
                        if atype.value == self.analysis_type.get():
                            self.analysis_combobox.set(desc)
                            break
                    sources = config.get("sources", {})
                    for label, var in self.source_vars.items():
                        var.set(sources.get(label, False))
                    run_profile = config.get("run_profile", {})
                    self.app.mcnp_jobs_var.set(run_profile.get("jobs", 3))
                    self.app.mcnp_folder_var.set(run_profile.get("folder", ""))
            except Exception as e:
                self.app.log(f"Failed to load config: {e}", logging.ERROR)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_analysis_type(self, event=None):
        selected_description = self.analysis_combobox.get()
        selected_type = self.analysis_type_reverse_map[selected_description]
        self.analysis_type.set(selected_type.value)

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
                        os.startfile(file_path)  # type: ignore[attr-defined]
                except Exception as e:
                    self.app.log(f"Failed to open file: {e}", logging.ERROR)

    # ------------------------------------------------------------------
    # Argument collection helpers
    # ------------------------------------------------------------------
    def _collect_args_type1(self, yield_value):
        file_path = He3_Plotter.select_file("Select MCNP Output File")
        if not file_path:
            self.app.log("Analysis cancelled.")
            return None
        return (AnalysisType.EFFICIENCY_NEUTRON_RATES, file_path, yield_value)

    def _collect_args_type2(self, yield_value):
        folder_path = He3_Plotter.select_folder("Select Folder with Simulated Data")
        if not folder_path:
            self.app.log("Analysis cancelled.")
            return None
        lab_data_path = He3_Plotter.select_file("Select Experimental Lab Data CSV")
        if not lab_data_path:
            self.app.log("Analysis cancelled.")
            return None
        return (
            AnalysisType.THICKNESS_COMPARISON,
            folder_path,
            lab_data_path,
            yield_value,
        )

    def _collect_args_type3(self, yield_value):
        folder_path = He3_Plotter.select_folder("Select Folder with Simulated Source Position CSVs")
        if not folder_path:
            self.app.log("Analysis cancelled.")
            return None
        return (AnalysisType.SOURCE_POSITION_ALIGNMENT, folder_path, yield_value)

    def _collect_args_type4(self, _=None):
        file_path = He3_Plotter.select_file("Select MCNP Output File for Gamma Analysis")
        if not file_path:
            self.app.log("Analysis cancelled.")
            return None
        return (AnalysisType.PHOTON_TALLY_PLOT, file_path)

    # ------------------------------------------------------------------
    # Analysis execution
    # ------------------------------------------------------------------
    def run_analysis_threaded(self):
        selected_sources = {
            "Small tank (1.25e6)": 1.25e6,
            "Big tank (2.5e6)": 2.5e6,
            "Graphite stack (7.5e6)": 7.5e6,
        }
        yield_value = sum(
            val for label, val in selected_sources.items() if self.source_vars[label].get()
        )
        if yield_value == 0:
            self.app.log("No neutron sources selected. Please select at least one.")
            return

        analysis = AnalysisType(self.analysis_type.get())
        collector = self._analysis_arg_collectors.get(analysis)
        if not collector:
            messagebox.showerror("Error", "Invalid analysis type selected.")
            return

        args = collector(yield_value)
        if not args:
            return

        t = threading.Thread(target=self.process_analysis, args=(args,))
        t.daemon = True
        t.start()

    def process_analysis(self, args):
        self.save_config()
        export_csv = self.app.save_csv_var.get()
        try:
            if args[0] == AnalysisType.EFFICIENCY_NEUTRON_RATES:
                _, file_path, yield_value = args
                He3_Plotter.run_analysis_type_1(
                    file_path, He3_Plotter.AREA, He3_Plotter.VOLUME, yield_value, export_csv
                )
            elif args[0] == AnalysisType.THICKNESS_COMPARISON:
                _, folder_path, lab_data_path, yield_value = args
                He3_Plotter.run_analysis_type_2(
                    folder_path, lab_data_path, He3_Plotter.AREA, He3_Plotter.VOLUME, yield_value, export_csv
                )
            elif args[0] == AnalysisType.SOURCE_POSITION_ALIGNMENT:
                _, folder_path, yield_value = args
                He3_Plotter.run_analysis_type_3(
                    folder_path, He3_Plotter.AREA, He3_Plotter.VOLUME, yield_value, export_csv
                )
            elif args[0] == AnalysisType.PHOTON_TALLY_PLOT:
                _, file_path = args
                He3_Plotter.run_analysis_type_4(file_path, export_csv)
        except Exception as e:
            self.app.log(f"Error during analysis: {e}", logging.ERROR)
