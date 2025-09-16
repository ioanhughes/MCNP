import os
import subprocess
import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

import ttkbootstrap as ttk

from ..he3_plotter.io_utils import select_file, select_folder
from ..he3_plotter.config import set_filename_tag, set_plot_extension, set_show_fig_heading
from ..he3_plotter.analysis import (
    run_analysis_type_1,
    run_analysis_type_2,
    run_analysis_type_3,
    run_analysis_type_4,
)
from .config_store import JsonConfigStore

CONFIG_FILE = Path(__file__).resolve().parents[3] / "config.json"

# ``tests/test_analysis_config.py`` replaces ``mcnp.he3_plotter`` with a light-weight
# stub, so we provide sensible defaults and load the real detector metadata lazily
# when the full application is running.
DEFAULT_DETECTOR = "He3"
DETECTORS: dict[str, Any] = {}


def _ensure_detectors_loaded() -> None:
    """Load detector metadata if it has not yet been imported."""

    if DETECTORS:
        return
    from ..he3_plotter.detectors import DETECTORS as DETECTOR_MAP, DEFAULT_DETECTOR as DETECTOR_DEFAULT

    DETECTORS.update(DETECTOR_MAP)
    globals()["DEFAULT_DETECTOR"] = DETECTOR_DEFAULT


class AnalysisType(Enum):
    """Enumeration of supported analysis types."""

    EFFICIENCY_NEUTRON_RATES = 1
    THICKNESS_COMPARISON = 2
    SOURCE_POSITION_ALIGNMENT = 3
    PHOTON_TALLY_PLOT = 4


@dataclass(slots=True)
class AnalysisConfigData:
    """Dataclass capturing persistent analysis configuration fields."""

    neutron_yield: str = "single"
    analysis_type: int = AnalysisType.EFFICIENCY_NEUTRON_RATES.value
    sources: dict[str, bool] = field(default_factory=dict)
    custom_enabled: bool = False
    custom_value: str = ""
    run_jobs: int = 3
    run_folder: str = ""
    file_tag: str = ""
    plot_ext: str = "pdf"
    detector: str = DEFAULT_DETECTOR
    show_fig_heading: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise configuration to a JSON-compatible dictionary."""

        return {
            "neutron_yield": self.neutron_yield,
            "analysis_type": self.analysis_type,
            "sources": self.sources,
            "custom_source": {
                "enabled": self.custom_enabled,
                "value": self.custom_value,
            },
            "run_profile": {
                "jobs": self.run_jobs,
                "folder": self.run_folder,
            },
            "file_tag": self.file_tag,
            "plot_ext": self.plot_ext,
            "detector": self.detector,
            "show_fig_heading": self.show_fig_heading,
        }

    @classmethod
    def from_view(cls, view: "AnalysisView") -> "AnalysisConfigData":
        """Create a configuration snapshot from ``view`` widgets."""

        return cls(
            neutron_yield=view.app.neutron_yield.get(),
            analysis_type=view.analysis_type.get(),
            sources={label: var.get() for label, var in view.source_vars.items()},
            custom_enabled=view.custom_var.get(),
            custom_value=view.custom_value_var.get(),
            run_jobs=view.app.mcnp_jobs_var.get(),
            run_folder=view.app.mcnp_folder_var.get(),
            file_tag=view.app.file_tag_var.get(),
            plot_ext=view.app.plot_ext_var.get(),
            detector=view.detector_var.get(),
            show_fig_heading=view.app.show_fig_heading_var.get(),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AnalysisConfigData":
        """Create a configuration instance from ``data``."""

        custom = data.get("custom_source", {})
        run_profile = data.get("run_profile", {})
        return cls(
            neutron_yield=data.get("neutron_yield", "single"),
            analysis_type=data.get(
                "analysis_type", AnalysisType.EFFICIENCY_NEUTRON_RATES.value
            ),
            sources=dict(data.get("sources", {})),
            custom_enabled=custom.get("enabled", False),
            custom_value=custom.get("value", ""),
            run_jobs=run_profile.get("jobs", 3),
            run_folder=run_profile.get("folder", ""),
            file_tag=data.get("file_tag", ""),
            plot_ext=data.get("plot_ext", "pdf"),
            detector=data.get("detector", DEFAULT_DETECTOR),
            show_fig_heading=data.get("show_fig_heading", True),
        )

    def apply_to_view(self, view: "AnalysisView") -> None:
        """Populate ``view`` widgets using stored configuration."""

        view.app.neutron_yield.set(self.neutron_yield)
        view.analysis_type.set(self.analysis_type)
        for atype, description in view.analysis_type_map.items():
            if atype.value == self.analysis_type:
                view.analysis_combobox.set(description)
                break
        for label, var in view.source_vars.items():
            var.set(self.sources.get(label, False))
        view.custom_var.set(self.custom_enabled)
        view.custom_value_var.set(self.custom_value)
        view.app.mcnp_jobs_var.set(self.run_jobs)
        view.app.mcnp_folder_var.set(self.run_folder)
        view.app.file_tag_var.set(self.file_tag)
        view.app.plot_ext_var.set(self.plot_ext)
        view.detector_var.set(self.detector)
        view.detector_combobox.set(self.detector)
        view.app.show_fig_heading_var.set(self.show_fig_heading)
class AnalysisView:
    """UI and logic for the analysis tab."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        """Initialise the analysis view.

        Parameters
        ----------
        app : Any
            Main application instance that holds shared state.
        parent : tk.Widget
            Parent widget into which the view is rendered.
        """

        _ensure_detectors_loaded()
        self.app = app
        self.frame = parent

        self.analysis_type = tk.IntVar(
            value=AnalysisType.EFFICIENCY_NEUTRON_RATES.value
        )
        self.detector_var = tk.StringVar(value=DEFAULT_DETECTOR)
        self._analysis_arg_collectors: dict[
            AnalysisType, Callable[[float], Optional[Tuple[Any, ...]]]
        ] = {
            AnalysisType.EFFICIENCY_NEUTRON_RATES: self._collect_args_type1,
            AnalysisType.THICKNESS_COMPARISON: self._collect_args_type2,
            AnalysisType.SOURCE_POSITION_ALIGNMENT: self._collect_args_type3,
            AnalysisType.PHOTON_TALLY_PLOT: self._collect_args_type4,
        }

        self._executor = ThreadPoolExecutor(max_workers=1)

        self.build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct all UI widgets for the analysis tab."""

        yield_frame = ttk.LabelFrame(self.frame, text="Neutron Source Selection")
        yield_frame.pack(fill="x", padx=10, pady=5)
        self.source_vars = {
            "Small tank (1.25e6)": tk.BooleanVar(),
            "Big tank (2.5e6)": tk.BooleanVar(),
            "Graphite stack (7.5e6)": tk.BooleanVar(),
        }
        for label, var in self.source_vars.items():
            ttk.Checkbutton(yield_frame, text=label, variable=var).pack(anchor="w", padx=10)

        # Custom neutron source
        self.custom_var = tk.BooleanVar()
        self.custom_value_var = tk.StringVar()
        custom_frame = ttk.Frame(yield_frame)
        custom_frame.pack(anchor="w", padx=10)
        ttk.Checkbutton(custom_frame, text="Custom", variable=self.custom_var).pack(side="left")
        vcmd = (self.frame.register(self._validate_float), "%P")
        ttk.Entry(
            custom_frame,
            textvariable=self.custom_value_var,
            width=10,
            validate="key",
            validatecommand=vcmd,
        ).pack(side="left", padx=5)

        _ensure_detectors_loaded()
        detector_frame = ttk.LabelFrame(self.frame, text="Detector")
        detector_frame.pack(fill="x", padx=10, pady=5)
        self.detector_combobox = ttk.Combobox(
            detector_frame,
            values=list(DETECTORS.keys()),
            state="readonly",
            textvariable=self.detector_var,
        )
        self.detector_combobox.pack(padx=10, pady=5)

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
        tag_frame = ttk.Frame(self.frame)
        tag_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(tag_frame, text="File tag:").pack(side="left")
        ttk.Entry(tag_frame, textvariable=self.app.file_tag_var, width=25).pack(
            side="left", padx=5
        )
        ttk.Label(tag_frame, text="File type:").pack(side="left", padx=(10, 0))
        self.file_type_combobox = ttk.Combobox(
            tag_frame,
            values=["pdf", "png"],
            state="readonly",
            textvariable=self.app.plot_ext_var,
            width=5,
        )
        self.file_type_combobox.pack(side="left")

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
        self.output_console = ScrolledText(output_frame, wrap=tk.WORD, height=6)
        self.output_console.pack(fill="both", expand=True)

        file_frame = ttk.LabelFrame(self.frame, text="Saved Plots")
        file_frame.pack(fill="both", expand=False, padx=10, pady=5)
        self.plot_listbox = tk.Listbox(file_frame, height=4)
        self.plot_listbox.pack(fill="both", expand=True)
        self.plot_listbox.bind("<Double-Button-1>", self.open_selected_plot)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def save_config(self) -> None:
        """Persist the current analysis configuration to ``CONFIG_FILE``."""

        store = JsonConfigStore(CONFIG_FILE)
        try:
            store.merge(AnalysisConfigData.from_view(self).to_dict())
        except Exception as e:
            self.app.log(f"Failed to save config: {e}", logging.ERROR)

    def load_config(self) -> None:
        """Load previously saved configuration from ``CONFIG_FILE`` if present."""

        if CONFIG_FILE.exists():
            try:
                data = JsonConfigStore(CONFIG_FILE).load()
                AnalysisConfigData.from_mapping(data).apply_to_view(self)
            except Exception as e:
                self.app.log(f"Failed to load config: {e}", logging.ERROR)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_analysis_type(self, event: Optional[tk.Event] = None) -> None:
        """Update ``analysis_type`` when the combobox selection changes."""

        selected_description = self.analysis_combobox.get()
        selected_type = self.analysis_type_reverse_map[selected_description]
        self.analysis_type.set(selected_type.value)

    def clear_output(self) -> None:
        """Remove all text from the output console widget."""

        self.output_console.delete("1.0", tk.END)

    def clear_saved_plots(self) -> None:
        """Clear the list of saved plot file paths."""

        self.plot_listbox.delete(0, tk.END)

    def open_selected_plot(self, event: tk.Event) -> None:
        """Open the plot file double-clicked by the user."""

        selection = self.plot_listbox.curselection()
        if selection:
            file_path = Path(self.plot_listbox.get(selection[0]))
            if file_path.exists():
                try:
                    if sys.platform.startswith("darwin"):
                        subprocess.run(["open", str(file_path)])
                    elif sys.platform.startswith("linux"):
                        subprocess.run(["xdg-open", str(file_path)])
                    elif sys.platform.startswith("win"):
                        os.startfile(str(file_path))  # type: ignore[attr-defined]
                except Exception as e:
                    self.app.log(f"Failed to open file: {e}", logging.ERROR)

    def _validate_float(self, P: str) -> bool:  # pragma: no cover - UI validation
        if P == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Argument collection helpers
    # ------------------------------------------------------------------
    def _collect_args_type1(
        self, yield_value: float
    ) -> Optional[Tuple[AnalysisType, str, float]]:
        """Collect arguments for analysis type 1.

        Parameters
        ----------
        yield_value : float
            Combined neutron source yield.

        Returns
        -------
        Optional[Tuple[AnalysisType, str, float]]
            Tuple containing the analysis type, selected file path and
            yield value, or ``None`` if cancelled.
        """

        file_path = select_file("Select MCNP Output File")
        if not file_path:
            self.app.log("Analysis cancelled.")
            return None
        return (AnalysisType.EFFICIENCY_NEUTRON_RATES, file_path, yield_value)

    def _collect_args_type2(
        self, yield_value: float
    ) -> Optional[Tuple[AnalysisType, list[str], Optional[str], float]]:
        """Collect arguments for analysis type 2, allowing multiple folders."""

        folder_paths: list[str] = []
        while True:
            folder_path = select_folder("Select Folder with Simulated Data")
            if not folder_path:
                break
            folder_paths.append(folder_path)
            if not messagebox.askyesno("Add Another", "Add another folder?"):
                break
        if not folder_paths:
            self.app.log("Analysis cancelled.")
            return None
        lab_data_path = select_file(
            "Select Experimental Lab Data CSV (Cancel to skip)"
        )
        if not lab_data_path:
            lab_data_path = None
        return (
            AnalysisType.THICKNESS_COMPARISON,
            folder_paths,
            lab_data_path,
            yield_value,
        )

    def _collect_args_type3(
        self, yield_value: float
    ) -> Optional[Tuple[AnalysisType, str, float]]:
        """Collect arguments for analysis type 3."""

        folder_path = select_folder("Select Folder with Simulated Source Position CSVs")
        if not folder_path:
            self.app.log("Analysis cancelled.")
            return None
        return (AnalysisType.SOURCE_POSITION_ALIGNMENT, folder_path, yield_value)

    def _collect_args_type4(
        self, _: Optional[float] = None
    ) -> Optional[Tuple[AnalysisType, str]]:
        """Collect arguments for analysis type 4 (gamma tally)."""

        file_path = select_file("Select MCNP Output File for Gamma Analysis")
        if not file_path:
            self.app.log("Analysis cancelled.")
            return None
        return (AnalysisType.PHOTON_TALLY_PLOT, file_path)


    # ------------------------------------------------------------------
    # Analysis execution
    # ------------------------------------------------------------------
    def run_analysis_threaded(self) -> None:
        """Gather arguments and start the analysis in a background thread."""

        _ensure_detectors_loaded()
        selected_sources = {
            "Small tank (1.25e6)": 1.25e6,
            "Big tank (2.5e6)": 2.5e6,
            "Graphite stack (7.5e6)": 7.5e6,
        }
        yield_value = sum(
            val for label, val in selected_sources.items() if self.source_vars[label].get()
        )
        if self.custom_var.get():
            try:
                yield_value += float(self.custom_value_var.get())
            except ValueError:
                self.app.log("Invalid custom neutron source value.")
                return
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
        detector_name = self.detector_var.get()
        geometry = DETECTORS.get(detector_name, DETECTORS[DEFAULT_DETECTOR])
        args = args + (geometry.area, geometry.volume)

        future = self._executor.submit(self.process_analysis, args)
        future.add_done_callback(self._handle_future_result)

    def _handle_future_result(self, future: Future) -> None:
        def callback() -> None:
            try:
                future.result()
                self.app.log("Analysis complete.")
            except Exception as e:
                self.app.log(f"Error during analysis: {e}", logging.ERROR)

        self.app.root.after(0, callback)

    def process_analysis(self, args: Tuple[Any, ...]) -> None:
        """Execute the chosen analysis using the provided argument tuple.

        Parameters
        ----------
        args : Tuple[Any, ...]
            Argument tuple returned by one of the ``_collect_args_type*``
            methods. The first element is always :class:`AnalysisType`.
        """

        self.save_config()
        export_csv = self.app.save_csv_var.get()
        set_filename_tag(self.app.file_tag_var.get())
        set_plot_extension(self.app.plot_ext_var.get())
        set_show_fig_heading(self.app.show_fig_heading_var.get())
        analysis_type = args[0]
        if analysis_type == AnalysisType.EFFICIENCY_NEUTRON_RATES:
            _, file_path, yield_value, area, volume = args
            run_analysis_type_1(file_path, area, volume, yield_value, export_csv)
        elif analysis_type == AnalysisType.THICKNESS_COMPARISON:
            _, folder_paths, lab_data_path, yield_value, area, volume = args
            run_analysis_type_2(
                folder_paths,
                lab_data_path=lab_data_path,
                area=area,
                volume=volume,
                neutron_yield=yield_value,
                export_csv=export_csv,
            )
        elif analysis_type == AnalysisType.SOURCE_POSITION_ALIGNMENT:
            _, folder_path, yield_value, area, volume = args
            run_analysis_type_3(folder_path, area, volume, yield_value, export_csv)
        elif analysis_type == AnalysisType.PHOTON_TALLY_PLOT:
            _, file_path, area, volume = args
            run_analysis_type_4(file_path, export_csv)
