import logging
from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
from tkinter.scrolledtext import ScrolledText
from typing import Any, Mapping
import threading

import pandas as pd
import numpy as np
import matplotlib

import os
import sys

try:  # Use TkAgg if available for interactive plots
    matplotlib.use("TkAgg")
except Exception:  # pragma: no cover - falls back to default backend
    pass
import matplotlib.pyplot as plt
from matplotlib import colors

from . import vedo_plotter as vp
from .stl_service import StlMeshService
from .vedo_plotter import AXES_LABELS

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox

from ..he3_plotter.io_utils import select_file, select_folder
from ..utils import msht_parser
from ..utils.mesh_bins_helper import plan_mesh_from_mesh
from .config_store import JsonConfigStore


CONFIG_FILE = Path(__file__).resolve().parents[3] / "config.json"


@dataclass(slots=True)
class MeshConfigData:
    """Dataclass representing persisted mesh tally preferences."""

    sources: dict[str, bool] = field(default_factory=dict)
    custom_enabled: bool = False
    custom_value: str = ""
    msht_path: str | None = None
    stl_folder: str | None = None
    slice_viewer: bool | None = None
    slice_axis: str | None = None
    slice_value: Any | None = None
    mesh_subdivision: int | None = None
    volume_sampling: bool | None = None
    dose_scale_enabled: bool | None = None
    dose_scale_quantile: float | None = None
    _present: set[str] = field(default_factory=set, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the configuration into a JSON-compatible structure."""

        data: dict[str, Any] = {
            "sources": self.sources,
            "custom_source": {
                "enabled": self.custom_enabled,
                "value": self.custom_value,
            },
            "msht_path": self.msht_path,
            "stl_folder": self.stl_folder,
        }
        for key in (
            "slice_viewer",
            "slice_axis",
            "slice_value",
            "mesh_subdivision",
            "volume_sampling",
            "dose_scale_enabled",
            "dose_scale_quantile",
        ):
            value = getattr(self, key)
            if key in self._present or value is not None:
                data[key] = value
        return data

    @classmethod
    def from_view(cls, view: "MeshTallyView") -> "MeshConfigData":
        """Capture the mesh tally configuration from the given ``view``."""

        present: set[str] = set()

        def _get_bool(var: Any, default: bool) -> bool:
            try:
                return bool(var.get())
            except Exception:  # pragma: no cover - Tk variable access
                return default

        def _get_value(var: Any, default: Any) -> Any:
            try:
                return var.get()
            except Exception:  # pragma: no cover - Tk variable access
                return default

        slice_viewer: bool | None = None
        if hasattr(view, "slice_viewer_var"):
            present.add("slice_viewer")
            slice_viewer = _get_bool(view.slice_viewer_var, True)

        slice_axis: str | None = None
        if hasattr(view, "axis_var"):
            present.add("slice_axis")
            slice_axis = _get_value(view.axis_var, "y")

        slice_value: Any | None = None
        if hasattr(view, "slice_var"):
            present.add("slice_value")
            slice_value = _get_value(view.slice_var, "")

        mesh_subdivision: int | None = None
        if hasattr(view, "subdivision_var"):
            present.add("mesh_subdivision")
            mesh_subdivision = _get_value(view.subdivision_var, 0)

        volume_sampling: bool | None = None
        if hasattr(view, "volume_sampling_var"):
            present.add("volume_sampling")
            volume_sampling = _get_bool(view.volume_sampling_var, False)

        dose_scale_enabled: bool | None = None
        if hasattr(view, "dose_scale_enabled_var"):
            present.add("dose_scale_enabled")
            dose_scale_enabled = _get_bool(view.dose_scale_enabled_var, True)

        dose_scale_quantile: float | None = None
        if hasattr(view, "dose_quantile_var"):
            present.add("dose_scale_quantile")
            previous = getattr(view, "_dose_scale_previous", None)
            try:
                if previous is not None:
                    dose_scale_quantile = float(previous)
                else:
                    dose_scale_quantile = float(view.dose_quantile_var.get())
            except Exception:  # pragma: no cover - Tk variable access
                dose_scale_quantile = 95.0

        return cls(
            sources={label: var.get() for label, var in view.source_vars.items()},
            custom_enabled=view.custom_var.get(),
            custom_value=view.custom_value_var.get(),
            msht_path=getattr(view, "msht_path", None),
            stl_folder=getattr(view, "stl_folder", None),
            slice_viewer=slice_viewer,
            slice_axis=slice_axis,
            slice_value=slice_value,
            mesh_subdivision=mesh_subdivision,
            volume_sampling=volume_sampling,
            dose_scale_enabled=dose_scale_enabled,
            dose_scale_quantile=dose_scale_quantile,
            _present=present,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MeshConfigData":
        """Create a configuration instance from stored ``data``."""

        custom = data.get("custom_source", {})
        return cls(
            sources=dict(data.get("sources", {})),
            custom_enabled=custom.get("enabled", False),
            custom_value=custom.get("value", ""),
            msht_path=data.get("msht_path"),
            stl_folder=data.get("stl_folder"),
            slice_viewer=data.get("slice_viewer"),
            slice_axis=data.get("slice_axis"),
            slice_value=data.get("slice_value"),
            mesh_subdivision=data.get("mesh_subdivision"),
            volume_sampling=data.get("volume_sampling"),
            dose_scale_enabled=data.get("dose_scale_enabled"),
            dose_scale_quantile=data.get("dose_scale_quantile"),
        )

    def apply_to_view(self, view: "MeshTallyView") -> None:
        """Populate ``view`` widgets using stored configuration data."""

        for label, var in view.source_vars.items():
            var.set(self.sources.get(label, False))
        view.custom_var.set(self.custom_enabled)
        view.custom_value_var.set(self.custom_value)

        view.msht_path = self.msht_path
        if view.msht_path and Path(view.msht_path).is_file():
            try:
                view.load_msht(view.msht_path)
            except Exception:  # pragma: no cover - optional parsing failure
                view.msht_path_var.set(f"MSHT file: {view.msht_path}")
        elif view.msht_path:
            view.msht_path_var.set(f"MSHT file: {view.msht_path}")

        view.stl_folder = self.stl_folder
        if view.stl_folder and Path(view.stl_folder).is_dir():
            try:
                view.load_stl_files(view.stl_folder)
            except Exception:  # pragma: no cover - optional parsing failure
                view.stl_folder_var.set(f"STL folder: {view.stl_folder}")
        elif view.stl_folder:
            view.stl_folder_var.set(f"STL folder: {view.stl_folder}")

        if hasattr(view, "slice_viewer_var"):
            value = self.slice_viewer if self.slice_viewer is not None else True
            view.slice_viewer_var.set(value)

        if hasattr(view, "volume_sampling_var"):
            value = self.volume_sampling if self.volume_sampling is not None else False
            view.volume_sampling_var.set(value)

        if hasattr(view, "axis_var"):
            view.axis_var.set(self.slice_axis if self.slice_axis is not None else "y")

        if hasattr(view, "slice_var"):
            view.slice_var.set(self.slice_value if self.slice_value is not None else "")

        if hasattr(view, "dose_quantile_var"):
            quantile = self.dose_scale_quantile if self.dose_scale_quantile is not None else 95.0
            view.dose_quantile_var.set(quantile)

        if hasattr(view, "dose_scale_enabled_var"):
            value = self.dose_scale_enabled if self.dose_scale_enabled is not None else True
            view.dose_scale_enabled_var.set(value)
            if hasattr(view, "_dose_scale_previous"):
                view._dose_scale_previous = None

        if hasattr(view, "dose_scale"):
            view._update_dose_scale_state()

        if hasattr(view, "subdivision_var"):
            # Always reset subdivision to zero for new sessions.
            view.subdivision_var.set(0)
class MeshTallyView:
    """UI for mesh tally related tools."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        self.app = app
        self.frame = parent

        # Variables for bin helper inputs (origin and mesh extents)
        self.xorigin_var = tk.StringVar()
        self.yorigin_var = tk.StringVar()
        self.zorigin_var = tk.StringVar()
        self.imesh_var = tk.StringVar()
        self.jmesh_var = tk.StringVar()
        self.kmesh_var = tk.StringVar()
        self.delta_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="uniform")

        # Source emission rate selection
        self.source_vars = {
            "Small tank (1.25e6)": tk.BooleanVar(),
            "Big tank (2.5e6)": tk.BooleanVar(),
            "Graphite stack (7.5e6)": tk.BooleanVar(),
        }
        self.custom_var = tk.BooleanVar()
        self.custom_value_var = tk.StringVar()

        # Options for 2-D dose slice plotting
        self.axis_var = tk.StringVar(value="y")
        self.slice_var = tk.DoubleVar()

        # Scaling for dose colour normalisation (percentile of max dose)
        self.dose_quantile_var = tk.DoubleVar(value=95.0)

        # Toggle for enabling/disabling dose scaling
        self.dose_scale_enabled_var = tk.BooleanVar(value=True)
        self._dose_scale_previous: float | None = None
        self.dose_scale_enabled_var.trace_add("write", lambda *_: self.save_config())

        # Toggle for logarithmic dose scaling
        self.log_scale_var = tk.BooleanVar(value=False)

        # Level of subdivision to apply to STL meshes
        self.subdivision_var = tk.IntVar(value=0)

        # Toggle for volume sampling vs surface sampling
        self.volume_sampling_var = tk.BooleanVar(value=False)

        self.msht_df: pd.DataFrame | None = None
        self.msht_path: str | None = None
        self._stl_service = StlMeshService(vp)

        # Display variables for selected file paths
        self.msht_path_var = tk.StringVar(value="MSHT file: None")
        self.stl_folder_var = tk.StringVar(value="STL folder: None")

        # Toggle for interactive 3-D slice viewer
        # Default to the slice viewer so 3-D plots open with interactive slices.
        self.slice_viewer_var = tk.BooleanVar(value=True)

        # Persist slice view selections when changed
        def _axis_changed(*_):
            self.save_config()
            self._update_slice_scale()

        self.axis_var.trace_add("write", _axis_changed)
        self.slice_var.trace_add("write", lambda *_: self.save_config())
        self.slice_viewer_var.trace_add("write", lambda *_: self.save_config())
        self.subdivision_var.trace_add("write", self._on_subdivision_changed)

        self.build()
        self.load_config()

    # ------------------------------------------------------------------
    @property
    def stl_service(self) -> StlMeshService:
        """Expose the STL management service."""

        if not hasattr(self, "_stl_service"):
            self._stl_service = StlMeshService(vp)
        return self._stl_service

    @property
    def stl_folder(self) -> str | None:
        """Compatibility accessor used by configuration helpers."""

        return self._stl_service.stl_folder

    @stl_folder.setter
    def stl_folder(self, value: str | None) -> None:
        self.stl_service.stl_folder = value

    @property
    def stl_files(self) -> list[str] | None:
        """Return filenames associated with the loaded STL meshes."""

        return self.stl_service.stl_files

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct the mesh tally tab widgets."""

        self._build_bin_helper()
        dose_frame = self._build_dose_frame()
        self._build_source_section(dose_frame)
        self._build_msht_section(dose_frame)
        self._build_display_settings_section(dose_frame)
        self._build_stl_section(dose_frame)
        self._build_plot3d_section(dose_frame)
        self._build_slice_controls(dose_frame)
        self._build_output_box()

    def _build_bin_helper(self) -> None:
        helper_frame = ttk.LabelFrame(self.frame, text="Bin Helper")
        helper_frame.pack(fill="x", padx=10, pady=10)

        vcmd = (self.frame.register(self._validate_float), "%P")

        origin_entries = [
            ("X0", self.xorigin_var),
            ("Y0", self.yorigin_var),
            ("Z0", self.zorigin_var),
        ]
        for i, (label, var) in enumerate(origin_entries):
            col = i * 2
            ttk.Label(helper_frame, text=label + ":").grid(
                row=0, column=col, sticky="e", padx=5, pady=2
            )
            ttk.Entry(
                helper_frame,
                textvariable=var,
                width=10,
                validate="key",
                validatecommand=vcmd,
            ).grid(row=0, column=col + 1, padx=5, pady=2)

        mesh_entries = [
            ("IMESH", self.imesh_var),
            ("JMESH", self.jmesh_var),
            ("KMESH", self.kmesh_var),
        ]
        for i, (label, var) in enumerate(mesh_entries):
            col = i * 2
            ttk.Label(helper_frame, text=label + ":").grid(
                row=1, column=col, sticky="e", padx=5, pady=2
            )
            ttk.Entry(
                helper_frame,
                textvariable=var,
                width=10,
                validate="key",
                validatecommand=vcmd,
            ).grid(row=1, column=col + 1, padx=5, pady=2)

        ttk.Label(helper_frame, text="delta:").grid(
            row=2, column=0, sticky="e", padx=5, pady=2
        )
        ttk.Entry(
            helper_frame,
            textvariable=self.delta_var,
            width=10,
            validate="key",
            validatecommand=vcmd,
        ).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(helper_frame, text="mode:").grid(
            row=2, column=2, sticky="e", padx=5, pady=2
        )
        mode_combo = ttk.Combobox(
            helper_frame,
            values=["uniform", "ratio"],
            state="readonly",
            textvariable=self.mode_var,
            width=10,
        )
        mode_combo.grid(row=2, column=3, padx=5, pady=2)

        ttk.Button(helper_frame, text="Compute", command=self.compute_bins).grid(
            row=2, column=4, columnspan=2, padx=5, pady=2
        )

        for col in range(6):
            helper_frame.columnconfigure(col, weight=1)

    def _build_dose_frame(self) -> ttk.LabelFrame:
        dose_frame = ttk.LabelFrame(self.frame, text="Dose Map")
        dose_frame.pack(fill="x", padx=10, pady=(0, 10))
        return dose_frame

    def _build_source_section(self, parent: ttk.LabelFrame) -> None:
        source_frame = ttk.LabelFrame(parent, text="Source Emission Rate")
        source_frame.pack(fill="x", padx=5, pady=5)
        for label, var in self.source_vars.items():
            ttk.Checkbutton(source_frame, text=label, variable=var).pack(
                anchor="w", padx=5
            )
        custom_frame = ttk.Frame(source_frame)
        custom_frame.pack(anchor="w", padx=5)
        ttk.Checkbutton(custom_frame, text="Other", variable=self.custom_var).pack(
            side="left"
        )
        vcmd = (self.frame.register(self._validate_float), "%P")
        ttk.Entry(
            custom_frame,
            textvariable=self.custom_value_var,
            width=10,
            validate="key",
            validatecommand=vcmd,
        ).pack(side="left", padx=5)

    def _build_msht_section(self, parent: ttk.LabelFrame) -> None:
        msht_frame = ttk.LabelFrame(parent, text="MSHT File")
        msht_frame.pack(fill="x", padx=5, pady=5)

        msht_button_frame = ttk.Frame(msht_frame)
        msht_button_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(
            msht_button_frame, text="Load MSHT File", command=self.load_msht
        ).pack(side="left", padx=5)
        ttk.Button(
            msht_button_frame, text="Save CSV", command=self.save_msht_csv
        ).pack(side="left", padx=5)

        ttk.Label(msht_frame, textvariable=self.msht_path_var).pack(
            fill="x", padx=5
        )

    def _build_display_settings_section(self, parent: ttk.LabelFrame) -> None:
        settings_frame = ttk.LabelFrame(parent, text="Display Settings")
        settings_frame.pack(fill="x", padx=5, pady=5)

        scale_frame = ttk.Frame(settings_frame)
        scale_frame.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(
            scale_frame, text="Log scale", variable=self.log_scale_var
        ).pack(side="right", padx=5)
        self.dose_scale_value = ttk.Label(scale_frame, text="95")
        self.dose_scale_value.pack(side="right", padx=5)
        ttk.Checkbutton(
            scale_frame,
            text="Enable dose scale",
            variable=self.dose_scale_enabled_var,
            command=self._update_dose_scale_state,
        ).pack(side="left", padx=5)
        ttk.Label(scale_frame, text="Dose scale (%):").pack(side="left")
        self.dose_scale = ttk.Scale(
            scale_frame,
            from_=50,
            to=100,
            orient="horizontal",
            variable=self.dose_quantile_var,
            command=lambda v: self.dose_scale_value.config(text=f"{float(v):.0f}")
        )
        self.dose_scale.pack(side="left", fill="x", expand=True, padx=5)
        self._update_dose_scale_state()

    def _build_stl_section(self, parent: ttk.LabelFrame) -> None:
        stl_frame = ttk.LabelFrame(parent, text="STL Meshes")
        stl_frame.pack(fill="x", padx=5, pady=5)

        stl_button_frame = ttk.Frame(stl_frame)
        stl_button_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(
            stl_button_frame, text="Load STL Files", command=self.load_stl_files
        ).pack(side="left", padx=5)

        subdiv_frame = ttk.Frame(stl_frame)
        subdiv_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(subdiv_frame, text="Subdivision level:").pack(side="left")
        ttk.Spinbox(
            subdiv_frame,
            from_=0,
            to=10,
            width=5,
            textvariable=self.subdivision_var,
        ).pack(side="left", padx=5)
        ttk.Button(
            subdiv_frame, text="Save STL Files", command=self.save_stl_files
        ).pack(side="left", padx=5)

        ttk.Label(stl_frame, textvariable=self.stl_folder_var).pack(
            fill="x", padx=5
        )

    def _build_plot3d_section(self, parent: ttk.LabelFrame) -> None:
        plot3d_frame = ttk.LabelFrame(parent, text="3D Plot")
        plot3d_frame.pack(fill="x", padx=5, pady=5)

        button_frame = ttk.Frame(plot3d_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(
            button_frame, text="Plot 3D Dose", command=self.plot_dose_map
        ).pack(side="left", padx=5)
        ttk.Checkbutton(
            button_frame,
            text="Slice Viewer",
            variable=self.slice_viewer_var,
        ).pack(side="left", padx=5)
        ttk.Checkbutton(
            button_frame,
            text="Volume sampling",
            variable=self.volume_sampling_var,
        ).pack(side="left", padx=5)

    def _build_slice_controls(self, parent: ttk.LabelFrame) -> None:
        slice_frame = ttk.LabelFrame(parent, text="2D Slice")
        slice_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(slice_frame, text="Slice axis:").pack(side="left")
        axis_combo = ttk.Combobox(
            slice_frame,
            values=["x", "y", "z"],
            state="readonly",
            textvariable=self.axis_var,
            width=5,
        )
        axis_combo.pack(side="left", padx=5)
        self.slice_scale = ttk.Scale(
            slice_frame,
            orient="horizontal",
            variable=self.slice_var,
            from_=0.0,
            to=0.0,
            command=self._on_slice_slider,
        )
        self.slice_scale.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(slice_frame, text="Value:").pack(side="left")
        ttk.Entry(slice_frame, textvariable=self.slice_var, width=10).pack(
            side="left", padx=5
        )
        ttk.Button(
            slice_frame, text="Plot 2D Dose Slice", command=self.plot_dose_slice
        ).pack(side="left", padx=5)

    def _build_output_box(self) -> None:
        self.output_box = ScrolledText(self.frame, wrap=tk.WORD, height=5)
        self.output_box.pack(fill="x", padx=10, pady=5)

    # ------------------------------------------------------------------
    def save_config(self) -> None:
        """Persist current source emission selections to ``CONFIG_FILE``."""

        if not hasattr(self, "source_vars") or not hasattr(self, "custom_var"):
            return
        app = getattr(self, "app", None)
        try:
            payload = MeshConfigData.from_view(self).to_dict()
            JsonConfigStore(CONFIG_FILE).merge(payload)
        except Exception as e:  # pragma: no cover - disk issues
            if app and hasattr(app, "log"):
                app.log(f"Failed to save config: {e}", logging.ERROR)

    # ------------------------------------------------------------------
    def load_config(self) -> None:
        """Load source emission selections from ``CONFIG_FILE`` if present."""

        if not hasattr(self, "source_vars") or not hasattr(self, "custom_var"):
            return
        app = getattr(self, "app", None)
        if CONFIG_FILE.exists():
            try:
                data = JsonConfigStore(CONFIG_FILE).load()
                MeshConfigData.from_mapping(data).apply_to_view(self)
            except Exception as e:  # pragma: no cover - disk issues
                if app and hasattr(app, "log"):
                    app.log(f"Failed to load config: {e}", logging.ERROR)

    # ------------------------------------------------------------------
    def compute_bins(self) -> None:
        """Compute mesh bins using IMESH/JMESH/KMESH."""

        try:
            xorigin = float(self.xorigin_var.get())
            yorigin = float(self.yorigin_var.get())
            zorigin = float(self.zorigin_var.get())
            imesh = float(self.imesh_var.get())
            jmesh = float(self.jmesh_var.get())
            kmesh = float(self.kmesh_var.get())
            delta = float(self.delta_var.get())
            result = plan_mesh_from_mesh(
                imesh=imesh,
                jmesh=jmesh,
                kmesh=kmesh,
                delta=delta,
                xorigin=xorigin,
                yorigin=yorigin,
                zorigin=zorigin,
                mode=self.mode_var.get(),
            )
        except Exception as e:  # pragma: no cover - GUI interaction
            Messagebox.show_error("Bin Helper Error", str(e))
            return

        self.output_box.delete("1.0", tk.END)
        data = result.get("result", {})
        lines = [
            f"iints: {data.get('iints')}",
            f"jints: {data.get('jints')}",
            f"kints: {data.get('kints')}",
            f"delta_x: {data.get('delta_x')}",
            f"delta_y: {data.get('delta_y')}",
            f"delta_z: {data.get('delta_z')}",
        ]
        self.output_box.insert("1.0", "\n".join(lines))

    # ------------------------------------------------------------------
    def _show_progress_dialog(self, message: str):
        """Display a modal progress dialog while background tasks run.

        Returns an object with a ``close`` method that dismisses the dialog.
        """

        parent = self.frame
        win = ttk.Toplevel(parent)
        win.title("Please wait")
        win.transient(parent.winfo_toplevel())
        win.grab_set()
        ttk.Label(win, text=message).pack(padx=20, pady=10)
        bar = ttk.Progressbar(win, mode="indeterminate")
        bar.pack(fill="x", padx=20, pady=(0, 10))
        bar.start()

        class _Dlg:
            def close(self_inner):  # pragma: no cover - simple closure
                try:
                    bar.stop()
                    win.grab_release()
                    win.destroy()
                except Exception:
                    pass

        return _Dlg()

    # ------------------------------------------------------------------
    def load_msht(self, path: str | None = None) -> None:
        """Load an MSHT file and preview its data without blocking UI."""

        app = getattr(self, "app", None)
        root = getattr(app, "root", None)

        try:
            rate = self._get_total_rate()
            if path is None:
                path = select_file("Select MSHT File")
            if not path:
                return
        except Exception as exc:  # pragma: no cover - GUI interaction
            Messagebox.show_error("MSHT Load Error", str(exc))
            return

        if root is None:  # Fallback to synchronous execution
            try:
                df = msht_parser.parse_msht(path)
                df["dose"] = df["result"] * rate * 3600 * 1e6
                df["dose_error"] = df["dose"] * df["rel_error"]
            except Exception as exc:  # pragma: no cover - GUI interaction
                Messagebox.show_error("MSHT Load Error", str(exc))
                return
            self.msht_df = df
            self.msht_path = path
            self.msht_path_var.set(f"MSHT file: {path}")
            self.output_box.delete("1.0", tk.END)
            rows, cols = df.shape
            preview = (
                f"Loaded MSHT file: {path}\n"
                f"DataFrame dimensions: {rows} rows x {cols} columns"
            )
            self.output_box.insert("1.0", preview)
            self.save_config()
            self._update_slice_scale()
            return

        progress = self._show_progress_dialog("Loading MSHT file...")

        def worker() -> None:
            try:
                df = msht_parser.parse_msht(path)
                df["dose"] = df["result"] * rate * 3600 * 1e6
                df["dose_error"] = df["dose"] * df["rel_error"]

                def on_complete() -> None:
                    progress.close()
                    self.msht_df = df
                    self.msht_path = path
                    self.msht_path_var.set(f"MSHT file: {path}")
                    self.output_box.delete("1.0", tk.END)
                    rows, cols = df.shape
                    preview = (
                        f"Loaded MSHT file: {path}\n"
                        f"DataFrame dimensions: {rows} rows x {cols} columns"
                    )
                    self.output_box.insert("1.0", preview)
                    self.save_config()
                    self._update_slice_scale()

                root.after(0, on_complete)
            except Exception as exc:  # pragma: no cover - GUI interaction

                def on_error() -> None:
                    progress.close()
                    Messagebox.show_error("MSHT Load Error", str(exc))

                root.after(0, on_error)

        t = threading.Thread(target=worker, daemon=True)
        self._msht_thread = t
        t.start()

    # ------------------------------------------------------------------
    def get_mesh_dataframe(self) -> pd.DataFrame:
        """Return the parsed mesh tally data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``['x', 'y', 'z', 'result', 'rel_error',
            'volume', 'result_vol', 'dose', 'dose_error']``.

        Raises
        ------
        ValueError
            If no mesh tally has been loaded yet.
        """

        if self.msht_df is None:
            raise ValueError("No MSHT data loaded")
        return self.msht_df

    # ------------------------------------------------------------------
    def _on_subdivision_changed(self, *_: Any) -> None:
        """Handle changes to the STL subdivision level."""

        self.save_config()
        # Subdividing meshes can be expensive, so defer the work until the
        # meshes are explicitly saved.

    def _apply_stl_load_result(
        self, folderpath: str, meshes: list[Any], stl_files: list[str]
    ) -> None:
        """Update internal state and UI after STL meshes are loaded."""

        self.stl_service.update_meshes(folderpath, meshes, stl_files)
        self.stl_folder_var.set(f"STL folder: {folderpath}")
        summary = (
            f"Loaded {len(stl_files)} STL file{'s' if len(stl_files) != 1 else ''} from: {folderpath}\n"
        )
        self.output_box.insert("end", summary)
        for file in stl_files:
            self.output_box.insert("end", f"  {os.path.join(folderpath, file)}\n")
        self.output_box.see("end")
        self.save_config()

    def load_stl_files(self, folderpath: str | None = None) -> None:
        """Load all STL files from a folder and store meshes without blocking."""

        if not self.stl_service.available:  # pragma: no cover - optional dependency
            self.stl_service.clear()
            return

        if folderpath is None:
            folderpath = select_folder("Select folder with STL files")
            if not folderpath:
                return

        app = getattr(self, "app", None)
        root = getattr(app, "root", None)

        if root is None:  # Fallback synchronous execution
            try:
                meshes, stl_files = self.stl_service.read_folder(folderpath)
            except OSError:
                logging.getLogger(__name__).error(
                    "Failed to list files in folder %s", folderpath
                )
                return
            except Exception as exc:  # pragma: no cover - vedo errors
                Messagebox.show_error("STL Load Error", str(exc))
                return
            self._apply_stl_load_result(folderpath, meshes, stl_files)
            return

        progress = self._show_progress_dialog("Loading STL files...")

        def worker() -> None:
            try:
                meshes, stl_files = self.stl_service.read_folder(folderpath)

                def on_complete() -> None:
                    progress.close()
                    self._apply_stl_load_result(folderpath, meshes, stl_files)

                root.after(0, on_complete)
            except Exception as exc:

                def on_error() -> None:
                    progress.close()
                    logging.getLogger(__name__).error(
                        "Failed to list files in folder %s", folderpath
                    )
                    Messagebox.show_error("STL Load Error", str(exc))

                root.after(0, on_error)

        t = threading.Thread(target=worker, daemon=True)
        self._stl_thread = t
        t.start()


    def save_stl_files(self, folderpath: str | None = None) -> None:
        """Save STL meshes with the current subdivision level."""

        if not self.stl_service.available:  # pragma: no cover - optional dependency
            Messagebox.show_error("STL Save Error", "Vedo library not available")
            return
        if not self.stl_service.has_meshes:
            Messagebox.show_error("STL Save Error", "No STL files loaded")
            return

        if folderpath is None:
            folderpath = select_folder("Select folder to save STL files")
            if not folderpath:
                return

        try:
            os.makedirs(folderpath, exist_ok=True)
        except Exception as exc:  # pragma: no cover - disk or vedo errors
            Messagebox.show_error("STL Save Error", str(exc))
            return

        try:
            level = int(self.subdivision_var.get())
        except (AttributeError, TypeError, ValueError):
            level = 0

        try:
            saved = self.stl_service.save_to_folder(folderpath, level)
        except ValueError as exc:
            Messagebox.show_error("STL Save Error", str(exc))
            return
        except RuntimeError as exc:  # pragma: no cover - optional dependency
            Messagebox.show_error("STL Save Error", str(exc))
            return

        self.output_box.insert(
            "end",
            f"Saved {saved} STL file{'s' if saved != 1 else ''} to: {folderpath}\n",
        )
        self.output_box.see("end")


    # ------------------------------------------------------------------
    def _resolve_dose_scaling(
        self, df: pd.DataFrame
    ) -> tuple[float, float, float, bool]:
        """Determine the active quantile, bounds, and log-scale setting.

        Parameters
        ----------
        df:
            DataFrame containing the dose column used for normalisation.
        """

        quant_var = getattr(self, "dose_quantile_var", None)
        enabled_var = getattr(self, "dose_scale_enabled_var", None)

        scale_enabled = True
        if enabled_var is not None:
            try:
                scale_enabled = bool(enabled_var.get())
            except Exception:  # pragma: no cover - Tk variable access issues
                scale_enabled = True

        if scale_enabled:
            quantile = 0.95
            if quant_var is not None:
                try:
                    quantile = float(quant_var.get()) / 100.0
                except Exception:  # pragma: no cover - Tk variable access issues
                    quantile = 0.95
        else:
            quantile = 1.0

        if quantile < 0.0:
            quantile = 0.0
        elif quantile > 1.0:
            quantile = 1.0

        dose_series = df["dose"].astype(float, copy=False)
        try:
            max_dose = float(dose_series.quantile(quantile))
        except Exception:  # pragma: no cover - defensive casting
            max_dose = float("nan")
        if not pd.notna(max_dose) or max_dose <= 0.0:
            max_dose = 1.0

        positive = dose_series[dose_series > 0.0]
        try:
            min_dose = float(positive.min())
        except Exception:  # pragma: no cover - defensive casting
            min_dose = float("nan")
        if not pd.notna(min_dose) or min_dose <= 0.0 or min_dose >= max_dose:
            min_dose = max_dose / 1e6

        log_scale = False
        log_var = getattr(self, "log_scale_var", None)
        if log_var is not None:
            try:
                log_scale = bool(log_var.get())
            except Exception:  # pragma: no cover - Tk variable access issues
                log_scale = False

        return quantile, min_dose, max_dose, log_scale

    # ------------------------------------------------------------------
    def plot_dose_map(self) -> None:
        """Render a 3-D dose map using ``vedo``."""

        if vp.Volume is None or vp.show is None:  # pragma: no cover - vedo missing
            Messagebox.show_error("Dose Map Error", "Vedo library not available")
            return

        stl_meshes = self.stl_service.get_base_meshes()

        try:
            df = self.get_mesh_dataframe()
            quantile, min_dose, max_dose, log_scale = self._resolve_dose_scaling(df)
            vol, meshes, cmap_name, min_dose, max_dose = vp.build_volume(
                df,
                stl_meshes,
                cmap_name="jet",
                dose_quantile=quantile * 100.0,
                min_dose=min_dose,
                max_dose=max_dose,
                log_scale=log_scale,
                warning_cb=Messagebox.show_warning,
                volume_sampling=self.volume_sampling_var.get(),
            )
        except ValueError as exc:  # pragma: no cover - GUI interaction
            Messagebox.show_error("Dose Map Error", str(exc))
            return

        show_dose_map = getattr(self, "_show_dose_map", vp.show_dose_map)
        if not os.environ.get("DISPLAY") and show_dose_map is vp.show_dose_map:
            try:
                Messagebox.show_error(
                    "Dose Map Error",
                    "Display not available for rendering dose map",
                )
            except Exception:
                logging.getLogger(__name__).warning(
                    "Display not available; skipping dose map rendering"
                )
            return

        try:
            show_dose_map(
                vol,
                meshes,
                cmap_name,
                min_dose,
                max_dose,
                slice_viewer=self.slice_viewer_var.get(),
                volume_sampling=self.volume_sampling_var.get(),
                axes=AXES_LABELS,
            )
        except RuntimeError as exc:  # pragma: no cover - optional dependency
            Messagebox.show_error("Dose Map Error", str(exc))


    def _update_dose_scale_state(self) -> None:
        """Enable or disable dose scaling controls and persist preferences."""

        scale = getattr(self, "dose_scale", None)
        value_label = getattr(self, "dose_scale_value", None)
        quant_var = getattr(self, "dose_quantile_var", None)
        enabled_var = getattr(self, "dose_scale_enabled_var", None)
        if scale is None or value_label is None or quant_var is None or enabled_var is None:
            return

        try:
            enabled = bool(enabled_var.get())
        except Exception:
            enabled = True

        if enabled:
            previous = getattr(self, "_dose_scale_previous", None)
            if previous is not None:
                quant_var.set(previous)
            self._dose_scale_previous = None
            try:
                scale.state(["!disabled"])
            except Exception:  # pragma: no cover - widget state differences
                scale.configure(state="normal")
            value = quant_var.get()
        else:
            self._dose_scale_previous = quant_var.get()
            quant_var.set(100.0)
            try:
                scale.state(["disabled"])
            except Exception:  # pragma: no cover - widget state differences
                scale.configure(state="disabled")
            value = 100.0

        try:
            value_label.config(text=f"{float(value):.0f}")
        except Exception:  # pragma: no cover - defensive UI update
            pass


    def _on_slice_slider(self, val: float | str) -> None:
        """Handle updates from the slice selection slider."""

        try:
            self.slice_var.set(float(val))
        except Exception:
            return
        # Don't automatically plot when adjusting the slider; only update the value
        # so the user can choose when to render the slice via the dedicated button.

    def _update_slice_scale(self) -> None:
        """Recompute slider limits based on the selected axis."""

        if self.msht_df is None or not hasattr(self, "slice_scale"):
            return
        axis = self.axis_var.get()
        if axis not in {"x", "y", "z"} or axis not in self.msht_df:
            return
        min_val = float(self.msht_df[axis].min())
        max_val = float(self.msht_df[axis].max())
        self.slice_scale.configure(from_=min_val, to=max_val)
        # Set the slider and variable to the minimum value without triggering a plot
        self.slice_scale.set(min_val)
        self.slice_var.set(min_val)

    def plot_dose_slice(self) -> None:
        """Render a 2-D slice of the dose map."""

        try:
            df = self.get_mesh_dataframe()
        except ValueError as exc:  # pragma: no cover - GUI interaction
            Messagebox.show_error("Dose Slice Error", str(exc))
            return

        try:
            slice_val = float(self.slice_var.get())
        except ValueError:
            Messagebox.show_error("Dose Slice Error", "Invalid slice value")
            return

        axis = self.axis_var.get()
        if axis not in {"x", "y", "z"}:
            Messagebox.show_error("Dose Slice Error", "Invalid axis")
            return

        nearest_idx = (df[axis] - slice_val).abs().idxmin()
        nearest_val = df.loc[nearest_idx, axis]
        self.slice_var.set(nearest_val)
        mask = (df[axis] - nearest_val).abs() < 1e-6
        slice_df = df[mask]
        if slice_df.empty:
            Messagebox.show_error("Dose Slice Error", "No data at specified slice")
            return

        axes = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
        x_axis, y_axis = axes[axis]

        fig, ax = plt.subplots()
        ax.set_title(f"{axis.upper()} Slice at ~{int(round(nearest_val))}")
        cmap = plt.get_cmap("jet")
        _quantile, min_dose, max_dose, log_scale = self._resolve_dose_scaling(slice_df)
        if log_scale:
            norm = colors.LogNorm(vmin=min_dose, vmax=max_dose)
        else:
            norm = colors.Normalize(vmin=min_dose, vmax=max_dose)
        norm_vals = norm(slice_df["dose"].clip(lower=min_dose, upper=max_dose))
        colors_arr = cmap(norm_vals)
        # Display 2-D slices without transparency for a clearer dose map
        ax.scatter(
            slice_df[x_axis],
            slice_df[y_axis],
            c=colors_arr,
            marker="s",
            s=20,
        )
        ax.set_xlabel(x_axis.upper())
        ax.set_ylabel(y_axis.upper())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Dose (µSv/h)")
        plt.show()

    # ------------------------------------------------------------------
    def save_msht_csv(self) -> None:
        """Save the loaded MSHT DataFrame to a CSV file."""

        try:
            df = self.get_mesh_dataframe()
        except ValueError:
            Messagebox.show_error("Save CSV Error", "No MSHT data loaded")
            return
        try:
            path = asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save MSHT Data As",
            )
            if not path:
                return
            df.to_csv(path, index=False)
        except Exception as exc:  # pragma: no cover - GUI interaction
            Messagebox.show_error("Save CSV Error", str(exc))

    # ------------------------------------------------------------------
    def _validate_float(self, P: str) -> bool:  # pragma: no cover - UI validation
        if P == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    def _get_total_rate(self) -> float:
        source_map = {
            "Small tank (1.25e6)": 1.25e6,
            "Big tank (2.5e6)": 2.5e6,
            "Graphite stack (7.5e6)": 7.5e6,
        }
        total_rate = sum(
            val for label, val in source_map.items() if self.source_vars[label].get()
        )
        if self.custom_var.get():
            try:
                total_rate += float(self.custom_value_var.get())
            except ValueError:
                raise ValueError("Invalid custom source value")
        if total_rate == 0:
            raise ValueError("No neutron source emission rate selected")
        return total_rate
