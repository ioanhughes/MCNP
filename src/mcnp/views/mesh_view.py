import json
import logging
from pathlib import Path
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
from tkinter.scrolledtext import ScrolledText
from typing import Any

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

try:  # Optional dependency for 3-D dose maps
    from vedo import Volume, show
except Exception:  # pragma: no cover - vedo not available
    Volume = None  # type: ignore[assignment]
    show = None  # type: ignore[assignment]

try:  # Optional imports for slice viewer
    import vedo  # pragma: no cover - optional dependency
    from vedo.applications import Slicer3DPlotter
except Exception:  # pragma: no cover - vedo not available
    vedo = None  # type: ignore[assignment]
    Slicer3DPlotter = None  # type: ignore[assignment]

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox

from ..he3_plotter.io_utils import select_file, select_folder
from ..utils import msht_parser
from ..utils.mesh_bins_helper import plan_mesh_from_mesh


CONFIG_FILE = Path(__file__).resolve().parents[3] / "config.json"


AXES_LABELS = {"xTitle": "x (cm)", "yTitle": "y (cm)", "zTitle": "z (cm)"}


class MeshTallyView:
    """UI for mesh tally related tools."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        self.app = app
        self.frame = parent

        # Variables for bin helper inputs (mesh extents)
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
        self.slice_var = tk.StringVar()

        # Scaling for dose colour normalisation (percentile of max dose)
        self.dose_quantile_var = tk.DoubleVar(value=95.0)

        # Toggle for logarithmic dose scaling
        self.log_scale_var = tk.BooleanVar(value=False)

        # Colour map for 3-D dose rendering
        self.cmap_var = tk.StringVar(value="jet")

        self.msht_df: pd.DataFrame | None = None
        self.msht_path: str | None = None
        self.stl_meshes: list[Any] | None = None
        self.stl_folder: str | None = None

        # Display variables for selected file paths
        self.msht_path_var = tk.StringVar(value="MSHT file: None")
        self.stl_folder_var = tk.StringVar(value="STL folder: None")

        # Toggle for interactive 3-D slice viewer
        self.slice_viewer_var = tk.BooleanVar(value=False)

        # Persist slice view selections when changed
        self.axis_var.trace_add("write", lambda *_: self.save_config())
        self.slice_var.trace_add("write", lambda *_: self.save_config())
        self.slice_viewer_var.trace_add("write", lambda *_: self.save_config())

        self.build()
        self.load_config()

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct the mesh tally tab widgets."""

        helper_frame = ttk.LabelFrame(self.frame, text="Bin Helper")
        helper_frame.pack(fill="x", padx=10, pady=10)

        # First row: IMESH/JMESH/KMESH
        entries = [
            ("IMESH", self.imesh_var),
            ("JMESH", self.jmesh_var),
            ("KMESH", self.kmesh_var),
        ]
        for i, (label, var) in enumerate(entries):
            col = i * 2
            ttk.Label(helper_frame, text=label + ":").grid(
                row=0, column=col, sticky="e", padx=5, pady=2
            )
            ttk.Entry(helper_frame, textvariable=var, width=10).grid(
                row=0, column=col + 1, padx=5, pady=2
            )

        # Second row: delta and mode
        ttk.Label(helper_frame, text="delta:").grid(
            row=1, column=0, sticky="e", padx=5, pady=2
        )
        ttk.Entry(helper_frame, textvariable=self.delta_var, width=10).grid(
            row=1, column=1, padx=5, pady=2
        )
        ttk.Label(helper_frame, text="mode:").grid(
            row=1, column=2, sticky="e", padx=5, pady=2
        )
        mode_combo = ttk.Combobox(
            helper_frame,
            values=["uniform", "ratio"],
            state="readonly",
            textvariable=self.mode_var,
            width=10,
        )
        mode_combo.grid(row=1, column=3, padx=5, pady=2)

        # Compute button on same row as delta and mode
        ttk.Button(helper_frame, text="Compute", command=self.compute_bins).grid(
            row=1, column=4, columnspan=2, padx=5, pady=2
        )

        for col in range(6):
            helper_frame.columnconfigure(col, weight=1)

        msht_frame = ttk.Frame(self.frame)
        msht_frame.pack(fill="x", padx=10, pady=(0, 10))

        source_frame = ttk.LabelFrame(msht_frame, text="Source Emission Rate")
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

        button_frame = ttk.Frame(msht_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(button_frame, text="Load MSHT File", command=self.load_msht).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Load STL Files", command=self.load_stl_files).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Save CSV", command=self.save_msht_csv).pack(
            side="left", padx=5
        )
        ttk.Button(
            button_frame, text="Plot Dose Map", command=self.plot_dose_map
        ).pack(side="left", padx=5)
        ttk.Checkbutton(
            button_frame,
            text="Slice Viewer",
            variable=self.slice_viewer_var,
        ).pack(side="left", padx=5)

        # Display currently selected file paths
        ttk.Label(msht_frame, textvariable=self.msht_path_var).pack(
            fill="x", padx=5
        )
        ttk.Label(msht_frame, textvariable=self.stl_folder_var).pack(
            fill="x", padx=5
        )

        # Slider to control dose scaling percentile
        scale_frame = ttk.Frame(msht_frame)
        scale_frame.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(
            scale_frame, text="Log scale", variable=self.log_scale_var
        ).pack(side="right", padx=5)
        self.dose_scale_value = ttk.Label(scale_frame, text="95")
        self.dose_scale_value.pack(side="right", padx=5)
        ttk.Label(scale_frame, text="Dose scale (%):").pack(side="left")
        ttk.Scale(
            scale_frame,
            from_=50,
            to=100,
            orient="horizontal",
            variable=self.dose_quantile_var,
            command=lambda v: self.dose_scale_value.config(text=f"{float(v):.0f}")
        ).pack(side="left", fill="x", expand=True, padx=5)

        cmap_frame = ttk.Frame(msht_frame)
        cmap_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(cmap_frame, text="Colour map:").pack(side="left")
        ttk.Combobox(
            cmap_frame,
            values=["jet", "Spectral", "viridis", "magma"],
            state="readonly",
            textvariable=self.cmap_var,
            width=10,
        ).pack(side="left", padx=5)

        slice_frame = ttk.Frame(msht_frame)
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
        ttk.Label(slice_frame, text="Value:").pack(side="left")
        ttk.Entry(slice_frame, textvariable=self.slice_var, width=10).pack(
            side="left", padx=5
        )
        ttk.Button(
            slice_frame, text="Plot Dose Slice", command=self.plot_dose_slice
        ).pack(side="left", padx=5)

        # Output box for results at bottom of the page
        self.output_box = ScrolledText(self.frame, wrap=tk.WORD, height=5)
        self.output_box.pack(fill="x", padx=10, pady=5)

    # ------------------------------------------------------------------
    def save_config(self) -> None:
        """Persist current source emission selections to ``CONFIG_FILE``."""

        if not hasattr(self, "source_vars") or not hasattr(self, "custom_var"):
            return
        app = getattr(self, "app", None)
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
            else:
                config = {}
            config.update(
                {
                    "sources": {
                        label: var.get() for label, var in self.source_vars.items()
                    },
                    "custom_source": {
                        "enabled": self.custom_var.get(),
                        "value": self.custom_value_var.get(),
                    },
                    "msht_path": getattr(self, "msht_path", None),
                    "stl_folder": getattr(self, "stl_folder", None),
                    "slice_viewer": self.slice_viewer_var.get()
                    if hasattr(self, "slice_viewer_var")
                    else False,
                    "slice_axis": self.axis_var.get()
                    if hasattr(self, "axis_var")
                    else "y",
                    "slice_value": self.slice_var.get()
                    if hasattr(self, "slice_var")
                    else "",
                }
            )
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
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
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                sources = config.get("sources", {})
                for label, var in self.source_vars.items():
                    var.set(sources.get(label, False))
                custom = config.get("custom_source", {})
                self.custom_var.set(custom.get("enabled", False))
                self.custom_value_var.set(custom.get("value", ""))
                self.msht_path = config.get("msht_path")
                if self.msht_path and Path(self.msht_path).is_file():
                    try:
                        self.load_msht(self.msht_path)
                    except Exception:
                        self.msht_path_var.set(f"MSHT file: {self.msht_path}")
                elif self.msht_path:
                    self.msht_path_var.set(f"MSHT file: {self.msht_path}")
                self.stl_folder = config.get("stl_folder")
                if self.stl_folder and Path(self.stl_folder).is_dir():
                    try:
                        self.load_stl_files(self.stl_folder)
                    except Exception:
                        self.stl_folder_var.set(f"STL folder: {self.stl_folder}")
                elif self.stl_folder:
                    self.stl_folder_var.set(f"STL folder: {self.stl_folder}")
                if hasattr(self, "slice_viewer_var"):
                    self.slice_viewer_var.set(config.get("slice_viewer", False))
                if hasattr(self, "axis_var"):
                    self.axis_var.set(config.get("slice_axis", "y"))
                if hasattr(self, "slice_var"):
                    self.slice_var.set(config.get("slice_value", ""))
            except Exception as e:  # pragma: no cover - disk issues
                if app and hasattr(app, "log"):
                    app.log(f"Failed to load config: {e}", logging.ERROR)

    # ------------------------------------------------------------------
    def compute_bins(self) -> None:
        """Compute mesh bins using IMESH/JMESH/KMESH."""

        try:
            imesh = float(self.imesh_var.get())
            jmesh = float(self.jmesh_var.get())
            kmesh = float(self.kmesh_var.get())
            delta = float(self.delta_var.get())
            result = plan_mesh_from_mesh(
                imesh=imesh,
                jmesh=jmesh,
                kmesh=kmesh,
                delta=delta,
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
    def load_msht(self, path: str | None = None) -> None:
        """Load an MSHT file and preview its data."""

        try:
            rate = self._get_total_rate()
            if path is None:
                path = select_file("Select MSHT File")
            if not path:
                return
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

    def load_stl_files(self, folderpath: str | None = None) -> list[Any]:
        """Load all STL files from a folder and store ``vedo`` meshes."""

        if vedo is None:  # pragma: no cover - optional dependency
            self.stl_meshes = []
            return []

        if folderpath is None:
            folderpath = select_folder("Select folder with STL files")
            if not folderpath:
                return []

        try:
            files_in_folder = os.listdir(folderpath)
        except OSError:
            logging.getLogger(__name__).error(
                "Failed to list files in folder %s", folderpath
            )
            return []

        stl_files = [f for f in files_in_folder if f.lower().endswith(".stl")]
        meshes: list[Any] = []
        for file in stl_files:
            full_path = os.path.join(folderpath, file)
            vedo_mesh = (
                vedo.Mesh(full_path).alpha(0.5).c("lightblue").wireframe(False)
            )
            meshes.append(vedo_mesh)

        self.stl_meshes = meshes
        self.stl_folder = folderpath
        self.stl_folder_var.set(f"STL folder: {folderpath}")
        self.output_box.insert(
            "end",
            f"Loaded {len(meshes)} STL file{'s' if len(meshes) != 1 else ''} from: {folderpath}\n",
        )
        for file in stl_files:
            self.output_box.insert("end", f"  {os.path.join(folderpath, file)}\n")
        self.save_config()
        return meshes





    def _build_volume(self) -> tuple[Any, list[Any], str, float, float]:
        """Construct the ``vedo`` volume and meshes for dose mapping."""

        df = self.get_mesh_dataframe()

        quant_var = getattr(self, "dose_quantile_var", None)
        quant = (quant_var.get() / 100) if quant_var else 0.95
        max_dose = df["dose"].quantile(quant)
        if max_dose == 0:
            max_dose = 1
        min_dose = df[df["dose"] > 0]["dose"].min()
        if not pd.notna(min_dose) or min_dose <= 0:
            min_dose = max_dose / 1e6

        xs = np.sort(df["x"].unique())
        ys = np.sort(df["y"].unique())
        zs = np.sort(df["z"].unique())
        nx, ny, nz = len(xs), len(ys), len(zs)

        def _check_uniform(arr: np.ndarray, label: str) -> None:
            if len(arr) > 1:
                diffs = np.diff(arr)
                if not np.allclose(diffs, diffs[0]):
                    Messagebox.show_warning(
                        "Non-uniform mesh spacing",
                        f"{label}-coordinates are not uniformly spaced; using first spacing value.",
                    )

        _check_uniform(xs, "X")
        _check_uniform(ys, "Y")
        _check_uniform(zs, "Z")

        grid = (
            df.pivot_table(index="z", columns=["y", "x"], values="dose")
            .fillna(0.0)
            .to_numpy()
            .reshape(nz, ny, nx)
            .transpose(2, 1, 0)
        )
        grid = np.clip(grid, min_dose, max_dose)

        if self.log_scale_var.get():
            grid = np.log10(grid)
            min_dose = np.log10(min_dose)
            max_dose = np.log10(max_dose)
            bar_title = "Log10 Dose (µSv/h)"
        else:
            bar_title = "Dose (µSv/h)"

        dx = xs[1] - xs[0] if nx > 1 else 1.0
        dy = ys[1] - ys[0] if ny > 1 else 1.0
        dz = zs[1] - zs[0] if nz > 1 else 1.0

        vol = Volume(grid, spacing=(dx, dy, dz), origin=(xs[0], ys[0], zs[0]))
        cmap_name = getattr(self, "cmap_var", None)
        cmap_name = cmap_name.get() if cmap_name else "jet"
        vol.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
        vol.add_scalarbar(title=bar_title)

        if self.stl_meshes is None:
            raise ValueError("No STL files loaded")
        meshes = self.stl_meshes
        return vol, meshes, cmap_name, min_dose, max_dose


    # ------------------------------------------------------------------
    def plot_dose_map(self) -> None:
        """Render a 3-D dose map using ``vedo``."""

        if Volume is None or show is None:  # pragma: no cover - vedo missing
            Messagebox.show_error("Dose Map Error", "Vedo library not available")
            return

        try:
            vol, meshes, cmap_name, min_dose, max_dose = self._build_volume()
        except ValueError as exc:  # pragma: no cover - GUI interaction
            Messagebox.show_error("Dose Map Error", str(exc))
            return

        if self.slice_viewer_var.get():
            if Slicer3DPlotter is None:  # pragma: no cover - optional dependency
                Messagebox.show_error("Dose Map Error", "Slice viewer not available")
                return
            plt = Slicer3DPlotter(vol, axes=AXES_LABELS)
            for mesh in meshes:
                mesh.probe(vol)
                mesh.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
                plt += mesh
            plt.show()
        else:
            plt = show(vol, meshes, axes=AXES_LABELS, interactive=False)
            if hasattr(plt, "interactive"):
                plt.interactive()


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
        self.slice_var.set(f"{nearest_val:g}")
        mask = (df[axis] - nearest_val).abs() < 1e-6
        slice_df = df[mask]
        if slice_df.empty:
            Messagebox.show_error("Dose Slice Error", "No data at specified slice")
            return

        axes = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}
        x_axis, y_axis = axes[axis]

        fig, ax = plt.subplots()
        cmap = plt.cm.jet
        quant_var = getattr(self, "dose_quantile_var", None)
        quant = (quant_var.get() / 100) if quant_var else 0.95
        max_dose = slice_df["dose"].quantile(quant)
        if max_dose == 0:
            max_dose = 1
        min_dose = slice_df[slice_df["dose"] > 0]["dose"].min()
        if not pd.notna(min_dose) or min_dose <= 0:
            min_dose = max_dose / 1e6
        if self.log_scale_var.get():
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
