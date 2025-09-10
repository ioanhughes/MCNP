import tkinter as tk
from tkinter.filedialog import asksaveasfilename
from tkinter.scrolledtext import ScrolledText
from typing import Any

import pandas as pd
import matplotlib

try:  # Use TkAgg if available for interactive plots
    matplotlib.use("TkAgg")
except Exception:  # pragma: no cover - falls back to default backend
    pass
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D proj

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox

from he3_plotter.io_utils import select_file
import msht_parser
from mesh_bins_helper import plan_mesh_from_mesh


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

        self.msht_df: pd.DataFrame | None = None

        self.build()

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
        ttk.Button(button_frame, text="Save CSV", command=self.save_msht_csv).pack(
            side="left", padx=5
        )
        ttk.Button(
            button_frame, text="Plot Dose Map", command=self.plot_dose_map
        ).pack(side="left", padx=5)

        # Output box for results at bottom of the page
        self.output_box = ScrolledText(self.frame, wrap=tk.WORD, height=5)
        self.output_box.pack(fill="x", padx=10, pady=5)

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
    def load_msht(self) -> None:
        """Load an MSHT file and preview its data."""

        try:
            rate = self._get_total_rate()
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
        self.output_box.delete("1.0", tk.END)
        rows, cols = df.shape
        preview = f"DataFrame dimensions: {rows} rows x {cols} columns"
        self.output_box.insert("1.0", preview)

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
    def plot_dose_map(self) -> None:
        """Render a 3-D scatter plot coloured by dose."""

        try:
            df = self.get_mesh_dataframe()
        except ValueError as exc:  # pragma: no cover - GUI interaction
            Messagebox.show_error("Dose Map Error", str(exc))
            return

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        max_dose = df["dose"].max()
        norm = colors.Normalize(vmin=0, vmax=max_dose if max_dose != 0 else 1)
        sc = ax.scatter(
            df["x"],
            df["y"],
            df["z"],
            c=df["dose"],
            cmap="viridis",
            norm=norm,
            marker="s",
            s=20,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig.colorbar(sc, label="Dose (µSv/h)")
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
