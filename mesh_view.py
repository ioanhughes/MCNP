import tkinter as tk
from tkinter.filedialog import asksaveasfilename
from tkinter.scrolledtext import ScrolledText
from typing import Any

import pandas as pd

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

        self.msht_df: pd.DataFrame | None = None

        self.build()

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct the mesh tally tab widgets."""

        helper_frame = ttk.LabelFrame(self.frame, text="Bin Helper")
        helper_frame.pack(fill="both", expand=True, padx=10, pady=10)

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

        # Output box for results
        self.output_box = ScrolledText(helper_frame, wrap=tk.WORD, height=10)
        self.output_box.grid(row=2, column=0, columnspan=6, pady=5, sticky="nsew")
        helper_frame.rowconfigure(2, weight=1)
        for col in range(6):
            helper_frame.columnconfigure(col, weight=1)

        msht_frame = ttk.Frame(self.frame)
        msht_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(msht_frame, text="Load MSHT File", command=self.load_msht).pack(
            side="left", padx=5
        )
        ttk.Button(msht_frame, text="Save CSV", command=self.save_msht_csv).pack(
            side="left", padx=5
        )

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
            Messagebox.showerror("Bin Helper Error", str(e))
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
            path = select_file("Select MSHT File")
            if not path:
                return
            df = msht_parser.parse_msht(path)
        except Exception as exc:  # pragma: no cover - GUI interaction
            Messagebox.showerror("MSHT Load Error", str(exc))
            return

        self.msht_df = df
        self.output_box.delete("1.0", tk.END)
        try:
            preview = df.head().to_string(index=False)
        except Exception as exc:  # pragma: no cover - defensive
            Messagebox.showerror("MSHT Preview Error", str(exc))
            preview = ""
        self.output_box.insert("1.0", preview)

    # ------------------------------------------------------------------
    def get_mesh_dataframe(self) -> pd.DataFrame:
        """Return the parsed mesh tally data.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``['x', 'y', 'z', 'result', 'rel_error',
            'volume', 'result_vol']``.

        Raises
        ------
        ValueError
            If no mesh tally has been loaded yet.
        """

        if self.msht_df is None:
            raise ValueError("No MSHT data loaded")
        return self.msht_df

    # ------------------------------------------------------------------
    def save_msht_csv(self) -> None:
        """Save the loaded MSHT DataFrame to a CSV file."""

        try:
            df = self.get_mesh_dataframe()
        except ValueError:
            Messagebox.showerror("Save CSV Error", "No MSHT data loaded")
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
            Messagebox.showerror("Save CSV Error", str(exc))
