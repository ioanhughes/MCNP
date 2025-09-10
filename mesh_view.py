import json
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox

from mesh_bins_helper import plan_mesh


class MeshTallyView:
    """UI for mesh tally related tools."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        self.app = app
        self.frame = parent

        # Variables for bin helper inputs
        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar()
        self.ymax_var = tk.StringVar()
        self.zmin_var = tk.StringVar()
        self.zmax_var = tk.StringVar()
        self.delta_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="uniform")

        self.build()

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct the mesh tally tab widgets."""

        helper_frame = ttk.LabelFrame(self.frame, text="Bin Helper")
        helper_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Grid for extent entries
        labels = [
            ("xmin", self.xmin_var),
            ("xmax", self.xmax_var),
            ("ymin", self.ymin_var),
            ("ymax", self.ymax_var),
            ("zmin", self.zmin_var),
            ("zmax", self.zmax_var),
            ("delta", self.delta_var),
        ]
        for i, (label, var) in enumerate(labels):
            ttk.Label(helper_frame, text=label+":").grid(row=i, column=0, sticky="e", padx=5, pady=2)
            ttk.Entry(helper_frame, textvariable=var, width=10).grid(row=i, column=1, padx=5, pady=2)

        ttk.Label(helper_frame, text="mode:").grid(row=len(labels), column=0, sticky="e", padx=5, pady=2)
        mode_combo = ttk.Combobox(
            helper_frame,
            values=["uniform", "ratio"],
            state="readonly",
            textvariable=self.mode_var,
            width=10,
        )
        mode_combo.grid(row=len(labels), column=1, padx=5, pady=2)

        ttk.Button(helper_frame, text="Compute", command=self.compute_bins).grid(
            row=len(labels)+1, column=0, columnspan=2, pady=(10, 5)
        )

        self.output_box = ScrolledText(helper_frame, wrap=tk.WORD, height=10)
        self.output_box.grid(row=len(labels)+2, column=0, columnspan=2, pady=5, sticky="nsew")
        helper_frame.rowconfigure(len(labels)+2, weight=1)
        helper_frame.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    def compute_bins(self) -> None:
        """Compute mesh bins using provided extents and update the output box."""

        try:
            xmin = float(self.xmin_var.get())
            xmax = float(self.xmax_var.get())
            ymin = float(self.ymin_var.get())
            ymax = float(self.ymax_var.get())
            zmin = float(self.zmin_var.get())
            zmax = float(self.zmax_var.get())
            delta = float(self.delta_var.get())
            result = plan_mesh(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                zmin=zmin,
                zmax=zmax,
                delta=delta,
                mode=self.mode_var.get(),
            )
        except Exception as e:  # pragma: no cover - GUI interaction
            Messagebox.showerror("Bin Helper Error", str(e))
            return

        self.output_box.delete("1.0", tk.END)
        self.output_box.insert("1.0", json.dumps(result, indent=2))
