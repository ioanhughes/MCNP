import json
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox

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

        self.build()

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Construct the mesh tally tab widgets."""

        helper_frame = ttk.LabelFrame(self.frame, text="Bin Helper")
        helper_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Grid for IMESH/JMESH/KMESH entries
        labels = [
            ("IMESH", self.imesh_var),
            ("JMESH", self.jmesh_var),
            ("KMESH", self.kmesh_var),
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
        self.output_box.insert("1.0", json.dumps(result, indent=2))
