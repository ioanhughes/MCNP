import tkinter as tk
from typing import Any

import ttkbootstrap as ttk


class DoseView:
    """Simple calculator to convert a result value into dose."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        self.app = app
        self.frame = parent

        self.result_var = tk.StringVar()
        self.dose_var = tk.StringVar()
        self.source_vars = {
            "Small tank (1.25e6)": tk.BooleanVar(),
            "Big tank (2.5e6)": tk.BooleanVar(),
            "Graphite stack (7.5e6)": tk.BooleanVar(),
        }
        self.custom_var = tk.BooleanVar()
        self.custom_value_var = tk.StringVar()

        self.build()

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Build the dose calculator widgets."""

        container = ttk.LabelFrame(self.frame, text="Dose Calculator")
        container.pack(fill="x", padx=10, pady=10)

        entry_frame = ttk.Frame(container)
        entry_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(entry_frame, text="Result:").pack(side="left")
        ttk.Entry(entry_frame, textvariable=self.result_var, width=15).pack(
            side="left", padx=5
        )

        source_frame = ttk.LabelFrame(container, text="Source Emission Rate")
        source_frame.pack(fill="x", padx=10, pady=5)
        for label, var in self.source_vars.items():
            ttk.Checkbutton(source_frame, text=label, variable=var).pack(
                anchor="w", padx=10
            )

        custom_frame = ttk.Frame(source_frame)
        custom_frame.pack(anchor="w", padx=10)
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

        button_frame = ttk.Frame(container)
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="Calculate", command=self.calculate_dose).pack(
            side="left", padx=5
        )
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(
            side="left", padx=5
        )

        output_frame = ttk.Frame(container)
        output_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(output_frame, text="Dose (\u00b5Sv/h):").pack(side="left")
        ttk.Entry(output_frame, textvariable=self.dose_var, state="readonly", width=25).pack(
            side="left", padx=5
        )

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset all fields in the calculator."""

        self.result_var.set("")
        self.dose_var.set("")
        for var in self.source_vars.values():
            var.set(False)
        self.custom_var.set(False)
        self.custom_value_var.set("")

    def _validate_float(self, P: str) -> bool:  # pragma: no cover - UI validation
        if P == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    def calculate_dose(self) -> None:
        """Compute dose from the provided result and selected sources."""

        try:
            result_val = float(self.result_var.get())
        except ValueError:
            self.app.log("Invalid result value.")
            return

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
                self.app.log("Invalid custom source value.")
                return
        if total_rate == 0:
            self.app.log("No neutron sources selected. Please select at least one.")
            return

        dose = result_val * total_rate * 3600 * 1e6
        self.dose_var.set(f"{dose:.3e}")
        self.app.log(f"Calculated dose: {self.dose_var.get()} \u00b5Sv/h")
