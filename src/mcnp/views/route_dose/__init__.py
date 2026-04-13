import logging
from pathlib import Path
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import ttkbootstrap as ttk

from ...he3_plotter.io_utils import select_file
from ...route_dose import (
    format_axis_label,
    load_route_csv,
    plot_route_csv,
    selectable_columns,
    suggest_x_column,
    suggest_y_columns,
)


class RouteDoseView:
    """GUI view for plotting route-dose CSV outputs."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        self.app = app
        self.frame = parent
        self.csv_path_var = tk.StringVar()
        self.x_var = tk.StringVar()
        self.show_relative_uncertainty_var = tk.BooleanVar(value=True)
        self.current_dataframe = None
        self.current_columns: list[str] = []
        self.y_column_vars: dict[str, tk.BooleanVar] = {}

        self.build()

    def build(self) -> None:
        """Construct all widgets for the route-dose tab."""

        file_frame = ttk.LabelFrame(self.frame, text="Route Dose CSV")
        file_frame.pack(fill="x", padx=10, pady=10)
        ttk.Entry(
            file_frame,
            textvariable=self.csv_path_var,
            state="readonly",
            width=80,
        ).pack(side="left", fill="x", expand=True, padx=10, pady=10)
        ttk.Button(file_frame, text="Browse CSV", command=self.load_csv).pack(
            side="left",
            padx=(0, 10),
            pady=10,
        )

        axes_frame = ttk.LabelFrame(self.frame, text="Plot Selection")
        axes_frame.pack(fill="both", expand=True, padx=10, pady=5)

        x_frame = ttk.Frame(axes_frame)
        x_frame.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(x_frame, text="X-axis:").pack(side="left")
        self.x_combobox = ttk.Combobox(
            x_frame,
            textvariable=self.x_var,
            state="readonly",
            values=[],
        )
        self.x_combobox.pack(side="left", fill="x", expand=True, padx=(10, 0))

        toggle_frame = ttk.Frame(axes_frame)
        toggle_frame.pack(fill="x", padx=10, pady=(0, 5))
        ttk.Checkbutton(
            toggle_frame,
            text="Show relative uncertainty subplot",
            variable=self.show_relative_uncertainty_var,
        ).pack(anchor="w")

        y_frame = ttk.Frame(axes_frame)
        y_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        ttk.Label(y_frame, text="Y-axis columns:").pack(anchor="w")
        self.y_checkbox_frame = ttk.Frame(y_frame)
        self.y_checkbox_frame.pack(fill="both", expand=True, pady=(5, 0))

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(button_frame, text="Plot", command=self.plot_selected).pack(
            side="left",
            padx=(0, 5),
        )
        ttk.Button(button_frame, text="Clear", command=self.clear).pack(side="left")

        status_frame = ttk.LabelFrame(self.frame, text="Status")
        status_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        self.status_console = ScrolledText(status_frame, wrap=tk.WORD, height=8)
        self.status_console.pack(fill="both", expand=True, padx=10, pady=10)

    def append_status(self, message: str, level: int = logging.INFO) -> None:
        """Append a status message locally and to the shared app logger."""

        self.status_console.insert("end", message + "\n")
        self.status_console.see("end")
        self.app.log(message, level)

    def clear_status(self) -> None:
        """Clear the status console."""

        self.status_console.delete("1.0", tk.END)

    def clear(self) -> None:
        """Reset the loaded CSV and current selections."""

        self.csv_path_var.set("")
        self.x_var.set("")
        self.current_dataframe = None
        self.current_columns = []
        self.x_combobox.configure(values=[])
        self.clear_y_checkboxes()
        self.clear_status()

    def load_csv(self) -> None:
        """Prompt for a CSV file and populate plot controls."""

        file_path = select_file("Select route dose CSV")
        self.load_csv_path(file_path)

    def load_csv_path(self, file_path: str | None) -> None:
        """Load a CSV from a specific path."""

        if not file_path:
            self.append_status("No CSV selected.", logging.WARNING)
            return

        try:
            df = load_route_csv(file_path)
        except Exception as exc:
            self.append_status(f"Failed to load CSV: {exc}", logging.ERROR)
            return

        self.current_dataframe = df
        self.current_columns = list(df.columns)
        self.csv_path_var.set(file_path)
        self.refresh_column_controls()
        self.append_status(
            f"Loaded {Path(file_path).name} with {len(df)} row(s) and {len(df.columns)} column(s)."
        )

    def clear_y_checkboxes(self) -> None:
        """Remove existing Y-axis checkbox widgets and state."""

        self.y_column_vars = {}
        if hasattr(self, "y_checkbox_frame"):
            for child in self.y_checkbox_frame.winfo_children():
                child.destroy()

    def refresh_column_controls(self) -> None:
        """Rebuild axis selectors for the current dataframe."""

        if self.current_dataframe is None:
            return

        columns = selectable_columns(self.current_dataframe)
        if not columns:
            columns = list(self.current_dataframe.columns)

        self.x_combobox.configure(values=columns)
        current_x = self.x_var.get()
        suggested_x = suggest_x_column(self.current_dataframe)
        if current_x in columns:
            self.x_var.set(current_x)
        elif suggested_x is not None and suggested_x in columns:
            self.x_var.set(suggested_x)
        elif columns:
            self.x_var.set(columns[0])
        else:
            self.x_var.set("")

        previous_selection = {
            column for column, var in self.y_column_vars.items() if var.get()
        }
        suggested_columns = set(suggest_y_columns(self.current_dataframe))
        self.clear_y_checkboxes()
        for column in columns:
            selected = column in previous_selection or (
                not previous_selection and column in suggested_columns
            )
            var = tk.BooleanVar(value=selected)
            self.y_column_vars[column] = var
            ttk.Checkbutton(
                self.y_checkbox_frame,
                text=format_axis_label(column),
                variable=var,
            ).pack(anchor="w")

    def get_selected_y_columns(self) -> list[str]:
        """Return the currently selected Y-axis columns."""

        return [
            column
            for column, var in self.y_column_vars.items()
            if var.get()
        ]

    def plot_selected(self) -> None:
        """Plot the currently loaded CSV using the selected axes."""

        if self.current_dataframe is None:
            self.append_status("Load a route dose CSV before plotting.", logging.WARNING)
            return

        y_columns = self.get_selected_y_columns()
        if not y_columns:
            self.append_status("Select at least one Y-axis column.", logging.WARNING)
            return

        x_column = self.x_var.get() or None
        try:
            plot_kind = plot_route_csv(
                self.current_dataframe,
                x_column,
                y_columns,
                self.csv_path_var.get(),
                show_relative_uncertainty=self.show_relative_uncertainty_var.get(),
            )
        except Exception as exc:
            self.append_status(f"Plot failed: {exc}", logging.ERROR)
            return

        self.append_status(f"Displayed {plot_kind} plot for {Path(self.csv_path_var.get()).name}.")


__all__ = ["RouteDoseView"]
