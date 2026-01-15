import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from typing import Any

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import logging

from ...utils import config_utils


class SettingsView:
    """User settings tab."""

    def __init__(self, app: Any, parent: tk.Widget) -> None:
        """Create the settings view."""

        self.app = app
        self.frame = parent
        self.mcnp_path_var = tk.StringVar(value=self.app.base_dir)
        self.default_jobs_var = tk.IntVar(value=self.app.mcnp_jobs_var.get())
        self.theme_var = self.app.theme_var

        self.build()

    def build(self) -> None:
        """Construct all widgets for the settings tab."""

        frame = ttk.LabelFrame(self.frame, text="User Preferences")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="MY_MCNP Path:").pack(anchor="w")
        path_entry = ttk.Entry(frame, textvariable=self.mcnp_path_var, state="readonly", width=60)
        path_entry.pack(fill="x", pady=5)
        ttk.Button(frame, text="Change Path", command=self.change_mcnp_path).pack()

        ttk.Label(frame, text="Default Parallel Jobs:").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.default_jobs_var).pack()

        ttk.Checkbutton(frame, text="Save analysis CSVs by default", variable=self.app.save_csv_var).pack(anchor="w", pady=10)

        ttk.Checkbutton(
            frame,
            text="Show plot titles",
            variable=self.app.show_fig_heading_var,
        ).pack(anchor="w")

        ttk.Label(frame, text="Default plot file type:").pack(anchor="w")
        self.plot_ext_combobox = ttk.Combobox(
            frame,
            textvariable=self.app.plot_ext_var,
            state="readonly",
            values=["pdf", "png"],
        )
        self.plot_ext_combobox.pack(fill="x", pady=5)

        figure_frame = ttk.LabelFrame(frame, text="Figure Display")
        figure_frame.pack(fill="x", expand=False, pady=(10, 0))
        ttk.Label(figure_frame, text="Axis label font size:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Spinbox(
            figure_frame,
            from_=6,
            to=48,
            textvariable=self.app.axis_label_fontsize_var,
            width=5,
            wrap=False,
        ).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(figure_frame, text="Tick label font size:").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Spinbox(
            figure_frame,
            from_=6,
            to=48,
            textvariable=self.app.tick_label_fontsize_var,
            width=5,
            wrap=False,
        ).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(figure_frame, text="Legend font size:").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        ttk.Spinbox(
            figure_frame,
            from_=6,
            to=48,
            textvariable=self.app.legend_fontsize_var,
            width=5,
            wrap=False,
        ).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Checkbutton(
            figure_frame,
            text="Show plot legends",
            variable=self.app.show_legend_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(4, 2))
        ttk.Checkbutton(
            figure_frame,
            text="Show plot text boxes",
            variable=self.app.show_text_boxes_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(
            figure_frame,
            text="Show grid on plots",
            variable=self.app.show_grid_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        figure_frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Select Theme:").pack(anchor="w", pady=(10, 0))
        self.theme_combobox = ttk.Combobox(frame, textvariable=self.theme_var, state="readonly")
        self.theme_combobox['values'] = ['flatly', 'darkly', 'superhero', 'cyborg', 'solar', 'vapor']
        self.theme_combobox.pack(fill="x", pady=5)
        self.theme_combobox.bind("<<ComboboxSelected>>", lambda e: self.toggle_theme())

        ttk.Button(frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        ttk.Button(frame, text="Reset Settings", command=self.reset_settings).pack(pady=10)

    # ------------------------------------------------------------------
    def change_mcnp_path(self) -> None:
        """Prompt the user to change the stored MCNP base path."""

        new_path = filedialog.askdirectory(title="Select your MY_MCNP directory")
        if new_path:
            self.app.base_dir = new_path
            self.mcnp_path_var.set(new_path)
            try:
                config_utils.save_settings({"MY_MCNP_PATH": new_path})
                os.environ["MY_MCNP"] = new_path
                try:
                    from ... import run_packages

                    run_packages.BASE_DIR = Path(new_path)
                except Exception:
                    pass
                self.app.log("MY_MCNP path updated.")
            except Exception as e:
                self.app.log(f"Failed to update MY_MCNP path: {e}", logging.ERROR)

    def toggle_theme(self) -> None:
        """Apply the selected ttkbootstrap theme to the application."""

        style = ttk.Style()
        try:
            selected_theme = self.theme_var.get()
            style.theme_use(selected_theme)
        except Exception:
            pass
        self.app.root.update_idletasks()

    def save_settings(self) -> None:
        """Persist current settings to disk and apply them."""

        self.app.mcnp_jobs_var.set(self.default_jobs_var.get())
        self.toggle_theme()
        if hasattr(self.app, "analysis_view"):
            self.app.analysis_view.save_config()
        if hasattr(self.app, "mesh_view"):
            self.app.mesh_view.save_config()
        try:
            settings = {
                "MY_MCNP_PATH": self.app.base_dir,
                "default_jobs": self.default_jobs_var.get(),
                "dark_mode": self.app.dark_mode_var.get(),
                "save_csv": self.app.save_csv_var.get(),
                "neutron_yield": self.app.neutron_yield.get(),
                "theme": self.theme_var.get(),
                "plot_ext": self.app.plot_ext_var.get(),
                "show_fig_heading": self.app.show_fig_heading_var.get(),
                "axis_label_fontsize": self.app.axis_label_fontsize_var.get(),
                "tick_label_fontsize": self.app.tick_label_fontsize_var.get(),
                "legend_fontsize": self.app.legend_fontsize_var.get(),
                "show_legend": self.app.show_legend_var.get(),
                "show_grid": self.app.show_grid_var.get(),
                "show_text_boxes": self.app.show_text_boxes_var.get(),
            }
            config_utils.save_settings(settings)
            self.app.log("Settings saved.")
        except Exception as e:
            self.app.log(f"Failed to save settings: {e}", logging.ERROR)

    def reset_settings(self) -> None:
        """Reset settings to defaults and exit the application."""

        if Messagebox.yesno("Reset Settings", "Are you sure you want to reset all settings to default?"):
            try:
                settings_file = Path(self.app.settings_path)
                if settings_file.exists():
                    settings_file.unlink()
                Messagebox.show_info("Reset Complete", "Settings reset to default. Please restart the application.")
                self.app.root.quit()
            except Exception as e:
                Messagebox.show_error("Error", f"Failed to reset settings: {e}")
