import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox
import logging


class SettingsView:
    """User settings tab."""

    def __init__(self, app, parent):
        self.app = app
        self.frame = parent
        self.mcnp_path_var = tk.StringVar(value=self.app.base_dir)
        self.default_jobs_var = tk.IntVar(value=self.app.mcnp_jobs_var.get())
        self.theme_var = self.app.theme_var

        self.build()

    def build(self):
        frame = ttk.LabelFrame(self.frame, text="User Preferences")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="MY_MCNP Path:").pack(anchor="w")
        path_entry = ttk.Entry(frame, textvariable=self.mcnp_path_var, state="readonly", width=60)
        path_entry.pack(fill="x", pady=5)
        ttk.Button(frame, text="Change Path", command=self.change_mcnp_path).pack()

        ttk.Label(frame, text="Default Parallel Jobs:").pack(anchor="w", pady=(10, 0))
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.default_jobs_var).pack()

        ttk.Checkbutton(frame, text="Save analysis CSVs by default", variable=self.app.save_csv_var).pack(anchor="w", pady=10)

        ttk.Label(frame, text="Default plot file type:").pack(anchor="w")
        self.plot_ext_combobox = ttk.Combobox(
            frame,
            textvariable=self.app.plot_ext_var,
            state="readonly",
            values=["pdf", "png"],
        )
        self.plot_ext_combobox.pack(fill="x", pady=5)

        ttk.Label(frame, text="Select Theme:").pack(anchor="w", pady=(10, 0))
        self.theme_combobox = ttk.Combobox(frame, textvariable=self.theme_var, state="readonly")
        self.theme_combobox['values'] = ['flatly', 'darkly', 'superhero', 'cyborg', 'solar', 'vapor']
        self.theme_combobox.pack(fill="x", pady=5)
        self.theme_combobox.bind("<<ComboboxSelected>>", lambda e: self.toggle_theme())

        ttk.Button(frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        ttk.Button(frame, text="Reset Settings", command=self.reset_settings).pack(pady=10)

    # ------------------------------------------------------------------
    def change_mcnp_path(self):
        new_path = filedialog.askdirectory(title="Select your MY_MCNP directory")
        if new_path:
            self.app.base_dir = new_path
            self.mcnp_path_var.set(new_path)
            try:
                with open(self.app.settings_path, "w") as f:
                    json.dump({"MY_MCNP_PATH": new_path}, f)
                self.app.log("MY_MCNP path updated.")
            except Exception as e:
                self.app.log(f"Failed to update MY_MCNP path: {e}", logging.ERROR)

    def toggle_theme(self):
        style = ttk.Style()
        try:
            selected_theme = self.theme_var.get()
            style.theme_use(selected_theme)
        except Exception:
            pass
        self.app.root.update_idletasks()

    def save_settings(self):
        self.app.mcnp_jobs_var.set(self.default_jobs_var.get())
        self.toggle_theme()
        if hasattr(self.app, "analysis_view"):
            self.app.analysis_view.save_config()
        try:
            settings = {
                "MY_MCNP_PATH": self.app.base_dir,
                "default_jobs": self.default_jobs_var.get(),
                "dark_mode": self.app.dark_mode_var.get(),
                "save_csv": self.app.save_csv_var.get(),
                "neutron_yield": self.app.neutron_yield.get(),
                "theme": self.theme_var.get(),
                "plot_ext": self.app.plot_ext_var.get(),
            }
            with open(self.app.settings_path, "w") as f:
                json.dump(settings, f)
            self.app.log("Settings saved.")
        except Exception as e:
            self.app.log(f"Failed to save settings: {e}", logging.ERROR)

    def reset_settings(self):
        if Messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to default?"):
            try:
                settings_file = Path(self.app.settings_path)
                if settings_file.exists():
                    settings_file.unlink()
                Messagebox.showinfo("Reset Complete", "Settings reset to default. Please restart the application.")
                self.app.root.quit()
            except Exception as e:
                Messagebox.showerror("Error", f"Failed to reset settings: {e}")
