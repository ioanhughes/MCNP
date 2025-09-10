import json
import os
import sys
import logging
import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox

try:
    import tkinterdnd2 as tkdnd
except ImportError:  # pragma: no cover - optional dependency
    tkdnd = None

from analysis_view import AnalysisView
from runner_view import RunnerView
from settings_view import SettingsView
from mesh_view import MeshTallyView
import logging_config

# Module-level logger for this module
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom logging handler to write to GUI widgets
# ---------------------------------------------------------------------------
class WidgetLoggerHandler(logging.Handler):
    def __init__(self, main_widget, listbox, secondary_widget=None):
        super().__init__()
        self.main_widget = main_widget
        self.secondary_widget = secondary_widget
        self.listbox = listbox

    def emit(self, record):
        """Emit a log record and update GUI widgets safely.

        Tkinter widgets are not thread-safe, yet analyses are executed in a
        background thread.  Any direct interaction with the widgets from that
        thread can therefore fail silently or raise errors.  To avoid this we
        schedule updates via each widget's ``after`` method so that the actual
        GUI manipulation happens on Tk's main thread.
        """

        message = self.format(record)

        if self.main_widget:
            def append_main() -> None:
                self.main_widget.insert("end", message + "\n")
                self.main_widget.see("end")

            self.main_widget.after(0, append_main)

        if self.secondary_widget:
            def append_secondary() -> None:
                self.secondary_widget.insert("end", message + "\n")
                self.secondary_widget.see("end")

            self.secondary_widget.after(0, append_secondary)

        if "Saved:" in message and self.listbox:
            path = message.split("Saved:", 1)[1].strip()

            def append_listbox() -> None:
                self.listbox.insert("end", path)

            self.listbox.after(0, append_listbox)


class He3PlotterApp:
    """Main application coordinating the various views."""

    def log(self, message, level=logging.INFO):
        """Log messages using the module-level logger."""
        logger.log(level, message)

    def __init__(self, root):
        self.root = root
        self.root.title("MCNP Tools")
        self.root.geometry("900x650")
        self.tkdnd = tkdnd

        # Paths and settings
        self.settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.base_dir = self.load_mcnp_path()
        if not self.base_dir:
            self.base_dir = filedialog.askdirectory(
                title="Select your MY_MCNP directory",
                initialdir=os.path.expanduser("~/Documents"),
                mustexist=True,
            )
            if not self.base_dir:
                Messagebox.showerror(
                    "Missing MY_MCNP Directory",
                    (
                        "You must select the folder that contains the MCNP_CODE directory and your simulation folders.\n\n"
                        "This is typically the 'MY_MCNP' folder inside your MCNP installation."
                    ),
                )
                sys.exit(1)
            else:
                try:
                    with open(self.settings_path, "w") as f:
                        json.dump({"MY_MCNP_PATH": self.base_dir}, f)
                except Exception:
                    pass

        # Load persisted user settings
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r") as f:
                    settings = json.load(f)
                default_jobs = settings.get("default_jobs", 3)
                self.default_jobs_var = tk.IntVar(value=default_jobs)
                self.mcnp_jobs_var = tk.IntVar(value=default_jobs)
                self.dark_mode_var = tk.BooleanVar(value=settings.get("dark_mode", False))
                self.save_csv_var = tk.BooleanVar(value=settings.get("save_csv", True))
                self.neutron_yield = tk.StringVar(value=settings.get("neutron_yield", "single"))
                self.theme_var = tk.StringVar(value=settings.get("theme", "flatly"))
                self.file_tag_var = tk.StringVar(value=settings.get("file_tag", ""))
                self.plot_ext_var = tk.StringVar(value=settings.get("plot_ext", "pdf"))
                self.show_fig_heading_var = tk.BooleanVar(value=settings.get("show_fig_heading", True))
            except Exception:
                self.default_jobs_var = tk.IntVar(value=3)
                self.mcnp_jobs_var = tk.IntVar(value=3)
                self.dark_mode_var = tk.BooleanVar(value=False)
                self.save_csv_var = tk.BooleanVar(value=True)
                self.neutron_yield = tk.StringVar(value="single")
                self.theme_var = tk.StringVar(value="flatly")
                self.file_tag_var = tk.StringVar(value="")
                self.plot_ext_var = tk.StringVar(value="pdf")
                self.show_fig_heading_var = tk.BooleanVar(value=True)
        else:
            self.default_jobs_var = tk.IntVar(value=3)
            self.mcnp_jobs_var = tk.IntVar(value=3)
            self.dark_mode_var = tk.BooleanVar(value=False)
            self.save_csv_var = tk.BooleanVar(value=True)
            self.neutron_yield = tk.StringVar(value="single")
            self.theme_var = tk.StringVar(value="flatly")
            self.file_tag_var = tk.StringVar(value="")
            self.plot_ext_var = tk.StringVar(value="pdf")
            self.show_fig_heading_var = tk.BooleanVar(value=True)

        # Shared variables for runner view
        self.mcnp_folder_var = tk.StringVar()

        # Build interface and views
        self.build_interface()
        self.analysis_view.load_config()
        self.settings_view.toggle_theme()

        # Logging to GUI widgets
        gui_handler = WidgetLoggerHandler(
            self.analysis_view.output_console,
            self.analysis_view.plot_listbox,
            self.runner_view.runner_output_console,
        )
        gui_handler.setFormatter(logging.Formatter("%(message)s"))

        # Attach the handler to the root logger so that logs from all modules
        # propagate to the GUI. Previously the handler was only attached to the
        # ``GUI`` module's logger which meant messages emitted from other
        # modules (e.g. ``he3_plotter.analysis``) were not intercepted. As a
        # result, paths to saved plots logged via ``logger.info("Saved: ...")``
        # never reached the ``WidgetLoggerHandler`` and the "Saved Plots" list
        # remained empty. By registering the handler with the root logger we
        # ensure that all log records, regardless of their originating module,
        # are captured and displayed in the application.
        root_logger = logging.getLogger()
        root_logger.addHandler(gui_handler)

    # ------------------------------------------------------------------
    def load_mcnp_path(self):
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r") as f:
                    return json.load(f).get("MY_MCNP_PATH")
        except Exception:
            pass
        fallback = os.path.expanduser("~/Documents/MCNP/MY_MCNP")
        if os.path.exists(fallback):
            return fallback
        return None

    # ------------------------------------------------------------------
    def build_interface(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        self.runner_tab = ttk.Frame(self.tabs)
        self.analysis_tab = ttk.Frame(self.tabs)
        self.mesh_tab = ttk.Frame(self.tabs)
        self.help_tab = ttk.Frame(self.tabs)
        self.settings_tab = ttk.Frame(self.tabs)

        self.tabs.add(self.runner_tab, text="Run MCNP")
        self.tabs.add(self.analysis_tab, text="Analysis")
        self.tabs.add(self.mesh_tab, text="Mesh Tally")
        self.tabs.add(self.help_tab, text="How to Use")
        self.tabs.add(self.settings_tab, text="Settings")

        self.runner_view = RunnerView(self, self.runner_tab)
        self.analysis_view = AnalysisView(self, self.analysis_tab)
        self.mesh_view = MeshTallyView(self, self.mesh_tab)
        self.settings_view = SettingsView(self, self.settings_tab)

        help_label = tk.Label(self.help_tab, text="How to Use MCNP Tools", font=("Arial", 14, "bold"))
        help_label.pack(pady=10)
        help_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "help_text.txt")
        try:
            with open(help_file, "r", encoding="utf-8") as f:
                help_text = f.read()
        except Exception as e:
            help_text = f"Could not load help text: {e}"
        help_box = ScrolledText(self.help_tab, wrap=tk.WORD, height=25)
        help_box.insert("1.0", help_text)
        help_box.configure(state="disabled")
        help_box.pack(fill="both", expand=True, padx=10, pady=10)


if __name__ == "__main__":
    logging_config.configure()
    if tkdnd:
        root = tkdnd.TkinterDnD.Tk()
    else:
        root = ttk.Window(themename="flatly")

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    try:
        icon_image = tk.PhotoImage(file=icon_path)
        root.iconphoto(True, icon_image)
    except Exception:
        pass

    app = He3PlotterApp(root)
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.mainloop()
