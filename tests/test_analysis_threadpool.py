import importlib
import logging
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyRoot:
    def __init__(self):
        self.calls = []

    def after(self, delay, func):
        self.calls.append(delay)
        func()


class DummyApp:
    def __init__(self):
        self.root = DummyRoot()
        self.save_csv_var = DummyVar(False)
        self.file_tag_var = DummyVar("")
        self.plot_ext_var = DummyVar("pdf")
        self.show_fig_heading_var = DummyVar(True)
        self.axis_label_fontsize_var = DummyVar(12)
        self.tick_label_fontsize_var = DummyVar(10)
        self.legend_fontsize_var = DummyVar(10)
        self.show_grid_var = DummyVar(True)
        self.logged = []

    def log(self, message, level=logging.INFO):
        self.logged.append((message, level))


def setup_view(monkeypatch, *, raise_error=False):
    # Stub external modules
    he3_pkg = types.ModuleType("mcnp.he3_plotter")
    io_utils = types.ModuleType("mcnp.he3_plotter.io_utils")
    io_utils.select_file = lambda *args, **kwargs: None
    io_utils.select_folder = lambda *args, **kwargs: None
    config = types.ModuleType("mcnp.he3_plotter.config")
    config.set_filename_tag = lambda *args, **kwargs: None
    config.set_plot_extension = lambda *args, **kwargs: None
    config.set_show_fig_heading = lambda *args, **kwargs: None
    config.set_axis_label_fontsize = lambda *args, **kwargs: None
    config.set_tick_label_fontsize = lambda *args, **kwargs: None
    config.set_legend_fontsize = lambda *args, **kwargs: None
    config.set_show_grid = lambda *args, **kwargs: None
    analysis = types.ModuleType("mcnp.he3_plotter.analysis")
    detectors = types.ModuleType("mcnp.he3_plotter.detectors")
    detectors.DETECTORS = {"He3": types.SimpleNamespace(area=1.0, volume=1.0)}
    detectors.DEFAULT_DETECTOR = "He3"

    def run1(*args, **kwargs):
        if raise_error:
            raise ValueError("boom")
        return None

    analysis.run_analysis_type_1 = run1
    analysis.run_analysis_type_2 = lambda *args, **kwargs: None
    analysis.run_analysis_type_3 = lambda *args, **kwargs: None
    analysis.run_analysis_type_4 = lambda *args, **kwargs: None
    analysis.AREA = analysis.VOLUME = None

    monkeypatch.setitem(sys.modules, "mcnp.he3_plotter", he3_pkg)
    monkeypatch.setitem(sys.modules, "mcnp.he3_plotter.io_utils", io_utils)
    monkeypatch.setitem(sys.modules, "mcnp.he3_plotter.config", config)
    monkeypatch.setitem(sys.modules, "mcnp.he3_plotter.analysis", analysis)
    monkeypatch.setitem(sys.modules, "mcnp.he3_plotter.detectors", detectors)

    ttk = types.ModuleType("ttkbootstrap")

    class _StubWidget:
        def __init__(self, *args, **kwargs):
            pass

        def pack(self, *args, **kwargs):
            return self

        def grid(self, *args, **kwargs):
            return self

        def place(self, *args, **kwargs):
            return self

        def configure(self, *args, **kwargs):
            return self

        def insert(self, *args, **kwargs):
            return self

        def delete(self, *args, **kwargs):
            return self

        def bind(self, *args, **kwargs):
            return self

        def columnconfigure(self, *args, **kwargs):
            return self

        def after(self, *args, **kwargs):
            return self

        def see(self, *args, **kwargs):
            return self

        def set(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return ""

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class _StubStyle:
        def __init__(self, *args, **kwargs):
            pass

        def theme_use(self, *args, **kwargs):
            return None

    ttk.LabelFrame = _StubWidget  # type: ignore[attr-defined]
    ttk.Frame = _StubWidget  # type: ignore[attr-defined]
    ttk.Button = _StubWidget  # type: ignore[attr-defined]
    ttk.Entry = _StubWidget  # type: ignore[attr-defined]
    ttk.Combobox = _StubWidget  # type: ignore[attr-defined]
    ttk.Checkbutton = _StubWidget  # type: ignore[attr-defined]
    ttk.Spinbox = _StubWidget  # type: ignore[attr-defined]
    ttk.Notebook = _StubWidget  # type: ignore[attr-defined]
    ttk.Style = _StubStyle  # type: ignore[attr-defined]
    ttk.Window = _StubWidget  # type: ignore[attr-defined]

    dialogs = types.ModuleType("ttkbootstrap.dialogs")
    dialogs.Messagebox = types.SimpleNamespace(  # type: ignore[attr-defined]
        yesno=lambda *args, **kwargs: True,
        show_info=lambda *args, **kwargs: None,
        show_error=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "ttkbootstrap", ttk)
    monkeypatch.setitem(sys.modules, "ttkbootstrap.dialogs", dialogs)

    # Reload module to pick up stubs
    module = importlib.import_module("mcnp.views.analysis")
    module = importlib.reload(module)
    AnalysisType = module.AnalysisType

    app = DummyApp()
    av = object.__new__(module.AnalysisView)
    av.app = app
    av.analysis_type = DummyVar(AnalysisType.EFFICIENCY_NEUTRON_RATES.value)
    av.source_vars = {
        "Small tank (1.25e6)": DummyVar(True),
        "Big tank (2.5e6)": DummyVar(False),
        "Graphite stack (7.5e6)": DummyVar(False),
    }
    av.custom_var = DummyVar(False)
    av.custom_value_var = DummyVar("")
    av._analysis_arg_collectors = {
        AnalysisType.EFFICIENCY_NEUTRON_RATES: lambda y: (
            AnalysisType.EFFICIENCY_NEUTRON_RATES,
            "path",
            y,
        )
    }
    av._executor = ThreadPoolExecutor(max_workers=1)
    av.save_config = lambda: None
    av.detector_var = DummyVar("He3")
    return av, app, module


def test_analysis_logs_error(monkeypatch):
    av, app, module = setup_view(monkeypatch, raise_error=True)
    av.run_analysis_threaded()
    av._executor.shutdown(wait=True)
    assert any("Error during analysis" in m for m, _ in app.logged)
    # Ensure callback executed on main thread via after
    assert app.root.calls


def test_analysis_logs_success(monkeypatch):
    av, app, module = setup_view(monkeypatch, raise_error=False)
    av.run_analysis_threaded()
    av._executor.shutdown(wait=True)
    assert ("Analysis complete.", logging.INFO) in app.logged
    assert app.root.calls
