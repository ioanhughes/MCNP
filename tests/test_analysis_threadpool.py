import sys
import types
import logging
import importlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).resolve().parent.parent))


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
        self.logged = []

    def log(self, message, level=logging.INFO):
        self.logged.append((message, level))


def setup_view(monkeypatch, *, raise_error=False):
    # Stub external modules
    he3_pkg = types.ModuleType("he3_plotter")
    io_utils = types.ModuleType("he3_plotter.io_utils")
    io_utils.select_file = lambda *args, **kwargs: None
    io_utils.select_folder = lambda *args, **kwargs: None
    config = types.ModuleType("he3_plotter.config")
    config.set_filename_tag = lambda *args, **kwargs: None
    config.set_plot_extension = lambda *args, **kwargs: None
    config.set_show_fig_heading = lambda *args, **kwargs: None
    analysis = types.ModuleType("he3_plotter.analysis")

    def run1(*args, **kwargs):
        if raise_error:
            raise ValueError("boom")
        return None

    analysis.run_analysis_type_1 = run1
    analysis.run_analysis_type_2 = lambda *args, **kwargs: None
    analysis.run_analysis_type_3 = lambda *args, **kwargs: None
    analysis.run_analysis_type_4 = lambda *args, **kwargs: None
    analysis.AREA = analysis.VOLUME = None

    monkeypatch.setitem(sys.modules, "he3_plotter", he3_pkg)
    monkeypatch.setitem(sys.modules, "he3_plotter.io_utils", io_utils)
    monkeypatch.setitem(sys.modules, "he3_plotter.config", config)
    monkeypatch.setitem(sys.modules, "he3_plotter.analysis", analysis)

    ttk = types.ModuleType("ttkbootstrap")
    monkeypatch.setitem(sys.modules, "ttkbootstrap", ttk)

    # Reload module to pick up stubs
    module = importlib.import_module("analysis_view")
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
