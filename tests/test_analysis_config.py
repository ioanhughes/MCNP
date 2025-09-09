import sys
from pathlib import Path
import json
import logging
import types

sys.path.append(str(Path(__file__).resolve().parent.parent))


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyCombobox:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class DummyApp:
    def __init__(self):
        self.neutron_yield = DummyVar("single")
        self.mcnp_jobs_var = DummyVar(3)
        self.mcnp_folder_var = DummyVar("")
        self.file_tag_var = DummyVar("")
        self.plot_ext_var = DummyVar("pdf")
        self.show_fig_heading_var = DummyVar(True)

    def log(self, *args, **kwargs):
        pass


def create_analysis_view(app, AnalysisType, analysis_view_module):
    av = object.__new__(analysis_view_module.AnalysisView)
    av.app = app
    av.analysis_type = DummyVar(AnalysisType.EFFICIENCY_NEUTRON_RATES.value)
    av.source_vars = {
        "Small tank (1.25e6)": DummyVar(False),
        "Big tank (2.5e6)": DummyVar(False),
        "Graphite stack (7.5e6)": DummyVar(False),
    }
    av.custom_var = DummyVar(False)
    av.custom_value_var = DummyVar("")
    av.analysis_type_map = {
        AnalysisType.EFFICIENCY_NEUTRON_RATES: "Efficiency & Neutron Rates",
        AnalysisType.THICKNESS_COMPARISON: "Thickness Comparison",
        AnalysisType.SOURCE_POSITION_ALIGNMENT: "Source Position Alignment",
        AnalysisType.PHOTON_TALLY_PLOT: "Photon Tally Plot",
    }
    av.analysis_combobox = DummyCombobox()
    av.detector_var = DummyVar("He3")
    av.detector_combobox = DummyCombobox()
    return av


def test_save_and_load_config(tmp_path, monkeypatch):
    # Stub heavy dependencies
    he3_pkg = types.ModuleType("he3_plotter")
    io_utils = types.ModuleType("he3_plotter.io_utils")
    io_utils.select_file = lambda *args, **kwargs: None
    io_utils.select_folder = lambda *args, **kwargs: None
    config = types.ModuleType("he3_plotter.config")
    config.set_filename_tag = lambda *args, **kwargs: None
    config.set_plot_extension = lambda *args, **kwargs: None
    config.set_show_fig_heading = lambda *args, **kwargs: None
    analysis = types.ModuleType("he3_plotter.analysis")
    analysis.run_analysis_type_1 = lambda *args, **kwargs: None
    analysis.run_analysis_type_2 = lambda *args, **kwargs: None
    analysis.run_analysis_type_3 = lambda *args, **kwargs: None
    analysis.run_analysis_type_4 = lambda *args, **kwargs: None
    analysis.AREA = analysis.VOLUME = None
    monkeypatch.setitem(sys.modules, "he3_plotter", he3_pkg)
    monkeypatch.setitem(sys.modules, "he3_plotter.io_utils", io_utils)
    monkeypatch.setitem(sys.modules, "he3_plotter.config", config)
    monkeypatch.setitem(sys.modules, "he3_plotter.analysis", analysis)

    import importlib
    analysis_view_module = importlib.import_module("analysis_view")
    AnalysisType = analysis_view_module.AnalysisType
    monkeypatch.setattr(analysis_view_module, "CONFIG_FILE", tmp_path / "config.json")

    # Set up instance and save config
    app = DummyApp()
    av = create_analysis_view(app, AnalysisType, analysis_view_module)
    app.neutron_yield.set("multi")
    av.analysis_type.set(AnalysisType.THICKNESS_COMPARISON.value)
    av.source_vars["Small tank (1.25e6)"].set(True)
    av.source_vars["Graphite stack (7.5e6)"].set(True)
    av.custom_var.set(True)
    av.custom_value_var.set("9e5")
    app.mcnp_jobs_var.set(5)
    app.mcnp_folder_var.set("/data")
    app.file_tag_var.set("tag")
    app.plot_ext_var.set("png")
    app.show_fig_heading_var.set(False)

    av.save_config()

    # Verify file contents
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["neutron_yield"] == "multi"
    assert data["analysis_type"] == AnalysisType.THICKNESS_COMPARISON.value
    assert data["sources"]["Small tank (1.25e6)"] is True
    assert data["custom_source"] == {"enabled": True, "value": "9e5"}
    assert data["run_profile"] == {"jobs": 5, "folder": "/data"}
    assert data["show_fig_heading"] is False

    # Load config into a fresh instance with different starting values
    app2 = DummyApp()
    av2 = create_analysis_view(app2, AnalysisType, analysis_view_module)
    app2.neutron_yield.set("single")
    av2.analysis_type.set(AnalysisType.EFFICIENCY_NEUTRON_RATES.value)
    av2.load_config()

    assert app2.neutron_yield.get() == "multi"
    assert av2.analysis_type.get() == AnalysisType.THICKNESS_COMPARISON.value
    assert av2.analysis_combobox.get() == "Thickness Comparison"
    assert av2.source_vars["Small tank (1.25e6)"].get() is True
    assert av2.source_vars["Big tank (2.5e6)"].get() is False
    assert av2.source_vars["Graphite stack (7.5e6)"].get() is True
    assert av2.custom_var.get() is True
    assert av2.custom_value_var.get() == "9e5"
    assert app2.mcnp_jobs_var.get() == 5
    assert app2.mcnp_folder_var.get() == "/data"
    assert app2.file_tag_var.get() == "tag"
    assert app2.plot_ext_var.get() == "png"
    assert app2.show_fig_heading_var.get() is False
