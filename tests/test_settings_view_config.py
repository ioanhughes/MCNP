import sys
import json
import logging
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mcnp.views import settings as settings_module
from mcnp.utils import config_utils


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):  # pragma: no cover - simple setter
        self.value = value


class DummyApp:
    def __init__(self, settings_path: Path):
        self.settings_path = str(settings_path)
        self.base_dir = "/path"
        self.mcnp_jobs_var = DummyVar(3)
        self.dark_mode_var = DummyVar(False)
        self.save_csv_var = DummyVar(True)
        self.neutron_yield = DummyVar("single")
        self.theme_var = DummyVar("flatly")
        self.plot_ext_var = DummyVar("pdf")
        self.show_fig_heading_var = DummyVar(True)
        self.analysis_view = types.SimpleNamespace(save_config=lambda: None)
        self.logs = []

    def log(self, message, level=logging.INFO):  # pragma: no cover - simple logger
        self.logs.append(message)


def make_view(app):
    view = settings_module.SettingsView.__new__(settings_module.SettingsView)
    view.app = app
    view.mcnp_path_var = DummyVar(app.base_dir)
    view.default_jobs_var = DummyVar(app.mcnp_jobs_var.get())
    view.theme_var = app.theme_var
    view.toggle_theme = lambda: None
    return view


def test_save_settings_merges_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(config_utils, "PROJECT_SETTINGS_PATH", tmp_path / "config.json")
    config_utils.save_settings({"sources": {"A": True}})
    app = DummyApp(config_utils.PROJECT_SETTINGS_PATH)
    view = make_view(app)
    view.save_settings()
    data = json.loads(config_utils.PROJECT_SETTINGS_PATH.read_text())
    assert data["sources"] == {"A": True}
    assert data["default_jobs"] == 3


def test_change_mcnp_path_preserves_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_utils, "PROJECT_SETTINGS_PATH", tmp_path / "config.json")
    config_utils.save_settings({"sources": {"A": True}})
    app = DummyApp(config_utils.PROJECT_SETTINGS_PATH)
    view = make_view(app)
    monkeypatch.setattr(settings_module.filedialog, "askdirectory", lambda title="": "/new/path")
    view.change_mcnp_path()
    data = json.loads(config_utils.PROJECT_SETTINGS_PATH.read_text())
    assert data["sources"] == {"A": True}
    assert data["MY_MCNP_PATH"] == "/new/path"
