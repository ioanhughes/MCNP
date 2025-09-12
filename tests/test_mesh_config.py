import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyApp:
    def log(self, *args, **kwargs):
        pass


def create_mesh_view(app, mesh_view_module):
    mv = object.__new__(mesh_view_module.MeshTallyView)
    mv.app = app
    mv.source_vars = {
        "Small tank (1.25e6)": DummyVar(False),
        "Big tank (2.5e6)": DummyVar(False),
        "Graphite stack (7.5e6)": DummyVar(False),
    }
    mv.custom_var = DummyVar(False)
    mv.custom_value_var = DummyVar("")
    mv.axis_var = DummyVar("y")
    mv.slice_var = DummyVar("0")
    mv.slice_viewer_var = DummyVar(False)
    mv.msht_path = None
    mv.stl_folder = None
    mv.msht_path_var = DummyVar("MSHT file: None")
    mv.stl_folder_var = DummyVar("STL folder: None")
    return mv


def test_mesh_view_config(tmp_path, monkeypatch):
    import importlib

    mesh_view_module = importlib.import_module("mcnp.views.mesh_view")
    monkeypatch.setattr(mesh_view_module, "CONFIG_FILE", tmp_path / "config.json")

    app = DummyApp()
    mv = create_mesh_view(app, mesh_view_module)

    # Prepopulate config with unrelated data
    (tmp_path / "config.json").write_text(json.dumps({"other": 1}))

    mv.source_vars["Big tank (2.5e6)"].set(True)
    mv.custom_var.set(True)
    mv.custom_value_var.set("3e6")
    mv.axis_var.set("x")
    mv.slice_var.set("1")
    mv.slice_viewer_var.set(True)
    mv.msht_path = "last.msht"
    mv.stl_folder = "stl_folder"
    mv.save_config()

    data = json.loads((tmp_path / "config.json").read_text())
    assert data["sources"]["Big tank (2.5e6)"] is True
    assert data["custom_source"] == {"enabled": True, "value": "3e6"}
    assert data["other"] == 1
    assert data["msht_path"] == "last.msht"
    assert data["stl_folder"] == "stl_folder"
    assert data["slice_viewer"] is True
    assert data["slice_axis"] == "x"
    assert data["slice_value"] == "1"

    mv2 = create_mesh_view(app, mesh_view_module)
    mv2.load_config()
    assert mv2.source_vars["Big tank (2.5e6)"].get() is True
    assert mv2.custom_var.get() is True
    assert mv2.custom_value_var.get() == "3e6"
    assert mv2.msht_path == "last.msht"
    assert mv2.stl_folder == "stl_folder"
    assert mv2.axis_var.get() == "x"
    assert mv2.slice_var.get() == "1"
    assert mv2.slice_viewer_var.get() is True
