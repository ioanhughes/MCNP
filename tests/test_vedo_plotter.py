import types
from pathlib import Path

import pytest

from mcnp.views import vedo_plotter


def test_load_stl_meshes_subdivision(tmp_path, monkeypatch):
    (tmp_path / "model.stl").write_text("", encoding="utf-8")

    class DummyMesh:
        def __init__(self, path):
            self.path = path
            self.subdivide_level = None
            self.triangulated = False

        def alpha(self, *a, **k):
            return self

        def c(self, *a, **k):
            return self

        def wireframe(self, *a, **k):
            return self

        def triangulate(self):
            self.triangulated = True
            return self

        def subdivide(self, level):
            self.subdivide_level = level
            return self

    dummy_vedo = types.SimpleNamespace(Mesh=DummyMesh)
    monkeypatch.setattr(vedo_plotter, "vedo", dummy_vedo)

    meshes, files = vedo_plotter.load_stl_meshes(str(tmp_path), subdivision=2)
    assert files == ["model.stl"]
    assert meshes[0].triangulated is True
    assert meshes[0].subdivide_level == 2

    meshes, _ = vedo_plotter.load_stl_meshes(str(tmp_path), subdivision=0)
    assert meshes[0].subdivide_level is None


def test_show_dose_map_probes(monkeypatch):
    calls = {}

    class DummyMesh:
        def probe(self, vol):
            calls["probed"] = calls.get("probed", 0) + 1
            return self

        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["mesh_cmap"] = (cmap_name, vmin, vmax)
            return self

    def fake_show(*a, **k):
        calls["show"] = True
        return object()

    monkeypatch.setattr(vedo_plotter, "show", fake_show)

    vedo_plotter.show_dose_map(object(), [DummyMesh()], "jet", 0.0, 1.0, slice_viewer=False)
    assert calls.get("probed") == 1
    assert calls["mesh_cmap"][0] == "jet"
    assert calls.get("show")
