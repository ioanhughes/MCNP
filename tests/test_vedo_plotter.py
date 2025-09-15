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

        def subdivide(self, level, method=1):
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


def test_show_dose_map_volume_sampling(monkeypatch):
    calls = {}

    class DummyMesh:
        def probe(self, vol):
            calls["probed"] = True
            return self

        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["mesh_cmap"] = (cmap_name, vmin, vmax)
            return self

        def print(self):  # pragma: no cover - simple stub
            calls["printed"] = True
            return self

    def fake_show(*a, **k):
        calls["show"] = True
        return object()

    monkeypatch.setattr(vedo_plotter, "show", fake_show)

    vedo_plotter.show_dose_map(
        object(), [DummyMesh()], "jet", 0.0, 1.0, slice_viewer=False, volume_sampling=True
    )
    # probe should not be called in volume sampling mode
    assert "probed" not in calls
    assert calls["mesh_cmap"][0] == "jet"
    assert calls.get("show")


def test_show_dose_map_slice_viewer(monkeypatch):
    calls = {}

    class DummyMesh:
        def probe(self, vol):
            calls["probed"] = True
            return self

        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["mesh_cmap"] = (cmap_name, vmin, vmax)
            return self

        def print(self):  # pragma: no cover - simple stub
            calls["printed"] = True
            return self

    class DummyPlotter:
        def __init__(self, vol, axes=None, cmaps=None, draggable=False):
            calls["axes"] = axes

        def __iadd__(self, mesh):
            return self

        def add(self, obj):
            calls["added"] = True

        def add_callback(self, event, func):
            calls["callback"] = event
            calls["probe_func"] = func

        def show(self):
            calls["show"] = True

    class DummyText:
        def __init__(self, *a, **k):
            pass

        def text(self, arg, *a, **k):
            calls["text"] = arg

    class DummyPoint:
        def __init__(self, coords):
            self.coords = coords

        def probe(self, vol):
            return types.SimpleNamespace(pointdata=[[1.23]])

    monkeypatch.setattr(vedo_plotter, "Slicer3DPlotter", DummyPlotter)
    monkeypatch.setattr(vedo_plotter, "Text2D", DummyText)
    monkeypatch.setattr(vedo_plotter, "vedo", types.SimpleNamespace(Point=DummyPoint))

    vedo_plotter.show_dose_map(object(), [DummyMesh()], "jet", 0.0, 1.0, slice_viewer=True)
    assert calls["axes"] == vedo_plotter.AXES_LABELS
    assert calls["mesh_cmap"][0] == "jet"
    assert calls["callback"] == "MouseMove"
    assert calls.get("added") is True
    assert calls["show"] is True

    calls["probe_func"](types.SimpleNamespace(picked3d=(1.0, 2.0, 3.0)))
    assert calls["text"] == "1.23 @ (1, 2, 3)"


def test_mesh_to_volume_calls(monkeypatch):
    calls = {}

    class DummyMesh:
        def voxelize(self):
            calls["voxelize"] = True
            return self

        def tovolume(self):
            calls["tovolume"] = True
            return self

        def binarize(self):
            calls["binarize"] = True
            return self

    monkeypatch.setattr(vedo_plotter, "vedo", object())
    res = vedo_plotter.mesh_to_volume(DummyMesh())
    assert res is not None
    assert calls.get("voxelize") and calls.get("tovolume")
    assert calls.get("binarize")


def test_build_volume_volume_sampling(monkeypatch):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        {"x": [0, 1], "y": [0, 0], "z": [0, 0], "dose": [1.0, 3.0]}
    )

    class DummyVol:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            self.grid = grid

        def cmap(self, *a, **k):
            return self

        def add_scalarbar(self, *a, **k):
            return self

    class DummyMask:
        def __init__(self):
            self.pointdata = [np.array([1.0, 3.0])]

        def probe(self, vol):
            return self

    class DummyMesh:
        npoints = 4

    monkeypatch.setattr(vedo_plotter, "Volume", DummyVol)
    monkeypatch.setattr(vedo_plotter, "mesh_to_volume", lambda m: DummyMask())

    vol, meshes, _, _, _ = vedo_plotter.build_volume(
        df,
        [DummyMesh()],
        dose_quantile=100,
        log_scale=False,
        volume_sampling=True,
    )
    assert isinstance(vol, DummyVol)
    assert meshes[0].pointdata["scalars"][0] == pytest.approx(2.0)
