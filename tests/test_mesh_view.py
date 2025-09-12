import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pandas.testing as pdt
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from mcnp.views import mesh_view

class DummyText:
    def __init__(self):
        self.content = ""
    def delete(self, start, end):
        self.content = ""
    def insert(self, index, text):
        self.content = text
    def get(self, start, end):
        return self.content

def make_view():
    view = mesh_view.MeshTallyView.__new__(mesh_view.MeshTallyView)
    view.output_box = DummyText()
    view.msht_df = None
    view._get_total_rate = lambda: 1.0
    class DummyVar:
        def __init__(self, value=""):
            self.value = value
        def get(self):
            return self.value
        def set(self, value):  # pragma: no cover - simple setter
            self.value = value
    view.axis_var = DummyVar("x")
    view.slice_var = DummyVar("0")
    view.slice_viewer_var = DummyVar(False)
    view.cmap_var = DummyVar("jet")
    view.log_scale_var = DummyVar(False)
    return view

def test_load_msht_and_save_csv(tmp_path, monkeypatch):
    content = (
        "Some preamble\n"
        "X         Y         Z     Result    Rel Error   Volume    Result*Volume\n"
        "1.0 2.0 3.0 4.0 0.5 6.0 24.0\n"
        "2.0 3.0 4.0 5.0 0.6 7.0 35.0\n"
    )
    file_path = tmp_path / "sample.msht"
    file_path.write_text(content, encoding="utf-8")

    view = make_view()
    monkeypatch.setattr(mesh_view, "select_file", lambda title='': str(file_path))
    view.load_msht()

    factor = 3600 * 1e6
    expected = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, 6.0, 24.0, 4.0 * factor, 4.0 * factor * 0.5],
            [2.0, 3.0, 4.0, 5.0, 0.6, 7.0, 35.0, 5.0 * factor, 5.0 * factor * 0.6],
        ],
        columns=[
            "x",
            "y",
            "z",
            "result",
            "rel_error",
            "volume",
            "result_vol",
            "dose",
            "dose_error",
        ],
    )
    pdt.assert_frame_equal(view.get_mesh_dataframe(), expected)
    assert "2 rows x 9 columns" in view.output_box.get("1.0", "end")

    csv_path = tmp_path / "out.csv"
    monkeypatch.setattr(mesh_view, "asksaveasfilename", lambda **kwargs: str(csv_path))
    view.save_msht_csv()
    saved = pd.read_csv(csv_path)
    pdt.assert_frame_equal(saved, expected)


def test_load_msht_parse_error(monkeypatch):
    view = make_view()
    monkeypatch.setattr(mesh_view, "select_file", lambda title='': "file.msht")
    monkeypatch.setattr(mesh_view.msht_parser, "parse_msht", lambda path: (_ for _ in ()).throw(ValueError("bad")))
    called = {}
    def fake_error(title, message):
        called["msg"] = (title, message)
    monkeypatch.setattr(mesh_view.Messagebox, "show_error", fake_error, raising=False)
    view.load_msht()
    with pytest.raises(ValueError):
        view.get_mesh_dataframe()
    assert called["msg"][0] == "MSHT Load Error"


def test_plot_dose_map(monkeypatch):
    view = make_view()

    # When no data loaded, should show error
    err = {}
    monkeypatch.setattr(
        mesh_view.Messagebox, "show_error", lambda title, msg: err.setdefault("t", title), raising=False
    )
    view.plot_dose_map()
    assert err.get("t") == "Dose Map Error"


    # Provide sample dataframe and ensure plotting functions are invoked
    view.msht_df = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [1.0, 1.0], "z": [0.0, 0.0], "dose": [1.0, 4.0]}
    )
    view.load_stl_files = lambda: []
    view.cmap_var.set("viridis")

    calls = {}

    class DummyVolume:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            calls["grid"] = grid.tolist()
            calls["spacing"] = spacing
            calls["origin"] = origin

        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["cmap"] = (cmap_name, vmin, vmax)
            return self

        def add_scalarbar(self, title=""):
            calls["scalarbar"] = title
            return self

    monkeypatch.setattr(mesh_view, "Volume", DummyVolume)
    monkeypatch.setattr(
        mesh_view, "show", lambda *a, **kw: calls.setdefault("show", kw.get("axes"))
    )

    # Linear scaling
    view.log_scale_var.set(False)
    view.plot_dose_map()
    linear_calls = calls.copy()
    calls.clear()

    # Log scaling
    view.log_scale_var.set(True)
    view.plot_dose_map()
    log_calls = calls.copy()

    # Linear path assertions
    assert linear_calls["grid"][0][0][0] == pytest.approx(1.0)
    # Second value is clipped to the chosen max dose
    assert linear_calls["grid"][1][0][0] == pytest.approx(linear_calls["cmap"][2])
    assert linear_calls["cmap"][0] == "viridis"
    assert linear_calls["scalarbar"] == "Dose (µSv/h)"
    assert linear_calls["show"] == mesh_view.AXES_LABELS

    # Log scaling assertions
    max_dose = view.msht_df["dose"].quantile(0.95)
    assert log_calls["grid"][0][0][0] == pytest.approx(np.log10(1.0))
    assert log_calls["grid"][1][0][0] == pytest.approx(np.log10(max_dose))
    assert log_calls["cmap"][1] == pytest.approx(np.log10(1.0))
    assert log_calls["cmap"][2] == pytest.approx(np.log10(max_dose))
    assert log_calls["show"] == mesh_view.AXES_LABELS


def test_plot_dose_map_slice_viewer(monkeypatch):
    view = make_view()
    view.msht_df = pd.DataFrame({"x": [1.0], "y": [1.0], "z": [0.0], "dose": [1.0]})
    view.cmap_var.set("magma")
    calls = {}

    class DummyVolume:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            calls["grid"] = grid.tolist()
        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["vol_cmap"] = (cmap_name, vmin, vmax)
            return self
        def add_scalarbar(self, title=""):
            return self

    class DummyMesh:
        def probe(self, vol):
            calls["probed"] = True
        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["mesh_cmap"] = (cmap_name, vmin, vmax)
            return self

    class DummyPlotter:
        def __init__(self, volume, axes=None):
            calls["axes"] = axes
        def __iadd__(self, obj):  # pragma: no cover - simple add
            return self
        def show(self):
            calls["show"] = True

    view.load_stl_files = lambda: [DummyMesh()]
    view.slice_viewer_var.set(True)

    monkeypatch.setattr(mesh_view, "Volume", DummyVolume)
    monkeypatch.setattr(mesh_view, "Slicer3DPlotter", DummyPlotter)
    monkeypatch.setattr(mesh_view, "show", lambda *a, **k: calls.setdefault("plain_show", True))

    view.plot_dose_map()
    assert calls["axes"] == mesh_view.AXES_LABELS
    assert calls["show"] is True
    assert calls["vol_cmap"][0] == "magma"
    assert calls["mesh_cmap"][0] == "magma"
    assert "plain_show" not in calls


def test_plot_dose_slice(monkeypatch):
    view = make_view()

    # Error when no data loaded
    err = {}
    monkeypatch.setattr(
        mesh_view.Messagebox,
        "show_error",
        lambda title, msg: err.setdefault("t", title),
        raising=False,
    )
    view.plot_dose_slice()
    assert err.get("t") == "Dose Slice Error"

    # Provide sample dataframe and set axis/value
    view.msht_df = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [1.0, 1.0], "z": [0.0, 1.0], "dose": [1.0, 4.0]}
    )
    view.axis_var.set("y")
    view.slice_var.set("1.0")

    def run_slice(expected_norm):
        calls = {}

        class DummyAx:
            def scatter(self, x, y, c, marker, s):
                calls["scatter"] = (list(x), list(y))
                calls["colors"] = c
                return object()

            def set_xlabel(self, label):
                calls["xlabel"] = label

            def set_ylabel(self, label):
                calls["ylabel"] = label

        class DummyFig:
            def colorbar(self, sc, ax=None, label=""):
                calls["colorbar"] = label

        base_norm = mesh_view.colors.Normalize

        class LogDummy(base_norm):
            def __init__(self, vmin=None, vmax=None, clip=False):
                calls["norm"] = "log"
                super().__init__(vmin=vmin, vmax=vmax, clip=clip)
            def __call__(self, values):
                return np.zeros(len(values))

        class LinDummy(base_norm):
            def __init__(self, vmin=None, vmax=None, clip=False):
                calls["norm"] = "linear"
                super().__init__(vmin=vmin, vmax=vmax, clip=clip)
            def __call__(self, values):
                return np.zeros(len(values))

        monkeypatch.setattr(mesh_view.plt, "subplots", lambda: (DummyFig(), DummyAx()))
        monkeypatch.setattr(mesh_view.plt, "show", lambda: calls.setdefault("show", True))
        monkeypatch.setattr(mesh_view.colors, "LogNorm", LogDummy)
        if expected_norm == "linear":
            monkeypatch.setattr(mesh_view.colors, "Normalize", LinDummy)
        else:
            monkeypatch.setattr(mesh_view.colors, "Normalize", base_norm)

        view.plot_dose_slice()
        assert calls.get("norm") == expected_norm
        return calls

    # Log scaling
    view.log_scale_var.set(True)
    calls = run_slice("log")
    assert calls["scatter"] == ([1.0, 2.0], [0.0, 1.0])
    assert calls["xlabel"] == "X"
    assert calls["ylabel"] == "Z"
    assert calls["colorbar"] == "Dose (µSv/h)"
    assert calls["show"] is True

    # Linear scaling
    view.log_scale_var.set(False)
    calls = run_slice("linear")

    # Test selection of nearest slice
    view.msht_df = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [0.0, 2.0], "z": [0.0, 1.0], "dose": [1.0, 4.0]}
    )
    view.slice_var.set("1.4")
    calls = run_slice("linear")
    assert calls["scatter"] == ([2.0], [1.0])
    assert view.slice_var.get() == "2"


def test_load_stl_files(tmp_path, monkeypatch):
    view = make_view()

    stl_file = tmp_path / "sample.stl"
    stl_file.write_text("", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")

    class DummyMesh:
        def __init__(self, path):
            self.path = path
        def alpha(self, *a, **k):
            return self
        def c(self, *a, **k):
            return self
        def wireframe(self, *a, **k):
            return self

    dummy_vedo = type("Vedo", (), {"Mesh": DummyMesh})
    monkeypatch.setattr(mesh_view, "vedo", dummy_vedo)

    meshes = view.load_stl_files(folderpath=str(tmp_path))
    assert len(meshes) == 1
    assert meshes[0].path == str(stl_file)


def test_save_dose_map(monkeypatch, tmp_path):
    view = make_view()
    view.msht_df = pd.DataFrame({"x": [1.0], "y": [1.0], "z": [0.0], "dose": [1.0]})
    view.load_stl_files = lambda: []

    calls: dict[str, Any] = {}

    class DummyVolume:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            calls["grid"] = grid
        def cmap(self, *a, **k):
            return self
        def add_scalarbar(self, *a, **k):
            return self

    class DummyPlotter:
        def close(self):
            calls["closed"] = True

    def dummy_show(*a, **k):
        calls["show"] = True
        return DummyPlotter()

    def dummy_screenshot(path, plotter=None):
        calls["screenshot"] = path

    monkeypatch.setattr(mesh_view, "Volume", DummyVolume)
    monkeypatch.setattr(mesh_view, "show", dummy_show)
    monkeypatch.setattr(mesh_view, "screenshot", dummy_screenshot)
    monkeypatch.setattr(
        mesh_view,
        "asksaveasfilename",
        lambda **k: str(tmp_path / "out.png"),
    )

    view.save_dose_map()
    assert calls["screenshot"].endswith("out.png")
    assert calls.get("closed") is True


def test_save_dose_map_slice_viewer(monkeypatch, tmp_path):
    view = make_view()
    view.msht_df = pd.DataFrame({"x": [1.0], "y": [1.0], "z": [0.0], "dose": [1.0]})
    view.slice_viewer_var.set(True)

    class DummyVolume:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            pass
        def cmap(self, *a, **k):
            return self
        def add_scalarbar(self, *a, **k):
            return self

    class DummyMesh:
        def probe(self, vol):
            pass
        def cmap(self, *a, **k):
            return self

    class DummyPlotter:
        def __init__(self, vol, axes=None):
            self.axes = axes
        def __iadd__(self, obj):
            return self
        def show(self, interactive=False):
            pass
        def close(self):
            pass

    calls: dict[str, Any] = {}

    monkeypatch.setattr(mesh_view, "Volume", DummyVolume)
    monkeypatch.setattr(mesh_view, "Slicer3DPlotter", DummyPlotter)
    monkeypatch.setattr(mesh_view, "screenshot", lambda path, plotter=None: calls.setdefault("screenshot", path))
    monkeypatch.setattr(mesh_view, "asksaveasfilename", lambda **k: str(tmp_path / "slice.png"))
    view.load_stl_files = lambda: [DummyMesh()]

    view.save_dose_map()
    assert calls["screenshot"].endswith("slice.png")
