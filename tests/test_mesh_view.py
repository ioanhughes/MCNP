import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
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
    monkeypatch.setattr(mesh_view, "show", lambda obj, axes=1: calls.setdefault("show", axes))

    view.plot_dose_map()
    assert calls["grid"][0][0][0] == pytest.approx(1.0)
    # Second value is clipped to the chosen max dose
    assert calls["grid"][1][0][0] == pytest.approx(calls["cmap"][2])
    assert calls["cmap"][0] == "jet"
    assert calls["scalarbar"] == "Dose (µSv/h)"
    assert calls["show"] == 1


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

    monkeypatch.setattr(mesh_view.plt, "subplots", lambda: (DummyFig(), DummyAx()))
    monkeypatch.setattr(mesh_view.plt, "show", lambda: calls.setdefault("show", True))

    view.plot_dose_slice()
    assert calls["scatter"] == ([1.0, 2.0], [0.0, 1.0])
    assert len(calls["colors"]) == 2
    alphas = [col[3] for col in calls["colors"]]
    assert all(alpha == pytest.approx(1.0) for alpha in alphas)
    assert calls["xlabel"] == "X"
    assert calls["ylabel"] == "Z"
    assert calls["colorbar"] == "Dose (µSv/h)"
    assert calls["show"] is True

    calls.clear()
    view.msht_df = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [0.0, 2.0], "z": [0.0, 1.0], "dose": [1.0, 4.0]}
    )
    view.slice_var.set("1.4")
    view.plot_dose_slice()
    assert calls["scatter"] == ([2.0], [1.0])
    assert view.slice_var.get() == "2"
