import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pandas.testing as pdt
import numpy as np
import pytest
import time

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from mcnp.views import mesh as mesh_view, vedo_plotter


class DummyText:
    def __init__(self):
        self.content = ""

    def delete(self, start, end):
        self.content = ""

    def insert(self, index, text):
        self.content = text

    def get(self, start, end):
        return self.content

    def see(self, index):  # pragma: no cover - simple stub
        pass


class DummyProgress:
    def __init__(self):
        self.closed = False

    def close(self):  # pragma: no cover - trivial
        self.closed = True


def make_view(collect_callbacks: bool = False):
    view = mesh_view.MeshTallyView.__new__(mesh_view.MeshTallyView)
    view.output_box = DummyText()
    view.msht_df = None
    view._stl_service = mesh_view.StlMeshService(mesh_view.vp)
    view._get_total_rate = lambda: 1.0

    class DummyVar:
        def __init__(self, value=""):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):  # pragma: no cover - simple setter
            self.value = value

        def trace_add(self, *_):  # pragma: no cover - tracing not needed in tests
            return None

    view.axis_var = DummyVar("x")
    view.slice_var = DummyVar("0")
    view.slice_viewer_var = DummyVar(True)
    view.volume_sampling_var = DummyVar(False)
    view.log_scale_var = DummyVar(False)
    view.subdivision_var = DummyVar(0)
    view.msht_path_var = DummyVar("MSHT file: None")
    view.stl_folder_var = DummyVar("STL folder: None")
    view.msht_path = None
    view.stl_folder = None

    view.dose_quantile_var = DummyVar(95.0)
    view.dose_scale_enabled_var = DummyVar(True)
    view._dose_scale_previous = None

    class DummyScale:
        def __init__(self):
            self.config: dict[str, Any] = {}
            self.value = None
            self.states: set[str] = set()

        def configure(self, **kwargs):  # pragma: no cover - simple setter
            self.config.update(kwargs)

        def set(self, value):  # pragma: no cover - simple setter
            self.value = value

        def state(self, states=None):  # pragma: no cover - simple state handler
            if states is None:
                return tuple(self.states)
            if isinstance(states, (list, tuple, set)):
                items = states
            else:
                items = [states]
            for item in items:
                if isinstance(item, str) and item.startswith("!"):
                    self.states.discard(item[1:])
                elif isinstance(item, str):
                    self.states.add(item)
            return tuple(self.states)

    view.slice_scale = DummyScale()
    view.dose_scale = DummyScale()

    class DummyButton:
        def __init__(self):
            self.states: set[str] = set()

        def state(self, states=None):  # pragma: no cover - simple state handler
            if states is None:
                return tuple(self.states)
            if isinstance(states, (list, tuple, set)):
                items = states
            else:
                items = [states]
            for item in items:
                if isinstance(item, str) and item.startswith("!"):
                    self.states.discard(item[1:])
                elif isinstance(item, str):
                    self.states.add(item)
            return tuple(self.states)

    view.msht_button = DummyButton()
    view.stl_button = DummyButton()
    view._msht_thread = None
    view._stl_thread = None

    class DummyLabel:
        def __init__(self, text=""):
            self.text = text

        def config(self, **kwargs):  # pragma: no cover - simple setter
            if "text" in kwargs:
                self.text = kwargs["text"]

    view.dose_scale_value = DummyLabel("95")

    callbacks: list[tuple] = []

    class DummyRoot:
        def after(self, delay, func, *args):
            if collect_callbacks:
                callbacks.append((func, args))
            else:
                func(*args)

    class DummyApp:
        def __init__(self):
            self.root = DummyRoot()

        def log(self, *a, **k):  # pragma: no cover - logging stub
            pass

    view.app = DummyApp()
    view.save_config = lambda: None

    progress_list: list[DummyProgress] = []

    def fake_progress(message: str):
        p = DummyProgress()
        progress_list.append(p)
        return p

    view._show_progress_dialog = fake_progress
    view.after_calls = callbacks
    view.progress_calls = progress_list
    return view


def _drain_after_callbacks(view):
    """Execute queued ``after`` callbacks synchronously for tests."""

    while view.after_calls:
        func, args = view.after_calls.pop(0)
        func(*args)

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
    view.load_msht(path=str(file_path))
    view._msht_thread.join()

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
    out_text = view.output_box.get("1.0", "end")
    assert "2 rows x 9 columns" in out_text
    assert f"Loaded MSHT file: {file_path}" in out_text

    csv_path = tmp_path / "out.csv"
    monkeypatch.setattr(mesh_view, "asksaveasfilename", lambda **kwargs: str(csv_path))
    view.save_msht_csv()
    saved = pd.read_csv(csv_path)
    pdt.assert_frame_equal(saved, expected)


def test_load_stl_files_builds_on_main_thread():
    view = make_view(collect_callbacks=True)

    calls: list[tuple] = []

    def fake_discover(folder):
        calls.append(("discover", threading.current_thread().name))
        return ["a.stl", "b.stl"]

    def fake_build(folder, name):
        calls.append(("build", name, threading.current_thread().name))
        return f"mesh-{name}"

    def fake_update(folder, meshes, files):
        calls.append(("update", threading.current_thread().name, list(meshes), list(files)))

    view.stl_service.discover_stl_files = fake_discover  # type: ignore[assignment]
    view.stl_service.build_mesh = fake_build  # type: ignore[assignment]
    view.stl_service.update_meshes = fake_update  # type: ignore[assignment]

    view.load_stl_files("/tmp/folder")
    view._stl_thread.join()

    _drain_after_callbacks(view)

    build_threads = [entry[2] for entry in calls if entry[0] == "build"]
    assert build_threads, "Expected build calls to be recorded"
    assert all(
        name == threading.main_thread().name for name in build_threads
    ), "Mesh construction must run on the main thread"

    discover_threads = [entry[1] for entry in calls if entry[0] == "discover"]
    assert discover_threads, "Discovery should run in background thread"
    assert all(
        name != threading.main_thread().name for name in discover_threads
    ), "STL discovery should not run on the main thread"

    # Progress dialog closed and button re-enabled
    assert view.progress_calls[0].closed is True
    assert "disabled" not in view.stl_button.state()

    # Update executed on main thread with expected data
    updates = [entry for entry in calls if entry[0] == "update"]
    assert updates and updates[0][1] == threading.main_thread().name
    assert updates[0][2] == ["mesh-a.stl", "mesh-b.stl"]
    assert updates[0][3] == ["a.stl", "b.stl"]

    assert view._stl_thread is None


def test_load_msht_parse_error(monkeypatch):
    view = make_view()
    monkeypatch.setattr(
        mesh_view.msht_parser,
        "parse_msht",
        lambda path: (_ for _ in ()).throw(ValueError("bad")),
    )
    called = {}

    def fake_error(title, message):
        called["msg"] = (title, message)

    monkeypatch.setattr(mesh_view.Messagebox, "show_error", fake_error, raising=False)
    view.load_msht(path="file.msht")
    view._msht_thread.join()
    with pytest.raises(ValueError):
        view.get_mesh_dataframe()
    assert called["msg"][0] == "MSHT Load Error"


def test_plot_dose_map(monkeypatch):
    view = make_view()
    # Use the volume viewer for this test
    view.slice_viewer_var.set(False)

    # When no data loaded, should show error
    err = {}
    monkeypatch.setattr(
        mesh_view.Messagebox, "show_error", lambda title, msg: err.setdefault("t", title), raising=False
    )
    view.plot_dose_map()
    assert err.get("t") == "Dose Map Error"

    # When STL files not loaded, should show error
    err.clear()
    view.msht_df = pd.DataFrame(
        {"x": [1.0], "y": [1.0], "z": [0.0], "dose": [1.0]}
    )
    view.plot_dose_map()
    assert err.get("t") == "Dose Map Error"

    # Provide sample dataframe and ensure plotting functions are invoked
    err.clear()
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

        def add_scalarbar(self, title="", size=None, font_size=None):
            calls["scalarbar"] = {
                "title": title,
                "size": size,
                "font_size": font_size,
            }
            return self

    monkeypatch.setattr(vedo_plotter, "Volume", DummyVolume)

    def fake_show_dose_map(
        vol,
        meshes,
        cmap_name,
        min_dose,
        max_dose,
        *,
        slice_viewer,
        volume_sampling,
        axes,
    ):
        calls["show_axes"] = axes

    view._show_dose_map = fake_show_dose_map

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
    assert linear_calls["cmap"][0] == "jet"
    assert linear_calls["scalarbar"]["title"] == "Dose (µSv/h)"
    assert linear_calls["scalarbar"]["size"] == (300, 900)
    assert linear_calls["scalarbar"]["font_size"] == 36
    assert linear_calls["show_axes"] == mesh_view.AXES_LABELS

    # Log scaling assertions
    max_dose = view.msht_df["dose"].quantile(0.95)
    assert log_calls["grid"][0][0][0] == pytest.approx(np.log10(1.0))
    assert log_calls["grid"][1][0][0] == pytest.approx(np.log10(max_dose))
    assert log_calls["cmap"][1] == pytest.approx(np.log10(1.0))
    assert log_calls["cmap"][2] == pytest.approx(np.log10(max_dose))
    assert log_calls["show_axes"] == mesh_view.AXES_LABELS


def test_update_dose_scale_state():
    view = make_view()
    view.dose_quantile_var.set(75.0)
    view.dose_scale_enabled_var.set(True)
    view._update_dose_scale_state()
    assert "disabled" not in view.dose_scale.states
    assert view.dose_scale_value.text == "75"

    view.dose_scale_enabled_var.set(False)
    view._update_dose_scale_state()
    assert "disabled" in view.dose_scale.states
    assert view._dose_scale_previous == 75.0
    assert view.dose_quantile_var.get() == 100.0
    assert view.dose_scale_value.text == "100"

    view.dose_scale_enabled_var.set(True)
    view._update_dose_scale_state()
    assert "disabled" not in view.dose_scale.states
    assert view.dose_quantile_var.get() == 75.0
    assert view.dose_scale_value.text == "75"


def test_plot_dose_map_dose_scale_toggle(monkeypatch):
    view = make_view()
    view.msht_df = pd.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0], "dose": [1.0, 4.0]}
    )
    captured: list[float] = []

    def fake_build_volume(df, meshes, *, cmap_name, dose_quantile, **kwargs):
        captured.append(dose_quantile)
        return object(), [], cmap_name, 0.0, 1.0

    view._show_dose_map = lambda *a, **k: None
    monkeypatch.setattr(mesh_view.vp, "build_volume", fake_build_volume)

    view.dose_quantile_var.set(80.0)
    view.dose_scale_enabled_var.set(True)
    view._update_dose_scale_state()
    view.plot_dose_map()
    assert captured[0] == pytest.approx(80.0)

    view.dose_scale_enabled_var.set(False)
    view._update_dose_scale_state()
    view.plot_dose_map()
    assert captured[1] == pytest.approx(100.0)

    view.dose_scale_enabled_var.set(True)
    view._update_dose_scale_state()
    view.plot_dose_map()
    assert captured[2] == pytest.approx(80.0)


def test_plot_dose_map_nonuniform_spacing(monkeypatch):
    view = make_view()
    # Use the volume viewer for this test
    view.slice_viewer_var.set(False)
    view.msht_df = pd.DataFrame(
        {"x": [0.0, 1.0, 3.0], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0], "dose": [1.0, 2.0, 3.0]}
    )

    warnings: dict[str, Any] = {}
    monkeypatch.setattr(
        mesh_view.Messagebox,
        "show_warning",
        lambda title, msg: warnings.setdefault(title, msg),
        raising=False,
    )

    class DummyVolume:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            warnings["spacing"] = spacing

        def cmap(self, *a, **k):
            return self

        def add_scalarbar(self, *a, **k):
            return self

    monkeypatch.setattr(vedo_plotter, "Volume", DummyVolume)

    class DummyPlotter:
        def add_button(self, *a, **k):
            pass

        def interactive(self):
            pass

        def close(self):  # pragma: no cover - not used
            pass

    monkeypatch.setattr(
        vedo_plotter, "show", lambda *a, **k: DummyPlotter()
    )

    view._show_dose_map = lambda *a, **k: None
    view.plot_dose_map()
    assert "Non-uniform mesh spacing" in warnings
    assert warnings["spacing"][0] == pytest.approx(1.0)


def test_plot_dose_map_slice_viewer(monkeypatch):
    view = make_view()
    view.msht_df = pd.DataFrame({"x": [1.0], "y": [1.0], "z": [0.0], "dose": [1.0]})
    calls = {}

    class DummyVolume:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            calls["grid"] = grid.tolist()
        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["vol_cmap"] = (cmap_name, vmin, vmax)
            return self
        def add_scalarbar(self, title="", size=None, font_size=None):  # pragma: no cover - simple stub
            return self

    class DummyMesh:
        def probe(self, vol):
            calls["probed"] = True
        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["mesh_cmap"] = (cmap_name, vmin, vmax)
            return self
        def print(self):  # pragma: no cover - simple stub
            calls["printed"] = True
            return self

    class DummyPlotter:
        def __init__(self, volume, axes=None, cmaps=None, draggable=False):
            calls["axes"] = axes

        def __iadd__(self, obj):  # pragma: no cover - simple add
            return self

        def add(self, obj):  # pragma: no cover - annotation add
            calls["added"] = True

        def add_callback(self, event, func):  # pragma: no cover - callback registration
            calls["callback"] = event

        def show(self):
            calls["show"] = True

        def close(self):  # pragma: no cover - not used
            pass

    mesh = DummyMesh()
    view.stl_service.update_meshes("dummy", [mesh], ["mesh.stl"])
    view.slice_viewer_var.set(True)

    monkeypatch.setattr(vedo_plotter, "Volume", DummyVolume)
    monkeypatch.setattr(vedo_plotter, "Slicer3DPlotter", DummyPlotter)

    class DummyText:
        def __init__(self, *a, **k):
            pass

        def text(self, *a, **k):  # pragma: no cover - simple setter
            calls["text"] = True

    monkeypatch.setattr(vedo_plotter, "Text2D", DummyText)

    class PlainPlotter:
        def add_button(self, *a, **k):
            calls["plain_button"] = True

        def interactive(self):
            pass

        def close(self):
            pass

    def fake_show(*a, **kw):
        calls.setdefault("plain_show", True)
        return PlainPlotter()

    monkeypatch.setattr(vedo_plotter, "show", fake_show)

    def fake_show_dose_map(
        vol,
        meshes,
        cmap_name,
        min_dose,
        max_dose,
        *,
        slice_viewer,
        volume_sampling,
        axes,
    ):
        calls["axes"] = axes
        if slice_viewer:
            plt = vedo_plotter.Slicer3DPlotter(vol, axes=axes, cmaps=[cmap_name], draggable=True)
            for mesh in meshes:
                if not volume_sampling:
                    mesh.probe(vol)
                mesh.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
                plt += mesh
            if hasattr(plt, "add"):
                annotation = vedo_plotter.Text2D("", pos="top-left", bg="w", alpha=0.5)
                plt.add(annotation)
            if hasattr(plt, "add_callback"):
                plt.add_callback("MouseMove", lambda *_: None)
            if hasattr(plt, "show"):
                plt.show()
        else:
            for mesh in meshes:
                if not volume_sampling:
                    mesh.probe(vol)
                mesh.cmap(cmap_name, vmin=min_dose, vmax=max_dose)
            plt = vedo_plotter.show(vol, meshes, axes=axes, interactive=False)
            if hasattr(plt, "add_callback"):
                plt.add_callback("MouseMove", lambda *_: None)
        calls["show"] = True

    view._show_dose_map = fake_show_dose_map

    view.plot_dose_map()
    assert calls["axes"] == mesh_view.AXES_LABELS
    assert calls["show"] is True
    assert calls["vol_cmap"][0] == "jet"
    assert calls["mesh_cmap"][0] == "jet"
    assert calls.get("callback") == "MouseMove"
    assert calls.get("added") is True
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

            def set_title(self, title):
                calls["title"] = title

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
        expected_title = f"{view.axis_var.get().upper()} Slice at ~{int(round(view.slice_var.get()))}"
        assert calls.get("title") == expected_title
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
    assert view.slice_var.get() == 2.0


def test_slice_slider_updates(monkeypatch):
    view = make_view()
    view.msht_df = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 2.0],
            "z": [0.0, 3.0],
            "dose": [1.0, 2.0],
        }
    )
    calls: list[Any] = []
    view.plot_dose_slice = lambda: calls.append(view.slice_var.get())

    view.axis_var.set("y")
    view._update_slice_scale()
    assert view.slice_scale.config["from_"] == 0.0
    assert view.slice_scale.config["to"] == 2.0
    # Slider update should not trigger plotting automatically
    assert calls == []
    assert view.slice_var.get() == 0.0

    view.axis_var.set("z")
    view._update_slice_scale()
    assert view.slice_scale.config["to"] == 3.0
    assert calls == []
    assert view.slice_var.get() == 0.0

    view._on_slice_slider(1.5)
    # Moving the slider should update the value without plotting
    assert calls == []
    assert view.slice_var.get() == 1.5

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
    monkeypatch.setattr(vedo_plotter, "vedo", dummy_vedo)

    view.load_stl_files(folderpath=str(tmp_path))
    view._stl_thread.join()
    meshes = view.stl_service.get_base_meshes()
    assert len(meshes) == 1
    assert meshes[0].path == str(stl_file)


def test_load_msht_nonblocking(tmp_path, monkeypatch):
    view = make_view(collect_callbacks=True)

    def fake_parse(path):
        time.sleep(0.2)
        return pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "result": [1.0],
                "rel_error": [0.1],
                "volume": [1.0],
                "result_vol": [1.0],
            }
        )

    monkeypatch.setattr(mesh_view.msht_parser, "parse_msht", fake_parse)
    start = time.time()
    view.load_msht(path=str(tmp_path / "dummy.msht"))
    elapsed = time.time() - start
    assert elapsed < 0.1
    assert view.msht_df is None
    view._msht_thread.join()
    for func, args in view.after_calls:
        func(*args)
    assert view.msht_df is not None
    assert view.progress_calls[0].closed
    assert view.after_calls


def test_load_stl_files_nonblocking(tmp_path, monkeypatch):
    view = make_view(collect_callbacks=True)

    class DummyMesh:
        def __init__(self, path):
            self.path = path

        def alpha(self, *a, **k):  # pragma: no cover - fluent interface stub
            return self

        def c(self, *a, **k):  # pragma: no cover - fluent interface stub
            return self

        def wireframe(self, *a, **k):  # pragma: no cover - fluent interface stub
            return self

    def fake_discover(folder):
        time.sleep(0.2)
        return ["sample.stl"]

    def fake_build(folder, name):
        return DummyMesh(str(tmp_path / name))

    view.stl_service.discover_stl_files = fake_discover  # type: ignore[assignment]
    view.stl_service.build_mesh = fake_build  # type: ignore[assignment]

    start = time.time()
    view.load_stl_files(folderpath=str(tmp_path))
    elapsed = time.time() - start
    assert elapsed < 0.1
    assert view.stl_service.get_base_meshes() == []
    view._stl_thread.join()
    assert view.after_calls
    _drain_after_callbacks(view)
    assert view.stl_service.get_base_meshes()
    assert view.progress_calls[0].closed
    assert not view.after_calls


def test_load_msht_disables_button_and_prevents_overlap(tmp_path, monkeypatch):
    view = make_view(collect_callbacks=True)

    call_count = {"count": 0}

    def fake_parse(path):
        call_count["count"] += 1
        time.sleep(0.1)
        return pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "result": [1.0],
                "rel_error": [0.1],
                "volume": [1.0],
                "result_vol": [1.0],
            }
        )

    monkeypatch.setattr(mesh_view.msht_parser, "parse_msht", fake_parse)
    view.load_msht(path=str(tmp_path / "dummy.msht"))
    assert "disabled" in view.msht_button.states

    # Second invocation should be ignored while the first is still running.
    view.load_msht(path=str(tmp_path / "dummy.msht"))
    assert call_count["count"] == 1
    assert len(view.progress_calls) == 1

    view._msht_thread.join()
    for func, args in view.after_calls:
        func(*args)

    assert "disabled" not in view.msht_button.states
    assert view._msht_thread is not None
    assert not view._msht_thread.is_alive()


def test_load_stl_disables_button_and_prevents_overlap(tmp_path, monkeypatch):
    view = make_view(collect_callbacks=True)

    class DummyMesh:
        def __init__(self, path):
            self.path = path

        def alpha(self, *a, **k):  # pragma: no cover - fluent interface stub
            return self

        def c(self, *a, **k):  # pragma: no cover - fluent interface stub
            return self

        def wireframe(self, *a, **k):  # pragma: no cover - fluent interface stub
            return self

    def fake_discover(folder):
        time.sleep(0.1)
        return ["sample.stl"]

    def fake_build(folder, name):
        return DummyMesh(str(tmp_path / name))

    view.stl_service.discover_stl_files = fake_discover  # type: ignore[assignment]
    view.stl_service.build_mesh = fake_build  # type: ignore[assignment]

    view.load_stl_files(folderpath=str(tmp_path))
    assert "disabled" in view.stl_button.states

    view.load_stl_files(folderpath=str(tmp_path))
    assert len(view.progress_calls) == 1

    view._stl_thread.join()
    _drain_after_callbacks(view)

    assert "disabled" not in view.stl_button.states
    assert view._stl_thread is None


def test_save_stl_files(tmp_path, monkeypatch):
    view = make_view()

    class DummyMesh:
        def __init__(self):
            self.triangulated = False
            self.subdivide_level = None

        def clone(self):
            return DummyMesh()

        def triangulate(self):
            self.triangulated = True
            return self

        def subdivide(self, level, method=1):
            self.subdivide_level = level
            return self

        def write(self, path, binary=True):
            Path(path).write_text(str(self.subdivide_level), encoding="utf-8")

    base_mesh = DummyMesh()
    view.stl_service.update_meshes(str(tmp_path), [base_mesh], ["sample.stl"])
    view.subdivision_var.set(2)
    monkeypatch.setattr(mesh_view.vp, "vedo", object())

    out_dir = tmp_path / "out"
    view.save_stl_files(folderpath=str(out_dir))
    assert (out_dir / "sample.stl").read_text(encoding="utf-8") == "2"
    # Saving should not replace the in-memory meshes with subdivided copies
    assert view.stl_service.get_base_meshes()[0] is base_mesh


def test_stl_service_subdivision_cache(monkeypatch, tmp_path):
    service = mesh_view.StlMeshService(mesh_view.vp)

    class DummyMesh:
        def __init__(self):
            self.subdivide_level = None
            self.triangulated = 0

        def clone(self):
            return DummyMesh()

        def triangulate(self):
            self.triangulated += 1
            return self

        def subdivide(self, level, method=1):
            self.subdivide_level = level
            return self

    base_mesh = DummyMesh()
    service.update_meshes(str(tmp_path), [base_mesh], ["sample.stl"])
    monkeypatch.setattr(mesh_view.vp, "vedo", object())

    assert service.get_base_meshes()[0] is base_mesh

    subdivided = service.get_meshes_for_level(1)[0]
    assert subdivided is not base_mesh
    assert subdivided.subdivide_level == 1

    subdivided.subdivide_level = 99
    assert service.get_meshes_for_level(1)[0].subdivide_level == 99

    assert service.get_base_meshes()[0] is base_mesh
