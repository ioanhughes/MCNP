import math
import types

import pandas as pd
import pytest

from mcnp.views import vedo_plotter


def test_material_properties_loaded_from_csv():
    water = vedo_plotter.MATERIAL_PROPERTIES[1]
    assert water["name"] == "Water"
    assert water["density"] == "0.997 g/cm^3"
    assert water["transparent"] is True

    helium = vedo_plotter.MATERIAL_PROPERTIES[2]
    assert helium["name"] == "Helium-3"
    assert helium["density"] == "4.925e-5 atoms/cm^3"
    assert helium["transparent"] is True
    assert math.isclose(vedo_plotter.MOLAR_MASS_G_PER_MOL["helium-3"], 3.016)


def _dose_bounds(df: pd.DataFrame, quantile: float) -> tuple[float, float]:
    max_dose = float(df["dose"].quantile(quantile))
    if not pd.notna(max_dose) or max_dose <= 0.0:
        max_dose = 1.0
    positive = df[df["dose"] > 0.0]["dose"]
    min_dose = float(positive.min()) if not positive.empty else float("nan")
    if not pd.notna(min_dose) or min_dose <= 0.0 or min_dose >= max_dose:
        min_dose = max_dose / 1e6
    return min_dose, max_dose


def test_density_conversion_for_number_density():
    metadata = {"material_id": 2}
    expected = 3.016 / 6.022_140_76e23
    result = vedo_plotter._density_to_g_per_cm3(1.0, "atoms/cm^3", metadata)
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_extract_density_handles_atoms_unit():
    metadata = {
        "material_name": "Helium-3",
        "material_id": 2,
        "density_value": 2.5,
        "density_unit": "atoms/cm^3",
    }
    expected = 2.5 * 3.016 / 6.022_140_76e23
    assert math.isclose(
        vedo_plotter._extract_density_in_g_cm3(metadata), expected, rel_tol=1e-12
    )


def test_load_stl_meshes_subdivision(tmp_path, monkeypatch):
    (tmp_path / "model.stl").write_text("", encoding="utf-8")

    class DummyMesh:
        def __init__(self, path):
            self.path = path
            self.subdivide_level = None
            self.triangulated = False
            self.alpha_value = None

        def alpha(self, value, *a, **k):
            self.alpha_value = value
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
    assert meshes[0].alpha_value == pytest.approx(1.0)

    meshes, _ = vedo_plotter.load_stl_meshes(str(tmp_path), subdivision=0)
    assert meshes[0].subdivide_level is None


def test_load_stl_meshes_metadata(tmp_path, monkeypatch):
    filename = "Large_Water_Tank_1.stl"
    (tmp_path / filename).write_text("", encoding="utf-8")

    class DummyMesh:
        def __init__(self, path):
            self.path = path
            self.alpha_value = None

        def alpha(self, value, *a, **k):
            self.alpha_value = value
            return self

        def c(self, *a, **k):
            return self

        def wireframe(self, *a, **k):
            return self

    dummy_vedo = types.SimpleNamespace(Mesh=DummyMesh)
    monkeypatch.setattr(vedo_plotter, "vedo", dummy_vedo)

    meshes, _ = vedo_plotter.load_stl_meshes(str(tmp_path))
    metadata = getattr(meshes[0], vedo_plotter.MESH_METADATA_ATTR, {})
    assert metadata["object_name"] == "Large Water Tank"
    assert metadata["material_id"] == 1
    assert metadata["material_name"] == "Water"
    assert metadata["density"] == "0.997 g/cm^3"
    assert metadata["transparent"] is True
    assert meshes[0].alpha_value == pytest.approx(vedo_plotter.TRANSPARENT_ALPHA)


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
        def __init__(self):
            setattr(
                self,
                vedo_plotter.MESH_METADATA_ATTR,
                {
                    "object_name": "Dummy Mesh",
                    "material_id": 1,
                    "material_name": "Water",
                    "density": "0.997 g/cm^3",
                    "dose_statistics": {
                        "mean_dose_rate": 2.5,
                        "voxel_count": 10,
                        "total_mass_g": 5.0,
                    },
                },
            )

        def probe(self, vol):
            calls["probed"] = True
            return self

        def cmap(self, cmap_name, vmin=None, vmax=None):
            calls["mesh_cmap"] = (cmap_name, vmin, vmax)
            return self

        def print(self):  # pragma: no cover - simple stub
            calls["printed"] = True
            return self

    class DummySlider:
        def __init__(self, axis):
            self.axis = axis
            self.value = 0.0
            self._callbacks = []

        def AddObserver(self, event, func):
            calls.setdefault("slider_events", []).append((self.axis, event))
            self._callbacks.append((event, func))

        def trigger(self, value):
            self.value = value
            for event_name, func in list(self._callbacks):
                func(self, event_name)

    class DummyPlotter:
        def __init__(self, vol, axes=None, cmaps=None, draggable=False):
            calls["axes"] = axes
            self.xslider = DummySlider("x")
            self.yslider = DummySlider("y")
            self.zslider = DummySlider("z")
            calls["plotter"] = self

        def __iadd__(self, mesh):
            return self

        def add(self, obj):
            calls.setdefault("added", []).append(getattr(obj, "pos", None))

        def add_callback(self, event, func):
            calls["callback"] = event
            calls["probe_func"] = func

        def show(self):
            calls["show"] = True

    class DummyText:
        def __init__(self, *a, **k):
            self.pos = k.get("pos")
            calls.setdefault("text_positions", []).append(self.pos)

        def text(self, arg, *a, **k):
            calls.setdefault(self.pos or "default", []).append(arg)

    class DummyPoint:
        value = 1.23

        def __init__(self, coords):
            self.coords = coords

        def probe(self, vol):
            return types.SimpleNamespace(pointdata=[[self.value]])

    monkeypatch.setattr(vedo_plotter, "Slicer3DPlotter", DummyPlotter)
    monkeypatch.setattr(vedo_plotter, "Text2D", DummyText)
    monkeypatch.setattr(vedo_plotter, "vedo", types.SimpleNamespace(Point=DummyPoint))

    volume_metadata = {"log_scale": False, "conversion_factor": None}

    class DummyVolume:
        def __init__(self):
            self._mcnp_dose_metadata = volume_metadata

        def origin(self):
            return (0.0, 0.0, 0.0)

        def spacing(self):
            return (1.0, 1.0, 1.0)

    vol = DummyVolume()
    mesh = DummyMesh()
    vedo_plotter.show_dose_map(vol, [mesh], "jet", 0.0, 1.0, slice_viewer=True)
    assert calls["axes"] == vedo_plotter.AXES_LABELS
    assert calls["mesh_cmap"][0] == "jet"
    assert calls["callback"] == "MouseMove"
    assert set(calls.get("added", [])) == {"top-left", "top-right"}
    assert calls["show"] is True
    assert "top-right" in calls.get("text_positions", [])
    assert calls["top-right"][0] == "Slice @ x: 0 cm | y: 0 cm | z: 0 cm"

    event = types.SimpleNamespace(picked3d=(1.0, 2.0, 3.0), actor=mesh)
    calls["probe_func"](event)
    assert (
        calls["top-left"][-1]
        == "Dose: 1.23 µSv/h @ (1, 2, 3)\n"
        "Object: Dummy Mesh | Material: Water (1) | Density: 0.997 g/cm^3\n"
        "Mean dose: 2.5 µSv/h | Voxels: 10 | Mass: 5 g"
    )

    DummyPoint.value = 2.0
    volume_metadata.update({"log_scale": True, "conversion_factor": 2.5})
    mesh_metadata = getattr(mesh, vedo_plotter.MESH_METADATA_ATTR)
    mesh_metadata.update({"material_id": 3, "material_name": "Cadmium", "density": "8.65 g/cm^3"})
    event = types.SimpleNamespace(picked3d=(4.0, 5.0, 6.0), actor=mesh)
    calls["probe_func"](event)
    assert (
        calls["top-left"][-1]
        == "Result: 40 | Dose: 100 µSv/h | log10: 2 @ (4, 5, 6)\n"
        "Object: Dummy Mesh | Material: Cadmium (3) | Density: 8.65 g/cm^3\n"
        "Mean dose: 2.5 µSv/h | Voxels: 10 | Mass: 5 g"
    )

    plotter = calls["plotter"]
    plotter.xslider.trigger(2.0)
    assert calls["top-right"][-1] == "Slice @ x: 2 cm | y: 0 cm | z: 0 cm"


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

    min_dose, max_dose = _dose_bounds(df, 1.0)
    vol, meshes, _, _, _ = vedo_plotter.build_volume(
        df,
        [DummyMesh()],
        dose_quantile=100,
        min_dose=min_dose,
        max_dose=max_dose,
        log_scale=False,
        volume_sampling=True,
    )
    assert isinstance(vol, DummyVol)
    assert meshes[0].pointdata["scalars"][0] == pytest.approx(2.0)


def test_build_volume_mesh_statistics(monkeypatch):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [0, 0, 0],
            "z": [0, 0, 0],
            "dose": [1.0, 2.0, 3.0],
        }
    )

    class DummyVol:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            self.grid = grid

        def cmap(self, *a, **k):
            return self

        def add_scalarbar(self, *a, **k):
            return self

    class DummyMesh:
        def __init__(self):
            setattr(
                self,
                vedo_plotter.MESH_METADATA_ATTR,
                {"density": "2 g/cm^3"},
            )

        def inside_points(self, coords, return_ids=False):
            assert return_ids is True
            return np.array([0, 1, 2])

    monkeypatch.setattr(vedo_plotter, "Volume", DummyVol)

    min_dose, max_dose = _dose_bounds(df, 1.0)
    vol, meshes, _, _, _ = vedo_plotter.build_volume(
        df,
        [DummyMesh()],
        dose_quantile=100,
        min_dose=min_dose,
        max_dose=max_dose,
        log_scale=False,
        volume_sampling=False,
    )

    assert isinstance(vol, DummyVol)
    metadata = getattr(meshes[0], vedo_plotter.MESH_METADATA_ATTR)
    stats = metadata["dose_statistics"]
    assert stats["mean_dose_rate"] == pytest.approx(2.0)
    assert stats["voxel_count"] == 3
    assert stats["total_mass_g"] == pytest.approx(6.0)


def test_build_volume_mesh_statistics_distance_fallback(monkeypatch):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "dose": [10.0, 20.0, 30.0, 40.0],
        }
    )

    class DummyVol:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            self.grid = grid

        def cmap(self, *a, **k):
            return self

        def add_scalarbar(self, *a, **k):
            return self

    class DummyPoints:
        def __init__(self, pts):
            self._pts = np.asarray(pts, dtype=float)

        def distance_to(self, mesh, signed=True):
            fn = getattr(mesh, "distance_to_points", None)
            if fn is None:
                raise AttributeError("distance_to_points not implemented")
            return fn(self._pts, signed=signed)

    class DummyMesh:
        def __init__(self):
            setattr(
                self,
                vedo_plotter.MESH_METADATA_ATTR,
                {"density": "1 g/cm^3"},
            )

        def inside_points(self, coords, return_ids=False):
            return np.array([], dtype=int)

        def bounds(self):
            return (0.0, 1.0, 0.0, 1.0, -0.1, 0.1)

        def compute_normals(self):
            return self

        def distance_to_points(self, pts, signed=True):
            pts = np.asarray(pts, dtype=float)
            centre = np.array([0.0, 0.0, 0.0])
            radius = 0.51
            deltas = pts - centre
            distances = np.linalg.norm(deltas, axis=1) - radius
            if signed:
                return distances
            return np.abs(distances)

    monkeypatch.setattr(vedo_plotter, "Volume", DummyVol)
    monkeypatch.setattr(vedo_plotter, "vedo", types.SimpleNamespace(Points=DummyPoints))

    min_dose, max_dose = _dose_bounds(df, 1.0)
    vol, meshes, _, _, _ = vedo_plotter.build_volume(
        df,
        [DummyMesh()],
        dose_quantile=100,
        min_dose=min_dose,
        max_dose=max_dose,
        log_scale=False,
        volume_sampling=False,
    )

    assert isinstance(vol, DummyVol)
    metadata = getattr(meshes[0], vedo_plotter.MESH_METADATA_ATTR)
    stats = metadata["dose_statistics"]
    assert stats["mean_dose_rate"] == pytest.approx(10.0)
    assert stats["voxel_count"] == 1
    assert stats["total_mass_g"] == pytest.approx(1.0)


def test_build_volume_metadata(monkeypatch):
    import pandas as pd

    df = pd.DataFrame(
        {
            "x": [0, 1],
            "y": [0, 0],
            "z": [0, 0],
            "dose": [1.0, 10.0],
            "result": [1e-6, 1e-5],
        }
    )

    class DummyVol:
        def __init__(self, grid, spacing=(1, 1, 1), origin=(0, 0, 0)):
            self.grid = grid

        def cmap(self, *a, **k):
            return self

        def add_scalarbar(self, *a, **k):
            return self

    monkeypatch.setattr(vedo_plotter, "Volume", DummyVol)

    min_dose, max_dose = _dose_bounds(df, 1.0)
    vol, meshes, _, _, _ = vedo_plotter.build_volume(
        df,
        [],
        dose_quantile=100,
        min_dose=min_dose,
        max_dose=max_dose,
        log_scale=True,
        volume_sampling=False,
    )
    assert isinstance(vol, DummyVol)
    assert meshes == []
    metadata = getattr(vol, "_mcnp_dose_metadata")
    assert metadata["log_scale"] is True
    assert metadata["conversion_factor"] == pytest.approx(1e6)
    assert metadata["dose_quantile"] == pytest.approx(100)
