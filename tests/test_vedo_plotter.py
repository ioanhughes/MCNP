from mcnp.views import vedo_plotter


def test_show_dose_map_handles_missing_setexit(monkeypatch):
    class DummyPlot:
        def __init__(self):
            self.interactor = object()
        def interactive(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(vedo_plotter, "show", lambda *a, **k: DummyPlot())
    vedo_plotter.show_dose_map(object(), [], "jet", 0.0, 1.0, slice_viewer=False)


def test_show_dose_map_slice_viewer_missing_setexit(monkeypatch):
    class DummyPlot:
        def __init__(self):
            self.interactor = object()
        def show(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(vedo_plotter, "Slicer3DPlotter", lambda *a, **k: DummyPlot())
    vedo_plotter.show_dose_map(object(), [], "jet", 0.0, 1.0, slice_viewer=True)
