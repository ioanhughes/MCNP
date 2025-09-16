import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


class DummyVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyApp:
    def __init__(self):
        self.logged = []

    def log(self, message, level=None):
        self.logged.append(message)


def setup_view(monkeypatch):
    ttk = types.ModuleType("ttkbootstrap")
    monkeypatch.setitem(sys.modules, "ttkbootstrap", ttk)

    import importlib
    module = importlib.import_module("mcnp.views.dose")
    module = importlib.reload(module)

    app = DummyApp()
    dv = object.__new__(module.DoseView)
    dv.app = app
    dv.result_var = DummyVar("2.0")
    dv.dose_var = DummyVar("")
    dv.source_vars = {
        "Small tank (1.25e6)": DummyVar(True),
        "Big tank (2.5e6)": DummyVar(False),
        "Graphite stack (7.5e6)": DummyVar(False),
    }
    dv.custom_var = DummyVar(True)
    dv.custom_value_var = DummyVar("1e6")
    return dv, app


def test_calculate_dose(monkeypatch):
    dv, app = setup_view(monkeypatch)
    dv.calculate_dose()
    expected = 2.0 * (1.25e6 + 1e6) * 3600 * 1e6
    assert dv.dose_var.get() == f"{expected:.3e}"
    assert any("Calculated dose" in msg for msg in app.logged)
