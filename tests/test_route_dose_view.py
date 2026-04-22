import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mcnp.views import route_dose as route_dose_view_module


class DummyVar:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyCombobox:
    def __init__(self):
        self.values = []

    def configure(self, **kwargs):
        if "values" in kwargs:
            self.values = list(kwargs["values"])


class DummyListbox:
    def __init__(self):
        self.items = []
        self.selected = set()

    def delete(self, start, end=None):
        self.items = []
        self.selected = set()

    def insert(self, index, value):
        self.items.append(value)

    def selection_set(self, index):
        self.selected.add(index)

    def curselection(self):
        return tuple(sorted(self.selected))

    def get(self, index):
        return self.items[index]


class DummyConsole:
    def __init__(self):
        self.text = []

    def insert(self, index, value):
        self.text.append(value)

    def see(self, index):
        return None

    def delete(self, start, end):
        self.text = []


class DummyApp:
    def __init__(self):
        self.logs = []

    def log(self, message, level=logging.INFO):
        self.logs.append((message, level))


class DummyCheckboxFrame:
    def __init__(self):
        self.children = []

    def winfo_children(self):
        return list(self.children)


class DummyChildWidget:
    def __init__(self, frame):
        self.frame = frame

    def pack(self, *args, **kwargs):
        return self

    def destroy(self):
        if self in self.frame.children:
            self.frame.children.remove(self)


def make_view():
    view = route_dose_view_module.RouteDoseView.__new__(route_dose_view_module.RouteDoseView)
    view.app = DummyApp()
    view.csv_path_var = DummyVar()
    view.x_var = DummyVar()
    view.show_relative_uncertainty_var = DummyVar(True)
    view.show_title_var = DummyVar(True)
    view.current_dataframe = None
    view.current_columns = []
    view.x_combobox = DummyCombobox()
    view.y_checkbox_frame = DummyCheckboxFrame()
    view.y_column_vars = {}
    view.status_console = DummyConsole()
    return view


def test_load_csv_populates_controls(monkeypatch):
    view = make_view()
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "cumulative_effective_dose_uSv": [1.0, 2.0],
            "effective_dose_rate_uSv_per_h": [3.0, 4.0],
            "effective_dose_rate_uncertainty_uSv_per_h": [0.1, 0.2],
        }
    )
    monkeypatch.setattr(route_dose_view_module, "load_route_csv", lambda path: df)
    monkeypatch.setattr(route_dose_view_module.ttk, "Checkbutton", fake_checkbutton_factory)
    monkeypatch.setattr(route_dose_view_module.tk, "BooleanVar", DummyVar)

    view.load_csv_path("/tmp/route.csv")

    assert view.csv_path_var.get() == "/tmp/route.csv"
    assert view.x_var.get() == "distance_cm"
    assert view.x_combobox.values == [
        "distance_cm",
        "cumulative_effective_dose_uSv",
        "effective_dose_rate_uSv_per_h",
    ]
    assert view.get_selected_y_columns() == [
        "cumulative_effective_dose_uSv",
        "effective_dose_rate_uSv_per_h",
    ]


def test_load_csv_logs_error_for_invalid_path(monkeypatch):
    view = make_view()

    def fail_load(path):
        raise ValueError("bad csv")

    monkeypatch.setattr(route_dose_view_module, "load_route_csv", fail_load)

    view.load_csv_path("/tmp/missing.csv")

    assert any("Failed to load CSV: bad csv" in message for message, _ in view.app.logs)


def test_plot_selected_requires_y_selection():
    view = make_view()
    view.current_dataframe = pd.DataFrame({"distance_cm": [0.0], "value": [1.0]})
    view.csv_path_var.set("/tmp/route.csv")
    view.x_var.set("distance_cm")
    view.y_column_vars = {}

    view.plot_selected()

    assert ("Select at least one Y-axis column.", logging.WARNING) in view.app.logs


def test_plot_selected_calls_shared_plotter(monkeypatch):
    view = make_view()
    view.current_dataframe = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "value": [1.0, 2.0],
        }
    )
    view.csv_path_var.set("/tmp/route.csv")
    view.x_var.set("distance_cm")
    view.y_column_vars = {
        "distance_cm": DummyVar(False),
        "value": DummyVar(True),
    }
    called = {}

    def fake_plot(
        df,
        x_col,
        y_cols,
        csv_path,
        show_relative_uncertainty=True,
        show_title=True,
    ):
        called["x_col"] = x_col
        called["y_cols"] = y_cols
        called["csv_path"] = csv_path
        called["show_relative_uncertainty"] = show_relative_uncertainty
        called["show_title"] = show_title
        return "scatter"

    monkeypatch.setattr(route_dose_view_module, "plot_route_csv", fake_plot)

    view.plot_selected()

    assert called == {
        "x_col": "distance_cm",
        "y_cols": ["value"],
        "csv_path": "/tmp/route.csv",
        "show_relative_uncertainty": True,
        "show_title": True,
    }


def test_plot_selected_can_disable_relative_uncertainty(monkeypatch):
    view = make_view()
    view.current_dataframe = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "value": [1.0, 2.0],
        }
    )
    view.csv_path_var.set("/tmp/route.csv")
    view.x_var.set("distance_cm")
    view.show_relative_uncertainty_var.set(False)
    view.y_column_vars = {"value": DummyVar(True)}
    called = {}

    def fake_plot(
        df,
        x_col,
        y_cols,
        csv_path,
        show_relative_uncertainty=True,
        show_title=True,
    ):
        called["show_relative_uncertainty"] = show_relative_uncertainty
        called["show_title"] = show_title
        return "scatter"

    monkeypatch.setattr(route_dose_view_module, "plot_route_csv", fake_plot)

    view.plot_selected()

    assert called["show_relative_uncertainty"] is False
    assert called["show_title"] is True


def test_plot_selected_can_disable_title(monkeypatch):
    view = make_view()
    view.current_dataframe = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "value": [1.0, 2.0],
        }
    )
    view.csv_path_var.set("/tmp/route.csv")
    view.x_var.set("distance_cm")
    view.show_title_var.set(False)
    view.y_column_vars = {"value": DummyVar(True)}
    called = {}

    def fake_plot(
        df,
        x_col,
        y_cols,
        csv_path,
        show_relative_uncertainty=True,
        show_title=True,
    ):
        called["show_title"] = show_title
        return "scatter"

    monkeypatch.setattr(route_dose_view_module, "plot_route_csv", fake_plot)

    view.plot_selected()

    assert called["show_title"] is False


def test_plot_selected_logs_error_for_more_than_two_unit_groups(monkeypatch):
    view = make_view()
    view.current_dataframe = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "effective_dose_rate_uSv_per_h": [1.0, 2.0],
            "cumulative_effective_dose_uSv": [3.0, 4.0],
            "cumulative_absorbed_dose_Gy": [5.0, 6.0],
        }
    )
    view.csv_path_var.set("/tmp/route.csv")
    view.x_var.set("distance_cm")
    view.y_column_vars = {
        "effective_dose_rate_uSv_per_h": DummyVar(True),
        "cumulative_effective_dose_uSv": DummyVar(True),
        "cumulative_absorbed_dose_Gy": DummyVar(True),
    }

    def fail_plot(*args, **kwargs):
        raise ValueError("Select Y-axis columns with at most two distinct units for a dual-axis plot.")

    monkeypatch.setattr(route_dose_view_module, "plot_route_csv", fail_plot)

    view.plot_selected()

    assert any(
        "at most two distinct units" in message for message, _ in view.app.logs
    )


def fake_checkbutton_factory(parent, text, variable, **kwargs):
    widget = DummyChildWidget(parent)
    parent.children.append(widget)
    return widget
