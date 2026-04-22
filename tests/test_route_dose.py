import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from mcnp import route_dose


def test_normalize_column_names_handles_micro_and_rate_text():
    columns = ["dose_rate_uSv/h", "incremental_dose_¬µSv", "distance_cm"]
    assert route_dose.normalize_column_names(columns) == [
        "dose_rate_uSv_per_h",
        "incremental_dose_µSv",
        "distance_cm",
    ]


def test_suggest_axis_columns_prefers_route_metrics():
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "cumulative_effective_dose_uSv": [1.0, 2.0],
            "effective_dose_rate_uSv_per_h": [3.0, 4.0],
            "other": [5.0, 6.0],
        }
    )
    assert route_dose.suggest_x_column(df) == "distance_cm"
    assert route_dose.suggest_y_columns(df) == [
        "cumulative_effective_dose_uSv",
        "effective_dose_rate_uSv_per_h",
    ]


def test_selectable_columns_hide_uncertainty_by_default():
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "effective_dose_rate_uSv_per_h": [1.0, 2.0],
            "effective_dose_rate_uncertainty_uSv_per_h": [0.1, 0.2],
        }
    )
    assert route_dose.selectable_columns(df) == [
        "distance_cm",
        "effective_dose_rate_uSv_per_h",
    ]


def test_uncertainty_partner_detects_matching_column():
    columns = pd.Index(
        [
            "effective_dose_rate_uSv_per_h",
            "effective_dose_rate_uncertainty_uSv_per_h",
        ]
    )
    assert (
        route_dose.uncertainty_partner("effective_dose_rate_uSv_per_h", columns)
        == "effective_dose_rate_uncertainty_uSv_per_h"
    )


def test_column_unit_key_extracts_metric_units():
    assert route_dose.column_unit_key("effective_dose_rate_uSv_per_h") == "uSv_per_h"
    assert route_dose.column_unit_key("cumulative_effective_dose_uSv") == "uSv"
    assert route_dose.column_unit_key("tissue_name") is None


def test_group_y_columns_by_unit_preserves_selection_order():
    grouped = route_dose.group_y_columns_by_unit(
        [
            "effective_dose_rate_uSv_per_h",
            "equivalent_dose_rate_uSv_per_h",
            "cumulative_effective_dose_uSv",
        ]
    )
    assert grouped == [
        ("uSv_per_h", ["effective_dose_rate_uSv_per_h", "equivalent_dose_rate_uSv_per_h"]),
        ("uSv", ["cumulative_effective_dose_uSv"]),
    ]


def test_group_y_columns_by_unit_allows_three_distinct_units_for_validation_step():
    grouped = route_dose.group_y_columns_by_unit(
        [
            "effective_dose_rate_uSv_per_h",
            "cumulative_effective_dose_uSv",
            "cumulative_absorbed_dose_Gy",
        ]
    )
    assert grouped == [
        ("uSv_per_h", ["effective_dose_rate_uSv_per_h"]),
        ("uSv", ["cumulative_effective_dose_uSv"]),
        ("Gy", ["cumulative_absorbed_dose_Gy"]),
    ]


def test_determine_plot_kind_uses_dataframe_shape_and_x_type():
    summary_df = pd.DataFrame({"total_effective_dose_uSv": [1.0]})
    text_df = pd.DataFrame({"tissue_name": ["a", "b"], "value": [1.0, 2.0]})
    numeric_df = pd.DataFrame({"distance_cm": [0.0, 1.0], "value": [1.0, 2.0]})

    assert route_dose.determine_plot_kind(summary_df, None) == "summary"
    assert route_dose.determine_plot_kind(text_df, "tissue_name") == "bar"
    assert route_dose.determine_plot_kind(numeric_df, "distance_cm") == "scatter"


def test_plot_route_csv_routes_to_expected_plot_family(monkeypatch):
    calls = []
    summary_df = pd.DataFrame({"total_effective_dose_uSv": [1.0]})
    text_df = pd.DataFrame({"tissue_name": ["a", "b"], "value": [1.0, 2.0]})
    numeric_df = pd.DataFrame({"distance_cm": [0.0, 1.0], "value": [1.0, 2.0]})

    monkeypatch.setattr(
        route_dose,
        "plot_single_row_summary",
        lambda *args, **kwargs: calls.append(("summary", kwargs.get("show_title", True))),
    )
    monkeypatch.setattr(
        route_dose,
        "plot_bar_chart",
        lambda *args, **kwargs: calls.append(("bar", kwargs.get("show_title", True))),
    )
    monkeypatch.setattr(
        route_dose,
        "plot_scatter_chart",
        lambda *args, **kwargs: calls.append(("scatter", kwargs.get("show_title", True))),
    )

    assert route_dose.plot_route_csv(summary_df, None, ["total_effective_dose_uSv"], "summary.csv") == "summary"
    assert route_dose.plot_route_csv(text_df, "tissue_name", ["value"], "bar.csv") == "bar"
    assert route_dose.plot_route_csv(numeric_df, "distance_cm", ["value"], "scatter.csv") == "scatter"
    assert calls == [("summary", True), ("bar", True), ("scatter", True)]


def test_plot_route_csv_passes_show_title_flag(monkeypatch):
    calls = []
    numeric_df = pd.DataFrame({"distance_cm": [0.0, 1.0], "value": [1.0, 2.0]})

    monkeypatch.setattr(
        route_dose,
        "plot_scatter_chart",
        lambda *args, **kwargs: calls.append(kwargs.get("show_title")),
    )

    route_dose.plot_route_csv(
        numeric_df,
        "distance_cm",
        ["value"],
        "scatter.csv",
        show_title=False,
    )

    assert calls == [False]


def test_plot_scatter_chart_raises_for_more_than_two_unit_groups():
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "effective_dose_rate_uSv_per_h": [1.0, 2.0],
            "cumulative_effective_dose_uSv": [3.0, 4.0],
            "cumulative_absorbed_dose_Gy": [5.0, 6.0],
        }
    )
    try:
        route_dose.plot_scatter_chart(
            df,
            "distance_cm",
            [
                "effective_dose_rate_uSv_per_h",
                "cumulative_effective_dose_uSv",
                "cumulative_absorbed_dose_Gy",
            ],
            Path("route.csv"),
        )
    except ValueError as exc:
        assert "at most two distinct units" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for more than two unit groups")


class _FakeLine:
    def __init__(self, color: str):
        self._color = color

    def get_color(self):
        return self._color


class _FakeAxes:
    def __init__(self, name: str, color: str = "C0"):
        self.name = name
        self.color = color
        self.calls = []
        self.ylabel = None
        self.legend_args = None
        self.twin = None
        self.xlabel = None
        self.xlim = None
        self.margin_calls = []

    def twinx(self):
        self.twin = _FakeAxes("right", "C1")
        return self.twin

    def errorbar(self, *args, **kwargs):
        self.calls.append(("errorbar", kwargs.get("label"), kwargs.get("color")))
        return (_FakeLine(kwargs.get("color", self.color)),)

    def plot(self, *args, **kwargs):
        self.calls.append(("plot", kwargs.get("label"), kwargs.get("color")))
        return [_FakeLine(kwargs.get("color", self.color))]

    def set_ylabel(self, value, fontsize=None):
        self.ylabel = value

    def set_xlabel(self, *args, **kwargs):
        self.xlabel = args[0] if args else None
        return None

    def set_title(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def legend(self, handles=None, labels=None, **kwargs):
        self.legend_args = (handles, labels)

    def margins(self, *args, **kwargs):
        self.margin_calls.append((args, kwargs))

    def set_xlim(self, left, right):
        self.xlim = (left, right)


def test_plot_scatter_chart_uses_dual_axes_for_two_unit_groups(monkeypatch):
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "effective_dose_rate_uSv_per_h": [1.0, 2.0],
            "cumulative_effective_dose_uSv": [3.0, 4.0],
        }
    )
    main_ax = _FakeAxes("left", "C0")
    monkeypatch.setattr(route_dose.plt, "subplots", lambda *args, **kwargs: (object(), main_ax))
    monkeypatch.setattr(route_dose.plt, "tight_layout", lambda: None)
    monkeypatch.setattr(route_dose.plt, "show", lambda: None)

    route_dose.plot_scatter_chart(
        df,
        "distance_cm",
        ["effective_dose_rate_uSv_per_h", "cumulative_effective_dose_uSv"],
        Path("route.csv"),
        show_relative_uncertainty=False,
    )

    assert main_ax.twin is not None
    assert any(
        call[:2] == ("plot", route_dose.format_axis_label("effective_dose_rate_uSv_per_h"))
        for call in main_ax.calls
    )
    assert any(
        call[:2] == ("plot", route_dose.format_axis_label("cumulative_effective_dose_uSv"))
        for call in main_ax.twin.calls
    )
    assert main_ax.ylabel == route_dose.format_axis_label("effective_dose_rate_uSv_per_h")
    assert main_ax.twin.ylabel == route_dose.format_axis_label("cumulative_effective_dose_uSv")
    left_color = next(
        call[2]
        for call in main_ax.calls
        if call[:2] == ("plot", route_dose.format_axis_label("effective_dose_rate_uSv_per_h"))
    )
    right_color = next(
        call[2]
        for call in main_ax.twin.calls
        if call[:2] == ("plot", route_dose.format_axis_label("cumulative_effective_dose_uSv"))
    )
    assert left_color != right_color


def test_plot_scatter_chart_attaches_errorbars_to_correct_axis(monkeypatch):
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "effective_dose_rate_uSv_per_h": [1.0, 2.0],
            "effective_dose_rate_uncertainty_uSv_per_h": [0.1, 0.2],
            "cumulative_effective_dose_uSv": [3.0, 4.0],
        }
    )
    main_ax = _FakeAxes("left", "C0")
    monkeypatch.setattr(route_dose.plt, "subplots", lambda *args, **kwargs: (object(), main_ax))
    monkeypatch.setattr(route_dose.plt, "tight_layout", lambda: None)
    monkeypatch.setattr(route_dose.plt, "show", lambda: None)

    route_dose.plot_scatter_chart(
        df,
        "distance_cm",
        ["effective_dose_rate_uSv_per_h", "cumulative_effective_dose_uSv"],
        Path("route.csv"),
        show_relative_uncertainty=False,
    )

    assert any(
        call[:2] == ("errorbar", route_dose.format_axis_label("effective_dose_rate_uSv_per_h"))
        for call in main_ax.calls
    )
    assert any(
        call[:2] == ("plot", route_dose.format_axis_label("cumulative_effective_dose_uSv"))
        for call in main_ax.twin.calls
    )


def test_plot_scatter_chart_uses_zero_hspace_and_single_legend(monkeypatch):
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0],
            "effective_dose_rate_uSv_per_h": [1.0, 2.0],
            "effective_dose_rate_uncertainty_uSv_per_h": [0.1, 0.2],
            "cumulative_effective_dose_uSv": [3.0, 4.0],
            "cumulative_effective_dose_uncertainty_uSv": [0.3, 0.4],
        }
    )
    main_ax = _FakeAxes("left", "C0")
    rel_ax = _FakeAxes("rel", "C2")
    captured = {}

    def fake_subplots(*args, **kwargs):
        captured["gridspec_kw"] = kwargs.get("gridspec_kw")
        return object(), np.array([main_ax, rel_ax], dtype=object)

    monkeypatch.setattr(route_dose.plt, "subplots", fake_subplots)
    monkeypatch.setattr(route_dose.plt, "tight_layout", lambda: None)
    monkeypatch.setattr(route_dose.plt, "show", lambda: None)

    route_dose.plot_scatter_chart(
        df,
        "distance_cm",
        ["effective_dose_rate_uSv_per_h", "cumulative_effective_dose_uSv"],
        Path("route.csv"),
        show_relative_uncertainty=True,
    )

    assert captured["gridspec_kw"] == {"height_ratios": [3, 1], "hspace": 0}
    assert main_ax.legend_args is not None
    assert rel_ax.legend_args is None


def test_plot_scatter_chart_removes_x_padding(monkeypatch):
    df = pd.DataFrame(
        {
            "distance_cm": [0.0, 1.0, 2.0],
            "effective_dose_rate_uSv_per_h": [1.0, 2.0, 3.0],
        }
    )
    main_ax = _FakeAxes("left", "C0")
    monkeypatch.setattr(route_dose.plt, "subplots", lambda *args, **kwargs: (object(), main_ax))
    monkeypatch.setattr(route_dose.plt, "tight_layout", lambda: None)
    monkeypatch.setattr(route_dose.plt, "show", lambda: None)

    route_dose.plot_scatter_chart(
        df,
        "distance_cm",
        ["effective_dose_rate_uSv_per_h"],
        Path("route.csv"),
        show_relative_uncertainty=False,
    )

    assert main_ax.xlim == (0.0, 2.0)
    assert (((), {"x": 0}) in main_ax.margin_calls)
