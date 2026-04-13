import sys
from pathlib import Path

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
        lambda *args, **kwargs: calls.append("summary"),
    )
    monkeypatch.setattr(
        route_dose,
        "plot_bar_chart",
        lambda *args, **kwargs: calls.append("bar"),
    )
    monkeypatch.setattr(
        route_dose,
        "plot_scatter_chart",
        lambda *args, **kwargs: calls.append("scatter"),
    )

    assert route_dose.plot_route_csv(summary_df, None, ["total_effective_dose_uSv"], "summary.csv") == "summary"
    assert route_dose.plot_route_csv(text_df, "tissue_name", ["value"], "bar.csv") == "bar"
    assert route_dose.plot_route_csv(numeric_df, "distance_cm", ["value"], "scatter.csv") == "scatter"
    assert calls == ["summary", "bar", "scatter"]
