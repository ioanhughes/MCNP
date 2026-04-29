"""Shared route-dose CSV loading, suggestion, and plotting helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL_OVERRIDES = {
    "sample": "Sample index",
    "step_index": "Step index",
    "tissue_id": "Tissue ID",
    "tissue_name": "Tissue name",
    "x_cm": r"$x$ (cm)",
    "y_cm": r"$y$ (cm)",
    "z_cm": r"$z$ (cm)",
    "distance_cm": "Distance (cm)",
    "total_length_cm": "Total length (cm)",
    "time_s": "Time (s)",
    "total_time_s": "Total time (s)",
    "dose_rate_µSv_per_h": r"Dose rate ($\mu$Sv h$^{-1}$)",
    "dose_rate_uSv_per_h": r"Dose rate ($\mu$Sv h$^{-1}$)",
    "incremental_dose_µSv": r"Incremental dose ($\mu$Sv)",
    "incremental_dose_uSv": r"Incremental dose ($\mu$Sv)",
    "cumulative_dose_µSv": r"Cumulative dose ($\mu$Sv)",
    "cumulative_dose_uSv": r"Cumulative dose ($\mu$Sv)",
    "absorbed_dose_rate_Gy_per_h": r"Absorbed dose rate (Gy h$^{-1}$)",
    "absorbed_dose_rate_uncertainty_Gy_per_h": r"Absorbed dose rate uncertainty (Gy h$^{-1}$)",
    "equivalent_dose_rate_uSv_per_h": r"Equivalent dose rate ($\mu$Sv h$^{-1}$)",
    "equivalent_dose_rate_uncertainty_uSv_per_h": r"Equivalent dose rate uncertainty ($\mu$Sv h$^{-1}$)",
    "effective_dose_rate_uSv_per_h": r"Effective dose rate ($\mu$Sv h$^{-1}$)",
    "effective_dose_rate_uncertainty_uSv_per_h": r"Effective dose rate uncertainty ($\mu$Sv h$^{-1}$)",
    "incremental_absorbed_dose_Gy": "Incremental absorbed dose (Gy)",
    "incremental_absorbed_dose_uncertainty_Gy": "Incremental absorbed dose uncertainty (Gy)",
    "cumulative_absorbed_dose_Gy": "Cumulative absorbed dose (Gy)",
    "cumulative_absorbed_dose_uncertainty_Gy": "Cumulative absorbed dose uncertainty (Gy)",
    "incremental_equivalent_dose_uSv": r"Incremental equivalent dose ($\mu$Sv)",
    "incremental_equivalent_dose_uncertainty_uSv": r"Incremental equivalent dose uncertainty ($\mu$Sv)",
    "cumulative_equivalent_dose_uSv": r"Cumulative equivalent dose ($\mu$Sv)",
    "cumulative_equivalent_dose_uncertainty_uSv": r"Cumulative equivalent dose uncertainty ($\mu$Sv)",
    "incremental_effective_dose_uSv": r"Incremental effective dose ($\mu$Sv)",
    "incremental_effective_dose_uncertainty_uSv": r"Incremental effective dose uncertainty ($\mu$Sv)",
    "cumulative_effective_dose_uSv": r"Cumulative effective dose ($\mu$Sv)",
    "cumulative_effective_dose_uncertainty_uSv": r"Cumulative effective dose uncertainty ($\mu$Sv)",
    "total_absorbed_dose_Gy": "Total absorbed dose (Gy)",
    "total_absorbed_dose_uncertainty_Gy": "Total absorbed dose uncertainty (Gy)",
    "total_equivalent_dose_uSv": r"Total equivalent dose ($\mu$Sv)",
    "total_equivalent_dose_uncertainty_uSv": r"Total equivalent dose uncertainty ($\mu$Sv)",
    "total_effective_dose_uSv": r"Total effective dose ($\mu$Sv)",
    "total_effective_dose_uncertainty_uSv": r"Total effective dose uncertainty ($\mu$Sv)",
    "cumulative_effective_contribution_uSv": r"Cumulative effective contribution ($\mu$Sv)",
}

UNIT_LABELS = {
    "cm": "cm",
    "mm": "mm",
    "m": "m",
    "s": "s",
    "min": "min",
    "h": "h",
    "Gy": "Gy",
    "Gy_per_h": r"Gy h$^{-1}$",
    "uSv": r"$\mu$Sv",
    "µSv": r"$\mu$Sv",
    "uSv_per_h": r"$\mu$Sv h$^{-1}$",
    "µSv_per_h": r"$\mu$Sv h$^{-1}$",
}

DEFAULT_TEXT_COLUMNS = {"tissue_name"}
DEFAULT_PLOT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]


def _split_metric_unit(name: str) -> tuple[str, str] | None:
    """Split a metric name into its base portion and unit suffix."""

    for unit in sorted(UNIT_LABELS, key=len, reverse=True):
        suffix = f"_{unit}"
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)], unit
    return None


def column_unit_key(column: str) -> str | None:
    """Return the parsed unit suffix for a metric column, if present."""

    split = _split_metric_unit(column)
    if split is None:
        return None
    return split[1]


def normalize_column_names(columns: Iterable[str]) -> list[str]:
    """Clean route-dose CSV column names for downstream use."""

    cleaned = []
    for column in columns:
        text = str(column)
        text = text.replace("¬µ", "µ")
        text = text.replace("uSv/h", "uSv_per_h")
        cleaned.append(text)
    return cleaned


def is_uncertainty_column(column: str) -> bool:
    """Return whether a column stores uncertainty values."""

    return "_uncertainty_" in column


def selectable_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that should appear in GUI axis selectors."""

    return [column for column in df.columns if not is_uncertainty_column(column)]


def load_route_csv(path: str | Path) -> pd.DataFrame:
    """Load a route-dose CSV and normalise its column names."""

    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    df.columns = normalize_column_names(df.columns)
    return df


def format_axis_label(col_name: str) -> str:
    """Return a user-friendly axis label for a CSV column."""

    name = col_name.replace("¬µ", "µ").replace("uSv/h", "uSv_per_h")

    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]

    split = _split_metric_unit(name)
    if split is not None:
        base_name, unit_str = split
        base = base_name.replace("_", " ").capitalize()
        unit_fmt = UNIT_LABELS.get(unit_str, unit_str.replace("µ", r"$\mu$"))
        return f"{base} ({unit_fmt})"

    return name.replace("_", " ").capitalize()


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a column converted to numeric values where possible."""

    return pd.to_numeric(df[column], errors="coerce")


def is_numeric_column(df: pd.DataFrame, column: str) -> bool:
    """Return whether a column contains any numeric values."""

    return numeric_series(df, column).notna().any()


def dataframe_has_single_row(df: pd.DataFrame) -> bool:
    """Return whether the dataframe represents a single summary row."""

    return len(df.index) == 1


def suggest_x_column(df: pd.DataFrame) -> str | None:
    """Return the preferred X-axis column for a route-dose dataframe."""

    available = set(selectable_columns(df))
    for candidate in (
        "distance_cm",
        "time_s",
        "step_index",
        "sample",
        "tissue_name",
        "tissue_id",
    ):
        if candidate in available:
            return candidate
    return None


def suggest_y_columns(df: pd.DataFrame) -> list[str]:
    """Return up to three preferred Y-axis columns."""

    preferred = [
        "cumulative_effective_dose_uSv",
        "cumulative_equivalent_dose_uSv",
        "cumulative_absorbed_dose_Gy",
        "total_effective_dose_uSv",
        "total_equivalent_dose_uSv",
        "total_absorbed_dose_Gy",
        "effective_dose_rate_uSv_per_h",
        "equivalent_dose_rate_uSv_per_h",
        "absorbed_dose_rate_Gy_per_h",
        "cumulative_effective_contribution_uSv",
        "cumulative_dose_uSv",
    ]
    found = [name for name in preferred if name in df.columns]
    return found[:3]


def uncertainty_partner(column: str, columns: pd.Index) -> str | None:
    """Return the matching uncertainty column when present."""

    if "_uncertainty_" in column:
        return None
    split = _split_metric_unit(column)
    if split is None:
        return None
    base_name, unit = split
    candidate = f"{base_name}_uncertainty_{unit}"
    return candidate if candidate in columns else None


def group_y_columns_by_unit(y_cols: list[str]) -> list[tuple[str | None, list[str]]]:
    """Group Y columns by unit while preserving first-seen order."""

    grouped: list[tuple[str | None, list[str]]] = []
    for column in y_cols:
        unit = column_unit_key(column)
        for existing_unit, columns in grouped:
            if existing_unit == unit:
                columns.append(column)
                break
        else:
            grouped.append((unit, [column]))
    return grouped


def relative_uncertainty_percent(values: pd.Series, uncertainties: pd.Series) -> pd.Series:
    """Compute relative uncertainty percentages for matching values."""

    with np.errstate(divide="ignore", invalid="ignore"):
        relative = 100.0 * uncertainties.astype(float) / np.abs(values.astype(float))
    relative[~np.isfinite(relative)] = np.nan
    return relative


def metric_summary_lines(
    values: pd.Series, uncertainties: pd.Series | None = None
) -> list[str]:
    """Return a concise textual summary for numeric series."""

    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_values.empty:
        return []

    lines = [
        f"  count: {len(clean_values)}",
        f"  min:   {clean_values.min():.6g}",
        f"  mean:  {clean_values.mean():.6g}",
        f"  max:   {clean_values.max():.6g}",
        f"  final: {clean_values.iloc[-1]:.6g}",
    ]

    if uncertainties is not None:
        clean_uncertainties = pd.to_numeric(uncertainties, errors="coerce")
        valid = clean_values.index.intersection(clean_uncertainties.dropna().index)
        if len(valid) > 0:
            rel = relative_uncertainty_percent(
                clean_values.loc[valid],
                clean_uncertainties.loc[valid],
            ).dropna()
            if not rel.empty:
                lines.append(f"  max rel. uncertainty:  {rel.max():.3f}%")
                lines.append(f"  mean rel. uncertainty: {rel.mean():.3f}%")
    return lines


def print_metric_summary(
    label: str, values: pd.Series, uncertainties: pd.Series | None = None
) -> None:
    """Print a metric summary to stdout."""

    lines = metric_summary_lines(values, uncertainties)
    if not lines:
        return
    print(f"\n{label}")
    for line in lines:
        print(line)


def plot_color_cycle(num_colors: int) -> list[str]:
    """Return a deterministic list of plotting colors."""

    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is not None:
        palette = colors.by_key().get("color", [])
    else:
        palette = []
    if not palette:
        palette = DEFAULT_PLOT_COLORS
    return [palette[index % len(palette)] for index in range(num_colors)]


def determine_plot_kind(df: pd.DataFrame, x_col: str | None) -> str:
    """Choose the plot family for the current dataframe and selection."""

    if dataframe_has_single_row(df):
        return "summary"
    if x_col is None:
        raise ValueError("An X-axis column is required for multi-row route plots.")
    if x_col in DEFAULT_TEXT_COLUMNS or not is_numeric_column(df, x_col):
        return "bar"
    return "scatter"


def plot_single_row_summary(
    df: pd.DataFrame,
    csv_path: Path,
    y_cols: list[str],
    *,
    show_title: bool = True,
    show_error_bars: bool = True,
) -> None:
    """Plot a single-row summary dataframe as a bar chart."""

    values = []
    errors = []
    labels = []
    has_any_errors = False
    for column in y_cols:
        series = numeric_series(df, column).dropna()
        if series.empty:
            print(f"Skipping non-numeric summary column: {column}")
            continue
        uncertainty_col = uncertainty_partner(column, df.columns)
        uncertainty_value = np.nan
        if uncertainty_col is not None:
            uncertainty_series = numeric_series(df, uncertainty_col).dropna()
            if not uncertainty_series.empty:
                uncertainty_value = float(uncertainty_series.iloc[0])
                has_any_errors = True
        labels.append(format_axis_label(column))
        values.append(float(series.iloc[0]))
        errors.append(uncertainty_value)
        print_metric_summary(
            format_axis_label(column),
            series,
            df[uncertainty_col] if uncertainty_col else None,
        )

    if not values:
        raise ValueError("No plottable numeric summary columns were selected.")

    plt.figure(figsize=(10, 6))
    yerr = np.array(errors, dtype=float) if has_any_errors and show_error_bars else None
    plt.bar(range(len(values)), values, yerr=yerr, capsize=6)
    plt.xticks(range(len(values)), labels, rotation=30, ha="right")
    plt.ylabel("Value", fontsize=16)
    if show_title:
        plt.title(csv_path.name)
    plt.grid(axis="y", which="major", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()


def plot_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    csv_path: Path,
    *,
    top_n_tissues: int | None = None,
    show_title: bool = True,
    show_error_bars: bool = True,
) -> None:
    """Plot bar-chart style route-dose data."""

    x_values = df[x_col].astype(str)
    if x_col == "tissue_name" and len(df) > 20 and top_n_tissues:
        sort_col = y_cols[0]
        sort_values = numeric_series(df, sort_col)
        ranked = sort_values.sort_values(ascending=False).head(top_n_tissues).index
        df = df.loc[ranked].copy()
        x_values = df[x_col].astype(str)

    if len(y_cols) == 1:
        y_col = y_cols[0]
        y_series = numeric_series(df, y_col)
        uncertainty_col = uncertainty_partner(y_col, df.columns)
        y_err = numeric_series(df, uncertainty_col) if uncertainty_col else None
        valid = y_series.notna()
        if y_err is not None:
            valid = valid & y_err.notna()

        fig, axes = plt.subplots(
            2 if y_err is not None else 1,
            1,
            figsize=(12, 9 if y_err is not None else 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]} if y_err is not None else None,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        main_ax = axes[0]
        main_ax.bar(
            x_values[valid],
            y_series[valid],
            yerr=y_err[valid] if y_err is not None and show_error_bars else None,
            capsize=4,
        )
        main_ax.set_ylabel(format_axis_label(y_col), fontsize=16)
        if show_title:
            main_ax.set_title(csv_path.name)
        main_ax.grid(axis="y", which="major", linestyle="--", linewidth=1)

        if y_err is not None:
            rel = relative_uncertainty_percent(y_series[valid], y_err[valid])
            axes[1].bar(x_values[valid], rel)
            axes[1].set_ylabel("Rel. unc. (%)", fontsize=14)
            axes[1].grid(axis="y", which="major", linestyle=":", linewidth=0.8)
            print_metric_summary(format_axis_label(y_col), y_series[valid], y_err[valid])
        else:
            print_metric_summary(format_axis_label(y_col), y_series[valid])

        axes[-1].set_xlabel(format_axis_label(x_col), fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        return

    subset = df[[x_col] + y_cols].copy()
    for column in y_cols:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
        print_metric_summary(format_axis_label(column), subset[column])
    melted = subset.melt(
        id_vars=x_col,
        value_vars=y_cols,
        var_name="metric",
        value_name="value",
    )
    melted = melted.dropna(subset=["value"])
    pivot = melted.pivot(index=x_col, columns="metric", values="value")
    ax = pivot.plot(kind="bar", figsize=(12, 7))
    ax.set_xlabel(format_axis_label(x_col), fontsize=16)
    ax.set_ylabel("Value", fontsize=16)
    if show_title:
        ax.set_title(csv_path.name)
    ax.legend([format_axis_label(col) for col in pivot.columns], fontsize=10)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    csv_path: Path,
    *,
    show_relative_uncertainty: bool = True,
    show_title: bool = True,
    use_main_markers: bool = False,
    show_error_bars: bool = True,
) -> None:
    """Plot line or error-bar route-dose data."""

    x_values = numeric_series(df, x_col)
    if x_values.isna().all():
        raise ValueError(
            f"X-axis column '{x_col}' does not contain numeric data for a line plot."
        )

    uncertainty_map = {y_col: uncertainty_partner(y_col, df.columns) for y_col in y_cols}
    has_uncertainty = any(col is not None for col in uncertainty_map.values())
    show_rel_uncertainty = has_uncertainty and show_relative_uncertainty
    unit_groups = group_y_columns_by_unit(y_cols)
    if len(unit_groups) > 2:
        raise ValueError(
            "Select Y-axis columns with at most two distinct units for a dual-axis plot."
        )

    fig, axes = plt.subplots(
        2 if show_rel_uncertainty else 1,
        1,
        figsize=(11, 9 if show_rel_uncertainty else 6),
        sharex=True,
        gridspec_kw=(
            {"height_ratios": [3, 1], "hspace": 0}
            if show_rel_uncertainty
            else None
        ),
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    main_ax = axes[0]
    rel_ax = axes[1] if show_rel_uncertainty else None
    secondary_ax = main_ax.twinx() if len(unit_groups) == 2 else None
    axis_map: dict[str, Any] = {}
    for index, (_, columns) in enumerate(unit_groups):
        target_ax = main_ax if index == 0 else secondary_ax
        assert target_ax is not None
        for column in columns:
            axis_map[column] = target_ax

    plotted = False
    legend_handles = []
    legend_labels = []
    color_map = dict(zip(y_cols, plot_color_cycle(len(y_cols))))
    for y_col in y_cols:
        y_values = numeric_series(df, y_col)
        uncertainty_col = uncertainty_map[y_col]
        y_uncertainty = numeric_series(df, uncertainty_col) if uncertainty_col else None
        target_ax = axis_map[y_col]
        color = color_map[y_col]
        valid = x_values.notna() & y_values.notna()
        if y_uncertainty is not None:
            valid = valid & y_uncertainty.notna()
        if not valid.any():
            print(f"Skipping non-numeric or empty Y column: {y_col}")
            continue

        x_valid = x_values[valid]
        y_valid = y_values[valid]
        label = format_axis_label(y_col)

        if y_uncertainty is not None:
            unc_valid = y_uncertainty[valid].abs()
            if show_error_bars:
                container = target_ax.errorbar(
                    x_valid,
                    y_valid,
                    yerr=unc_valid,
                    fmt="o-" if use_main_markers else "-",
                    linewidth=1.5,
                    elinewidth=1,
                    capsize=2,
                    alpha=0.85,
                    color=color,
                    label=label,
                )
                legend_handles.append(container[0])
            else:
                line = target_ax.plot(
                    x_valid,
                    y_valid,
                    linewidth=1.5,
                    alpha=0.85,
                    color=color,
                    label=label,
                    marker="o" if use_main_markers else None,
                )[0]
                legend_handles.append(line)
            legend_labels.append(label)
            if rel_ax is not None:
                rel = relative_uncertainty_percent(y_valid, unc_valid)
                rel_ax.plot(
                    x_valid,
                    rel,
                    linewidth=1.0,
                    alpha=0.8,
                    color=color,
                    label=label,
                )
            print_metric_summary(format_axis_label(y_col), y_valid, unc_valid)
        else:
            line = target_ax.plot(
                x_valid,
                y_valid,
                linewidth=1.5,
                alpha=0.85,
                color=color,
                label=label,
                marker="o" if use_main_markers else None,
            )[0]
            legend_handles.append(line)
            legend_labels.append(label)
            print_metric_summary(format_axis_label(y_col), y_valid)

        plotted = True

    if not plotted:
        raise ValueError("None of the selected Y columns contained plottable numeric data.")

    x_clean = x_values.dropna()
    if not x_clean.empty:
        x_min = float(x_clean.min())
        x_max = float(x_clean.max())
        axes_to_tighten = [main_ax]
        if secondary_ax is not None:
            axes_to_tighten.append(secondary_ax)
        if rel_ax is not None:
            axes_to_tighten.append(rel_ax)
        for axis in axes_to_tighten:
            axis.margins(x=0)
            if x_min != x_max:
                axis.set_xlim(x_min, x_max)

    main_ax.set_xlabel(format_axis_label(x_col), fontsize=16)
    if len(unit_groups) == 1:
        main_ax.set_ylabel(
            format_axis_label(y_cols[0]) if len(y_cols) == 1 else "Value",
            fontsize=16,
        )
    else:
        main_ax.set_ylabel(format_axis_label(unit_groups[0][1][0]), fontsize=16)
        assert secondary_ax is not None
        secondary_ax.set_ylabel(format_axis_label(unit_groups[1][1][0]), fontsize=16)
    if show_title:
        main_ax.set_title(csv_path.name)
    main_ax.grid(which="major", linestyle="--", linewidth=1)
    main_ax.grid(which="minor", linestyle=":", linewidth=0.5)
    if len(y_cols) > 1 or has_uncertainty:
        main_ax.legend(
            legend_handles,
            legend_labels,
            loc="lower left",
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            mode="expand",
            borderaxespad=0,
            ncol=min(len(legend_labels), 2),
        )

    if rel_ax is not None:
        rel_ax.set_xlabel(format_axis_label(x_col), fontsize=16)
        rel_ax.set_ylabel("Rel. unc. (%)", fontsize=14)
        rel_ax.grid(which="major", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    plt.show()


def plot_route_csv(
    df: pd.DataFrame,
    x_col: str | None,
    y_cols: list[str],
    csv_path: str | Path,
    *,
    show_relative_uncertainty: bool = True,
    top_n_tissues: int | None = None,
    show_title: bool = True,
    use_main_markers: bool = False,
    show_error_bars: bool = True,
) -> str:
    """Plot a route-dose dataframe using the appropriate chart family."""

    if not y_cols:
        raise ValueError("At least one Y-axis column is required.")

    path = Path(csv_path)
    plot_kind = determine_plot_kind(df, x_col)
    if plot_kind == "summary":
        plot_single_row_summary(
            df,
            path,
            y_cols,
            show_title=show_title,
            show_error_bars=show_error_bars,
        )
    elif plot_kind == "bar":
        assert x_col is not None
        plot_bar_chart(
            df,
            x_col,
            y_cols,
            path,
            top_n_tissues=top_n_tissues,
            show_title=show_title,
            show_error_bars=show_error_bars,
        )
    else:
        assert x_col is not None
        plot_scatter_chart(
            df,
            x_col,
            y_cols,
            path,
            show_relative_uncertainty=show_relative_uncertainty,
            show_title=show_title,
            use_main_markers=use_main_markers,
            show_error_bars=show_error_bars,
        )
    return plot_kind


__all__ = [
    "DEFAULT_TEXT_COLUMNS",
    "determine_plot_kind",
    "format_axis_label",
    "group_y_columns_by_unit",
    "is_numeric_column",
    "load_route_csv",
    "normalize_column_names",
    "numeric_series",
    "plot_route_csv",
    "column_unit_key",
    "relative_uncertainty_percent",
    "suggest_x_column",
    "suggest_y_columns",
    "uncertainty_partner",
]
