#!/usr/bin/env python3
"""Interactive CSV plotter for route dosimetry outputs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from mcnp.route_dose import (
    determine_plot_kind,
    load_route_csv,
    plot_route_csv,
    suggest_x_column,
    suggest_y_columns,
)


def _print_columns(columns: list[str]) -> None:
    print("\nAvailable columns:")
    for i, col in enumerate(columns):
        print(f"  [{i}] {col}")
    print("")


def _resolve_column(columns: list[str], choice: str) -> str:
    choice = choice.strip()
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(columns):
            return columns[idx]
        raise ValueError(f"Index {idx} is out of range.")
    if choice in columns:
        return choice
    raise ValueError(f"'{choice}' is not a valid column name or index.")


def _choose_columns(columns: list[str], prompt: str, allow_multiple: bool) -> list[str]:
    while True:
        choice = input(prompt)
        raw_parts = [part.strip() for part in choice.split(",") if part.strip()]
        if not raw_parts:
            print("Please choose at least one column.")
            continue
        try:
            selected = [_resolve_column(columns, part) for part in raw_parts]
        except ValueError as exc:
            print(exc)
            continue
        if not allow_multiple and len(selected) != 1:
            print("Choose exactly one column.")
            continue
        return selected


def _choose_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        value = input(f"{prompt} {suffix}: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _choose_x_and_y(columns: list[str], default_x: str | None, default_y: list[str]) -> tuple[str, list[str]]:
    _print_columns(columns)
    if default_x is not None:
        print(f"Suggested X-axis: {default_x}")
    if default_y:
        print(f"Suggested Y-axis: {', '.join(default_y)}")

    x_input = input("Choose X-axis column (press Enter for suggested): ").strip()
    if x_input:
        x_col = _resolve_column(columns, x_input)
    elif default_x is not None:
        x_col = default_x
    else:
        x_col = _choose_columns(columns, "Choose X-axis column: ", allow_multiple=False)[0]

    y_input = input("Choose Y-axis column(s) (comma-separated, Enter for suggested): ").strip()
    if y_input:
        y_cols = [_resolve_column(columns, part.strip()) for part in y_input.split(",") if part.strip()]
    elif default_y:
        y_cols = default_y
    else:
        y_cols = _choose_columns(columns, "Choose Y-axis column(s): ", allow_multiple=True)

    if not y_cols:
        raise ValueError("At least one Y-axis column is required.")
    return x_col, y_cols


def _choose_summary_columns(columns: list[str], default_y: list[str]) -> list[str]:
    _print_columns(columns)
    if default_y:
        print(f"Suggested summary metrics: {', '.join(default_y)}")
    y_input = input("Choose summary value column(s) (Enter for suggested): ").strip()
    if y_input:
        return [_resolve_column(columns, part.strip()) for part in y_input.split(",") if part.strip()]
    if default_y:
        return default_y
    return _choose_columns(columns, "Choose summary value column(s): ", allow_multiple=True)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python plot_route_csv.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(argv[0])
    if not csv_path.is_file():
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    df = load_route_csv(csv_path)
    columns = list(df.columns)
    default_x = suggest_x_column(df)
    default_y = suggest_y_columns(df)

    try:
        if determine_plot_kind(df, default_x) == "summary":
            y_cols = _choose_summary_columns(columns, default_y)
            plot_route_csv(df, None, y_cols, csv_path)
            return

        x_col, y_cols = _choose_x_and_y(columns, default_x, default_y)
        top_n_tissues = None
        if x_col == "tissue_name" and len(df) > 20:
            top_n = input(
                "Limit tissue bar chart to top N rows by the first Y metric? (Enter for all, e.g. 15): "
            ).strip()
            if top_n:
                top_n_tissues = max(1, int(top_n))
        show_rel_uncertainty = _choose_yes_no(
            "Add a relative uncertainty subplot?",
            default=True,
        )
        plot_route_csv(
            df,
            x_col,
            y_cols,
            csv_path,
            show_relative_uncertainty=show_rel_uncertainty,
            top_n_tissues=top_n_tissues,
        )
    except ValueError as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
