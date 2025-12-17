import os
import re
import logging
import pandas as pd
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# ``FigureCanvasAgg`` is used directly when emitting plots from background
# threads so that we do not rely on Matplotlib's global backend state.  This
# avoids clobbering any interactive backend selected by the GUI layer while
# still allowing deterministic file generation.

from .io_utils import get_output_path, select_file
from .plots import plot_efficiency_and_rates
from .detectors import DETECTORS, DEFAULT_DETECTOR
from .config import config

logger = logging.getLogger(__name__)


def _sort_and_deduplicate_thickness(df, label):
    """Sort a dataframe by thickness and warn about duplicate entries.

    The first occurrence of each thickness value is retained so downstream
    merges can be validated as one-to-one.
    """

    df_sorted = df.sort_values("thickness").reset_index(drop=True)
    duplicate_mask = df_sorted.duplicated(subset=["thickness"], keep="first")
    if duplicate_mask.any():
        duplicates = sorted(df_sorted.loc[duplicate_mask, "thickness"].unique())
        logger.warning(
            "%s contains duplicate thickness values %s; keeping the first occurrence",
            label,
            duplicates,
        )
        df_sorted = df_sorted.loc[~duplicate_mask].reset_index(drop=True)
    return df_sorted


def read_tally_blocks_to_df(file_path, tally_ids=None):
    """Read MCNP tally blocks and return DataFrames for each tally type.

    Parameters
    ----------
    file_path : str
        Path to the MCNP output file.
    tally_ids : iterable of str, optional
        Specific tally IDs to search for. If ``None``, all tally IDs found in the
        file after ``1tally`` markers are used.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(df_neutron, df_photon)`` where ``df_neutron`` contains merged incident
        and detected neutron tallies for each surface and ``df_photon`` contains
        photon tallies. Both DataFrames include a ``surface`` column identifying
        the associated surface number.
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Automatically collect tally IDs if none are provided
    if tally_ids is None:
        tally_ids = []
        for line in lines:
            match = re.search(r"1tally\s+(\d+)", line)
            if match:
                tally_ids.append(match.group(1))
        tally_ids = sorted(set(tally_ids))

    dataframes = {}

    for tally_id in tally_ids:
        start_index = None
        for i, line in enumerate(lines):
            if re.search(rf"1tally\s+{tally_id}\b", line):
                start_index = i
                break
        if start_index is not None:
            data_lines = []
            for j in range(start_index, len(lines)):
                if "energy" in lines[j].lower():
                    data_lines = lines[j + 1 :]
                    break
            parsed = []
            for line in data_lines:
                if not line.strip() or "total" in line.lower():
                    break
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        energy, value, error = map(float, parts)
                        parsed.append((energy, value, error))
                    except Exception:
                        continue
            if parsed:
                df = pd.DataFrame(parsed, columns=["energy", "value", "error"])
                dataframes[tally_id] = df

    incident = {tid[1:]: df for tid, df in dataframes.items() if tid.startswith("1")}
    detected = {tid[1:]: df for tid, df in dataframes.items() if tid.startswith("2")}
    photon = {tid[1:]: df for tid, df in dataframes.items() if tid.startswith("3")}

    neutron_frames = []
    for surf in sorted(set(incident.keys()) & set(detected.keys())):
        df = pd.merge(
            incident[surf],
            detected[surf],
            on="energy",
            suffixes=("_incident", "_detected"),
        )
        df.rename(
            columns={
                "value_incident": "neutrons_incident_cm2",
                "error_incident": "frac_error_incident_cm2",
                "value_detected": "neutrons_detected_cm2",
                "error_detected": "frac_error_detected_cm2",
            },
            inplace=True,
        )
        df["surface"] = int(surf)
        neutron_frames.append(df)

    df_combined = (
        pd.concat(neutron_frames, ignore_index=True)
        if neutron_frames
        else pd.DataFrame()
    )

    photon_frames = []
    for surf, df in photon.items():
        df = df.rename(
            columns={
                "energy": "photon_energy",
                "value": "photons",
                "error": "photon_error",
            }
        )
        df["surface"] = int(surf)
        photon_frames.append(df)

    df_photon = (
        pd.concat(photon_frames, ignore_index=True) if photon_frames else pd.DataFrame()
    )

    if df_combined.empty:
        logger.warning(f"No valid tally data found in {file_path}")

    return df_combined, df_photon


def calculate_rates(df, area, volume, neutron_yield):
    df["rate_incident"] = df["neutrons_incident_cm2"] * neutron_yield * area
    df["rate_detected"] = df["neutrons_detected_cm2"] * neutron_yield * volume

    rate_detected = df["rate_detected"].to_numpy()
    rate_incident = df["rate_incident"].to_numpy()
    df["efficiency"] = np.divide(
        rate_detected,
        rate_incident,
        out=np.zeros_like(rate_detected),
        where=rate_incident != 0,
    )

    df["rate_incident_err2"] = (
        df["frac_error_incident_cm2"] * df["rate_incident"]
    ) ** 2
    df["rate_detected_err2"] = (
        df["frac_error_detected_cm2"] * df["rate_detected"]
    ) ** 2
    df["rate_incident_err"] = np.sqrt(df["rate_incident_err2"])
    df["rate_detected_err"] = np.sqrt(df["rate_detected_err2"])

    rate_detected_err = df["rate_detected_err"].to_numpy()
    rate_incident_err = df["rate_incident_err"].to_numpy()
    term_detected = np.divide(
        rate_detected_err,
        rate_detected,
        out=np.zeros_like(rate_detected_err),
        where=rate_detected != 0,
    )
    term_incident = np.divide(
        rate_incident_err,
        rate_incident,
        out=np.zeros_like(rate_incident_err),
        where=rate_incident != 0,
    )
    df["efficiency_err"] = df["efficiency"] * np.sqrt(
        term_detected**2 + term_incident**2
    )
    return df


def process_simulation_file(file_path, area, volume, neutron_yield):
    df_neutron, _ = read_tally_blocks_to_df(file_path)
    if df_neutron.empty:
        return None
    return calculate_rates(df_neutron, area, volume, neutron_yield)


def export_summary_to_csv(df, filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_dir = os.path.dirname(filename)
    summary_data = {
        "Total Incident Neutron": [df["rate_incident"].sum()],
        "Incident Error": [np.sqrt(df["rate_incident_err2"].sum())],
        "Total Detected Neutron": [df["rate_detected"].sum()],
        "Detected Error": [np.sqrt(df["rate_detected_err2"].sum())],
    }
    summary_df = pd.DataFrame(summary_data)
    csv_path = get_output_path(
        base_dir, base_name, "summary", extension="csv", subfolder="csvs"
    )
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")


def calculate_chi_squared(obs, exp, obs_err, exp_err):
    chi2 = np.sum(((obs - exp) ** 2) / (obs_err**2 + exp_err**2))
    dof = max(len(obs) - 1, 0)
    return chi2, dof, chi2 / dof if dof > 0 else np.nan


def compute_best_scale_factor(simulated, observed, sigma):
    """Return the weighted best-fit scale factor between simulated and observed data."""

    simulated = np.asarray(simulated, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    valid = (sigma > 0) & np.isfinite(simulated) & np.isfinite(observed)
    if not np.any(valid):
        logger.warning("Cannot compute scale factor: no valid data points.")
        return 1.0

    inv_var = 1.0 / sigma[valid] ** 2
    numerator = np.sum(inv_var * simulated[valid] * observed[valid])
    denominator = np.sum(inv_var * simulated[valid] ** 2)
    if denominator == 0:
        logger.warning("Cannot compute scale factor: zero denominator.")
        return 1.0

    return numerator / denominator


def compute_thickness_residuals(combined_df, experimental_df):
    """Calculate residuals and chi-squared statistics for thickness scans."""

    residual_frames = []
    stats = []
    experimental_df = _sort_and_deduplicate_thickness(experimental_df, "Experimental data")
    experimental_thickness = set(experimental_df["thickness"])

    for dataset in combined_df["dataset"].unique():
        df_label = combined_df[combined_df["dataset"] == dataset]
        df_label = _sort_and_deduplicate_thickness(df_label, f"Simulated dataset '{dataset}'")

        missing_in_experiment = sorted(set(df_label["thickness"]) - experimental_thickness)
        missing_in_simulation = sorted(experimental_thickness - set(df_label["thickness"]))
        if missing_in_experiment:
            logger.warning(
                "Simulated points without experimental match for %s: %s", dataset, missing_in_experiment
            )
        if missing_in_simulation:
            logger.warning(
                "Experimental points without simulated match for %s: %s", dataset, missing_in_simulation
            )

        try:
            merged = pd.merge(
                df_label,
                experimental_df,
                on="thickness",
                how="inner",
                sort=True,
                validate="one_to_one",
            )
        except ValueError as exc:
            logger.warning(
                "Unable to validate one-to-one merge for %s (%s); using first occurrences only",
                dataset,
                exc,
            )
            merged = pd.merge(
                df_label.drop_duplicates(subset=["thickness"]),
                experimental_df.drop_duplicates(subset=["thickness"]),
                on="thickness",
                how="inner",
                sort=True,
            )

        if merged.empty:
            logger.warning("No overlapping thickness values for %s; skipping residuals", dataset)
            continue
        merged = merged.sort_values("thickness").reset_index(drop=True)
        merged["combined_uncertainty"] = np.sqrt(
            merged["simulated_error"] ** 2 + merged["error_cps"] ** 2
        )
        merged["raw_residual_unscaled"] = merged["cps"] - merged["simulated_detected"]
        merged["relative_residual_pct_unscaled"] = np.where(
            merged["cps"] != 0,
            100 * merged["raw_residual_unscaled"] / merged["cps"],
            np.nan,
        )
        merged["standardised_residual_unscaled"] = np.divide(
            merged["raw_residual_unscaled"],
            merged["combined_uncertainty"],
            out=np.full_like(merged["raw_residual_unscaled"], np.nan, dtype=float),
            where=merged["combined_uncertainty"] != 0,
        )

        chi_squared_before = np.nansum(
            merged["standardised_residual_unscaled"] ** 2
        )
        valid_before = np.isfinite(merged["standardised_residual_unscaled"])
        dof_before = max(valid_before.sum() - 1, 0)
        reduced_chi_squared_before = (
            chi_squared_before / dof_before if dof_before > 0 else np.nan
        )

        scale_factor = compute_best_scale_factor(
            merged["simulated_detected"],
            merged["cps"],
            merged["combined_uncertainty"],
        )
        merged["scale_factor"] = scale_factor
        merged["scaled_simulated_detected"] = merged["simulated_detected"] * scale_factor
        merged["scaled_simulated_error"] = merged["simulated_error"] * scale_factor
        merged["combined_uncertainty_scaled"] = np.sqrt(
            merged["scaled_simulated_error"] ** 2 + merged["error_cps"] ** 2
        )
        merged["raw_residual_scaled"] = merged["cps"] - merged["scaled_simulated_detected"]
        merged["relative_residual_pct_scaled"] = np.where(
            merged["cps"] != 0,
            100 * merged["raw_residual_scaled"] / merged["cps"],
            np.nan,
        )
        merged["standardised_residual_scaled"] = np.divide(
            merged["raw_residual_scaled"],
            merged["combined_uncertainty_scaled"],
            out=np.full_like(merged["raw_residual_scaled"], np.nan, dtype=float),
            where=merged["combined_uncertainty_scaled"] != 0,
        )

        chi_squared_after = np.nansum(merged["standardised_residual_scaled"] ** 2)
        valid_after = np.isfinite(merged["standardised_residual_scaled"])
        dof_after = max(valid_after.sum() - 1, 0)
        reduced_chi_squared_after = (
            chi_squared_after / dof_after if dof_after > 0 else np.nan
        )
        stats.append(
            {
                "dataset": dataset,
                "scale_factor": scale_factor,
                "chi_squared_before": chi_squared_before,
                "dof_before": dof_before,
                "reduced_chi_squared_before": reduced_chi_squared_before,
                "chi_squared_after": chi_squared_after,
                "dof_after": dof_after,
                "reduced_chi_squared_after": reduced_chi_squared_after,
            }
        )

        residual_frames.append(
            merged[
                [
                    "thickness",
                    "raw_residual_unscaled",
                    "relative_residual_pct_unscaled",
                    "standardised_residual_unscaled",
                    "raw_residual_scaled",
                    "relative_residual_pct_scaled",
                    "standardised_residual_scaled",
                    "scaled_simulated_detected",
                    "scaled_simulated_error",
                    "combined_uncertainty",
                    "combined_uncertainty_scaled",
                    "scale_factor",
                    "dataset",
                ]
            ]
        )

    residuals_df = (
        pd.concat(residual_frames, ignore_index=True).sort_values(["dataset", "thickness"]).reset_index(drop=True)
        if residual_frames
        else pd.DataFrame()
    )
    stats_df = pd.DataFrame(stats)
    return residuals_df, stats_df


def parse_thickness_from_filename(filename):
    """Extract thickness from an output filename.

    Accepts common MCNP naming patterns, case-insensitively:
    - "..._10o", "..._10cm.o", "..._10cmo"
    - Also tolerates no underscore separator (e.g., "...10cmo") and
      decimal thickness using '_' or '.' as a separator (e.g., "..._2_5cmo").

    Returns an ``int`` when the thickness is a whole number, otherwise a
    ``float``; returns ``None`` if no thickness token is found.
    """
    base = os.path.basename(filename)
    # Look for a number before optional 'cm' and a trailing output extension
    # Example matches: _10o, _10cmo, _10cm.o, _10cm.out, 10cmo, 2_5cmo
    m = re.search(
        r"(\d+(?:[._]\d+)?)\s*(?:cm)?(?:\.o|o|\.out|out)$",
        base,
        re.IGNORECASE,
    )
    if not m:
        return None
    token = m.group(1).replace("_", ".")
    try:
        value = float(token)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


# Geometry constants
_DEFAULT_GEOM = DETECTORS[DEFAULT_DETECTOR]
LENGTH_CM = _DEFAULT_GEOM.length_cm
RADIUS_CM = _DEFAULT_GEOM.radius_cm
DIAMETER_CM = RADIUS_CM * 2.0
AREA = _DEFAULT_GEOM.area
VOLUME = _DEFAULT_GEOM.volume

EXP_RATE = 247.0333333
EXP_ERR = 0.907438397

SINGLE_SOURCE_YIELD = 2.5e6
THREE_SOURCE_YIELD = (2.5e6) + (1.25e6) + (7.5e6)


# Neutron yield selection logic --------------------------------------------
def select_neutron_yield():
    choice = input(
        "Select neutron yield configuration:\n"
        "1. Single source (2.5e6 n/s)\n"
        "2. Three sources (weighted sum)\n"
        "Enter 1 or 2: "
    ).strip()
    if choice == "1":
        return SINGLE_SOURCE_YIELD
    if choice == "2":
        return THREE_SOURCE_YIELD
    logger.warning("Invalid selection. Defaulting to single source (2.5e6 n/s).")
    return SINGLE_SOURCE_YIELD


def prompt_for_valid_file(title="Select MCNP Output File"):
    while True:
        file_path = select_file(title)
        if not file_path:
            logger.warning("No file selected. Please try again.")
            continue
        result = read_tally_blocks_to_df(file_path)
        if result is None:
            logger.warning(
                "Invalid file selected. No tally data found. Please select another file."
            )
            continue
        df_neutron, _ = result
        if df_neutron is not None and not df_neutron.empty:
            return file_path, result
        logger.warning(
            "Invalid file selected. No tally data found. Please select another file."
        )


# Analysis entry points -----------------------------------------------------


def run_analysis_type_1(file_path, area, volume, neutron_yield, export_csv=True):
    df = process_simulation_file(file_path, area, volume, neutron_yield)
    if export_csv:
        df_neutron, df_photon = read_tally_blocks_to_df(file_path)
        if df_neutron is not None and not df_neutron.empty:
            neutron_csv_path = get_output_path(
                os.path.dirname(file_path),
                os.path.splitext(os.path.basename(file_path))[0],
                "neutron tallies",
                extension="csv",
                subfolder="csvs",
            )
            df_neutron.to_csv(neutron_csv_path, index=False)
            logger.info(f"Saved: {neutron_csv_path}")
        if df_photon is not None and not df_photon.empty:
            photon_csv_path = get_output_path(
                os.path.dirname(file_path),
                os.path.splitext(os.path.basename(file_path))[0],
                "photon tally",
                extension="csv",
                subfolder="csvs",
            )
            df_photon.to_csv(photon_csv_path, index=False)
            logger.info(f"Saved: {photon_csv_path}")
    if df is None:
        return
    logger.info(
        f"Total Incident Neutron: {df['rate_incident'].sum():.3e} ± {np.sqrt(df['rate_incident_err2'].sum()):.3e}"
    )
    logger.info(
        f"Total Detected Neutron: {df['rate_detected'].sum():.3e} ± {np.sqrt(df['rate_detected_err2'].sum()):.3e}"
    )
    plot_paths = plot_efficiency_and_rates(df, file_path)
    if export_csv:
        export_summary_to_csv(df, file_path)
    return df, plot_paths


def run_analysis_type_2(
    folder_paths,
    labels=None,
    lab_data_path=None,
    area=AREA,
    volume=VOLUME,
    neutron_yield=1.0,
    export_csv=True,
):
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    if labels is None:
        labels = [os.path.basename(p.rstrip("/")) for p in folder_paths]

    experimental_df = None
    if lab_data_path:
        experimental_df = pd.read_csv(lab_data_path)
        experimental_df.columns = experimental_df.columns.str.strip()
        if "thickness" in experimental_df.columns:
            experimental_df = _sort_and_deduplicate_thickness(
                experimental_df, "Experimental data"
            )

    all_results = []
    for folder_path, label in zip(folder_paths, labels):
        results = []
        for filename in os.listdir(folder_path):
            # Accept common MCNP output extensions like '.o' and '.out'
            if not filename.lower().endswith(("o", "out")):
                continue
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                continue
            thickness = parse_thickness_from_filename(filename)
            if thickness is None:
                continue
            result = read_tally_blocks_to_df(file_path)
            if result is None:
                continue
            df_neutron, _ = result
            if df_neutron.empty:
                continue
            df = calculate_rates(df_neutron, area, volume, neutron_yield)
            total_detected = df["rate_detected"].sum()
            total_error = np.sqrt(df["rate_detected_err2"].sum())
            results.append(
                {
                    "thickness": thickness,
                    "simulated_detected": total_detected,
                    "simulated_error": total_error,
                    "dataset": label,
                }
            )
        if results:
            all_results.append(pd.DataFrame(results).sort_values(by="thickness"))

    if not all_results:
        logger.warning("No matching simulated CSV files found in folders.")
        return

    combined_df = (
        pd.concat(all_results, ignore_index=True)
        .sort_values(["dataset", "thickness"])
        .reset_index(drop=True)
    )

    residuals_df = pd.DataFrame()
    residual_stats = pd.DataFrame()
    if export_csv:
        base_dir = os.path.commonpath(folder_paths)
        csv_path = get_output_path(
            base_dir,
            "multi_thickness",
            "comparison data",
            extension="csv",
            subfolder="csvs",
        )
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
    if experimental_df is not None:
        residuals_df, residual_stats = compute_thickness_residuals(
            combined_df, experimental_df
        )
        for _, row in residual_stats.iterrows():
            logger.info(
                f"{row['dataset']}: k = {row['scale_factor']:.3g}, "
                f"Chi-squared (before) = {row['chi_squared_before']:.2f}, "
                f"Chi-squared (scaled) = {row['chi_squared_after']:.2f}, "
                f"DoF (before/after) = {int(row['dof_before'])}/{int(row['dof_after'])}, "
                f"Reduced $\\chi^2_\\nu$ (before/after) = "
                f"{row['reduced_chi_squared_before']:.2f}/"
                f"{row['reduced_chi_squared_after']:.2f}"
            )

    experimental_df_local = experimental_df
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "h", "*"]
    for i, label in enumerate(combined_df["dataset"].unique()):
        df_label = combined_df[combined_df["dataset"] == label]
        color = f"C{i}"
        ax.errorbar(
            df_label["thickness"],
            df_label["simulated_detected"],
            yerr=df_label["simulated_error"],
            fmt=markers[i % len(markers)],
            linestyle="-",
            label=label,
            capsize=5,
            color=color,
        )
        scaled_df = pd.DataFrame()
        if not residuals_df.empty:
            scaled_df = (
                residuals_df[residuals_df["dataset"] == label]
                .drop_duplicates(subset=["thickness"])
                .sort_values("thickness")
            )
        if not scaled_df.empty:
            ax.errorbar(
                scaled_df["thickness"],
                scaled_df["scaled_simulated_detected"],
                yerr=scaled_df.get("scaled_simulated_error"),
                fmt=markers[i % len(markers)],
                linestyle="--",
                label=f"{label} (scaled)",
                capsize=5,
                color=color,
                alpha=0.9,
            )
    if experimental_df_local is not None:
        ax.errorbar(
            experimental_df_local["thickness"],
            experimental_df_local["cps"],
            yerr=experimental_df_local["error_cps"],
            fmt="k--",
            label="Experimental",
            capsize=5,
        )
        ax.plot(
            experimental_df_local["thickness"],
            experimental_df_local["cps"],
            linestyle="--",
            color="black",
        )
        if config.show_fig_heading:
            tag = (
                f" - {config.filename_tag.strip()}"
                if config.filename_tag.strip()
                else ""
            )
            ax.set_title(f"Simulated vs Experimental Neutron Detection{tag}")
    else:
        if config.show_fig_heading:
            tag = (
                f" - {config.filename_tag.strip()}"
                if config.filename_tag.strip()
                else ""
            )
            ax.set_title(f"Simulated Neutron Detection{tag}")
    ax.set_xlabel("Moderator Thickness (cm)", fontsize=config.axis_label_fontsize)
    ax.set_ylabel("Count Rate, (Counts/s)", fontsize=config.axis_label_fontsize)
    ax.grid(config.show_grid)
    ax.legend(fontsize=config.legend_fontsize)
    ax.tick_params(labelsize=config.tick_label_fontsize)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    base_dir = os.path.commonpath(folder_paths)
    save_path = get_output_path(
        base_dir, "multi_thickness", "comparison plot", subfolder="plots"
    )
    fig.savefig(save_path)
    fig.clf()
    logger.info(f"Saved: {save_path}")

    residual_plot_path = None
    if experimental_df_local is not None and not residuals_df.empty:
        resid_fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(resid_fig)
        ax_resid = resid_fig.add_subplot(111)
        for i, label in enumerate(combined_df["dataset"].unique()):
            df_resid = residuals_df[residuals_df["dataset"] == label]
            if df_resid.empty:
                continue
            color = f"C{i}"
            ax_resid.plot(
                df_resid["thickness"],
                df_resid["standardised_residual_unscaled"],
                marker=markers[i % len(markers)],
                linestyle=":",
                label=f"{label} (before)",
                color=color,
            )
            ax_resid.plot(
                df_resid["thickness"],
                df_resid["standardised_residual_scaled"],
                marker=markers[i % len(markers)],
                linestyle="-",
                label=f"{label} (scaled)",
                color=color,
            )

        for level, style in zip([0, 1, 2, 3], ["-", "--", ":", ":"]):
            ax_resid.axhline(y=level, color="gray", linestyle=style, linewidth=1)
            if level:
                ax_resid.axhline(
                    y=-level, color="gray", linestyle=style, linewidth=1
                )
        if not residual_stats.empty:
            text_lines = [
                (
                    f"{row['dataset']}: k = {row['scale_factor']:.3g}, "
                    f"$\\chi^2_\\nu$ before = {row['reduced_chi_squared_before']:.2f}, "
                    f"after = {row['reduced_chi_squared_after']:.2f}"
                )
                for _, row in residual_stats.iterrows()
                if row.get("dof_after", 0) > 0
            ]
            if text_lines:
                ax_resid.text(
                    0.02,
                    0.95,
                    "\n".join(text_lines),
                    transform=ax_resid.transAxes,
                    va="top",
                    ha="left",
                    fontsize=config.legend_fontsize,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
                )

        ax_resid.set_xlabel("Moderator Thickness (cm)", fontsize=config.axis_label_fontsize)
        ax_resid.set_ylabel(
            "Standardised Residual, z", fontsize=config.axis_label_fontsize
        )
        ax_resid.grid(config.show_grid)
        ax_resid.legend(fontsize=config.legend_fontsize)
        ax_resid.tick_params(labelsize=config.tick_label_fontsize)
        resid_fig.tight_layout()

        residual_plot_path = get_output_path(
            base_dir, "multi_thickness", "residuals plot", subfolder="plots"
        )
        resid_fig.savefig(residual_plot_path)
        resid_fig.clf()
        logger.info(f"Saved: {residual_plot_path}")

    return (
        combined_df,
        experimental_df_local,
        save_path,
        residuals_df,
        residual_plot_path,
        residual_stats,
    )


def run_analysis_type_3(
    folder_path,
    area,
    volume,
    neutron_yield,
    export_csv=True,
):
    results = []
    for filename in os.listdir(folder_path):
        if not filename.endswith("o"):
            continue
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            match = re.search(r"([-+]?\d+_\d+|\d+)", filename)
            if match:
                distance_str = match.group(1).replace("_", ".")
                try:
                    distance = float(distance_str)
                except ValueError:
                    continue
                result = read_tally_blocks_to_df(file_path)
                if result is None:
                    continue
                df_neutron, _ = result
                df = calculate_rates(df_neutron, area, volume, neutron_yield)
                total_detected = df["rate_detected"].sum()
                total_error = np.sqrt(df["rate_detected_err2"].sum())
                results.append(
                    {
                        "distance": distance,
                        "rate_detected": total_detected,
                        "rate_error": total_error,
                    }
                )
    if not results:
        logger.warning("No matching simulated CSV files found in folder.")
        return
    distance_df = pd.DataFrame(results).sort_values(by="distance")

    folder_name = os.path.basename(folder_path.rstrip("/"))
    if export_csv:
        csv_path = get_output_path(
            folder_path,
            folder_name,
            "source shift data",
            extension="csv",
            subfolder="csvs",
        )
        distance_df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")

    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.errorbar(
        distance_df["distance"],
        distance_df["rate_detected"],
        yerr=distance_df["rate_error"],
        fmt="o",
        capsize=5,
        label="Simulated",
    )
    fit_coeffs, cov_matrix = np.polyfit(
        distance_df["distance"], distance_df["rate_detected"], 1, cov=True
    )
    slope, intercept = fit_coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov_matrix))
    fitted_values = np.polyval(fit_coeffs, distance_df["distance"])
    residuals = distance_df["rate_detected"] - fitted_values
    chi_squared_fit = np.sum((residuals / distance_df["rate_error"]) ** 2)
    dof_fit = len(distance_df) - 2
    reduced_chi_squared_fit = chi_squared_fit / dof_fit
    logger.info(f"Chi-squared of linear fit: {chi_squared_fit:.2f}")
    logger.info(f"Degrees of freedom: {dof_fit}")
    logger.info(f"Reduced Chi-squared: {reduced_chi_squared_fit:.2f}")
    if slope != 0:
        x_intersect = (EXP_RATE - intercept) / slope
        x_intersect_err = x_intersect * np.sqrt(
            (intercept_err / (EXP_RATE - intercept)) ** 2 + (slope_err / slope) ** 2
        )
        logger.info(
            f"Intersection of fit line with experimental value at x = {x_intersect:.3f} ± {x_intersect_err:.3f} cm"
        )
        extended_margin = 1.0
        min_x = min(distance_df["distance"].min(), x_intersect - extended_margin)
        max_x = max(distance_df["distance"].max(), x_intersect + extended_margin)
        ax.set_xlim(min_x, max_x)
        ax.plot(distance_df["distance"], fitted_values, linestyle="--", label="Fit")
        ax.axvline(x=x_intersect, color="gray", linestyle=":", label="Fit vs Exp")
        ax.text(
            x_intersect,
            EXP_RATE,
            f"{x_intersect:.3f}±{x_intersect_err:.3f}",
            fontsize=10,
            color="black",
        )
    else:
        logger.warning(
            "Fit line is horizontal; no intersection with experimental line."
        )
    ax.axhline(
        y=EXP_RATE, color="red", linestyle="--", label=f"Experimental = {EXP_RATE:.2e}"
    )
    ax.axhspan(
        EXP_RATE - EXP_ERR,
        EXP_RATE + EXP_ERR,
        color="red",
        alpha=0.2,
        label="Experimental Uncertainty",
    )
    ax.set_xlabel("Source Displacement (cm)", fontsize=config.axis_label_fontsize)
    ax.set_ylabel("Total Detected Rate", fontsize=config.axis_label_fontsize)
    if config.show_fig_heading:
        tag = f" - {config.filename_tag.strip()}" if config.filename_tag.strip() else ""
        ax.set_title(f"Detected Rate vs Source Displacement{tag}")
    ax.grid(config.show_grid)
    ax.legend(fontsize=config.legend_fontsize)
    ax.tick_params(labelsize=config.tick_label_fontsize)
    fig.tight_layout()
    ax.text(
        0.98,
        0.02,
        f"$\\chi^2_\\nu$ = {reduced_chi_squared_fit:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    save_path = get_output_path(
        folder_path, folder_name, "source shift plot", subfolder="plots"
    )
    fig.savefig(save_path)
    fig.clf()
    logger.info(f"Saved: {save_path}")
    metadata = {
        "distance_df": distance_df,
        "fit_coeffs": fit_coeffs,
        "fitted_values": fitted_values,
        "chi_squared": chi_squared_fit,
        "dof": dof_fit,
        "reduced_chi_squared": reduced_chi_squared_fit,
        "x_intersect": locals().get("x_intersect"),
        "x_intersect_err": locals().get("x_intersect_err"),
        "slope": slope,
        "intercept": intercept,
        "slope_err": slope_err,
        "intercept_err": intercept_err,
        "exp_rate": EXP_RATE,
        "exp_err": EXP_ERR,
    }
    return metadata, save_path


def run_analysis_type_4(file_path, export_csv=True):
    _, df_photon = read_tally_blocks_to_df(file_path)
    if df_photon is None or df_photon.empty:
        logger.warning("No photon tally data found.")
        return

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_dir = os.path.dirname(file_path)
    if export_csv:
        photon_csv_path = get_output_path(
            base_dir, base_name, "photon tally", extension="csv", subfolder="csvs"
        )
        df_photon.to_csv(photon_csv_path, index=False)
        logger.info(f"Saved: {photon_csv_path}")

    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(df_photon["photon_energy"], df_photon["photons"], label="Photons")
    ax.set_xlabel("Photon Energy (MeV)", fontsize=config.axis_label_fontsize)
    ax.set_ylabel("Photon Counts", fontsize=config.axis_label_fontsize)
    if config.show_fig_heading:
        tag = f" - {config.filename_tag.strip()}" if config.filename_tag.strip() else ""
        ax.set_title(f"Photon Tally (Tally 34){tag}")
    ax.grid(config.show_grid)
    ax.legend(fontsize=config.legend_fontsize)
    ax.tick_params(labelsize=config.tick_label_fontsize)
    fig.tight_layout()

    save_path = get_output_path(
        base_dir, base_name, "photon tally plot", subfolder="plots"
    )
    fig.savefig(save_path)
    fig.clf()
    logger.info(f"Saved: {save_path}")
    return df_photon, save_path
