import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib

# Force the non-interactive Agg backend so plots can be generated safely when
# this module is executed from a background thread.  The GUI imports elsewhere
# in the application may have already selected an interactive backend (e.g.
# ``TkAgg``), and attempting to create figures from a worker thread with that
# backend triggers ``UserWarning: Starting a Matplotlib GUI outside of the main
# thread will likely fail``.  Using ``force=True`` ensures we override any
# previously chosen backend without relying on global import order.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from .io_utils import get_output_path, select_file
from .plots import plot_efficiency_and_rates
from .detectors import DETECTORS, DEFAULT_DETECTOR
from .config import config

logger = logging.getLogger(__name__)


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
    dof = len(obs)
    return chi2, dof, chi2 / dof if dof > 0 else np.nan


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

    combined_df = pd.concat(all_results, ignore_index=True)

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
        for label in combined_df["dataset"].unique():
            df_label = combined_df[combined_df["dataset"] == label]
            merged = pd.merge(df_label, experimental_df, on="thickness")
            if merged.empty:
                continue
            chi_squared, dof, reduced_chi_squared = calculate_chi_squared(
                merged["simulated_detected"],
                merged["cps"],
                merged["simulated_error"],
                merged["error_cps"],
            )
            logger.info(
                f"{label}: Chi-squared: {chi_squared:.2f}, DoF: {dof}, Reduced Chi-squared: {reduced_chi_squared:.2f}"
            )

    experimental_df_local = experimental_df
    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "^", "d", "v", "<", ">", "p", "h", "*"]
    for i, label in enumerate(combined_df["dataset"].unique()):
        df_label = combined_df[combined_df["dataset"] == label]
        plt.errorbar(
            df_label["thickness"],
            df_label["simulated_detected"],
            yerr=df_label["simulated_error"],
            fmt=markers[i % len(markers)],
            linestyle="-",
            label=label,
            capsize=5,
        )
    if experimental_df_local is not None:
        plt.errorbar(
            experimental_df_local["thickness"],
            experimental_df_local["cps"],
            yerr=experimental_df_local["error_cps"],
            fmt="k--",
            label="Experimental",
            capsize=5,
        )
        plt.plot(
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
            plt.title(f"Simulated vs Experimental Neutron Detection{tag}")
    else:
        if config.show_fig_heading:
            tag = (
                f" - {config.filename_tag.strip()}"
                if config.filename_tag.strip()
                else ""
            )
            plt.title(f"Simulated Neutron Detection{tag}")
    plt.xlabel("Moderator Thickness (cm)")
    plt.ylabel("Count Rate, (Counts/s)")
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()

    base_dir = os.path.commonpath(folder_paths)
    save_path = get_output_path(
        base_dir, "multi_thickness", "comparison plot", subfolder="plots"
    )
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")
    return combined_df, experimental_df_local, save_path


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

    plt.figure(figsize=(10, 6))
    plt.errorbar(
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
        plt.xlim(min_x, max_x)
        plt.plot(distance_df["distance"], fitted_values, linestyle="--", label="Fit")
        plt.axvline(x=x_intersect, color="gray", linestyle=":", label="Fit vs Exp")
        plt.text(
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
    plt.axhline(
        y=EXP_RATE, color="red", linestyle="--", label=f"Experimental = {EXP_RATE:.2e}"
    )
    plt.axhspan(
        EXP_RATE - EXP_ERR,
        EXP_RATE + EXP_ERR,
        color="red",
        alpha=0.2,
        label="Experimental Uncertainty",
    )
    plt.xlabel("Source Displacement (cm)")
    plt.ylabel("Total Detected Rate")
    if config.show_fig_heading:
        tag = f" - {config.filename_tag.strip()}" if config.filename_tag.strip() else ""
        plt.title(f"Detected Rate vs Source Displacement{tag}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.text(
        0.98,
        0.02,
        f"$\\chi^2_\\nu$ = {reduced_chi_squared_fit:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    save_path = get_output_path(
        folder_path, folder_name, "source shift plot", subfolder="plots"
    )
    plt.savefig(save_path)
    plt.close()
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

    plt.figure(figsize=(10, 6))
    plt.plot(df_photon["photon_energy"], df_photon["photons"], label="Photons")
    plt.xlabel("Photon Energy (MeV)")
    plt.ylabel("Photon Counts")
    if config.show_fig_heading:
        tag = f" - {config.filename_tag.strip()}" if config.filename_tag.strip() else ""
        plt.title(f"Photon Tally (Tally 34){tag}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = get_output_path(
        base_dir, base_name, "photon tally plot", subfolder="plots"
    )
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")
    return df_photon, save_path
