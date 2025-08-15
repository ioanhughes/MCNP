import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI blocking

# Global toggle for exporting CSVs
EXPORT_CSV = True
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from datetime import datetime
import logging
import atexit

logger = logging.getLogger(__name__)

# ---- Utility Functions ----
_hidden_root = None


def _get_hidden_root():
    """Create a single transparent Tk root for file dialogs."""
    global _hidden_root
    if _hidden_root is None:
        _hidden_root = Tk()
        _hidden_root.attributes("-alpha", 0)
        _hidden_root.withdraw()
        atexit.register(_hidden_root.destroy)
    return _hidden_root


def _show_dialog(dialog_func, **kwargs):
    """Show a file dialog using a hidden root near the cursor position."""
    root = _get_hidden_root()
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    root.geometry(f"+{root.winfo_pointerx()}+{root.winfo_pointery()}")
    try:
        return dialog_func(parent=root, **kwargs)
    finally:
        root.attributes("-topmost", False)
        root.withdraw()


def select_file(title="Select a file"):
    return _show_dialog(askopenfilename, title=title)


def select_folder(title="Select a folder"):
    return _show_dialog(askdirectory, title=title)

def make_plot_dir(base_path):
    plot_dir = os.path.join(base_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

# Utility to get output path and ensure directory exists
def get_output_path(base_path, filename_prefix, descriptor, extension="pdf", subfolder="plots"):
    output_dir = os.path.join(base_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{filename_prefix} {descriptor} {date_str}.{extension}"
    return os.path.join(output_dir, filename)

def process_simulation_file(file_path, area, volume, neutron_yield):
    df_neutron, _ = read_tally_blocks_to_df(file_path)
    if df_neutron.empty:
        return None
    return calculate_rates(df_neutron, area, volume, neutron_yield)

# ---- Function to read MCNP tally blocks and convert to DataFrame ----
def read_tally_blocks_to_df(file_path, tally_ids=("14", "24", "34"), context_lines=30):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    dataframes = {}

    for tally_id in tally_ids:
        start_index = None
        for i, line in enumerate(lines):
            if re.search(rf"1tally\s+{tally_id}\b", line):
                start_index = i
                break
        if start_index is not None:
            for j in range(start_index, len(lines)):
                if "energy" in lines[j].lower():
                    data_lines = lines[j+1:]
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
                    except:
                        continue
            df = pd.DataFrame(parsed, columns=["energy", "value", "error"])
            dataframes[tally_id] = df

    if "14" not in dataframes or "24" not in dataframes:
        # If required tallies are missing, return empty DataFrames to avoid
        # unpacking errors in calling code.
        logger.warning(f"No valid tally data found in {file_path}")
        return pd.DataFrame(), pd.DataFrame()

    df_combined = pd.merge(dataframes["14"], dataframes["24"], on="energy", suffixes=("_incident", "_detected"))
    df_combined.rename(columns={
        "value_incident": "neutrons_incident_cm2",
        "error_incident": "frac_error_incident_cm2",
        "value_detected": "neutrons_detected_cm2",
        "error_detected": "frac_error_detected_cm2"
    }, inplace=True)
    df_photon = dataframes["34"].rename(columns={
        "energy": "photon_energy",
        "value": "photons",
        "error": "photon_error"
    }) if "34" in dataframes else pd.DataFrame()
    return df_combined, df_photon

# ---- Function to calculate rates and propagated errors ----
def calculate_rates(df, area, volume, neutron_yield):
    df["rate_incident"] = df["neutrons_incident_cm2"] * neutron_yield * area
    df["rate_detected"] = df["neutrons_detected_cm2"] * neutron_yield * volume
    df["efficiency"] = df["rate_detected"] / df["rate_incident"]

    df["rate_incident_err2"] = (df["frac_error_incident_cm2"] * df["rate_incident"]) ** 2
    df["rate_detected_err2"] = (df["frac_error_detected_cm2"] * df["rate_detected"]) ** 2
    df["rate_incident_err"] = np.sqrt(df["rate_incident_err2"])
    df["rate_detected_err"] = np.sqrt(df["rate_detected_err2"])

    df["efficiency_err"] = df["efficiency"] * np.sqrt(
        (df["rate_detected_err"] / df["rate_detected"]) ** 2 +
        (df["rate_incident_err"] / df["rate_incident"]) ** 2
    )
    return df

# ---- Function to plot and save neutron rate and efficiency curves ----
def plot_efficiency_and_rates(df, filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_dir = os.path.dirname(filename)

    plt.figure(figsize=(10, 6))
    plt.errorbar(df["energy"], df["rate_incident"], yerr=df["rate_incident_err"], label="Incident Rate", fmt='o-', markersize=3, capsize=2)
    plt.errorbar(df["energy"], df["rate_detected"], yerr=df["rate_detected_err"], label="Detected Rate", fmt='s-', markersize=3, capsize=2)
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Neutron Rate")
    plt.title("Neutron Rates vs Energy")
    plt.legend()
    plt.grid(True)
    plt.semilogx()
    plt.tight_layout()
    rate_path = get_output_path(base_dir, base_name, "Neutron rate plot", extension="pdf", subfolder="plots")
    plt.savefig(rate_path)
    plt.close()
    logger.info(f"Saved: {rate_path}")

    plt.figure(figsize=(8, 6))
    plt.errorbar(df["energy"], df["efficiency"], yerr=df["efficiency_err"], fmt='^-', color='green', markersize=3, capsize=2)
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Detection Efficiency")
    plt.title("Efficiency vs Energy")
    plt.grid(True)
    plt.semilogx()
    plt.tight_layout()
    eff_path = get_output_path(base_dir, base_name, "efficiency curve", extension="pdf", subfolder="plots")
    plt.savefig(eff_path)
    plt.close()
    logger.info(f"Saved: {eff_path}")
    # plt.show()


# ---- Function to export neutron analysis summary to CSV ----
def export_summary_to_csv(df, filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_dir = os.path.dirname(filename)
    summary_data = {
        "Total Incident Neutron": [df["rate_incident"].sum()],
        "Incident Error": [np.sqrt(df["rate_incident_err2"].sum())],
        "Total Detected Neutron": [df["rate_detected"].sum()],
        "Detected Error": [np.sqrt(df["rate_detected_err2"].sum())]
    }
    summary_df = pd.DataFrame(summary_data)
    csv_path = get_output_path(base_dir, base_name, "summary", extension="csv", subfolder="csvs")
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

# ---- Function to compute chi-squared statistics between observed and expected values ----
def calculate_chi_squared(obs, exp, obs_err, exp_err):
    chi2 = np.sum(((obs - exp) ** 2) / (obs_err ** 2 + exp_err ** 2))
    dof = len(obs)
    return chi2, dof, chi2 / dof if dof > 0 else np.nan

# ---- Function to extract moderator thickness from filename ----
def parse_thickness_from_filename(filename):
    match = re.search(r'_(\d+)cmo', filename)
    return int(match.group(1)) if match else None

# ---- Define constants ----
SINGLE_SOURCE_YIELD = 2.5e6
THREE_SOURCE_YIELD = (2.5e6) + (1.25e6) + (7.5e6)

LENGTH_CM = 100.0     # Length of the cylinder in cm
DIAMETER_CM = 5.0     # Diameter of the cylinder in cm
RADIUS_CM = DIAMETER_CM / 2.0
AREA = LENGTH_CM * DIAMETER_CM
VOLUME = np.pi * (RADIUS_CM ** 2) * LENGTH_CM

EXP_RATE = 247.0333333
EXP_ERR = 0.907438397


# ---- Neutron yield selection logic ----
def select_neutron_yield():
    choice = input(
        "Select neutron yield configuration:\n"
        "1. Single source (2.5e6 n/s)\n"
        "2. Three sources (weighted sum)\n"
        "Enter 1 or 2: "
    ).strip()
    if choice == "1":
        return SINGLE_SOURCE_YIELD
    elif choice == "2":
        return THREE_SOURCE_YIELD
    else:
        logger.warning("Invalid selection. Defaulting to single source (2.5e6 n/s).")
        return SINGLE_SOURCE_YIELD

# ---- Start user interaction loop ----
def prompt_for_valid_file(title="Select MCNP Output File"):
    while True:
        file_path = select_file(title)
        if not file_path:
            logger.warning("No file selected. Please try again.")
            continue
        result = read_tally_blocks_to_df(file_path)
        if result is not None:
            return file_path, result
        logger.error("Invalid file selected. No tally data found. Please select another file.")

def run_analysis_type_1(file_path, area, volume, neutron_yield):
    df = process_simulation_file(file_path, area, volume, neutron_yield)
    # --- Export neutron and photon tally blocks to CSV ---
    if EXPORT_CSV:
        df_neutron, df_photon = read_tally_blocks_to_df(file_path)
        if df_neutron is not None and not df_neutron.empty:
            neutron_csv_path = get_output_path(
                os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0],
                "neutron tallies", extension="csv", subfolder="csvs"
            )
            df_neutron.to_csv(neutron_csv_path, index=False)
            logger.info(f"Saved: {neutron_csv_path}")
        if df_photon is not None and not df_photon.empty:
            photon_csv_path = get_output_path(
                os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0],
                "photon tally", extension="csv", subfolder="csvs"
            )
            df_photon.to_csv(photon_csv_path, index=False)
            logger.info(f"Saved: {photon_csv_path}")
    if df is None:
        return
    logger.info(f"Total Incident Neutron: {df['rate_incident'].sum():.3e} ± {np.sqrt(df['rate_incident_err2'].sum()):.3e}")
    logger.info(f"Total Detected Neutron: {df['rate_detected'].sum():.3e} ± {np.sqrt(df['rate_detected_err2'].sum()):.3e}")
    plot_efficiency_and_rates(df, file_path)
    if EXPORT_CSV:
        export_summary_to_csv(df, file_path)

def run_analysis_type_2(folder_path, lab_data_path, area, volume, neutron_yield):
    experimental_df = pd.read_csv(lab_data_path)
    experimental_df.columns = experimental_df.columns.str.strip()
    results = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            thickness = parse_thickness_from_filename(filename)
            if thickness is not None:
                result = read_tally_blocks_to_df(file_path)
                if result is None:
                    continue
                df_neutron, _ = result
                df = calculate_rates(df_neutron, area, volume, neutron_yield)
                total_detected = df["rate_detected"].sum()
                total_error = np.sqrt(df["rate_detected_err2"].sum())
                results.append({
                    "thickness": thickness,
                    "simulated_detected": total_detected,
                    "simulated_error": total_error
                })
    if not results:
        logger.warning("No matching simulated CSV files found in folder.")
        return
    simulated_df = pd.DataFrame(results).sort_values(by="thickness")
    combined_df = pd.merge(simulated_df, experimental_df, on="thickness")
    # Export comparison data to CSV
    folder_name = os.path.basename(folder_path.rstrip('/'))
    if EXPORT_CSV:
        csv_path = get_output_path(folder_path, folder_name, "thickness comparison data", extension="csv", subfolder="csvs")
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
    chi_squared, dof, reduced_chi_squared = calculate_chi_squared(
        combined_df["simulated_detected"],
        combined_df["cps"],
        combined_df["simulated_error"],
        combined_df["error_cps"]
    )
    logger.info(f"\nChi-squared: {chi_squared:.2f}")
    logger.info(f"Degrees of Freedom: {dof}")
    logger.info(f"Reduced Chi-squared: {reduced_chi_squared:.2f}")
    plt.figure(figsize=(10, 6))
    plt.errorbar(combined_df["thickness"], combined_df["simulated_detected"],
                 yerr=combined_df["simulated_error"], fmt='o', label="Simulated", capsize=5)
    plt.plot(combined_df["thickness"], combined_df["simulated_detected"], linestyle='-', color='blue', alpha=0.7)
    plt.errorbar(combined_df["thickness"], combined_df["cps"],
                 yerr=combined_df["error_cps"], fmt='s', label="Experimental", capsize=5)
    plt.plot(combined_df["thickness"], combined_df["cps"], linestyle='-', color='orange', alpha=0.7)
    plt.xlabel("Moderator Thickness (cm)")
    plt.ylabel("Counts Per Second (CPS)")
    plt.title("Simulated vs Experimental Neutron Detection")
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    parent_folder = os.path.dirname(folder_path.rstrip('/'))
    save_path = get_output_path(parent_folder, folder_name, f"{folder_name} plot", extension="pdf", subfolder="plots")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")
    # plt.show()

def run_analysis_type_3(folder_path, area, volume, neutron_yield):
    exp_rate = EXP_RATE
    exp_err = EXP_ERR
    results = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            match = re.search(r'([-+]?\d+_\d+|\d+)', filename)
            if match:
                distance_str = match.group(1).replace('_', '.')
                try:
                    distance = float(distance_str)
                except ValueError:
                    continue
                result = read_tally_blocks_to_df(file_path)
                if result is None:
                    continue
                df_neutron, _ = result
                df = calculate_rates(df_neutron, AREA, VOLUME, neutron_yield)
                total_detected = df["rate_detected"].sum()
                total_error = np.sqrt(df["rate_detected_err2"].sum())
                results.append({
                    "distance": distance,
                    "rate_detected": total_detected,
                    "rate_error": total_error
                })
    if not results:
        logger.warning("No matching simulated CSV files found in folder.")
        return
    distance_df = pd.DataFrame(results).sort_values(by="distance")
    # Export displacement data to CSV
    folder_name = os.path.basename(folder_path.rstrip('/'))
    if EXPORT_CSV:
        csv_path = get_output_path(folder_path, folder_name, "source shift data", extension="csv", subfolder="csvs")
        distance_df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
    plt.figure(figsize=(10, 6))
    plt.errorbar(distance_df["distance"], distance_df["rate_detected"], yerr=distance_df["rate_error"], fmt='o', capsize=5, label="Simulated")
    fit_coeffs, cov_matrix = np.polyfit(distance_df["distance"], distance_df["rate_detected"], 1, cov=True)
    slope, intercept = fit_coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov_matrix))
    fitted_values = np.polyval(fit_coeffs, distance_df["distance"])
    residuals = distance_df["rate_detected"] - fitted_values
    chi_squared_fit = np.sum((residuals / distance_df["rate_error"]) ** 2)
    dof_fit = len(distance_df) - 2  # 2 parameters in linear fit
    reduced_chi_squared_fit = chi_squared_fit / dof_fit
    logger.info(f"Chi-squared of linear fit: {chi_squared_fit:.2f}")
    logger.info(f"Degrees of freedom: {dof_fit}")
    logger.info(f"Reduced Chi-squared: {reduced_chi_squared_fit:.2f}")
    if slope != 0:
        x_intersect = (exp_rate - intercept) / slope
        x_intersect_err = x_intersect * np.sqrt((intercept_err / (exp_rate - intercept)) ** 2 + (slope_err / slope) ** 2)
        logger.info(f"Intersection of fit line with experimental value at x = {x_intersect:.3f} ± {x_intersect_err:.3f} cm")
        extended_margin = 1.0
        min_x = min(distance_df["distance"].min(), x_intersect - extended_margin)
        max_x = max(distance_df["distance"].max(), x_intersect + extended_margin)
        fit_x = np.linspace(min_x, max_x, 500)
        fit_y = np.polyval(fit_coeffs, fit_x)
        plt.plot(fit_x, fit_y, linestyle='--', color='blue', label="Best Fit Line")
        plt.annotate(
            f'Intersection: {x_intersect:.2f} ± {x_intersect_err:.2f} cm',
            xy=(x_intersect, exp_rate),
            xytext=(x_intersect + 0.5, exp_rate + 0.05 * exp_rate),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10, color='black'
        )
    else:
        logger.warning("Fit line is horizontal; no intersection with experimental line.")
    plt.axhline(y=exp_rate, color='red', linestyle='--', label=f"Experimental = {exp_rate:.2e}")
    plt.axhspan(exp_rate - exp_err, exp_rate + exp_err, color='red', alpha=0.2, label="Experimental Uncertainty")
    plt.xlabel("Source Displacement (cm)")
    plt.ylabel("Total Detected Rate")
    plt.title("Detected Rate vs Source Displacement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.text(
        0.98, 0.02,
        f"$\\chi^2_\\nu$ = {reduced_chi_squared_fit:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )
    save_path = get_output_path(folder_path, folder_name, "source shift plot", extension="pdf", subfolder="plots")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")
    # plt.show()


# ---- Function for Analysis Type 4: Photon Tally Plot ----
def run_analysis_type_4(file_path):
    _, df_photon = read_tally_blocks_to_df(file_path)
    if df_photon is None or df_photon.empty:
        logger.warning("No photon tally data found.")
        return

    # --- Save photon tally to CSV ---
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_dir = os.path.dirname(file_path)
    if EXPORT_CSV:
        photon_csv_path = get_output_path(base_dir, base_name, "photon tally", extension="csv", subfolder="csvs")
        df_photon.to_csv(photon_csv_path, index=False)
        logger.info(f"Saved: {photon_csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(df_photon["photon_energy"], df_photon["photons"], label="Photons")
    plt.xlabel("Photon Energy (MeV)")
    plt.ylabel("Photon Counts")
    plt.title("Photon Tally (Tally 34)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = get_output_path(base_dir, base_name, "photon tally plot", extension="pdf", subfolder="plots")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved: {save_path}")
    # plt.show()


# ---- Main execution logic ----
def main():
    neutron_yield = select_neutron_yield()
    while True:
        Tk().withdraw()
        analysis_type = input(
            "Select analysis type:\n"
            "1. Efficiency & Neutron Rates (single simulated CSV)\n"
            "2. Thickness Comparison (multiple simulated + experimental)\n"
            "3. Source Position Alignment (varying source distance, no moderator)\n"
            "4. Photon Tally Plot (single simulated CSV)\n"
            "Enter 1, 2, 3, or 4: "
        ).strip()

        if analysis_type == "1":
            file_path, _ = prompt_for_valid_file("Select MCNP Output File")
            run_analysis_type_1(file_path, AREA, VOLUME, neutron_yield)
        elif analysis_type == "2":
            folder_path = select_folder("Select Folder with Simulated Data")
            if not folder_path:
                logger.warning("No folder selected.")
                continue
            lab_data_path = select_file("Select Experimental Lab Data CSV")
            if not lab_data_path:
                logger.warning("No experimental CSV selected.")
                continue
            run_analysis_type_2(folder_path, lab_data_path, AREA, VOLUME, neutron_yield)
        elif analysis_type == "3":
            folder_path = select_folder("Select Folder with Simulated Source Position CSVs")
            if not folder_path:
                logger.warning("No folder selected.")
                continue
            run_analysis_type_3(folder_path, AREA, VOLUME, neutron_yield)
        elif analysis_type == "4":
            file_path, _ = prompt_for_valid_file("Select MCNP Output File for Gamma Analysis")
            run_analysis_type_4(file_path)
        else:
            logger.warning("Invalid selection.")

        cont = input("\nWould you like to run another analysis? (Y/N): ").strip().lower()
        if cont != "y":
            logger.info("Exiting analysis tool.")
            break


# ---- Entry point ----
if __name__ == "__main__":
    main()