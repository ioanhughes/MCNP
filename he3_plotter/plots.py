import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import get_output_path


def plot_efficiency_and_rates(df, filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_dir = os.path.dirname(filename)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        df["energy"],
        df["rate_incident"],
        yerr=df["rate_incident_err"],
        label="Incident Rate",
        fmt="o-",
        markersize=3,
        capsize=2,
    )
    plt.errorbar(
        df["energy"],
        df["rate_detected"],
        yerr=df["rate_detected_err"],
        label="Detected Rate",
        fmt="s-",
        markersize=3,
        capsize=2,
    )
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Neutron Rate")
    plt.title("Neutron Rates vs Energy")
    plt.legend()
    plt.grid(True)
    plt.semilogx()
    plt.tight_layout()
    rate_path = get_output_path(base_dir, base_name, "Neutron rate plot")
    plt.savefig(rate_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        df["energy"],
        df["efficiency"],
        yerr=df["efficiency_err"],
        fmt="^-",
        color="green",
        markersize=3,
        capsize=2,
    )
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Detection Efficiency")
    plt.title("Efficiency vs Energy")
    plt.grid(True)
    plt.semilogx()
    plt.tight_layout()
    eff_path = get_output_path(base_dir, base_name, "efficiency curve")
    plt.savefig(eff_path)
    plt.close()
    return rate_path, eff_path
