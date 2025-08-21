import os
import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import get_output_path


logger = logging.getLogger(__name__)


def plot_efficiency_and_rates(df, filename):
    """Plot incident/detected rates and efficiencies.

    If a ``surface`` column is present, each surface is plotted separately so
    multiple tally pairs can be visualised on the same axes.
    """

    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_dir = os.path.dirname(filename)

    plt.figure(figsize=(10, 6))

    if "surface" in df.columns:
        for i, (surface, grp) in enumerate(df.groupby("surface")):
            color = f"C{i}"
            plt.errorbar(
                grp["energy"],
                grp["rate_incident"],
                yerr=grp["rate_incident_err"],
                label=f"Incident Rate (Surface {surface})",
                fmt="o-",
                color=color,
                markersize=3,
                capsize=2,
            )
            plt.errorbar(
                grp["energy"],
                grp["rate_detected"],
                yerr=grp["rate_detected_err"],
                label=f"Detected Rate (Surface {surface})",
                fmt="s-",
                color=color,
                markersize=3,
                capsize=2,
            )
    else:
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
    logger.info(f"Saved: {rate_path}")

    plt.figure(figsize=(8, 6))

    if "surface" in df.columns:
        for i, (surface, grp) in enumerate(df.groupby("surface")):
            color = f"C{i}"
            plt.errorbar(
                grp["energy"],
                grp["efficiency"],
                yerr=grp["efficiency_err"],
                fmt="^-",
                color=color,
                markersize=3,
                capsize=2,
                label=f"Surface {surface}",
            )
    else:
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
    logger.info(f"Saved: {eff_path}")
    return rate_path, eff_path
