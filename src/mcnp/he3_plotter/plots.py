import os
import logging
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class _PyplotShim:
    """Light-weight substitute for ``matplotlib.pyplot`` used in tests."""

    def __init__(self) -> None:
        self.title = lambda text: None  # type: ignore[assignment]


plt = _PyplotShim()

from .io_utils import get_output_path
from .config import config


logger = logging.getLogger(__name__)


def plot_efficiency_and_rates(df, filename):
    """Plot incident/detected rates and efficiencies.

    If a ``surface`` column is present, each surface is plotted separately so
    multiple tally pairs can be visualised on the same axes.
    """

    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_dir = os.path.dirname(filename)

    rate_fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(rate_fig)
    rate_ax = rate_fig.add_subplot(111)

    if "surface" in df.columns:
        for i, (surface, grp) in enumerate(df.groupby("surface")):
            color = f"C{i}"
            rate_ax.errorbar(
                grp["energy"],
                grp["rate_incident"],
                yerr=grp["rate_incident_err"],
                label=f"Incident Rate (Surface {surface})",
                fmt="o-",
                color=color,
                markersize=3,
                capsize=2,
            )
            rate_ax.errorbar(
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
        rate_ax.errorbar(
            df["energy"],
            df["rate_incident"],
            yerr=df["rate_incident_err"],
            label="Incident Rate",
            fmt="o-",
            markersize=3,
            capsize=2,
        )
        rate_ax.errorbar(
            df["energy"],
            df["rate_detected"],
            yerr=df["rate_detected_err"],
            label="Detected Rate",
            fmt="s-",
            markersize=3,
            capsize=2,
        )

    rate_ax.set_xlabel("Energy (MeV)")
    rate_ax.set_ylabel("Neutron Rate")
    if config.show_fig_heading:
        tag = f" - {config.filename_tag.strip()}" if config.filename_tag.strip() else ""
        title = f"Neutron Rates vs Energy{tag}"
        rate_ax.set_title(title)
        plt.title(title)
    rate_ax.legend()
    rate_ax.grid(True)
    rate_ax.set_xscale("log")
    rate_fig.tight_layout()
    rate_path = get_output_path(base_dir, base_name, "Neutron rate plot")
    rate_fig.savefig(rate_path)
    logger.info(f"Saved: {rate_path}")

    eff_fig = Figure(figsize=(8, 6))
    FigureCanvasAgg(eff_fig)
    eff_ax = eff_fig.add_subplot(111)

    if "surface" in df.columns:
        for i, (surface, grp) in enumerate(df.groupby("surface")):
            color = f"C{i}"
            eff_ax.errorbar(
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
        eff_ax.errorbar(
            df["energy"],
            df["efficiency"],
            yerr=df["efficiency_err"],
            fmt="^-",
            color="green",
            markersize=3,
            capsize=2,
        )

    eff_ax.set_xlabel("Energy (MeV)")
    eff_ax.set_ylabel("Detection Efficiency")
    if config.show_fig_heading:
        tag = f" - {config.filename_tag.strip()}" if config.filename_tag.strip() else ""
        title = f"Efficiency vs Energy{tag}"
        eff_ax.set_title(title)
        plt.title(title)
    eff_ax.grid(True)
    eff_ax.set_xscale("log")
    eff_fig.tight_layout()
    eff_path = get_output_path(base_dir, base_name, "efficiency curve")
    eff_fig.savefig(eff_path)
    logger.info(f"Saved: {eff_path}")
    return rate_path, eff_path
