from dataclasses import dataclass


@dataclass
class PlotterConfig:
    """Configuration settings for the He3 plotter package."""

    filename_tag: str = ""
    plot_extension: str = "pdf"
    show_fig_heading: bool = True
    axis_label_fontsize: int = 12
    tick_label_fontsize: int = 10
    legend_fontsize: int = 10
    show_grid: bool = True


config = PlotterConfig()


def set_filename_tag(tag: str) -> None:
    """Set a tag to append to output filenames."""

    config.filename_tag = tag.strip()


def set_plot_extension(extension: str) -> None:
    """Configure the file extension used for saved plots."""

    ext = extension.strip().lower()
    config.plot_extension = ext if ext else "pdf"


def set_show_fig_heading(show: bool) -> None:
    """Enable or disable plot titles."""

    config.show_fig_heading = bool(show)


def set_axis_label_fontsize(size: int) -> None:
    """Set the font size used for axis labels."""

    try:
        config.axis_label_fontsize = max(1, int(size))
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        config.axis_label_fontsize = 12


def set_tick_label_fontsize(size: int) -> None:
    """Set the font size for tick labels."""

    try:
        config.tick_label_fontsize = max(1, int(size))
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        config.tick_label_fontsize = 10


def set_legend_fontsize(size: int) -> None:
    """Set the font size for legend text."""

    try:
        config.legend_fontsize = max(1, int(size))
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        config.legend_fontsize = 10


def set_show_grid(show: bool) -> None:
    """Toggle whether plots should display grid lines."""

    config.show_grid = bool(show)
