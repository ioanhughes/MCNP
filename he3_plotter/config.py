from dataclasses import dataclass


@dataclass
class PlotterConfig:
    """Configuration settings for the He3 plotter package."""

    filename_tag: str = ""
    plot_extension: str = "pdf"
    show_fig_heading: bool = True


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
