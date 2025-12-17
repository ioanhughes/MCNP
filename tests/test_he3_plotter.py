import tempfile
import sys
from pathlib import Path
import pytest

# Ensure project root is on path so the he3_plotter package can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from mcnp.he3_plotter.analysis import (
    process_simulation_file,
    prompt_for_valid_file,
    read_tally_blocks_to_df,
    compute_thickness_residuals,
    run_analysis_type_3,
    parse_thickness_from_filename,
    run_analysis_type_2,
)
from mcnp.he3_plotter.io_utils import get_output_path
from mcnp.he3_plotter.config import (
    config as plotter_config,
    set_axis_label_fontsize,
    set_filename_tag,
    set_legend_fontsize,
    set_plot_extension,
    set_show_fig_heading,
    set_show_grid,
    set_tick_label_fontsize,
)
from mcnp.he3_plotter.plots import plot_efficiency_and_rates
from mcnp.he3_plotter.detectors import DETECTORS

def test_li6i_detector_geometry():
    geom = DETECTORS["Li6I(Eu)"]
    assert geom.length_cm == 2.5
    assert geom.radius_cm == 0.3
    assert geom.area == 2.5 * 0.6
    assert abs(geom.volume - (3.141592653589793 * 0.09 * 2.5)) < 1e-6

def test_parse_thickness_from_filename_handles_optional_cm():
    assert parse_thickness_from_filename("example_10cmo") == 10
    assert parse_thickness_from_filename("example_10o") == 10
    assert parse_thickness_from_filename("example_10cm.out") == 10
    assert parse_thickness_from_filename("example_o") is None

def test_process_simulation_file_no_tally():
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.write(b"no tally data\n")
        tmp.close()
        result = process_simulation_file(tmp.name, area=1.0, volume=1.0, neutron_yield=1.0)
        assert result is None
    finally:
        Path(tmp.name).unlink()


def test_prompt_for_valid_file_requires_neutron_data(tmp_path, monkeypatch, caplog):
    empty_file = tmp_path / "empty.out"
    empty_file.write_text("no tally data\n")

    valid_content = (
        "1tally    14\n"
        "energy value error\n"
        "0.1 2.0 0.1\n"
        "total\n"
        "1tally    24\n"
        "energy value error\n"
        "0.1 1.0 0.05\n"
        "total\n"
    )
    valid_file = tmp_path / "valid.out"
    valid_file.write_text(valid_content)

    selections = iter([str(empty_file), str(valid_file)])

    def fake_select_file(title):
        try:
            return next(selections)
        except StopIteration:
            return ""

    monkeypatch.setattr("mcnp.he3_plotter.analysis.select_file", fake_select_file)

    with caplog.at_level("WARNING"):
        file_path, (df_neutron, _) = prompt_for_valid_file()

    assert file_path == str(valid_file)
    assert not df_neutron.empty
    assert any(
        "Invalid file selected. No tally data found." in record.getMessage()
        for record in caplog.records
    )


def test_read_tally_blocks_to_df_parses_data():
    content = """
1tally    14
something
energy value error
0.1 2.0 0.1
0.2 3.0 0.2
total
1tally    24
something
energy value error
0.1 1.0 0.05
0.2 2.0 0.1
total
1tally    34
something
energy value error
0.5 5.0 0.5
total
1tally    15
something
energy value error
0.1 4.0 0.2
total
1tally    25
something
energy value error
0.1 2.0 0.1
total
1tally    35
something
energy value error
0.5 6.0 0.6
total
"""
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        tmp.write(content)
        tmp.close()
        df_neutron, df_photon = read_tally_blocks_to_df(tmp.name)
        assert list(df_neutron["energy"]) == [0.1, 0.2, 0.1]
        assert list(df_neutron["neutrons_incident_cm2"]) == [2.0, 3.0, 4.0]
        assert list(df_neutron["neutrons_detected_cm2"]) == [1.0, 2.0, 2.0]
        assert list(df_neutron["surface"]) == [4, 4, 5]
        assert list(df_photon["photon_energy"]) == [0.5, 0.5]
        assert list(df_photon["photons"]) == [5.0, 6.0]
        assert list(df_photon["surface"]) == [4, 5]
    finally:
        Path(tmp.name).unlink()


def test_run_analysis_type_3_ignores_non_output_files(tmp_path):
    """run_analysis_type_3 should ignore files that do not end with 'o'."""
    folder = tmp_path / "sim"
    folder.mkdir()

    # Create input files without tally data and corresponding output files
    (folder / "0_0").write_text("no tally data\n")
    (folder / "1_0").write_text("no tally data\n")
    (folder / "2_0").write_text("no tally data\n")
    content = (
        "1tally    14\nenergy value error\n0.1 2.0 0.1\n0.2 3.0 0.2\ntotal\n"
        "1tally    24\nenergy value error\n0.1 1.0 0.05\n0.2 2.0 0.1\ntotal\n"
    )
    (folder / "0_0o").write_text(content)
    (folder / "1_0o").write_text(content)
    (folder / "2_0o").write_text(content)

    run_analysis_type_3(str(folder), area=1.0, volume=1.0, neutron_yield=1.0)

    csv_dir = folder / "csvs"
    csv_files = list(csv_dir.glob("*.csv"))
    assert len(csv_files) == 1, "Expected exactly one CSV output file"
    import pandas as pd

    df = pd.read_csv(csv_files[0])
    # Only the output files should have been processed
    assert len(df) == 3


def test_run_analysis_type_2_without_lab_data(tmp_path):
    folder = tmp_path / "sim"
    folder.mkdir()
    content = (
        "1tally    14\nenergy value error\n0.1 2.0 0.1\n0.2 3.0 0.2\ntotal\n"
        "1tally    24\nenergy value error\n0.1 1.0 0.05\n0.2 2.0 0.1\ntotal\n"
    )
    (folder / "example_1o").write_text(content)
    (folder / "example_2cm.out").write_text(content)
    run_analysis_type_2(
        str(folder), lab_data_path=None, area=1.0, volume=1.0, neutron_yield=1.0
    )
    csv_dir = folder / "csvs"
    csv_files = list(csv_dir.glob("*.csv"))
    assert len(csv_files) == 1, "Expected exactly one CSV output file"
    import pandas as pd

    df = pd.read_csv(csv_files[0])
    assert list(df.columns) == [
        "thickness",
        "simulated_detected",
        "simulated_error",
        "dataset",
    ]
    assert set(df["dataset"]) == {"sim"}
    assert set(df["thickness"]) == {1, 2}


def test_run_analysis_type_2_multiple_folders_without_lab_data(tmp_path):
    folder1 = tmp_path / "lib1"
    folder2 = tmp_path / "lib2"
    folder1.mkdir()
    folder2.mkdir()
    content = (
        "1tally    14\nenergy value error\n0.1 2.0 0.1\n0.2 3.0 0.2\ntotal\n"
        "1tally    24\nenergy value error\n0.1 1.0 0.05\n0.2 2.0 0.1\ntotal\n"
    )
    (folder1 / "example_1o").write_text(content)
    (folder2 / "example_1o").write_text(content)
    run_analysis_type_2(
        [str(folder1), str(folder2)],
        labels=["lib1", "lib2"],
        lab_data_path=None,
        area=1.0,
        volume=1.0,
        neutron_yield=1.0,
    )
    csv_dir = tmp_path / "csvs"
    csv_files = list(csv_dir.glob("*.csv"))
    assert len(csv_files) == 1, "Expected one combined CSV output file"
    import pandas as pd

    df = pd.read_csv(csv_files[0])
    assert set(df["dataset"]) == {"lib1", "lib2"}
    assert list(df.columns) == [
        "thickness",
        "simulated_detected",
        "simulated_error",
        "dataset",
    ]


def test_compute_thickness_residuals_builds_standardised_values():
    import pandas as pd

    combined_df = pd.DataFrame(
        {
            "thickness": [1, 2],
            "simulated_detected": [10.0, 20.0],
            "simulated_error": [1.0, 2.0],
            "dataset": ["sim", "sim"],
        }
    )
    experimental_df = pd.DataFrame(
        {
            "thickness": [1, 2],
            "cps": [12.0, 19.0],
            "error_cps": [1.5, 1.0],
        }
    )

    residuals_df, stats_df = compute_thickness_residuals(
        combined_df, experimental_df
    )

    assert not residuals_df.empty
    assert set(residuals_df.columns) == {
        "thickness",
        "raw_residual_unscaled",
        "relative_residual_pct_unscaled",
        "standardised_residual_unscaled",
        "raw_residual_scaled",
        "relative_residual_pct_scaled",
        "standardised_residual_scaled",
        "combined_uncertainty",
        "scale_factor",
        "dataset",
    }
    # thickness 1: residual = 2, sigma = sqrt(1^2 + 1.5^2)
    assert pytest.approx(
        residuals_df.loc[0, "standardised_residual_unscaled"], rel=1e-3
    ) == 1.109
    # thickness 2: residual = -1, sigma = sqrt(2^2 + 1^2)
    assert pytest.approx(
        residuals_df.loc[1, "standardised_residual_unscaled"], rel=1e-3
    ) == -0.447
    assert pytest.approx(residuals_df.loc[0, "scale_factor"], rel=1e-3) == 1.019
    assert pytest.approx(
        residuals_df.loc[0, "standardised_residual_scaled"], rel=1e-3
    ) == 1.002
    assert pytest.approx(
        residuals_df.loc[1, "standardised_residual_scaled"], rel=1e-3
    ) == -0.621

    assert not stats_df.empty
    assert stats_df.loc[0, "dof_before"] == 1
    assert pytest.approx(stats_df.loc[0, "chi_squared_before"], rel=1e-3) == pytest.approx(
        1.109**2 + (-0.447) ** 2, rel=1e-3
    )
    assert pytest.approx(stats_df.loc[0, "chi_squared_after"], rel=1e-3) == pytest.approx(
        1.3888888889, rel=1e-3
    )


def test_get_output_path_includes_tag(tmp_path):
    set_filename_tag("experiment")
    path = Path(get_output_path(tmp_path, "base", "desc"))
    set_filename_tag("")
    assert "experiment" in path.name


def test_plot_titles_respect_tag_and_toggle(tmp_path, monkeypatch):
    set_filename_tag("tag1")
    from mcnp.he3_plotter.config import set_show_fig_heading
    set_show_fig_heading(True)

    import pandas as pd
    from mcnp.he3_plotter import plots

    df = pd.DataFrame(
        {
            "energy": [1.0],
            "rate_incident": [1.0],
            "rate_detected": [0.5],
            "rate_incident_err": [0.1],
            "rate_detected_err": [0.05],
            "efficiency": [0.5],
            "efficiency_err": [0.01],
        }
    )
    dummy = tmp_path / "test.o"
    dummy.write_text("dummy")

    titles = []

    def capture_title(text):
        titles.append(text)

    monkeypatch.setattr(plots.plt, "title", capture_title)
    plots.plot_efficiency_and_rates(df, dummy)
    assert titles and all("tag1" in t for t in titles)

    titles.clear()
    set_show_fig_heading(False)
    plots.plot_efficiency_and_rates(df, dummy)
    assert titles == []

    set_filename_tag("")
    set_show_fig_heading(True)


def test_set_plot_extension_saves_png(tmp_path):
    set_plot_extension("png")
    import pandas as pd

    df = pd.DataFrame(
        {
            "energy": [1.0],
            "rate_incident": [1.0],
            "rate_detected": [0.5],
            "rate_incident_err": [0.1],
            "rate_detected_err": [0.05],
            "efficiency": [0.5],
            "efficiency_err": [0.01],
        }
    )
    dummy = tmp_path / "test.o"
    dummy.write_text("dummy")
    plot_efficiency_and_rates(df, dummy)
    png_files = list((tmp_path / "plots").glob("*.png"))
    assert png_files, "Expected PNG plot to be saved"
    set_plot_extension("pdf")


def test_plotter_fontsize_and_grid_setters():
    original_axis = plotter_config.axis_label_fontsize
    original_tick = plotter_config.tick_label_fontsize
    original_legend = plotter_config.legend_fontsize
    original_grid = plotter_config.show_grid

    try:
        set_axis_label_fontsize(24)
        set_tick_label_fontsize(18)
        set_legend_fontsize(16)
        set_show_grid(False)

        assert plotter_config.axis_label_fontsize == 24
        assert plotter_config.tick_label_fontsize == 18
        assert plotter_config.legend_fontsize == 16
        assert plotter_config.show_grid is False
    finally:
        set_axis_label_fontsize(original_axis)
        set_tick_label_fontsize(original_tick)
        set_legend_fontsize(original_legend)
        set_show_grid(original_grid)
