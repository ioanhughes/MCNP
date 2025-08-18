import tempfile, os, sys

# Ensure project root is on path so He3_Plotter can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from He3_Plotter import (
    process_simulation_file,
    read_tally_blocks_to_df,
    run_analysis_type_3,
)

def test_process_simulation_file_no_tally():
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.write(b"no tally data\n")
        tmp.close()
        result = process_simulation_file(tmp.name, area=1.0, volume=1.0, neutron_yield=1.0)
        assert result is None
    finally:
        os.unlink(tmp.name)


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
"""
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        tmp.write(content)
        tmp.close()
        df_neutron, df_photon = read_tally_blocks_to_df(tmp.name)
        assert list(df_neutron["energy"]) == [0.1, 0.2]
        assert list(df_neutron["neutrons_incident_cm2"]) == [2.0, 3.0]
        assert list(df_neutron["neutrons_detected_cm2"]) == [1.0, 2.0]
        assert list(df_photon["photon_energy"]) == [0.5]
        assert list(df_photon["photons"]) == [5.0]
    finally:
        os.unlink(tmp.name)


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
