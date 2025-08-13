import tempfile, os, sys

# Ensure project root is on path so He3_Plotter can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from He3_Plotter import process_simulation_file

def test_process_simulation_file_no_tally():
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmp.write(b"no tally data\n")
        tmp.close()
        result = process_simulation_file(tmp.name, area=1.0, volume=1.0, neutron_yield=1.0)
        assert result is None
    finally:
        os.unlink(tmp.name)
