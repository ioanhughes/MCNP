import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import mesh_view

class DummyText:
    def __init__(self):
        self.content = ""
    def delete(self, start, end):
        self.content = ""
    def insert(self, index, text):
        self.content = text
    def get(self, start, end):
        return self.content

def make_view():
    view = mesh_view.MeshTallyView.__new__(mesh_view.MeshTallyView)
    view.output_box = DummyText()
    view.msht_df = None
    view._get_total_rate = lambda: 1.0
    return view

def test_load_msht_and_save_csv(tmp_path, monkeypatch):
    content = (
        "Some preamble\n"
        "X         Y         Z     Result    Rel Error   Volume    Result*Volume\n"
        "1.0 2.0 3.0 4.0 0.5 6.0 24.0\n"
        "2.0 3.0 4.0 5.0 0.6 7.0 35.0\n"
    )
    file_path = tmp_path / "sample.msht"
    file_path.write_text(content, encoding="utf-8")

    view = make_view()
    monkeypatch.setattr(mesh_view, "select_file", lambda title='': str(file_path))
    view.load_msht()

    factor = 3600 * 1e6
    expected = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, 6.0, 24.0, 4.0 * factor, 4.0 * factor * 0.5],
            [2.0, 3.0, 4.0, 5.0, 0.6, 7.0, 35.0, 5.0 * factor, 5.0 * factor * 0.6],
        ],
        columns=[
            "x",
            "y",
            "z",
            "result",
            "rel_error",
            "volume",
            "result_vol",
            "dose",
            "dose_error",
        ],
    )
    pdt.assert_frame_equal(view.get_mesh_dataframe(), expected)
    assert "1.0" in view.output_box.get("1.0", "end")

    csv_path = tmp_path / "out.csv"
    monkeypatch.setattr(mesh_view, "asksaveasfilename", lambda **kwargs: str(csv_path))
    view.save_msht_csv()
    saved = pd.read_csv(csv_path)
    pdt.assert_frame_equal(saved, expected)


def test_load_msht_parse_error(monkeypatch):
    view = make_view()
    monkeypatch.setattr(mesh_view, "select_file", lambda title='': "file.msht")
    monkeypatch.setattr(mesh_view.msht_parser, "parse_msht", lambda path: (_ for _ in ()).throw(ValueError("bad")))
    called = {}
    def fake_error(title, message):
        called["msg"] = (title, message)
    monkeypatch.setattr(mesh_view.Messagebox, "show_error", fake_error, raising=False)
    view.load_msht()
    with pytest.raises(ValueError):
        view.get_mesh_dataframe()
    assert called["msg"][0] == "MSHT Load Error"
