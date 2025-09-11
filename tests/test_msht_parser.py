from pathlib import Path
import sys
import pandas as pd
import pandas.testing as pdt

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from mcnp.utils.msht_parser import parse_msht


def test_parse_msht(tmp_path: Path) -> None:
    content = (
        "Some preamble\n"
        "X         Y         Z     Result    Rel Error   Volume    Result*Volume\n"
        "1.0 2.0 3.0 4.0 0.5 6.0 24.0\n"
        "2.0 3.0 4.0 5.0 0.6 7.0 35.0\n"
        "\n"
        "Other text\n"
    )
    file_path = tmp_path / "sample.msht"
    file_path.write_text(content, encoding="utf-8")

    df = parse_msht(file_path)
    expected = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 0.5, 6.0, 24.0],
            [2.0, 3.0, 4.0, 5.0, 0.6, 7.0, 35.0],
        ],
        columns=["x", "y", "z", "result", "rel_error", "volume", "result_vol"],
    )

    pdt.assert_frame_equal(df, expected)


def test_parse_msht_header_flexible(tmp_path: Path) -> None:
    """Header detection should ignore spacing and case."""
    content = (
        "Random text\n"
        "   x    y    z    result    rel error   volume    result*volume\n"
        "1 2 3 4 0.5 6 24\n"
    )
    file_path = tmp_path / "sample.msht"
    file_path.write_text(content, encoding="utf-8")

    df = parse_msht(file_path)
    expected = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 0.5, 6.0, 24.0]],
        columns=["x", "y", "z", "result", "rel_error", "volume", "result_vol"],
    )
    pdt.assert_frame_equal(df, expected)
