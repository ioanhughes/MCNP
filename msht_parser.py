"""Parser for MCNP mesh tally (.msht) files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd


def _to_floats(parts: Iterable[str]) -> list[float]:
    """Convert an iterable of strings to floats.

    Parameters
    ----------
    parts : Iterable[str]
        String tokens to convert.

    Returns
    -------
    list[float]
        Converted float values.

    Raises
    ------
    ValueError
        If any token cannot be converted to ``float``.
    """

    try:
        return [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - defensive, exercised via parse_msht
        raise ValueError("non-numeric data encountered") from exc


def parse_msht(path: str | Path) -> pd.DataFrame:
    """Parse an MSHT mesh tally file into a :class:`pandas.DataFrame`.

    The parser searches for the line starting with ``"X         Y         Z     Result"``
    which marks the beginning of the mesh tally table. Subsequent non-blank
    lines are read until another blank line or a line containing non-numeric
    data is encountered. Each valid record is split into seven float columns
    corresponding to ``x``, ``y``, ``z``, ``result``, ``rel_error``, ``volume``
    and ``result_vol``.

    Parameters
    ----------
    path : str | Path
        Path to the ``.msht`` file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``['x', 'y', 'z', 'result', 'rel_error',
        'volume', 'result_vol']``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is malformed or does not contain a table.
    """

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        raise

    # The header line that marks the start of the tally table can vary in
    # spacing or capitalization between MCNP versions.  Instead of matching the
    # entire string exactly, split the line into tokens and compare the first
    # four fields case-insensitively.  This makes the parser resilient to
    # leading whitespace or different column spacing.
    header_tokens = ["x", "y", "z", "result"]
    try:
        start = next(
            i
            for i, line in enumerate(lines)
            if [t.lower() for t in line.split()[:4]] == header_tokens
        )
    except StopIteration as exc:
        raise ValueError("MSHT table header not found") from exc

    data: list[list[float]] = []
    for raw_line in lines[start + 1:]:
        if not raw_line.strip():
            break
        parts = raw_line.split()
        # Stop at the first non-numeric record
        try:
            floats = _to_floats(parts)
        except ValueError:
            break
        if len(floats) != 7:
            raise ValueError("expected 7 columns in MSHT data line")
        data.append(floats)

    if not data:
        raise ValueError("no data lines found in MSHT file")

    columns = ["x", "y", "z", "result", "rel_error", "volume", "result_vol"]
    return pd.DataFrame(data, columns=columns)
