"""Utilities for creating 3D volumes from scatter data."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[int] | np.ndarray


def scatter_to_array(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    values: ArrayLike,
    shape: Tuple[int, int, int],
    dtype: type[np.number] = np.float32,
) -> np.ndarray:
    """Convert scatter coordinates into a 3D numpy array.

    Parameters
    ----------
    x, y, z:
        Sequences of the same length representing the coordinates of the
        scattered points.
    values:
        Values associated with each coordinate triple.
    shape:
        Shape of the resulting 3D array (nx, ny, nz).
    dtype:
        Desired dtype of the output array. Defaults to ``np.float32``.

    Returns
    -------
    numpy.ndarray
        A 3D array with ``values`` placed at the specified coordinates and
        zeros elsewhere.
    """

    arr = np.zeros(shape, dtype=dtype)
    arr[np.asarray(x), np.asarray(y), np.asarray(z)] = values
    return arr


def array_to_volume(array: np.ndarray):
    """Create a ``vedo.Volume`` from a 3D numpy array.

    The import of :mod:`vedo` happens inside the function so that projects that
    merely rely on :func:`scatter_to_array` do not require the optional
    ``vedo`` dependency.
    """

    from vedo import Volume

    return Volume(array)


if __name__ == "__main__":  # pragma: no cover - example usage
    from vedo import show

    data_matrix = scatter_to_array(
        x=[0, 30, 50],
        y=[0, 30, 60],
        z=[0, 30, 70],
        values=[1, 2, 3],
        shape=(70, 80, 90),
        dtype=np.uint8,
    )

    vol = array_to_volume(data_matrix)
    vol.cmap(["white", "b", "g", "r"]).mode(1)
    vol.add_scalarbar()

    show(vol, __doc__, axes=1).close()
