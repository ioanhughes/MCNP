import numpy as np

from mcnp.utils import scatter_to_array


def test_scatter_to_array_basic():
    x = [0, 1]
    y = [0, 1]
    z = [0, 1]
    values = [5, 10]
    arr = scatter_to_array(x, y, z, values, shape=(2, 2, 2))

    expected = np.zeros((2, 2, 2), dtype=np.float32)
    expected[0, 0, 0] = 5
    expected[1, 1, 1] = 10

    assert np.array_equal(arr, expected)
