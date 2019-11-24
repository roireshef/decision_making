from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.numpy_utils import UniformGrid
import numpy as np
import pytest

def test_div_divideByNonZero_sameAsNumpyDivide():
    x = np.random.randn(100)
    y = np.random.randn(100) + np.finfo(np.float32).eps
    assert np.any(NumpyUtils.div(x, y) == np.divide(x, y))


def test_div_divideByZero_equalsZero():
    x = np.random.randn(100)
    y = np.zeros(100)
    assert np.any(NumpyUtils.div(x, y) == y)


def test_getIndex_interpolation_returnsCorrectIndices():
    grid = UniformGrid(np.array([0, 100]), 1.)

    values = np.array([3., 50.])
    expected_indices = [3, 50]

    for v, ind in zip(values, expected_indices):
        assert ind == grid.get_index(v)


def test_getIndex_extrapolation_raisesException():
    grid = UniformGrid(np.array([0, 100]), 1.)

    values = np.array([-1., 101.])

    for v in values:
        with pytest.raises(Exception):
            _ = grid.get_index(v)


def test_getIndex_bonBoundaries_returnsCorrectIndices():
    grid = UniformGrid(np.array([0, 100]), 1.)

    values = np.array([0., 100.])
    expected_indices = [0, 100]

    for v, ind in zip(values, expected_indices):
        assert ind == grid.get_index(v)


def test_isInLimits2D_twoOnOneDimensionalArray_returnsExpecetedValuesWell():
    limits = np.array([[0, 2], [5, 6]])
    arr = np.array([[[ 1,  2,  3],
        [ 4,  5,  6]],
       [[10, 20, 30],
        [40, 50, 60]]])

    expected = np.array([[[ True,  True, False],
        [False,  True,  True]],
       [[False, False, False],
        [False, False, False]]])
    np.testing.assert_array_equal(NumpyUtils.is_in_limits_2d(arr, limits), expected)


def test_isInLimits2D_twoOnThreeDimensionalArray_returnsExpecetedValuesWell():
    limits = np.array([[0, 2], [5, 6]])
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    expected = np.array([[ True,  True, False], [False,  True,  True]])
    np.testing.assert_array_equal(NumpyUtils.is_in_limits_2d(arr, limits), expected)


def test_isInLimits2D_twoOnOneDimensionalArray_returnsExpecetedValuesWell():
    limits = np.array([[0, 2], [5, 6]])
    arr = np.array([[1], [8]])

    expected = np.array([[True], [False]])
    np.testing.assert_array_equal(NumpyUtils.is_in_limits_2d(arr, limits), expected)

