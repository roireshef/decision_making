from decision_making.src.planning.utils.numpy_utils import NumpyUtils
import numpy as np


def test_div_divideByNonZero_sameAsNumpyDivide():
    x = np.random.randn(100)
    y = np.random.randn(100) + np.finfo(np.float32).eps
    assert np.any(NumpyUtils.div(x, y) == np.divide(x, y))


def test_div_divideByZero_equalsZero():
    x = np.random.randn(100)
    y = np.zeros(100)
    assert np.any(NumpyUtils.div(x, y) == y)