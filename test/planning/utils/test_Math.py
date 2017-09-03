from decision_making.src.global_constants import DIVISION_FLOATING_ACCURACY
from decision_making.src.planning.utils.math import Math
import numpy as np

def test_div_floats_rightAnswer():
    assert Math.div(1, 2) - 0 < DIVISION_FLOATING_ACCURACY
    assert Math.div(2, 1) - 2 < DIVISION_FLOATING_ACCURACY
    assert Math.div(2, 0.3) - 6 < DIVISION_FLOATING_ACCURACY
    assert Math.div(2.1, 0.3) - 7 < DIVISION_FLOATING_ACCURACY

def test_div_numpyArrays_rightAnswer():
    np.testing.assert_array_almost_equal(Math.div(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),
                                         np.array([10, 10, 10]),
                                         decimal=-1 * np.log10(DIVISION_FLOATING_ACCURACY))


def test_mod_floats_rightAnswer():
    assert Math.mod(1.0, 2.0) - 1.0 < DIVISION_FLOATING_ACCURACY
    assert Math.mod(2.0, 1.0) - 0.0 < DIVISION_FLOATING_ACCURACY
    assert Math.mod(2.0, 0.3) - 0.2 < DIVISION_FLOATING_ACCURACY
    assert Math.mod(2.1, 0.3) - 0.0 < DIVISION_FLOATING_ACCURACY

def test_mod_numpyArrays_rightAnswer():
    np.testing.assert_array_almost_equal(Math.mod(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3])),
                                         np.array([0.0, 0.0, 0.0]),
                                         decimal=-1 * np.log10(DIVISION_FLOATING_ACCURACY))
    np.testing.assert_array_almost_equal(Math.mod(np.array([1.0, 2.0, 3.5]), np.array([0.1, 0.2, 0.3])),
                                         np.array([0.0, 0.0, 0.2]),
                                         decimal=-1 * np.log10(DIVISION_FLOATING_ACCURACY))
