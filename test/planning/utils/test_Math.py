import numpy as np
from decision_making.src.planning.utils.math_utils import Math, DIVISION_FLOATING_ACCURACY
from decision_making.src.planning.utils.numpy_utils import NumpyUtils


def test_polyVal2D_exampleMatrices_accurateComputation():
    # polynomial coefficients (4 cubic polynomials)
    p = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])

    # 2 value points for the evaluation of polynomials
    x = np.array([5, 10])

    polyvals = Math.polyval2d(p, x)

    expected_polyvals = np.array([[np.polyval(poly, t) for t in x] for poly in p])

    np.testing.assert_array_equal(polyvals, expected_polyvals)


def test_findRealRootsInLimits_compareFoundRootsWithNumpyRoots_rootsShouldBeTheSame():
    limits = np.array([-1000, 1000])

    for poly_dim in [2, 3]:
        poly_sq = np.random.rand(10000, poly_dim)

        # calculate real roots in limits using np.roots
        roots1 = np.apply_along_axis(np.roots, 1, poly_sq.astype(complex))
        real_roots1 = np.real(roots1)
        is_real = np.isclose(np.imag(roots1), 0.0)
        is_in_limits = NumpyUtils.is_in_limits(real_roots1, limits)
        real_roots1_in_limits = real_roots1[np.logical_and(is_real, is_in_limits)]

        # calculate real roots in limits using Math.find_real_roots_in_limits
        roots2 = Math.find_real_roots_in_limits(poly_sq, limits)
        real_roots2_in_limits = roots2[np.logical_not(np.isnan(roots2))]

        assert np.isclose(real_roots1_in_limits, real_roots2_in_limits).all()


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
