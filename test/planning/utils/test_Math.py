import numpy as np
from decision_making.src.planning.utils.math_utils import Math
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


def test_findRealRootsInLimits_leadingZeroCoefs_rootsShouldBeTheSame():
    limits = np.array([-1000, 1000])
    poly_sq = np.random.rand(1000, 6)

    for first_zero_coef in range(1, 5):

        # set zero the first coefficients
        poly_sq[:, :first_zero_coef] = 0
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


def test_solveQuadratic_compareFindRealRootsInLimits_rootsShouldBeTheSame():
    limits = np.array([-1000, 1000])
    poly_sq = np.random.rand(10000, 3)

    # calculate real roots in limits using Math.solve_quadratic
    roots1 = Math.solve_quadratic(poly_sq.astype(complex))
    is_in_limits = NumpyUtils.is_in_limits(roots1, limits)
    roots1_in_limits = roots1[is_in_limits]

    # calculate real roots in limits using Math.find_real_roots_in_limits
    roots2 = Math.find_real_roots_in_limits(poly_sq, limits)
    real_roots2_in_limits = roots2[np.logical_not(np.isnan(roots2))]

    assert np.isclose(roots1_in_limits, real_roots2_in_limits).all()


def test_floorToStep_compareFloatSeriesWithExpected_Accurate():

    step = 0.1
    floats = np.array([-3.01, -3.59, -2.3, -0.15, 0.15, 1.6, 1.52, 1.19])
    expected_floored_results = np.array([-3.1, -3.6, -2.3, -0.2, 0.1, 1.6, 1.5, 1.1])

    floored_results = Math.floor_to_step(floats, step)
    np.testing.assert_array_equal(expected_floored_results, floored_results)


def test_floorToStep_compareFloatSeriesWithNumpyRoundWithStepOne_Accurate():
    step = 1
    floats = np.array([-3.01, -3.59, -2.3, -0.125, 0.125, 1.6, 1.52, 1.19])
    expected_floored_results = np.floor(floats)

    rounded_results = Math.floor_to_step(floats, step)
    np.testing.assert_array_equal(expected_floored_results, rounded_results)


def test_ceilToStep_compareFloatSeriesWithExpected_Accurate():

    step = 0.1
    floats = np.array([-3.01, -3.59, -2.3, -0.15, 0.15, 1.6, 1.52, 1.19])
    expected_ceiled_results = np.array([-3.0, -3.5, -2.3, -0.1, 0.2, 1.6, 1.6, 1.2])

    ceiled_results = Math.ceil_to_step(floats, step)
    np.testing.assert_array_equal(expected_ceiled_results, ceiled_results)


def test_ceilToStep_compareFloatSeriesWithNumpyRoundWithStepOne_Accurate():

    step = 1
    floats = np.array([-3.01, -3.59, -2.3, -0.125, 0.125, 1.6, 1.52, 1.19])
    expected_rounded_results = np.ceil(floats)

    rounded_results = Math.ceil_to_step(floats, step)
    np.testing.assert_array_equal(expected_rounded_results, rounded_results)


def test_roundToStep_compareFloatSeriesWithExpected_Accurate():

    step = 0.1
    floats = np.array([-3.01, -3.59, -2.3, -0.125, 0.125, 1.6, 1.52, 1.19])
    expected_rounded_results = np.array([-3.0, -3.6, -2.3, -0.1, 0.1, 1.6, 1.5, 1.2])

    rounded_results = Math.round_to_step(floats, step)
    np.testing.assert_array_equal(expected_rounded_results, rounded_results)


def test_roundToStep_compareFloatSeriesWithNumpyRoundWithStepOne_Accurate():

    step = 1
    floats = np.array([-3.01, -3.59, -2.3, -0.125, 0.125, 1.6, 1.52, 1.19])
    expected_rounded_results = np.round(floats)

    rounded_results = Math.round_to_step(floats, step)
    np.testing.assert_array_equal(expected_rounded_results, rounded_results)
