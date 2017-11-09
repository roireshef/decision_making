import numpy as np
from decision_making.src.planning.utils.math import Math


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