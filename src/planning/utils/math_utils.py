from typing import Union, TypeVar

import numpy as np

from decision_making.src.global_constants import EXP_CLIP_TH
from decision_making.src.planning.types import Limits, LIMIT_MIN, LIMIT_MAX
from decision_making.src.planning.utils.numpy_utils import NumpyUtils


class Math:
    T = TypeVar('T', bound=Union[float, np.ndarray])

    @staticmethod
    def clipped_exponent(x: np.ndarray, w: float, k: float,
                         min_clip: float = -EXP_CLIP_TH, max_clip: float = EXP_CLIP_TH) -> np.ndarray:

        return np.multiply(w, np.exp(np.clip(k * x, min_clip, max_clip)))

    @staticmethod
    def clipped_sigmoid(x: np.ndarray, w: float, k: float,
                         min_clip: float = -EXP_CLIP_TH, max_clip: float = EXP_CLIP_TH) -> np.ndarray:
        """
        compute sigmoid with clipping the exponent between [-EXP_CLIP_TH, EXP_CLIP_TH]
        :param x: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param w: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param k: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param min_clip: clips the (k * x) part for too low values. default threshold is -EXP_CLIP_TH
        :param max_clip: clips the (k * x) part for too high values. default threshold is EXP_CLIP_TH
        :return: numpy array of exponentiated values
        """
        return np.divide(w, np.add(1.0, Math.clipped_exponent(x, 1, -k, min_clip, max_clip)))

    @staticmethod
    def polyval2d(p, x):
        """
        Functionality similar to numpy.polyval, except now p can be multiple poly1d instances - one in each row,
        while enjoying matrix-operations efficiency
        :param p: a 2d numpy array [MxL] having in each of the M rows the L polynomial coefficients vector
        :param x: a 1d numpy array [N] of samples
        :return: a 2d numpy array [MxN] of polynom values for each poly1d instance and sample
        """
        m = len(p)
        l = p.shape[1]
        n = len(x)

        y = np.zeros(shape=[m, n])
        for i in range(l):
            y = np.einsum('ij,j->ij', y, x) + np.repeat(p[:, i, np.newaxis], n, axis=1)

        return y

    @staticmethod
    def zip_polyval2d(p, x):
        """
        Functionality similar to numpy.polyval, except now p holds multiple poly1d instances - one in each row,
        and x holds in each row a vector of values to assign to the corresponding poly1d in p.
        this enjoys matrix-operations efficiency
        :param p: a 2d numpy array [MxL] having in each of the M rows the L polynomial coefficients vector
        :param x: a 2d numpy array [N] of samples
        :return: a 2d numpy array [MxN] of polynom values for each poly1d instance and sample
        """
        assert p.shape[0] == x.shape[0], 'number of values and polynomials is not equal'
        m = p.shape[0]
        l = p.shape[1]
        n = x.shape[1]

        y = np.zeros(shape=[m, n])
        for i in range(l):
            y = np.einsum('ij,ij->ij', y, x) + np.repeat(p[:, i, np.newaxis], n, axis=1)

        return y

    @staticmethod
    def polyder2d(p, m):
        """
        Functionality similar to numpy.polyval, except now p can be multiple poly1d instances - one in each row,
        while enjoying matrix-operations efficiency
        :param p: a 2d numpy array [MxL] having in each of the M rows the L polynomial coefficients vector
        :param x: a 1d numpy array [N] of samples
        :return: a 2d numpy array [MxN] of polynom values for each poly1d instance and sample
        """
        n = p.shape[1] - 1
        y = p[:, :-1] * np.arange(n, 0, -1)
        if m == 0:
            val = p
        else:
            val = Math.polyder2d(y, m - 1)
        return val

    @staticmethod
    def round_to_step(value, step):
        """
        Round the value to nearest multiple of step
        :param value: the value to be rounded.
        :param step: the rounding step
        :return: a value rounded to a multiple of step
        """
        rounded_val = np.round(value * (1 / step)) / (1 / step)

        return rounded_val

    @staticmethod
    def floor_to_step(value, step):
        """
        Floor the value to nearest multiple of step
        :param value: the value to be floored.
        :param step: the rounding step
        :return: a value floored to a multiple of step
        """
        floored_val = np.floor(value * (1 / step)) / (1 / step)

        return floored_val

    @staticmethod
    def ceil_to_step(value, step):
        """
        Ceils the value to nearest multiple of step
        :param value: the value to be ceiled.
        :param step: the rounding step
        :return: a value ceiled to a multiple of step
        """
        ceiled_val = np.ceil(value * (1 / step)) / (1 / step)

        return ceiled_val

    @staticmethod
    def roots(p):
        """
        Return the roots of polynomials with coefficients given in the rows of p.
        The values in each row of the matrix `p` are coefficients of a polynomial.
        If the length of a row in `p` is n+1 then the polynomial is described by:
            p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
        :param p: A matrix of size (num_of_poly X (poly_degree+1)) which contains polynomial coefficients.
                  num_of_poly has to be greater than 1.
        :return: A matrix containing the roots of the polynomials (a set of roots in each row corresponding
                to the polynomial in the input matrix) [ndarray]

        """
        n = p.shape[-1]
        A = np.zeros(p.shape[:1]+(n-1, n-1), float)
        A[..., 1:, :-1] = np.eye(n-2)
        A[..., 0, :] = -p[..., 1:]/p[..., None, 0]
        return np.linalg.eigvals(A)

    @staticmethod
    def find_real_roots(coef_matrix: np.ndarray):
        """
        Given a matrix of polynomials coefficients, returns their Real roots within boundaries.
        NOTE THAT IN ORDER FOR THIS TO WORK, K has to be >=2 and to have no zeros in its first column (degenerate polynomial)
        :param coef_matrix: 2D numpy array [NxK] full with coefficients of N polynomials of degree (K-1)
        :return: 2D numpy array [Nx(K-1)]
        """
        if np.any(coef_matrix[..., 0] == 0):
            raise NotImplementedError("find_real_roots_in_limits can not find roots for degenerated polynomials, "
                                      "please clip the polynomial")

        # if polynomial is of degree 0 (f(x) = c), it has no roots
        if coef_matrix.shape[-1] < 2:
            return np.full(coef_matrix.shape, np.nan), np.full(coef_matrix.shape, False)

        roots = np.roots(coef_matrix) if coef_matrix.ndim == 1 else Math.roots(coef_matrix)
        real_roots = np.real(roots)
        is_real = np.isclose(np.imag(roots), 0.0)
        return real_roots, is_real

    @staticmethod
    def find_real_roots_in_limits(coef_matrix: np.ndarray, value_limits: Limits):
        """
        Given a matrix of polynomials coefficients, returns their Real roots within boundaries.
        NOTE THAT IN ORDER FOR THIS TO WORK, K has to be >=2 and to have no zeros in its first column (degenerate polynomial)
        :param coef_matrix: 2D numpy array [NxK] full with coefficients of N polynomials of degree (K-1)
        :param value_limits: Boundaries for desired roots to look for.
        :return: 2D numpy array [Nx(K-1)]
        """
        roots_real_part, is_real = Math.find_real_roots(coef_matrix)
        is_in_limits = NumpyUtils.is_in_limits(roots_real_part, value_limits)
        roots_real_part[~np.logical_and(is_real, is_in_limits)] = np.nan
        return roots_real_part

    @staticmethod
    def zip_find_real_roots_in_limits(coef_matrix: np.ndarray, value_limits: np.ndarray):
        """
        Given a matrix of polynomials coefficients, returns their Real roots within a matrix of boundaries.
        NOTE THAT IN ORDER FOR THIS TO WORK, K has to be >=2 and to have no zeros in its first column (degenerate polynomial)
        :param coef_matrix: 2D numpy array [NxK] full with coefficients of N polynomials of degree (K-1)
        :param value_limits: 2D numpy array [Nx2]: [min, max] boundary for desired roots for each polynomial.
        :return: 2D numpy array [Nx(K-1)]
        """
        roots_real_part, is_real = Math.find_real_roots(coef_matrix)
        is_in_limits = np.logical_and(roots_real_part >= value_limits[:, :1], roots_real_part <= value_limits[:, 1:])
        roots_real_part[~np.logical_and(is_real, is_in_limits)] = np.nan
        return roots_real_part
