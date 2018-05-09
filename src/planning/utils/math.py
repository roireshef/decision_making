from typing import Union, TypeVar

import numpy as np

from decision_making.src.global_constants import EXP_CLIP_TH


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
    def ind_on_uniform_axis(value, axis):
        """
        Returns index of closest value on equally-spaced axis
        :param value: the value to be looked for on axis
        :param axis: the axis step
        :return: index of the closest value on the equally-spaced axis
        """
        index = int(np.round((value-axis[0])/(axis[1]-axis[0])))
        return min(index, axis[-1])
