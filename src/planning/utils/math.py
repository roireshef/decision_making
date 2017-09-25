import math
from typing import Union, TypeVar

from decision_making.src.global_constants import DIVISION_FLOATING_ACCURACY, EXP_CLIP_TH
import numpy as np


class Math:
    T = TypeVar('T', bound=Union[float, np.ndarray])

    @staticmethod
    def clipped_exponent(x: np.ndarray, w: float, k: float,
                         min_clip: float=-EXP_CLIP_TH, max_clip: float=EXP_CLIP_TH) -> np.ndarray:
        """
        compute sigmoid with clipping the exponent between [-EXP_CLIP_TH, EXP_CLIP_TH]
        :param x: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param w: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param k: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param min_clip: clips the (k * x) part for too low values. default threshold is -EXP_CLIP_TH
        :param max_clip: clips the (k * x) part for too high values. default threshold is EXP_CLIP_TH
        :return: numpy array of exponentiated values
        """
        return np.multiply(w, np.exp(np.clip(k * x, min_clip, max_clip)))

    @staticmethod
    def div(a: T,  b: T, precision: float = DIVISION_FLOATING_ACCURACY) -> T:
        """
        divides a/b with desired floating-point precision
        """
        div = np.divide(a, b).astype(np.int_)
        mod = np.subtract(a, np.multiply(div, b))
        add_ones = 1 * (np.fabs(mod - b) < precision)

        if isinstance(div, np.ndarray):
            return (div + add_ones).astype(np.int_)
        else:
            return int(div + add_ones)

    @staticmethod
    def mod(a: T,  b: T, precision: float = DIVISION_FLOATING_ACCURACY) -> T:
        """
        modulo a % b with desired floating-point precision
        """
        div = np.divide(a, b).astype(np.int_)
        mod = np.subtract(a, np.multiply(div, b))

        return b * (np.fabs(mod - b) < precision) + mod * (np.fabs(mod - b) > precision) * (np.fabs(mod) > precision)
