import math

from decision_making.src.global_constants import DIVISION_FLOATING_ACCURACY, EXP_CLIP_TH
import numpy as np


class Math:
    @staticmethod
    def clipped_exponent(x: np.ndarray, w: float, k: float,
                         min_clip: float=0, max_clip: float=EXP_CLIP_TH) -> np.ndarray:
        """
        compute sigmoid with clipping the exponent between [0, EXP_CLIP_TH]
        :param x: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param w: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param k: sigmoid function is f(x) = w / (1 + exp(k * x))
        :param min_clip: clips the (k * x) part for too low values. default threshold is 0
        :param max_clip: clips the (k * x) part for too high values. default threshold is EXP_CLIP_TH
        :return: numpy array of exponentiated values
        """
        return w * np.sum(np.exp(np.clip(k * x, min_clip, max_clip)), axis=1)

    @staticmethod
    def div(a: float,  b: float, precision: float = DIVISION_FLOATING_ACCURACY):
        """
        divides a/b with desired floating-point precision
        """
        div, mod = divmod(a, b)
        if math.fabs(mod - b) < precision:
            return int(div + 1)
        else:
            return int(div)

    @staticmethod
    def mod(a,  b, precision: float = DIVISION_FLOATING_ACCURACY):
        """
        modulo a % b with desired floating-point precision
        """
        div, mod = divmod(a, b)
        if math.fabs(mod - b) < precision:
            return 0
        else:
            return mod
