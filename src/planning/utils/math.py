from typing import Union, TypeVar

from decision_making.src.global_constants import EXP_CLIP_TH
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

