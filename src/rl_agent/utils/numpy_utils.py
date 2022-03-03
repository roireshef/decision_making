from typing import List, NamedTuple, Union, Tuple

import numpy as np
from decision_making.src.planning.types import Limits, LIMIT_MIN, LIMIT_MAX


class NumpyUtils:
    @staticmethod
    def from_list_of_tuples(objs: List[Union[NamedTuple, Tuple]]) -> np.ndarray:
        """
        Utility function for casting a list of tuples to a numpy array of object dtype. It is necessary in cases
        of original objects are either of type Tuple/NamedTuple (numpy casts those into multi-dimensional arrays by
        default
        :param objs: list of tuple objects to convert to an array of tuples
        :return: 1d numpy array of dtype object
        """
        arr = np.empty(len(objs), dtype=object)
        arr[:] = objs
        return arr
