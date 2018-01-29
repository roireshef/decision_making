from typing import Union
import numpy as np
from numbers import Number

from decision_making.src.planning.types import FrenetState2D
from decision_making.src.planning.utils.numpy_utils import NumpyUtils


class FrenetConstraints:
    """
    this class stores in its fields values for grid-search over frenet-frame parameters for the werling planner
    """
    def __init__(self, sx: Union[np.ndarray, Number], sv: Union[np.ndarray, Number], sa: Union[np.ndarray, Number],
                 dx: Union[np.ndarray, Number], dv: Union[np.ndarray, Number], da: Union[np.ndarray, Number]):
        """

        :param sx: location in [m] in s-coordinate (longitudinal) of Frenet frame
        :param sv: velocity in [m/s] in s-coordinate (longitudinal) of Frenet frame
        :param sa: acceleration in [m/s^2] in s-coordinate (longitudinal) of Frenet frame
        :param dx: location in [m] in d-coordinate (lateral) of Frenet frame
        :param dv: velocity in [m/s] in d-coordinate (lateral) of Frenet frame
        :param da: acceleration in [m/s^2] in d-coordinate (lateral) of Frenet frame
        """
        self._sx = np.array(sx)
        self._sv = np.array(sv)
        self._sa = np.array(sa)
        self._dx = np.array(dx)
        self._dv = np.array(dv)
        self._da = np.array(da)

    def __str__(self):
        return "FrenetConstraints(%s)" % ["%s: %s" % (k, NumpyUtils.str_log(v)) for k, v in self.__dict__.items()]

    @classmethod
    def from_state(cls, state: FrenetState2D):
        return cls(state[0], state[1], state[2], state[3], state[4], state[5])

    def get_grid_s(self) -> np.ndarray:
        """
        Generates a grid (cartesian product) of all (position, velocity and acceleration) on dimension S
        :return: numpy array of shape [n, 3] where n is the resulting number of constraints
        """
        return np.array(np.meshgrid(self._sx, self._sv, self._sa)).T.reshape(-1, 3)

    def get_grid_d(self) -> np.ndarray:
        """
        Generates a grid (cartesian product) of all (position, velocity and acceleration) on dimension D
        :return: numpy array of shape [n, 3] where n is the resulting number of constraints
        """
        return np.array(np.meshgrid(self._dx, self._dv, self._da)).T.reshape(-1, 3)
