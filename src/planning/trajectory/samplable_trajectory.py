import numpy as np
from abc import ABCMeta, abstractmethod

from decision_making.src.planning.types import CartesianExtendedTrajectory


class SamplableTrajectory(metaclass=ABCMeta):
    def __init__(self, timestamp_in_sec: float, T: float):
        """
        Abstract class that holds all the statistics to sample points on a specific planned trajectory
        :param timestamp_in_sec: [sec] global timestamp *in seconds* to use as a reference
                (other timestamps will be given relative to it)
        :param T: [sec] longitudinal trajectory duration (relative to self.timestamp).
        """
        self.timestamp_in_sec = timestamp_in_sec
        self.T = T

    @property
    def max_sample_time(self):
        return self.timestamp_in_sec + self.T

    @abstractmethod
    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """
        This function takes an array of time stamps and returns an array of points <x, y, theta, v, acceleration,
        curvature> along the trajectory
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: 2D numpy array with every row having the format of <x, y, yaw, velocity, a, k>
        """
        pass

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})