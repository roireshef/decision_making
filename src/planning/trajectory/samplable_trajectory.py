from typing import Optional

import numpy as np
from abc import ABCMeta, abstractmethod

from decision_making.src.planning.types import CartesianExtendedTrajectory, FrenetTrajectory2D


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

    @abstractmethod
    def sample_frenet(self, time_points: np.ndarray) -> FrenetTrajectory2D:
        """
        Sample cartesian states at the given time points, find the containing road and convert them to Frenet trajectory
        :param time_points: time points for the sampling
        :return: Frenet trajectory
        """
        pass

    @abstractmethod
    def get_time_from_longitude(self, road_id: int, longitude: float) -> Optional[float]:
        """
        Given longitude, calculate time, for which the samplable trajectory reaches this longitude.
        :param road_id: road id (is used in another derived class of SamplableTrajectory)
        :param longitude: [m] the required longitude on the road
        :return: [sec] global time, for which the samplable trajectory reaches the longitude, or None if it doesn't.
        """
        pass

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})
