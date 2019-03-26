import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List

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
        This function takes an array of time stamps and returns an array of Frenet states along the trajectory.
        We sample from s-axis polynomial (longitudinal) and partially (up to some time-horizon cached in
        self.lon_plan_horizon) from d-axis polynomial (lateral) and extrapolate the rest of the states in d-axis
        to conform to the trajectory's total duration.
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: Frenet Trajectory
        """
        pass

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})


class CombinedSamplableTrajectory(SamplableTrajectory):
    def __init__(self, trajectories: List[SamplableTrajectory]):
        super().__init__(trajectories[0].timestamp_in_sec, trajectories[-1].timestamp_in_sec + trajectories[-1].T)
        self.trajectories = trajectories

    @property
    def time_offsets(self):
        return np.array([trajectory.timestamp_in_sec for trajectory in self.trajectories])

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        assert ~np.any(time_points > self.timestamp_in_sec + self.T), "queried time out of trajectory with time range"
        trajectory_idxs = self._get_index_by_time(time_points)

        return np.array([self.trajectories[idx].sample(np.array([time]))[0] for time, idx in zip(time_points, trajectory_idxs)])

    def sample_frenet(self, time_points: np.ndarray) -> FrenetTrajectory2D:
        assert ~np.any(time_points > self.timestamp_in_sec + self.T), "queried time out of trajectory"
        trajectory_idxs = self._get_index_by_time(time_points)

        return np.array([self.trajectories[idx].sample_frenet(time) for time, idx in zip(time_points, trajectory_idxs)])

    def _get_index_by_time(self, time_points: np.ndarray):
        segments_idxs = np.searchsorted(self.time_offsets, time_points) - 1
        segments_idxs[time_points == 0] = 0
        return segments_idxs