import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Sequence

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

    def __str__(self):
        return str({k: str(v) for (k, v) in self.__dict__.items()})

    def sample_frenet(self, time_points: np.ndarray) -> FrenetTrajectory2D:
        """
        This function takes an array of time stamps and returns an array of Frenet states along the trajectory.
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: Frenet Trajectory
        """
        pass


class ExtendedSamplableTrajectory(SamplableTrajectory):
    def __init__(self, samplable_trajectories: Sequence[SamplableTrajectory]):
        super().__init__(timestamp_in_sec=samplable_trajectories[0].timestamp_in_sec,
                         T=sum([straj.T for straj in samplable_trajectories]))
        self.samplable_trajectories = np.array(samplable_trajectories)

    def get_trajectory_idx_by_time(self, time: np.ndarray) -> np.ndarray:
        """

        :param time:
        :return:
        """
        time_segments = np.concatenate([straj.timestamp_in_sec for straj in self.samplable_trajectories], [self.T])
        time_flat = time.flat
        idxs = np.searchsorted(time_segments, time_flat) - 1
        idxs[idxs<0 & time_flat==time_segments[0]] = 0
        idxs.resize(time.shape)
        return idxs

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        return np.apply_over_axes(lambda t, _: self.samplable_trajectories[self.get_trajectory_idx_by_time(t)].sample(t), time_points, axes=0)
