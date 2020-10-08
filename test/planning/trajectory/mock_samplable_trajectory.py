from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.types import CartesianExtendedTrajectory
import numpy as np


class MockSamplableTrajectory(SamplableTrajectory):
    """
    Assumes the samplable trajectory is already initialized with samples at the required timepoints
    """

    def __init__(self, fixed_trajectory: CartesianExtendedTrajectory, duration: float = np.inf):
        super().__init__(0, T=duration)
        self._fixed_trajectory = fixed_trajectory

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """
        This function takes an array of time stamps and returns aCartesianExtendedTrajectory.

        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: CartesianExtendedTrajectory
        """

        return self._fixed_trajectory
