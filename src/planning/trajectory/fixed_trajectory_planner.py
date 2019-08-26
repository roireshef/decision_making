import time
from logging import Logger
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import raises
from decision_making.src.global_constants import NEGLIGIBLE_DISPOSITION_LON, NEGLIGIBLE_DISPOSITION_LAT, \
    WERLING_TIME_RESOLUTION, MAX_NUM_POINTS_FOR_VIZ
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.types import C_V, \
    CartesianExtendedState, CartesianTrajectories, CartesianExtendedTrajectory, CartesianPoint2D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State
from decision_making.src.exceptions import NotTriggeredException
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame


class FixedSamplableTrajectory(SamplableTrajectory):

    def __init__(self, fixed_trajectory: CartesianExtendedTrajectory, timestamp_in_sec: float = 0, T:float = np.inf):
        """
        This class holds a CartesianExtendedTrajectory object with the 'timestamp_in_sec' member as its initial
        timestamp and T as the total horizon. It samples from the Trajectory object upon request by returning the
        closest point in time on the discrete trajectory.
        :param fixed_trajectory: a CartesianExtendedTrajectory object
        :param timestamp_in_sec: Initial timestamp [s]
        :param T: Trajectory time horizon [s] ("length")
        """
        super().__init__(timestamp_in_sec, T)
        self._fixed_trajectory = fixed_trajectory

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """
        This function takes an array of timestamps and returns a CartesianExtendedTrajectory.
        Note: Since the trajectory is not actually samplable - the closest time points on the trajectory are returned.
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: CartesianExtendedTrajectory
        """

        relative_time_points = time_points - self.timestamp_in_sec

        # Make sure no unplanned extrapolation will occur due to overreaching time points
        # This check is done in relative-to-ego units
        assert max(relative_time_points) <= self.T, \
            'In timestamp %f : self.T=%f <= max(relative_time_points)=%f' % \
            (self.timestamp_in_sec, self.T, max(relative_time_points))

        indices_of_closest_time_points = np.round(relative_time_points / WERLING_TIME_RESOLUTION).astype(int)

        return self._fixed_trajectory[indices_of_closest_time_points]


class FixedTrajectoryPlanner(TrajectoryPlanner):
    """
            FixedTrajectoryPlanner purpose is once the ego reached the trigger position,
            every time the trajectory planner is called, output a trajectory
            that advances incrementally on fixed_trajectory by step size
    """

    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, fixed_trajectory: CartesianExtendedTrajectory, step_size: int,
                 trigger_pos: CartesianPoint2D, sleep_std: float, sleep_mean: float):
        """
        :param logger:
        :param fixed_trajectory: a fixed trajectory to advance on
        :param step_size: the size by which to advance on the trajectory each time
        :param trigger_pos: the position that triggers the first trajectory output
        """
        super().__init__(logger, predictor)
        self._fixed_trajectory = fixed_trajectory
        self._step_size = int(step_size)
        self._trajectory_advancing = 0
        self._trigger_pos = trigger_pos
        self._triggered = False
        self._sleep_std = sleep_std
        self._sleep_mean = sleep_mean

    @raises(NotTriggeredException)
    def plan(self, state: State, reference_route: FrenetSerret2DFrame, goal: CartesianExtendedState, T_target_horizon: float,
             T_trajectory_end_horizon: float, cost_params: TrajectoryCostParams) -> \
            Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """
        Once the ego reached the trigger position, every time the trajectory planner is called, output a trajectory
        that advances incrementally on fixed_trajectory by step size. Otherwise raise NotTriggeredException
        :param state: environment & ego state object
        :param reference_route: ignored
        :param goal: ignored
        :param T_trajectory_end_horizon: the length of the trajectory snippet (seconds)
        :param cost_params: ignored
        :return: a tuple of: (samplable representation of the fixed trajectory, tensor of the fixed trajectory,
         and numpy array of zero as the trajectory's cost)
        """
        # add a Gaussian noise to sleep time, to simulate time delays in control
        time.sleep(max(self._sleep_std * np.random.randn(), 0) + self._sleep_mean)
        current_pos = np.array([state.ego_state.x, state.ego_state.y])

        if not self._triggered and np.all(np.linalg.norm(current_pos - self._trigger_pos) <
                                          np.linalg.norm(np.array([NEGLIGIBLE_DISPOSITION_LON,
                                                                   NEGLIGIBLE_DISPOSITION_LAT]))):
            self._triggered = True

        if self._triggered:
            # A trajectory snippet in the size required for the visualization message is outputted.
            current_trajectory = self._fixed_trajectory[
                                 self._trajectory_advancing:(self._trajectory_advancing + MAX_NUM_POINTS_FOR_VIZ)]

            self._trajectory_advancing += self._step_size

            # Currently no one does anything with the cost, the array here is dummy
            zero_trajectory_cost = np.array([0])

            return FixedSamplableTrajectory(current_trajectory, state.ego_state.timestamp_in_sec), \
                   np.array([current_trajectory[:, :(C_V + 1)]]), zero_trajectory_cost
        else:
            raise NotTriggeredException("Didn't reach trigger point yet [%s]. Current localization is [%s]" %
                                        (self._trigger_pos, current_pos))
