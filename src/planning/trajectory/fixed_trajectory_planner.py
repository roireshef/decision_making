from logging import Logger
from typing import Tuple

import numpy as np
import time

from decision_making.src.exceptions import raises
from decision_making.src.global_constants import NEGLIGIBLE_DISPOSITION_LON, NEGLIGIBLE_DISPOSITION_LAT, \
    TRAJECTORY_NUM_POINTS, EGO_ORIGIN_LON_FROM_REAR
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.types import C_V, \
    CartesianExtendedState, CartesianTrajectories, CartesianPath2D, CartesianExtendedTrajectory, CartesianPoint2D
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State
from decision_making.test.exceptions import NotTriggeredException


class FixedSamplableTrajectory(SamplableTrajectory):

    def __init__(self, fixed_trajectory: CartesianExtendedTrajectory):
        super().__init__(timestamp_in_sec=0, T=np.inf)
        self._fixed_trajectory = fixed_trajectory

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """
        This function takes an array of time stamps and returns aCartesianExtendedTrajectory.
        Note: this function ignores the time_points parameter and returns the
        fixed trajectory which is already sampled.
        :param time_points: 1D numpy array of time stamps *in seconds* (global self.timestamp)
        :return: CartesianExtendedTrajectory
        """
        return self._fixed_trajectory


class FixedTrajectoryPlanner(TrajectoryPlanner):
    """
            FixedTrajectoryPlanner purpose is once the ego reached the trigger position,
            every time the trajectory planner is called, output a trajectory
            that advances incrementally on fixed_trajectory by step size
    """

    def __init__(self, logger: Logger, predictor: Predictor, fixed_trajectory: CartesianExtendedTrajectory, step_size: int,
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
    def plan(self, state: State, reference_route: CartesianPath2D, goal: CartesianExtendedState, time_horizon: float,
             cost_params: TrajectoryCostParams) -> Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """
        Once the ego reached the trigger position, every time the trajectory planner is called, output a trajectory
        that advances incrementally on fixed_trajectory by step size. Otherwise raise NotTriggeredException
        :param time_horizon: the length of the trajectory snippet (seconds)
        :param state: environment & ego state object
        :param reference_route: ignored
        :param goal: ignored
        :param cost_params: ignored
        :return: a tuple of: (samplable representation of the fixed trajectory, tensor of the fixed trajectory,
         and numpy array of zero as the trajectory's cost)
        """
        time.sleep(max(self._sleep_std * np.random.randn(), 0) + self._sleep_mean)
        current_pos = np.array([state.ego_state.x, state.ego_state.y])

        # Since we want to compare current ego position to a point on trajectory, and ego_state was transformed to be
        # around vehicle center, we have to transform the state back.
        current_pos += (EGO_ORIGIN_LON_FROM_REAR - state.ego_state.size.length / 2) * \
                       np.array([np.cos(state.ego_state.yaw), np.sin(state.ego_state.yaw)])

        if not self._triggered and np.all(np.linalg.norm(current_pos - self._trigger_pos) <
                                          np.linalg.norm(np.array([NEGLIGIBLE_DISPOSITION_LON,
                                                                   NEGLIGIBLE_DISPOSITION_LAT]))):
            self._triggered = True

        if self._triggered:
            current_trajectory = self._fixed_trajectory[
                                 self._trajectory_advancing:(self._trajectory_advancing + TRAJECTORY_NUM_POINTS)]

            self._trajectory_advancing += self._step_size

            # Currently no one does anything with the cost, the array here is dummy
            zero_trajectory_cost = np.array([0])

            return FixedSamplableTrajectory(current_trajectory), \
                   np.array([current_trajectory[:, :(C_V + 1)]]), zero_trajectory_cost
        else:
            raise NotTriggeredException("Didn't reach trigger point yet [%s]. Current localization is [%s]" %
                                        (self._trigger_pos, current_pos))
