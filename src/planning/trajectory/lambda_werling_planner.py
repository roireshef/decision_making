import numpy as np

from decision_making.src.planning.trajectory.trajectory_planner import SamplableTrajectory
from decision_making.src.planning.trajectory.werling_planner import WerlingPlanner
from decision_making.src.planning.types import CartesianExtendedTrajectory
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.optimal_control.poly1d import DynamicsCallables


class SamplableBehavioralWerlingTrajectory(SamplableTrajectory):
    def __init__(self, timestamp_in_sec: float, T_s: float, T_d: float, frenet_frame: FrenetSerret2DFrame,
                 s_dynamic_functions: DynamicsCallables, d_dynamic_functions: DynamicsCallables):
        """
        To represent a trajectory that is a result of Werling planner, we store the frenet frame used and
        two dynamics callables class (for dimensions s and d)
        """
        super().__init__(timestamp_in_sec, T_s)
        self.T_d = T_d
        self.frenet_frame = frenet_frame
        self.s_dynamic_functions = s_dynamic_functions
        self.d_dynamic_functions = d_dynamic_functions

    @property
    def T_s(self):
        return self.T

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """See base method for API. In this specific representation of the trajectory, we sample from s-axis polynomial
        (longitudinal) and partially (up to some time-horizon cached in self.lon_plan_horizon) from d-axis polynomial
        (lateral) and extrapolate the rest of the states in d-axis to conform to the trajectory's total duration"""

        # # A Cartesian-Frame trajectory: a numpy matrix of CartesianExtendedState [:, [C_X, C_Y, C_YAW, C_V, C_A, C_K]]
        # CartesianExtendedTrajectory = np.ndarray

        relative_time_points = time_points - self.timestamp_in_sec

        # Make sure no unplanned extrapolation will occur due to overreaching time points
        # This check is done in relative-to-ego units
        assert max(relative_time_points) <= self.T_s

        # assign values from <time_points> in s-axis polynomial
        fstates_s = self.s_dynamic_functions.dynamics_for_time(relative_time_points)
        fstates_d = np.empty(shape=np.append(time_points.shape, 3))

        is_within_horizon_d = relative_time_points <= self.T_d

        fstates_d[is_within_horizon_d] = self.d_dynamic_functions.dynamics_for_time(relative_time_points[is_within_horizon_d])

        # Expand lateral solution to the size of the longitudinal solution with its final positions replicated
        # NOTE: we assume that velocity and accelerations = 0 !!
        end_of_horizon_state_d = self.d_dynamic_functions.dynamics_for_time(np.array([self.T_d]))
        extrapolation_state_d = WerlingPlanner.repeat_1d_state(
            fstate=end_of_horizon_state_d,
            repeats=1,
            override_values=np.zeros(3),
            override_mask=np.array([0, 1, 1]))

        fstates_d[np.logical_not(is_within_horizon_d)] = extrapolation_state_d

        fstates = np.hstack((np.array([fstates_s]), fstates_d))

        # project from road coordinates to cartesian coordinate frame
        cstates = self.frenet_frame.ftrajectory_to_ctrajectory(fstates)

        return cstates
