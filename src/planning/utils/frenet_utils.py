import numpy as np
import rte.python.profiler as prof
from decision_making.src.messages.trajectory_parameters import TrajectoryParams
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.trajectory.samplable_trajectory import SamplableTrajectory
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX, FrenetState2D, FS_SV, FrenetStates2D, S2
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


class FrenetUtils:
    @staticmethod
    @prof.ProfileFunction()
    def generate_baseline_trajectory(timestamp: float, action_spec: ActionSpec, trajectory_parameters: TrajectoryParams,
                                     ego_fstate: FrenetState2D) -> SamplableTrajectory:
        """
        Creates a SamplableTrajectory as a reference trajectory for a given ActionSpec, assuming T_d=T_s
        :param timestamp: [s] ego timestamp in seconds
        :param action_spec: action specification that contains all relevant info about the action's terminal state
        :param trajectory_parameters: the parameters (of the required trajectory) that will be sent to TP
        :param ego_fstate: ego Frenet state w.r.t. reference_route
        :return: a SamplableWerlingTrajectory object
        """
        poly_coefs_s, poly_coefs_d = FrenetUtils.generate_baseline_polynomials(action_spec, ego_fstate)

        minimal_horizon = trajectory_parameters.trajectory_end_time - timestamp

        return SamplableWerlingTrajectory(timestamp_in_sec=timestamp,
                                          T_s=action_spec.t,
                                          T_d=action_spec.t,
                                          T_extended=minimal_horizon,
                                          frenet_frame=trajectory_parameters.reference_route,
                                          poly_s_coefs=poly_coefs_s,
                                          poly_d_coefs=poly_coefs_d)

    @staticmethod
    @prof.ProfileFunction()
    def generate_baseline_polynomials(action_spec: ActionSpec, ego_fstate: FrenetState2D) -> (np.ndarray, np.ndarray):
        """
        Creates a SamplableTrajectory as a reference trajectory for a given ActionSpec, assuming T_d=T_s
        :param action_spec: action specification that contains all relevant info about the action's terminal state
        :param ego_fstate: ego Frenet state w.r.t. reference_route
        :return: a SamplableWerlingTrajectory object
        """
        # Note: We create the samplable trajectory as a reference trajectory of the current action.
        goal_fstate = action_spec.as_fstate()
        if action_spec.only_padding_mode:
            # in case of very short action, create samplable trajectory using linear polynomials from ego time,
            # such that it passes through the goal at goal time
            ego_by_goal_state = FrenetUtils.create_ego_by_goal_state(goal_fstate, action_spec.t)
            poly_coefs_s, poly_coefs_d = FrenetUtils.create_linear_profile_polynomial_pair(ego_by_goal_state)
        else:
            # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
            A_inv = QuinticPoly1D.inverse_time_constraints_matrix(action_spec.t)

            constraints_s = np.concatenate((ego_fstate[FS_SX:(FS_SA + 1)], goal_fstate[FS_SX:(FS_SA + 1)]))
            constraints_d = np.concatenate((ego_fstate[FS_DX:(FS_DA + 1)], goal_fstate[FS_DX:(FS_DA + 1)]))

            poly_coefs_s = QuinticPoly1D.solve(A_inv, constraints_s[np.newaxis, :])[0]
            poly_coefs_d = QuinticPoly1D.solve(A_inv, constraints_d[np.newaxis, :])[0]

        return poly_coefs_s, poly_coefs_d

    @staticmethod
    def create_ego_by_goal_state(goal_frenet_state: FrenetState2D, ego_to_goal_time: float) -> FrenetState2D:
        """
        calculate Frenet state in ego time, such that its constant-velocity prediction in goal time is goal_frenet_state
        :param goal_frenet_state: goal Frenet state
        :param ego_to_goal_time: the difference between the goal time and ego time
        :return: ego by goal frenet state
        """
        return np.array([goal_frenet_state[FS_SX] - ego_to_goal_time * goal_frenet_state[FS_SV],
                         goal_frenet_state[FS_SV], 0, 0, 0, 0])

    @staticmethod
    def create_linear_profile_polynomial_pair(frenet_state: FrenetState2D) -> (np.ndarray, np.ndarray):
        """
        Given a frenet state, create two (s, d) polynomials that assume constant velocity (we keep the same momentary
        velocity). Those polynomials are degenerate to s(t)=v*t+x form
        :param frenet_state: the current frenet state to pull positions and velocities from
        :return: a tuple of (s(t), d(t)) polynomial coefficient arrays
        """
        poly_s, poly_d = FrenetUtils.create_linear_profile_polynomial_pairs(frenet_state[np.newaxis])
        return poly_s[0], poly_d[0]

    @staticmethod
    def create_linear_profile_polynomial_pairs(frenet_states: FrenetStates2D) -> (np.ndarray, np.ndarray):
        """
        Given N frenet states, create two Nx6 matrices (s, d) of polynomials that assume constant velocity
        (we keep the same momentary velocity). Those polynomials are degenerate to s(t)=v*t+x form
        :param frenet_states: the current frenet states to pull positions and velocities from
        :return: a tuple of Nx6 matrices (s(t), d(t)) polynomial coefficient arrays
        """
        # zero 4 highest coefficients of poly_s: from x^5 until x^2 (including)
        poly_s = np.c_[np.zeros((frenet_states.shape[0], S2+1)), frenet_states[:, FS_SV], frenet_states[:, FS_SX]]
        # We zero out the lateral polynomial because we strive for being in the lane center with zero lateral velocity
        poly_d = np.zeros((frenet_states.shape[0], QuinticPoly1D.num_coefs()))
        return poly_s, poly_d
