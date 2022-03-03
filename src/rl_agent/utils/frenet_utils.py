import numpy as np
from decision_making.src.planning.types import FS_DA, FS_SA, FS_SX, FS_DX, FrenetState2D, FS_SV, FrenetStates2D, S2
from decision_making.src.planning.types import S1, S0
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionSpec


class FrenetUtils:
    @staticmethod
    def generate_polynomials_1d(x_0, v_0, a_0, x_T, v_T, T) -> np.ndarray:
        """
        Given sets boundary conditions (excluding terminal acceleraiton which is always set to 0), generate a
        1D polynomial for each set. This applies for both longitudinal and lateral dimensions, and performs checks for
        tight timeframes (replaces with padding)
        :param x_0: initial position (numpy 1d array)
        :param v_0: initial velocity (numpy 1d array)
        :param a_0: initial acceleration (numpy 1d array)
        :param x_T: terminal position (numpy 1d array)
        :param v_T: terminal velocity (numpy 1d array)
        :param T: time to perform maneuver (numpy 1d array)
        :return: 2D numpy array with 5th order polynomial coefficients for each input set
        """
        poly_coefs = np.full((len(T), 6), np.nan)

        # Handle T=NaN cases (TODO: should move outside?)
        is_only_padding = np.full(T.shape, False)
        is_only_padding[~np.isnan(T)] = RLActionSpec.is_only_padding_mode(T[~np.isnan(T)])

        # Create linear profile for actions with short horizon (overcomes numerical issues of solving boundary
        # conditions for the jerk-optimal case). This is basically using goal velocity as constant and takes goal
        # position and moves it back by the time difference to end of action.
        poly_coefs[is_only_padding, :S1] = 0
        poly_coefs[is_only_padding, S1] = v_T[is_only_padding]
        poly_coefs[is_only_padding, S0] = x_T[is_only_padding] - T[is_only_padding] * v_T[is_only_padding]

        # Solve boundary conditions for the jerk-optimal case for actions with long enough horizons
        non_padding = ~is_only_padding
        A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T[non_padding])
        constraints = np.c_[x_0[non_padding], v_0[non_padding], a_0[non_padding],
                            x_T[non_padding], v_T[non_padding], np.zeros_like(non_padding[non_padding])]

        poly_coefs[non_padding] = QuinticPoly1D.zip_solve(A_inv, constraints)

        return poly_coefs

    @staticmethod
    def generate_polynomials_2d(ego_fstate: FrenetState2D, goal_fstate: FrenetState2D, T: float) -> (np.ndarray, np.ndarray):
        """
        OBSOLETE - method to generate polynomials from boundary conditions, assuming T_d=T_s
        """
        # Note: We create the samplable trajectory as a reference trajectory of the current action.
        if RLActionSpec.is_only_padding_mode(T):
            # in case of very short action, create samplable trajectory using linear polynomials from ego time,
            # such that it passes through the goal at goal time
            ego_by_goal_state = FrenetUtils.create_ego_by_goal_state(goal_fstate, T)
            poly_coefs_s, poly_coefs_d = FrenetUtils.create_linear_profile_polynomial_pair(ego_by_goal_state)
        else:
            # We assume correctness only of the longitudinal axis, and set T_d to be equal to T_s.
            A_inv = QuinticPoly1D.inverse_time_constraints_matrix(T)

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
        poly_s = np.c_[np.zeros((frenet_states.shape[0], S2 + 1)), frenet_states[:, FS_SV], frenet_states[:, FS_SX]]
        # We zero out the lateral polynomial because we strive for being in the lane center with zero lateral velocity
        poly_d = np.zeros((frenet_states.shape[0], QuinticPoly1D.num_coefs()))
        return poly_s, poly_d
