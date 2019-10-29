from abc import abstractmethod
from typing import Union

import numpy as np

from decision_making.src.planning.types import Limits
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils


class Poly1D:
    @staticmethod
    @abstractmethod
    def num_coefs():
        pass

    @staticmethod
    @abstractmethod
    def time_constraints_tensor(terminal_times: np.ndarray) -> np.ndarray:
        """
        Given polynomial setting, this function returns A as a tensor with the first dimension iterating
        over different values of T (time-horizon) provided in <terminal_times>
        :param terminal_times: 1D numpy array of different values for T
        :return: 3D numpy array of shape [len(terminal_times), self.num_coefs(), self.num_coefs()]
        """
        pass

    @staticmethod
    def inverse_time_constraints_tensor(terminal_times: np.ndarray) -> np.ndarray:
        """
        Given polynomial setting, this function returns array of inverse matrices of time constraints tensors
        with the first dimension iterating over different values of T (time-horizon) provided in <terminal_times>
        :param terminal_times: 1D numpy array of different values for T
        :return: 3D numpy array of shape [len(terminal_times), self.num_coefs(), self.num_coefs()]
        """
        pass

    @staticmethod
    @abstractmethod
    def cumulative_jerk(poly_coefs: np.ndarray, T: Union[float, np.ndarray]):
        """
        Computes cumulative jerk from time 0 to time T for the x(t) whose coefficients are given in <poly_coefs>
        :param poly_coefs: distance polynomial coefficients
        :param T: relative time in seconds
        :return: [float] the cummulative jerk: sum(x'''(t)^2)
        """
        pass

    @classmethod
    def jerk_between(cls, poly_coefs: np.ndarray, a: Union[float, np.ndarray], b: Union[float, np.ndarray]):
        return cls.cumulative_jerk(poly_coefs, b) - cls.cumulative_jerk(poly_coefs, a)

    @staticmethod
    def solve(A_inv: np.ndarray, constraints: np.ndarray) -> np.ndarray:
        """
        Given a 1D polynom x(t) with self.num_coefs() differential constraints on t0 (initial time) and tT (terminal time),
        this code solves the problem of minimizing the sum of its squared third-degree derivative sum[ x'''(t) ^ 2 ]
        according to:
        {Local path planning and motion control for AGV in positioning. In IEEE/RSJ International Workshop on
        Intelligent Robots and Systems’ 89. The Autonomous Mobile Robots and Its Applications. IROS’89.
        Proceedings., pages 392–397, 1989}
        :param A_inv: given that the constraints are Ax = B, and x are the polynom coeficients to seek,
        this is the A ^ -1
        :param constraints: given that the constraints are Ax = B, and x are the polynom coeficients to seek,
        every row in here is a B (so that this variable can hold a set of B's that results in a set of solutions)
        :return: x(t) coefficients
        """
        poly_coefs = np.fliplr(np.dot(constraints, A_inv.transpose()))
        return poly_coefs

    @staticmethod
    def zip_solve(A_inv: np.ndarray, constraints: np.ndarray) -> np.ndarray:
        poly_coefs = np.fliplr(np.einsum('ijk,ik->ij', A_inv, constraints))
        return poly_coefs

    @staticmethod
    def polyval_with_derivatives(poly_coefs: np.ndarray, time_samples: np.ndarray) -> np.ndarray:
        """
        For each x(t) position polynomial(s) and time-sample it generates 3 values:
          1. position (evaluation of the polynomial)
          2. velocity (evaluation of the 1st derivative of the polynomial)
          2. acceleration (evaluation of the 2st derivative of the polynomial)
        :param poly_coefs: 2d numpy array [MxL] of the (position) polynomials coefficients, where
         each row out of the M is a different polynomial and contains L coefficients
        :param time_samples: 1d numpy array [K] of the time stamps for the evaluation of the polynomials
        :return: 3d numpy array [M,K,3] with the following dimensions:
            [position value, velocity value, acceleration value]
        """
        # compute the coefficients of the polynom's 1st derivative (m=1)
        poly_dot_coefs = Math.polyder2d(poly_coefs, m=1)
        # compute the coefficients of the polynom's 2nd derivative (m=2)
        poly_dotdot_coefs = Math.polyder2d(poly_coefs, m=2)

        x_vals = Math.polyval2d(poly_coefs, time_samples)
        x_dot_vals = Math.polyval2d(poly_dot_coefs, time_samples)
        x_dotdot_vals = Math.polyval2d(poly_dotdot_coefs, time_samples)

        return np.dstack((x_vals, x_dot_vals, x_dotdot_vals))

    @staticmethod
    def zip_polyval_with_derivatives(poly_coefs: np.ndarray, time_samples: np.ndarray) -> np.ndarray:
        """
        For n-th position polynomial and k-th element in n-th row of time-samples matrix (where n runs on all
        polynomials), it generates 3 values:
          1. position (evaluation of the polynomial)
          2. velocity (evaluation of the 1st derivative of the polynomial)
          3. acceleration (evaluation of the 2st derivative of the polynomial)
        :param poly_coefs: 2d numpy array [NxL] of the (position) polynomials coefficients, where
         each row out of the N is a different polynomial and contains L coefficients
        :param time_samples: 2d numpy array [NxK] of the time stamps for the evaluation of the polynomials
        :return: 3d numpy array [N,K,3] with the following dimensions:
            [position value, velocity value, acceleration value]
        """
        # compute the coefficients of the polynom's 1st derivative (m=1)
        poly_dot_coefs = Math.polyder2d(poly_coefs, m=1)
        # compute the coefficients of the polynom's 2nd derivative (m=2)
        poly_dotdot_coefs = Math.polyder2d(poly_coefs, m=2)

        x_vals = Math.zip_polyval2d(poly_coefs, time_samples)
        x_dot_vals = Math.zip_polyval2d(poly_dot_coefs, time_samples)
        x_dotdot_vals = Math.zip_polyval2d(poly_dotdot_coefs, time_samples)

        return np.dstack((x_vals, x_dot_vals, x_dotdot_vals))

    @classmethod
    def time_constraints_matrix(cls, T: float) -> np.ndarray:
        """
        Given the polynomial setting, this function returns A as a tensor with the first dimension iterating
        over different values of T (time-horizon) provided in <terminal_times>
        :param T: 1D numpy array of different values for T
        :return: 3D numpy array of shape [len(terminal_times), cls.num_coefs(), cls.num_coefs()]
        """
        return cls.time_constraints_tensor(np.array([T]))[0]

    @staticmethod
    def are_derivatives_in_limits(degree: int, poly_coefs: np.ndarray, T_vals: np.ndarray, limits: Limits):
        """
        Applies the following on a vector of polynomials and planning-times: given coefficients vector of a
        polynomial x(t), and restrictions on its <degree> derivative, return True if restrictions are met,
        False otherwise
        :param degree:
        :param poly_coefs: 2D numpy array with N polynomials and <cls.num_coefs()> coefficients each
        :param T_vals: 1D numpy array of planning-times [N]
        :param limits: minimal and maximal allowed values for the <degree> derivative of x(t)
        :return: 1D numpy array of booleans where True means the restrictions are met.
        """
        # a(0) and a(T) checks are omitted as they they are provided by the user.
        # compute extrema points, by finding the roots of the 3rd derivative
        poly_der = Math.polyder2d(poly_coefs, m=degree+1)
        poly = Math.polyder2d(poly_coefs, m=degree)

        # TODO: implement tests for those cases
        if poly_der.shape[-1] == 0:  # No derivative - polynomial is constant
            if poly.shape[-1] == 0:  # Also polynomial is zero (null)
                return NumpyUtils.is_in_limits(np.full((poly.shape[0], 1), 0), limits)
            else:
                return NumpyUtils.is_in_limits(poly[:, 0], limits)
        elif poly_der.shape[-1] == 1:  # 1st order derivative is constant - Polynomial is a*x+b
            # No need to test for t=0 (assuming it's valid), only t=T
            return NumpyUtils.is_in_limits(Math.polyval2d(poly, T_vals), limits)

        #  Find roots of jerk_poly (nan for complex or negative roots).
        acc_suspected_points = Math.find_real_roots_in_limits(poly_der, value_limits=np.array([0, np.inf]))
        acc_suspected_values = Math.zip_polyval2d(poly, acc_suspected_points)

        # are extrema points out of [0, T] range and are they non-complex
        is_suspected_point_in_time_range = (acc_suspected_points <= T_vals[:, np.newaxis])

        # check if extrema values are within [a_min, a_max] limits or very close to the limits
        is_suspected_value_in_limits = NumpyUtils.is_almost_in_limits(acc_suspected_values, limits)

        # for all non-complex extrema points that are inside the time range, verify their values are in [a_min, a_max]
        return np.all(np.logical_or(np.logical_not(is_suspected_point_in_time_range), is_suspected_value_in_limits), axis=1)

    @classmethod
    def are_accelerations_in_limits(cls, poly_coefs: np.ndarray, T_vals: np.ndarray, acc_limits: Limits) -> np.ndarray:
        return cls.are_derivatives_in_limits(degree=2, poly_coefs=poly_coefs, T_vals=T_vals, limits=acc_limits)

    @classmethod
    def is_acceleration_in_limits(cls, poly_coefs: np.ndarray, T: float, acc_limits: Limits) -> bool:
        """
        given coefficients vector of a polynomial x(t), and restrictions on the acceleration values,
        return True if restrictions are met, False otherwise
        :param poly_coefs: 1D numpy array of x(t)'s coefficients
        :param T: planning time horizon [sec]
        :param acc_limits: minimal and maximal allowed values of acceleration/deceleration [m/sec^2]
        :return: True if restrictions are met, False otherwise
        """
        return cls.are_accelerations_in_limits(np.array([poly_coefs]), np.array([T]), acc_limits)[0]

    @classmethod
    def are_velocities_in_limits(cls, poly_coefs: np.ndarray, T_vals: np.ndarray, vel_limits: Limits) -> np.ndarray:
        """
        Applies the following on a vector of polynomials and planning-times: given coefficients vector of a
        polynomial x(t), and restrictions on the velocity values, return True if restrictions are met,
        False otherwise
        :param poly_coefs: 2D numpy array with N polynomials and 6 coefficients each [Nx6]
        :param T_vals: 1D numpy array of planning-times [N]
        :param vel_limits: minimal and maximal allowed values of velocities [m/sec]
        :return: 1D numpy array of booleans where True means the restrictions are met.
        """
        return cls.are_derivatives_in_limits(degree=1, poly_coefs=poly_coefs, T_vals=T_vals, limits=vel_limits)

    @classmethod
    def is_velocity_in_limits(cls, poly_coefs: np.ndarray, T: float, vel_limits: Limits) -> bool:
        """
        given coefficients vector of a polynomial x(t), and restrictions on the velocity values,
        return True if restrictions are met, False otherwise
        :param poly_coefs: 1D numpy array of x(t)'s coefficients
        :param T: planning time horizon [sec]
        :param vel_limits: minimal and maximal allowed values of velocities [m/sec]
        :return: True if restrictions are met, False otherwise
        """
        return cls.are_velocities_in_limits(np.array([poly_coefs]), np.array([T]), vel_limits)[0]

    @classmethod
    def solve_1d_bvp(cls, constraints: np.ndarray, T: float) -> np.ndarray:
        """
        Solves the two-point boundary value problem in 1D, given a set of constraints over the initial and terminal states.
        :param constraints: 3D numpy array of a set of constraints over the initial and terminal states
        :param T: longitudinal/lateral trajectory duration (sec.), relative to ego. T has to be a multiple of WerlingPlanner.dt
        :return: a poly-coefficients-matrix of rows in the form [c0_s, c1_s, ... c5_s] or [c0_d, ..., c5_d]
        """
        assert constraints.shape[-1] == cls.num_coefs(), "%s should get constraints of size %s (got %s)" % \
                                                         (cls.__name__, cls.num_coefs(), constraints.shape[-1])
        A = cls.time_constraints_matrix(T)
        A_inv = np.linalg.inv(A)
        poly_coefs = cls.solve(A_inv, constraints)
        return poly_coefs


class QuarticPoly1D(Poly1D):
    @staticmethod
    def num_coefs():
        return 5

    @staticmethod
    def is_tracking_mode(v_0: np.array, v_T: np.array, a_0: np.array) -> np.array:
        """
        Checks if agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        zero.
        :param v_0: a vector of initial velocities
        :param v_T: a vector of terminal velocities
        :param a_0: a vector of initial accelerations
        :return: a vector of boolean values indicating if ego is in tracking mode, meaning it actually wants to stay at
        its current velocity (usually when it stabilizes on the desired velocity in a following action)
        """
        return np.logical_and(np.isclose(v_0, v_T, atol=1e-3, rtol=0), np.isclose(a_0, 0.0, atol=1e-3, rtol=0))

    @staticmethod
    def cumulative_jerk(poly_coefs: np.ndarray, T: Union[float, np.ndarray]):
        """
        Computes cumulative jerk from time 0 to time T for the x(t) whose coefficients are given in <poly_coefs>
        :param poly_coefs: distance polynomial coefficients
        :param T: relative time in seconds
        :return: [float] the cummulative jerk: sum(x'''(t)^2)
        """
        a4, a3, a2, a1, a0 = np.split(poly_coefs, 5, axis=-1)
        a4, a3, a2, a1, a0 = a4.flatten(), a3.flatten(), a2.flatten(), a1.flatten(), a0.flatten()
        return 36 * (a3 ** 2) * T + \
               144 * a3 * a4 * T ** 2 + \
               192 * a4 ** 2 * T ** 3

    @staticmethod
    def time_constraints_tensor(terminal_times: np.ndarray) -> np.ndarray:
        """
        Given the quartic polynomial setting, this function returns A as a tensor with the first dimension iterating
        over different values of T (time-horizon) provided in <terminal_times>
        :param terminal_times: 1D numpy array of different values for T
        :return: 3D numpy array of shape [len(terminal_times), 6, 6]
        """
        return np.array(
            [[[1.0, 0.0, 0.0, 0.0, 0.0],  # x(0)
              [0.0, 1.0, 0.0, 0.0, 0.0],  # x_dot(0)
              [0.0, 0.0, 2.0, 0.0, 0.0],  # x_dotdot(0)
              [0.0, 1.0, 2.0 * T, 3.0 * T ** 2, 4.0 * T ** 3],  # x_dot(T)
              [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T ** 2]]  # x_dotdot(T)
             for T in terminal_times], dtype=np.float)

    @staticmethod
    def inverse_time_constraints_tensor(T: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(T)
        tensor = np.array([
            [1 + zeros, zeros, zeros, zeros, zeros],
            [zeros, 1 + zeros, zeros, zeros, zeros],
            [zeros, zeros, 0.5 + zeros, zeros, zeros],
            [zeros, -1/T**2, -2/(3*T), 1/T**2, -1/(3*T)],
            [zeros, 1/(2*T**3), 1/(4*T**2), -1/(2*T**3), 1/(4*T**2)]])
        return np.transpose(tensor, (2, 0, 1))

    @staticmethod
    def time_cost_function_derivative_coefs(w_T: np.ndarray, w_J: np.ndarray, a_0: np.ndarray, v_0: np.ndarray,
                                            v_T: np.ndarray):
        """
        For given weights and constraints on a jerk-optimal polynomial solution, this function returns a matrix that
        contains (in each row:) the coefficients of the derivative of the cost function use for finding the optimal time
        horizon: f(T) = w_T * T + w_J * J(T) where J(T) is the accumulated jerk for given time horizon T.
        :param w_T: weight for Time component
        :param w_J: weight for Jerk component
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :return: coefficient matrix for all possibilities
        """
        zeros = np.zeros(w_T.shape[0])
        return np.c_[w_T,
                     zeros,
                     - 4 * a_0 ** 2 * w_J, + 24 * (a_0 * v_T * w_J - a_0 * v_0 * w_J),
                     - 36 * v_0 ** 2 * w_J + 72 * v_0 * v_T * w_J - 36 * v_T ** 2 * w_J]

    @staticmethod
    def distance_profile_function(a_0: float, v_0: float, v_T: float, T: float):
        """
        relative distance travelled by ego at time t, given a solution to the conditions in the parameters
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param T: [sec] horizon
        :return: lambda function(s) that takes relative time in seconds and returns the relative distance
        travelled since time 0
        """
        return lambda t: t * (6 * T ** 3 * (a_0 * t + 2 * v_0) - 4 * T * t ** 2 * (2 * T * a_0 + 3 * v_0 - 3 * v_T) +
                              3 * t ** 3 * (T * a_0 + 2 * v_0 - 2 * v_T)) / (12 * T ** 3)

    @staticmethod
    def velocity_profile_function(a_0: float, v_0: float, v_T: float, T: float):
        """
        velocity of ego at time t, given a solution to the conditions in the parameters
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param T: [sec] horizon
        :return: lambda function(s) that takes relative time in seconds and returns the velocity
        """
        return lambda t: (T ** 3 * (a_0 * t + v_0)
                          - T * t ** 2 * (2 * T * a_0 + 3 * v_0 - 3 * v_T)
                          + t ** 3 * (T * a_0 + 2 * v_0 - 2 * v_T)) / T ** 3

    @staticmethod
    def velocity_profile_derivative_coefs(a_0: float, v_0: float, v_T: float, T: float):
        """
        coefficients of the derivative of the velocity profile
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param T: [sec] horizon
        :return: coefficients of the derivative of the velocity profile polynomial
        """
        coefs = np.array([3 * (T * a_0 + 2 * v_0 - 2 * v_T),
                          - 2 * T * (2 * T * a_0 + 3 * v_0 - 3 * v_T),
                          T ** 3 * a_0]) / T ** 3
        return coefs

    @staticmethod
    def acceleration_profile_function(a_0: float, v_0: float, v_T: float, T: float):
        """
        acceleration of ego at time t, given a solution to the conditions in the parameters
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param T: [sec] horizon
        :return: lambda function(s) that takes relative time in seconds and returns the acceleration
        """
        return lambda t: np.inner(
            QuarticPoly1D.velocity_profile_derivative_coefs(a_0, v_0, v_T, T),
            np.array([t ** 2, t, 1]))

    @staticmethod
    def acceleration_profile_derivative_coefs(a_0: float, v_0: float, v_T: float, T: float):
        """
        coefficients of the derivative of the acceleration profile
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param T: [sec] horizon
        :return: coefficients of the derivative of the acceleration profile polynomial
        """
        coefs = np.array([6*(T * a_0 + 2 * v_0 - 2 * v_T),
                          - 2*T * (2 * T * a_0 + 3 * v_0 - 3 * v_T)]) / T ** 3
        return coefs

    @staticmethod
    def position_profile_coefficients(a_0: np.array, v_0: np.array, v_T: np.array, T: np.array):
        """
        Given a set of quartic actions, i.e. arrays of v_0, v_T, a_0 and T (all arrays of the same size), calculate
        coefficients for longitudinal polynomial profile for each action.
        :param a_0: array of initial accelerations
        :param v_0: array of initial velocities
        :param v_T: array of target velocities
        :param T: [sec] array of action times
        :return: 2D matrix of polynomials of shape Nx6, where N = T.shape[0]
        """
        return np.c_[
            (0.25 * T * a_0 + 0.5 * v_0 - 0.5 * v_T) / T ** 3,
            (-0.666666666666667 * T * a_0 - 1.0 * v_0 + 1.0 * v_T) / T ** 2,
            0.5 * a_0,
            v_0,
            np.zeros_like(v_0)]


class QuinticPoly1D(Poly1D):
    """
    In the quintic polynomial setting we model our 2P-BVP problem as linear system of constraints Ax=b where
    x is a vector of the quintic polynomial coefficients; b is a vector of the values (p[t] is the value of the
    polynomial at time t): [p[0], p_dot[0], p_dotdot[0], p[T], p_dot[T], p_dotdot[T]]; A's rows are the
    polynomial elements at time 0 (first 3 rows) and T (last 3 rows) - the 3 rows in each block correspond to
    p, p_dot, p_dotdot.
    """

    @staticmethod
    def num_coefs():
        return 6

    @staticmethod
    def is_tracking_mode(v_0: float, v_T: np.array, a_0: float, s_0: np.array, T_m: float) -> np.array:
        """
        Checks if agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        zero.
        :param v_0: initial velocity
        :param v_T: a vector of terminal velocities
        :param a_0: initial acceleration
        :param s_0: a vector of initial distance to target
        :param T_m: headway (seconds to be behind a target)
        :return: a vector of boolean values indicating if ego is in tracking mode, meaning it actually wants to stay at
        its current velocity (usually when it stabilizes on the desired velocity in a following action)
        """
        return np.logical_and(np.isclose(v_0, v_T, atol=1e-3, rtol=0),
                              np.isclose(s_0, T_m*v_0, atol=1e-3, rtol=0)) if np.isclose(a_0, 0.0, atol=1e-3, rtol=0) else np.full(v_T.shape, False)

    @staticmethod
    def time_constraints_tensor(terminal_times: np.ndarray) -> np.ndarray:
        """
        Given the quintic polynomial setting, this function returns A as a tensor with the first dimension iterating
        over different values of T (time-horizon) provided in <terminal_times>
        :param terminal_times: 1D numpy array of different values for T
        :return: 3D numpy array of shape [len(terminal_times), 6, 6]
        """
        return np.array(
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x(0)
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # x_dot(0)
              [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],  # x_dotdot(0)
              [1.0, T, T ** 2, T ** 3, T ** 4, T ** 5],  # x(T)
              [0.0, 1.0, 2.0 * T, 3.0 * T ** 2, 4.0 * T ** 3, 5.0 * T ** 4],  # x_dot(T)
              [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T ** 2, 20.0 * T ** 3]]  # x_dotdot(T)
             for T in terminal_times], dtype=np.float)

    @staticmethod
    def inverse_time_constraints_tensor(T: np.ndarray) -> np.ndarray:
        zeros = np.zeros_like(T)
        tensor = np.array([
            [1 + zeros, zeros, zeros, zeros, zeros, zeros],
            [zeros, 1 + zeros, zeros, zeros, zeros, zeros],
            [zeros, zeros, 0.5 + zeros, zeros, zeros, zeros],
            [-10 / T ** 3, -6 / T ** 2, -3 / (2 * T), 10 / T ** 3, -4 / T ** 2, 1 / (2 * T)],
            [15 / T ** 4, 8 / T ** 3, 3 / (2 * T ** 2), -15 / T ** 4, 7 / T ** 3, -1 / T ** 2],
            [-6 / T ** 5, -3 / T ** 4, -1 / (2 * T ** 3), 6 / T ** 5, -3 / T ** 4, 1 / (2 * T ** 3)]])
        return np.transpose(tensor, (2, 0, 1))

    @staticmethod
    def cumulative_jerk(poly_coefs: np.ndarray, T: Union[float, np.ndarray]):
        """
        Computes cumulative jerk from time 0 to time T for the x(t) whose coefficients are given in <poly_coefs>
        :param poly_coefs: distance polynomial coefficients
        :param T: relative time in seconds
        :return: [float] the cummulative jerk: sum(x'''(t)^2)
        """
        a5, a4, a3, a2, a1, a0 = np.split(poly_coefs, 6, axis=-1)
        a5, a4, a3, a2, a1, a0 = a5.flatten(), a4.flatten(), a3.flatten(), a2.flatten(), a1.flatten(), a0.flatten()
        return 36 * a3 ** 2 * T + \
               144 * a3 * a4 * T ** 2 + \
               (240 * a3 * a5 + 192 * a4 ** 2) * T ** 3 + \
               720 * a4 * a5 * T ** 4 + \
               720 * a5 ** 2 * T ** 5

    @staticmethod
    def time_cost_function_derivative_coefs(w_T: np.ndarray, w_J: np.ndarray, a_0: np.ndarray, v_0: np.ndarray,
                                            v_T: np.ndarray, dx: np.ndarray, T_m: np.ndarray):
        """
        For given weights and constraints on a jerk-optimal polynomial solution, this function returns a matrix that
        contains (in each row:) the coefficients of the derivative of the cost function use for finding the optimal time
        horizon: f(T) = w_T * T + w_J * J(T) where J(T) is the accumulated jerk for given time horizon T.
        :param w_T: weight for Time component
        :param w_J: weight for Jerk component
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] distance to travel between time 0 and time T
        :param T_m: T_m: [sec] T_m * v_T is added to dx
        :return: coefficient matrix for all possibilities
        """
        zeros = np.zeros(w_T.shape[0])
        return np.c_[w_T,
                     zeros,
                     -9 * a_0 ** 2 * w_J,
                     144 * w_J * a_0 * (v_T - v_0),
                     -72 * w_J * (5 * a_0 * (T_m * v_T - dx) + 8 * (v_T - v_0) ** 2),
                     2880 * w_J * (T_m * v_T - dx) * (v_T - v_0),
                     -3600 * w_J * (T_m * v_T - dx) ** 2]

    @staticmethod
    def position_profile_coefficients(a_0: np.array, v_0: np.array, v_T: np.array, dx: np.array, T: np.array):
        """
        Given a set of quintic actions, i.e. arrays of v_0, v_T, a_0, dx and T (all arrays of the same size), calculate
        coefficients for longitudinal polynomial profile for each action.
        :param a_0: array of initial accelerations
        :param v_0: array of initial velocities
        :param v_T: array of target velocities
        :param dx: [m] array of distances to travel between time 0 and time T
        :param T: [sec] array of action times
        :return: 2D matrix of polynomials of shape Nx6, where N = T.shape[0]
        """
        zeros = np.zeros_like(T)
        return np.c_[
            -a_0 / (2 * T ** 3) - 3 * v_0 / T ** 4 - 3 * v_T / T ** 4 + 6 * dx / T ** 5,
            3 * a_0 / (2 * T ** 2) + 8 * v_0 / T ** 3 + 7 * v_T / T ** 3 - 15 * dx / T ** 4,
            -3 * a_0 / (2 * T) - 6 * v_0 / T ** 2 - 4 * v_T / T ** 2 + 10 * dx / T ** 3,
            0.5 * a_0 + zeros,
            v_0 + zeros,
            zeros]

    @staticmethod
    def distance_profile_function(a_0: float, v_0: float, v_T: float, dx: float, T: float, T_m: float):
        """
        relative distance travelled by ego at time t, given a solution to the conditions in the parameters
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] distance to travel between time 0 and time T (see T_m as well)
        :param T: [sec] horizon
        :param T_m: [sec] T_m * v_T is added to dx
        :return: lambda function(s) that takes relative time in seconds and returns the relative distance
        travelled since time 0
        """
        return lambda t: t * (T ** 5 * (a_0 * t + 2 * v_0) + T ** 2 * t ** 2 * (
                -3 * T ** 2 * a_0 - 4 * T * (3 * v_0 + 2 * v_T) + 20 * dx + 20 * v_T * (T - T_m)) + T * t ** 3 * (
                                      3 * T ** 2 * a_0 + 2 * T * (8 * v_0 + 7 * v_T) - 30 * dx - 30 * v_T * (
                                      T - T_m)) + t ** 4 * (
                                      -T ** 2 * a_0 - 6 * T * (v_0 + v_T) + 12 * dx + 12 * v_T * (T - T_m))) / (
                                     2 * T ** 5)

    @staticmethod
    def distance_from_target(a_0: float, v_0: float, v_T: float, dx: float, T: float, T_m: float):
        """
        relative distance travelled by ego at time t, given a solution to the conditions in the parameters
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] initial distance to target in time 0
        :param T: [sec] horizon
        :return: lambda function(s) that takes relative time in seconds and returns the relative distance
        travelled since time 0
        """
        return lambda t: (-T ** 5 * t * (a_0 * t + 2 * v_0) + 2 * T ** 5 * (dx + t * v_T) + T ** 2 * t ** 3 * (
            3 * T ** 2 * a_0 + 4 * T * (3 * v_0 + 2 * v_T)
            - 20 * dx - 20 * v_T * (T - T_m)) - T * t ** 4 * (
                              3 * T ** 2 * a_0 + 2 * T * (8 * v_0 + 7 * v_T) - 30 * dx - 30 * v_T * (T - T_m))
                          + t ** 5 * (T ** 2 * a_0 + 6 * T * (v_0 + v_T) - 12 * dx - 12 * v_T * (T - T_m))) / (
                                     2 * T ** 5)

    @staticmethod
    def distance_from_target_derivative_coefs(a_0: float, v_0: float, v_T: float, dx: float, T: float, T_m: float):
        """
        coefficients of the derivative of the distance profile to a target vehicle
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] distance to travel between time 0 and time T (see T_m as well)
        :param T: [sec] horizon
        :param T_m: [sec] T_m * v_T is added to dx
        :return: lambda function(s) that takes relative time in seconds and returns the velocity
        """
        coefs = np.array([5 * (T ** 2 * a_0 + 6 * T * (v_0 + v_T) - 12 * dx - 12 * v_T * (T - T_m)),
                          -4 * T * (3 * T ** 2 * a_0 + 2 * T * (8 * v_0 + 7 * v_T) - 30 * dx - 30 * v_T * (T - T_m)),
                          +3 * T ** 2 * (
                                  3 * T ** 2 * a_0 + 4 * T * (3 * v_0 + 2 * v_T) - 20 * dx - 20 * v_T * (T - T_m)),
                          -2 * T ** 5 * a_0,
                          2 * T ** 5 * (v_T - v_0)])
        return coefs

    @staticmethod
    def velocity_profile_function(a_0: float, v_0: float, v_T: float, dx: float, T: float, T_m: float):
        """
        velocity of ego at time t, given a solution to the conditions in the parameters
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] distance to travel between time 0 and time T (see T_m as well)
        :param T: [sec] horizon
        :param T_m: [sec] T_m * v_T is added to dx
        :return: lambda function(s) that takes relative time in seconds and returns the velocity
        """
        return lambda t: (2 * T ** 5 * (a_0 * t + v_0) + 3 * T ** 2 * t ** 2 * (
                -3 * T ** 2 * a_0 - 4 * T * (3 * v_0 + 2 * v_T) + 20 * dx +
                20 * v_T * (T - T_m)) + 4 * T * t ** 3 * (
                                  3 * T ** 2 * a_0 + 2 * T * (8 * v_0 + 7 * v_T) - 30 * dx - 30 * v_T * (T - T_m))
                          + 5 * t ** 4 * (-T ** 2 * a_0 - 6 * T * (v_0 + v_T) + 12 * dx + 12 * v_T * (T - T_m))) / (
                                 2 * T ** 5)

    @staticmethod
    def velocity_profile_derivative_coefs(a_0: float, v_0: float, v_T: float, dx: float, T: float, T_m: float):
        """
        coefficients of the derivative of the velocity profile to a target vehicle
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] distance to travel between time 0 and time T
        :param T: [sec] horizon
        :param T_m: [sec] T_m * v_T is added to dx
        :return: lambda function(s) that takes relative time in seconds and returns the velocity
        """
        coefs = np.array([10 * (-T ** 2 * a_0 - 6 * T * (v_0 + v_T) + 12 * dx + 12 * v_T * (T - T_m)),
                          + 6 * T * (3 * T ** 2 * a_0 + 2 * T * (8 * v_0 + 7 * v_T) - 30 * dx - 30 * v_T * (T - T_m)),
                          - 3 * T ** 2 * (
                              3 * T ** 2 * a_0 + 4 * T * (3 * v_0 + 2 * v_T) - 20 * dx - 20 * v_T * (T - T_m)),
                          T ** 5 * a_0]) / T ** 5
        return coefs

    @staticmethod
    def acceleration_profile_function(a_0: float, v_0: float, v_T: float, dx: float, T: float, T_m: float):
        """
        acceleration of ego at time t, given a solution to the conditions in the parameters
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] distance to travel between time 0 and time T
        :param T: [sec] horizon
        :param T_m: [sec] T_m * v_T is added to dx
        :return: lambda function(s) that takes relative time in seconds and returns the velocity
        """
        return lambda t: np.inner(
            QuinticPoly1D.velocity_profile_derivative_coefs(a_0, v_0, v_T, dx, T, T_m),
            np.array([t ** 3, t ** 2, t, 1]))

    @staticmethod
    def acceleration_profile_derivative_coefs(a_0: float, v_0: float, v_T: float, dx: float, T: float, T_m: float):
        """
        coefficients of the derivative of the acceleration profile
        :param a_0: [m/sec^2] acceleration at time 0
        :param v_0: [m/sec] velocity at time 0
        :param v_T: [m/sec] terminal velocity (at time T)
        :param dx: [m] distance to travel between time 0 and time T
        :param T: [sec] horizon
        :return: lambda function(s) that takes relative time in seconds and returns the velocity
        """
        coefs = np.array([30 * (-T ** 2 * a_0 - 6 * T * (v_0 + v_T) + 12 * dx + 12 * v_T * (T - T_m)),
                          + 12 * T * (3 * T ** 2 * a_0 + 2 * T * (8 * v_0 + 7 * v_T) - 30 * dx - 30 * v_T * (T - T_m)),
                          - 3 * T ** 2 * (3 * T ** 2 * a_0 + 4 * T * (3 * v_0 + 2 * v_T) - 20 * dx - 20 * v_T * (T - T_m))]) / T ** 5
        return coefs
