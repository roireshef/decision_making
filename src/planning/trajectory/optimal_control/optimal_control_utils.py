from abc import abstractmethod

import numpy as np

from decision_making.src.planning.types import Limits
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils


class Poly1D:
    @staticmethod
    @abstractmethod
    def num_coefs():
        pass

    @staticmethod
    @abstractmethod
    def time_constraints_tensor(terminal_times: np.ndarray) -> np.ndarray:
        pass

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
    def polyval_with_derivatives(poly_coefs: np.ndarray, time_samples: np.ndarray) -> np.ndarray:
        """
        For each x(t) position polynomial(s) and time-sample it generates 3 values:
          1. position (evaluation of the polynomial)
          2. velocity (evaluation of the 1st derivative of the polynomial)
          2. acceleration (evaluation of the 2st derivative of the polynomial)
        :param poly_coefs: 2d numpy array [MxL] of the quartic (position) polynomials coefficients, where
         each row out of the M is a different polynomial and contains L coefficients
        :param time_samples: 1d numpy array [K] of the time stamps for the evaluation of the polynomials
        :return: 3d numpy array [M,K,3] with the following dimnesions:
            1. solution (corresponds to a given polynomial coefficients  vector in <poly_coefs>)
            2. time stamp
            3. [position value, velocity value, acceleration value]
        """
        # compute the coefficients of the polynom's 1st derivative (m=1)
        poly_dot_coefs = np.apply_along_axis(func1d=np.polyder, axis=1, arr=poly_coefs, m=1)
        # compute the coefficients of the polynom's 2nd derivative (m=2)
        poly_dotdot_coefs = np.apply_along_axis(func1d=np.polyder, axis=1, arr=poly_coefs, m=2)

        x_vals = Math.polyval2d(poly_coefs, time_samples)
        x_dot_vals = Math.polyval2d(poly_dot_coefs, time_samples)
        x_dotdot_vals = Math.polyval2d(poly_dotdot_coefs, time_samples)

        return np.dstack((x_vals, x_dot_vals, x_dotdot_vals))

    @classmethod
    def time_constraints_matrix(cls, T: float) -> np.ndarray:
        """
        Given the polynomial setting, this function returns A as a tensor with the first dimension iterating
        over different values of T (time-horizon) provided in <terminal_times>
        :param terminal_times: 1D numpy array of different values for T
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
                :param polys_coefs: 2D numpy array with N polynomials and <cls.num_coefs()> coefficients each
                :param T_vals: 1D numpy array of planning-times [N]
                :param limits: minimal and maximal allowed values for the <degree> derivative of x(t)
                :return: 1D numpy array of booleans where True means the restrictions are met.
                """
        # TODO: a(0) and a(T) checks are omitted as they they are provided by the user.
        # compute extrema points, by finding the roots of the 3rd derivative
        jerk_poly = Math.polyder2d(poly_coefs, m=degree + 1)
        acc_poly = Math.polyder2d(poly_coefs, m=degree)
        acc_suspected_points = np.apply_along_axis(np.roots, 1, jerk_poly)  # TODO: this should use matrix operations!
        acc_suspected_values = Math.zip_polyval2d(acc_poly, acc_suspected_points)

        # are extrema points out of [0, T] range
        is_suspected_point_in_time_range = np.greater_equal(acc_suspected_points, 0) & \
                                           np.less_equal(acc_suspected_points, T_vals[:, np.newaxis])

        # check if extrema values are within [a_min, a_max] limits
        is_suspected_value_in_limits = NumpyUtils.is_in_limits(acc_suspected_values, limits)

        # for all extrema points that are inside the time range, verify that their values are inside [a_min, a_max]
        return np.all(np.logical_or(np.logical_not(is_suspected_point_in_time_range), is_suspected_value_in_limits),
                      axis=1)

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
        :param polys_coefs: 2D numpy array with N polynomials and 6 coefficients each [Nx6]
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


class QuarticPoly1D(Poly1D):
    @staticmethod
    def num_coefs():
        return 5

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
