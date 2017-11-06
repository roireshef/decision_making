import numpy as np
from decision_making.src.planning.utils.math import Math


class OptimalControlUtils:
    class QuinticPoly1D:
        @staticmethod
        def solve(A_inv: np.ndarray, constraints: np.ndarray) -> np.ndarray:
            """
            Given a 1D quintic polynom x(t) with 6 differential constraints on t0 (initial time) and tT (terminal time),
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
        def polyval_with_derivatives(quintic_poly_coefs: np.ndarray, time_samples: np.ndarray) -> np.ndarray:
            """
            For each (quintic) position polynomial(s) and time-sample it generates 3 values:
              1. position (evaluation of the polynomial)
              2. velocity (evaluation of the 1st derivative of the polynomial)
              2. acceleration (evaluation of the 2st derivative of the polynomial)
            :param quintic_poly_coefs: 2d numpy array [MxL] of the quintic (position) polynomials coefficients, where
             each row out of the M is a different polynomial and contains L coefficients
            :param time_samples: 1d numpy array [K] of the time stamps for the evaluation of the polynomials
            :return: 3d numpy array [M,K,3] with the following dimnesions:
                1. solution (corresponds to a given polynomial coefficients  vector in <quintic_poly_coefs>)
                2. time stamp
                3. [position value, velocity value, acceleration value]
            """
            # compute the coefficients of the polynom's 1st derivative (m=1)
            poly_dot_coefs = np.apply_along_axis(func1d=np.polyder, axis=1, arr=quintic_poly_coefs, m=1)
            # compute the coefficients of the polynom's 2nd derivative (m=2)
            poly_dotdot_coefs = np.apply_along_axis(func1d=np.polyder, axis=1, arr=quintic_poly_coefs, m=2)

            x_vals = Math.polyval2d(quintic_poly_coefs, time_samples)
            x_dot_vals = Math.polyval2d(poly_dot_coefs, time_samples)
            x_dotdot_vals = Math.polyval2d(poly_dotdot_coefs, time_samples)

            return np.dstack((x_vals, x_dot_vals, x_dotdot_vals))

        @staticmethod
        def time_constraints_tensor(terminal_times: np.ndarray) -> np.ndarray:
            """

            :param terminal_times:
            :return:
            """
            return np.array(
                [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],                                   # x(0)
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],                                   # x_dot(0)
                  [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],                                   # x_dotdot(0)
                  [1.0, T, T ** 2, T ** 3, T ** 4, T ** 5],                         # x(T)
                  [0.0, 1.0, 2.0 * T, 3.0 * T ** 2, 4.0 * T ** 3, 5.0 * T ** 4],    # x_dot(T)
                  [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T ** 2, 20.0 * T ** 3]]           # x_dotdot(T)
                 for T in terminal_times], dtype=np.float)

        @staticmethod
        def time_constraints_matrix(T: float) -> np.ndarray:
            """

            :param T:
            :return:
            """
            return OptimalControlUtils.QuinticPoly1D.time_constraints_tensor(np.array([T]))[0]

