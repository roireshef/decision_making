import numpy as np


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
            :return: x(t) coeficients, x_dot(t) coeficients, x_dot_dot coeficients - all concatenated in a numpy array
            """
            poly_coefs = np.fliplr(np.dot(constraints, A_inv.transpose()))
            # compute the coefficients of the polynom's 1st derivative (m=1)
            poly_dot_coefs = np.apply_along_axis(func1d=np.polyder, axis=1, arr=poly_coefs, m=1)
            # compute the coefficients of the polynom's 2nd derivative (m=2)
            poly_dotdot_coefs = np.apply_along_axis(func1d=np.polyder, axis=1, arr=poly_coefs, m=2)

            return np.concatenate((poly_coefs, poly_dot_coefs, poly_dotdot_coefs), axis=1)

        @staticmethod
        def time_constraints_tensor(terminal_times: np.ndarray) -> np.ndarray:
            return np.array(
                [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],                                   # x(0)
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],                                   # x_dot(0)
                  [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],                                   # x_dotdot(0)
                  [1.0, T, T ** 2, T ** 3, T ** 4, T ** 5],                         # x(T)
                  [0.0, 1.0, 2.0 * T, 3.0 * T ** 2, 4.0 * T ** 3, 5.0 * T ** 4],    # x_dot(T)
                  [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T ** 2, 20.0 * T ** 3]]           # x_dotdot(T)
                 for T in terminal_times], dtype=np.float16)

        @staticmethod
        def time_constraints_matrix(T: float) -> np.ndarray:
            return OptimalControlUtils.QuinticPoly1D.time_constraints_tensor(np.array([T]))[0]

        @staticmethod
        def find_second_der_extrema(poly_coefs: np.ndarray) -> np.ndarray:
            """
            find the extremas of the second derivative of the quintic-polynom, by finding the roots of the 3rd
            derivative (which is itself a 2nd degree polynomial)
            :param poly_coefs: 1D numpy array corresponds to coefficients of [x**5, x**4, ..., 1]
            :return: numpy array with two values corresponds to the x-values of the extremas
            """
            return np.roots(np.array([60.0, 24.0, 6.0]) * poly_coefs[:3])
