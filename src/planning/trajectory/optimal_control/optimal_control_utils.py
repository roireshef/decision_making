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
            :return: the x(t) polynom's coeficients after optimization
            """
            poly_coefs = np.fliplr(np.dot(constraints, A_inv.transpose()))
            poly_dot_coefs = np.apply_along_axis(np.polyder, 1, poly_coefs, 1)
            poly_dotdot_coefs = np.apply_along_axis(np.polyder, 1, poly_coefs, 2)

            return np.concatenate((poly_coefs, poly_dot_coefs, poly_dotdot_coefs), axis=1)

