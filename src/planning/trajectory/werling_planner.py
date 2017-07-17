from numbers import Number
from typing import Union

import numpy as np

from src.planning.global_constants import WERLING_DT
from src.planning.trajectory.cost_function import CostParams
from src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from src.planning.utils.columns import *
from src.planning.utils.geometry_utils import FrenetMovingFrame


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, dt=WERLING_DT):
        self.dt = dt

    # TODO: link state to its type once interface is merged
    def plan(self, state, reference_route: np.ndarray, goal: np.ndarray, cost_params: CostParams):
        frenet = FrenetMovingFrame(reference_route)
        route = frenet.curve

        # TODO: replace this object
        ego = state.ego_state

        ego_in_frenet = frenet.cpoint_to_fpoint(np.array([0, 0]))      # the ego-vehicle origin in the road-frenet-frame
        ego_theta_diff = route[0, C_THETA]

        # TODO: fix velocity jitters at the State level
        ego_v_x = np.max(ego.v_x, 0)

        fconstraints_t0 = FrenetConstraints(0,                  np.cos(ego_theta_diff) * ego_v_x,   0,
                                            ego_in_frenet[1],   np.sin(ego_theta_diff) * ego_v_x,   0)

        goal_in_frenet = frenet.cpoint_to_fpoint(goal[[EGO_X, EGO_Y]])
        goal_sx, goal_dx = goal_in_frenet[0], goal_in_frenet[1]

        goal_theta_diff = goal[C_THETA] - route[frenet.sx_to_s_idx(goal_sx), C_THETA]

        # TODO: parameterize this smartly
        sv_range = np.linspace(-2, 2, 6)
        sx_range = np.linspace(-3, 0, 3)
        dx_range = np.linspace(-1.5, 1.5, 3)
        fconstraints_tT = FrenetConstraints(sx_range + goal_sx, sv_range + np.cos(goal_theta_diff) * goal[EGO_V],   0,
                                            dx_range + goal_dx, np.sin(goal_theta_diff) * goal[EGO_V],              0)

        ftrajectories = self.__solve_quintic_poly_frenet(fconstraints_t0, fconstraints_tT, cost_params.T)
        ctrajectories = np.array([frenet.ftrajectory_to_ctrajectory(ftraj) for ftraj in ftrajectories])

        # TODO: copy and clean the rest

        pass

    # solves the 2-point boundary value problem for each cell on a frenet grid
    # with constant time (T)
    def __solve_quintic_poly_frenet(self, fconst_0, fconst_t, T):
        """
        Solves the two-point boundary value problem, given a set of constraints over the initial state
        and a set of constraints over the terminal state. The solution is a cartesian product of the solutions returned
        from solving two 1D problems (one for each Frenet dimension)
        :param fconst_0: a set of constraints over the initial state
        :param fconst_t: a set of constraints over the terminal state
        :param T: trajectory duration (sec.)
        :param dt: trajectory time-resolution (sec.)
        :return: a matrix of rows of the form [sx, sv, sa, dx, dv, da]
        """
        A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],                                       # dx0/sx0
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],                                       # dv0/sv0
                      [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],                                       # da0/sa0
                      [1.0, T, T ** 2, T ** 3, T ** 4, T ** 5],                             # dxT/sxT
                      [0.0, 1.0, 2.0 * T, 3.0 * T ** 2, 4.0 * T ** 3, 5.0 * T ** 4],        # dvT/svT
                      [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T ** 2, 20.0 * T ** 3]],              # daT/saT
                     dtype=np.float64)

        A_inv = np.linalg.inv(A)
        time_samples = np.arange(0.0, T, self.dt)

        # solve for dimesion d
        constraints_d = self.__cartesian_product_rows(fconst_0.get_grid_d(), fconst_t.get_grid_d())
        poly_all_coefs_d = self.__solve_quintic_poly_1d(A_inv, constraints_d)

        # solve for dimesion s
        constraints_s = self.__cartesian_product_rows(fconst_0.get_grid_s(), fconst_t.get_grid_s())
        poly_all_coefs_s = self.__solve_quintic_poly_1d(A_inv, constraints_s)

        # concatenate all polynomial coefficients (both dimensions, up to 2nd derivative)
        # [6 poly_coef_d, 5 poly_dot_coef_d, 4 poly_dotodot_coef_d, ...
        # 6 poly_coef_s, 5 poly_dot_coef_s, 4 poly_dotodot_coef_s]
        poly_all_coefs = self.__cartesian_product_rows(poly_all_coefs_s, poly_all_coefs_d)

        trajectories = np.array([np.array([np.polyval(coefs[0:6], time_samples),  # sx
                                           np.polyval(coefs[6:11], time_samples),  # sv
                                           np.polyval(coefs[11:15], time_samples),  # sa
                                           np.polyval(coefs[15:21], time_samples),  # dx
                                           np.polyval(coefs[21:26], time_samples),  # dv
                                           np.polyval(coefs[26:30], time_samples),  # da
                                           ]).transpose() for coefs in poly_all_coefs])

        return trajectories

    @staticmethod
    def __solve_quintic_poly_1d(A_inv, constraints):

        poly_coefs = np.fliplr(np.dot(constraints, A_inv.transpose()))
        poly_dot_coefs = np.apply_along_axis(np.polyder, 1, poly_coefs, 1)
        poly_dotdot_coefs = np.apply_along_axis(np.polyder, 1, poly_coefs, 2)

        return np.concatenate((poly_coefs, poly_dot_coefs, poly_dotdot_coefs), axis=1)

    @staticmethod
    def __cartesian_product_rows(mat1, mat2):
        return np.array([np.concatenate((mat1[idx1, :], mat2[idx2, :]))
                         for idx1 in range(mat1.shape[0])
                         for idx2 in range(mat2.shape[0])])

class FrenetConstraints:
    def __init__(self, sx: Union[np.ndarray, Number], sv: Union[np.ndarray, Number], sa: Union[np.ndarray, Number],
                 dx: Union[np.ndarray, Number], dv: Union[np.ndarray, Number], da: Union[np.ndarray, Number]):
        self.sx = np.array(sx)
        self.sv = np.array(sv)
        self.sa = np.array(sa)
        self.dx = np.array(dx)
        self.dv = np.array(dv)
        self.da = np.array(da)

    def get_grid_s(self):
        return np.array(np.meshgrid(self.sx, self.sv, self.sa)).T.reshape(-1, 3)

    def get_grid_d(self):
        return np.array(np.meshgrid(self.dx, self.dv, self.da)).T.reshape(-1, 3)
