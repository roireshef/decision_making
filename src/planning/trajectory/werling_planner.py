from numbers import Number
from typing import Union, Tuple

import numpy as np

from src.planning.global_constants import *
from src.planning.trajectory.cost_function import CostParams, SigmoidStatic2DBoxObstacle
from src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from src.planning.utils.columns import *
from src.planning.utils.geometry_utils import FrenetMovingFrame
from src.state.enriched_state import State as EnrichedState


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, dt=WERLING_TIME_RESOLUTION):
        self.dt = dt

    def plan(self, state: EnrichedState, reference_route: np.ndarray, goal: np.ndarray, cost_params: CostParams) -> \
            Tuple[np.ndarray, float, dict]:
        """
        see parent method in TrajectoryPlanner
        """
        frenet = FrenetMovingFrame(reference_route)
        route = frenet.curve

        # TODO: replace this object
        ego = state.ego_state

        ego_in_frenet = frenet.cpoint_to_fpoint(np.array([0, 0]))  # the ego-vehicle origin in the road-frenet-frame
        ego_theta_diff = route[0, R_THETA]

        # TODO: fix velocity jitters at the State level
        ego_v_x = np.max(ego.v_x, 0)

        fconstraints_t0 = FrenetConstraints(0, np.cos(ego_theta_diff) * ego_v_x, 0,
                                            ego_in_frenet[1], np.sin(ego_theta_diff) * ego_v_x, 0)

        goal_in_frenet = frenet.cpoint_to_fpoint(goal[[EGO_X, EGO_Y]])
        goal_sx, goal_dx = goal_in_frenet[0], goal_in_frenet[1]

        goal_theta_diff = goal[R_THETA] - route[frenet.sx_to_s_idx(goal_sx), R_THETA]

        sx_range = np.linspace(SX_OFFSET_MIN, SX_OFFSET_MAX, SX_RES)
        sv_range = np.linspace(SV_OFFSET_MIN, SV_OFFSET_MAX, SV_RES)
        dx_range = np.linspace(DX_OFFSET_MIN, DX_OFFSET_MAX, DX_RES)
        fconstraints_tT = FrenetConstraints(sx_range + goal_sx, sv_range + np.cos(goal_theta_diff) * goal[EGO_V], 0,
                                            dx_range + goal_dx, np.sin(goal_theta_diff) * goal[EGO_V], 0)

        ftrajectories = self._solve_quintic_poly_frenet(fconstraints_t0, fconstraints_tT, cost_params.T)

        # TODO: rewrite as tensor multiplication in FrenetMovingFrame
        ctrajectories = np.array([frenet.ftrajectory_to_ctrajectory(ftraj) for ftraj in ftrajectories])

        trajectory_costs = self._compute_cost(ctrajectories, ftrajectories, state)
        sorted_idxs = trajectory_costs.argsort()

        debug_results = {'trajectories': ctrajectories[sorted_idxs[:NUM_ALTERNATIVE_TRAJECTORIES], :, :EGO_V],
                         'costs': trajectory_costs[sorted_idxs[:NUM_ALTERNATIVE_TRAJECTORIES]]}

        return ctrajectories[sorted_idxs[0], :, :EGO_V], trajectory_costs[sorted_idxs[0]], debug_results

    @staticmethod
    def _compute_cost(ctrajectories: np.ndarray, ftrajectories: np.ndarray, state: EnrichedState, params: CostParams):
        """
        Takes trajectories (in both frenet-frame repr. and cartesian-frame repr.) and computes a cost for each one
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param params: parameters for the cost function (from behavioral layer)
        :return:
        """
        ''' OBSTACLES (Sigmoid cost from bounding-box) '''

        static_obstacles = [SigmoidStatic2DBoxObstacle.from_object_state(obs) for obs in state.static_objects]
        # TODO: consider max over trajectory points?
        close_obstacles = filter(lambda o: np.linalg.norm([o.x, o.y]) < MAXIMAL_OBSTACLE_PROXIMITY, static_obstacles)

        # TODO: make it use tensor operations instead
        obstacles_costs = np.sum([[obs.compute_cost(ctraj[:, 0:2]) for obs in close_obstacles]
                                  for ctraj in ctrajectories], axis=1)

        ''' DISTANCE FROM REFERENCE ROUTE ( DX ^ 2 ) '''

        ref_deviation_costs = np.sum(ftrajectories[:, :, F_DX] ** 2, axis=1)

        ''' DEVIATION FROM ROAD/LANE '''

        left_offsets = ftrajectories[:, :, F_DX] - params.left_lane_offset
        left_deviations_costs = np.sum(np.exp(np.clip(params.left_deviation_exp * left_offsets, 0, EXP_CLIP_TH)),
                                       axis=1)

        right_offsets = np.negative(ftrajectories[:, :, F_DX]) - params.right_lane_offset
        right_deviations_costs = np.sum(np.exp(np.clip(params.right_deviation_exp * right_offsets, 0, EXP_CLIP_TH)),
                                        axis=1)

        return params.lane_deviation_weight * (left_deviations_costs + right_deviations_costs) + \
               params.ref_deviation_weight * ref_deviation_costs + \
               params.obstacle_weight * obstacles_costs

    def _solve_quintic_poly_frenet(self, fconst_0, fconst_t, T):
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
        A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # dx0/sx0
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # dv0/sv0
                      [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],  # da0/sa0
                      [1.0, T, T ** 2, T ** 3, T ** 4, T ** 5],  # dxT/sxT
                      [0.0, 1.0, 2.0 * T, 3.0 * T ** 2, 4.0 * T ** 3, 5.0 * T ** 4],  # dvT/svT
                      [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T ** 2, 20.0 * T ** 3]],  # daT/saT
                     dtype=np.float64)

        A_inv = np.linalg.inv(A)
        time_samples = np.arange(0.0, T, self.dt)

        # solve for dimesion d
        constraints_d = self._cartesian_product_rows(fconst_0.get_grid_d(), fconst_t.get_grid_d())
        poly_all_coefs_d = self._solve_quintic_poly_1d(A_inv, constraints_d)

        # solve for dimesion s
        constraints_s = self._cartesian_product_rows(fconst_0.get_grid_s(), fconst_t.get_grid_s())
        poly_all_coefs_s = self._solve_quintic_poly_1d(A_inv, constraints_s)

        # concatenate all polynomial coefficients (both dimensions, up to 2nd derivative)
        # [6 poly_coef_d, 5 poly_dot_coef_d, 4 poly_dotodot_coef_d, ...
        # 6 poly_coef_s, 5 poly_dot_coef_s, 4 poly_dotodot_coef_s]
        poly_all_coefs = self._cartesian_product_rows(poly_all_coefs_s, poly_all_coefs_d)

        trajectories = np.array([np.array([np.polyval(coefs[0:6], time_samples),  # sx
                                           np.polyval(coefs[6:11], time_samples),  # sv
                                           np.polyval(coefs[11:15], time_samples),  # sa
                                           np.polyval(coefs[15:21], time_samples),  # dx
                                           np.polyval(coefs[21:26], time_samples),  # dv
                                           np.polyval(coefs[26:30], time_samples),  # da
                                           ]).transpose() for coefs in poly_all_coefs])

        return trajectories

    @staticmethod
    def _solve_quintic_poly_1d(A_inv, constraints):
        poly_coefs = np.fliplr(np.dot(constraints, A_inv.transpose()))
        poly_dot_coefs = np.apply_along_axis(np.polyder, 1, poly_coefs, 1)
        poly_dotdot_coefs = np.apply_along_axis(np.polyder, 1, poly_coefs, 2)

        return np.concatenate((poly_coefs, poly_dot_coefs, poly_dotdot_coefs), axis=1)

    @staticmethod
    def _cartesian_product_rows(mat1, mat2):
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
