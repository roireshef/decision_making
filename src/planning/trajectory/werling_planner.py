from numbers import Number
from typing import Union, Tuple

import numpy as np

from decision_making.src.global_constants import *
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.cost_function import SigmoidStatic2DBoxObstacle
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.planning.utils.columns import *
from decision_making.src.planning.utils.geometry_utils import FrenetMovingFrame
from decision_making.src.planning.utils.math import Math
from decision_making.src.state.state import State
from logging import Logger


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, logger: Logger, dt=WERLING_TIME_RESOLUTION):
        super().__init__(logger)
        self._dt = dt

    @property
    def dt(self): return self._dt

    def plan(self, state: State, reference_route: np.ndarray, goal: np.ndarray,
             cost_params: TrajectoryCostParams) -> Tuple[np.ndarray, float, TrajectoryVisualizationMsg]:
        # create road coordinate-frame
        frenet = FrenetMovingFrame(reference_route)

        # the convention is that the reference_route is given in the vehicle's coordinate-frame, so that the vehicle
        # is always at the origin. Nonetheless, the vehicle doesn't need to lay parallel to the road
        ego_in_frenet = frenet.cpoint_to_fpoint(np.array([0, 0]))
        ego_theta_diff = frenet.curve[0, R_THETA]

        # TODO: fix velocity jitters at the State level
        ego_v_x = np.max((state.ego_state.v_x, 0))

        # TODO: translate velocity (better) and acceleration of initial state
        # define constraints for the initial state
        fconstraints_t0 = FrenetConstraints(0, np.cos(ego_theta_diff) * ego_v_x, 0,
                                            ego_in_frenet[1], np.sin(ego_theta_diff) * ego_v_x, 0)

        # define constraints for the terminal (goal) state
        goal_in_frenet = frenet.cpoint_to_fpoint(goal[[EGO_X, EGO_Y]])
        goal_sx, goal_dx = goal_in_frenet[0], goal_in_frenet[1]

        goal_theta_diff = goal[EGO_THETA] - frenet.curve[frenet.sx_to_s_idx(goal_sx), R_THETA]

        sx_range = np.linspace(np.max((SX_OFFSET_MIN + goal_sx, 0)),
                               np.min((SX_OFFSET_MAX + goal_sx, frenet.length * frenet.resolution)),
                               SX_STEPS)
        sv_range = np.linspace(
            np.max((SV_OFFSET_MIN + np.cos(goal_theta_diff) * goal[EGO_V], cost_params.v_x_min_limit)),
            np.min((SV_OFFSET_MAX + np.cos(goal_theta_diff) * goal[EGO_V], cost_params.v_x_max_limit)),
            SV_STEPS)
        dx_range = np.linspace(DX_OFFSET_MIN + goal_dx,
                               DX_OFFSET_MAX + goal_dx,
                               DX_STEPS)

        fconstraints_tT = FrenetConstraints(sx_range, sv_range, 0,
                                            dx_range, np.sin(goal_theta_diff) * goal[EGO_V], 0)

        # solve problem in frenet-frame
        ftrajectories = self._solve_quintic_poly_frenet(fconstraints_t0, fconstraints_tT, cost_params.time)

        # filter resulting trajectories by velocity and acceleration
        ftrajectories_filtered = self._filter_limits(ftrajectories, cost_params)

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories_filtered)

        # compute trajectory costs
        trajectory_costs = self._compute_cost(ctrajectories, ftrajectories_filtered, state, cost_params)
        sorted_idxs = trajectory_costs.argsort()

        debug_results = TrajectoryVisualizationMsg(frenet.curve,
                                                   ctrajectories[sorted_idxs[:NUM_ALTERNATIVE_TRAJECTORIES], :, :EGO_V],
                                                   trajectory_costs[sorted_idxs[:NUM_ALTERNATIVE_TRAJECTORIES]])

        return ctrajectories[sorted_idxs[0], :, :EGO_V], trajectory_costs[sorted_idxs[0]], debug_results

    @staticmethod
    def _filter_limits(ftrajectories: np.ndarray, cost_params: TrajectoryCostParams) -> np.ndarray:
        """
        filters trajectories in their frenet-frame representation according to velocity and acceleration limits
        :param ftrajectories: trajectories in frenet-frame. A numpy array of shape [t, p, 6] with t trajectories,
        p points in each, and 6 frenet-frame axes
        :param cost_params: A CostParams instance specifying the required limitations
        :return: a numpy array of valid trajectories. shape is [reduced_t, p, 6]
        """
        conforms = np.all((ftrajectories[:, :, F_SV] >= cost_params.v_x_min_limit) &
                          (ftrajectories[:, :, F_SV] <= cost_params.v_x_max_limit) &
                          (ftrajectories[:, :, F_SA] >= cost_params.a_x_min_limit) &
                          (ftrajectories[:, :, F_SA] <= cost_params.a_x_max_limit), axis=1)
        return ftrajectories[conforms]

    @staticmethod
    def _compute_cost(ctrajectories: np.ndarray, ftrajectories: np.ndarray, state: State,
                      params: TrajectoryCostParams):
        """
        Takes trajectories (in both frenet-frame repr. and cartesian-frame repr.) and computes a cost for each one
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param params: parameters for the cost function (from behavioral layer)
        :return:
        """
        # TODO: add jerk cost
        # TODO: handle dynamic objects?

        # TODO: max instead of sum? what if close_obstacles is empty?
        ''' OBSTACLES (Sigmoid cost from bounding-box) '''
        # TODO: validate that both obstacles and ego are in world coordinates. if not, change the filter cond.
        with state.ego_state as ego, params.obstacle_cost as exp:
            close_obstacles = [SigmoidStatic2DBoxObstacle.from_object(obs, exp.k, exp.offset)
                               for obs in state.dynamic_objects
                               if np.linalg.norm([obs.x - ego.x, obs.y - ego.y]) < MAXIMAL_OBSTACLE_PROXIMITY]

            cost_per_obstacle = [obs.compute_cost(ctrajectories[:, :, 0:2]) for obs in close_obstacles]
            obstacles_costs = exp.w * np.sum(cost_per_obstacle, axis=0)

        ''' DISTANCE FROM REFERENCE ROUTE ( DX ^ 2 ) '''
        dist_from_ref_costs = params.dist_from_ref_sq_coef * np.sum(np.power(ftrajectories[:, :, F_DX], 2), axis=1)

        ''' DEVIATIONS FROM LANE/SHOULDER/ROAD '''
        deviations_costs = np.zeros(ftrajectories.shape[0])

        # add to deviations_costs the costs of deviations from the left [lane, shoulder, road]
        for exp in [params.left_lane_cost, params.left_shoulder_cost, params.left_road_cost]:
            left_offsets = ftrajectories[:, :, F_DX] - exp.offset
            deviations_costs += Math.clipped_exponent(left_offsets, exp.k, exp.w)

        # add to deviations_costs the costs of deviations from the right [lane, shoulder, road]
        for exp in [params.right_lane_cost, params.right_shoulder_cost, params.right_road_cost]:
            right_offsets = np.negative(ftrajectories[:, :, F_DX]) - exp.offset
            deviations_costs += Math.clipped_exponent(right_offsets, exp.k, exp.w)

        ''' TOTAL '''
        return obstacles_costs + dist_from_ref_costs + deviations_costs

    def _solve_quintic_poly_frenet(self, fconst_0, fconst_t, T):
        """
        Solves the two-point boundary value problem, given a set of constraints over the initial state
        and a set of constraints over the terminal state. The solution is a cartesian product of the solutions returned
        from solving two 1D problems (one for each Frenet dimension)
        :param fconst_0: a set of constraints over the initial state
        :param fconst_t: a set of constraints over the terminal state
        :param T: trajectory duration (sec.)
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
    # this class stores in its fields values for grid-search over frenet-frame parameters for the werling planner
    def __init__(self, sx: Union[np.ndarray, Number], sv: Union[np.ndarray, Number], sa: Union[np.ndarray, Number],
                 dx: Union[np.ndarray, Number], dv: Union[np.ndarray, Number], da: Union[np.ndarray, Number]):
        self._sx = np.array(sx)
        self._sv = np.array(sv)
        self._sa = np.array(sa)
        self._dx = np.array(dx)
        self._dv = np.array(dv)
        self._da = np.array(da)

    def get_grid_s(self) -> np.ndarray:
        """
        Generates a grid (cartesian product) of all (position, velocity and acceleration) on dimension S
        :return:
        """
        return np.array(np.meshgrid(self._sx, self._sv, self._sa)).T.reshape(-1, 3)

    def get_grid_d(self):
        return np.array(np.meshgrid(self._dx, self._dv, self._da)).T.reshape(-1, 3)
