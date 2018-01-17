from logging import Logger
from typing import Tuple

import numpy as np
import copy

from decision_making.src.exceptions import NoValidTrajectoriesFound
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, SX_STEPS, SV_OFFSET_MIN, SV_OFFSET_MAX, \
    SV_STEPS, DX_OFFSET_MIN, DX_OFFSET_MAX, DX_STEPS, VISUALIZATION_PREDICTION_RESOLUTION, NUM_ALTERNATIVE_TRAJECTORIES, \
    TRAJECTORY_OBSTACLE_LOOKAHEAD, EGO_ORIGIN_LON_FROM_REAR
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.cost_function import SigmoidDynamicBoxObstacle
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import OptimalControlUtils as OC
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner
from decision_making.src.planning.types import FP_SX, CURVE_YAW, FP_DX, C_X, C_Y, C_YAW, C_V, FS_SV, \
    FS_SA, FS_SX, FS_DX, LIMIT_MIN, LIMIT_MAX, CartesianPoint2D, CURVE_X, CURVE_Y
from decision_making.src.planning.types import FrenetTrajectories, CartesianExtendedTrajectories, FrenetPoint
from decision_making.src.planning.utils.frenet_moving_frame import FrenetMovingFrame
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.tensor_ops import TensorOps
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, logger: Logger, predictor: Predictor, dt=WERLING_TIME_RESOLUTION):
        super().__init__(logger, predictor)
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    def plan(self, state: State, reference_route: np.ndarray, goal: np.ndarray, goal_time: float,
             cost_params: TrajectoryCostParams) -> Tuple[np.ndarray, float, TrajectoryVisualizationMsg]:
        """ see base class """

        # we shift ego origin from front axle to ego center, so shift ego position accordingly
        shift = EGO_ORIGIN_LON_FROM_REAR - state.ego_state.size.length/2
        shift_vec = shift * np.array([np.cos(state.ego_state.yaw), np.sin(state.ego_state.yaw)])
        shifted_ego_pos = np.array([state.ego_state.x - shift_vec[0], state.ego_state.y - shift_vec[1]])

        # create road coordinate-frame
        frenet = FrenetMovingFrame(reference_route)

        # The reference_route, the goal, ego and the dynamic objects are given in the global coordinate-frame.
        # The vehicle doesn't need to lay parallel to the road.
        ego_in_frenet = frenet.cpoint_to_fpoint(shifted_ego_pos)
        ego_theta_diff = frenet.curve[0, CURVE_YAW] - state.ego_state.yaw

        # TODO: translate acceleration of initial state
        # define constraints for the initial state
        fconstraints_t0 = FrenetConstraints(sx=ego_in_frenet[FP_SX],
                                            sv=np.cos(ego_theta_diff) * state.ego_state.v_x +
                                               np.sin(ego_theta_diff) * state.ego_state.v_y,
                                            sa=0,
                                            dx=ego_in_frenet[FP_DX],
                                            dv=-np.sin(ego_theta_diff) * state.ego_state.v_x +
                                               np.cos(ego_theta_diff) * state.ego_state.v_y,
                                            da=0)

        # define constraints for the terminal (goal) state
        goal_in_frenet = frenet.cpoint_to_fpoint(goal[[C_X, C_Y]])
        goal_sx, goal_dx = goal_in_frenet[FP_SX], goal_in_frenet[FP_DX]

        goal_theta_diff = goal[C_YAW] - frenet.curve[frenet.sx_to_s_idx(goal_sx), CURVE_YAW]

        # TODO: Determine desired final state search grid - this should be fixed with introducing different T_s, T_d
        # sx_range = np.linspace(np.max((SX_OFFSET_MIN + goal_sx, 0)) / 2,
        #                        np.min((SX_OFFSET_MAX + goal_sx, frenet.length * frenet.resolution)),
        #                        SX_STEPS)
        sx_range = np.linspace(goal_sx / 2, goal_sx, SX_STEPS)

        sv_range = np.linspace(
            np.max((SV_OFFSET_MIN + np.cos(goal_theta_diff) * goal[C_V], cost_params.velocity_limits[0])),
            np.min((SV_OFFSET_MAX + np.cos(goal_theta_diff) * goal[C_V], cost_params.velocity_limits[1])),
            SV_STEPS)

        dx_range = np.linspace(DX_OFFSET_MIN + goal_dx,
                               DX_OFFSET_MAX + goal_dx,
                               DX_STEPS)
        dv = np.sin(goal_theta_diff) * goal[C_V]

        fconstraints_tT = FrenetConstraints(sx=sx_range, sv=sv_range, sa=0, dx=dx_range, dv=dv, da=0)

        # TODO: Understand what's the best resolution for cases of planning_horizon<0
        # planning is done on the time dimension relative to an anchor (currently the timestamp of the ego vehicle)
        # so time points are from t0 = 0 until some T (planning_horizon)
        planning_horizon = goal_time - state.ego_state.timestamp_in_sec
        planning_time_points = np.arange(0.0, planning_horizon, self.dt)
        assert planning_horizon >= 0

        # solve problem in frenet-frame
        ftrajectories, _ = self._solve_optimization(fconstraints_t0, fconstraints_tT, planning_horizon,
                                                    planning_time_points)

        # filter resulting trajectories by velocity and acceleration
        ftrajectories_filtered = self._filter_limits(ftrajectories, cost_params)

        self._logger.debug("TP has found %d valid trajectories to choose from", len(ftrajectories_filtered))

        if ftrajectories_filtered is None or len(ftrajectories_filtered) == 0:
            raise NoValidTrajectoriesFound("No valid trajectories found. time: %f, goal: %s, state: %s",
                                           planning_horizon, goal, state)

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories_filtered)

        # compute trajectory costs at sampled times
        global_time_sample = planning_time_points + state.ego_state.timestamp_in_sec
        trajectory_costs = self._compute_cost(ctrajectories, ftrajectories_filtered, state, shifted_ego_pos,
                                              goal_in_frenet, cost_params, global_time_sample, self._predictor)

        sorted_idxs = trajectory_costs.argsort()

        alternative_ids_skip_range = range(0, len(ctrajectories),
                                           max(int(len(ctrajectories) / NUM_ALTERNATIVE_TRAJECTORIES), 1))

        prediction_timestamps = np.arange(state.ego_state.timestamp_in_sec,
                                          state.ego_state.timestamp_in_sec + planning_horizon,
                                          VISUALIZATION_PREDICTION_RESOLUTION, float)

        # TODO: move this to visualizer.
        # Curently we are predicting the state at ego's timestamp and at the end of the traj execution time.
        predicted_states = self._predictor.predict_state(state=state, prediction_timestamps=prediction_timestamps)
        # predicted_states[0] is the current state
        # predicted_states[1] is the predicted state in the end of the execution of traj.
        debug_results = TrajectoryVisualizationMsg(reference_route,
                                                   ctrajectories[sorted_idxs[alternative_ids_skip_range], :, :C_V],
                                                   trajectory_costs[sorted_idxs[alternative_ids_skip_range]],
                                                   predicted_states[0],
                                                   predicted_states[1:],
                                                   planning_horizon)

        #TODO: ego origin: transform trajectories to the front axle

        return ctrajectories[sorted_idxs[0], :, :C_V + 1], trajectory_costs[sorted_idxs[0]], debug_results

    @staticmethod
    def _filter_limits(ftrajectories: FrenetTrajectories, cost_params: TrajectoryCostParams) -> FrenetTrajectories:
        """
        filters trajectories in their frenet-frame representation according to velocity and acceleration limits
        :param ftrajectories: Frenet-frame trajectories (tensor)
        :param cost_params: A CostParams instance specifying the required limitations
        :return: a tensor of valid trajectories. shape is [reduced_t, p, 6]
        """
        conforms = np.all(
            (np.greater_equal(ftrajectories[:, :, FS_SV], cost_params.velocity_limits[LIMIT_MIN])) &
            (np.less_equal(ftrajectories[:, :, FS_SV], cost_params.velocity_limits[LIMIT_MAX])) &
            (np.greater_equal(ftrajectories[:, :, FS_SA], cost_params.acceleration_limits[LIMIT_MIN])) &
            (np.less_equal(ftrajectories[:, :, FS_SA], cost_params.acceleration_limits[LIMIT_MAX])), axis=1)
        return ftrajectories[conforms]

    @staticmethod
    def _compute_cost(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories, state: State,
                      shifted_ego_pos: np.ndarray, goal_in_frenet: FrenetPoint, params: TrajectoryCostParams,
                      global_time_samples: np.ndarray, predictor: Predictor):
        """
        Takes trajectories (in both frenet-frame repr. and cartesian-frame repr.) and computes a cost for each one
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param shifted_ego_pos: shifted global ego position by shifting origin to ego center
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor:
        :param goal_in_frenet: target state of ego
        :return:
        """
        # TODO: add jerk cost
        # TODO: handle dynamic objects?
        # TODO: max instead of sum? what if close_obstacles is empty?
        ''' OBSTACLES (Sigmoid cost from bounding-box) '''
        offset = np.array([params.obstacle_cost_x.offset, params.obstacle_cost_y.offset])
        close_obstacles = \
            [SigmoidDynamicBoxObstacle.from_object(obj=obs, k=params.obstacle_cost_x.k, offset=offset,
                                                    time_samples=global_time_samples, predictor=predictor)
             for obs in state.dynamic_objects
             if np.linalg.norm([obs.x - shifted_ego_pos[0], obs.y - shifted_ego_pos[1]]) < TRAJECTORY_OBSTACLE_LOOKAHEAD]

        cost_per_obstacle = [obs.compute_cost(ctrajectories[:, :, 0:2]) for obs in close_obstacles]
        obstacles_costs = params.obstacle_cost_x.w * np.sum(cost_per_obstacle, axis=0)

        ''' SQUARED DISTANCE FROM GOAL SCORE '''
        # make theta_diff to be in [-pi, pi]
        last_fpoints = ftrajectories[:, -1, :]
        dist_from_goal_costs = \
            params.dist_from_goal_lon_sq_cost * np.square(last_fpoints[:, FS_SX] - goal_in_frenet[FP_SX]) + \
            params.dist_from_goal_lat_sq_cost * np.square(last_fpoints[:, FS_DX] - goal_in_frenet[FP_DX])
        dist_from_ref_costs = params.dist_from_ref_sq_cost * np.sum(np.power(ftrajectories[:, :, FS_DX], 2), axis=1)

        ''' DEVIATIONS FROM LANE/SHOULDER/ROAD '''
        deviations_costs = np.zeros(ftrajectories.shape[0])

        # add to deviations_costs the costs of deviations from the left [lane, shoulder, road]
        for exp in [params.left_lane_cost, params.left_shoulder_cost, params.left_road_cost]:
            left_offsets = ftrajectories[:, :, FS_DX] - exp.offset
            deviations_costs += np.mean(Math.clipped_sigmoid(left_offsets, exp.w, exp.k), axis=1)

        # add to deviations_costs the costs of deviations from the right [lane, shoulder, road]
        for exp in [params.right_lane_cost, params.right_shoulder_cost, params.right_road_cost]:
            right_offsets = np.negative(ftrajectories[:, :, FS_DX]) - exp.offset
            deviations_costs += np.mean(Math.clipped_sigmoid(right_offsets, exp.w, exp.k), axis=1)

        ''' TOTAL '''
        return obstacles_costs + dist_from_ref_costs + dist_from_goal_costs + deviations_costs

    def _solve_optimization(self, fconst_0: FrenetConstraints, fconst_t: FrenetConstraints, T: float,
                            time_samples: np.ndarray):
        """
        Solves the two-point boundary value problem, given a set of constraints over the initial state
        and a set of constraints over the terminal state. The solution is a cartesian product of the solutions returned
        from solving two 1D problems (one for each Frenet dimension)
        :param fconst_0: a set of constraints over the initial state
        :param fconst_t: a set of constraints over the terminal state
        :param T: trajectory duration (sec.)
        :param time_samples: [sec] from 0 to T with step=self.dt
        :return: a tuple: (points-matrix of rows in the form [sx, sv, sa, dx, dv, da],
        poly-coefficients-matrix of rows in the form [c0_s, c1_s, ... c5_s, c0_d, ..., c5_d])
        """
        A = OC.QuinticPoly1D.time_constraints_matrix(T)
        A_inv = np.linalg.inv(A)

        # solve for dimesion d
        constraints_d = TensorOps.cartesian_product_matrix_rows(fconst_0.get_grid_d(), fconst_t.get_grid_d())
        poly_d = OC.QuinticPoly1D.solve(A_inv, constraints_d)
        solutions_d = OC.QuinticPoly1D.polyval_with_derivatives(poly_d, time_samples)

        # solve for dimesion s
        constraints_s = TensorOps.cartesian_product_matrix_rows(fconst_0.get_grid_s(), fconst_t.get_grid_s())
        poly_s = OC.QuinticPoly1D.solve(A_inv, constraints_s)
        solutions_s = OC.QuinticPoly1D.polyval_with_derivatives(poly_s, time_samples)

        return (TensorOps.cartesian_product_matrix_rows(solutions_s, solutions_d), (poly_s, poly_d))


