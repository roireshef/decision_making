from logging import Logger
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import NoValidTrajectoriesFound
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, SX_STEPS, SV_OFFSET_MIN, SV_OFFSET_MAX, \
    SV_STEPS, DX_OFFSET_MIN, DX_OFFSET_MAX, DX_STEPS, NUM_ALTERNATIVE_TRAJECTORIES, \
    TRAJECTORY_OBSTACLE_LOOKAHEAD, SX_OFFSET_MIN, SX_OFFSET_MAX
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.messages.visualization.trajectory_visualization_message import TrajectoryVisualizationMsg
from decision_making.src.planning.trajectory.cost_function import SigmoidDynamicBoxObstacle
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import OptimalControlUtils as OC
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, FS_SV, \
    FS_SA, FS_SX, FS_DX, LIMIT_MIN, LIMIT_MAX, CartesianExtendedTrajectory, \
    CartesianTrajectories, FS_DV, FS_DA, CartesianExtendedState
from decision_making.src.planning.types import FrenetTrajectories, CartesianExtendedTrajectories, FrenetPoint
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.tensor_ops import TensorOps
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class SamplableWerlingTrajectory(SamplableTrajectory):
    def __init__(self, timestamp: float, max_sample_time: float, frenet_frame: FrenetSerret2DFrame,
                 poly_s_coefs: np.ndarray, poly_d_coefs: np.ndarray):
        """To represent a trajectory that is a result of Werling planner, we store the frenet frame used and
        two polynomial coefficients vectors (for dimensions s and d)"""
        super().__init__(timestamp, max_sample_time)
        self.frenet_frame = frenet_frame
        self.poly_s_coefs = poly_s_coefs
        self.poly_d_coefs = poly_d_coefs

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """ see base method """
        relative_time_points = time_points - self.timestamp

        # assign values from <time_points> in both s and d polynomials
        fstates_s = OC.QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_s_coefs]), relative_time_points)[0]
        fstates_d = OC.QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_d_coefs]), relative_time_points)[0]
        fstates = np.hstack((fstates_s, fstates_d))

        # project from road coordinates to cartesian coordinate frame
        # TODO: currently acceleration is computed in ftrajectory_to_ctrajectory as pseudo-derivative
        # TODO (cont.): so acceleration values here are wrong
        cstates = self.frenet_frame.ftrajectory_to_ctrajectory(fstates)

        return cstates


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, logger: Logger, predictor: Predictor, dt=WERLING_TIME_RESOLUTION):
        super().__init__(logger, predictor)
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    def plan(self, state: State, reference_route: np.ndarray, goal: CartesianExtendedState, goal_time: float,
             cost_params: TrajectoryCostParams) -> Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """ see base class """
        # create road coordinate-frame
        frenet = FrenetSerret2DFrame(reference_route)

        # The reference_route, the goal, ego and the dynamic objects are given in the global coordinate-frame.
        # The vehicle doesn't need to lay parallel to the road.
        ego_cartesian_state = np.array([state.ego_state.x, state.ego_state.y, state.ego_state.yaw, state.ego_state.v_x,
                                        state.ego_state.acceleration_lon, state.ego_state.curvature])

        ego_frenet_state = frenet.ctrajectory_to_ftrajectory(np.array([ego_cartesian_state]))[0]

        # THIS HANDLES CURRENT STATES WHERE THE VEHICLE IS STANDING STILL
        if np.any(np.isnan(ego_frenet_state)):
            self._logger.warning("Werling planner tried to convert current EgoState from cartesian-frame (%s)"
                                 "to frenet-frame (%s) and encoutered nan values. Those values are zeroed by default",
                                 str(ego_cartesian_state), str(ego_frenet_state))
            ego_frenet_state[np.isnan(ego_frenet_state)] = 0.0

        # define constraints for the initial state
        fconstraints_t0 = FrenetConstraints.from_state(ego_frenet_state)

        # define constraints for the terminal (goal) state
        goal_frenet_state = frenet.ctrajectory_to_ftrajectory(np.array([goal]))[0]

        # TODO: Determine desired final state search grid - this should be fixed with introducing different T_s, T_d
        sx_range = np.linspace(np.max((SX_OFFSET_MIN + goal_frenet_state[FS_SX], 0)) / 2,
                               np.min((SX_OFFSET_MAX + goal_frenet_state[FS_SX], (len(frenet.O)-1) * frenet.ds)),
                               SX_STEPS)
        # sx_range = np.linspace(goal_frenet_state[FS_SX]  / 2, goal_frenet_state[FS_SX], SX_STEPS)

        sv_range = np.linspace(
            np.max((SV_OFFSET_MIN + goal_frenet_state[FS_SV], cost_params.velocity_limits[0])),
            np.min((SV_OFFSET_MAX + goal_frenet_state[FS_SV], cost_params.velocity_limits[1])),
            SV_STEPS)

        dx_range = np.linspace(DX_OFFSET_MIN + goal_frenet_state[FS_DX],
                               DX_OFFSET_MAX + goal_frenet_state[FS_DX],
                               DX_STEPS)

        fconstraints_tT = FrenetConstraints(sx=sx_range, sv=sv_range, sa=goal_frenet_state[FS_SA],
                                            dx=dx_range, dv=goal_frenet_state[FS_DV], da=goal_frenet_state[FS_DA])

        # TODO: Understand what's the best resolution for cases of planning_horizon<0
        # planning is done on the time dimension relative to an anchor (currently the timestamp of the ego vehicle)
        # so time points are from t0 = 0 until some T (planning_horizon)
        planning_horizon = goal_time - state.ego_state.timestamp_in_sec
        planning_time_points = np.arange(0.0, planning_horizon, self.dt)
        assert planning_horizon >= 0

        # solve problem in frenet-frame
        ftrajectories, poly_coefs = WerlingPlanner._solve_optimization(fconstraints_t0, fconstraints_tT, planning_horizon,
                                                 planning_time_points)

        # filter resulting trajectories by velocity and acceleration
        ftrajectories_filtered, filtered_indices = self._filter_limits(ftrajectories, cost_params)

        self._logger.debug("TP has found %d valid trajectories to choose from", len(ftrajectories_filtered))

        if ftrajectories_filtered is None or len(ftrajectories_filtered) == 0:
            min_vel, max_vel = np.min(ftrajectories[:, :, FS_SV]), np.max(ftrajectories[:, :, FS_SV])
            min_acc, max_acc = np.min(ftrajectories[:, :, FS_SA]), np.max(ftrajectories[:, :, FS_SA])
            raise NoValidTrajectoriesFound("No valid trajectories found. time: %f, goal: %s, state: %s. "
                                           "planned velocities range [%s, %s]. planned accelerations range [%s, %s]",
                                           planning_horizon, goal, state, min_vel, max_vel, min_acc, max_acc)

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories_filtered)

        # compute trajectory costs at sampled times
        global_time_sample = planning_time_points + state.ego_state.timestamp_in_sec
        trajectory_costs = self._compute_cost(ctrajectories, ftrajectories_filtered, state, goal_frenet_state,
                                              cost_params, global_time_sample, self._predictor)

        sorted_idxs = trajectory_costs.argsort()

        samplable_trajectory = SamplableWerlingTrajectory(
            timestamp=state.ego_state.timestamp_in_sec,
            max_sample_time=state.ego_state.timestamp_in_sec + planning_horizon,
            frenet_frame=frenet,
            poly_s_coefs=poly_coefs[filtered_indices[sorted_idxs[0]]][:6],
            poly_d_coefs=poly_coefs[filtered_indices[sorted_idxs[0]]][6:]
        )

        # splice alternative trajectories by skipping indices - for visualization
        alternative_ids_skip_range = range(0, len(ctrajectories),
                                           max(int(len(ctrajectories) / NUM_ALTERNATIVE_TRAJECTORIES), 1))

        return samplable_trajectory, \
               ctrajectories[sorted_idxs[alternative_ids_skip_range], :, :(C_V+1)], \
               trajectory_costs[sorted_idxs[alternative_ids_skip_range]]

    @staticmethod
    def _filter_limits(ftrajectories: FrenetTrajectories, cost_params: TrajectoryCostParams) -> (FrenetTrajectories, np.ndarray):
        """
        filters trajectories in their frenet-frame representation according to velocity and acceleration limits
        :param ftrajectories: Frenet-frame trajectories (tensor) of shape [t, p, 6] with t trajectories,
        p points in each, and 6 frenet-frame axes
        :param cost_params: A CostParams instance specifying the required limitations
        :return: (a tensor of valid trajectories. shape is [reduced_t, p, 6], array of booleans - true if trajectory is
        valid, false if it breaks the limits.
        """
        conforms = np.all(
            (np.greater_equal(ftrajectories[:, :, FS_SV], cost_params.velocity_limits[LIMIT_MIN])) &
            (np.less_equal(ftrajectories[:, :, FS_SV], cost_params.velocity_limits[LIMIT_MAX])) &
            (np.greater_equal(ftrajectories[:, :, FS_SA], cost_params.acceleration_limits[LIMIT_MIN])) &
            (np.less_equal(ftrajectories[:, :, FS_SA], cost_params.acceleration_limits[LIMIT_MAX])), axis=1)
        return ftrajectories[conforms], np.argwhere(conforms).flatten()

    @staticmethod
    def _compute_cost(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories, state: State,
                      goal_in_frenet: FrenetPoint, params: TrajectoryCostParams, global_time_samples: np.ndarray,
                      predictor: Predictor) -> np.ndarray:
        """
        Takes trajectories (in both frenet-frame repr. and cartesian-frame repr.) and computes a cost for each one
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor: predictor instance to use to compute future localizations for DyanmicObjects
        :param goal_in_frenet: target state of ego
        :return: numpy array (1D) of the total cost per trajectory (in ctrajectories and ftrajectories)
        """
        # TODO: add jerk cost
        # TODO: handle dynamic objects?
        # TODO: max instead of sum? what if close_obstacles is empty?
        ''' OBSTACLES (Sigmoid cost from bounding-box) '''
        close_obstacles = \
            [SigmoidDynamicBoxObstacle.from_object(obs, params.obstacle_cost.k, params.obstacle_cost.offset,
                                                   global_time_samples, predictor)
             for obs in state.dynamic_objects
             if np.linalg.norm([obs.x - state.ego_state.x, obs.y - state.ego_state.y]) < TRAJECTORY_OBSTACLE_LOOKAHEAD]

        cost_per_obstacle = [obs.compute_cost(ctrajectories[:, :, 0:2]) for obs in close_obstacles]
        obstacles_costs = params.obstacle_cost.w * np.sum(cost_per_obstacle, axis=0)

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

    @staticmethod
    def _solve_optimization(fconst_0: FrenetConstraints, fconst_t: FrenetConstraints, T: float,
                            time_samples: np.ndarray) -> Tuple[FrenetTrajectories, np.ndarray]:
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

        return TensorOps.cartesian_product_matrix_rows(solutions_s, solutions_d), \
               TensorOps.cartesian_product_matrix_rows(poly_s, poly_d)


