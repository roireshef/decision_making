from logging import Logger
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import NoValidTrajectoriesFound
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, SX_STEPS, SV_OFFSET_MIN, SV_OFFSET_MAX, \
    SV_STEPS, DX_OFFSET_MIN, DX_OFFSET_MAX, DX_STEPS, TRAJECTORY_OBSTACLE_LOOKAHEAD, SX_OFFSET_MIN, SX_OFFSET_MAX, \
    TD_STEPS, LAT_ACC_LIMITS
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.cost_function import SigmoidDynamicBoxObstacle
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import OptimalControlUtils as OC
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, FS_SV, \
    FS_SA, FS_SX, FS_DX, LIMIT_MIN, LIMIT_MAX, CartesianExtendedTrajectory, \
    CartesianTrajectories, FS_DV, FS_DA, CartesianExtendedState, FrenetState2D, C_A, C_K, FrenetState1D, \
    FrenetTrajectories1D
from decision_making.src.planning.types import FrenetTrajectories2D, CartesianExtendedTrajectories, FrenetPoint
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class SamplableWerlingTrajectory(SamplableTrajectory):
    def __init__(self, timestamp: float, lon_plan_horizon: float, lat_plan_horizon: float, frenet_frame: FrenetSerret2DFrame,
                 poly_s_coefs: np.ndarray, poly_d_coefs: np.ndarray):
        """To represent a trajectory that is a result of Werling planner, we store the frenet frame used and
        two polynomial coefficients vectors (for dimensions s and d)"""
        super().__init__(timestamp, lon_plan_horizon)
        self.lat_plan_horizon = lat_plan_horizon
        self.frenet_frame = frenet_frame
        self.poly_s_coefs = poly_s_coefs
        self.poly_d_coefs = poly_d_coefs

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """See base method for API. In this specific representation of the trajectory, we sample from s-axis polynomial
        and partially from d-axis polynomial and extrapolate the d-axis to conform to the trajectory's total duration"""

        relative_time_points = time_points - self.timestamp

        # Make sure no unplanned extrapolation will occur due to overreaching time points
        # This check is done in relative-to-ego units
        assert max(relative_time_points) < self.lon_plan_horizon

        # assign values from <time_points> in s-axis polynomial
        fstates_s = OC.QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_s_coefs]), relative_time_points)[0]

        # assign values from <time_points> in d-axis polynomial
        in_range_relative_time_points = relative_time_points[relative_time_points <= self.lat_plan_horizon]
        partial_fstates_d = OC.QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_d_coefs]),
                                                                      in_range_relative_time_points)[0]

        # Expand lateral solution to the size of the longitudinal solution with its final positions replicated
        # NOTE: we assume that velocity and accelerations = 0 !!
        full_fstates_d = WerlingPlanner.extrapolate_solution_1d(
            fstates=np.array([partial_fstates_d]),
            length=relative_time_points.size - in_range_relative_time_points.size,
            override_values=np.zeros(3),
            override_mask=np.array([0, 1, 1])
        )[0]

        fstates = np.hstack((fstates_s, full_fstates_d))

        # project from road coordinates to cartesian coordinate frame
        cstates = self.frenet_frame.ftrajectory_to_ctrajectory(fstates)

        return cstates


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, logger: Logger, predictor: Predictor, dt=WERLING_TIME_RESOLUTION):
        super().__init__(logger, predictor)
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    def plan(self, state: State, reference_route: np.ndarray, goal: CartesianExtendedState, lon_plan_horizon: float,
             cost_params: TrajectoryCostParams) -> Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """ see base class """

        # create road coordinate-frame
        frenet = FrenetSerret2DFrame(reference_route)

        # The reference_route, the goal, ego and the dynamic objects are given in the global coordinate-frame.
        # The vehicle doesn't need to lay parallel to the road.
        ego_cartesian_state = np.array([state.ego_state.x, state.ego_state.y, state.ego_state.yaw, state.ego_state.v_x,
                                        state.ego_state.acceleration_lon, state.ego_state.curvature])

        ego_frenet_state: FrenetState2D = frenet.cstate_to_fstate(ego_cartesian_state)

        # THIS HANDLES CURRENT STATES WHERE THE VEHICLE IS STANDING STILL
        if np.any(np.isnan(ego_frenet_state)):
            self._logger.warning("Werling planner tried to convert current EgoState from cartesian-frame (%s)"
                                 "to frenet-frame (%s) and encountered nan values. Those values are zeroed by default",
                                 str(ego_cartesian_state), str(ego_frenet_state))
            ego_frenet_state[np.isnan(ego_frenet_state)] = 0.0

        # define constraints for the initial state
        fconstraints_t0 = FrenetConstraints.from_state(ego_frenet_state)

        # define constraints for the terminal (goal) state
        goal_frenet_state: FrenetState2D = frenet.cstate_to_fstate(goal)

        # TODO: Determine desired final state search grid - this should be fixed with introducing different T_s, T_d
        sx_range = np.linspace(np.max((SX_OFFSET_MIN + goal_frenet_state[FS_SX], 0)),
                               np.min((SX_OFFSET_MAX + goal_frenet_state[FS_SX], (len(frenet.O) - 1) * frenet.ds)),
                               SX_STEPS)
        # sx_range = np.linspace(goal_frenet_state[FS_SX]  / 2, goal_frenet_state[FS_SX], SX_STEPS)

        sv_range = np.linspace(
            np.max((SV_OFFSET_MIN + goal_frenet_state[FS_SV], cost_params.velocity_limits[LIMIT_MIN])),
            np.min((SV_OFFSET_MAX + goal_frenet_state[FS_SV], cost_params.velocity_limits[LIMIT_MAX])),
            SV_STEPS)

        dx_range = np.linspace(DX_OFFSET_MIN + goal_frenet_state[FS_DX],
                               DX_OFFSET_MAX + goal_frenet_state[FS_DX],
                               DX_STEPS)

        fconstraints_tT = FrenetConstraints(sx=sx_range, sv=sv_range, sa=goal_frenet_state[FS_SA],
                                            dx=dx_range, dv=goal_frenet_state[FS_DV], da=goal_frenet_state[FS_DA])

        assert lon_plan_horizon >= self.dt
        # Make sure T_s values are multiples of dt (or else the matrix, calculated using T_s, and the latitudinal
        #  time axis, lon_time_samples, won't fit).
        lon_plan_horizon = Math.round_to_step(lon_plan_horizon, self.dt)

        # planning is done on the time dimension relative to an anchor (currently the timestamp of the ego vehicle)
        # so time points are from t0 = 0 until some T (lon_plan_horizon)
        planning_time_points = np.arange(self.dt, lon_plan_horizon + np.finfo(np.float16).eps, self.dt)

        # Latitudinal planning horizon(Td) lower bound, now approximated from x=a*t^2
        # TODO: determine lower bound according to physical constraints and ego control limitations
        min_lat_movement = np.min(np.abs(fconstraints_tT.get_grid_d()[:, 0] - fconstraints_t0.get_grid_d()[0, 0]))
        low_bound_lat_plan_horizon = max(np.sqrt((2*min_lat_movement) / LAT_ACC_LIMITS[LIMIT_MAX]), self.dt)

        assert lon_plan_horizon >= low_bound_lat_plan_horizon

        self._logger.debug('WerlingPlanner is planning from %s (frenet) to %s (frenet) in %s seconds' %
                           (NumpyUtils.str_log(ego_frenet_state), NumpyUtils.str_log(goal_frenet_state),
                            lon_plan_horizon))

        lat_plan_horizon_vals = WerlingPlanner._create_lat_horizon_grid(lon_plan_horizon, low_bound_lat_plan_horizon, self.dt)

        # solve problem in frenet-frame
        ftrajectories, poly_coefs, poly_lat_horizons = WerlingPlanner._solve_optimization(fconstraints_t0, fconstraints_tT,
                                                                       lon_plan_horizon,
                                                                       lat_plan_horizon_vals, self.dt)

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories: CartesianExtendedTrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories)

        # filter resulting trajectories by velocity and accelerations limits - this is now done in Cartesian frame
        # which takes into account the curvature of the road applied to trajectories planned in the Frenet frame
        filtered_indices = self._filter_limits(ctrajectories, cost_params)
        ctrajectories_filtered = ctrajectories[filtered_indices]
        ftrajectories_filtered = ftrajectories[filtered_indices]

        self._logger.debug("TP has found %d valid trajectories to choose from", len(ctrajectories_filtered))

        if len(ctrajectories_filtered) == 0:
            lat_acc = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
            raise NoValidTrajectoriesFound("No valid trajectories found. time: %f, goal: %s, state: %s. "
                                           "planned velocities range [%s, %s] (limits: %s); "
                                           "planned lon. accelerations range [%s, %s] (limits: %s); "
                                           "planned lat. accelerations range [%s, %s] (limits: %s); " %
                                           (lon_plan_horizon, NumpyUtils.str_log(goal), str(state).replace('\n', ''),
                                            np.min(ctrajectories[:, :, C_V]), np.max(ctrajectories[:, :, C_V]),
                                            NumpyUtils.str_log(cost_params.velocity_limits),
                                            np.min(ctrajectories[:, :, C_A]), np.max(ctrajectories[:, :, C_A]),
                                            NumpyUtils.str_log(cost_params.lon_acceleration_limits),
                                            np.min(lat_acc), np.max(lat_acc),
                                            NumpyUtils.str_log(cost_params.lat_acceleration_limits)))

        # compute trajectory costs at sampled times
        global_time_sample = planning_time_points + state.ego_state.timestamp_in_sec
        filtered_trajectory_costs = self._compute_cost(ctrajectories_filtered, ftrajectories_filtered, state,
                                                       goal_frenet_state, cost_params, global_time_sample, self._predictor)

        sorted_filtered_idxs = filtered_trajectory_costs.argsort()

        samplable_trajectory = SamplableWerlingTrajectory(
            timestamp=state.ego_state.timestamp_in_sec,
            lon_plan_horizon=lon_plan_horizon,
            lat_plan_horizon=poly_lat_horizons[filtered_indices[sorted_filtered_idxs[0]]],
            frenet_frame=frenet,
            poly_s_coefs=poly_coefs[filtered_indices[sorted_filtered_idxs[0]]][:6],
            poly_d_coefs=poly_coefs[filtered_indices[sorted_filtered_idxs[0]]][6:]
        )

        return samplable_trajectory, \
               ctrajectories_filtered[sorted_filtered_idxs, :, :(C_V + 1)], \
               filtered_trajectory_costs[sorted_filtered_idxs]

    @staticmethod
    def _filter_limits(ctrajectories: CartesianExtendedTrajectories, cost_params: TrajectoryCostParams) -> np.ndarray:
        """
        Given a set of trajectories in Cartesian coordinate-frame, it validates them against the following limits:
        longitudinal velocity, longitudinal acceleration, lateral acceleration (via curvature and lon. velocity)
        :param ctrajectories: CartesianExtendedTrajectories object of trajectories to validate
        :param cost_params: TrajectoryCostParams object that holds desired limits (for validation)
        :return: Indices along the 1st dimension in <ctrajectories> (trajectory index) for valid trajectories
        """
        lon_acceleration = ctrajectories[:, :, C_A]
        lat_acceleration = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
        lon_velocity = ctrajectories[:, :, C_V]

        conforms = np.all(
            NumpyUtils.is_in_limits(lon_velocity, cost_params.velocity_limits) &
            NumpyUtils.is_in_limits(lon_acceleration, cost_params.lon_acceleration_limits) &
            NumpyUtils.is_in_limits(lat_acceleration, cost_params.lat_acceleration_limits), axis=1)

        return np.argwhere(conforms).flatten()

    @staticmethod
    def _compute_cost(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D, state: State,
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
    def _create_lat_horizon_grid(T_s: float, T_d_low_bound: float, dt: float) -> np.ndarray:
        """
        Receives the lower bound of the lateral time horizon T_d_low_bound and the longitudinal time horizon T_s
        and returns a grid of possible lateral planning time values.
        :param T_s: longitudinal trajectory duration (sec.), relative to ego. Ts has to be a multiple of dt.
        :param T_d_low_bound: lower bound on latitudinal trajectory duration (sec.), relative to ego. Higher bound is Ts.
        :param dt: [sec] basic time unit from constructor.
        :return: numpy array (1D) of the possible lateral planning horizons
        """
        T_d_vals = np.array([T_d_low_bound])
        if T_s != T_d_low_bound:
            T_d_vals = np.linspace(T_d_low_bound, T_s, TD_STEPS)

        # Make sure T_d_vals values are multiples of dt (or else the matrix, calculated using T_d, and the latitudinal
        #  time axis, lat_time_samples, won't fit).
        T_d_vals = Math.round_to_step(T_d_vals, dt)

        return T_d_vals

    @staticmethod
    def _get_werling_poly(constraints: np.ndarray, T: float) -> np.ndarray:
        """
        Solves the two-point boundary value problem, given a set of constraints over the initial and terminal states.
        :param constraints: 3D numpy array of a set of constraints over the initial and terminal states
        :param T: longitudinal/lateral trajectory duration (sec.), relative to ego. T has to be a multiple of WerlingPlanner.dt
        :return: a poly-coefficients-matrix of rows in the form [c0_s, c1_s, ... c5_s, c0_d, ..., c5_d]
        """
        A = OC.QuinticPoly1D.time_constraints_matrix(T)
        A_inv = np.linalg.inv(A)
        poly = OC.QuinticPoly1D.solve(A_inv, constraints)

        return poly

    @staticmethod
    def extrapolate_solution_1d(fstates: FrenetTrajectories1D, length: int, override_values: FrenetState1D,
                                override_mask: FrenetState1D):
        """
        Given a partial 1D trajectory, this function appends to the end of it an extrapolation-block of specified length
        with values taken from the trajectory's last state (or values-overrides).
        :param fstates: the set of 1D trajectories to extrapolate
        :param length: length of extrapolation block (number of replicates)
        :param override_values: 1D frenet state vector (or NaN values whereas actual values are to be taken from the
        last point in the partial 1D trajectory)
        :param override_mask: mask vector for <override_values>. On cells where mask values == 1, override will apply.
        :return:
        """
        extrapolation_vector = np.logical_not(override_mask) * fstates[:, -1, :] + \
                               override_mask * np.repeat(override_values[np.newaxis, :], repeats=fstates.shape[0], axis=0)
        extrapolation_block = np.repeat(extrapolation_vector[:, np.newaxis, :], repeats=length, axis=1)

        return np.concatenate((fstates, extrapolation_block), axis=-2)

    @staticmethod
    def _solve_optimization(fconst_0: FrenetConstraints, fconst_t: FrenetConstraints, T_s: float, T_d_vals: np.ndarray,
                            dt: float) -> Tuple[FrenetTrajectories2D, np.ndarray, np.ndarray]:
        """
        Solves the two-point boundary value problem, given a set of constraints over the initial state
        and a set of constraints over the terminal state. The solution is a cartesian product of the solutions returned
        from solving two 1D problems (one for each Frenet dimension). The solutions for the latitudinal direction are
        aggregated along different Td possible values.When Td happens to be lower than Ts, we expand the latitudinal
        solution: the last position stays the same while the velocity and acceleration are set to zero.
        :param fconst_0: a set of constraints over the initial state
        :param fconst_t: a set of constraints over the terminal state
        :param T_s: longitudinal trajectory duration (sec.), relative to ego. Ts has to be a multiple of dt.
        :param T_d_vals: lateral trajectory possible durations (sec.), relative to ego. Higher bound is Ts.
        :param dt: [sec] basic time unit from constructor.
        :return: a tuple: (points-matrix of rows in the form [sx, sv, sa, dx, dv, da],
        poly-coefficients-matrix of rows in the form [c0_s, c1_s, ... c5_s, c0_d, ..., c5_d],
        array of the Td values associated with the polynomials)
        """

        time_samples_s = np.arange(dt, T_s + np.finfo(np.float16).eps, dt)

        # Define constraints
        constraints_s = NumpyUtils.cartesian_product_matrix_rows(fconst_0.get_grid_s(), fconst_t.get_grid_s())
        constraints_d = NumpyUtils.cartesian_product_matrix_rows(fconst_0.get_grid_d(), fconst_t.get_grid_d())

        # solve for dimension s
        poly_s = WerlingPlanner._get_werling_poly(constraints_s, T_s)

        # generate trajectories for the polynomials of dimension s
        solutions_s = OC.QuinticPoly1D.polyval_with_derivatives(poly_s, time_samples_s)

        # store a vector of time-horizons for solutions of dimension s
        horizons_s = np.repeat([T_s], len(constraints_s))

        # Iterate over different time-horizons for dimension d
        poly_d = np.empty(shape=(0, 6))
        solutions_d = np.empty(shape=(0, len(time_samples_s), 3))
        horizons_d = np.empty(shape=0)
        for T_d in T_d_vals:
            time_samples_d = np.arange(dt, T_d + np.finfo(np.float16).eps, dt)

            # solve for dimension d (with time-horizon T_d)
            partial_poly_d = WerlingPlanner._get_werling_poly(constraints_d, T_d)
            partial_solutions_d = OC.QuinticPoly1D.polyval_with_derivatives(partial_poly_d, time_samples_d)

            # Expand lateral solutions (dimension d) to the size of the longitudinal solutions (dimension s)
            # with its final positions replicated. NOTE: we assume that final (dim d) velocities and accelerations = 0 !
            full_solutions_d = WerlingPlanner.extrapolate_solution_1d(
                fstates=partial_solutions_d,
                length=time_samples_s.size - time_samples_d.size,
                override_values=np.zeros(3),
                override_mask=np.array([0, 1, 1])
            )

            # append polynomials, trajectories and time-horizons to the dimensions d buffers
            poly_d = np.vstack((poly_d, partial_poly_d))
            solutions_d = np.vstack((solutions_d, full_solutions_d))
            horizons_d = np.append(horizons_d, np.repeat(T_d, len(constraints_d)))

        # generate 2D trajectories by Cartesian product of {horizons, polynomials, and 1D trajectories}
        # of dimensions {s,d}
        horizons = NumpyUtils.cartesian_product_matrix_rows(horizons_s[:, np.newaxis], horizons_d[:, np.newaxis])
        polynoms = NumpyUtils.cartesian_product_matrix_rows(poly_s, poly_d)
        solutions = NumpyUtils.cartesian_product_matrix_rows(solutions_s, solutions_d)

        # slice the results according to the rule T_s >= T_d since we don't want to generate trajectories whose
        valid_traj_slice = horizons[:, FP_SX] >= horizons[:, FP_DX]

        return solutions[valid_traj_slice], polynoms[valid_traj_slice], horizons[valid_traj_slice, FP_DX]
