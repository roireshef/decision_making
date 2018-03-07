from logging import Logger
from typing import Tuple

import numpy as np

from decision_making.src.exceptions import NoValidTrajectoriesFound, CouldNotGenerateTrajectories
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, SX_STEPS, SV_OFFSET_MIN, SV_OFFSET_MAX, \
    SV_STEPS, DX_OFFSET_MIN, DX_OFFSET_MAX, DX_STEPS, SX_OFFSET_MIN, SX_OFFSET_MAX, \
    TD_STEPS, LAT_ACC_LIMITS, TD_MIN_DT, LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.cost_function import Costs
from decision_making.src.planning.trajectory.optimal_control.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.optimal_control.optimal_control_utils import QuinticPoly1D, Poly1D
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, FS_SV, \
    FS_SA, FS_SX, FS_DX, LIMIT_MIN, LIMIT_MAX, CartesianExtendedTrajectory, \
    CartesianTrajectories, FS_DV, FS_DA, CartesianExtendedState, FrenetState2D, C_A, C_K, FrenetState1D, \
    FrenetTrajectory1D, D5, Limits, C_Y
from decision_making.src.planning.types import FrenetTrajectories2D, CartesianExtendedTrajectories
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.prediction.predictor import Predictor
from decision_making.src.state.state import State


class SamplableWerlingTrajectory(SamplableTrajectory):
    def __init__(self, timestamp_in_sec: float, T_s: float, T_d: float, frenet_frame: FrenetSerret2DFrame,
                 poly_s_coefs: np.ndarray, poly_d_coefs: np.ndarray):
        """To represent a trajectory that is a result of Werling planner, we store the frenet frame used and
        two polynomial coefficients vectors (for dimensions s and d)"""
        super().__init__(timestamp_in_sec, T_s)
        self.T_d = T_d
        self.frenet_frame = frenet_frame
        self.poly_s_coefs = poly_s_coefs
        self.poly_d_coefs = poly_d_coefs

    @property
    def T_s(self):
        return self.T

    def sample(self, time_points: np.ndarray) -> CartesianExtendedTrajectory:
        """See base method for API. In this specific representation of the trajectory, we sample from s-axis polynomial
        (longitudinal) and partially (up to some time-horizon cached in self.lon_plan_horizon) from d-axis polynomial
        (lateral) and extrapolate the rest of the states in d-axis to conform to the trajectory's total duration"""

        relative_time_points = time_points - self.timestamp_in_sec

        # Make sure no unplanned extrapolation will occur due to overreaching time points
        # This check is done in relative-to-ego units
        assert max(relative_time_points) <= self.T_s

        # assign values from <time_points> in s-axis polynomial
        fstates_s = QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_s_coefs]), relative_time_points)[0]

        fstates_d = np.empty(shape=np.append(time_points.shape, 3))

        is_within_horizon_d = relative_time_points <= self.T_d

        fstates_d[is_within_horizon_d] = QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_d_coefs]),
                                                                                   relative_time_points[is_within_horizon_d])

        # Expand lateral solution to the size of the longitudinal solution with its final positions replicated
        # NOTE: we assume that velocity and accelerations = 0 !!
        end_of_horizon_state_d = QuinticPoly1D.polyval_with_derivatives(np.array([self.poly_d_coefs]),
                                                                           np.array([self.T_d]))[0]
        extrapolation_state_d = WerlingPlanner.repeat_1d_state(
            fstate=end_of_horizon_state_d,
            repeats=1,
            override_values=np.zeros(3),
            override_mask=np.array([0, 1, 1]))

        fstates_d[np.logical_not(is_within_horizon_d)] = extrapolation_state_d

        fstates = np.hstack((fstates_s, fstates_d))

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

    def plan(self, state: State, reference_route: np.ndarray, goal: CartesianExtendedState, time_horizon: float,
             cost_params: TrajectoryCostParams) -> \
            Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """ see base class """
        T_s = time_horizon

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

        sx_range = np.linspace(np.max((SX_OFFSET_MIN + goal_frenet_state[FS_SX],
                                       (goal_frenet_state[FS_SX] + ego_frenet_state[FS_SX]) / 2)),
                               np.min((SX_OFFSET_MAX + goal_frenet_state[FS_SX], (len(frenet.O) - 1) * frenet.ds)),
                               SX_STEPS)

        sv_range = np.linspace(
            np.max((SV_OFFSET_MIN + goal_frenet_state[FS_SV], cost_params.velocity_limits[LIMIT_MIN])),
            np.min((SV_OFFSET_MAX + goal_frenet_state[FS_SV], cost_params.velocity_limits[LIMIT_MAX])),
            SV_STEPS)

        dx_range = np.linspace(DX_OFFSET_MIN + goal_frenet_state[FS_DX],
                               DX_OFFSET_MAX + goal_frenet_state[FS_DX],
                               DX_STEPS)

        fconstraints_tT = FrenetConstraints(sx=sx_range, sv=sv_range, sa=goal_frenet_state[FS_SA],
                                            dx=dx_range, dv=goal_frenet_state[FS_DV], da=goal_frenet_state[FS_DA])

        assert T_s >= self.dt
        # Make sure T_s values are multiples of dt (or else the matrix, calculated using T_s, and the latitudinal
        #  time axis, lon_time_samples, won't fit).
        T_s = Math.round_to_step(T_s, self.dt)

        # planning is done on the time dimension relative to an anchor (currently the timestamp of the ego vehicle)
        # so time points are from t0 = 0 until some T (lon_plan_horizon)
        planning_time_points = np.arange(0, T_s + np.finfo(np.float16).eps, self.dt)

        # Latitudinal planning horizon(Td) lower bound, now approximated from x=a*t^2
        lower_bound_T_d = self._low_bound_lat_horizon(fconstraints_t0, fconstraints_tT, self.dt)

        assert T_s >= lower_bound_T_d

        self._logger.debug('WerlingPlanner is planning from %s (frenet) to %s (frenet) in %s seconds' %
                           (NumpyUtils.str_log(ego_frenet_state), NumpyUtils.str_log(goal_frenet_state),
                            T_s))

        # create a grid on T_d (lateral movement time-grid)
        T_d_grid = WerlingPlanner._create_lat_horizon_grid(T_s, lower_bound_T_d, self.dt)

        self._logger.debug("Lateral horizon grid considered is: {}".format(str(T_d_grid)))

        # solve problem in frenet-frame
        ftrajectories, poly_coefs, T_d_vals = WerlingPlanner._solve_optimization(fconstraints_t0, fconstraints_tT,
                                                                                 T_s, T_d_grid, self.dt)

        # filter resulting trajectories by progress on curve, velocity and (lateral) accelerations limits in frenet
        frenet_filtered_indices = self._filter_by_frenet_limits(ftrajectories, poly_coefs[:, D5:], T_d_vals,
                                                                cost_params, frenet.s_limits)

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories: CartesianExtendedTrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories[frenet_filtered_indices])

        # filter resulting trajectories by velocity and accelerations limits - this is now done in Cartesian frame
        # which takes into account the curvature of the road applied to trajectories planned in the Frenet frame
        cartesian_refiltered_indices = self._filter_by_cartesian_limits(ctrajectories, cost_params)

        refiltered_indices = frenet_filtered_indices[cartesian_refiltered_indices]

        ctrajectories_filtered = ctrajectories[cartesian_refiltered_indices]
        ftrajectories_refiltered = ftrajectories[frenet_filtered_indices][cartesian_refiltered_indices]

        self._logger.debug(LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, len(ctrajectories_filtered))

        if len(ctrajectories) == 0:
            raise CouldNotGenerateTrajectories("No valid cartesian trajectories. time: %f, goal: %s, state: %s. "
                                               "Longitudes range: [%s, %s] (limits: %s)"
                                               "Min frenet velocity: %s"
                                               "number of trajectories passed according to Frenet limits: %s/%s;" %
                                               (T_s, NumpyUtils.str_log(goal), str(state).replace('\n', ''),
                                                np.min(ftrajectories[:, :, FS_SX]), np.max(ftrajectories[:, :, FS_SX]),
                                                frenet.s_limits,
                                                np.min(ftrajectories[:, :, FS_SV]),
                                                len(frenet_filtered_indices), len(ftrajectories)))
        elif len(ctrajectories_filtered) == 0:
            lat_acc = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
            raise NoValidTrajectoriesFound("No valid trajectories found. time: %f, goal: %s, state: %s. "
                                           "planned velocities range [%s, %s] (limits: %s); "
                                           "planned lon. accelerations range [%s, %s] (limits: %s); "
                                           "planned lat. accelerations range [%s, %s] (limits: %s); "
                                           "number of trajectories passed according to Frenet limits: %s/%s;"
                                           "number of trajectories passed according to Cartesian limits: %s/%s;"
                                           "number of trajectories passed according to all limits: %s/%s;" %
                                           (T_s, NumpyUtils.str_log(goal), str(state).replace('\n', ''),
                                            np.min(ctrajectories[:, :, C_V]), np.max(ctrajectories[:, :, C_V]),
                                            NumpyUtils.str_log(cost_params.velocity_limits),
                                            np.min(ctrajectories[:, :, C_A]), np.max(ctrajectories[:, :, C_A]),
                                            NumpyUtils.str_log(cost_params.lon_acceleration_limits),
                                            np.min(lat_acc), np.max(lat_acc),
                                            NumpyUtils.str_log(cost_params.lat_acceleration_limits),
                                            len(frenet_filtered_indices), len(ftrajectories),
                                            len(cartesian_refiltered_indices), len(ctrajectories),
                                            len(refiltered_indices), len(ftrajectories)))

        # compute trajectory costs at sampled times
        global_time_sample = planning_time_points + state.ego_state.timestamp_in_sec
        filtered_trajectory_costs = \
            self._compute_cost(ctrajectories_filtered, ftrajectories_refiltered, state, goal_frenet_state, cost_params,
                               global_time_sample, self._predictor, self.dt)

        sorted_filtered_idxs = filtered_trajectory_costs.argsort()

        self._logger.debug("Chosen trajectory planned with lateral horizon : {}".format(T_d_vals[refiltered_indices[sorted_filtered_idxs[0]]]))

        samplable_trajectory = SamplableWerlingTrajectory(
            timestamp_in_sec=state.ego_state.timestamp_in_sec,
            T_s=T_s,
            T_d=T_d_vals[refiltered_indices[sorted_filtered_idxs[0]]],
            frenet_frame=frenet,
            poly_s_coefs=poly_coefs[refiltered_indices[sorted_filtered_idxs[0]]][:6],
            poly_d_coefs=poly_coefs[refiltered_indices[sorted_filtered_idxs[0]]][6:]
        )

        return samplable_trajectory, \
               ctrajectories_filtered[sorted_filtered_idxs, :, :(C_V + 1)], \
               filtered_trajectory_costs[sorted_filtered_idxs]

    @staticmethod
    def _filter_by_cartesian_limits(ctrajectories: CartesianExtendedTrajectories,
                                    cost_params: TrajectoryCostParams) -> np.ndarray:
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
    def _filter_by_frenet_limits(ftrajectories: FrenetTrajectories2D, poly_coefs_d: np.ndarray,
                                 T_d_vals: np.ndarray, cost_params: TrajectoryCostParams,
                                 reference_route_limits: Limits) -> np.ndarray:
        """
        Given a set of trajectories in Frenet coordinate-frame, it validates them against the following limits:
        (longitudinal progress on the frenet frame curve, positive longitudinal velocity, acceleration of the lateral
        polynomial used in planning is in the allowed limits)
        :param ftrajectories: FrenetTrajectories2D object of trajectories to validate
        :param poly_coefs_d: 2D numpy array of each row has 6 poly coefficients of lateral polynomial
        :param T_d_vals: 1D numpy array with lateral planning-horizons (correspond to each trajectory)
        :param cost_params: TrajectoryCostParams object that holds desired limits (for validation)
        :param reference_route_limits: the minimal and maximal progress (s value) on the reference route used
        in the frenet frame used for planning
        :return: Indices along the 1st dimension in <ctrajectories> (trajectory index) for valid trajectories
        """
        # validate the progress on the reference-route curve doesn't extrapolate, and that velocity is non-negative
        conforms = np.all(
            NumpyUtils.is_in_limits(ftrajectories[:, :, FS_SX], reference_route_limits) &
            np.greater_equal(ftrajectories[:, :, FS_SV], 0.0), axis=1)

        # here we validate feasible lateral acceleration *directly from the lateral polynomial* because our
        # discretization of the trajectory (via sampling with constant self.dt) can overlook cases where there is a high
        # lateral acceleration between two adjacent sampled points (critical in the lateral case because we allow
        # shorter lateral maneuvers
        frenet_lateral_movement_is_feasible = \
            QuinticPoly1D.are_accelerations_in_limits(poly_coefs_d, T_d_vals, cost_params.lat_acceleration_limits)

        return np.argwhere(np.logical_and(conforms, frenet_lateral_movement_is_feasible)).flatten()

    @staticmethod
    def _compute_cost(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D, state: State,
                      goal_in_frenet: FrenetState2D, params: TrajectoryCostParams, global_time_samples: np.ndarray,
                      predictor: Predictor, dt: float) -> np.ndarray:
        """
        Takes trajectories (in both frenet-frame repr. and cartesian-frame repr.) and computes a cost for each one
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param goal_in_frenet: A 1D numpy array of the desired ego-state to plan towards, represented in current
                global-coordinate-frame (see EGO_* in planning.utils.types.py for the fields)
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor: predictor instance to use to compute future localizations for DyanmicObjects
        :param dt: time step of ctrajectories
        :return: numpy array (1D) of the total cost per trajectory (in ctrajectories and ftrajectories)
        """
        ''' deviation from goal cost '''
        last_fpoints = ftrajectories[:, -1, :]
        trajectory_end_goal_diff = np.array([last_fpoints[:, FS_SX] - goal_in_frenet[FS_SX],
                                             last_fpoints[:, FS_DX] - goal_in_frenet[FS_DX]])
        trajectory_end_goal_dist = np.sqrt(trajectory_end_goal_diff[0] ** 2 +
                                           (params.dist_from_goal_lat_factor * trajectory_end_goal_diff[1]) ** 2)
        dist_from_goal_costs = Math.clipped_sigmoid(trajectory_end_goal_dist - params.dist_from_goal_cost.offset,
                                                    params.dist_from_goal_cost.w, params.dist_from_goal_cost.k)

        ''' point-wise costs: obstacles, deviations, jerk '''
        pointwise_costs = Costs.compute_pointwise_costs(ctrajectories, ftrajectories, state, params,
                                                                  global_time_samples, predictor, dt)

        return np.sum(pointwise_costs, axis=(1, 2)) + dist_from_goal_costs

    # TODO: determine tighter lower bound according to physical constraints and ego control limitations
    def _low_bound_lat_horizon(self, fconstraints_t0: FrenetConstraints, fconstraints_tT: FrenetConstraints,
                               dt) -> float:
        """
        Calculates the lower bound for the lateral time horizon based on the physical constraints.
        :param fconstraints_t0: a set of constraints over the initial state
        :param fconstraints_tT: a set of constraints over the terminal state
        :param dt: [sec] basic time unit from constructor
        :return: Low bound for lateral time horizon.
        """
        min_lat_movement = np.min(np.abs(fconstraints_tT.get_grid_d()[:, 0] - fconstraints_t0.get_grid_d()[0, 0]))
        low_bound_lat_plan_horizon = max(np.sqrt((2*min_lat_movement) / LAT_ACC_LIMITS[LIMIT_MAX]), dt)
        return max(low_bound_lat_plan_horizon, TD_MIN_DT * self.dt)

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
    def _solve_1d_poly(constraints: np.ndarray, T: float, poly_impl: Poly1D) -> np.ndarray:
        """
        Solves the two-point boundary value problem, given a set of constraints over the initial and terminal states.
        :param constraints: 3D numpy array of a set of constraints over the initial and terminal states
        :param T: longitudinal/lateral trajectory duration (sec.), relative to ego. T has to be a multiple of WerlingPlanner.dt
        :param poly_impl: OptimalControlUtils 1d polynomial implementation class
        :return: a poly-coefficients-matrix of rows in the form [c0_s, c1_s, ... c5_s, c0_d, ..., c5_d]
        """
        A = poly_impl.time_constraints_matrix(T)
        A_inv = np.linalg.inv(A)
        poly_coefs = poly_impl.solve(A_inv, constraints)
        return poly_coefs

    @staticmethod
    def repeat_1d_state(fstate: FrenetState1D, repeats: int,
                        override_values: FrenetState1D, override_mask: FrenetState1D):
        """please see documentation in used method"""
        return WerlingPlanner.repeat_1d_states(fstate[np.newaxis, :], repeats, override_values, override_mask)[0]

    @staticmethod
    def repeat_1d_states(fstates: FrenetTrajectory1D, repeats: int,
                         override_values: FrenetState1D, override_mask: FrenetState1D):
        """
        Given an array of 1D-frenet-states [x, x_dot, x_dotdot], this function builds a block of replicates of each one
        of them <repeats> times while giving the option for overriding values (part or all) their values.
        :param fstates: the set of 1D-frenet-states to repeat
        :param repeats: length of repeated block (number of replicates for each state)
        :param override_values: 1D-frenet-state vector of values to override while repeating every state in <fstates>
        :param override_mask: mask vector for <override_values>. Where mask values == 1, override will apply, whereas
        mask value of 0 will incur no value override
        :return:
        """
        repeating_slice = np.logical_not(override_mask) * fstates + \
                          override_mask * np.repeat(override_values[np.newaxis, :], repeats=fstates.shape[0], axis=0)
        repeated_block = np.repeat(repeating_slice[:, np.newaxis, :], repeats=repeats, axis=1)
        return repeated_block

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
        time_samples_s = np.arange(0, T_s + np.finfo(np.float16).eps, dt)

        # Define constraints
        constraints_s = NumpyUtils.cartesian_product_matrix_rows(fconst_0.get_grid_s(), fconst_t.get_grid_s())
        constraints_d = NumpyUtils.cartesian_product_matrix_rows(fconst_0.get_grid_d(), fconst_t.get_grid_d())

        # solve for dimension s
        poly_s = WerlingPlanner._solve_1d_poly(constraints_s, T_s, QuinticPoly1D)

        # generate trajectories for the polynomials of dimension s
        solutions_s = QuinticPoly1D.polyval_with_derivatives(poly_s, time_samples_s)

        # store a vector of time-horizons for solutions of dimension s
        horizons_s = np.repeat([T_s], len(constraints_s))

        # Iterate over different time-horizons for dimension d
        poly_d = np.empty(shape=(0, 6))
        solutions_d = np.empty(shape=(0, len(time_samples_s), 3))
        horizons_d = np.empty(shape=0)
        for T_d in T_d_vals:
            time_samples_d = np.arange(dt, T_d + np.finfo(np.float16).eps, dt)

            # solve for dimension d (with time-horizon T_d)
            partial_poly_d = WerlingPlanner._solve_1d_poly(constraints_d, T_d, QuinticPoly1D)

            # generate the trajectories for the polynomials of dimension d - within the horizon T_d
            partial_solutions_d = QuinticPoly1D.polyval_with_derivatives(partial_poly_d, time_samples_d)

            # Expand lateral solutions (dimension d) to the size of the longitudinal solutions (dimension s)
            # with its final positions replicated. NOTE: we assume that final (dim d) velocities and accelerations = 0 !
            solutions_extrapolation_d = WerlingPlanner.repeat_1d_states(
                fstates=partial_solutions_d[:, -1, :],
                repeats=time_samples_s.size - time_samples_d.size,
                override_values=np.zeros(3),
                override_mask=np.array([0, 1, 1])
            )
            full_horizon_solutions_d = np.concatenate((partial_solutions_d, solutions_extrapolation_d), axis=-2)

            # append polynomials, trajectories and time-horizons to the dimensions d buffers
            poly_d = np.vstack((poly_d, partial_poly_d))
            solutions_d = np.vstack((solutions_d, full_horizon_solutions_d))
            horizons_d = np.append(horizons_d, np.repeat(T_d, len(constraints_d)))

        # generate 2D trajectories by Cartesian product of {horizons, polynomials, and 1D trajectories}
        # of dimensions {s,d}
        horizons = NumpyUtils.cartesian_product_matrix_rows(horizons_s[:, np.newaxis], horizons_d[:, np.newaxis])
        polynoms = NumpyUtils.cartesian_product_matrix_rows(poly_s, poly_d)
        solutions = NumpyUtils.cartesian_product_matrix_rows(solutions_s, solutions_d)

        # slice the results according to the rule T_s >= T_d since we don't want to generate trajectories whose
        valid_traj_slice = horizons[:, FP_SX] >= horizons[:, FP_DX]

        return solutions[valid_traj_slice], polynoms[valid_traj_slice], horizons[valid_traj_slice, FP_DX]
