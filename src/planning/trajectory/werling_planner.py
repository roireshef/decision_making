from logging import Logger
from typing import Tuple

import numpy as np
import rte.python.profiler as prof
from decision_making.src.exceptions import CartesianLimitsViolated
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, SX_STEPS, SV_OFFSET_MIN, SV_OFFSET_MAX, \
    SV_STEPS, DX_OFFSET_MIN, DX_OFFSET_MAX, DX_STEPS, SX_OFFSET_MIN, SX_OFFSET_MAX, \
    TD_STEPS, LAT_ACC_LIMITS, TD_MIN_DT, LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, EPS, \
    CLOSE_TO_ZERO_NEGATIVE_VELOCITY
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.cost_function import TrajectoryPlannerCosts
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.werling_utils import WerlingUtils
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, FS_SV, \
    FS_SA, FS_SX, FS_DX, LIMIT_MIN, LIMIT_MAX, CartesianTrajectories, FS_DV, FS_DA, CartesianExtendedState, \
    FrenetState2D, C_A, C_K
from decision_making.src.planning.types import FrenetTrajectories2D, CartesianExtendedTrajectories
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.planning.behavioral.state import State


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, dt=WERLING_TIME_RESOLUTION):
        super().__init__(logger, predictor)
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    @prof.ProfileFunction()
    def plan(self, state: State, reference_route: FrenetSerret2DFrame, goal: CartesianExtendedState,
             T_target_horizon: float, T_trajectory_end_horizon: float, cost_params: TrajectoryCostParams) -> Tuple[
                SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """ see base class """

        # The reference_route, the goal, ego and the dynamic objects are given in the global coordinate-frame.
        # The vehicle doesn't need to lay parallel to the road.

        ego_frenet_state: FrenetState2D = reference_route.cstate_to_fstate(state.ego_state.cartesian_state)

        # define constraints for the initial state
        fconstraints_t0 = FrenetConstraints.from_state(ego_frenet_state)

        # define constraints for the terminal (goal) state
        goal_frenet_state: FrenetState2D = reference_route.cstate_to_fstate(goal)

        sx_range = np.linspace(np.max((SX_OFFSET_MIN + goal_frenet_state[FS_SX],
                                       (goal_frenet_state[FS_SX] + ego_frenet_state[FS_SX]) / 2)),
                               np.min((SX_OFFSET_MAX + goal_frenet_state[FS_SX], reference_route.s_max)),
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

        planning_horizon = max(T_trajectory_end_horizon, T_target_horizon)

        assert planning_horizon >= self.dt + EPS, 'planning_horizon (=%f) is too short and is less than one trajectory' \
                                                  ' timestamp (=%f)' % (planning_horizon, self.dt)

        self._logger.debug(
            'WerlingPlanner is planning from %s (frenet) to %s (frenet) in %s seconds and extrapolating to %s seconds',
            NumpyUtils.str_log(ego_frenet_state), NumpyUtils.str_log(goal_frenet_state),
            T_target_horizon, planning_horizon)

        is_target_ahead = T_target_horizon > self.dt and goal_frenet_state[FS_SX] > ego_frenet_state[FS_SX]

        # calculate frenet state in ego time, such that its prediction in goal time is goal_frenet_state
        # it is used only when not is_target_ahead
        ego_by_goal_state = KinematicUtils.create_ego_by_goal_state(goal_frenet_state, T_target_horizon)

        # solve the optimization problem in frenet-frame from t=0 to t=T
        # Actual trajectory planning is needed because T_s > 0.1 and the target is ahead of us
        if is_target_ahead:
            # Lateral planning horizon(Td) lower bound, now approximated from x=a*t^2
            lower_bound_T_d = self._low_bound_lat_horizon(fconstraints_t0, fconstraints_tT, T_target_horizon, self.dt)

            # create a grid on T_d (lateral movement time-grid)
            T_d_grid = WerlingPlanner._create_lat_horizon_grid(T_target_horizon, lower_bound_T_d)

            # solve problem in frenet-frame
            ftrajectories_optimization, poly_coefs, T_d_vals = WerlingPlanner._solve_optimization(fconstraints_t0,
                                                                                                  fconstraints_tT,
                                                                                                  T_target_horizon, T_d_grid, self.dt)
            ftrajectories = WerlingPlanner._correct_velocity_values(ftrajectories_optimization)

            # trajectory was planned up to a certain time, the rest should be padded with constant
            # velocity prediction
            if planning_horizon > T_target_horizon:        # add padding
                terminal_d = np.repeat(fconstraints_tT.get_grid_d(), len(T_d_grid), axis=0)
                terminal_s = fconstraints_tT.get_grid_s()
                terminal_states = NumpyUtils.cartesian_product_matrix_rows(terminal_s, terminal_d)

                time_samples = np.arange(Math.ceil_to_step(T_target_horizon, self.dt) - T_target_horizon, planning_horizon - T_target_horizon + EPS, self.dt)
                extrapolated_fstates_s = self.predictor.predict_2d_frenet_states(terminal_states, time_samples)
                ftrajectories = np.hstack((ftrajectories, extrapolated_fstates_s))
        else:
            # only pad
            ftrajectories = self.predictor.predict_2d_frenet_states(ego_by_goal_state[np.newaxis, :],
                                                                    np.arange(0, planning_horizon + EPS, self.dt))
            ftrajectories = WerlingPlanner._correct_velocity_values(ftrajectories)

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories: CartesianExtendedTrajectories = reference_route.ftrajectories_to_ctrajectories(ftrajectories)

        # TODO: desired velocity is dynamically changing when transitioning between road/lane segments
        # filter resulting trajectories by velocity and accelerations limits - this is now done in Cartesian frame
        # which takes into account the curvature of the road applied to trajectories planned in the Frenet frame
        cartesian_filter_results = KinematicUtils.filter_by_cartesian_limits(ctrajectories, cost_params.velocity_limits,
                                                                             cost_params.lon_acceleration_limits,
                                                                             cost_params.lat_acceleration_limits)

        cartesian_filtered_indices = np.argwhere(cartesian_filter_results).flatten()

        ctrajectories_filtered = ctrajectories[cartesian_filtered_indices]

        self._logger.debug(LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, len(ctrajectories_filtered))

        if len(ctrajectories_filtered) == 0:
            WerlingPlanner._raise_error(state, ftrajectories, ctrajectories, reference_route, T_target_horizon,
                                        planning_horizon, goal, ego_frenet_state, goal_frenet_state, cost_params)

        # planning is done on the time dimension relative to an anchor (currently the timestamp of the ego vehicle)
        # so time points are from t0 = 0 until some T (lon_plan_horizon)
        total_planning_time_points = np.arange(0, planning_horizon + EPS, self.dt)

        # compute trajectory costs at sampled times
        global_time_samples = total_planning_time_points + state.ego_state.timestamp_in_sec
        filtered_trajectory_costs = \
            self._compute_cost(ctrajectories_filtered, ftrajectories[cartesian_filtered_indices], state,
                               goal_frenet_state, cost_params,
                               global_time_samples, self._predictor, self.dt, reference_route)

        sorted_filtered_idxs = filtered_trajectory_costs.argsort()

        if is_target_ahead:  # Actual werling planning has occurred because T_s > 0.1 and the target is ahead of us
            # TODO: what if future sampling from poly_s will result with negative velocity (uncorrected for negative velocity)?
            samplable_trajectory = SamplableWerlingTrajectory(
                timestamp_in_sec=state.ego_state.timestamp_in_sec,
                T_s=T_target_horizon,
                T_d=T_d_vals[cartesian_filtered_indices[sorted_filtered_idxs[0]]],
                frenet_frame=reference_route,
                poly_s_coefs=poly_coefs[cartesian_filtered_indices[sorted_filtered_idxs[0]]][:6],
                poly_d_coefs=poly_coefs[cartesian_filtered_indices[sorted_filtered_idxs[0]]][6:],
                T_extended=planning_horizon
            )
        else:  # Publish a fixed trajectory, containing just padding
            poly_s, poly_d = KinematicUtils.create_linear_profile_polynomial_pair(ego_by_goal_state)
            samplable_trajectory = SamplableWerlingTrajectory(state.ego_state.timestamp_in_sec,
                                                              planning_horizon, planning_horizon, planning_horizon,
                                                              reference_route, poly_s, poly_d)
        return samplable_trajectory, \
           ctrajectories_filtered[sorted_filtered_idxs, :, :(C_V + 1)], \
           filtered_trajectory_costs[sorted_filtered_idxs]

    @staticmethod
    def _correct_velocity_values(ftrajectories: FrenetTrajectories2D) -> FrenetTrajectories2D:
        """
        Velocity values of Werling trajectories can be received with minor numerical deviations.
        This method verifies that if such deviations exist, they are indeed minor, and corrects them
        to the right accurate values.
        :param ftrajectories: trajectories in frenet frame
        :return:Corrected trajectories in frenet frame
        """
        traj_velocities = ftrajectories[:, :, FS_SV]
        is_velocities_close_to_zero = np.logical_and(traj_velocities > CLOSE_TO_ZERO_NEGATIVE_VELOCITY,
                                                     traj_velocities < 0)
        ftrajectories[is_velocities_close_to_zero, FS_SV] = 0.0
        return ftrajectories

    @staticmethod
    def _compute_cost(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D, state: State,
                      goal_in_frenet: FrenetState2D, params: TrajectoryCostParams, global_time_samples: np.ndarray,
                      predictor: EgoAwarePredictor, dt: float, reference_route: FrenetSerret2DFrame) -> np.ndarray:
        """
        Takes trajectories (in both frenet-frame repr. and cartesian-frame repr.) and computes a cost for each one
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param goal_in_frenet: A 1D numpy array of the desired ego-state to plan towards, represented in current
                global-coordinate-frame (see EGO_* in planning.utils.types.py for the fields)
        :param params: parameters for the cost function (from behavioral layer)
        :param global_time_samples: [sec] time samples for prediction (global, not relative)
        :param predictor: predictor instance to use to compute future localizations for DynamicObjects
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
        pointwise_costs = TrajectoryPlannerCosts.compute_pointwise_costs(ctrajectories, ftrajectories, state, params,
                                                                         global_time_samples, predictor, dt,
                                                                         reference_route)

        return np.sum(pointwise_costs, axis=(1, 2)) + dist_from_goal_costs

    # TODO: determine tighter lower bound according to physical constraints and ego control limitations
    def _low_bound_lat_horizon(self, fconstraints_t0: FrenetConstraints, fconstraints_tT: FrenetConstraints,
                               T_s: float, dt: float) -> float:
        """
        Calculates the lower bound for the lateral time horizon based on the physical constraints.
        :param fconstraints_t0: a set of constraints over the initial state
        :param fconstraints_tT: a set of constraints over the terminal state
        :param T_s: longitudinal action time horizon
        :param dt: [sec] basic time unit from constructor
        :return: Low bound for lateral time horizon.
        """
        min_lat_movement = np.min(np.abs(fconstraints_tT.get_grid_d()[:, 0] - fconstraints_t0.get_grid_d()[0, 0]))
        low_bound_lat_plan_horizon = max(np.sqrt((2 * min_lat_movement) / LAT_ACC_LIMITS[LIMIT_MAX]), dt)
        return min(max(low_bound_lat_plan_horizon, TD_MIN_DT * self.dt), T_s)

    @staticmethod
    def _create_lat_horizon_grid(T_s: float, T_d_low_bound: float) -> np.ndarray:
        """
        Receives the lower bound of the lateral time horizon T_d_low_bound and the longitudinal time horizon T_s
        and returns a grid of possible lateral planning time values.
        :param T_s: longitudinal trajectory duration (sec.), relative to ego.
        :param T_d_low_bound: lower bound on lateral trajectory duration (sec.), relative to ego. Higher bound is Ts.
        :return: numpy array (1D) of the possible lateral planning horizons
        """
        return np.flip(np.linspace(T_s, T_d_low_bound, TD_STEPS), axis=0)



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
        time_samples_s = np.arange(0, T_s + EPS, dt)

        # Define constraints
        constraints_s = NumpyUtils.cartesian_product_matrix_rows(fconst_0.get_grid_s(), fconst_t.get_grid_s())
        constraints_d = NumpyUtils.cartesian_product_matrix_rows(fconst_0.get_grid_d(), fconst_t.get_grid_d())

        # solve for dimension s
        poly_s = QuinticPoly1D.solve_1d_bvp(constraints_s, T_s)

        # generate trajectories for the polynomials of dimension s
        solutions_s = QuinticPoly1D.polyval_with_derivatives(poly_s, time_samples_s)

        # store a vector of time-horizons for solutions of dimension s
        horizons_s = np.repeat([T_s], len(constraints_s))

        # Iterate over different time-horizons for dimension d
        poly_d = np.empty(shape=(0, 6))
        solutions_d = np.empty(shape=(0, len(time_samples_s), 3))
        horizons_d = np.empty(shape=0)
        for T_d in T_d_vals:
            time_samples_d = np.arange(0, T_d + EPS, dt)

            # solve for dimension d (with time-horizon T_d)
            partial_poly_d = QuinticPoly1D.solve_1d_bvp(constraints_d, T_d)

            # generate the trajectories for the polynomials of dimension d - within the horizon T_d
            partial_solutions_d = QuinticPoly1D.polyval_with_derivatives(partial_poly_d, time_samples_d)

            # Expand lateral solutions (dimension d) to the size of the longitudinal solutions (dimension s)
            # with its final positions replicated. NOTE: we assume that final (dim d) velocities and accelerations = 0 !
            solutions_extrapolation_d = WerlingUtils.repeat_1d_states(
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

    @staticmethod
    def _raise_error(state: State, ftrajectories: FrenetTrajectories2D,ctrajectories: CartesianExtendedTrajectories,
                     reference_route: FrenetSerret2DFrame, T_target_horizon: float, planning_horizon: float,
                     goal: CartesianExtendedState, ego_frenet_state: FrenetState2D, goal_frenet_state: FrenetState2D,
                     cost_params: TrajectoryCostParams) -> None:
        """
        Raise error and print error message, when all trajectories were filtered in Werling.
        See the parameters description in function plan().
        :param state:
        :param ftrajectories:
        :param ctrajectories:
        :param reference_route:
        :param T_target_horizon:
        :param planning_horizon:
        :param goal:
        :param ego_frenet_state:
        :param goal_frenet_state:
        :param cost_params:
        :return:
        """
        np.set_printoptions(suppress=True)
        lat_acc = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
        lat_acc[ctrajectories[:, :, C_V] == 0] = 0
        # find the "best" trajectory, whose maximal lateral acceleration among all timestamps is minimal
        lat_acc_traj_idx = np.argmin(np.max(np.abs(lat_acc), axis=1))
        # find the "worst" timestamp in the "best" trajectory, for which the lateral acceleration is maximal
        lat_acc_t_idx = np.argmax(np.abs(lat_acc[lat_acc_traj_idx]))
        lat_acc_v = ctrajectories[lat_acc_traj_idx, lat_acc_t_idx, C_V]
        lat_acc_k = ctrajectories[lat_acc_traj_idx, lat_acc_t_idx, C_K]
        init_idx = final_idx = 0
        if not NumpyUtils.is_in_limits(lat_acc, cost_params.lat_acceleration_limits).all(axis=1).any():
            # if all trajectories violate lat_acc limits, get range of nominal points around the "worst" timestamp
            lat_acc_traj_s = ftrajectories[lat_acc_traj_idx, lat_acc_t_idx, FS_SX]
            nominal_idxs = reference_route.get_closest_index_on_frame(np.array([lat_acc_traj_s]))[0]
            nominal_points_around_worst_point = 11  # how many nominal points to print around the worst point
            if len(nominal_idxs) > 0:  # get a range of at most 11 nominal points near the problematic trajectory point
                init_idx = max(0, nominal_idxs[0] - nominal_points_around_worst_point//2)
                final_idx = min(reference_route.k.shape[0] - 1, nominal_idxs[0] + nominal_points_around_worst_point//2)
        raise CartesianLimitsViolated("No valid trajectories. "
                                      "timestamp_in_sec: %f, time horizon: %f, "
                                      "extrapolated time horizon: %f\ngoal: %s\nstate: %s.\n"
                                      "[highest minimal velocity, lowest maximal velocity] [%s, %s] (limits: %s)\n"
                                      "[highest minimal lon_acc, lowest maximal lon_acc] [%s, %s] (limits: %s)\n"
                                      "[highest minimal lat_acc, lowest maximal lat_acc] [%s, %s] (limits: %s)\n"
                                      "original trajectories #: %s\nego_frenet = %s\ngoal_frenet = %s\n"
                                      "distance from ego to goal = %f, time*approx_velocity = %f\n"
                                      "worst_lat_acc: t=%.1f v=%.3f k=%f; nominal_points.k=%s" %
                                      (state.ego_state.timestamp_in_sec, T_target_horizon, planning_horizon,
                                       NumpyUtils.str_log(goal), str(state).replace('\n', ''),
                                       np.max(np.min(ctrajectories[:, :, C_V], axis=1)),
                                       np.min(np.max(ctrajectories[:, :, C_V], axis=1)),
                                       NumpyUtils.str_log(cost_params.velocity_limits),
                                       np.max(np.min(ctrajectories[:, :, C_A], axis=1)),
                                       np.min(np.max(ctrajectories[:, :, C_A], axis=1)),
                                       NumpyUtils.str_log(cost_params.lon_acceleration_limits),
                                       np.max(np.min(lat_acc, axis=1)), np.min(np.max(lat_acc, axis=1)),
                                       NumpyUtils.str_log(cost_params.lat_acceleration_limits), len(ctrajectories),
                                       NumpyUtils.str_log(ego_frenet_state), NumpyUtils.str_log(goal_frenet_state),
                                       goal_frenet_state[FS_SX] - ego_frenet_state[FS_SX],
                                       T_target_horizon * (ego_frenet_state[FS_SV] + goal_frenet_state[FS_SV]) * 0.5,
                                       lat_acc_t_idx * WERLING_TIME_RESOLUTION, lat_acc_v, lat_acc_k,
                                       reference_route.k[init_idx:final_idx + 1, 0]))
