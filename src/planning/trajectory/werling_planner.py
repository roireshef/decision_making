from decision_making.src.planning.utils.safety_utils import SafetyUtils
from logging import Logger
from typing import Tuple, List, Dict

import numpy as np
import rte.python.profiler as prof

from decision_making.src.exceptions import CartesianLimitsViolated, NoSafeTrajectoriesFound
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, SX_STEPS, SV_OFFSET_MIN, SV_OFFSET_MAX, \
    SV_STEPS, DX_OFFSET_MIN, DX_OFFSET_MAX, DX_STEPS, SX_OFFSET_MIN, SX_OFFSET_MAX, \
    TD_STEPS, LAT_ACC_LIMITS, TD_MIN_DT, LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, EPS, TRAJECTORY_NUM_POINTS
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
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State, ObjectSize, DynamicObject


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, dt=WERLING_TIME_RESOLUTION):
        super().__init__(logger, predictor)
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    @prof.ProfileFunction()
    def plan(self, state: State, reference_route: FrenetSerret2DFrame, projected_obj_fstates: Dict[int, FrenetState2D],
             goal: CartesianExtendedState, T_target_horizon: float, T_trajectory_end_horizon: float,
             cost_params: TrajectoryCostParams) -> Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """ see base class """

        # The reference_route, the goal, ego and the dynamic objects are given in the global coordinate-frame.
        # The vehicle doesn't need to lay parallel to the road.

        ego_frenet_state: FrenetState2D = projected_obj_fstates[state.ego_state.obj_id]

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
            ftrajectories = self.predictor.predict_2d_frenet_states(ego_frenet_state[np.newaxis, :],
                                                                    np.arange(0, planning_horizon + EPS, self.dt))
            ftrajectories = WerlingPlanner._correct_velocity_values(ftrajectories)

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories: CartesianExtendedTrajectories = reference_route.ftrajectories_to_ctrajectories(ftrajectories)

        # filter resulting trajectories by velocity and accelerations limits - this is now done in Cartesian frame
        # which takes into account the curvature of the road applied to trajectories planned in the Frenet frame
        cartesian_filter_results = KinematicUtils.filter_by_cartesian_limits(ctrajectories, cost_params.velocity_limits,
                                                                             cost_params.lon_acceleration_limits,
                                                                             cost_params.lat_acceleration_limits)
        filtered_indices = np.argwhere(cartesian_filter_results).flatten()

        ctrajectories_filtered = ctrajectories[filtered_indices]
        ftrajectories_filtered = ftrajectories[filtered_indices]

        self._logger.debug(LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, len(ctrajectories_filtered))

        if len(ctrajectories_filtered) == 0:
            lat_acc = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
            lat_acc[ctrajectories[:, :, C_V] == 0] = 0
            raise CartesianLimitsViolated("No valid trajectories. "
                                          "timestamp_in_sec: %f, time horizon: %f, "
                                          "extrapolated time horizon: %f. goal: %s, state: %s.\n"
                                          "[highest minimal velocity, lowest maximal velocity] [%s, %s] (limits: %s); "
                                          "[highest minimal lon_acc, lowest maximal lon_acc] [%s, %s] (limits: %s); "
                                          "planned lat. accelerations range [%s, %s] (limits: %s); "
                                          "number of trajectories passed according to Cartesian limits: %s/%s;"
                                          "goal_frenet = %s; distance from ego to goal = %f, time*approx_velocity = %f" %
                                          (state.ego_state.timestamp_in_sec, T_target_horizon, planning_horizon,
                                           NumpyUtils.str_log(goal), str(state).replace('\n', ''),
                                           np.max(np.min(ctrajectories[:, :, C_V], axis=1)),
                                           np.min(np.max(ctrajectories[:, :, C_V], axis=1)),
                                           NumpyUtils.str_log(cost_params.velocity_limits),
                                           np.max(np.min(ctrajectories[:, :, C_A], axis=1)),
                                           np.min(np.max(ctrajectories[:, :, C_A], axis=1)),
                                           NumpyUtils.str_log(cost_params.lon_acceleration_limits),
                                           np.min(lat_acc), np.max(lat_acc),
                                           NumpyUtils.str_log(cost_params.lat_acceleration_limits),
                                           len(filtered_indices), len(ctrajectories),
                                           goal_frenet_state, goal_frenet_state[FS_SX] - ego_frenet_state[FS_SX],
                                           planning_horizon * (
                                                   ego_frenet_state[FS_SV] + goal_frenet_state[FS_SV]) * 0.5))

        # planning is done on the time dimension relative to an anchor (currently the timestamp of the ego vehicle)
        # so time points are from t0 = 0 until some T (lon_plan_horizon)
        total_planning_time_points = np.arange(0, planning_horizon + EPS, self.dt)

        # filter trajectories by RSS safety
        safe_traj_indices, safe_distances = \
            self.filter_trajectories_by_safety(state, projected_obj_fstates, total_planning_time_points, ftrajectories_filtered)
        # Throw an error if no safe trajectory is found
        if len(safe_traj_indices) == 0:
            raise NoSafeTrajectoriesFound("No safe trajectories found\ntime: %f, goal_frenet: %s\nstate: %s\n"
                                          "ego_fstate: %s\nobjects_fstates: %s\nsafe times:\n%s" %
                                          (T_target_horizon, NumpyUtils.str_log(goal_frenet_state),
                                           str(state).replace('\n', ''),
                                           NumpyUtils.str_log(ftrajectories[0, 0, :]),
                                           NumpyUtils.str_log(np.array(list(projected_obj_fstates.values()))),
                                           (safe_distances[..., 0] > 0).astype(int)))

        ftrajectories_filtered_safe = ftrajectories_filtered[safe_traj_indices]
        ctrajectories_filtered_safe = ctrajectories_filtered[safe_traj_indices]
        filtered_indices_safe = filtered_indices[safe_traj_indices]
        safe_distances_safe = safe_distances[safe_traj_indices]

        # compute trajectory costs at sampled times
        global_time_samples = total_planning_time_points + state.ego_state.timestamp_in_sec
        filtered_trajectory_costs = \
            self._compute_cost(ctrajectories_filtered_safe, ftrajectories_filtered_safe, state,
                               projected_obj_fstates, goal_frenet_state, safe_distances_safe, cost_params,
                               global_time_samples, self._predictor, self.dt, reference_route)

        sorted_filtered_idxs = filtered_trajectory_costs.argsort()

        if is_target_ahead:  # Actual werling planning has occurred because T_s > 0.1 and the target is ahead of us
            # TODO: what if future sampling from poly_s will result with negative velocity (uncorrected for negative velocity)?
            samplable_trajectory = SamplableWerlingTrajectory(
                timestamp_in_sec=state.ego_state.timestamp_in_sec,
                T_s=T_target_horizon,
                T_d=T_d_vals[filtered_indices_safe[sorted_filtered_idxs[0]]],
                frenet_frame=reference_route,
                poly_s_coefs=poly_coefs[filtered_indices_safe[sorted_filtered_idxs[0]]][:6],
                poly_d_coefs=poly_coefs[filtered_indices_safe[sorted_filtered_idxs[0]]][6:],
                T_extended=planning_horizon
            )
        else:  # Publish a fixed trajectory, containing just padding
            poly_s, poly_d = WerlingPlanner._create_linear_profile_polynomials(ego_frenet_state)
            samplable_trajectory = SamplableWerlingTrajectory(state.ego_state.timestamp_in_sec,
                                                              planning_horizon, planning_horizon, planning_horizon,
                                                              reference_route, poly_s, poly_d)
        return samplable_trajectory, \
           ctrajectories_filtered[sorted_filtered_idxs, :, :(C_V + 1)], \
           filtered_trajectory_costs[sorted_filtered_idxs]

    @staticmethod
    def _create_linear_profile_polynomials(frenet_state: FrenetState2D) -> (np.ndarray, np.ndarray):
        """
        Given a frenet state, create two (s, d) polynomials that assume constant velocity (we keep the same momentary
        velocity). Those polynomials are degenerate to s(t)=v*t+x form
        :param frenet_state: the current frenet state to pull positions and velocities from
        :return: a tuple of (s(t), d(t)) polynomial coefficient arrays
        """
        poly_s = np.array([0, 0, 0, 0, frenet_state[FS_SV], frenet_state[FS_SX]])
        poly_d = np.array([0, 0, 0, 0, frenet_state[FS_DV], frenet_state[FS_DX]])
        return poly_s, poly_d

    @staticmethod
    def _correct_velocity_values(ftrajectories: FrenetTrajectories2D) -> \
            FrenetTrajectories2D:
        """
        Velocity values of werling trajectories can be received with minor numerical deviations.
        This method verifies that if such deviations exist, they are indeed minor, and corrects them
        to the right accurate values.
        :param ftrajectories: trajectories in frenet frame
        :return:Corrected trajectories in frenet frame
        """
        traj_velocities = ftrajectories[:, :, FS_SV]
        # TODO: extract constants to global constants
        is_velocities_close_to_zero = np.logical_and(traj_velocities > -0.1, traj_velocities < 0)
        ftrajectories[is_velocities_close_to_zero, FS_SV] = 0.0

        return ftrajectories

    @staticmethod
    def _compute_cost(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D, state: State,
                      projected_obj_fstates: Dict[int, FrenetState2D], goal_in_frenet: FrenetState2D, safe_distances: np.array, params: TrajectoryCostParams, global_time_samples: np.ndarray,
                      predictor: EgoAwarePredictor, dt: float, reference_route: FrenetSerret2DFrame) -> np.ndarray:
        """
        Takes trajectories (in both frenet-frame repr. and cartesian-frame repr.) and computes a cost for each one
        :param ctrajectories: numpy tensor of trajectories in cartesian-frame
        :param ftrajectories: numpy tensor of trajectories in frenet-frame
        :param state: the state object (that includes obstacles, etc.)
        :param projected_obj_fstates: dict from obj_id to projected Frenet state of the dynamic object on reference_route
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
        pointwise_costs = TrajectoryPlannerCosts.compute_pointwise_costs(ctrajectories, ftrajectories, state,
                                                                         projected_obj_fstates, params,
                                                                         global_time_samples, predictor, dt,
                                                                         reference_route, safe_distances)

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
    def _solve_1d_poly(constraints: np.ndarray, T: float, poly_impl: Poly1D) -> np.ndarray:
        """
        Solves the two-point boundary value problem, given a set of constraints over the initial and terminal states.
        :param constraints: 3D numpy array of a set of constraints over the initial and terminal states
        :param T: longitudinal/lateral trajectory duration (sec.), relative to ego. T has to be a multiple of WerlingPlanner.dt
        :param poly_impl: OptimalControlUtils 1d polynomial implementation class
        :return: a poly-coefficients-matrix of rows in the form [c0_s, c1_s, ... c5_s] or [c0_d, ..., c5_d]
        """
        if T > 0:  # prevent division by zero
            A = poly_impl.time_constraints_matrix(T)
            A_inv = np.linalg.inv(A)
            poly_coefs = poly_impl.solve(A_inv, constraints)
        else:  # for T == 0 return polynomials for constant velocity extrapolation, starting from the start constraints
            poly_coefs = np.concatenate((np.zeros(constraints.shape[0], constraints.shape[1] - 2),
                                         constraints[:, 1], constraints[:, 0]), axis=-1)
        return poly_coefs

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
            time_samples_d = np.arange(0, T_d + EPS, dt)

            # solve for dimension d (with time-horizon T_d)
            partial_poly_d = WerlingPlanner._solve_1d_poly(constraints_d, T_d, QuinticPoly1D)

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

    @prof.ProfileFunction()
    def filter_trajectories_by_safety(self, state: State, objects_fstates: Dict[int, FrenetState2D],
                                      time_samples: np.ndarray, ego_ftrajectories: FrenetTrajectories2D) -> \
            [np.array, np.array]:
        """
        Filter frenet trajectories by RSS safety (both longitudinal & lateral).
        The naive objects prediction in Frenet frame is used.
        :param state: the current state
        :param objects_fstates: dict from obj_id to projected Frenet state of the dynamic object on reference_route
        :param time_samples: time samples of ego trajectories including extrapolation (if T_s < 2)
        :param ego_ftrajectories: ego Frenet trajectories
        :return: indices of safe trajectories and point-wise safe distances
        """
        if len(time_samples) == 0:
            return np.full(ego_ftrajectories.shape[0], True), np.full(ego_ftrajectories.shape, 1000.)

        # find relevant objects for the safety test
        relevant_objects = WerlingPlanner._choose_relevant_objects_for_safety(
            state.ego_state.size, state.dynamic_objects, objects_fstates, time_samples, ego_ftrajectories, self.predictor)

        # extract Frenet states and sizes of the relevant objects
        objects_frenet_states_list = []
        obj_sizes = []
        for obj in relevant_objects:
            objects_frenet_states_list.append(objects_fstates[obj.obj_id])
            obj_sizes.append(obj.size)
        if len(objects_frenet_states_list) == 0:
            return np.array(range(ego_ftrajectories.shape[0])), \
                   np.zeros((ego_ftrajectories.shape[0], ego_ftrajectories.shape[1], 2))

        # create a matrix of all objects' predictions and a list of objects' sizes
        obj_ftraj = self.predictor.predict_2d_frenet_states(np.array(objects_frenet_states_list),
                                                            time_samples)

        # calculate RSS safety for all trajectories, all objects and all timestamps
        safe_distances = SafetyUtils.get_safe_distances(ego_ftrajectories,
                                                        state.ego_state.size, obj_ftraj, obj_sizes)
        safe_distances_for_all_objects = np.min(safe_distances, axis=1)  # min on objects

        # AND over all objects and all timestamps, OR on (lon,lat)
        safe_trajectories = (safe_distances[:, :, :TRAJECTORY_NUM_POINTS, :] > 0).all(axis=(1, 2)).any(axis=-1)

        return np.where(safe_trajectories)[0], safe_distances_for_all_objects

    @staticmethod
    def _choose_relevant_objects_for_safety(ego_size: ObjectSize, dynamic_objects: List[DynamicObject],
                                            projecte_obj_fstates: Dict[int, FrenetState2D],
                                            time_samples: np.ndarray, ego_ftrajectories: FrenetTrajectories2D,
                                            predictor: EgoAwarePredictor) -> List[DynamicObject]:
        """
        Choose at most 3 objects relevant for the safety test: F, LF/RF, RF/RB
        :param ego_size:
        :param dynamic_objects: list of all N dynamic objects in the state
        :param projecte_obj_fstates: dictionary: obj_id -> object's Frenet state on the reference route
        :param time_samples: time samples of ego trajectories
        :param ego_ftrajectories: ego Frenet trajectories
        :return: objects list relevant for the safety
        """
        # find at most 3 dynamic objects relevant for safety
        # TODO: the following logic is correct only for naive predictions, when dynamic objects don't change lane
        ego_fstate = ego_ftrajectories[0, 0]

        if len(projecte_obj_fstates.values()) == 0:
            return []

        obj_id_to_orig_idx = dict([(obj.obj_id, orig_idx) for orig_idx, obj in enumerate(dynamic_objects)])
        obj_orig_indices = np.array([obj_id_to_orig_idx[obj_id] for obj_id in projecte_obj_fstates.keys()])
        objects_widths = np.array([dynamic_objects[obj_id_to_orig_idx[obj_id]].size.width
                                   for obj_id in projecte_obj_fstates.keys()])
        common_widths = 0.5 * (ego_size.width + objects_widths)

        relevant_objects = []
        objects_fstates_array = np.array(list(projecte_obj_fstates.values()))
        # find front object F: the object is in front of ego longitudinally and is close to ego laterally
        start_front_idxs = np.where(np.logical_and(
            ego_fstate[FS_SX] < objects_fstates_array[:, FS_SX],
            np.abs(ego_fstate[FS_DX] - objects_fstates_array[:, FS_DX]) < common_widths))[0]
        if len(start_front_idxs) > 0:
            closest_front_idx = start_front_idxs[np.argmin(objects_fstates_array[:, FS_SX][start_front_idxs])]
            relevant_objects.append(dynamic_objects[obj_orig_indices[closest_front_idx]])

        # find the objects LF/RF and LB/RB
        ego_end_x = ego_ftrajectories[0, -1, FS_SX]
        objects_end_state = predictor.predict_2d_frenet_states(objects_fstates_array, time_samples[-1:])[:, 0]
        objects_end_x, objects_end_y = objects_end_state[:, FS_SX], objects_end_state[:, FS_DX]

        # find the closest back objects to the final ego position
        end_back_idxs = np.where(np.logical_and(ego_end_x >= objects_end_x, np.abs(objects_end_y) < common_widths))[0]
        if len(end_back_idxs) > 0:
            closest_end_back = end_back_idxs[np.argmax(objects_end_x[end_back_idxs])]
            relevant_objects.append(dynamic_objects[obj_orig_indices[closest_end_back]])

        # find the closest front objects to the final ego position
        end_front_idxs = np.where(np.logical_and(ego_end_x < objects_end_x, np.abs(objects_end_y) < common_widths))[0]
        if len(end_front_idxs) > 0:
            closest_front_idx = end_front_idxs[np.argmin(objects_end_x[end_front_idxs])]
            relevant_objects.append(dynamic_objects[obj_orig_indices[closest_front_idx]])

        # remove duplicated objects (e.g. if F becomes LF)
        seen = set()
        # The statement if seen.add(...) always returns False
        return [obj for obj in relevant_objects if obj.obj_id not in seen and not seen.add(obj.obj_id)]
