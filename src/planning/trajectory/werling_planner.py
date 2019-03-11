import numpy as np
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from logging import Logger
from typing import Tuple, List, Dict
import rte.python.profiler as prof

from decision_making.src.exceptions import NoValidTrajectoriesFound, CouldNotGenerateTrajectories, \
    NoSafeTrajectoriesFound
from decision_making.src.global_constants import WERLING_TIME_RESOLUTION, SX_STEPS, SV_OFFSET_MIN, SV_OFFSET_MAX, \
    SV_STEPS, DX_OFFSET_MIN, DX_OFFSET_MAX, DX_STEPS, SX_OFFSET_MIN, SX_OFFSET_MAX, \
    TD_STEPS, LAT_ACC_LIMITS, TD_MIN_DT, LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, EPS, VELOCITY_LIMITS, \
    LON_ACC_LIMITS, TRAJECTORY_NUM_POINTS
from decision_making.src.messages.trajectory_parameters import TrajectoryCostParams
from decision_making.src.planning.trajectory.cost_function import TrajectoryPlannerCosts
from decision_making.src.planning.trajectory.frenet_constraints import FrenetConstraints
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.trajectory.trajectory_planner import TrajectoryPlanner, SamplableTrajectory
from decision_making.src.planning.trajectory.werling_utils import WerlingUtils
from decision_making.src.planning.types import FP_SX, FP_DX, C_V, FS_SV, \
    FS_SA, FS_SX, FS_DX, LIMIT_MIN, LIMIT_MAX, CartesianTrajectories, FS_DV, FS_DA, CartesianExtendedState, \
    FrenetState2D, C_A, C_K, D5, Limits
from decision_making.src.planning.types import FrenetTrajectories2D, CartesianExtendedTrajectories
from decision_making.src.planning.utils.frenet_serret_frame import FrenetSerret2DFrame
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, Poly1D
from decision_making.src.prediction.ego_aware_prediction.ego_aware_predictor import EgoAwarePredictor
from decision_making.src.state.state import State, DynamicObject, ObjectSize


class WerlingPlanner(TrajectoryPlanner):
    def __init__(self, logger: Logger, predictor: EgoAwarePredictor, dt=WERLING_TIME_RESOLUTION):
        super().__init__(logger, predictor)
        self._dt = dt

    @property
    def dt(self):
        return self._dt

    def plan(self, state: State, reference_route: FrenetSerret2DFrame, projected_obj_fstates: Dict[int, FrenetState2D],
             goal: CartesianExtendedState, time_horizon: float, minimal_required_horizon: float,
             cost_params: TrajectoryCostParams) -> \
            Tuple[SamplableTrajectory, CartesianTrajectories, np.ndarray]:
        """ see base class """

        # The reference_route, the goal, ego and the dynamic objects are given in the global coordinate-frame.
        # The vehicle doesn't need to lay parallel to the road.

        ego_frenet_state: FrenetState2D = reference_route.cstate_to_fstate(state.ego_state.cartesian_state)

        # define constraints for the terminal (goal) state
        goal_frenet_state: FrenetState2D = reference_route.cstate_to_fstate(goal)

        if ego_frenet_state[FS_SX] > goal_frenet_state[FS_SX]:
            self._logger.warning('Goal longitudinal %s is behind ego longitudinal %s', goal_frenet_state[FS_SX], ego_frenet_state[FS_SX])

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

        # If ego is very close to goal laterally (DX,DV,DA), set the start lateral constraints identical to the goal's.
        # The purpose: when T_d is tiny, there is no feasible trajectory to perform the lateral maneuver.
        if np.isclose(ego_frenet_state[FS_DX:], goal_frenet_state[FS_DX:], atol=0.01).all():
            # define constraints for the initial state, such that d_constraints are identical to the goal
            fconstraints_t0 = FrenetConstraints(ego_frenet_state[FS_SX], ego_frenet_state[FS_SV], ego_frenet_state[FS_SA],
                                                goal_frenet_state[FS_DX], goal_frenet_state[FS_DV], goal_frenet_state[FS_DA])
        else:  # not very close to goal laterally
            # define constraints for the initial state
            fconstraints_t0 = FrenetConstraints.from_state(ego_frenet_state)

        fconstraints_tT = FrenetConstraints(sx=sx_range, sv=sv_range, sa=goal_frenet_state[FS_SA],
                                            dx=dx_range, dv=goal_frenet_state[FS_DV], da=goal_frenet_state[FS_DA])

        T_s = max(time_horizon, 0)
        planning_horizon = max(minimal_required_horizon, T_s) + EPS

        # TODO: should we make sure T_s values are multiples of dt?
        # (Otherwise the matrix, calculated using T_s,and the longitudinal time axis, lon_time_samples, won't fit).
        # T_s = Math.round_to_step(T_s, self.dt)

        # Lateral planning horizon(Td) lower bound, now approximated from x=a*t^2
        lower_bound_T_d = self._low_bound_lat_horizon(fconstraints_t0, fconstraints_tT, T_s, self.dt)

        # create a grid on T_d (lateral movement time-grid)
        T_d_grid = WerlingPlanner._create_lat_horizon_grid(T_s, lower_bound_T_d)

        self._logger.debug("Lateral horizon grid considered is: {}".format(str(T_d_grid)))

        self._logger.debug(
            'WerlingPlanner is planning from %s (frenet) to %s (frenet) in %s seconds and extrapolating to %s seconds',
            NumpyUtils.str_log(ego_frenet_state), NumpyUtils.str_log(goal_frenet_state),
            T_s, planning_horizon)

        base_traj_idx = None

        if T_s > 0 and goal_frenet_state[FS_SX] > ego_frenet_state[FS_SX]:

            # solve problem in frenet-frame
            ftrajectories, poly_coefs, T_d_vals = WerlingPlanner._solve_optimization(
                fconstraints_t0, fconstraints_tT, T_s, T_d_grid, self.dt)

            # TODO: remove it
            end_fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs[:, :6], np.array([T_s]))
            end_fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs[:, 6:], np.array([T_s]))
            ds = np.abs(end_fstates_s[:, 0, 0] - goal_frenet_state[FS_SX])
            dd = np.abs(end_fstates_d[:, 0, 0] - goal_frenet_state[FS_DX])
            min_dist = np.min(ds + dd)
            base_traj_idx = np.where(np.isclose(ds + dd, min_dist))[0][-1]  # trajectory with maximal T_d
            if min_dist > 0.001:
                print('_solve_optimization: min_dist=%.3f ds=%.3f dd=%.3f' % (min_dist, ds[base_traj_idx], dd[base_traj_idx]))

            if planning_horizon > T_s:
                time_samples = np.arange(self.dt, planning_horizon - Math.floor_to_step(T_s, self.dt) + EPS, self.dt)
                extrapolated_fstates = self.predictor.predict_2d_frenet_states(ftrajectories[:, -1, :], time_samples)
                ftrajectories = np.hstack((ftrajectories, extrapolated_fstates))

            lat_frenet_filtered_indices = self._filter_by_lateral_frenet_limits(poly_coefs[:, D5:], T_d_vals, cost_params)
        else:
            # Goal is behind us
            time_samples = np.arange(0, planning_horizon + EPS, self.dt)
            # Create only one trajectory which is actually a constant-velocity predictor of current state
            ftrajectories = self.predictor.predict_2d_frenet_states(np.array([ego_frenet_state]), time_samples)
            # here we just take the current state and extrapolate it linearly in time with constant velocity,
            # meaning no lateral motion is carried out.
            T_d_vals = np.array([0])
            lat_frenet_filtered_indices = np.array([0])
            poly_coefs = np.array([np.array([0,0,0,0, ego_frenet_state[FS_SV], ego_frenet_state[FS_SX], 0,0,0,0,0,0])])

        # filter resulting trajectories by progress on curve, velocity and (lateral) accelerations limits in frenet
        lon_frenet_filtered_indices = self._filter_by_longitudinal_frenet_limits(ftrajectories, reference_route.s_limits)
        frenet_filtered_indices = np.intersect1d(lat_frenet_filtered_indices, lon_frenet_filtered_indices)

        # TODO: remove it
        if base_traj_idx is not None and base_traj_idx not in lon_frenet_filtered_indices:
            print('base traj filtered by lon')
        if base_traj_idx is not None and base_traj_idx not in lat_frenet_filtered_indices:
            print('base traj filtered by lat')

        # project trajectories from frenet-frame to vehicle's cartesian frame
        ctrajectories: CartesianExtendedTrajectories = reference_route.ftrajectories_to_ctrajectories(
            ftrajectories[frenet_filtered_indices])

        # filter resulting trajectories by velocity and accelerations limits - this is now done in Cartesian frame
        # which takes into account the curvature of the road applied to trajectories planned in the Frenet frame
        cartesian_refiltered_indices = self._filter_by_cartesian_limits(ctrajectories, cost_params)

        refiltered_indices = frenet_filtered_indices[cartesian_refiltered_indices]
        ctrajectories_filtered = ctrajectories[cartesian_refiltered_indices]
        ftrajectories_refiltered = ftrajectories[frenet_filtered_indices][cartesian_refiltered_indices]

        # TODO: remove it
        if base_traj_idx is not None and base_traj_idx in frenet_filtered_indices and base_traj_idx not in refiltered_indices:
            idx = np.where(frenet_filtered_indices == base_traj_idx)[0][0]
            lon_acceleration = ctrajectories[idx, :, C_A]
            lat_acceleration = ctrajectories[idx, :, C_V] ** 2 * ctrajectories[idx, :, C_K]
            lon_velocity = ctrajectories[idx, :, C_V]
            vel = np.all(NumpyUtils.is_almost_in_limits(lon_velocity, cost_params.velocity_limits))
            acc = np.all(NumpyUtils.is_in_limits(lon_acceleration, cost_params.lon_acceleration_limits))
            lat = np.all(NumpyUtils.is_in_limits(lat_acceleration, cost_params.lat_acceleration_limits))
            badacc = lon_acceleration[np.logical_or(lon_acceleration < LON_ACC_LIMITS[0], lon_acceleration > LON_ACC_LIMITS[1])]
            badlat = lon_acceleration[np.logical_or(lat_acceleration < LAT_ACC_LIMITS[0], lat_acceleration > LAT_ACC_LIMITS[1])]
            print('base traj filtered by cartesian: vel=%d acc=%d lat=%d; badacc=%s badlat=%s' % (vel, acc, lat, badacc, badlat))

        self._logger.debug(LOG_MSG_TRAJECTORY_PLANNER_NUM_TRAJECTORIES, len(ctrajectories_filtered))

        if len(ctrajectories) == 0:
            raise CouldNotGenerateTrajectories("No valid cartesian trajectories. time: %f, goal: %s\nstate: %s\n"
                                               "Longitudes range: [%s, %s] (limits: %s), "
                                               "MinMax frenet velocity: (%s, %s)\n"
                                               "number of trajectories passed according to Frenet limits: %s/%s; "
                                               "pass_lon=%d pass_lat=%d\nego_frenet_state=%s\ngoal_frenet_state=%s" %
                                               (T_s, NumpyUtils.str_log(goal), str(state).replace('\n', ''),
                                                np.min(ftrajectories[:, :, FS_SX]), np.max(ftrajectories[:, :, FS_SX]),
                                                reference_route.s_limits,
                                                np.min(ftrajectories[:, :, FS_SV]), np.max(ftrajectories[:, :, FS_SV]),
                                                len(frenet_filtered_indices), len(ftrajectories),
                                                len(lon_frenet_filtered_indices), len(lat_frenet_filtered_indices),
                                                ego_frenet_state, goal_frenet_state))

        elif len(ctrajectories_filtered) == 0:
            lat_acc = ctrajectories[:, :, C_V] ** 2 * ctrajectories[:, :, C_K]
            valid_lat_acc = (np.abs(lat_acc) < cost_params.lat_acceleration_limits[1]).all(axis=-1)
            valid_lon_acc = NumpyUtils.is_in_limits(ctrajectories[:, :, C_A], cost_params.lon_acceleration_limits).all(axis=-1)
            end_fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs[:, :6], np.array([T_s]))
            end_fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs[:, 6:], np.array([T_s]))
            bad_lat_acc_traj = np.argmin(np.abs(end_fstates_s[:, 0, 0] - goal_frenet_state[FS_SX]) +
                                         np.abs(end_fstates_d[:, 0, 0] - goal_frenet_state[FS_DX]))
            worst_lat_acc_time_sample = np.argmax(np.abs(lat_acc[bad_lat_acc_traj, :]))
            worst_lat_acc = lat_acc[bad_lat_acc_traj, worst_lat_acc_time_sample]
            rr_idx = reference_route.get_index_on_frame_from_s(
                np.array([ftrajectories[bad_lat_acc_traj, worst_lat_acc_time_sample, FS_SX]]))[0][0]
            raise NoValidTrajectoriesFound(
                "No valid trajectories found. timestamp: %.3f horizon_time: %.3f, goal: %s, state: %s.\n"
                "planned velocities range [%s, %s] (limits: %s); "
                "planned lon. accelerations range [%s, %s] (limits: %s); "
                "planned lat. accelerations range [%s, %s] (limits: %s)\n"
                "passed Frenet limits: %s/%s; passed Cartesian limits: %s/%s; passed all limits: %s/%s;\n"
                "ego_frenet = %s\ngoal_frenet = %s\ndistance from ego to goal = %f, time*approx_velocity = %f\n"
                "valid_lon_acc=%d valid_lat_acc=%d valid_acc=%d worst_lat_acc_time=%.3f worst_lat_acc=%.3f"
                "\nroad_curvature=%s" %
                (state.ego_state.timestamp_in_sec, T_s, NumpyUtils.str_log(goal),
                str(state).replace('\n', ''),
                np.min(ctrajectories[:, :, C_V]), np.max(ctrajectories[:, :, C_V]), list(cost_params.velocity_limits),
                np.min(ctrajectories[:, :, C_A]), np.max(ctrajectories[:, :, C_A]), list(cost_params.lon_acceleration_limits),
                np.min(lat_acc), np.max(lat_acc), list(cost_params.lat_acceleration_limits),
                len(frenet_filtered_indices), len(ftrajectories), len(cartesian_refiltered_indices), len(ctrajectories),
                len(refiltered_indices), len(ftrajectories), ego_frenet_state,
                goal_frenet_state, goal_frenet_state[FS_SX] - ego_frenet_state[FS_SX],
                time_horizon * (ego_frenet_state[FS_SV] + goal_frenet_state[FS_SV])*0.5,
                np.sum(valid_lon_acc), np.sum(valid_lat_acc), np.sum(np.logical_and(valid_lon_acc, valid_lat_acc)),
                worst_lat_acc_time_sample*self.dt, worst_lat_acc, reference_route.k[rr_idx-3:rr_idx+4]))

        # planning is done on the time dimension relative to an anchor (currently the timestamp of the ego vehicle)
        # so time points are from t0 = 0 until some T (lon_plan_horizon)
        planning_time_points = np.arange(0, ftrajectories.shape[1] * self.dt - EPS, self.dt)

        # filter trajectories by RSS safety
        safe_traj_indices, safe_distances = \
            self.filter_trajectories_by_safety(state, projected_obj_fstates, planning_time_points, ftrajectories_refiltered)
        # Throw an error if no safe trajectory is found
        if len(safe_traj_indices) == 0:
            raise NoSafeTrajectoriesFound("No safe trajectories found\ntime: %f, goal_frenet: %s\nstate: %s\n"
                                          "ego_fstate: %s\nobjects_fstates: %s\nsafe times:\n%s" %
                                          (T_s, NumpyUtils.str_log(goal_frenet_state), str(state).replace('\n', ''),
                                           NumpyUtils.str_log(ftrajectories[0, 0, :]),
                                           NumpyUtils.str_log(np.array(list(projected_obj_fstates.values()))),
                                           (safe_distances[..., 0] > 0).astype(int)))

        ftrajectories_refiltered_safe = ftrajectories_refiltered[safe_traj_indices]
        ctrajectories_filtered_safe = ctrajectories_filtered[safe_traj_indices]
        refiltered_indices_safe = refiltered_indices[safe_traj_indices]
        safe_distances_safe = safe_distances[safe_traj_indices]

        # TODO: remove it
        if base_traj_idx is not None and base_traj_idx not in refiltered_indices_safe and base_traj_idx in refiltered_indices:
            print('base traj filtered by safety')

        # compute trajectory costs at sampled times
        global_time_sample = planning_time_points + state.ego_state.timestamp_in_sec
        filtered_trajectory_costs, pointwise_costs, dist_from_goal_costs = \
            self._compute_cost(ctrajectories_filtered_safe, ftrajectories_refiltered_safe, state, projected_obj_fstates,
                               goal_frenet_state, safe_distances_safe, cost_params, global_time_sample,
                               self._predictor, self.dt, reference_route, poly_coefs[refiltered_indices_safe], T_s)

        sorted_filtered_idxs = filtered_trajectory_costs.argsort()

        # TODO: remove it
        if base_traj_idx is not None and refiltered_indices_safe[sorted_filtered_idxs[0]] != base_traj_idx:
            safe_idxs = np.where(refiltered_indices_safe == base_traj_idx)[0]
            if len(safe_idxs) > 0:
                i = safe_idxs[0]
                print('base traj has no best cost: goal=%.3f obs=%.3f dev=%.3f jerk=%.3f safety=%.3f' %
                      (dist_from_goal_costs[i], np.sum(pointwise_costs[i, :, 0]), np.sum(pointwise_costs[i, :, 1]),
                       np.sum(pointwise_costs[i, :, 2]), np.sum(pointwise_costs[i, :, 3])))

        if T_s > 0 and goal_frenet_state[FS_SX] > ego_frenet_state[FS_SX]:
            samplable_trajectory = SamplableWerlingTrajectory(
                timestamp_in_sec=state.ego_state.timestamp_in_sec,
                T_s=T_s,
                T_d=T_d_vals[refiltered_indices_safe[sorted_filtered_idxs[0]]],
                frenet_frame=reference_route,
                poly_s_coefs=poly_coefs[refiltered_indices_safe[sorted_filtered_idxs[0]]][:6],
                poly_d_coefs=poly_coefs[refiltered_indices_safe[sorted_filtered_idxs[0]]][6:],
                total_time=planning_horizon
            )
        else:  # the goal is behind ego (by time or by s)
            samplable_trajectory = SamplableWerlingTrajectory(
                timestamp_in_sec=state.ego_state.timestamp_in_sec,
                T_s=T_s,
                T_d=T_s,
                frenet_frame=reference_route,
                poly_s_coefs=np.array([0, 0, 0, 0, ego_frenet_state[FS_SV], ego_frenet_state[FS_SX]]),
                poly_d_coefs=np.zeros(6),
                total_time=planning_horizon
            )

        self._logger.debug("Chosen trajectory planned with lateral horizon : {}".format(
            T_d_vals[refiltered_indices_safe[sorted_filtered_idxs[0]]]))

        return samplable_trajectory, \
               ctrajectories_filtered_safe[sorted_filtered_idxs, :, :(C_V + 1)], \
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
            NumpyUtils.is_almost_in_limits(lon_velocity, cost_params.velocity_limits) &
            NumpyUtils.is_in_limits(lon_acceleration, cost_params.lon_acceleration_limits) &
            NumpyUtils.is_in_limits(lat_acceleration, cost_params.lat_acceleration_limits), axis=1)

        return np.argwhere(conforms).flatten()

    @staticmethod
    def _filter_by_longitudinal_frenet_limits(ftrajectories: FrenetTrajectories2D, reference_route_limits: Limits) -> np.ndarray:
        """
        Given a set of trajectories in Frenet coordinate-frame, it validates them against the following limits:
        (longitudinal progress on the frenet frame curve, positive longitudinal velocity)
        :param ftrajectories: FrenetTrajectories2D object of trajectories to validate
        :param reference_route_limits: the minimal and maximal progress (s value) on the reference route used
        in the frenet frame used for planning
        :return: Indices along the 1st dimension in <ctrajectories> (trajectory index) for valid trajectories
        """
        # validate the progress on the reference-route curve doesn't extrapolate, and that velocity is non-negative
        conforms = np.all(
            NumpyUtils.is_in_limits(ftrajectories[:, :, FS_SX], reference_route_limits) &
            np.greater_equal(ftrajectories[:, :, FS_SV], VELOCITY_LIMITS[LIMIT_MIN] - EPS), axis=1)

        return np.argwhere(conforms).flatten()

    @staticmethod
    def _filter_by_lateral_frenet_limits(poly_coefs_d: np.ndarray, T_d_vals: np.ndarray,
                                         cost_params: TrajectoryCostParams) -> np.ndarray:
        """
        Given a set of trajectories in Frenet coordinate-frame, it validates that the acceleration of the lateral
        polynomial used in planning is in the allowed limits.
        :param poly_coefs_d: 2D numpy array of each row has 6 poly coefficients of lateral polynomial
        :param T_d_vals: 1D numpy array with lateral planning-horizons (correspond to each trajectory)
        :param cost_params: TrajectoryCostParams object that holds desired limits (for validation)
        :return: Indices along the 1st dimension in <ctrajectories> (trajectory index) for valid trajectories
        """
        # here we validate feasible lateral acceleration *directly from the lateral polynomial* because our
        # discretization of the trajectory (via sampling with constant self.dt) can overlook cases where there is a high
        # lateral acceleration between two adjacent sampled points (critical in the lateral case because we allow
        # shorter lateral maneuvers
        frenet_lateral_movement_is_feasible = \
            QuinticPoly1D.are_accelerations_in_limits(poly_coefs_d, T_d_vals, cost_params.lat_acceleration_limits)

        return np.argwhere(frenet_lateral_movement_is_feasible).flatten()

    @staticmethod
    def _compute_cost(ctrajectories: CartesianExtendedTrajectories, ftrajectories: FrenetTrajectories2D,
                      state: State, projected_obj_fstates: Dict[int, FrenetState2D],
                      goal_in_frenet: FrenetState2D, safe_distances: np.array, params: TrajectoryCostParams,
                      global_time_samples: np.ndarray, predictor: EgoAwarePredictor, dt: float,
                      reference_route: FrenetSerret2DFrame, poly_coefs: np.array, T_s: float) -> [np.ndarray, np.array, np.array]:
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
        end_fstates_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs[:, :6], np.array([T_s]))
        end_fstates_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs[:, 6:], np.array([T_s]))
        dist_from_goal_s = end_fstates_s[:, 0, 0] - goal_in_frenet[FS_SX]
        dist_from_goal_d = end_fstates_d[:, 0, 0] - goal_in_frenet[FS_DX]

        trajectory_end_goal_dist = np.sqrt(dist_from_goal_s ** 2 + (params.dist_from_goal_lat_factor * dist_from_goal_d) ** 2)
        dist_from_goal_costs = Math.clipped_sigmoid(trajectory_end_goal_dist - params.dist_from_goal_cost.offset,
                                                    params.dist_from_goal_cost.w, params.dist_from_goal_cost.k) - \
                               Math.clipped_sigmoid(-params.dist_from_goal_cost.offset,
                                                    params.dist_from_goal_cost.w, params.dist_from_goal_cost.k)

        ''' point-wise costs: obstacles, deviations, jerk '''
        pointwise_costs = TrajectoryPlannerCosts.compute_pointwise_costs(
            ctrajectories, ftrajectories, state, projected_obj_fstates, params, global_time_samples, predictor, dt,
            reference_route, safe_distances)

        return np.sum(pointwise_costs, axis=(1, 2)) + dist_from_goal_costs, \
               pointwise_costs, dist_from_goal_costs  # TODO: remove it

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
        T_d_vals = np.array([T_s])
        if T_s != T_d_low_bound:
            T_d_vals = np.linspace(T_d_low_bound, T_s, TD_STEPS)

        return T_d_vals

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
        valid_traj_slice = horizons[:, FP_SX] + EPS >= horizons[:, FP_DX]

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
