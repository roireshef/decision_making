from collections import defaultdict

import numpy as np
from decision_making.src.planning.utils.safety_utils import SafetyUtils
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.state.state import State
from rte.python.logger.AV_logger import AV_Logger
from typing import List

import rte.python.profiler as prof
from decision_making.src.global_constants import LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, MAX_SAFETY_T_D_GRID_SIZE
from decision_making.src.global_constants import EPS, WERLING_TIME_RESOLUTION, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS, FILTER_V_0_GRID, FILTER_V_T_GRID, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition, StaticActionRecipe, RelativeLane
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SA, FS_DX, FS_SV, FS_SX, FrenetState2D, FrenetTrajectories2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils, BrakingDistances
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.utils.map_utils import MapUtils
from typing import List
from decision_making.src.planning.behavioral.filtering.constraint_spec_filter import ConstraintSpecFilter


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState, state: State) -> List[bool]:
        return [(action_spec and behavioral_state) is not None and ~np.isnan(action_spec.t) for action_spec in action_specs]


class FilterForKinematics(ActionSpecFilter):
    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState, state: State) -> List[bool]:
        """ Builds a baseline trajectory out of the action specs (terminal states) and validates them against:
            - max longitudinal position (available in the reference frame)
            - longitudinal velocity limits - both in Frenet (analytical) and Cartesian (by sampling)
            - longitudinal acceleration limits - both in Frenet (analytical) and Cartesian (by sampling)
            - lateral acceleration limits - in Cartesian (by sampling) - this isn't tested in Frenet, because Frenet frame
            conceptually "straightens" the road's shape.
        :param action_specs: list of action specs
        :param behavioral_state:
        :return: boolean list per action spec: True if a spec passed the filter
        """
        # extract all relevant information for boundary conditions
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[spec.relative_lane] for spec in action_specs])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        T = np.array([spec.t for spec in action_specs])

        # create boolean arrays indicating whether the specs are in tracking mode
        padding_mode = np.array([spec.only_padding_mode for spec in action_specs])
        not_padding_mode = np.logical_not(padding_mode)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T[not_padding_mode])

        # represent initial and terminal boundary conditions (for two Frenet axes s,d) for non-tracking specs
        constraints_s = np.concatenate((initial_fstates[not_padding_mode, :FS_DX], terminal_fstates[not_padding_mode, :FS_DX]), axis=1)
        constraints_d = np.concatenate((initial_fstates[not_padding_mode, FS_DX:], terminal_fstates[not_padding_mode, FS_DX:]), axis=1)

        # solve for s(t) and d(t)
        poly_coefs_s, poly_coefs_d = np.zeros((len(action_specs), 6)), np.zeros((len(action_specs), 6))
        poly_coefs_s[not_padding_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d[not_padding_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_d)
        # in tracking mode (constant velocity) the s polynomials have "approximately" only two non-zero coefficients
        poly_coefs_s[padding_mode, 4:] = np.c_[terminal_fstates[padding_mode, FS_SV], terminal_fstates[padding_mode, FS_SX]]

        are_valid = []
        for poly_s, poly_d, t, spec in zip(poly_coefs_s, poly_coefs_d, T, action_specs):
            # TODO: in the future, consider leaving only a single action (for better "learnability")

            # extract the relevant (cached) frenet frame per action according to the destination lane
            frenet_frame = behavioral_state.extended_lane_frames[spec.relative_lane]

            if not spec.only_padding_mode:
                # if the action is static, there's a chance the 5th order polynomial is actually a degenerate one
                # (has lower degree), so we clip the first zero coefficients and send a polynomial with lower degree
                # TODO: This handling of polynomial coefficients being 5th or 4th order should happen in an inner context and get abstracted from this method
                first_non_zero = np.argmin(np.equal(poly_s, 0)) if isinstance(spec.recipe, StaticActionRecipe) else 0
                is_valid_in_frenet = KinematicUtils.filter_by_longitudinal_frenet_limits(
                    poly_s[np.newaxis, first_non_zero:], np.array([t]), LON_ACC_LIMITS, VELOCITY_LIMITS, frenet_frame.s_limits)[0]

                # frenet checks are analytical and do not require conversions so they are faster. If they do not pass,
                # we can save time by not checking cartesian limits
                if not is_valid_in_frenet:
                    are_valid.append(False)
                    continue

            total_time = max(MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, t)
            time_samples = np.arange(0, total_time + EPS, WERLING_TIME_RESOLUTION)

            # generate a SamplableWerlingTrajectory (combination of s(t), d(t) polynomials applied to a Frenet frame)
            samplable_trajectory = SamplableWerlingTrajectory(0, t, t, total_time, frenet_frame, poly_s, poly_d)
            cartesian_points = samplable_trajectory.sample(time_samples)  # sample cartesian points from the solution

            # validate cartesian points against cartesian limits
            is_valid_in_cartesian = KinematicUtils.filter_by_cartesian_limits(
                cartesian_points[np.newaxis, ...], VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS)[0]
            are_valid.append(is_valid_in_cartesian)

        return are_valid


class FilterUnsafeExpectedTrajectory(ActionSpecFilter):
    """
    This filter checks full safety (by RSS) toward the followed vehicle
    """
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState, state: State) -> List[ActionSpec]:
        """
        This filter checks for each action_spec if it's trajectory is safe w.r.t. the followed vehicle.
        :param action_specs:
        :param behavioral_state:
        :return: boolean array of size len(action_specs)
        """
        safe_specs = FilterUnsafeExpectedTrajectory._check_actions_safety(action_specs, behavioral_state, state)
        return [spec if safe_specs[i] else None for i, spec in enumerate(action_specs)]

    @staticmethod
    def _check_actions_safety(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState, state: State) -> np.array:
        """
        Check RSS safety for all action specs, for which action_specs_mask is true.
        An action spec is considered safe if it's safe wrt all dynamic objects for all timestamps < spec.t.
        :param action_specs: list of action specifications
        :param behavioral_state:
        :return: boolean list of safe specifications. The list's size is equal to the original action_specs size.
        Specifications filtered by action_specs_mask are considered "unsafe".
        """
        logger = AV_Logger.get_logger('_check_actions_safety')
        predictor = RoadFollowingPredictor(logger)
        ego_size = behavioral_state.ego_state.size
        behavioral_state.marginal_safety_per_action = np.zeros(len(action_specs))

        # collect action specs into at most 3 groups, according to their target lane
        safe_specs = np.full(len(action_specs), True)
        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)

        for rel_lane in specs_by_rel_lane.keys():
            lane_specs = specs_by_rel_lane[rel_lane]
            lane_frenet = behavioral_state.extended_lane_frames[rel_lane]
            ego_init_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            T_d_grid_size = MAX_SAFETY_T_D_GRID_SIZE

            # find relevant objects for safety: front and from the target lane
            relevant_objects = []
            front_cell = (rel_lane, RelativeLongitudinalPosition.FRONT)
            if front_cell in behavioral_state.road_occupancy_grid:
                relevant_objects.append(behavioral_state.road_occupancy_grid[front_cell][0].dynamic_object)
            else:
                # if there is no front cell in behavioral state, maybe there is a far front object beyond
                # lookahead horizon, which may be relevant for safety for long follow-lane actions
                lane_ids = np.array([obj.map_state.lane_id if obj._cached_map_state is not None else 0
                                     for obj in state.dynamic_objects])
                same_lane_orig_idxs = np.where(lane_frenet.has_segment_ids(lane_ids))[0]
                if len(same_lane_orig_idxs) > 0:
                    same_lane_fstates = np.array([state.dynamic_objects[idx].map_state.lane_fstate
                                                  for idx in same_lane_orig_idxs])
                    projected_obj_fstates = lane_frenet.convert_from_segment_states(same_lane_fstates,
                                                                                    lane_ids[same_lane_orig_idxs])
                    front_idxs = np.where(projected_obj_fstates[:, FS_SX] > ego_init_fstate[FS_SX])[0]
                    if len(front_idxs) > 0:
                        closest_idx = same_lane_orig_idxs[front_idxs[np.argmin(projected_obj_fstates[front_idxs, FS_SX])]]
                        relevant_objects.append(state.dynamic_objects[closest_idx])

            if rel_lane != RelativeLane.SAME_LANE:
                for lon_cell in RelativeLongitudinalPosition:
                    if (rel_lane, lon_cell) in behavioral_state.road_occupancy_grid:
                        relevant_objects.append(behavioral_state.road_occupancy_grid[(rel_lane, lon_cell)][0].dynamic_object)

            obj_sizes = []
            obj_map_states = []
            for obj in relevant_objects:
                obj_sizes.append(obj.size)
                obj_map_states.append(lane_frenet.convert_from_segment_state(obj.map_state.lane_fstate, obj.map_state.lane_id))
            if len(obj_sizes) == 0:
                continue  # if there are no objects on the target lane, then all actions are safe

            # predict objects' trajectories
            time_points = np.arange(0, BP_ACTION_T_LIMITS[1] + EPS, TRAJECTORY_TIME_RESOLUTION)
            obj_trajectories = predictor.predict_2d_frenet_states(np.array(obj_map_states), time_points)

            # generate baseline Frenet trajectories for the relative-lane actions, with T_d grid
            ftrajectories = FilterUnsafeExpectedTrajectory._generate_frenet_trajectories_for_relative_lane(
                ego_init_fstate, lane_specs, T_d_grid_size, time_points)

            # calculate safety for each trajectory, each object, each timestamp
            marginal_safety = SafetyUtils.get_safe_distances(ftrajectories, ego_size, obj_trajectories, obj_sizes)
            safe_times = (marginal_safety > 0).any(axis=-1)  # OR on lon & lat safe distances

            # trajectory is considered safe if it's safe wrt all dynamic objects for all timestamps
            safe_trajectories = safe_times.all(axis=(1, 2))

            # marginal safety is the difference between actual distance from obstacle and the minimal RSS-safe distance
            behavioral_state.marginal_safety_per_action[np.array(indices_by_rel_lane[rel_lane])] = \
                np.max(np.min(marginal_safety[..., 0], axis=(1, 2)).reshape(len(lane_specs), T_d_grid_size), axis=1)

            # find trajectories that are safe for any T_d
            safe_specs[np.array(indices_by_rel_lane[rel_lane])] = \
                safe_trajectories.reshape(len(lane_specs), T_d_grid_size).any(axis=1)  # OR on different T_d

        return safe_specs

    @staticmethod
    def _generate_frenet_trajectories_for_relative_lane(ego_init_fstate: FrenetState2D, lane_specs: List[ActionSpec],
                                                        T_d_grid_size: int, time_points: np.array) -> FrenetTrajectories2D:
        """
        Generate baseline Frenet trajectories for action specifications, with T_d grid.
        :param ego_init_fstate: initial ego frenet state
        :param lane_specs: action specifications list
        :param T_d_grid_size: the size of T_d grid
        :param time_points: array of time samples for the generated trajectories
        :return: the generated Frenet trajectories (3D numpy array)
        """
        # convert the specifications list to 2D matrix, where rows represent different specifications
        spec_arr = np.array([[spec.t, spec.s, spec.v] for i, spec in enumerate(lane_specs)])
        specs_t, specs_s, specs_v = np.split(spec_arr, 3, axis=1)
        specs_t, specs_s, specs_v = specs_t.flatten(), specs_s.flatten(), specs_v.flatten()

        # duplicate initial frenet states and create target frenet states based on the specifications
        zeros = np.zeros_like(specs_t)
        init_fstates = np.tile(ego_init_fstate, len(specs_t)).reshape(-1, len(ego_init_fstate))
        goal_fstates = np.c_[specs_s, specs_v, zeros, zeros, zeros, zeros]

        # calculate A_inv_d as a concatenation of inverse matrices for maximal T_d (= T_s) and for minimal T_d
        A_inv_s = QuinticPoly1D.inverse_time_constraints_tensor(specs_t)
        T_d_arr = FilterUnsafeExpectedTrajectory._calc_T_d_grid(ego_init_fstate[FS_DX:], specs_t, T_d_grid_size)
        A_inv_d = QuinticPoly1D.inverse_time_constraints_tensor(T_d_arr)

        # create ftrajectories_s and duplicated ftrajectories_d (for max_T_d and min_T_d)
        constraints_s = np.concatenate((init_fstates[:, :FS_DX], goal_fstates[:, :FS_DX]), axis=1)
        constraints_d = np.concatenate((init_fstates[:, FS_DX:], goal_fstates[:, FS_DX:]), axis=1)
        duplicated_constraints_d = np.tile(constraints_d, T_d_grid_size).reshape(-1, init_fstates.shape[-1])
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv_s, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv_d, duplicated_constraints_d)
        ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_points)
        ftrajectories_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_points)
        # for any T_d < T_s, complement ftrajectories_d to the length of T_s by setting zeros
        last_t_d_indices = np.floor(T_d_arr / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
        for i, ftrajectory_d in enumerate(ftrajectories_d):
            ftrajectory_d[last_t_d_indices[i]:] = 0

        # extrapolate points beyond spec.t
        end_traj_indices = np.floor(specs_t / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
        for i, ftrajectory_s in enumerate(ftrajectories_s):
            ftrajectory_s[end_traj_indices[i]:, FS_SX] = specs_s[i] + (time_points[end_traj_indices[i]:] - specs_t[i]) * specs_v[i]
            ftrajectory_s[end_traj_indices[i]:, FS_SV:] = np.array([specs_v[i], 0])

        # duplicate each longitudinal trajectory T_d_num times to be aligned with lateral trajectories
        duplicated_ftrajectories_s = np.tile(ftrajectories_s, (1, T_d_grid_size, 1)).reshape(-1, len(time_points), 3)

        # return full Frenet trajectories
        return np.concatenate((duplicated_ftrajectories_s, ftrajectories_d), axis=-1)

    @staticmethod
    def _calc_T_d_grid(fstate_d: np.array, T_s: np.array, T_d_num: int) -> np.array:
        """
        Calculate the lower bound of the lateral time horizon T_d_low_bound and return a grid of possible lateral
        planning time values based on lateral acceleration limits.
        :param fstate_d: 1D array containing: current latitude, lateral velocity and lateral acceleration
        :param T_s: [m] longitudinal time horizon
        :return: numpy array (1D) of the possible lateral planning horizons
                The output array size is len(T_s) * T_d_num, such that for each T_s value there are T_d_num
                values of T_d.
        """
        T_d_grid = np.arange(TRAJECTORY_TIME_RESOLUTION, np.max(T_s) + EPS, TRAJECTORY_TIME_RESOLUTION)
        A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T_d_grid)
        constraints_d = np.concatenate((fstate_d, np.zeros(3)))
        duplicated_constraints = np.tile(constraints_d, len(T_d_grid)).reshape(-1, 6)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, duplicated_constraints)
        acc_in_limits = Poly1D.are_accelerations_in_limits(poly_coefs_d, T_d_grid, LAT_ACC_LIMITS)
        T_d_min = T_d_grid[np.argmax(acc_in_limits)]
        # Create array of size len(T_s) * T_d_num. If for example T_d_num=3, then the output array looks:
        # [T_d_min, (T_s[0]+T_d_min)/2, T_s[0],
        #  T_d_min, (T_s[1]+T_d_min)/2, T_s[1],
        #  ...]
        T_d = np.repeat(T_s, T_d_num)
        T_d[0::T_d_num] = T_d_min
        T_d[1::T_d_num] = (T_d_min + T_s) / 2
        # T_d[2::T_d_num] = T_s
        return T_d


class StaticTrafficFlowControlFilter(ActionSpecFilter):
    """
    Checks if there is a StaticTrafficFlowControl between ego and the goal
    Currently treats every 'StaticTrafficFlowControl' as a stop event.

    """

    @staticmethod
    def _has_stop_bar_until_goal(action_spec: ActionSpec, behavioral_state: BehavioralGridState) -> bool:
        """
        Checks if there is a stop_bar between current ego location and the action_spec goal
        :param action_spec: the action_spec to be considered
        :param behavioral_state: BehavioralGridState in context
        :return: if there is a stop_bar between current ego location and the action_spec goal
        """
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        stop_bar_locations = MapUtils.get_static_traffic_flow_controls_s(target_lane_frenet)
        ego_location = behavioral_state.projected_ego_fstates[action_spec.relative_lane][FS_SX]

        return np.logical_and(ego_location <= stop_bar_locations, stop_bar_locations < action_spec.s).any()

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        return [not StaticTrafficFlowControlFilter._has_stop_bar_until_goal(action_spec, behavioral_state)
                for action_spec in action_specs]


class BeyondSpecStaticTrafficFlowControlFilter(ConstraintSpecFilter):
    """
    Filter for "BeyondSpec" stop sign - If there are any stop signs beyond the spec target (s > spec.s)
    this filter filters out action specs that cannot guarantee braking to velocity 0 before reaching the closest
    stop bar (beyond the spec target).
    Assumptions:  Actions where the stop_sign in between location and goal are filtered.
    """

    def __init__(self):
        super(BeyondSpecStaticTrafficFlowControlFilter, self).__init__(extend_short_action_specs=True)
        self.distances = BrakingDistances.create_braking_distances()

    def _get_first_stop_s(self, target_lane_frenet: GeneralizedFrenetSerretFrame, action_spec_s) -> int:
        """
        Returns the s value of the closest StaticTrafficFlow. Returns -1 is none exist
        :param target_lane_frenet:
        :param action_spec_s:
        :return:  Returns the s value of the closest StaticTrafficFlow. Returns -1 is none exist
        """
        traffic_control_s = MapUtils.get_static_traffic_flow_controls_s(target_lane_frenet)
        traffic_control_s = traffic_control_s[traffic_control_s >= action_spec_s]
        return -1 if len(traffic_control_s) == 0 else traffic_control_s[0]

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.array:
        """
        Checks if there are stop signs. Returns the `s` of the first (closest) stop-sign
        :param behavioral_state:
        :param action_spec:
        :return: The index of the end point
        """
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        stop_bar_s = self._get_first_stop_s(target_lane_frenet, action_spec.s)
        if stop_bar_s == -1:
            # no stop bars
            self._raise_true()
        if action_spec.s >= target_lane_frenet.s_max:
            self._raise_false()
        return np.array([stop_bar_s])

    def _target_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec,
                         points: np.ndarray) -> np.ndarray:
        """
        Braking distance from current velocity to 0
        The return value is a single scalar in a np.array (for consistency)
        :param behavioral_state:
        :param action_spec:
        :param points:
        :return:
        """
        # retrieve distances of static actions for the most aggressive level, since they have the shortest distances
        brake_dist = self.distances[FILTER_V_0_GRID.get_index(action_spec.v), FILTER_V_T_GRID.get_index(0)]
        return np.array([brake_dist])

    def _constraint_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec,
                             points: np.array) -> float:
        """
        Returns the distance from the action_spec goal to the closest stop sign.
        The return value is a single scalar in a np.array (for consistency)
        :param behavioral_state: The context  behavioral grid
        :param action_spec: ActionSpec to filter
        :param points: Current goal s
        :return: An array with a single point containing the distance from the action_spec goal to the closest stop sign
        """
        dist_to_points = points - action_spec.s
        assert dist_to_points[0] >= 0, 'Stop Sign must be ahead'
        return dist_to_points

    def _condition(self, target_values: np.array, constraints_values: np.array) -> bool:
        """
        Checks if braking distance from action_spec.v to 0 is smaller than the distance to stop sign
        :param target_values: braking distance from action_spec.v to 0
        :param constraints_values: the distance to stop sign
        :return: a single boolean
        """
        return target_values[0] < constraints_values[0]

