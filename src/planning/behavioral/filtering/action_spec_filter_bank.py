from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import List, Any

import numpy as np
import rte.python.profiler as prof
import six
from decision_making.src.global_constants import EPS, BP_ACTION_T_LIMITS, PARTIAL_GFF_END_PADDING, \
    VELOCITY_LIMITS, LON_ACC_LIMITS, FILTER_V_0_GRID, FILTER_V_T_GRID, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    SAFETY_HEADWAY, REL_LAT_ACC_LIMITS, LAT_ACC_LIMITS_LANE_CHANGE, \
    BP_LAT_ACC_STRICT_COEF, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, ZERO_SPEED, LAT_ACC_LIMITS_BY_K, \
    STOP_BAR_DISTANCE_IND, TIME_THRESHOLDS, SPEED_THRESHOLDS, TRAJECTORY_TIME_RESOLUTION, \
    LON_SAFETY_BACK_ACTOR_MAX_DECEL, LON_SAFETY_ACCEL_DURING_RESPONSE
from decision_making.src.planning.behavioral.data_objects import ActionSpec, RelativeLongitudinalPosition, \
    AggressivenessLevel, RoadSignActionRecipe, RelativeLane
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.behavioral.filtering.constraint_spec_filter import ConstraintSpecFilter
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState, SemanticGridCell
from decision_making.src.planning.types import FS_DX, FS_SV, BoolArray, LIMIT_MAX, LIMIT_MIN, C_K, FS_SX, \
    FrenetTrajectories2D, CartesianExtendedTrajectories, C_YAW, FrenetTrajectory2D
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, GFFType
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.braking_distances import BrakingDistances
from decision_making.src.utils.map_utils import MapUtils


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        return np.array([(action_spec and behavioral_state) is not None and ~np.isnan(action_spec.t)
                         for action_spec in action_specs])


class FilterForSLimit(ActionSpecFilter):
    """
    Check if target s value of action spec is inside s limit of the appropriate GFF.
    """
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        return np.array([spec.s <= behavioral_state.extended_lane_frames[spec.relative_lane].s_max for spec in action_specs])


class FilterForKinematics(ActionSpecFilter):
    def __init__(self, logger: Logger):
        self._logger = logger

    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        """
        Builds a baseline trajectory out of the action specs (terminal states) and validates them against:
            - max longitudinal position (available in the reference frame)
            - longitudinal velocity limits - both in Frenet (analytical) and Cartesian (by sampling)
            - longitudinal acceleration limits - both in Frenet (analytical) and Cartesian (by sampling)
            - lateral acceleration limits - in Cartesian (by sampling) - this isn't tested in Frenet, because Frenet frame
            conceptually "straightens" the road's shape.
        :param action_specs: list of action specs
        :param behavioral_state:
        :return: boolean array per action spec: True if a spec passed the filter
        """
        # Determine which actions are lane change actions
        action_spec_rel_lanes = [spec.relative_lane for spec in action_specs]
        lane_change_mask = behavioral_state.lane_change_state.get_lane_change_mask(action_spec_rel_lanes,
                                                                                   behavioral_state.extended_lane_frames)
        not_lane_change_mask = [not i for i in lane_change_mask]

        # for each point in the non-lane change trajectories,
        # compute the corresponding lateral acceleration (per point-wise curvature)
        baseline_curvatures = np.array([behavioral_state.extended_lane_frames[spec.relative_lane].get_curvature(ftrajectory[:, FS_SX])
                                        for spec, ftrajectory in zip(action_specs, ftrajectories)])
        nominal_abs_lat_acc_limits = KinematicUtils.get_lateral_acceleration_limit_by_curvature(baseline_curvatures, LAT_ACC_LIMITS_BY_K)

        # multiply the nominal lateral acceleration limits by the TP constant (acceleration limits may differ between
        # TP and BP), and duplicate the vector with negative sign to create boundaries for lateral acceleration
        two_sided_lat_acc_limits = BP_LAT_ACC_STRICT_COEF * \
                                   np.stack((-nominal_abs_lat_acc_limits, nominal_abs_lat_acc_limits), -1)

        self._log_debug_message(np.array(action_specs)[not_lane_change_mask].tolist(),
                                ctrajectories[not_lane_change_mask,:,C_K],
                                two_sided_lat_acc_limits[not_lane_change_mask])

        # Initialize conforms_limits array
        conforms_limits = np.full(ftrajectories.shape[0], False)

        # Filter for absolute limits for actions that are NOT part of a lane change
        conforms_limits[not_lane_change_mask] = KinematicUtils.filter_by_cartesian_limits(
            ctrajectories[not_lane_change_mask], VELOCITY_LIMITS, LON_ACC_LIMITS, two_sided_lat_acc_limits[not_lane_change_mask])

        # Deal with lane change actions if they exist
        if any(lane_change_mask):
            # Get the GFF that corresponds to the lane change's target (rel_lane depending on lane change state)
            target_gff = behavioral_state.lane_change_state.get_target_lane_gff(behavioral_state.extended_lane_frames)

            # Generate new limits based on lane change requirements
            num_lc_trajectories, num_lc_points, _ = ctrajectories[lane_change_mask].shape
            lane_change_max_lat_accel_limits = np.tile(
                LAT_ACC_LIMITS_LANE_CHANGE, reps=(num_lc_trajectories, num_lc_points)).reshape((num_lc_trajectories, num_lc_points, 2))

            # Filter for both relative and absolute limits for lane change actions
            conforms_limits[lane_change_mask] = np.logical_and(
                KinematicUtils.filter_by_relative_lateral_acceleration_limits(ftrajectories[lane_change_mask],
                                                                              ctrajectories[lane_change_mask],
                                                                              REL_LAT_ACC_LIMITS,
                                                                              nominal_abs_lat_acc_limits[lane_change_mask],
                                                                              target_gff),
                KinematicUtils.filter_by_cartesian_limits(
                    ctrajectories[lane_change_mask], VELOCITY_LIMITS, LON_ACC_LIMITS, lane_change_max_lat_accel_limits))

        return conforms_limits

    def _log_debug_message(self, action_specs: List[ActionSpec], curvatures: np.ndarray, acc_limits: np.ndarray):
        max_curvature_idxs = np.argmax(curvatures, axis=1)
        self._logger.debug("FilterForKinematics is working on %s action_specs with max curvature of %s "
                           "(acceleration limits %s)" % (len(action_specs),
                                                         curvatures[np.arange(len(curvatures)), max_curvature_idxs],
                                                         acc_limits[np.arange(len(acc_limits)), max_curvature_idxs]))


class FilterForLaneSpeedLimits(ActionSpecFilter):
    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        """
        Builds a baseline trajectory out of the action specs (terminal states) and validates them against:
            - max longitudinal position (available in the reference frame)
            - longitudinal velocity limits - both in Frenet (analytical) and Cartesian (by sampling)
            - longitudinal acceleration limits - both in Frenet (analytical) and Cartesian (by sampling)
            - lateral acceleration limits - in Cartesian (by sampling) - this isn't tested in Frenet, because Frenet frame
            conceptually "straightens" the road's shape.
        :param action_specs: list of action specs
        :param behavioral_state:
        :return: boolean array per action spec: True if a spec passed the filter
        """
        specs_by_rel_lane, indices_by_rel_lane = ActionSpecFilter._group_by_lane(action_specs)

        num_points = ftrajectories.shape[1]
        nominal_speeds = np.empty((len(action_specs), num_points), dtype=np.float)
        for relative_lane, lane_frame in behavioral_state.extended_lane_frames.items():
            if len(indices_by_rel_lane[relative_lane]) > 0:
                nominal_speeds[indices_by_rel_lane[relative_lane]] = self._pointwise_nominal_speed(
                    ftrajectories[indices_by_rel_lane[relative_lane]], lane_frame)

        T = np.array([spec.t for spec in action_specs])
        return KinematicUtils.filter_by_velocity_limit(ctrajectories, nominal_speeds, T)

    @staticmethod
    def _pointwise_nominal_speed(ftrajectories: np.ndarray, frenet: GeneralizedFrenetSerretFrame) -> np.ndarray:
        """
        :param ftrajectories: The frenet trajectories to which to calculate the nominal speeds
        :return: A matrix of (Trajectories x Time_samples) of lane-based nominal speeds (by e_v_nominal_speed).
        """

        # get the lane ids
        lane_ids_matrix = frenet.convert_to_segment_states(ftrajectories)[0]
        lane_to_nominal_speed = {lane_id: MapUtils.get_lane(lane_id).e_v_nominal_speed
                                 for lane_id in np.unique(lane_ids_matrix)}
        # creates an ndarray with the same shape as of `lane_ids_list`,
        # where each element is replaced by the maximal speed limit (according to lane)
        return np.vectorize(lane_to_nominal_speed.get)(lane_ids_matrix)


class FilterForSafetyTowardsTargetVehicle(ActionSpecFilter):
    def __init__(self, logger: Logger):
        self._logger = logger

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        """
        Calculate full RSS safety w.r.t. front, target and rear actors:
            Front same-lane actor is tested by longitudinal RSS until crossing the lanes border.
            Target-lane rear actor is tested by longitudinal RSS only in the point where ego touches the lanes border.
            Target actor is tested by longitudinal RSS starting from point where ego touches the lanes border.
        """
        specs_by_rel_lane, indices_by_rel_lane = ActionSpecFilter._group_by_lane(action_specs)
        are_safe = np.zeros(len(action_specs)).astype(bool)

        # loop over relative target lanes
        for target_lane, target_frenet in behavioral_state.extended_lane_frames.items():
            if len(indices_by_rel_lane[target_lane]) == 0:
                continue

            # build ego Frenet trajectories for the current target_lane
            specs_t = np.array([spec.t for spec in specs_by_rel_lane[target_lane]])
            trajectory_lengths = (np.maximum(specs_t, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
            max_trajectory_length = np.max(trajectory_lengths)
            ego_ftrajectories = ftrajectories[indices_by_rel_lane[target_lane], :max_trajectory_length]
            ego_ctrajectories = ctrajectories[indices_by_rel_lane[target_lane], :max_trajectory_length]

            # if there is an actor in parallel cell, all actions to relative_lane are not safe
            if target_lane != RelativeLane.SAME_LANE:
                parallel_cell = (target_lane, RelativeLongitudinalPosition.PARALLEL)
                if parallel_cell in behavioral_state.road_occupancy_grid and len(behavioral_state.road_occupancy_grid[parallel_cell]) > 0:
                    return np.zeros(len(action_specs)).astype(bool)

            time_ranges_dict = FilterForSafetyTowardsTargetVehicle._calculate_relevant_time_ranges(
                behavioral_state, ego_ftrajectories, ego_ctrajectories, target_lane, trajectory_lengths)

            are_safe[indices_by_rel_lane[target_lane]] = np.ones(ego_ftrajectories.shape[0]).astype(bool)
            for actor_cell, time_ranges in time_ranges_dict.items():
                safety_dist = FilterForSafetyTowardsTargetVehicle._check_safety_for_actor(
                    behavioral_state, ego_ftrajectories, actor_cell, time_ranges, self._logger)
                are_safe[indices_by_rel_lane[target_lane]] &= (safety_dist > 0)
                if not are_safe[indices_by_rel_lane[target_lane]].any():
                    break

        return are_safe

    @staticmethod
    def _calculate_relevant_time_ranges(behavioral_state: BehavioralGridState, ego_ftrajectories: FrenetTrajectories2D,
                                        ego_ctrajectories: CartesianExtendedTrajectories, target_lane: RelativeLane,
                                        trajectory_lengths: np.array):
        """
        For each relevant actor for safety (there are at most 3 such actors) calculate time ranges for all actions,
        where the safety should be tested:
           front source-lane actor is tested by longitudinal RSS until crossing the lanes border.
           target-lane rear actor is tested by longitudinal RSS only in the point where ego touches the lanes border.
           target actor is tested by longitudinal RSS starting from point where ego touches the lanes border.
        :param behavioral_state: behavioral grid state
        :param ego_ftrajectories: ego frenet trajectories relative to the target lane
        :param ego_ctrajectories: ego cartesian trajectories
        :param target_lane: the target lane
        :param trajectory_lengths: the real length of the trajectories according to spec.t
        :return: dictionary from actor's semantic cell to the relevant time ranges for all actions:
            time ranges are represented by a matrix Nx2, where N is the actions number, first column is from-time index,
            the second column is until-time index.
        """
        actor_cells = [(RelativeLane.SAME_LANE, RelativeLongitudinalPosition.FRONT),
                       (target_lane, RelativeLongitudinalPosition.FRONT),
                       (target_lane, RelativeLongitudinalPosition.REAR)]

        # in case of NOT lane change action, check safety for full trajectory w.r.t. the front actor
        if target_lane == RelativeLane.SAME_LANE:
            return {actor_cells[0]: np.c_[np.zeros(ego_ftrajectories.shape[0]).astype(int), trajectory_lengths]} \
                if actor_cells[0] in behavioral_state.road_occupancy_grid and \
                   len(behavioral_state.road_occupancy_grid[actor_cells[0]]) > 0 else {}

        # calculate distance from the target lane center to the target lane border
        target_frenet = behavioral_state.extended_lane_frames[target_lane]
        lane_id, lane_fstate = target_frenet.convert_to_segment_state(behavioral_state.projected_ego_fstates[target_lane])
        dist_to_lane_borders = MapUtils.get_dist_to_lane_borders(lane_id, lane_fstate[FS_SX])
        dist_to_lane_border = dist_to_lane_borders[0] if target_lane == RelativeLane.LEFT_LANE else dist_to_lane_borders[1]

        # for each ego_trajectory calculate the first time when some host's vertex touches the target lane
        rel_yaw = ego_ctrajectories[:, :, C_YAW] - target_frenet.get_yaw(ego_ftrajectories[:, :, FS_SX])
        l, w = behavioral_state.ego_state.size.length / 2, behavioral_state.ego_state.size.width / 2
        dx = ego_ftrajectories[:, :, FS_DX]
        # d of closest host's vertex from target lane
        touch_target_d = dx + l * np.sin(rel_yaw) - np.sign(dx) * w * np.cos(rel_yaw)
        # time of first touching target lane
        touch_target_lane_idxs = np.argmax(np.abs(touch_target_d) < dist_to_lane_border, axis=1)

        # For each trajectory calculate time range, in which the longitudinal safety should be tested:
        #    front source-lane actor is tested by longitudinal RSS until crossing the lanes border.
        #    target-lane rear actor is tested by longitudinal RSS only in the point where ego touches the lanes border.
        #    target actor is tested by longitudinal RSS starting from point where ego touches the lanes border.
        # front source-lane actor
        time_ranges_dict = dict()
        for cell in actor_cells:
            # skip cell if its actor does not exist
            if cell not in behavioral_state.road_occupancy_grid or len(behavioral_state.road_occupancy_grid[cell]) == 0:
                continue

            if cell[0] == RelativeLane.SAME_LANE:  # source-lane front actor
                time_ranges = np.c_[np.zeros(ego_ftrajectories.shape[0]).astype(int),
                                    np.argmax(np.abs(ego_ftrajectories[:, :, FS_DX]) < dist_to_lane_border, axis=1)]
            elif cell[1] == RelativeLongitudinalPosition.FRONT:  # front target-lane actor
                time_ranges = np.c_[np.maximum(0, touch_target_lane_idxs - 1),
                                    np.full(ego_ftrajectories.shape[0], ego_ftrajectories.shape[1])]
            else:  # rear target-lane actor
                time_ranges = np.c_[np.maximum(0, touch_target_lane_idxs - 1), touch_target_lane_idxs]

            time_ranges[:, 1] = np.minimum(trajectory_lengths, time_ranges[:, 1])
            time_ranges_dict[cell] = time_ranges

        return time_ranges_dict

    @staticmethod
    def _check_safety_for_actor(behavioral_state: BehavioralGridState, ego_ftrajectories: FrenetTrajectories2D,
                                actor_cell: SemanticGridCell, time_ranges: np.array, logger: Logger) -> np.array:
        """
        For each ego trajectory calculate longitudinal RSS safety distance: minimum on time of differences between
        actual distance and minimal safe distance. A trajectory is safe if its safety distance is positive.
        :param behavioral_state: behavioral grid state
        :param ego_ftrajectories: ego frenet trajectories w.r.t. the target lane
        :param actor_cell: semantic cell of the given actor
        :param time_ranges: matrix Nx2, where N is the actions number, first column is from-time index,
            the second column is until-time index
        :param logger:
        :return: array of size ego_ftrajectories.shape[0] of safety distances (minimum on time of differences between
        actual distance and minimal safe distance)
        """
        predictor = RoadFollowingPredictor(logger)
        actor = behavioral_state.road_occupancy_grid[actor_cell][0].dynamic_object
        ego_length = behavioral_state.ego_state.size.length

        min_from_time_idx = np.min(time_ranges[:, 0])
        max_till_time_idx = np.max(time_ranges[:, 1])

        # predict actor's Frenet trajectory
        margin = 0.5 * (ego_length + actor.size.length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT

        # align obj_trajectory to global time from TRAJECTORY_TIME_RESOLUTION grid
        time_offset = (-behavioral_state.ego_state.timestamp_in_sec/TRAJECTORY_TIME_RESOLUTION) % 1
        if 1 - time_offset < EPS:
            time_offset = 0

        # predict actor's trajectory
        # TODO: use real predictor
        target_fstate = behavioral_state.extended_lane_frames[actor_cell[0]].convert_from_segment_state(
            actor.map_state.lane_fstate, actor.map_state.lane_id)
        obj_ftrajectory = predictor.predict_2d_frenet_states(
            target_fstate[np.newaxis], np.arange(min_from_time_idx + time_offset, max_till_time_idx) * TRAJECTORY_TIME_RESOLUTION)[0]

        # calculate safety distances for all trajectories points
        safety_dist = FilterForSafetyTowardsTargetVehicle._get_lon_safe_dist(
            ego_ftrajectories[:, min_from_time_idx:max_till_time_idx], SAFETY_HEADWAY,
            obj_ftrajectory, SAFETY_HEADWAY, time_ranges[:, 0] - min_from_time_idx, time_ranges[:, 1] - min_from_time_idx,
            margin, actor_cell[1] == RelativeLongitudinalPosition.FRONT, logger)
        return np.min(safety_dist, axis=1)  # minimum on time

    @staticmethod
    def _get_lon_safe_dist(ego_trajectories: FrenetTrajectories2D, ego_response_time: float,
                           obj_trajectory: FrenetTrajectory2D, obj_response_time: float,
                           from_time_idx: np.array, till_time_idx: np.array, margin: float, front_actor: bool, logger: Logger,
                           ego_behind_max_brake: float = -LON_ACC_LIMITS[0],
                           ego_ahead_max_brake: float = LON_SAFETY_BACK_ACTOR_MAX_DECEL) -> np.array:
        """
        Calculate longitudinal safety between ego and another object for all timestamps.
        Longitudinal safety between two objects considers only their longitudinal data: longitude and longitudinal velocity.
        Longitudinal RSS formula considers distance reduction during the reaction time and difference between
        objects' braking distances.
        An object is defined safe if it's safe either longitudinally OR laterally.
        :param ego_trajectories: ego Frenet trajectories: 3D tensor of shape: traj_num x timestamps_num x 6
        :param ego_response_time: [sec] ego response time
        :param obj_trajectory: object's Frenet trajectory: 2D matrix: timestamps_num x 6
        :param obj_response_time: [sec] object's response time
        :param from_time_idx: array of first relevant time indices for each ego_trajectory
        :param till_time_idx: array of last relevant time indices for each ego_trajectory
        :param margin: [m] cars' lengths half sum
        :param front_actor: True if the actor is in front of ego
        :param logger:
        :param ego_behind_max_brake: [m/s^2] maximal deceleration of both objects for front actor
        :param ego_ahead_max_brake: [m/s^2] maximal deceleration of both objects for back actor
        :return: normalized longitudinal safety distance per timestamp. 2D matrix shape: traj_num x timestamps_num
        """
        # extract the relevant longitudinal data from the trajectories
        ego_lon, ego_vel, ego_lat = ego_trajectories[..., FS_SX], ego_trajectories[..., FS_SV], ego_trajectories[..., FS_DX]
        ego_trajectories_s = ego_trajectories[..., :FS_DX]
        if ego_trajectories.ndim == 2:  # single ego trajectory
            ego_lon = ego_lon[np.newaxis]
            ego_vel = ego_vel[np.newaxis]
            ego_trajectories_s = ego_trajectories_s[np.newaxis]

        obj_lon, obj_vel, obj_lat = obj_trajectory[:, FS_SX], obj_trajectory[:, FS_SV], obj_trajectory[:, FS_DX]

        # set infinite s for irrelevant time indices
        irrelevant_s = -np.inf if front_actor else np.inf
        for ego_s, from_i, till_i in zip(ego_lon, from_time_idx, till_time_idx):
            ego_s[till_i:] = irrelevant_s
            ego_s[:from_i] = irrelevant_s

        if front_actor:
            # extrapolate ego trajectories ego_response_time seconds beyond their end state
            dt = TRAJECTORY_TIME_RESOLUTION
            predictor = RoadFollowingPredictor(logger)
            extrapolated_times = np.arange(dt, ego_response_time + EPS, dt)
            last_ego_states_s = ego_trajectories_s[range(ego_trajectories_s.shape[0]), till_time_idx - 1]
            ego_extrapolation = predictor.predict_1d_frenet_states(last_ego_states_s, extrapolated_times)
            delay_shift = ego_extrapolation.shape[1]

            ext_ego_lon = np.concatenate((ego_lon, np.zeros_like(ego_extrapolation[..., FS_SX])), axis=1)
            ext_ego_vel = np.concatenate((ego_vel, np.zeros_like(ego_extrapolation[..., FS_SV])), axis=1)
            for i in range(delay_shift):
                ext_ego_lon[range(ext_ego_lon.shape[0]), till_time_idx + i] = ego_extrapolation[:, i, FS_SX]
                ext_ego_vel[range(ext_ego_vel.shape[0]), till_time_idx + i] = ego_extrapolation[:, i, FS_SV]

            # we assume ego continues its trajectory during its reaction time, so we compute the difference between
            # object's braking distance from any moment and delayed braking distance of ego
            braking_distances_diff = np.maximum(0, ext_ego_vel[:, delay_shift:] ** 2 - obj_vel ** 2) / (2 * ego_behind_max_brake)
            marginal_safe_dist = obj_lon - ext_ego_lon[:, delay_shift:] - braking_distances_diff - margin

        else:
            # The worst-case velocity of the rear object (either ego or another object) may increase during its reaction
            # time, since it may accelerate before it starts to brake.
            obj_vel_after_reaction_time = obj_vel + obj_response_time * LON_SAFETY_ACCEL_DURING_RESPONSE

            # longitudinal RSS formula considers distance reduction during the reaction time and difference between
            # objects' braking distances
            obj_acceleration_dist = 0.5 * LON_SAFETY_ACCEL_DURING_RESPONSE * obj_response_time ** 2
            min_safe_dist = np.maximum((obj_vel_after_reaction_time ** 2 - ego_vel ** 2) / (2 * ego_ahead_max_brake), 0) + \
                            (obj_vel * obj_response_time + obj_acceleration_dist) + margin
            marginal_safe_dist = ego_lon - obj_lon - min_safe_dist

        return marginal_safe_dist if ego_trajectories.ndim > 2 else marginal_safe_dist[0]


class StaticTrafficFlowControlFilter(ActionSpecFilter):
    """
    Checks if there is a StopBar between ego and the goal
    If no bar is found, action is invalidated.
    """
    @staticmethod
    def _has_stop_bar_until_goal(action_spec: ActionSpec, behavioral_state: BehavioralGridState) -> bool:
        """
        Checks if there is a stop_bar between current ego location and the action_spec goal
        :param action_spec: the action_spec to be considered
        :param behavioral_state: BehavioralGridState in context
        :return: if there is a stop_bar between current ego location and the action_spec goal
        """
        closest_TCB_ant_its_distance = behavioral_state.get_closest_stop_bar(action_spec.relative_lane)
        return closest_TCB_ant_its_distance is not None and \
               closest_TCB_ant_its_distance[STOP_BAR_DISTANCE_IND] < action_spec.s

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        return np.array([not StaticTrafficFlowControlFilter._has_stop_bar_until_goal(action_spec, behavioral_state)
                         for action_spec in action_specs])


@six.add_metaclass(ABCMeta)
class BeyondSpecBrakingFilter(ConstraintSpecFilter):
    """
    An ActionSpecFilter which implements a predefined constraint.
     The filter is defined by:
     (1) x-axis (select_points method)
     (2) the function to test on these points (_target_function)
     (3) the constraint function (_constraint_function)
     (4) the condition function between target and constraints (_condition function)

     Usage:
        extend ConstraintSpecFilter class and implement the following functions:
            _target_function
            _constraint_function
            _condition
        To terminate the filter calculation use _raise_true/_raise_false  at any stage.
    """
    def __init__(self, aggresiveness_level: AggressivenessLevel = AggressivenessLevel.CALM):
        super(BeyondSpecBrakingFilter, self).__init__()
        self.braking_distances = BrakingDistances.create_braking_distances(aggressiveness_level=aggresiveness_level.value)

    @abstractmethod
    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> [np.array, np.array]:
        """
         selects relevant points (or other information) based on the action_spec (e.g., select all points in the
          trajectory defined by action_spec that require slowing down due to curvature).
         This method is not restricted to returns a specific type, and can be used to pass any relevant information to
         the target and constraint functions.
        :param behavioral_state:  The behavioral_state as the context of this filtering
        :param action_spec:  The action spec in question
        :return: Any type that should be used by the _target_function and _constraint_function
        """
        pass

    def _target_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec, points: Any) -> np.array:
        """
        Calculate the braking distances from action_spec.s to the selected points, using static
        actions with STANDARD aggressiveness level.
        :param behavioral_state:  A behavioral grid state
        :param action_spec: the action spec which to filter
        :return: array of braking distances from action_spec.s to the selected points
        """
        return self._braking_distances(action_spec, points[1])

    def _constraint_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec, points: Any) -> np.array:
        """
        Calculate the actual distances from action_spec.s to the selected points.
        :param behavioral_state:  A behavioral grid state at the context of filtering
        :param action_spec: the action spec which to filter
        :return: array of actual distances from action_spec.s to the selected points
        """
        return self._actual_distances(action_spec, points[0])

    def _condition(self, target_values: np.array, constraints_values: np.array) -> bool:
        """
        The test condition to apply on the results of target and constraint values
        :param target_values: the (externally calculated) target function
        :param constraints_values: the (externally calculated) constraint values
        :return: a single boolean indicating whether this action_spec should be filtered or not
        """
        return np.all(target_values < constraints_values)

    def _braking_distances(self, action_spec: ActionSpec, slow_points_velocity_limits: np.array) -> np.ndarray:
        """
        The braking distance required by using the CALM aggressiveness level to brake from the spec velocity
        to the given points' velocity limits.
        :param action_spec:
        :param slow_points_velocity_limits: velocity limits of selected points
        :return: braking distances from the spec velocity to the given velocities
        """
        return self.braking_distances[FILTER_V_0_GRID.get_index(action_spec.v),
                                      FILTER_V_T_GRID.get_indices(slow_points_velocity_limits)]

    def _actual_distances(self, action_spec: ActionSpec, slow_points_s: np.array) -> np.ndarray:
        """
        The distance from current points to the 'slow points'
        :param action_spec:
        :param slow_points_s: s coordinates of selected points
        :return: distances from the spec's endpoint (spec.s) to the given points
        """
        return slow_points_s - action_spec.s


class BeyondSpecStaticTrafficFlowControlFilter(BeyondSpecBrakingFilter):
    """
    Filter for "BeyondSpec" stop sign - If there are any stop signs beyond the spec target (s > spec.s)
    this filter filters out action specs that cannot guarantee braking to velocity 0 before reaching the closest
    stop bar (beyond the spec target).
    Assumptions:  Actions where the stop_sign in between location and goal are filtered.
    """

    def __init__(self):
        # Using a STANDARD aggressiveness. If we try to use CALM, then when testing FOLLOW_VEHICLE and failing,
        # a CALM FOLLOW_ROAD_SIGN action will not be ready yet.
        # TODO a better solution may be to calculate the BeyondSpecStaticTrafficFlowControlFilter relative to the
        #  start of the action, or 1 BP cycle later, instead of relative to the end of the action.
        super().__init__(aggresiveness_level=AggressivenessLevel.STANDARD)

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> [np.ndarray, np.ndarray]:
        """
        Checks if there are stop signs. Returns the `s` of the first (closest) stop-sign
        :param behavioral_state: behavioral grid state
        :param action_spec: action specification
        :return: s of the next stop bar, target velocity (zero)
        """
        closest_TCB_and_its_distance = behavioral_state.get_closest_stop_bar(action_spec.relative_lane)
        if closest_TCB_and_its_distance is None:  # no stop bars
            self._raise_true()
        return np.array([closest_TCB_and_its_distance[STOP_BAR_DISTANCE_IND]]), np.array([0])


class BeyondSpecCurvatureFilter(BeyondSpecBrakingFilter):
    """
    Checks if it is possible to brake from the action's goal to all nominal points beyond action_spec.s
    (until the end of the frenet frame), without violation of the lateral acceleration limits.
    the following edge cases are treated by raise_true/false:
    (A) action_spec.v == 0 (return True)
    (B) there are no selected (slow) points (return True)
    """
    def __init__(self):
        super().__init__()

    def _get_velocity_limits_of_points(self, action_spec: ActionSpec, frenet_frame: GeneralizedFrenetSerretFrame) -> \
            [np.array, np.array]:
        """
        Returns s and velocity limits of the points that needs to slow down
        :param action_spec:
        :param frenet_frame:
        :return:
        """
        # get the worst case braking distance from spec.v to 0
        max_braking_distance = self.braking_distances[FILTER_V_0_GRID.get_index(action_spec.v), FILTER_V_T_GRID.get_index(0)]
        max_relevant_s = min(action_spec.s + max_braking_distance, frenet_frame.s_max)

        # get the Frenet point indices near spec.s and near the worst case braking distance beyond spec.s
        # beyond_spec_range[0] must be BEYOND spec.s because the actual distances from spec.s to the
        # selected points have to be positive.
        beyond_spec_range = frenet_frame.get_closest_index_on_frame(np.array([action_spec.s, max_relevant_s]))[0] + 1

        # get s for all points in the range
        points_s = frenet_frame.get_s_from_index_on_frame(np.arange(beyond_spec_range[LIMIT_MIN], beyond_spec_range[LIMIT_MAX]), 0)

        # get curvatures for all points in the range
        curvatures = np.maximum(np.abs(frenet_frame.k[beyond_spec_range[LIMIT_MIN]:beyond_spec_range[LIMIT_MAX], 0]), EPS)

        # for each point in the trajectories, compute the corresponding lateral acceleration (per point-wise curvature)
        lat_acc_limits = KinematicUtils.get_lateral_acceleration_limit_by_curvature(curvatures, LAT_ACC_LIMITS_BY_K)

        points_velocity_limits = np.sqrt(BP_LAT_ACC_STRICT_COEF * lat_acc_limits / curvatures)

        return points_s, points_velocity_limits

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> [np.array, np.array]:
        """
        Finds 'slow' points indices
        :param behavioral_state:
        :param action_spec:
        :return:
        """
        if action_spec.v == 0:
            # When spec velocity is 0, there is no problem to "brake" beyond spec. In this case the filter returns True.
            self._raise_true()
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        beyond_spec_s, points_velocity_limits = self._get_velocity_limits_of_points(action_spec, target_lane_frenet)
        slow_points = np.where(points_velocity_limits < action_spec.v)[0]  # points that require braking after spec
        # set edge case
        if len(slow_points) == 0:
            self._raise_true()
        return beyond_spec_s[slow_points], points_velocity_limits[slow_points]


class BeyondSpecSpeedLimitFilter(BeyondSpecBrakingFilter):
    """
    Checks if the speed limit will be exceeded.
    This filter assumes that the CALM aggressiveness will be used, and only checks the points that are before
    the worst case stopping distance.
    The braking distances are calculated upon initialization and cached.

    If the upcoming speeds are greater or equal than the target velocity or if the worst case braking distance is 0,
    this filter will raise true.
    """

    def __init__(self):
        super(BeyondSpecSpeedLimitFilter, self).__init__()

    def _get_upcoming_speed_limits(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> (
    int, float):
        """
        Finds speed limits of the lanes ahead
        :param behavioral_state
        :param action_spec:
        :return: tuple of (Frenet indices of start points of lanes ahead, speed limits at those indices)
        """
        # get the lane the action_spec wants to drive in
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]

        # get all subsegments in current GFF and get the ones that contain points ahead of the action_spec.s
        subsegments = target_lane_frenet.segments
        subsegments_ahead = [subsegment for subsegment in subsegments if subsegment.e_i_SStart > action_spec.s]

        # if no lane segments ahead, there will be no speed limit changes
        if len(subsegments_ahead) == 0:
            self._raise_true()

        # get lane initial s points and lane ids from subsegments ahead
        lanes_s_start_ahead = [subsegment.e_i_SStart for subsegment in subsegments_ahead]
        lane_ids_ahead = [subsegment.e_i_SegmentID for subsegment in subsegments_ahead]

        # find speed limits of points at the start of the lane (should be in mps)
        speed_limits = [MapUtils.get_lane(lane_id).e_v_nominal_speed for lane_id in lane_ids_ahead]

        return (np.array(lanes_s_start_ahead), np.array(speed_limits))

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> any:
        """
        Find points with speed limit slower than the spec velocity
        :param behavioral_state:
        :param action_spec:
        :return: points that require braking after the spec
        """

        # skip checking speed limits if the vehicle will be stopped
        if action_spec.v == 0:
            self._raise_true()

        # get speed limits after the action_spec.s
        lane_s_start_ahead, speed_limits = self._get_upcoming_speed_limits(behavioral_state, action_spec)
        # find points that require braking after spec
        slow_points = np.where(np.array(speed_limits) < action_spec.v)[0]

        # skip filtering if there are no points that require slowing down
        if len(slow_points) == 0:
            self._raise_true()

        return lane_s_start_ahead[slow_points], speed_limits[slow_points]


class BeyondSpecPartialGffFilter(BeyondSpecBrakingFilter):
    """
    Checks if an action will make the vehicle unable to stop before the end of a Partial GFF.

    This filter assumes that the CALM aggressiveness will be used, and only checks the points that are before
    the worst case stopping distance.
    The braking distances are calculated upon initialization and cached.

    The filter will return True if the GFF is not a Partial or AugmentedPartial GFF.
    """

    def __init__(self):
        super(BeyondSpecPartialGffFilter, self).__init__()

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> any:
        """
        Finds the point some distance before the end of a partial GFF. This distance is determined by PARTIAL_GFF_END_PADDING.
        :param behavioral_state:
        :param action_spec:
        :return: points that require braking after the spec
        """
        target_gff = behavioral_state.extended_lane_frames[action_spec.relative_lane]

        # skip checking if the GFF is not a partial GFF
        if target_gff.gff_type not in [GFFType.Partial, GFFType.AugmentedPartial]:
            self._raise_true()

        # pad end of GFF with PARTIAL_GFF_END_PADDING as the host should not be at the very end of the Partial GFF
        gff_end_s = target_gff.s_max - PARTIAL_GFF_END_PADDING \
            if target_gff.s_max - PARTIAL_GFF_END_PADDING > 0 \
            else target_gff.s_max

        # skip checking for end of Partial GFF if vehicle will be stopped before the padded end
        if action_spec.v == ZERO_SPEED and action_spec.s < gff_end_s:
            self._raise_true()

        # must be able to achieve 0 velocity before the end of the GFF
        return np.array([gff_end_s]), np.array([ZERO_SPEED])


class FilterStopActionIfTooSoonByTime(ActionSpecFilter):
    @staticmethod
    def _is_time_to_stop(action_spec: ActionSpec, behavioral_state: BehavioralGridState) -> bool:
        """
        Checks whether a stop action should start, or if it is too early for it to act.
        The allowed values are defined by the system requirements. See R14 in UltraCruise use cases
        :param action_spec: the action spec to test
        :param behavioral_state: state of the world
        :return: True if the action should start, False otherwise
        """
        assert max(TIME_THRESHOLDS) < BP_ACTION_T_LIMITS[1]  # sanity check
        ego_speed = behavioral_state.projected_ego_fstates[action_spec.recipe.relative_lane][FS_SV]
        # lookup maximal time threshold
        indices = np.where(SPEED_THRESHOLDS > ego_speed)[0]
        maximal_stop_time = TIME_THRESHOLDS[indices[0]] \
            if len(indices) > 0 else BP_ACTION_T_LIMITS[1]

        return action_spec.t < maximal_stop_time

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> BoolArray:
        return np.array([(not isinstance(action_spec.recipe, RoadSignActionRecipe)) or
                         FilterStopActionIfTooSoonByTime._is_time_to_stop(action_spec, behavioral_state)
                         if (action_spec is not None) else False for action_spec in action_specs])

