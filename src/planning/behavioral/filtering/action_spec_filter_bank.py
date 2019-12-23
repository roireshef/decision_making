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
    STOP_BAR_DISTANCE_IND, TIME_THRESHOLDS, SPEED_THRESHOLDS, TRAJECTORY_TIME_RESOLUTION
from decision_making.src.planning.behavioral.data_objects import ActionSpec, RelativeLongitudinalPosition, \
    AggressivenessLevel, RoadSignActionRecipe
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.behavioral.filtering.constraint_spec_filter import ConstraintSpecFilter
from decision_making.src.planning.behavioral.state.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.types import FS_DX, FS_SV, BoolArray, LIMIT_MAX, LIMIT_MIN, C_K, FS_SX, \
    FrenetTrajectories2D, CartesianExtendedTrajectories
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame, GFFType
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils, BrakingDistances
from decision_making.src.planning.utils.safety_utils import SafetyRSS
from decision_making.src.prediction.ego_aware_prediction.road_following_predictor import RoadFollowingPredictor
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
               ftrajectories: FrenetTrajectories2D, ctrajectories: CartesianExtendedTrajectories) -> List[bool]:
        return [spec.s <= behavioral_state.extended_lane_frames[spec.relative_lane].s_max for spec in action_specs]


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
        """ This is a temporary filter that replaces a more comprehensive test suite for safety w.r.t the target vehicle
         of a dynamic action or towards a leading vehicle in a static action. The condition under inspection is of
         maintaining the required safety-headway + constant safety-margin. Also used for action FOLLOW_ROAD_SIGN to
         verify ego maintains enough safety towards closest vehicle"""
        specs_by_rel_lane, indices_by_rel_lane = ActionSpecFilter._group_by_lane(action_specs)
        ego_length = behavioral_state.ego_state.size.length
        predictor = RoadFollowingPredictor(self._logger)  # TODO: replace by real predictor

        are_safe = np.zeros(len(action_specs)).astype(bool)

        for relative_lane, lane_frame in behavioral_state.extended_lane_frames.items():
            if len(indices_by_rel_lane[relative_lane]) == 0:
                continue
            front_vehicle = behavioral_state.road_occupancy_grid[(relative_lane, RelativeLongitudinalPosition.FRONT)][0].dynamic_object
            margin = 0.5 * (ego_length + front_vehicle.size.length) + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
            specs_t = np.array([spec.t for spec in specs_by_rel_lane[relative_lane]])
            trajectory_lengths = (np.maximum(specs_t, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
            max_trajectory_length = np.max(trajectory_lengths)

            target_fstate = behavioral_state.extended_lane_frames[relative_lane].convert_from_segment_state(
                front_vehicle.map_state.lane_fstate, front_vehicle.map_state.lane_id)
            obj_ftrajectory = predictor.predict_2d_frenet_states(
                target_fstate[np.newaxis], np.arange(max_trajectory_length)*TRAJECTORY_TIME_RESOLUTION)[0]
            ego_ftrajectories = ftrajectories[indices_by_rel_lane[relative_lane], :max_trajectory_length]

            safety_dist = SafetyRSS.get_lon_safe_dist(ego_ftrajectories, trajectory_lengths, SAFETY_HEADWAY,
                                                      obj_ftrajectory, SAFETY_HEADWAY, margin, self._logger)
            are_safe[indices_by_rel_lane[relative_lane]] = (safety_dist > 0).all(axis=1)

        return are_safe


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

