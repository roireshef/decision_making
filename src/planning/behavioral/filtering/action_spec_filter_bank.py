import numpy as np
import rte.python.profiler as prof
import six
from abc import ABCMeta, abstractmethod
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, EPS, BP_ACTION_T_LIMITS, \
    TRAJECTORY_TIME_RESOLUTION
from decision_making.src.global_constants import VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, \
    FILTER_V_0_GRID, FILTER_V_T_GRID, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, \
    BP_LAT_ACC_STRICT_COEF
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition, RelativeLane
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.behavioral.filtering.constraint_spec_filter import ConstraintSpecFilter
from decision_making.src.planning.types import FS_DX, FS_SX, C_V
from decision_making.src.planning.types import LAT_CELL
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils, BrakingDistances
from decision_making.src.utils.map_utils import MapUtils
from decision_making.src.planning.behavioral.data_objects import AggressivenessLevel
from typing import List
from typing import List, Union, Any


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None and ~np.isnan(action_spec.t) for action_spec in action_specs]


class FilterForKinematics(ActionSpecFilter):
    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Builds a baseline trajectory out of the action specs (terminal states) and validates them against:
            - max longitudinal position (available in the reference frame)
            - longitudinal velocity limits - both in Frenet (analytical) and Cartesian (by sampling)
            - longitudinal acceleration limits - both in Frenet (analytical) and Cartesian (by sampling)
            - lateral acceleration limits - in Cartesian (by sampling) - this isn't tested in Frenet, because Frenet frame
            conceptually "straightens" the road's shape.
        :param action_specs: list of action specs
        :param behavioral_state:
        :return: boolean list per action spec: True if a spec passed the filter
        """
        _, ctrajectories = self._build_trajectories(action_specs, behavioral_state)

        return list(KinematicUtils.filter_by_cartesian_limits(
            ctrajectories, VELOCITY_LIMITS, LON_ACC_LIMITS, BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS))


class FilterForLaneSpeedLimits(ActionSpecFilter):
    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Builds a baseline trajectory out of the action specs (terminal states) and validates them against:
            - max longitudinal position (available in the reference frame)
            - longitudinal velocity limits - both in Frenet (analytical) and Cartesian (by sampling)
            - longitudinal acceleration limits - both in Frenet (analytical) and Cartesian (by sampling)
            - lateral acceleration limits - in Cartesian (by sampling) - this isn't tested in Frenet, because Frenet frame
            conceptually "straightens" the road's shape.
        :param action_specs: list of action specs
        :param behavioral_state:
        :return: boolean list per action spec: True if a spec passed the filter
        """
        ftrajectories, ctrajectories = self._build_trajectories(action_specs, behavioral_state)

        specs_by_rel_lane, indices_by_rel_lane = ActionSpecFilter._group_by_lane(action_specs)

        num_points = ftrajectories.shape[1]
        nominal_speeds = np.empty((len(action_specs), num_points), dtype=np.float)
        for relative_lane, lane_frame in behavioral_state.extended_lane_frames.items():
            if len(indices_by_rel_lane[relative_lane]) > 0:
                nominal_speeds[indices_by_rel_lane[relative_lane]] = self._pointwise_nominal_speed(
                    ftrajectories[indices_by_rel_lane[relative_lane]], lane_frame)

        return list(KinematicUtils.filter_by_nominal_velocity(ctrajectories, nominal_speeds))

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
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """ This is a temporary filter that replaces a more comprehensive test suite for safety w.r.t the target vehicle
         of a dynamic action or towards a leading vehicle in a static action. The condition under inspection is of
         maintaining the required safety-headway + constant safety-margin"""
        # Extract the grid cell relevant for that action (for static actions it takes the front cell's actor,
        # so this filter is actually applied to static actions as well). Then query the cell for the target vehicle
        relative_cells = [(spec.recipe.relative_lane,
                           spec.recipe.relative_lon if isinstance(spec.recipe, DynamicActionRecipe) else RelativeLongitudinalPosition.FRONT)
                          for spec in action_specs]
        target_vehicles = [behavioral_state.road_occupancy_grid[cell][0]
                           if len(behavioral_state.road_occupancy_grid[cell]) > 0 else None
                           for cell in relative_cells]
        T = np.array([spec.t for spec in action_specs])

        # represent initial and terminal boundary conditions (for s axis)
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[cell[LAT_CELL]] for cell in relative_cells])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])

        # create boolean arrays indicating whether the specs are in tracking mode
        padding_mode = np.array([spec.only_padding_mode for spec in action_specs])

        poly_coefs_s, _ = KinematicUtils.calc_poly_coefs(T, initial_fstates[:, :FS_DX], terminal_fstates[:, :FS_DX], padding_mode)

        are_valid = []
        for poly_s, t, cell, target in zip(poly_coefs_s, T, relative_cells, target_vehicles):
            if target is None:
                are_valid.append(True)
                continue

            target_fstate = behavioral_state.extended_lane_frames[cell[LAT_CELL]].convert_from_segment_state(
                target.dynamic_object.map_state.lane_fstate, target.dynamic_object.map_state.lane_id)
            target_poly_s, _ = KinematicUtils.create_linear_profile_polynomial_pair(target_fstate)

            # minimal margin used in addition to headway (center-to-center of both objects)
            margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # validate distance keeping (on frenet longitudinal axis)
            is_safe = KinematicUtils.is_maintaining_distance(poly_s, target_poly_s, margin, SAFETY_HEADWAY,
                                                             np.array([0, t]))

            are_valid.append(is_safe)

        return are_valid


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
    def __init__(self):
        super(BeyondSpecBrakingFilter, self).__init__()
        self.braking_distances = BrakingDistances.create_braking_distances()

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
        super().__init__()

    def _get_first_stop_s(self, target_lane_frenet: GeneralizedFrenetSerretFrame, action_spec_s) -> Union[float, None]:
        """
        Returns the s value of the closest StaticTrafficFlow. Returns -1 is none exist
        :param target_lane_frenet:
        :param action_spec_s:
        :return:  Returns the s value of the closest StaticTrafficFlow. Returns -1 is none exist
        """
        traffic_control_s = MapUtils.get_static_traffic_flow_controls_s(target_lane_frenet)
        traffic_control_s = traffic_control_s[traffic_control_s >= action_spec_s]
        return traffic_control_s[0] if len(traffic_control_s) > 0 else None

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> [np.ndarray, np.ndarray]:
        """
        Checks if there are stop signs. Returns the `s` of the first (closest) stop-sign
        :param behavioral_state:
        :param action_spec:
        :return: The index of the end point
        """
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        stop_bar_s = self._get_first_stop_s(target_lane_frenet, action_spec.s)
        if stop_bar_s is None:  # no stop bars
            self._raise_true()
        return np.array([stop_bar_s]), np.array([0])


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
        points_s = frenet_frame.get_s_from_index_on_frame(np.array(range(beyond_spec_range[0], beyond_spec_range[1])), 0)
        # get velocity limits for all points in the range
        curvatures = np.maximum(np.abs(frenet_frame._k_max[beyond_spec_range[0]:beyond_spec_range[1]]), EPS)
        points_velocity_limits = np.sqrt(BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS[1] / curvatures)
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
    This filter assumes that the STANDARD aggressiveness will be used, and only checks the points that are before
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
        :return: tuple of (Frenet indicies of start points of lanes ahead, speed limits at those indicies)
        """
        # get the lane the action_spec wants to drive in
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]

        # get all subsegments in current GFF and get the ones that contain points ahead of the action_spec.s
        subsegments = target_lane_frenet.segments
        subsegments_ahead = [subsegment for subsegment in subsegments if
                             subsegment.e_i_SStart > action_spec.s]


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

