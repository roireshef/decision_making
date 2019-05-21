import copy
import numpy as np
import rte.python.profiler as prof
import six
from abc import ABCMeta, abstractmethod
from decision_making.src.exceptions import ConstraintFilterHaltWithValue
from decision_making.src.global_constants import EPS, WERLING_TIME_RESOLUTION, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS, FILTER_V_0_GRID, FILTER_V_T_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS, \
    LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition, StaticActionRecipe, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SA, FS_DX, FS_SX
from decision_making.src.planning.types import LAT_CELL
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils, BrakingDistances
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.utils.map_utils import MapUtils
from typing import List, Any


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
        # extract all relevant information for boundary conditions
        relative_lanes = np.array([spec.relative_lane for spec in action_specs])
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[lane] for lane in relative_lanes])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])

        # represent initial and terminal boundary conditions (for two Frenet axes s,d)
        constraints_s = np.concatenate((initial_fstates[:, :(FS_SA+1)], terminal_fstates[:, :(FS_SA+1)]), axis=1)
        constraints_d = np.concatenate((initial_fstates[:, FS_DX:], terminal_fstates[:, FS_DX:]), axis=1)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        T = np.array([spec.t for spec in action_specs])
        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T))

        # solve for s(t) and d(t)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d = QuinticPoly1D.zip_solve(A_inv, constraints_d)

        are_valid = []
        for poly_s, poly_d, t, lane, spec in zip(poly_coefs_s, poly_coefs_d, T, relative_lanes, action_specs):
            # TODO: in the future, consider leaving only a single action (for better "learnability")
            if spec.only_padding_mode:
                are_valid.append(True)
                continue

            # extract the relevant (cached) frenet frame per action according to the destination lane
            frenet_frame = behavioral_state.extended_lane_frames[lane]

            # if the action is static, there's a chance the 5th order polynomial is actually a degenerate one
            # (has lower degree), so we clip the first zero coefficients and send a polynomial with lower degree
            # TODO: This handling of polynomial coefficients being 5th or 4th order should happen in an inner context and get abstracted from this method
            first_non_zero = np.argmin(np.equal(poly_s, 0)) if isinstance(spec.recipe, StaticActionRecipe) else 0
            is_valid_in_frenet = KinematicUtils.filter_by_longitudinal_frenet_limits(poly_s[np.newaxis, first_non_zero:], np.array([t]),
                                                                                     LON_ACC_LIMITS, VELOCITY_LIMITS, frenet_frame.s_limits)[0]

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
            is_valid_in_cartesian = KinematicUtils.filter_by_cartesian_limits(cartesian_points[np.newaxis, ...],
                                                                 VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS)[0]

            are_valid.append(is_valid_in_cartesian)

        return are_valid


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

        # represent initial and terminal boundary conditions (for s axis)
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[cell[LAT_CELL]] for cell in relative_cells])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        constraints_s = np.concatenate((initial_fstates[:, :(FS_SA+1)], terminal_fstates[:, :(FS_SA+1)]), axis=1)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        T = np.array([spec.t for spec in action_specs])
        A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T))

        # solve for s(t)
        poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, constraints_s)

        are_valid = []
        for poly_s, t, cell, target in zip(poly_coefs_s, T, relative_cells, target_vehicles):
            if target is None:
                are_valid.append(True)
                continue

            target_fstate = behavioral_state.extended_lane_frames[cell[LAT_CELL]].convert_from_segment_state(
                target.dynamic_object.map_state.lane_fstate, target.dynamic_object.map_state.lane_id)
            target_poly_s, _ = KinematicUtils.create_linear_profile_polynomials(target_fstate)

            # minimal margin used in addition to headway (center-to-center of both objects)
            margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # validate distance keeping (on frenet longitudinal axis)
            is_safe = KinematicUtils.is_maintaining_distance(poly_s, target_poly_s, margin, SAFETY_HEADWAY, np.array([0, t]))

            are_valid.append(is_safe)

        # TODO: remove - for debug only
        had_dynmiacs = sum([isinstance(spec.recipe, DynamicActionRecipe) for spec in action_specs]) > 0
        valid_dynamics = sum([valid and isinstance(spec.recipe, DynamicActionRecipe) for spec, valid in zip(action_specs, are_valid)])

        return are_valid


@six.add_metaclass(ABCMeta)
class ConstraintSpecFilter(ActionSpecFilter):
    """
    An ActionSpecFilter which implements a predefined constraint.
     The filter is defined by:
     (1) x-axis (select_points method)
     (2) the function to test on these points (_target_function)
     (3) the constraint function (_constraint_function)
     (4) the condition function between target and constraints (_condition function)

     Usage:
        extend ConstraintSpecFilter class and implement the appropriate functions: (at least the four methods described
        above).
        To terminate the filter calculation use _raise_true/_raise_false  at any stage.
    """

    @abstractmethod
    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> Any:
        """
        selects relevant points from the action_spec (e.g., slow points)
        :param action_spec:
        :return:
        """
        pass

    @abstractmethod
    def _target_function(self, behavioral_state: BehavioralGridState,
                         action_spec: ActionSpec, points: Any) -> np.ndarray:
        """
        The definition of the function to be tested.
        :param behavioral_state:  A behavioral grid state
        :param action_spec: the action spec which to filter
        :return: the result of the target function as an np.ndarray
        """
        pass

    @abstractmethod
    def _constraint_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec, points: Any) \
            -> np.ndarray:
        """
        Defines the constraint function over points.

        :param behavioral_state:  A behavioral grid state
        :param action_spec: the action spec which to filter
        :return: the result of the constraint function as an np.ndarray
        """
        pass

    @abstractmethod
    def _condition(self, target_values, constraints_values) -> bool:
        """
        The test condition to apply on the results of target and constraint values
        :param target_values: the (externally calculated) target function
        :param constraints_values: the (externally calculated) constraint values
        :return:
        """
        pass

    def _raise_false(self):
        """
        Terminates the execution of the filter with a False value.
        No need to implement this method in your subtype
        :return: None
        """
        raise ConstraintFilterHaltWithValue(False)

    def _raise_true(self):
        """
        Terminates the execution of the filter with a True value.
        No need to implement this method in your subtype
        :return:
        """
        raise ConstraintFilterHaltWithValue(True)

    def _check_condition(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> bool:
        """
        Tests the condition defined by this filter
        No need to implement this method in your subtype

        :param behavioral_state:
        :param action_spec:
        :return:
        """
        points_under_test = self._select_points(behavioral_state, action_spec)
        return self._condition(self._target_function(behavioral_state, action_spec, points_under_test),
                               self._constraint_function(behavioral_state, action_spec, points_under_test))

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        The filter function
        No need to implement this method in your subtype
        :param action_specs:
        :param behavioral_state:
        :return:
        """
        mask = []
        for action_spec in action_specs:
            try:
                mask_value = self._check_condition(behavioral_state, action_spec)
            except ConstraintFilterHaltWithValue as e:
                mask_value = e.value
            mask.append(mask_value)
        return mask


@six.add_metaclass(ABCMeta)
class BeyondSpecConstraintFilter(ConstraintSpecFilter):
    """
    A ConstraintSpecFilter class with additional access to beyond spec indices.
    Also this adds an action_spec 'extension' for short action specs.
    """

    @staticmethod
    def extend_spec(spec: ActionSpec, min_action_time: float = MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) -> ActionSpec:
        """
        Extend Short action specs.
        :param spec: the original ActionSpec
        :param min_action_time: the minimal action time to which spec should be extended
        :return: An extended copy of the action_spec assuming a constant velocity
        """
        extended_spec = copy.copy(spec)
        extended_spec.s += (min_action_time - spec.t) * spec.v
        return extended_spec

    @staticmethod
    def extend_action_specs(action_specs: List[ActionSpec]) -> List[ActionSpec]:
        """
        Compensate for delays in super short action specs to increase filter efficiency.
        If we move with constant velocity by using very short actions, at some point we reveal that there is no ability
        to brake before a slow point, and it's too late to brake calm.
        Therefore, in case of actions shorter than 2 seconds we extend these actions by assuming constant velocity
        following the spec.
        :param action_specs:
        :return: A list of action specs with potentially extended copies of short ones
        """
        return [spec if spec.t >= MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
                else BeyondSpecConstraintFilter.extend_spec(spec) for spec in action_specs]

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Overridden  filter function
        :param action_specs:
        :param behavioral_state:
        :return:
        """
        return super(BeyondSpecConstraintFilter, self).filter(
            BeyondSpecConstraintFilter.extend_action_specs(action_specs),
            behavioral_state)

    def _get_beyond_spec_frenet_idxs(self, action_spec, behavioral_state) -> np.array:
        """
        Returns the indices beyond the action_spec goal
        :param action_spec:
        :param behavioral_state:
        :return: the indices beyond the action_spec goal
        """
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        if action_spec.s >= target_lane_frenet.s_max:
            self._raise_false()
        # get the Frenet point index near the goal action_spec.s
        spec_s_point_idx = target_lane_frenet.get_index_on_frame_from_s(np.array([action_spec.s]))[0][0]
        beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx + 1, len(target_lane_frenet.k), 4))
        return beyond_spec_frenet_idxs


class StaticTrafficFlowControlFilter(ActionSpecFilter):
    """
    Checks if there is a StaticTrafficFlowControl between ego and the goal
    Currently treats every 'StaticTrafficFlowControl' as a stop event.

    """

    @staticmethod
    def _has_stop_bar_until_goal(action_spec: ActionSpec, behavioral_state: BehavioralGridState):
        """
        Checks if there is a stop_bar between current ego location and the action_spec goal
        :param action_spec:
        :param behavioral_state:
        :return:
        """

        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        stop_bar_locations = MapUtils.get_static_traffic_flow_controls_s(target_lane_frenet)
        ego_location = behavioral_state.projected_ego_fstates[action_spec.relative_lane][FS_SX]

        return np.logical_and(ego_location <= stop_bar_locations, stop_bar_locations < action_spec.s).any()

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        return [not StaticTrafficFlowControlFilter._has_stop_bar_until_goal(action_spec, behavioral_state)
                for action_spec in action_specs]


class BeyondSpecStaticTrafficFlowControlFilter(BeyondSpecConstraintFilter):
    """
    Filter for "BeyondSpec" stop sign - If there are any stop signs beyond the spec target (s > spec.s)
    this filter filters out action specs that cannot guarantee braking to velocity 0 before reaching the closest
    stop bar (beyond the spec target).
    Assumptions:  Actions where the stop_sign in between location and goal are filtered.
    """

    def __init__(self):
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

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
        """
        Basically just checks if there are stop signs. Returns the `s` of the first stop-sign
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
        :param behavioral_state:
        :param action_spec:
        :param points:
        :return:
        """
        # retrieve distances of static actions for the most aggressive level, since they have the shortest distances
        brake_dist = self.distances[FILTER_V_0_GRID.get_index(action_spec.v), FILTER_V_T_GRID.get_index(0)]
        return brake_dist

    def _constraint_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec,
                             points: np.ndarray) -> np.ndarray:
        """
        The distance from first stop sign to current location
        :param behavioral_state:
        :param action_spec:
        :param points:
        :return: The distance from the goal to the
        """
        dist_to_points = points[0] - action_spec.s
        assert dist_to_points >= 0, 'Stop Sign must be ahead'
        return dist_to_points

    def _condition(self, target_values, constraints_values) -> bool:
        return target_values < constraints_values


