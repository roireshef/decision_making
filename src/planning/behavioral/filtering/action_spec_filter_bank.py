import copy

import numpy as np
import rte.python.profiler as prof
import six
from abc import ABCMeta, abstractmethod
from decision_making.src.exceptions import ConstraintFilterHaltWithValue
from decision_making.src.global_constants import EPS, WERLING_TIME_RESOLUTION, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS
from decision_making.src.global_constants import LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    FILTER_V_0_GRID, FILTER_V_T_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS, BP_LAT_ACC_STRICT_COEF, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.global_constants import SAFETY_HEADWAY
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition, StaticActionRecipe, AggressivenessLevel, RelativeLane
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SA, FS_DX
from decision_making.src.planning.types import FS_SX, FS_SV, LAT_CELL
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.utils.map_utils import MapUtils
from typing import List, Any, Union


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None and ~np.isnan(action_spec.t) for action_spec in
                action_specs]


class FilterForKinematics(ActionSpecFilter):
    @prof.ProfileFunction()
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """ Builds a baseline trajectory out of the action specs (terminal states) and validates them against:
            - max longitudinal position (available in the reference frame)
            - longitudinal velocity limits - both in Frenet (analytical) and Cartesian (by sampling)
            - longitudinal acceleration limits - both in Frenet (analytical) and Cartesian (by sampling)
            - lateral acceleration limits - in Cartesian (by sampling) - this isn't tested in Frenet, because Frenet frame
            conceptually "straightens" the road's shape.
         """
        # extract all relevant information for boundary conditions
        initial_fstates = np.array(
            [behavioral_state.projected_ego_fstates[spec.relative_lane] for spec in action_specs])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        T = np.array([spec.t for spec in action_specs])

        # creare boolean arrays indicating whether the specs are in tracking mode
        in_track_mode = np.array([spec.in_track_mode for spec in action_specs])
        no_track_mode = np.logical_not(in_track_mode)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T[no_track_mode])

        # represent initial and terminal boundary conditions (for two Frenet axes s,d) for non-tracking specs
        constraints_s = np.concatenate(
            (initial_fstates[no_track_mode, :FS_DX], terminal_fstates[no_track_mode, :FS_DX]), axis=1)
        constraints_d = np.concatenate(
            (initial_fstates[no_track_mode, FS_DX:], terminal_fstates[no_track_mode, FS_DX:]), axis=1)

        # solve for s(t) and d(t)
        poly_coefs_s, poly_coefs_d = np.zeros((len(action_specs), 6)), np.zeros((len(action_specs), 6))
        poly_coefs_s[no_track_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d[no_track_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_d)
        # in tracking mode (constant velocity) the s polynomials have only two non-zero coefficients
        poly_coefs_s[in_track_mode, 4:] = np.c_[
            initial_fstates[in_track_mode, FS_SV], initial_fstates[in_track_mode, FS_SX]]

        are_valid = []
        for poly_s, poly_d, t, spec in zip(poly_coefs_s, poly_coefs_d, T, action_specs):
            # TODO: in the future, consider leaving only a single action (for better "learnability")

            # extract the relevant (cached) frenet frame per action according to the destination lane
            frenet_frame = behavioral_state.extended_lane_frames[spec.relative_lane]

            if not spec.in_track_mode:
                # if the action is static, there's a chance the 5th order polynomial is actually a degnerate one
                # (has lower degree), so we clip the first zero coefficients and send a polynomial with lower degree
                first_non_zero = np.argmin(np.equal(poly_s, 0)) if isinstance(spec.recipe, StaticActionRecipe) else 0
                is_valid_in_frenet = KinematicUtils.filter_by_longitudinal_frenet_limits(
                    poly_s[np.newaxis, first_non_zero:], np.array([t]), LON_ACC_LIMITS, VELOCITY_LIMITS,
                    frenet_frame.s_limits)[0]

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
                                                                              VELOCITY_LIMITS, LON_ACC_LIMITS,
                                                                              BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS)[
                0]

            are_valid.append(is_valid_in_cartesian)

        # TODO: remove it
        if not any(are_valid):
            ego_fstate = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]
            frenet = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
            init_idx = frenet.get_index_on_frame_from_s(ego_fstate[:1])[0][0]
            print('ERROR in BP %.3f: ego_fstate=%s; nominal_k=%s' %
                  (behavioral_state.ego_state.timestamp_in_sec, NumpyUtils.str_log(ego_fstate),
                   frenet.k[init_idx:init_idx + 10, 0]))

        return are_valid


class FilterForSafetyTowardsTargetVehicle(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """ This is a temporary filter that replaces a more comprehensive test suite for safety w.r.t the target vehicle
         of a dynamic action or towards a leading vehicle in a static action. The condition under inspection is of
         maintaining the required safety-headway + constant safety-margin"""
        # Extract the grid cell relevant for that action (for static actions it takes the front cell's actor,
        # so this filter is actually applied to static actions as well). Then query the cell for the target vehicle
        relative_cells = [(spec.recipe.relative_lane,
                           spec.recipe.relative_lon if isinstance(spec.recipe,
                                                                  DynamicActionRecipe) else RelativeLongitudinalPosition.FRONT)
                          for spec in action_specs]
        target_vehicles = [behavioral_state.road_occupancy_grid[cell][0]
                           if len(behavioral_state.road_occupancy_grid[cell]) > 0 else None
                           for cell in relative_cells]

        # represent initial and terminal boundary conditions (for s axis)
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[cell[LAT_CELL]] for cell in relative_cells])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        constraints_s = np.concatenate((initial_fstates[:, :(FS_SA + 1)], terminal_fstates[:, :(FS_SA + 1)]), axis=1)

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
            target_poly_s = np.array([0, 0, 0, 0, target_fstate[FS_SV], target_fstate[FS_SX]])

            # minimal margin used in addition to headway (center-to-center of both objects)
            margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # validate distance keeping (on frenet longitudinal axis)
            is_safe = KinematicUtils.is_maintaining_distance(poly_s, target_poly_s, margin, SAFETY_HEADWAY,
                                                             np.array([0, t]))

            are_valid.append(is_safe)

        return are_valid


@six.add_metaclass(ABCMeta)
class BeyondSpecConstraintFilter(ActionSpecFilter):
    """
    An ActionSpecFilter which implements a predefined constraint.
     The filter is defined by:
     (1) x-axis (select_points method)
     (2) the function to test on these points (_target_function)
     (3) the constraint function (_constraint_function)
     (4) the condition function between target and constraints (_condition function)

     Usage:
        extend BeyondSpecConstraintFilter class and implement the appropriate functions: (at least the four methods
        described above).
        To terminate the filter calculation use _raise_true/_raise_false  at any stage.
    """
    def __init__(self):
        self.distances = BreakingDistances.create_braking_distances()

    @abstractmethod
    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> Any:
        """
        selects relevant points from the action_spec (e.g., slow points)
        :param action_spec:
        :return:
        """
        pass

    @abstractmethod
    def _target_function(self, action_spec: ActionSpec, points: Any) -> np.ndarray:
        """
        The definition of the function to be tested.
        :param action_spec: the action spec which to filter
        :return: the result of the target function as an np.ndarray
        """
        pass

    @abstractmethod
    def _constraint_function(self, action_spec: ActionSpec, points: Any) -> np.ndarray:
        """
        Defines the constraint function over points.
        :param action_spec: the action spec which to filter
        :return: the result of the constraint function as an np.ndarray
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
        return (self._target_function(action_spec, points_under_test) <
                self._constraint_function(action_spec, points_under_test)).all()

    @staticmethod
    def _extend_spec(spec: ActionSpec) -> ActionSpec:
        """
        # TODO: Carefully Explain why this is necessary (add use cases maybe)
        Compensate for delays in super short action specs to increase filter efficiency
        :param spec:
        :return: A list of action specs with potentially extended copies of short ones
        """
        min_t = MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
        return spec if spec.t >= min_t else ActionSpec(min_t, spec.v, spec.s + (min_t - spec.t) * spec.v, spec.d, spec.recipe)

    def _get_beyond_spec_frenet_range(self, action_spec: ActionSpec, frenet_frame: GeneralizedFrenetSerretFrame) -> np.array:
        """
        Returns range of Frenet frame indices of type np.array([from_idx, till_idx]) beyond the action_spec goal
        :param action_spec:
        :param frenet_frame:
        :return: the range of indices beyond the action_spec goal
        """
        # get the worst case braking distance from spec.v to 0
        max_braking_distance = self.distances[FILTER_V_0_GRID.get_index(action_spec.v), FILTER_V_T_GRID.get_index(0)]
        max_relevant_s = min(action_spec.s + max_braking_distance, frenet_frame.s_max)
        # get the Frenet point indices near spec.s and near the worst case braking distance beyond spec.s
        return frenet_frame.get_index_on_frame_from_s(np.array([action_spec.s, max_relevant_s]))[0] + 1

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
                mask_value = self._check_condition(behavioral_state, BeyondSpecConstraintFilter._extend_spec(action_spec))
            except ConstraintFilterHaltWithValue as e:
                mask_value = e.value
            mask.append(mask_value)
        return mask


class BeyondSpecLateralAccelerationFilter(BeyondSpecConstraintFilter):
    """
    Checks if it is possible to break to desired lateral acceleration limit from the goal to the end of the frenet frame
    the following edge cases are treated by raise_true/false:
    (A) spec is not None
    (B) (spec.s >= target_lane_frenet.s_max then continue)
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
        max_braking_distance = self.distances[FILTER_V_0_GRID.get_index(action_spec.v), FILTER_V_T_GRID.get_index(0)]
        max_relevant_s = min(action_spec.s + max_braking_distance, frenet_frame.s_max)
        # get the Frenet point indices near spec.s and near the worst case braking distance beyond spec.s
        beyond_spec_range = frenet_frame.get_index_on_frame_from_s(np.array([action_spec.s, max_relevant_s]))[0] + 1
        # get s for all points in the range
        points_s = frenet_frame.get_s_from_index_on_frame(np.array(range(beyond_spec_range[0], beyond_spec_range[1])),
                                                          delta_s=0)
        # get velocity limits for all points in the range
        curvatures = np.maximum(np.abs(frenet_frame.k[beyond_spec_range[0]:beyond_spec_range[1], 0]), EPS)
        points_velocity_limits = np.sqrt(BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS[1] / curvatures)
        return points_s, points_velocity_limits

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> Any:
        """
        Finds 'slow' points indices
        :param behavioral_state:
        :param action_spec:
        :return:
        """
        if action_spec.v == 0:
            self._raise_true()
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        beyond_spec_s, points_velocity_limits = self._get_velocity_limits_of_points(action_spec, target_lane_frenet)
        slow_points = np.where(points_velocity_limits < action_spec.v)[0]  # points that require braking after spec
        # set edge case
        if len(slow_points) == 0:
            self._raise_true()
        return beyond_spec_s[slow_points], points_velocity_limits[slow_points]

    def _target_function(self, action_spec: ActionSpec, points: Any) -> np.ndarray:
        """
        The braking distance required by using the CALM aggressiveness level for slow points.
        :param action_spec:
        :param points:
        :return:
        """
        _, slow_points_velocity_limits = points
        brake_dist = self.distances[FILTER_V_0_GRID.get_index(action_spec.v), :]
        return brake_dist[FILTER_V_T_GRID.get_indices(slow_points_velocity_limits)]

    def _constraint_function(self, action_spec: ActionSpec, points: Any) -> np.ndarray:
        """
        The distance from current points to the 'slow points'
        :param action_spec:
        :param points:
        :return:
        """
        slow_points_s, _ = points
        dist_to_points = slow_points_s - action_spec.s
        return dist_to_points


class StaticTrafficFlowControlFilter(ActionSpecFilter):
    """
    Checks if there is a StaticTrafficFlowControl between ego and the goal
    Currently treats every 'StaticTrafficFlowControl' as a stop event.
    """

    @staticmethod
    def _has_stop_bar_until_goal(action_spec: ActionSpec, behavioral_state: BehavioralGridState):
        """
        Checks if there is a stop_bar between current ego location and the goal
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
    Filter for "BeyondSpec" stop sign
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

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
        """
        Basically just checks if there are stop signs. Returns the `s` of the first stop-sign
        :param behavioral_state:
        :param action_spec:
        :return: The index of the end point
        """
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        stop_bar_s = self._get_first_stop_s(target_lane_frenet, action_spec.s)
        if stop_bar_s is None:  # no stop bars
            self._raise_true()
        return np.array([stop_bar_s])

    def _target_function(self, action_spec: ActionSpec, points: np.ndarray) -> np.ndarray:
        """
        Braking distance from current velocity to 0
        :param action_spec:
        :param points:
        :return:
        """
        # retrieve distances of static actions for the most aggressive level, since they have the shortest distances
        brake_dist = self.distances[FILTER_V_0_GRID.get_index(action_spec.v), FILTER_V_T_GRID.get_index(0)]
        return brake_dist

    def _constraint_function(self, action_spec: ActionSpec, points: np.ndarray) -> np.ndarray:
        """
        The distance from first stop sign to current location
        :param action_spec:
        :param points:
        :return: The distance from the goal to the
        """
        dist_to_points = points[0] - action_spec.s
        assert dist_to_points >= 0, 'Stop Sign must be ahead'
        return dist_to_points


class BreakingDistances:
    """
    Calculates breaking distances
    """
    # TODO: make it singleton, since a few filters use the braking distances

    @staticmethod
    def create_braking_distances(aggresiveness_level=AggressivenessLevel.CALM.value) -> np.array:
        """
        Creates distances of all follow_lane CALM braking actions with a0 = 0
        :return: the actions' distances
        """
        # create v0 & vT arrays for all braking actions
        v0, vT = np.meshgrid(FILTER_V_0_GRID.array, FILTER_V_T_GRID.array, indexing='ij')
        v0, vT = np.ravel(v0), np.ravel(vT)
        # calculate distances for braking actions
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[aggresiveness_level]
        distances = np.zeros_like(v0)
        distances[v0 > vT] = BreakingDistances._calc_actions_distances_for_given_weights(w_T, w_J, v0[v0 > vT],
                                                                                         vT[v0 > vT])
        return distances.reshape(len(FILTER_V_0_GRID), len(FILTER_V_T_GRID))

    @staticmethod
    def _calc_actions_distances_for_given_weights(w_T, w_J, v_0: np.array, v_T: np.array) -> np.array:
        """
        Calculate the distances for the given actions' weights and scenario params
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param v_T: array of desired final velocities [m/s]
        :return: actions' distances; actions not meeting acceleration limits have infinite distance
        """
        # calculate actions' planning time
        a_0 = np.zeros_like(v_0)
        T = BreakingDistances.calc_T_s(w_T, w_J, v_0, a_0, v_T)

        # check acceleration limits
        poly_coefs = QuarticPoly1D.s_profile_coefficients(a_0, v_0, v_T, T)
        in_limits = QuarticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)[:, 0]

        # calculate actions' distances, assuming a_0 = 0
        distances = T * (v_0 + v_T) / 2
        distances[np.logical_not(in_limits)] = np.inf
        return distances

    @staticmethod
    def calc_T_s(w_T: float, w_J: float, v_0: np.array, a_0: np.array, v_T: np.array):
        """
        given initial & end constraints and time-jerk weights, calculate longitudinal planning time
        :param w_T: weight of Time component in time-jerk cost function
        :param w_J: weight of longitudinal jerk component in time-jerk cost function
        :param v_0: array of initial velocities [m/s]
        :param a_0: array of initial accelerations [m/s^2]
        :param v_T: array of final velocities [m/s]
        :return: array of longitudinal trajectories' lengths (in seconds) for all sets of constraints
        """
        # Agent is in tracking mode, meaning the required velocity change is negligible and action time is actually
        # zero. This degenerate action is valid but can't be solved analytically.
        non_zero_actions = np.logical_not(np.logical_and(np.isclose(v_0, v_T, atol=1e-3, rtol=0),
                                                         np.isclose(a_0, 0.0, atol=1e-3, rtol=0)))
        w_T_array = np.full(v_0[non_zero_actions].shape, w_T)
        w_J_array = np.full(v_0[non_zero_actions].shape, w_J)

        # Get polynomial coefficients of time-jerk cost function derivative for our settings
        time_cost_derivative_poly_coefs = QuarticPoly1D.time_cost_function_derivative_coefs(
            w_T_array, w_J_array, a_0[non_zero_actions], v_0[non_zero_actions], v_T[non_zero_actions])

        # Find roots of the polynomial in order to get extremum points
        cost_real_roots = Math.find_real_roots_in_limits(time_cost_derivative_poly_coefs, np.array([0, np.inf]))

        # return T as the minimal real root
        T = np.zeros_like(v_0)
        T[non_zero_actions] = np.fmin.reduce(cost_real_roots, axis=-1)
        return T
