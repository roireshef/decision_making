import numpy as np
import rte.python.profiler as prof
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED
from decision_making.src.global_constants import EPS, WERLING_TIME_RESOLUTION, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS, FILTER_V_0_GRID, FILTER_V_T_GRID, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition, StaticActionRecipe
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.behavioral.filtering.constraint_spec_filter import ConstraintSpecFilter
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SA, FS_DX, FS_SV, FS_SX
from decision_making.src.planning.types import LAT_CELL
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils, BrakingDistances
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from decision_making.src.utils.map_utils import MapUtils
from typing import List


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
                    if isinstance(spec.recipe, DynamicActionRecipe):
                        print('filtered: Kinematic Frenet at time %.3f: %s' % (
                        behavioral_state.ego_state.timestamp_in_sec, spec.recipe.aggressiveness))
                    continue

            total_time = max(MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, t)
            time_samples = np.arange(0, total_time + EPS, WERLING_TIME_RESOLUTION)

            # generate a SamplableWerlingTrajectory (combination of s(t), d(t) polynomials applied to a Frenet frame)
            samplable_trajectory = SamplableWerlingTrajectory(0, t, t, total_time, frenet_frame, poly_s, poly_d)
            cartesian_points = samplable_trajectory.sample(time_samples)  # sample cartesian points from the solution

            # validate cartesian points against cartesian limits
            is_valid_in_cartesian = KinematicUtils.filter_by_cartesian_limits(
                cartesian_points[np.newaxis, ...],
                VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS,
                BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)[0]

            # TODO: remove it
            if not is_valid_in_cartesian and isinstance(spec.recipe, DynamicActionRecipe):
                print('filtered: Kinematic Cartesian at time %.3f: %s ego_va=%s\nvel=%s\nacc=%s' %
                      (behavioral_state.ego_state.timestamp_in_sec, spec.recipe.aggressiveness,
                       behavioral_state.ego_state.cartesian_state[3:5], cartesian_points[:, 3], cartesian_points[:, 4]))

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
        T = np.array([spec.t for spec in action_specs])

        dynamic = [isinstance(spec.recipe, DynamicActionRecipe) for spec in action_specs]
        aggressiveness = [spec.recipe.aggressiveness for spec in action_specs]

        # represent initial and terminal boundary conditions (for s axis)
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[cell[LAT_CELL]] for cell in relative_cells])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        constraints_s = np.concatenate((initial_fstates[:, :FS_DX], terminal_fstates[:, :FS_DX]), axis=1)
        poly_coefs_s = np.empty(shape=(len(action_specs), QuinticPoly1D.num_coefs()), dtype=np.float)

        # create boolean arrays indicating whether the specs are in tracking mode
        padding_mode = np.array([spec.only_padding_mode for spec in action_specs])
        not_padding_mode = np.logical_not(padding_mode)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        if not_padding_mode.any():
            # solve for s(t)
            A_inv = np.linalg.inv(QuinticPoly1D.time_constraints_tensor(T[not_padding_mode]))
            poly_coefs_s[not_padding_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_s[not_padding_mode])
        poly_coefs_s[padding_mode], _ = KinematicUtils.create_linear_profile_polynomial_pairs(terminal_fstates[padding_mode])

        are_valid = []
        for poly_s, t, cell, target, dyn, aggr in zip(poly_coefs_s, T, relative_cells, target_vehicles, dynamic, aggressiveness):
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
            if not is_safe and dyn:
                print('filtered: Safety at time %.3f: %s' % (behavioral_state.ego_state.timestamp_in_sec, aggr))

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

