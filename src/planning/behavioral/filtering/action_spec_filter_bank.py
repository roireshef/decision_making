from collections import defaultdict

import numpy as np
import rte.python.profiler as prof
from decision_making.src.global_constants import BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED, BP_ACTION_T_LIMITS, \
    TRAJECTORY_TIME_RESOLUTION, EPS
from decision_making.src.global_constants import VELOCITY_LIMITS, LON_ACC_LIMITS, LAT_ACC_LIMITS, \
    FILTER_V_0_GRID, FILTER_V_T_GRID, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, \
    MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.behavioral.filtering.constraint_spec_filter import ConstraintSpecFilter
from decision_making.src.planning.types import FS_DX, FS_SV, FS_SX, S1, C_V
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
        ctrajectories, _ = self.build_trajectories(action_specs, behavioral_state)
        return list(KinematicUtils.filter_by_cartesian_limits(ctrajectories, VELOCITY_LIMITS, LON_ACC_LIMITS,
                                                         LAT_ACC_LIMITS, BEHAVIORAL_PLANNING_DEFAULT_DESIRED_SPEED)[0])

    @staticmethod
    def get_nominal_speeds(ftrajectories_s: np.ndarray, frenet: GeneralizedFrenetSerretFrame) -> np.ndarray:
        """
        :param ftrajectories_s: The frenet trajectories to which to calculate the nominal speeds (only s)
        :return: A matrix of Trajectories x Time_samples x Max_limits
        """
        #patched_lane_ids = [13148163, 19670531, 58375172]
        # get lane_the ids
        lane_ids_list = frenet.convert_to_segment_states(ftrajectories_s)[0]
        max_velocities = {lane_id: MapUtils.get_lane(lane_id).e_v_nominal_speed
                          for lane_id in np.unique(lane_ids_list)}
        return np.vectorize(max_velocities.get)(lane_ids_list)


    def build_trajectories(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState,
                           build_nominal_speeds = False):
        # group all specs and their indices by the relative lanes
        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)
        time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
        ctrajectories = np.zeros((len(action_specs), len(time_samples), 6), dtype=float)
        lane_segment_velocity_limits = np.zeros((len(action_specs), len(time_samples)), dtype=float)
        # loop on the target relative lanes and calculate lateral accelerations for all relevant specs
        for rel_lane, lane_specs in specs_by_rel_lane.items():
            specs_t = np.array([spec.t for spec in lane_specs])
            pad_mode = np.array([spec.only_padding_mode for spec in lane_specs])
            goal_fstates = np.array([spec.as_fstate() for spec in lane_specs])

            frenet = behavioral_state.extended_lane_frames[rel_lane]  # the target GFF
            ego_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            ego_fstates = np.tile(ego_fstate, len(lane_specs)).reshape((len(lane_specs), -1))

            # calculate polynomials
            poly_coefs_s, poly_coefs_d = KinematicUtils.calc_poly_coefs(specs_t, ego_fstates, goal_fstates, pad_mode)

            # create Frenet trajectories for s axis for all trajectories of rel_lane and for all time samples
            ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_samples)
            ftrajectories_d = QuinticPoly1D.polyval_with_derivatives(poly_coefs_d, time_samples)

            # Pad (extrapolate) short trajectories from spec.t until minimal action time.
            # Beyond the maximum between spec.t and minimal action time the Frenet trajectories are set to zero.
            ftrajectories = FilterForKinematics.pad_trajectories_beyond_spec(
                lane_specs, ftrajectories_s, ftrajectories_d, specs_t, pad_mode)

            # convert Frenet trajectories to cartesian trajectories
            ctrajectories[indices_by_rel_lane[rel_lane]] = frenet.ftrajectories_to_ctrajectories(ftrajectories)
            if build_nominal_speeds:
                lane_segment_velocity_limits[indices_by_rel_lane[rel_lane]] = FilterForKinematics.get_nominal_speeds(
                    ftrajectories_s, frenet)
        return ctrajectories, lane_segment_velocity_limits

    @staticmethod
    def pad_trajectories_beyond_spec(action_specs: List[ActionSpec], ftrajectories_s: np.array, ftrajectories_d: np.array,
                                     T: np.array, in_padding_mode: np.array) -> np.array:
        """
        Given action specs and their Frenet trajectories, pad (extrapolate) short trajectories from spec.t until
        minimal action time. Beyond the maximum between spec.t and minimal action time Frenet trajectories are set to
        zero.
        Important! Here we assume that zero Frenet states converted to Cartesian states pass all kinematic Cartesian
        filters.
        :param action_specs: list of actions spec
        :param ftrajectories_s: matrix Nx3 of N Frenet trajectories for s component
        :param ftrajectories_d: matrix Nx3 of N Frenet trajectories for d component
        :param T: array of size N: time horizons for each action
        :param in_padding_mode: boolean array of size N: True if an action is in padding mode
        :return: full Frenet trajectories (s & d)
        """
        # calculate trajectory time indices for all spec.t
        spec_t_idxs = (T / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
        spec_t_idxs[in_padding_mode] = 0

        # calculate trajectory time indices for t = max(spec.t, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON)
        last_pad_idxs = (np.maximum(T, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1

        # pad short ftrajectories beyond spec.t until MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
        for (spec_t_idx, last_pad_idx, trajectory_s, trajectory_d, spec) in \
                zip(spec_t_idxs, last_pad_idxs, ftrajectories_s, ftrajectories_d, action_specs):
            # if spec.t < MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, pad ftrajectories_s from spec.t to
            # MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
            if spec_t_idx < last_pad_idx:
                times_beyond_spec = np.arange(spec_t_idx, last_pad_idx) * TRAJECTORY_TIME_RESOLUTION - spec.t
                trajectory_s[spec_t_idx:last_pad_idx] = np.c_[spec.s + times_beyond_spec * spec.v,
                                                              np.full(times_beyond_spec.shape, spec.v),
                                                              np.zeros_like(times_beyond_spec)]
            trajectory_s[last_pad_idx:] = 0
            trajectory_d[spec_t_idx:] = 0

        # return full Frenet trajectories
        return np.c_[ftrajectories_s, ftrajectories_d]


class FilterForLaneSpeedLimits(FilterForKinematics):

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
        ctrajectories, nominal_lane_speeds = self.build_trajectories(action_specs, behavioral_state,
                                                                     build_nominal_speeds=True)
        conforms_end_specs = [action_spec.v for action_spec in action_specs] <= nominal_lane_speeds[:, -1]
        lon_velocity = ctrajectories[:, :, C_V]
        conforms_trajectories = lon_velocity <= nominal_lane_speeds + EPS
        return np.logical_and(conforms_end_specs, conforms_trajectories)





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

