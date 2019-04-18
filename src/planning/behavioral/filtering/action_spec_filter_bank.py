from decision_making.paths import Paths
import os
import copy

from collections import defaultdict

import numpy as np
from decision_making.src.planning.utils.math_utils import Math
from typing import List

import rte.python.profiler as prof
import six
from abc import ABCMeta, abstractmethod
from decision_making.src.global_constants import BP_ACTION_T_LIMITS, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, \
    FILTER_V_0_GRID, FILTER_A_0_GRID, \
    FILTER_V_T_GRID, BP_JERK_S_JERK_D_TIME_WEIGHTS, BP_LAT_ACC_STRICT_COEF, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
from decision_making.src.global_constants import EPS, WERLING_TIME_RESOLUTION, VELOCITY_LIMITS, LON_ACC_LIMITS, \
    LAT_ACC_LIMITS
from decision_making.src.global_constants import SAFETY_HEADWAY
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, DynamicActionRecipe, \
    RelativeLongitudinalPosition, StaticActionRecipe, AggressivenessLevel, ActionType, RelativeLane
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.trajectory.samplable_werling_trajectory import SamplableWerlingTrajectory
from decision_making.src.planning.types import FS_SA, FS_DX, C_V, C_K
from decision_making.src.planning.types import FS_SX, FS_SV, LAT_CELL
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None and ~np.isnan(action_spec.t) for action_spec in action_specs]


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
        initial_fstates = np.array([behavioral_state.projected_ego_fstates[spec.relative_lane] for spec in action_specs])
        terminal_fstates = np.array([spec.as_fstate() for spec in action_specs])
        T = np.array([spec.t for spec in action_specs])

        # creare boolean arrays indicating whether the specs are in tracking mode
        in_track_mode = np.array([spec.in_track_mode for spec in action_specs])
        no_track_mode = np.logical_not(in_track_mode)

        # extract terminal maneuver time and generate a matrix that is used to find jerk-optimal polynomial coefficients
        A_inv = QuinticPoly1D.inverse_time_constraints_tensor(T[no_track_mode])

        # represent initial and terminal boundary conditions (for two Frenet axes s,d) for non-tracking specs
        constraints_s = np.concatenate((initial_fstates[no_track_mode, :FS_DX], terminal_fstates[no_track_mode, :FS_DX]), axis=1)
        constraints_d = np.concatenate((initial_fstates[no_track_mode, FS_DX:], terminal_fstates[no_track_mode, FS_DX:]), axis=1)

        # solve for s(t) and d(t)
        poly_coefs_s, poly_coefs_d = np.zeros((len(action_specs), 6)), np.zeros((len(action_specs), 6))
        poly_coefs_s[no_track_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_s)
        poly_coefs_d[no_track_mode] = QuinticPoly1D.zip_solve(A_inv, constraints_d)
        # in tracking mode (constant velocity) the s polynomials have only two non-zero coefficients
        poly_coefs_s[in_track_mode, 4:] = np.c_[initial_fstates[in_track_mode, FS_SV], initial_fstates[in_track_mode, FS_SX]]

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
            is_valid_in_cartesian = KinematicUtils.filter_by_cartesian_limits(cartesian_points[np.newaxis, ...],
                                        VELOCITY_LIMITS, LON_ACC_LIMITS, BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS)[0]

            are_valid.append(is_valid_in_cartesian)

            # TODO: remove it
            # if is_valid_in_cartesian and abs(spec.v - 3) < 1.2:
            #     lat_acc = np.abs(cartesian_points[:, C_V] ** 2 * cartesian_points[:, C_K])
            #     worst_t = np.argmax(lat_acc)
            #     worst_v = cartesian_points[worst_t, C_V]
            #     worst_k = cartesian_points[worst_t, C_K]
            #     print('BP %.3f: spec.t=%.3f spec.v=%.3f; worst_lat_acc: t=%.1f v=%.3f k=%.3f' %
            #           (behavioral_state.ego_state.timestamp_in_sec, spec.t, spec.v, worst_t * 0.1, worst_v, worst_k))

        # TODO: remove it
        if not any(are_valid):
            ego_fstate = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]
            frenet = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
            init_idx = frenet.get_index_on_frame_from_s(ego_fstate[:1])[0][0]
            print('ERROR in BP %.3f: ego_fstate=%s; nominal_k=%s' %
                  (behavioral_state.ego_state.timestamp_in_sec, NumpyUtils.str_log(ego_fstate),
                   frenet.k[init_idx:init_idx+10, 0]))

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
            target_poly_s = np.array([0, 0, 0, 0, target_fstate[FS_SV], target_fstate[FS_SX]])

            # minimal margin used in addition to headway (center-to-center of both objects)
            margin = LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT + \
                     behavioral_state.ego_state.size.length / 2 + target.dynamic_object.size.length / 2

            # validate distance keeping (on frenet longitudinal axis)
            is_safe = KinematicUtils.is_maintaining_distance(poly_s, target_poly_s, margin, SAFETY_HEADWAY, np.array([0, t]))

            are_valid.append(is_safe)

        return are_valid


@six.add_metaclass(ABCMeta)
class ConstraintSpecFilter(ActionSpecFilter):
    """
    A filter to allow constraint
    """

    @abstractmethod
    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
        """
        selects relevant points from the action_spec (e.g., slow points)
        :param action_spec:
        :return:
        """
        pass

    @abstractmethod
    def _target_function(self, behavioral_state: BehavioralGridState,
                         action_spec: ActionSpec, points: np.ndarray) -> np.ndarray:
        """
        function under test.
        :param action_spec:
        :return:
        """
        pass

    @abstractmethod
    def _constraint(self, behavioral_state: BehavioralGridState,
                    action_spec: ActionSpec, points: np.ndarray) -> np.ndarray:
        """
        Defines a constraint function over points
        :return:
        """
        pass

    @abstractmethod
    def _condition(self, target_values, constraints_values) -> bool:
        """
        Applying a condition on target and constraints values
        :param target_values:
        :param constraints_values:
        :return:
        """
        pass

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        # TODO: What about parallelism on all ActionSpecs?
        mask = []
        for action_spec in action_specs:
            mask.append(self._condition(
                self._target_function(self._select_points(behavioral_state, action_spec)),
                self._constraint(behavioral_state, action_spec ,self._select_points(behavioral_state, action_spec))))
        return mask


class ConstraintBrakeLateralAccelerationFilter(ConstraintSpecFilter):
    """
    Testing the constraint design with this extension.
    Checks if it is possible to break from the goal to the end of the frenet frame
    """

    def __init__(self, path: str):
        # TODO: Change this
        self.distances = ConstraintBrakeLateralAccelerationFilter.read_distances(path)

    @staticmethod
    def read_distances(path):
        """
        This method reads maps from file into a dictionary mapping a tuple of (action_type,weights) to a LUT.
        :param path: The directory holding all maps (.npy files)
        :return: a dictionary mapping a tuple of (action_type,weights) to a binary LUT.
        """
        directory = Paths.get_resource_absolute_path_filename(path)
        distances = {}
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                predicate_path = Paths.get_resource_absolute_path_filename('%s/%s' % (path, filename))
                action_type = filename.split('.bin')[0].split('_distances')[0]
                wT, wJ = [float(filename.split('.bin')[0].split('_')[4]),
                          float(filename.split('.bin')[0].split('_')[6])]
                distances[(action_type, wT, wJ)] = np.load(file=predicate_path)
        return distances

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        # get the Frenet point index near the goal action_spec.s
        spec_s_point_idx = target_lane_frenet.get_index_on_frame_from_s(np.array([action_spec.s]))[0][0]
        # find all Frenet points beyond spec.s, where velocity limit (by curvature) is lower then spec.v
        beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx + 1, len(target_lane_frenet.k), 4))
        curvatures = np.maximum(np.abs(target_lane_frenet.k[beyond_spec_frenet_idxs, 0]), EPS)
        points_velocity_limits = np.sqrt(BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS[1] / curvatures)
        slow_points = np.where(points_velocity_limits < action_spec.v)[0]  # points that require braking after spec

        return beyond_spec_frenet_idxs[slow_points]

    def _target_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec, points: np.ndarray) -> np.ndarray:
        # retrieve distances of static actions for the most aggressive level, since they have the shortest distances
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
        brake_dist = self.distances[(ActionType.FOLLOW_LANE.name.lower(), wT, wJ)][
                     FILTER_V_0_GRID.get_index(action_spec.v), FILTER_A_0_GRID.get_index(0), :]
        vel_limit_in_points = 0 # TODO
        return brake_dist[FILTER_V_T_GRID.get_indices(vel_limit_in_points)]

    def _constraint(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec, points: np.ndarray) -> np.ndarray:
        frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]
        # create constraints for static actions per point beyond the given spec
        dist_to_points = frenet.get_s_from_index_on_frame(points, delta_s=0) - action_spec.s
        return dist_to_points

    def _condition(self, target_values, constraints_values) -> bool:
        return target_values < constraints_values


class ConstraintStoppingAtLocationFilter(ConstraintSpecFilter):
    """
    A filter for actions that may arrive at location too fast
    Checks if it's possible to break to the stop location
    """
    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
        """
         TODO: Selects all points from the end of the action_spec until the 'stop' location
         return empty if there is no 'stop' location
        :param behavioral_state:
        :param action_spec:
        :return:
        """
        pass

    def _target_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec,
                         points: np.ndarray) -> np.ndarray:
        """
        TODO: the aggressive velocity to get to the stop sign
        :param behavioral_state:
        :param action_spec:
        :param points:
        :return:
        """
        pass

    def _constraint(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec,
                    points: np.ndarray) -> np.ndarray:
        """
        TODO: Similar to the braking condition for lateral acceleration, assuming braking to 0
        :param behavioral_state:
        :param action_spec:
        :param points:
        :return:
        """
        pass

    def _condition(self, target_values, constraints_values) -> bool:
        return target_values <= constraints_values


class FilterByLateralAcceleration(ActionSpecFilter):
    def __init__(self):
        self.distances = FilterByLateralAcceleration._create_braking_distances()

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[bool]:
        """
        Check violation of lateral acceleration for action_specs, and beyond action_specs check ability to brake
        before all future curves using any static action.
        :return: specs list that passed the lateral acceleration filter
        """
        # now check ability to break before future curves beyond the baseline specs' trajectories
        filtering_res = [False] * len(action_specs)
        for spec_idx, spec in enumerate(action_specs):
            # this should be taken care of by the kineatic_filter
            if spec is None:
                continue

            # in case of too short spec extend it to MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
            extended_spec = copy.copy(spec)
            min_action_time = MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
            if spec.t < min_action_time:
                extended_spec.t = min_action_time
                extended_spec.s += (min_action_time - spec.t) * spec.v

            target_lane_frenet = behavioral_state.extended_lane_frames[spec.relative_lane]  # the target GFF
            if extended_spec.s >= target_lane_frenet.s_max:
                continue
            # get the Frenet point index near the goal action_spec.s
            spec_s_point_idx = target_lane_frenet.get_index_on_frame_from_s(np.array([extended_spec.s]))[0][0]
            # find all Frenet points beyond spec.s, where velocity limit (by curvature) is lower then spec.v
            beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx + 1, len(target_lane_frenet.k), 4))
            curvatures = np.maximum(np.abs(target_lane_frenet.k[beyond_spec_frenet_idxs, 0]), EPS)

            points_velocity_limits = np.sqrt(BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS[1] / curvatures)
            slow_points = np.where(points_velocity_limits < spec.v)[0]  # points that require braking after spec

            # if all points beyond the spec have velocity limit higher than spec.v, so no need to brake
            if len(slow_points) == 0:
                filtering_res[spec_idx] = True
                continue  # the spec passes the filter

            # check the ability to brake beyond the spec for all points with limited velocity
            is_able_to_brake = FilterByLateralAcceleration._check_ability_to_brake_beyond_spec(
                extended_spec, target_lane_frenet, beyond_spec_frenet_idxs[slow_points],
                points_velocity_limits[slow_points], self.distances)

            filtering_res[spec_idx] = is_able_to_brake

        return filtering_res

    @staticmethod
    def check_lateral_acceleration_limits(action_specs: List[ActionSpec],
                                          behavioral_state: BehavioralGridState) -> np.array:
        """
        check meeting of lateral acceleration limits for the given specs list
        :param action_specs:
        :param behavioral_state:
        :return: bool array of size len(action_specs)
        """
        # group all specs and their indices by the relative lanes
        specs_by_rel_lane = defaultdict(list)
        indices_by_rel_lane = defaultdict(list)
        for i, spec in enumerate(action_specs):
            if spec is not None:
                specs_by_rel_lane[spec.relative_lane].append(spec)
                indices_by_rel_lane[spec.relative_lane].append(i)

        time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], WERLING_TIME_RESOLUTION)
        lateral_accelerations = np.zeros((len(action_specs), len(time_samples)))

        # loop on the target relative lanes and calculate lateral accelerations for all relevant specs
        for rel_lane in specs_by_rel_lane.keys():
            lane_specs = specs_by_rel_lane[rel_lane]
            specs_t = np.array([spec.t for spec in lane_specs])
            goal_fstates = np.array([spec.as_fstate() for spec in lane_specs])

            # don't test lateral acceleration beyond spec.t, so set 0 for time_samples > spec.t
            # create 2D time samples: a line for each spec with non-zero size according to spec.t
            time_samples_2d = np.tile(time_samples, len(lane_specs)).reshape(len(lane_specs), len(time_samples))
            for i, spec_samples in enumerate(time_samples_2d):
                spec_samples[(int(specs_t[i] / WERLING_TIME_RESOLUTION) + 1):] = 0

            frenet = behavioral_state.extended_lane_frames[rel_lane]  # the target GFF
            ego_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            ego_fstates = np.tile(ego_fstate, len(lane_specs)).reshape((len(lane_specs), -1))

            # calculate polynomial coefficients of the spec's Frenet trajectory for s axis
            A_inv = QuinticPoly1D.inverse_time_constraints_tensor(specs_t)
            poly_coefs_s = QuinticPoly1D.zip_solve(A_inv, np.hstack(
                (ego_fstates[:, FS_SX:FS_DX], goal_fstates[:, FS_SX:FS_DX])))

            # create Frenet trajectories for s axis for all trajectories of rel_lane and for all time samples
            ftrajectories_s = QuinticPoly1D.zip_polyval_with_derivatives(poly_coefs_s, time_samples_2d)

            # calculate ftrajectories_s beyond spec.t
            last_time_idxs = (np.maximum(specs_t, MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON) / WERLING_TIME_RESOLUTION).astype(int) + 1
            for i, trajectory in enumerate(ftrajectories_s):
                # calculate trajectory time index for t = max(specs_t[i], BP_ACTION_T_LIMITS[0])
                # if spec.t < MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON, extrapolate trajectories between spec.t and
                # MINIMUM_REQUIRED_TRAJECTORY_TIME_HORIZON
                if specs_t[i] < last_time_idxs[i]:
                    spec_t_idx = int(specs_t[i] / WERLING_TIME_RESOLUTION) + 1
                    times_beyond_spec = np.arange(spec_t_idx, last_time_idxs[i]) * WERLING_TIME_RESOLUTION - specs_t[
                        i]
                    trajectory[spec_t_idx:last_time_idxs[i]] = np.c_[
                        lane_specs[i].s + times_beyond_spec * lane_specs[i].v,
                        np.full(times_beyond_spec.shape, lane_specs[i].v),
                        np.zeros_like(times_beyond_spec)]
                # assign near-zero velocity to ftrajectories_s beyond last_time_idx
                trajectory[last_time_idxs[i]:] = np.array([EPS, EPS, 0])

            # assign zeros to the lateral movement of ftrajectories
            ftrajectories = np.concatenate((ftrajectories_s, np.zeros_like(ftrajectories_s)), axis=-1)

            # convert Frenet to cartesian trajectories
            lane_center_ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories)

            # calculate lateral accelerations
            lateral_accelerations[np.array(indices_by_rel_lane[rel_lane])] = \
                lane_center_ctrajectories[..., C_K] * lane_center_ctrajectories[..., C_V] ** 2

        return NumpyUtils.is_in_limits(lateral_accelerations, BP_LAT_ACC_STRICT_COEF * LAT_ACC_LIMITS).all(axis=-1)

    @staticmethod
    def _check_ability_to_brake_beyond_spec(action_spec: ActionSpec, frenet: GeneralizedFrenetSerretFrame,
                                            frenet_points_idxs: np.array, vel_limit_in_points: np.array,
                                            action_distances: np.array):
        """
        Given action spec and velocity limits on a subset of Frenet points, check if it's possible to brake enough
        before arriving to these points. The ability to brake is verified using static actions distances.
        :param action_spec: action specification
        :param frenet: generalized Frenet Serret frame
        :param frenet_points_idxs: array of indices of the Frenet frame points, having limited velocity
        :param vel_limit_in_points: array of maximal velocities at frenet_points_idxs
        :param action_distances: dictionary of distances of static actions
        :return: True if the agent can brake before each given point to its limited velocity
        """
        # create constraints for static actions per point beyond the given spec
        dist_to_points = frenet.get_s_from_index_on_frame(frenet_points_idxs, delta_s=0) - action_spec.s

        # retrieve distances of static actions for the most aggressive level, since they have the shortest distances
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
        brake_dist = action_distances[FILTER_V_0_GRID.get_index(action_spec.v), :]
        return (dist_to_points > brake_dist[FILTER_V_T_GRID.get_indices(vel_limit_in_points)]).all()

    @staticmethod
    def _create_braking_distances() -> np.array:
        """
        Creates distances of all follow_lane CALM braking actions with a0 = 0
        :return: the actions' distances
        """
        # create v0 & vT arrays for all braking actions
        v0, vT = np.meshgrid(FILTER_V_0_GRID.array, FILTER_V_T_GRID.array, indexing='ij')
        v0, vT = np.ravel(v0), np.ravel(vT)
        # calculate distances for braking actions
        w_J, _, w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
        distances = np.zeros_like(v0)
        distances[v0 > vT] = FilterByLateralAcceleration._calc_actions_distances_for_given_weights(w_T, w_J, v0[v0 > vT], vT[v0 > vT])
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
        T = FilterByLateralAcceleration.calc_T_s(w_T, w_J, v_0, a_0, v_T)

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
