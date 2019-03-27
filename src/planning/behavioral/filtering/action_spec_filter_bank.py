from collections import defaultdict

import six
from abc import ABCMeta, abstractmethod
from decision_making.src.global_constants import LAT_ACC_LIMITS, EPS, FILTER_A_0_GRID, FILTER_V_0_GRID, \
    TRAJECTORY_TIME_RESOLUTION, BP_ACTION_T_LIMITS, FILTER_V_T_GRID
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec, ActionType, AggressivenessLevel
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterLimitsViolatingTrajectory
from decision_making.src.planning.types import FS_DX, FS_SX, C_V, C_K
from decision_making.src.planning.utils.generalized_frenet_serret_frame import GeneralizedFrenetSerretFrame
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D
from typing import List
import numpy as np


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[bool]:
        return [(action_spec and behavioral_state) is not None for action_spec in action_specs]


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

    def __init__(self, predicates_dir: str):
        self.limit_predicates = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'limits')
        self.distance_predicates = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'distances')

    def _select_points(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec) -> np.ndarray:
        target_lane_frenet = behavioral_state.extended_lane_frames[action_spec.relative_lane]  # the target GFF
        # get the Frenet point index near the goal action_spec.s
        spec_s_point_idx = target_lane_frenet.get_index_on_frame_from_s(np.array([action_spec.s]))[0][0]
        # find all Frenet points beyond spec.s, where velocity limit (by curvature) is lower then spec.v
        beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx + 1, len(target_lane_frenet.k), 4))
        curvatures = np.maximum(np.abs(target_lane_frenet.k[beyond_spec_frenet_idxs, 0]), EPS)
        points_velocity_limits = np.sqrt(LAT_ACC_LIMITS[1] / curvatures)
        slow_points = np.where(points_velocity_limits < action_spec.v)[0]  # points that require braking after spec

        return beyond_spec_frenet_idxs[slow_points]

    def _target_function(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec, points: np.ndarray) -> np.ndarray:
        # retrieve distances of static actions for the most aggressive level, since they have the shortest distances
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS_FOLLOW_LANE[AggressivenessLevel.CALM.value]
        brake_dist = self.distance_predicates[(ActionType.FOLLOW_LANE.name.lower(), wT, wJ)][
                     FILTER_V_0_GRID.get_index(action_spec.v), FILTER_A_0_GRID.get_index(0), :]
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


    def _constraint(self, behavioral_state: BehavioralGridState, action_spec: ActionSpec,
                    points: np.ndarray) -> np.ndarray:
        """
        TODO: Similar to the braking condition for lateral acceleration, assuming braking to 0
        :param behavioral_state:
        :param action_spec:
        :param points:
        :return:
        """

    def _condition(self, target_values, constraints_values) -> bool:
        return target_values <= constraints_values


class FilterByLateralAcceleration(ActionSpecFilter):
    def __init__(self, predicates_dir: str):
        self.predicates = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'limits')
        self.distances = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'distances')

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState, state: State) -> List[bool]:
        """
        Check violation of lateral acceleration for action_specs, and beyond action_specs check ability to brake
        before all future curves using any static action.
        :return: specs list that passed the lateral acceleration filter
        """
        # first check lateral acceleration limits for all baseline trajectories of all action_specs
        meet_limits = FilterByLateralAcceleration.check_lateral_acceleration_limits(action_specs, behavioral_state)

        # now check ability to break before future curves beyond the baseline specs' trajectories
        filtering_res = [False] * len(action_specs)
        for spec_idx, spec in enumerate(action_specs):

            if spec is None or not meet_limits[spec_idx]:
                continue
            target_lane_frenet = behavioral_state.extended_lane_frames[spec.relative_lane]  # the target GFF
            if spec.s >= target_lane_frenet.s_max:
                continue
            # get the Frenet point index near the goal action_spec.s
            spec_s_point_idx = target_lane_frenet.get_index_on_frame_from_s(np.array([spec.s]))[0][0]
            # find all Frenet points beyond spec.s, where velocity limit (by curvature) is lower then spec.v
            beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx + 1, len(target_lane_frenet.k), 4))
            curvatures = np.maximum(np.abs(target_lane_frenet.k[beyond_spec_frenet_idxs, 0]), EPS)

            points_velocity_limits = np.sqrt(LAT_ACC_LIMITS[1] / curvatures)
            slow_points = np.where(points_velocity_limits < spec.v)[0]  # points that require braking after spec

            # if all points beyond the spec have velocity limit higher than spec.v, so no need to brake
            if len(slow_points) == 0:
                filtering_res[spec_idx] = True
                continue  # the spec passes the filter

            # check the ability to brake beyond the spec for all points with limited velocity
            is_able_to_brake = FilterByLateralAcceleration._check_ability_to_brake_beyond_spec(
                spec, behavioral_state.extended_lane_frames[spec.relative_lane],
                beyond_spec_frenet_idxs[slow_points], points_velocity_limits[slow_points], self.distances)

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

        time_samples = np.arange(0, BP_ACTION_T_LIMITS[1], TRAJECTORY_TIME_RESOLUTION)
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
                spec_samples[(int(specs_t[i] / TRAJECTORY_TIME_RESOLUTION) + 1):] = 0

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
            last_time_idxs = (np.maximum(specs_t, BP_ACTION_T_LIMITS[0]) / TRAJECTORY_TIME_RESOLUTION).astype(int) + 1
            for i, trajectory in enumerate(ftrajectories_s):
                # calculate trajectory time index for t = max(specs_t[i], BP_ACTION_T_LIMITS[0])
                # if spec.t < BP_ACTION_T_LIMITS[0], extrapolate trajectories between spec.t and BP_ACTION_T_LIMITS[0]
                if specs_t[i] < last_time_idxs[i]:
                    spec_t_idx = int(specs_t[i] / TRAJECTORY_TIME_RESOLUTION) + 1
                    times_beyond_spec = np.arange(spec_t_idx, last_time_idxs[i]) * TRAJECTORY_TIME_RESOLUTION - specs_t[
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

        return NumpyUtils.is_in_limits(lateral_accelerations, LAT_ACC_LIMITS).all(axis=-1)

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
        wJ, _, wT = BP_JERK_S_JERK_D_TIME_WEIGHTS_FOLLOW_LANE[AggressivenessLevel.CALM.value]
        brake_dist = action_distances[(ActionType.FOLLOW_LANE.name.lower(), wT, wJ)][
                     FILTER_V_0_GRID.get_index(action_spec.v), FILTER_A_0_GRID.get_index(0), :]
        return (brake_dist[FILTER_V_T_GRID.get_indices(vel_limit_in_points)] < dist_to_points).all()
