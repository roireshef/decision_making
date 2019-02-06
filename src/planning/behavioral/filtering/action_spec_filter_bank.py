from collections import defaultdict

from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, EPS, LAT_ACC_LIMITS, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
import numpy as np
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterLimitsViolatingTrajectory
from decision_making.src.planning.types import C_K, C_V, FS_SX, FS_DX
from decision_making.src.planning.utils.numpy_utils import NumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D

from typing import List


class AlwaysFalse(ActionSpecFilter):
    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralState) -> List[ActionSpec]:
        return [None]*len(action_specs)


class FilterByLateralAcceleration(ActionSpecFilter):
    def __init__(self, predicates_dir: str):
        self.predicates = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'limits')
        self.distances = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'distances')

    def filter(self, action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> List[ActionSpec]:
        """
        Check violation of lateral acceleration for action_specs, and beyond action_specs check ability to brake
        before all future curves using any static action.
        :return: specs list that passed the lateral acceleration filter
        """
        # first check lateral acceleration limits for all baseline trajectories of all action_specs
        meet_limits = FilterByLateralAcceleration.check_lateral_acceleration_limits(action_specs, behavioral_state)

        # now check ability to break before future curves beyond the baseline specs' trajectories
        resulting_specs = []
        for spec_idx, spec in enumerate(action_specs):

            if spec is None or not meet_limits[spec_idx]:
                resulting_specs.append(None)
                continue

            target_lane_frenet = behavioral_state.extended_lane_frames[spec.relative_lane]  # the target GFF
            # get the Frenet point index near the goal action_spec.s
            spec_s_point_idx = target_lane_frenet.get_index_on_frame_from_s(np.array([spec.s]))[0][0]
            # find all Frenet points beyond spec.s, where velocity limit (by curvature) is lower then spec.v
            beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx + 1, len(target_lane_frenet.k), 4))
            curvatures = np.maximum(np.abs(target_lane_frenet.k[beyond_spec_frenet_idxs, 0]), EPS)

            points_velocity_limits = np.sqrt(LAT_ACC_LIMITS[1] / curvatures)
            slow_points = np.where(points_velocity_limits < spec.v)[0]  # points that require braking after spec

            # if all points beyond the spec have velocity limit higher than spec.v, so no need to brake
            if len(slow_points) == 0:
                resulting_specs.append(spec)
                continue  # don't remove the spec

            # check the ability to brake beyond the spec for all points with limited velocity
            is_able_to_brake = ActionSpecFilter.check_ability_to_brake_beyond_spec(
                spec, behavioral_state.extended_lane_frames[spec.relative_lane],
                beyond_spec_frenet_idxs[slow_points], points_velocity_limits[slow_points], self.distances)

            resulting_specs.append(spec if is_able_to_brake else None)

        return resulting_specs

    @staticmethod
    def check_lateral_acceleration_limits(action_specs: List[ActionSpec], behavioral_state: BehavioralGridState) -> np.array:
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

            frenet = behavioral_state.extended_lane_frames[rel_lane]  # the target GFF
            ego_fstate = behavioral_state.projected_ego_fstates[rel_lane]
            ego_fstates = np.tile(ego_fstate, len(lane_specs)).reshape((len(lane_specs), -1))

            # calculate polynomial coefficients of the spec's Frenet trajectory for s axis
            A_inv = QuinticPoly1D.inverse_time_constraints_tensor(specs_t)
            poly_coefs_s = np.einsum(
                'ijk, ik -> ij', A_inv, np.hstack((ego_fstates[:, FS_SX:FS_DX], goal_fstates[:, FS_SX:FS_DX])))

            # create Frenet trajectories for s axis for all trajectories of rel_lane and for all time samples
            ftrajectories_s = QuinticPoly1D.polyval_with_derivatives(poly_coefs_s, time_samples)
            # assign near-zero velocity to ftrajectories_s beyond spec.t
            for i, trajectory in enumerate(ftrajectories_s):
                trajectory[int(specs_t[i] / TRAJECTORY_TIME_RESOLUTION) + 1:] = np.array([EPS, EPS, 0])
            # assign zeros to the lateral movement of ftrajectories
            ftrajectories = np.concatenate((ftrajectories_s, np.zeros_like(ftrajectories_s)), axis=-1)

            # convert Frenet to cartesian trajectories
            lane_center_ctrajectories = frenet.ftrajectories_to_ctrajectories(ftrajectories)

            # calculate lateral accelerations
            lateral_accelerations[np.array(indices_by_rel_lane[rel_lane])] = \
                lane_center_ctrajectories[..., C_K] * lane_center_ctrajectories[..., C_V] ** 2

        return NumpyUtils.is_in_limits(lateral_accelerations, LAT_ACC_LIMITS).all(axis=-1)
