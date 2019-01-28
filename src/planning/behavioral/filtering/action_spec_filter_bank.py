from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, EPS, LAT_ACC_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
import numpy as np
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterLimitsViolatingTrajectory
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import CostBasedBehavioralPlanner
from decision_making.src.planning.types import C_K, C_V


class FilterIfNone(ActionSpecFilter):
    def filter(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        return (action_spec and behavioral_state) is not None


class AlwaysFalse(ActionSpecFilter):
    def filter(self, action_spec: ActionSpec, behavioral_state: BehavioralState) -> bool:
        return False


class FilterByLateralAcceleration(ActionSpecFilter):
    def __init__(self, predicates_dir: str):
        self.predicates = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir)

    def filter(self, action_spec: ActionSpec, behavioral_state: BehavioralGridState) -> bool:
        timestamp = behavioral_state.ego_state.timestamp_in_sec
        rel_lane = action_spec.relative_lane
        frenet = behavioral_state.extended_lane_frames[rel_lane]

        samplable_trajectory = CostBasedBehavioralPlanner.generate_baseline_trajectory(
            timestamp, action_spec, frenet, behavioral_state.projected_ego_fstates[rel_lane])
        time_samples = timestamp + np.arange(min(3, action_spec.t/2), action_spec.t + EPS, 2*TRAJECTORY_TIME_RESOLUTION)
        trajectory = samplable_trajectory.sample(time_samples)
        lateral_accelerations = np.abs(trajectory[:, C_K] * trajectory[:, C_V] ** 2)
        low_acc_limit = LAT_ACC_LIMITS[1] - 0.2
        if not (lateral_accelerations <= low_acc_limit).all():
            print('lat filter: v=%.2f spec.v=%.2f lat_a=%4.2f' %
                  (behavioral_state.ego_state.cartesian_state[C_V], action_spec.v, np.max(lateral_accelerations)))
            return False

        spec_s_point_idx, _ = frenet.get_index_on_frame_from_s(np.array([action_spec.s]))
        beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx[0] + 1, len(frenet.k), 4))
        curvatures = frenet.k[beyond_spec_frenet_idxs, 0]
        max_velocities = np.sqrt(low_acc_limit / np.maximum(np.abs(curvatures), EPS))
        suspected_idxs = np.where(max_velocities < action_spec.v)[0]
        return ActionSpecFilter.check_velocity_limit_beyond_spec(
            action_spec, behavioral_state.extended_lane_frames[action_spec.relative_lane],
            beyond_spec_frenet_idxs[suspected_idxs], max_velocities[suspected_idxs], self.predicates)
