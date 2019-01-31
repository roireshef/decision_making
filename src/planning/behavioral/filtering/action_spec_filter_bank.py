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
        self.predicates = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'limits')
        self.distances = FilterLimitsViolatingTrajectory.read_predicates(predicates_dir, 'distances')

    def filter(self, action_spec: ActionSpec, behavioral_state: BehavioralGridState) -> bool:
        """
        Check violation of lateral acceleration by given action_spec, and an ability to brake by any static action
        beyond the action_spec.
        :return: True if there is no violation of lateral acceleration
        """
        timestamp = behavioral_state.ego_state.timestamp_in_sec
        rel_lane = action_spec.relative_lane
        frenet = behavioral_state.extended_lane_frames[rel_lane]  # the target GFF

        # First create the action's baseline trajectory and check it's violation of lateral acceleration limit
        samplable_trajectory = CostBasedBehavioralPlanner.generate_baseline_trajectory(
            timestamp, action_spec, frenet, behavioral_state.projected_ego_fstates[rel_lane])

        # The current state can slightly violate the filter's acceleration limit (LAT_ACC_LIMITS[1] - 0.2)
        # because of control errors. In such case any trajectory will not pass the filter.
        # Therefore we check only the second half of the trajectory or starting from time = 3 seconds.
        time_samples = timestamp + np.arange(min(3, action_spec.t/2), action_spec.t + EPS, 2*TRAJECTORY_TIME_RESOLUTION)
        baseline_trajectory = samplable_trajectory.sample(time_samples)
        lateral_acceleration_samples = np.abs(baseline_trajectory[:, C_K] * baseline_trajectory[:, C_V] ** 2)
        # The filter uses a bit tighter lateral acceleration limit than in TP because of control errors.
        acc_limit = LAT_ACC_LIMITS[1] - 0.2
        if not (lateral_acceleration_samples <= acc_limit).all():  # then the action violates the lat_accel limit
            return False

        # find all Frenet points beyond the action spec, where velocity limit (by curvature) is lower then spec.v
        spec_s_point_idx, _ = frenet.get_index_on_frame_from_s(np.array([action_spec.s]))
        beyond_spec_frenet_idxs = np.array(range(spec_s_point_idx[0] + 1, len(frenet.k), 4))
        points_curvatures = np.maximum(np.abs(frenet.k[beyond_spec_frenet_idxs, 0]), EPS)
        points_velocity_limits = np.sqrt(acc_limit / points_curvatures)
        slower_points = np.where(points_velocity_limits < action_spec.v)[0]  # points that require braking after spec
        if len(slower_points) == 0:
            return True  # all points beyond the spec have velocity limit higher than spec.v, so no need to brake

        # check the ability to brake beyond the spec for all points with limited velocity
        return ActionSpecFilter.check_ability_to_brake_beyond_spec(
            action_spec, behavioral_state.extended_lane_frames[action_spec.relative_lane],
            beyond_spec_frenet_idxs[slower_points], points_velocity_limits[slower_points], self.distances)
