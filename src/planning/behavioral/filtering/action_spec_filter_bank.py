from decision_making.src.global_constants import TRAJECTORY_TIME_RESOLUTION, EPS, LAT_ACC_LIMITS, \
    SPECIFICATION_MARGIN_TIME_DELAY, BP_ACTION_T_LIMITS
from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.behavioral_state import BehavioralState
from decision_making.src.planning.behavioral.data_objects import ActionSpec
from decision_making.src.planning.behavioral.filtering.action_spec_filtering import \
    ActionSpecFilter
import numpy as np
from decision_making.src.planning.behavioral.filtering.recipe_filter_bank import FilterLimitsViolatingTrajectory
from decision_making.src.planning.behavioral.planner.cost_based_behavioral_planner import CostBasedBehavioralPlanner
from decision_making.src.planning.types import C_K, C_V, C_A, FS_SA, FS_SV, FS_SX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D


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
        ego_fstate = behavioral_state.projected_ego_fstates[rel_lane]

        a_0 = np.array([ego_fstate[FS_SA]])
        v_0 = np.array([ego_fstate[FS_SV]])
        v_T = np.array([action_spec.v])
        dx = np.array([action_spec.s - ego_fstate[FS_SX]])
        T = np.array([action_spec.t])
        s_coefs = QuinticPoly1D.s_profile_coefficients(a_0, v_0, v_T, dx, T, SPECIFICATION_MARGIN_TIME_DELAY)[0]
        vel_poly = np.polyder(s_coefs, m=1)
        acc_coefs = np.polyder(vel_poly, m=1)
        suspected_times = Math.find_real_roots_in_limits(acc_coefs, value_limits=BP_ACTION_T_LIMITS)
        real_suspected_times = suspected_times[np.isfinite(suspected_times)]
        suspected_velocities = np.concatenate((v_0, v_T, np.polyval(vel_poly, real_suspected_times)))
        max_velocity = np.max(suspected_velocities)

        spec_s_point_idxs, _ = frenet.get_index_on_frame_from_s(np.array([ego_fstate[FS_SX], action_spec.s]))
        max_spec_curvature = np.max(np.abs(frenet.k[spec_s_point_idxs[0]:spec_s_point_idxs[1], 0]))
        acc_limit = LAT_ACC_LIMITS[1] - 0.2
        spec_velocity_limit = np.sqrt(acc_limit / max(max_spec_curvature, EPS))
        if max_velocity > spec_velocity_limit:

            # First create the action's baseline trajectory and check it's violation of lateral acceleration limit
            samplable_trajectory = CostBasedBehavioralPlanner.generate_baseline_trajectory(
                timestamp, action_spec, frenet, behavioral_state.projected_ego_fstates[rel_lane])

            # The current state can slightly violate the filter's acceleration limit (LAT_ACC_LIMITS[1] - 0.2)
            # because of control errors. In such case any trajectory will not pass the filter.
            # Therefore we check only the second half of the trajectory or starting from time = 3 seconds.
            time_samples = np.arange(min(3, action_spec.t/2), action_spec.t + EPS, 2*TRAJECTORY_TIME_RESOLUTION)

            # sample the samplable trajectory only for s axis
            fstates_s = QuinticPoly1D.polyval_with_derivatives(np.array([samplable_trajectory.poly_s_coefs]), time_samples)[0]
            fstates = np.hstack((fstates_s, np.zeros_like(fstates_s)))
            baseline_trajectory = frenet.ftrajectory_to_ctrajectory(fstates)

            lateral_acceleration_samples = np.abs(baseline_trajectory[:, C_K] * baseline_trajectory[:, C_V] ** 2)
            # The filter uses a bit tighter lateral acceleration limit than in TP because of control errors.
            if not (lateral_acceleration_samples <= acc_limit).all():  # then the action violates the lat_accel limit
                return False

        # find all Frenet points beyond the action spec, where velocity limit (by curvature) is lower then spec.v
        beyond_spec_frenet_idxs = np.array(range(spec_s_point_idxs[1] + 1, len(frenet.k), 4))
        curvatures = np.maximum(np.abs(frenet.k[beyond_spec_frenet_idxs, 0]), EPS)
        # extract regions with increasing curvature
        increasing_curvature_idxs = np.where(np.concatenate(([True], curvatures[:-1] + EPS < curvatures[1:])))[0]
        curvatures = curvatures[increasing_curvature_idxs]
        points_velocity_limits = np.sqrt(acc_limit / curvatures)
        slow_points = np.where(points_velocity_limits < action_spec.v)[0]  # points that require braking after spec
        if len(slow_points) == 0:
            return True  # all points beyond the spec have velocity limit higher than spec.v, so no need to brake

        # check the ability to brake beyond the spec for all points with limited velocity
        return ActionSpecFilter.check_ability_to_brake_beyond_spec(
            action_spec, behavioral_state.extended_lane_frames[action_spec.relative_lane],
            beyond_spec_frenet_idxs[increasing_curvature_idxs[slow_points]], points_velocity_limits[slow_points], self.distances)
