from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from decision_making.src.global_constants import LON_ACC_LIMITS, TRAJECTORY_TIME_RESOLUTION, \
    EPS
from decision_making.src.planning.types import FS_DA
from decision_making.src.planning.types import FS_SV, FS_SA, BoolArray, Limits, FS_SX
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.numpy_utils import NumpyUtils as DMNumpyUtils
from decision_making.src.planning.utils.optimal_control.poly1d import QuarticPoly1D
from decision_making.src.rl_agent.environments.action_space.common.data_objects import RLActionSpec, LaneChangeEvent
from decision_making.src.rl_agent.environments.action_space.filters.action_spec_filter import ActionSpecFilter
from decision_making.src.rl_agent.environments.state_space.common.data_objects import EgoCentricState, RelativeLane
from decision_making.src.rl_agent.environments.uc_rl_map import MapAnchor
from decision_making.src.rl_agent.global_types import LateralDirection
from decision_making.src.rl_agent.global_types import LongitudinalDirection
from decision_making.src.rl_agent.global_types import RSSParams
from decision_making.src.rl_agent.utils.primitive_utils import ListUtils
from decision_making.src.rl_agent.utils.safety_utils import SafetyUtils
from decision_making.src.rl_agent.utils.samplable_werling_trajectory import SamplableWerlingTrajectory, Utils


class ActionSpecFilterWithId(ActionSpecFilter, metaclass=ABCMeta):
    def __init__(self):
        """
        Base class for filter implementations that act on ActionSpec and returns a boolean value that corresponds to
        whether the ActionSpec satisfies the constraint in the filter. All filters have to get as input ActionSpec
        (or one of its children) and  BehavioralGridState (or one of its children) even if they don't actually use them.
        """
        # a unique automatically incremented ID for each filter class
        self.id = inverse_filter_map[self.__class__]


class FilterIfNone(ActionSpecFilterWithId):
    """ Filter if action_spec is None """

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        return np.array([False if spec is None else True for spec in action_specs])


class FilterForTerminalLaneSpeedLimits(ActionSpecFilterWithId):
    """ Filter if the terminal velocity doesn't comply with maximum limit of the currently driven lane """
    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        cur_seg_speed_limit = state.self_gff.get_speed_limit_for_lane_segment(state.ego_state.lane_segment)
        v_T = np.array([aspec.v for aspec in action_specs])
        return v_T <= cur_seg_speed_limit


class FilterForTrajectoryLaneSpeedLimits(ActionSpecFilterWithId):
    """ Filter speed within the next decision window (vs map speed limit) """
    def __init__(self, dt: float, horizon: float):
        super().__init__()

        self.dt = dt
        self.horizon = horizon
        self.timesteps = np.arange(0, self.horizon + EPS, dt)

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        seg_speeds = np.concatenate((state.self_gff.segments_speed_limits, [np.inf]))

        poly_s_coefs = np.array([aspec.baseline_trajectory.poly_s_coefs for aspec in action_specs])
        s_values = Math.polyval2d(poly_s_coefs, self.timesteps)
        v_values = Math.polyval2d(Math.polyder2d(poly_s_coefs, 1), self.timesteps)

        # s to lane index in gff
        pointwise_speed_limits = seg_speeds[np.searchsorted(state.self_gff.segments_s_offsets, s_values, side='right')-1]

        return np.all(v_values <= pointwise_speed_limits, axis=1)


class FilterForSpeedLimits(ActionSpecFilterWithId):
    """ Filter speed within the next decision window (vs map speed limit) """
    def __init__(self, dt: float, horizon: float, speed_limit: float):
        super().__init__()

        self.dt = dt
        self.horizon = horizon
        self.timesteps = np.arange(0, self.horizon + EPS, dt)
        self.speed_limit = speed_limit

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        poly_s_coefs = np.array([aspec.baseline_trajectory.poly_s_coefs for aspec in action_specs])
        v_values = Math.polyval2d(Math.polyder2d(poly_s_coefs, 1), self.timesteps)

        return np.all(v_values <= self.speed_limit, axis=1)


class FilterForTargetLaneExists(ActionSpecFilterWithId):
    """ Filters lane change actions if target lane does not exist. """

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        target_relative_lanes = np.array([action_spec.relative_lane for action_spec in action_specs])
        return np.isin(target_relative_lanes, state.valid_target_lanes)


class FilterCommitLaneChangesCommitments(ActionSpecFilterWithId):
    """ This filter is applicable only to RLCommitLaneChangeRecipe based action_specs """

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        if state.ego_state.lane_change_state.is_committing:
            original_lateral_dir = LateralDirection(state.ego_state.lane_change_state.target_relative_lane.value)
            return np.array([action_spec.recipe.lateral_dir == original_lateral_dir for action_spec in action_specs])

        return np.full(len(action_specs), True)


class FilterLateralOffsetCommitments(ActionSpecFilterWithId):
    """ This filter is applicable only to RLLateralOffsetRecipe based action_specs """

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        target_relative_lanes = np.array([action_spec.relative_lane for action_spec in action_specs])

        if state.ego_state.lane_change_state.is_aborting:
            return target_relative_lanes == RelativeLane.SAME_LANE

        if state.ego_state.lane_change_state.is_committing:
            moved_offset = state.roads_map.gff_lateral_offsets[(state.ego_state.lane_change_state.source_gff,
                                                                state.ego_state.gff_id)]
            return target_relative_lanes == state.ego_state.lane_change_state.target_relative_lane - moved_offset

        return np.full(len(action_specs), True)


class FilterForFrenetCurvature(ActionSpecFilterWithId):
    """ Parallel implementation of lateral acceleration validation based on trajectory's curvatures and velocities """
    # TODO: generalize to use real road geometry and compute curvature in Cartesian
    def __init__(self, limits: Limits, dt: float):
        super().__init__()
        self.limits = limits
        self.dt = dt

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        T_d = np.array([aspec.baseline_trajectory.T_d for aspec in action_specs])
        trajectories = np.array([aspec.baseline_trajectory for aspec in action_specs])

        result = np.full(len(action_specs), True)

        # if trajectory is shorter than sim-time-window (usually 0.1sec), then skip the filtering for it
        is_relevant = T_d > self.dt
        if not np.any(is_relevant):
            return result

        # compute frenet-curvature for all relevant trajectories and override results at the corresponding positions
        relevant_fcurvatures = self._compute_curvatures(trajectories[is_relevant], T_d[is_relevant])
        result[is_relevant] = np.all(DMNumpyUtils.is_in_limits(relevant_fcurvatures, self.limits), axis=1)

        return result

    def _compute_curvatures(self, trajectories: List[SamplableWerlingTrajectory], T_d: np.ndarray):
        # generate a 2d array of timepoints for all trajectories (if T_d differs between them, it will zero-padded)
        # if T_d is the same for all trajectories, the first dimension of time_points is null
        # otherwise, send a time_points array zipped to each trajectory
        if np.all(T_d == T_d[0]):
            time_points = np.arange(0, T_d[0]+EPS, self.dt)[np.newaxis, :]
            interpolation_mask = 1
        else:
            # np.arange(0, T_d) for each trajectory's T_d and zero-pad according to longest row
            relevant_lens = (T_d + EPS) // self.dt + 1
            interpolation_mask = relevant_lens[:, None] > np.arange(relevant_lens.max())
            time_points = np.zeros(interpolation_mask.shape, dtype=float)
            time_points[interpolation_mask] = np.concatenate([np.arange(0, t+EPS, self.dt) for t in T_d])

        # sample trajectories with time_points (that are zero-padded after lane change is over)
        fstates = Utils.sample_frenet(trajectories, time_points)

        # compute curvatures = acceleration / velocity ** 2
        return DMNumpyUtils.div(fstates[..., FS_DA] * interpolation_mask, fstates[..., FS_SV] ** 2 * interpolation_mask)


class FilterForLonAcceleration(ActionSpecFilterWithId):
    def __init__(self, lon_acceleration_limits: Limits = LON_ACC_LIMITS):
        """
        Filters by longitudinal acceleration
        :param lon_acceleration_limits: min/max acceleration limits to filter according to
        """
        super().__init__()

        self._lon_acceleration_limits = lon_acceleration_limits

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        v_0 = state.ego_state.fstate[FS_SV]
        a_0 = state.ego_state.fstate[FS_SA]

        v_T_array, T_array = np.array([[action_spec.v, action_spec.t] for action_spec in action_specs]).T

        # validate acceleration limits of the initial quartic action
        poly_s = np.zeros((T_array.shape[0], QuarticPoly1D.num_coefs()))
        tracking_mode_T = (T_array < TRAJECTORY_TIME_RESOLUTION)
        validT = np.logical_and(~np.isnan(T_array), np.logical_not(tracking_mode_T))
        a_0_array = np.full_like(v_T_array[validT], a_0)
        v_0_array = np.full_like(v_T_array[validT], v_0)
        poly_s[validT] = QuarticPoly1D.position_profile_coefficients(a_0_array, v_0_array, v_T_array[validT],
                                                                     T_array[validT])
        valid_acc = np.zeros_like(T_array).astype(bool)
        valid_acc[validT] = QuarticPoly1D.are_accelerations_in_limits(poly_s[validT], T_array[validT],
                                                                      self._lon_acceleration_limits)
        valid_acc[tracking_mode_T] = True

        return valid_acc


class FilterForLonAccelerationAndJerk(ActionSpecFilterWithId):
    def __init__(self,  lon_acceleration_limits: Limits = LON_ACC_LIMITS, jerk_limits: Limits = np.array([-5., 5.])):
        """
        Filters by longitudinal acceleration
        :param lon_acceleration_limits: min/max acceleration limits to filter according to
        """
        super().__init__()

        self._jerk_limits = jerk_limits
        self._lon_acceleration_limits = lon_acceleration_limits

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        v_0 = state.ego_state.fstate[FS_SV]
        a_0 = state.ego_state.fstate[FS_SA]

        v_T_array = np.array([action_spec.v for action_spec in action_specs])
        T_array = np.array([action_spec.t for action_spec in action_specs])

        # validate acceleration limits of the initial quartic action
        poly_s = np.zeros((T_array.shape[0], QuarticPoly1D.num_coefs()))
        tracking_mode_T = (T_array < TRAJECTORY_TIME_RESOLUTION)
        validT = np.logical_and(~np.isnan(T_array), np.logical_not(tracking_mode_T))
        a_0_array = np.full_like(v_T_array[validT], a_0)
        v_0_array = np.full_like(v_T_array[validT], v_0)
        poly_s[validT] = QuarticPoly1D.position_profile_coefficients(a_0_array, v_0_array, v_T_array[validT],
                                                                     T_array[validT])
        valid_jerk = np.zeros_like(T_array).astype(bool)
        valid_acc = np.zeros_like(T_array).astype(bool)

        valid_acc[validT] = QuarticPoly1D.are_accelerations_in_limits(poly_s[validT], T_array[validT], self._lon_acceleration_limits)
        valid_jerk[validT] = QuarticPoly1D.are_derivatives_in_limits(3, poly_s[validT], T_array[validT], self._jerk_limits)

        return np.logical_or(np.logical_and(valid_jerk, valid_acc), tracking_mode_T)


class FilterForSafetyGeneric(ActionSpecFilterWithId, metaclass=ABCMeta):
    """ Abstract class for safety filter based on RSS for lane change actions that use polynomial-based trajectories """

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        relative_lanes, T_d = np.array([[action_spec.relative_lane, action_spec.baseline_trajectory.T_d]
                                            for action_spec in action_specs]).T
        trajectories = np.array([action_spec.baseline_trajectory for action_spec in action_specs])
        poly_d = np.array([action_spec.baseline_trajectory.poly_d_coefs for action_spec in action_specs])

        if state.ego_state.is_on_source_gff:
            source_relative_lanes = np.full(len(action_specs), RelativeLane.SAME_LANE)
            target_relative_lanes = relative_lanes
            is_lane_change_action = target_relative_lanes != RelativeLane.SAME_LANE
            lane_change_direction = target_relative_lanes - source_relative_lanes
        else:
            source_relative_lanes = np.full(len(action_specs), ~state.ego_state.lane_change_state.target_relative_lane)
            target_relative_lanes = np.full(len(action_specs), RelativeLane.SAME_LANE)
            is_lane_change_action = np.full(len(action_specs), True)
            lane_change_direction = target_relative_lanes - source_relative_lanes

        # Check versus lead vehicle in target lane (normal headway control at target lane)
        # This applies to both LC and non-LC actions based on target lane
        safe_ahead_on_target_lane = self.is_safe_ahead_on_target_lane(state, lane_change_direction, trajectories,
                                                                      is_lane_change_action, T_d, poly_d,
                                                                      source_relative_lanes, target_relative_lanes)

        safe_ahead_on_source_lane = np.full(len(action_specs), True)
        safe_behind_on_target_lane = np.full(len(action_specs), True)

        indices_lc = np.where(is_lane_change_action)[0]
        if len(indices_lc) > 0:
            T_d_lc = T_d[indices_lc]
            poly_d_lc = poly_d[indices_lc]
            direction_lc = lane_change_direction[indices_lc]
            trajectories_lc = trajectories[indices_lc]
            source_relative_lanes_lc = source_relative_lanes[indices_lc]
            target_relative_lanes_lc = target_relative_lanes[indices_lc]

            # get target-lane's width
            source_gff = state.ego_state.lane_change_state.source_gff or state.ego_state.gff_id
            adjacent_lane_widths = {
                LateralDirection(rel_lane.value):
                    state.roads_map.get_gff_width_at(adj_gff_id, state.ego_fstate_on_gffs[adj_gff_id][FS_SX])
                for rel_lane, adj_gff_id in state.roads_map.gff_adjacency[source_gff].items()
            }
            target_lane_widths_lc = np.array([adjacent_lane_widths[rel_dir] for rel_dir in direction_lc])

            current_dx_on_target_gff = Math.polyval2d(poly_d_lc, np.array([0])).squeeze(-1)

            # Note: This offset is relative to target lane(!) since poly_d is relative to target lane
            cut_st_offsets_lc = -direction_lc.astype(int) * (target_lane_widths_lc + state.ego_state.size.width) / 2
            is_before_cut_st = abs(current_dx_on_target_gff) >= np.abs(cut_st_offsets_lc)
            if state.ego_state.lane_change_state.is_inactive or state.ego_state.lane_change_state.is_negotiating or \
                    (state.ego_state.lane_change_state.is_committing and state.ego_state.is_on_source_gff
                     and np.any(is_before_cut_st)):
                safe_behind_on_target_lane[indices_lc] = self.is_safe_behind_on_target_lane(
                    state, direction_lc, cut_st_offsets_lc,
                    trajectories_lc, T_d_lc, poly_d_lc,
                    source_relative_lanes_lc, target_relative_lanes_lc)

            # Check versus lead vehicle in source lane - only up to cut-in end (leaving source lane)
            # or next decision time (earliest of the two)
            cut_en_offsets_lc = -direction_lc.astype(int) * (target_lane_widths_lc - state.ego_state.size.width) / 2
            is_before_cut_en = np.abs(current_dx_on_target_gff) >= np.abs(cut_en_offsets_lc)
            if np.any(is_before_cut_en):
                safe_ahead_on_source_lane[indices_lc] = self.is_safe_ahead_on_source_lane(
                    state, direction_lc, cut_en_offsets_lc,
                    trajectories_lc, T_d_lc, poly_d_lc,
                    source_relative_lanes_lc, target_relative_lanes_lc)

        return safe_ahead_on_source_lane & safe_ahead_on_target_lane & safe_behind_on_target_lane

    @abstractmethod
    def is_safe_ahead_on_target_lane(self, state: EgoCentricState, lc_dir: np.ndarray, trajectories: np.ndarray,
                                     is_lane_change_action: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                     source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        pass

    @abstractmethod
    def is_safe_behind_on_target_lane(self, state: EgoCentricState, lc_dir: np.ndarray, cut_st_offsets: np.ndarray,
                                      trajectories: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                      source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        pass

    @abstractmethod
    def is_safe_ahead_on_source_lane(self, state: EgoCentricState, lc_dir: np.ndarray, cut_en_offset: np.ndarray,
                                     trajectories: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                     source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        pass

    @staticmethod
    def _find_time(poly: np.ndarray, offset: float, T: float) -> np.ndarray:
        """
        solve A(t) = B for t within limits [0, T], where A(t) is a polynomial and B a scalar
        :param poly: 1d numpy array of 6 coefficients of a polynomial A(t)
        :param offset: scalar B
        :param T: scalar [sec]. upper time limit
        :return:
        """
        roots = Math.find_real_roots_in_limits(polynomials=poly - offset * np.array([0, 0, 0, 0, 0, 1]),
                                               value_limits=np.array([0, T]))
        return np.fmin.reduce(roots, axis=-1)


class FilterForSafetyWithSpeedChanges(FilterForSafetyGeneric):
    def __init__(self, rss_params: RSSParams, dt: float, headway: float, perception_horizon: float):
        """
        Safety (RSS) Filter for lane change actions that allow negotiation and can change speeds in every decision step.
        Note that if headway < decision resolution then safety check isn't bulletproof
        :param rss_params: RSS parameters to use for safety checks
        :param dt: [sec] time resolution to use for sampling (generally simulation time-step)
        :param headway: [sec] time horizon to use for safety check against lead vehicles (headway control)
        lane changes (after cut-end). If set to False, lane changes are tested for completion with constant speed
        :param perception_horizon: [m] agent's perception horizon to use if no actual vehicle is found (for dummy vehicles)
        """
        super().__init__()
        self.rss_params = rss_params
        self.dt = dt
        self.headway = headway
        self.perception_horizon = perception_horizon

    def is_safe_ahead_on_target_lane(self, state: EgoCentricState, lc_dir: np.ndarray, trajectories: np.ndarray,
                                     is_lane_change_action: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                     source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        return SafetyUtils.is_future_safe_directional(
            lon_direction=LongitudinalDirection.AHEAD,
            state=state,
            timestamps=np.arange(0, self.headway + EPS, self.dt)[np.newaxis, :],
            lanes=target_relative_lanes,
            trajectories=trajectories,
            rss_params=self.rss_params.with_values(ro=0),
            perception_horizon=self.perception_horizon)

    def is_safe_behind_on_target_lane(self, state: EgoCentricState, lc_dir: np.ndarray, cut_st_offsets: np.ndarray,
                                      trajectories: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                      source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        # Note: this assumes the lateral motion of all lane change actions in the arguments are symmetric!
        T_cut_st = max(0, np.nan_to_num(self._find_time(poly_d[0], cut_st_offsets[0], T_d[0])).item())

        return SafetyUtils.is_future_safe_directional(
            lon_direction=LongitudinalDirection.BEHIND,
            state=state,
            timestamps=np.arange(Math.floor_to_step(T_cut_st, self.dt),
                                 Math.ceil_to_step(T_cut_st, self.dt) + EPS, self.dt)[np.newaxis, :],
            lanes=target_relative_lanes,
            trajectories=trajectories,
            rss_params=self.rss_params,
            perception_horizon=self.perception_horizon)

    def is_safe_ahead_on_source_lane(self, state: EgoCentricState, lc_dir: np.ndarray, cut_en_offset: np.ndarray,
                                     trajectories: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                     source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        # Note: this assumes the lateral motion of all lane change actions in the arguments are symmetric!
        T_cut_en = np.nan_to_num(self._find_time(poly_d[0], cut_en_offset[0], T_d[0])).item()

        timestamps = np.arange(0, Math.ceil_to_step(np.minimum(T_cut_en, self.headway) + EPS, self.dt) + EPS, self.dt)
        return SafetyUtils.is_future_safe_directional(
            lon_direction=LongitudinalDirection.AHEAD,
            state=state,
            timestamps=timestamps[np.newaxis, :],
            lanes=source_relative_lanes,
            trajectories=trajectories,
            rss_params=self.rss_params.with_values(ro=0),
            perception_horizon=self.perception_horizon)


class FilterForSafetyWithConstantSpeedAtLaneChange(FilterForSafetyGeneric):
    def __init__(self, rss_params: RSSParams, dt: float, headway: float, perception_horizon: float):
        """
        Safety (RSS) Filter for lane change actions that do not change speeds during lane changes.
        Note that if headway < decision resolution then safety check isn't bulletproof.

        Note: Only use it in case you are using the ChangeLaneAtConstantSpeedActionSpaceAdapter which has different
        underlying assumptions for creating actions than the 2-phase (negotiation) action space (for lane changes)
        :param rss_params: RSS parameters to use for safety checks
        :param dt: [sec] time resolution to use for sampling (generally simulation time-step)
        :param headway: [sec] time horizon to use for safety check against lead vehicles (headway control)
        lane changes (after cut-end). If set to False, lane changes are tested for completion with constant speed
        :param perception_horizon: [m] agent's perception horizon to use if no actual vehicle is found (for dummy vehicles)
        """
        super().__init__()
        self.rss_params = rss_params
        self.dt = dt
        self.headway = headway
        self.perception_horizon = perception_horizon

    def is_safe_ahead_on_target_lane(self, state: EgoCentricState, lc_dir: np.ndarray, trajectories: np.ndarray,
                                     is_lane_change_action: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                     source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        safe_ahead_on_target_lane = np.full(len(is_lane_change_action), True)
        if np.any(is_lane_change_action):
            # This assumes all lane change actions have same T_d!
            safe_ahead_on_target_lane[is_lane_change_action] = SafetyUtils.is_future_safe_directional(
                lon_direction=LongitudinalDirection.AHEAD,
                state=state,
                # TODO: should probably add self.headway here?
                timestamps=np.arange(0, T_d[is_lane_change_action][0] + EPS, self.dt)[np.newaxis, :],
                lanes=target_relative_lanes[is_lane_change_action],
                trajectories=trajectories[is_lane_change_action],
                rss_params=self.rss_params.with_values(ro=0),
                perception_horizon=self.perception_horizon)

        if np.any(~is_lane_change_action):
            safe_ahead_on_target_lane[~is_lane_change_action] = SafetyUtils.is_future_safe_directional(
                lon_direction=LongitudinalDirection.AHEAD,
                state=state,
                timestamps=np.arange(0, self.headway + EPS, self.dt)[np.newaxis, :],
                lanes=target_relative_lanes[~is_lane_change_action],
                trajectories=trajectories[~is_lane_change_action],
                rss_params=self.rss_params.with_values(ro=0),
                perception_horizon=self.perception_horizon)

        return safe_ahead_on_target_lane

    def is_safe_behind_on_target_lane(self, state: EgoCentricState, lc_dir: np.ndarray, cut_st_offsets: np.ndarray,
                                      trajectories: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                      source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        # Note: this assumes the lateral motion of all lane change actions in the arguments are symmetric!
        T_cut_st = max(0, np.nan_to_num(self._find_time(poly_d[0], cut_st_offsets[0], T_d[0])).item())

        return SafetyUtils.is_future_safe_directional(
            lon_direction=LongitudinalDirection.BEHIND,
            state=state,
            timestamps=np.arange(Math.floor_to_step(T_cut_st, self.dt),
                                 Math.ceil_to_step(T_cut_st, self.dt) + EPS, self.dt)[np.newaxis, :],
            lanes=target_relative_lanes,
            trajectories=trajectories,
            rss_params=self.rss_params,
            perception_horizon=self.perception_horizon)

    def is_safe_ahead_on_source_lane(self, state: EgoCentricState, lc_dir: np.ndarray, cut_en_offset: np.ndarray,
                                     trajectories: np.ndarray, T_d: np.ndarray, poly_d: np.ndarray,
                                     source_relative_lanes: np.ndarray, target_relative_lanes: np.ndarray):
        # Note: this assumes the lateral motion of all lane change actions in the arguments are symmetric!
        T_cut_en = np.nan_to_num(self._find_time(poly_d[0], cut_en_offset[0], T_d[0])).item()

        timestamps = np.arange(0, Math.ceil_to_step(np.minimum(T_cut_en, self.headway) + EPS, self.dt) + EPS, self.dt)
        return SafetyUtils.is_future_safe_directional(
            lon_direction=LongitudinalDirection.AHEAD,
            state=state,
            timestamps=timestamps[np.newaxis, :],
            lanes=source_relative_lanes,
            trajectories=trajectories,
            rss_params=self.rss_params.with_values(ro=0),
            perception_horizon=self.perception_horizon)


class DontCommitLCIntoIntersections(ActionSpecFilterWithId):
    """ Filters out lane change actions that do net terminate before next intersection segment """
    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        if state.ego_state.is_on_source_gff:
            target_relative_lanes = np.array([action_spec.relative_lane for action_spec in action_specs])
            is_in_lc = target_relative_lanes != RelativeLane.SAME_LANE
        else:
            target_relative_lanes = np.full(len(action_specs), RelativeLane.SAME_LANE)
            is_in_lc = np.full(len(action_specs), True)

        is_valid = np.full(len(action_specs), True)
        if np.any(is_in_lc):
            # get relative lane (and gff) of first is_in_lc action
            # (all of them are the same - committing to same target)
            first_lc_idx = np.argwhere(is_in_lc)[0].item()
            target_gff = action_specs[first_lc_idx].baseline_trajectory.gff_id
            ego_sx_on_target_gff = state.ego_fstate_on_adj_lanes[target_relative_lanes[first_lc_idx]][FS_SX]

            sx_limit = state.roads_map.gffs[target_gff].next_intersection_offset(sx=ego_sx_on_target_gff)

            is_valid[is_in_lc] = [aspec.baseline_trajectory.get_lon_position_at_Td() < sx_limit
                                  for aspec, lc in zip(action_specs, is_in_lc) if lc]

        return is_valid


class OnlyAllowNoOpWhenChangingLanes(ActionSpecFilterWithId):
    """ Only allows NOOP action to be executed while lane change is active. This is useful in cases where
      ChangeLaneAtConstantSpeedActionSpaceAdapter is used (use it for all action spaces involved) """
    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        if state.ego_state.lane_change_state.is_active:
            return np.array([aspec.recipe.is_noop for aspec in action_specs], dtype=bool)
        else:
            return np.full(len(action_specs), True)


# TODO: Generalize beyond reacting for Merge-specific MapAnchor.YIELD_LINE (Use full PoNR interface)
class FilterForPointOfNoReturn(ActionSpecFilterWithId):
    def __init__(self, horizon: float, brake_decel: float, rss_params: RSSParams, perception_horizon: float,
                 brake_slack: float):
        """
        Filter out actions that don't enable to brake before a speed limit (such as traffic control device, yield line)
        :param horizon: point in time relative to beginning of action [sec] to check for
        :param brake_decel: maximal braking deceleration for point of no return calculation. it is necessary that the
        vehicle will have an action that has equal or greater deceleration, to maintain consistency with the safety
        analysis in this filter.
        :param brake_slack: minimal allowed distance in meters to stop before the anchor
        """
        super().__init__()
        self.rss_params = rss_params
        self.horizon = horizon
        self.brake_decel = brake_decel
        self.perception_horizon = perception_horizon
        self.brake_slack = brake_slack

    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        # if committed to lane change, safety is already guaranteed (taken care of by other filters)
        if state.ego_state.lane_change_state.is_committing:
            return np.full(len(action_specs), True)

        # get anchor TODO: generalize to target velocity (other than 0)
        anchor_sx = np.array([state.map_anchors[aspec.baseline_trajectory.gff_id].get(MapAnchor.YIELD_LINE, np.inf)
                              for aspec in action_specs])

        # sample trajectories in parallel using Utils class
        trajectories = [aspec.baseline_trajectory for aspec in action_specs]
        now_fstate, horizon_fstate = Utils.sample_frenet(
            trajectories, np.array([[0, self.horizon]])).transpose([1, 0, 2])

        # braking distance = (v_0 - v_t)^ 2 / 2*a
        braking_distances = horizon_fstate[:, FS_SV] ** 2 / (2 * self.brake_decel)

        # is taking this action makes ego cross PoNR at time <self.horizon> from now?
        can_brake_to_anchor = horizon_fstate[:, FS_SX] + braking_distances < anchor_sx - self.brake_slack
        already_past_anchor = now_fstate[:, FS_SX] >= anchor_sx

        is_safe = can_brake_to_anchor | already_past_anchor
        if np.sum(~is_safe) > 0:
            is_safe[~is_safe] = self._action_is_safe_for_crossing_intersection(
                action_specs=ListUtils.slice_list(action_specs, ~is_safe),
                lon_targets=anchor_sx[~is_safe],
                state=state
            )

        return is_safe

    def _action_is_safe_for_crossing_intersection(self, action_specs: List[RLActionSpec], lon_targets: np.ndarray,
                                                  state: EgoCentricState) -> BoolArray:
        """
        Checks safety against read and front vehicles in a merge. The underlying heuristic policy for each action is to
        follow its velocity profile until terminal state, and then keep constant speed. Doing so, the "cut-in" moment
        is tested for RSS at the longitudinal targets (merge zone entrance)
        :param action_specs: list of action specs to validate safety for. Each serves as a policy for crossing the merge
        :param lon_targets: the sx value of the entrance to the merge zone (ego CG, relative to GFF)
        :param state: the state of the environment
        :return: array of boolean result (action is safe or not)
        """
        if state.ego_state.is_on_source_gff:
            next_intersection_segments = state.roads_map.next_intersection_segments_on_gff(
                state.ego_state.gff_id, state.ego_state.fstate[FS_SX])
            next_merges = [intersection_segment for intersection_segment in next_intersection_segments
                           if intersection_segment in state.roads_map.merges.keys()]
            relevant_lane = state.roads_map.merges[next_merges[0]]

            target_lanes = np.full(len(action_specs), relevant_lane)
        else:
            target_lanes = np.full(len(action_specs), RelativeLane.SAME_LANE)

        trajectories = np.array([action_spec.baseline_trajectory for action_spec in action_specs])

        # get time of arrival to anchor, trajectory full durations (T_ext)
        # check if time of arrival to anchor isn't too far (beyond trajectory length)
        T_anchor, T_ext = np.array([[aspec.baseline_trajectory.get_lon_time_of_arrival(lon_target),
                                     aspec.baseline_trajectory.T_extended]
                                    for aspec, lon_target in zip(action_specs, lon_targets)]).T
        T_anchor_active = np.logical_and(T_anchor >= 0, T_anchor <= T_ext)

        # Check safety ahead
        safe_ahead = np.full(len(action_specs), True)
        safe_ahead[T_anchor_active] = SafetyUtils.is_future_safe_directional(
            lon_direction=LongitudinalDirection.AHEAD,
            state=state,
            timestamps=T_anchor[T_anchor_active, np.newaxis],
            lanes=target_lanes[T_anchor_active],
            trajectories=trajectories[T_anchor_active],
            rss_params=self.rss_params,
            perception_horizon=self.perception_horizon)

        # Check safety behind
        safe_behind = np.full(len(action_specs), True)
        safe_behind[T_anchor_active] = SafetyUtils.is_future_safe_directional(
            lon_direction=LongitudinalDirection.BEHIND,
            state=state,
            timestamps=T_anchor[T_anchor_active, np.newaxis],
            lanes=target_lanes[T_anchor_active],
            trajectories=trajectories[T_anchor_active],
            rss_params=self.rss_params,
            perception_horizon=self.perception_horizon)

        return safe_ahead & safe_behind


class FilterMovingLaterallyToNonExistingLanes(ActionSpecFilterWithId):
    """ Filters lateral movements towards the direction of non existing lane, except when lane change is already active
    which is when abort action is allowed although it might not be applicable according to this rule """
    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        if state.ego_state.lane_change_state.is_active:
            return np.full(len(action_specs), True)

        # this virtually creates the FINAL target (adjacent lane) for actions that start negotiations
        target_relative_lanes = np.array([RelativeLane(aspec.recipe.lateral_dir.value) for aspec in action_specs])
        return np.isin(target_relative_lanes, state.valid_target_lanes)


class PostponeCommitAfterNegotiation(ActionSpecFilterWithId):
    """ Postpone commit until 1 sec after negotiating. Useful for enforcing when planning frequency is not 1Hz,
    otherwise already captured in "specify" """
    def filter(self, action_specs: List[RLActionSpec], state: EgoCentricState) -> BoolArray:
        if not state.ego_state.lane_change_state.is_negotiating \
                or state.ego_state.lane_change_state.time_since_start(state.timestamp_in_sec) > 0.95:
            return np.full(len(action_specs), True)

        return np.array([aspec.lane_change_event == LaneChangeEvent.UNCHANGED for aspec in action_specs])


# AUTO-ID CODE #

def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


# Mapping between all subclasses of ActionSpecFilter and their ordinal ID number (and inverse mapping)
filter_map = {idx: cls for idx, cls in enumerate(get_all_subclasses(ActionSpecFilterWithId))}
inverse_filter_map = dict(zip(filter_map.values(), filter_map.keys()))

if __name__ == "__main__":
    # Prints mapping between filter hashes and their name
    print("\nFilter List (by ID):\n----------")
    for k, v in filter_map.items():
        print("%s: %s" % (k, v.__name__))
