from typing import Optional, List
import numpy as np
from decision_making.src.global_constants import BP_JERK_S_JERK_D_TIME_WEIGHTS, LON_ACC_LIMITS, VELOCITY_LIMITS, \
    EGO_LENGTH, LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT, SAFETY_HEADWAY, SPECIFICATION_HEADWAY, BP_ACTION_T_LIMITS, \
    TRAJECTORY_TIME_RESOLUTION

from decision_making.src.planning.behavioral.behavioral_grid_state import BehavioralGridState
from decision_making.src.planning.behavioral.data_objects import RelativeLane, AggressivenessLevel, ActionSpec
from decision_making.src.planning.types import FS_SX, FS_SV, FS_SA, FS_2D_LEN, FrenetState1D
from decision_making.src.planning.utils.kinematics_utils import KinematicUtils, BrakingDistances
from decision_making.src.planning.utils.math_utils import Math
from decision_making.src.planning.utils.optimal_control.poly1d import QuinticPoly1D, QuarticPoly1D
from decision_making.src.state.state import State, ObjectSize
from decision_making.src.utils.map_utils import MapUtils

MAX_BACK_HORIZON = 300   # on the main road
MAX_AHEAD_HORIZON = 100  # on the main road
MERGE_LOOKAHEAD = 300    # on the ego road


class ActorState:
    def __init__(self, size: ObjectSize, fstate: FrenetState1D):
        self.size = size
        self.fstate = fstate


class LaneMergeState:
    def __init__(self, ego_fstate: FrenetState1D, ego_size: ObjectSize, actors: List[ActorState],
                 merge_point_red_line_dist: float):
        self.ego_fstate = ego_fstate  # SX is negative: -dist_to_red_line
        self.ego_size = ego_size
        self.actors = actors
        self.merge_point_red_line_dist = merge_point_red_line_dist


class RuleBasedLaneMerge:

    TIME_GRID_RESOLUTION = 0.2
    VEL_GRID_RESOLUTION = 0.5

    @staticmethod
    def create_lane_merge_state(state: State, behavioral_state: BehavioralGridState) -> [Optional[LaneMergeState], float]:
        """
        Given the current state, find the lane merge ahead and cars on the main road upstream the merge point.
        :param state: current state
        :param behavioral_state: current behavioral grid state
        :return: lane merge state or None (if no merge), s of merge-point in GFF
        """
        gff = behavioral_state.extended_lane_frames[RelativeLane.SAME_LANE]
        ego_fstate_on_gff = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]

        # Find the lanes before and after the merge point
        merge_lane_id = MapUtils.get_closest_lane_merge(gff, ego_fstate_on_gff[FS_SX], merge_lookahead=MERGE_LOOKAHEAD)
        if merge_lane_id is None:
            return None, 0
        after_merge_lane_id = MapUtils.get_downstream_lanes(merge_lane_id)[0]
        main_lane_ids, main_lanes_s = MapUtils.get_straight_upstream_downstream_lanes(
            after_merge_lane_id, max_back_horizon=MAX_BACK_HORIZON, max_ahead_horizon=MAX_AHEAD_HORIZON)
        if len(main_lane_ids) == 0:
            return None, 0

        # calculate s of the red line, as s on GFF of the merge lane segment origin
        red_line_in_gff = gff.convert_from_segment_state(np.zeros(FS_2D_LEN), merge_lane_id)[FS_SX]
        # calculate s of the merge point, as s on GFF of segment origin of after-merge lane
        merge_point_in_gff = gff.convert_from_segment_state(np.zeros(FS_2D_LEN), after_merge_lane_id)[FS_SX]
        # calculate distance from ego to the merge point
        dist_to_merge_point = merge_point_in_gff - ego_fstate_on_gff[FS_SX]

        # check existence of cars on the upstream main road
        actors = []
        main_lane_ids_arr = np.array(main_lane_ids)
        for obj in state.dynamic_objects:
            if obj.map_state.lane_id in main_lane_ids:
                lane_idx = np.where(main_lane_ids_arr == obj.map_state.lane_id)[0][0]
                obj_s = main_lanes_s[lane_idx] + obj.map_state.lane_fstate[FS_SX]
                if -MAX_BACK_HORIZON < obj_s < MAX_AHEAD_HORIZON:
                    obj_fstate = np.concatenate(([obj_s], obj.map_state.lane_fstate[FS_SV:]))
                    actors.append(ActorState(obj.size, obj_fstate))

        ego_in_lane_merge = np.array([-dist_to_merge_point, ego_fstate_on_gff[FS_SV], ego_fstate_on_gff[FS_SA]])
        return LaneMergeState(ego_in_lane_merge, behavioral_state.ego_state.size, actors,
                              merge_point_in_gff - red_line_in_gff), merge_point_in_gff

    @staticmethod
    def create_safe_merge_actions(lane_merge_state: LaneMergeState, merge_point_in_gff: float) -> List[ActionSpec]:
        """
        Create all possible actions to the merge point, filter unsafe actions, filter actions exceeding vel-acc limits,
        calculate time-jerk cost for the remaining actions.
        :param lane_merge_state: LaneMergeState containing distance to merge and actors
        :param merge_point_in_gff: s of merge point on GFF
        :return: list of action specs
        """
        import time
        st = time.time()

        t_arr = np.arange(RuleBasedLaneMerge.TIME_GRID_RESOLUTION, BP_ACTION_T_LIMITS[1], RuleBasedLaneMerge.TIME_GRID_RESOLUTION)
        v_arr = np.arange(0, VELOCITY_LIMITS[1], RuleBasedLaneMerge.VEL_GRID_RESOLUTION)
        # TODO: check safety also on the red line, besides the merge point
        safety_matrix = RuleBasedLaneMerge._create_safety_matrix(lane_merge_state.actors, t_arr, v_arr)
        vi, ti = np.where(safety_matrix)
        T, v_T = t_arr[ti], v_arr[vi]

        # calculate the actions' distance (the same for all actions)
        ds = -lane_merge_state.ego_fstate[FS_SX] - lane_merge_state.ego_size.length / 2
        ego_fstate = lane_merge_state.ego_fstate

        # calculate s_profile coefficients for all actions
        quintic_actions, violates_high_vel_limit, violates_low_vel_limit = \
            RuleBasedLaneMerge._create_quintic_actions(ego_fstate, T, v_T, ds, merge_point_in_gff)
        max_vel_actions = stop_actions = []

        # create sequences of actions, such that each sequence includes quartic + const_max_vel + quartic
        if violates_high_vel_limit:
            max_vel_actions = RuleBasedLaneMerge._create_max_vel_actions(ego_fstate, T, v_T, ds, merge_point_in_gff)

        # create sequences of actions, such that each sequence includes quartic + zero_vel + quartic
        if violates_low_vel_limit:
            stop_actions = RuleBasedLaneMerge._create_stop_actions(ego_fstate, T, v_T, ds, merge_point_in_gff)
        actions = quintic_actions + max_vel_actions + stop_actions

        print('time = %f' % (time.time() - st))
        return actions

    @staticmethod
    def _create_quintic_actions(ego_fstate: FrenetState1D, T: np.array, v_T: np.array, ds: float,
                                merge_point_in_gff: float) -> [List[ActionSpec], bool, bool]:
        # calculate s_profile coefficients for all actions
        initial_fstates = np.c_[np.zeros_like(T), np.full(T.shape, ego_fstate[FS_SV]), np.full(T.shape, ego_fstate[FS_SA])]
        terminal_fstates = np.c_[np.full(T.shape, ds), v_T, np.zeros_like(T)]
        poly_coefs, _ = KinematicUtils.calc_poly_coefs(T, initial_fstates, terminal_fstates, T < TRAJECTORY_TIME_RESOLUTION)

        # the fast analytic kinematic filter reduce the load on the regular kinematic filter
        # calculate actions that don't violate acceleration limits
        valid_acc = QuinticPoly1D.are_accelerations_in_limits(poly_coefs, T, LON_ACC_LIMITS)

        # calculate separately actions that don't violate maximal vel_limit and minimal vel_limit
        not_too_high_vel = QuinticPoly1D.are_velocities_in_limits(poly_coefs[valid_acc], T[valid_acc],
                                                                  np.array([-np.inf, VELOCITY_LIMITS[1]]))
        non_negative_vel = QuinticPoly1D.are_velocities_in_limits(poly_coefs[valid_acc], T[valid_acc],
                                                                  np.array([VELOCITY_LIMITS[0], np.inf]))
        valid_vel = np.logical_and(non_negative_vel, not_too_high_vel)
        valid_idxs = np.where(valid_acc)[0][valid_vel]

        # create specs of valid actions
        actions = [ActionSpec(t, v_T, merge_point_in_gff + ego_fstate[FS_SX] + ds, 0, None)
                   for t, v_T in zip(T[valid_idxs], v_T[valid_idxs])]

        return actions, (~not_too_high_vel).any(), (~non_negative_vel).any()

    @staticmethod
    def _create_max_vel_actions(ego_fstate: FrenetState1D, T: np.array, v_T: np.array, ds: float,
                                merge_point_in_gff: float) -> List[ActionSpec]:

        # create grid for jerk-time weights
        w_J_grid = np.geomspace(16, 0.001, 32)  # jumps of factor 1.37
        w_T = BP_JERK_S_JERK_D_TIME_WEIGHTS[0, 2]
        w_T_grid = np.full(w_J_grid.shape, w_T)

        # calculate initial quartic actions
        v_0, a_0 = ego_fstate[FS_SV], ego_fstate[FS_SA]
        T_init, s_init = RuleBasedLaneMerge._specify(v_0, a_0, VELOCITY_LIMITS[1], w_T_grid, w_J_grid)

        # calculate final quartic actions
        v_T_unique, unique_inverse = np.unique(v_T, return_inverse=True)
        v_T_meshgrid, _ = np.meshgrid(v_T_unique, w_J_grid, indexing='ij')
        v_T_meshgrid = v_T_meshgrid.ravel()
        w_J_meshgrid = np.tile(w_J_grid, v_T_unique.shape[0])
        w_T_meshgrid = np.full(w_J_meshgrid.shape, w_T)
        T_end, s_end = RuleBasedLaneMerge._specify(VELOCITY_LIMITS[1], 0, v_T_meshgrid, w_T_meshgrid, w_J_meshgrid)
        T_end = T_end.reshape(v_T_unique.shape[0], -1)
        s_end = s_end.reshape(v_T_unique.shape[0], -1)

        # TODO: validate acc limits

        # calculate middle constant (maximal) velocity actions
        s_mid = ds - s_init - s_end
        valid_actions = np.logical_and(~np.isnan(s_mid), s_mid >= 0)
        if not valid_actions.any():
            return []
        T_mid = s_mid / VELOCITY_LIMITS[1]
        T_total = T_init + T_mid + T_end

        # for each pair v_T and T, choose w_J such that T_total is closest to T
        T_error = np.abs(T_total[unique_inverse] - T[:, np.newaxis])
        closest_w_idx = np.nanargmin(T_error, axis=1)
        closest_T_error = T_error[np.arange(T_error.shape[0]), closest_w_idx]

        # filter out pairs (v_T, T) with significant time error
        chosen_pair_idxs = np.where(closest_T_error < 0.02)[0]
        chosen_v_T = v_T[chosen_pair_idxs]
        chosen_w_J = w_J_grid[closest_w_idx[chosen_pair_idxs]]
        chosen_w_T = np.full(chosen_w_J.shape, w_T)

        # specify actions for the chosen v_T & w_J
        chosen_T_init, chosen_s_init = RuleBasedLaneMerge._specify(v_0, a_0, VELOCITY_LIMITS[1], chosen_w_T, chosen_w_J)
        chosen_T_end, chosen_s_end = RuleBasedLaneMerge._specify(VELOCITY_LIMITS[1], 0, chosen_v_T, chosen_w_T, chosen_w_J)
        chosen_T_mid = T[chosen_pair_idxs] - chosen_T_init - chosen_T_end
        chosen_s_mid = chosen_T_mid * VELOCITY_LIMITS[1]

        # create actions specs
        spec_init = ActionSpec(chosen_T_init, VELOCITY_LIMITS[1], merge_point_in_gff + ego_fstate[FS_SX] + chosen_s_init, 0, None)
        spec_mid = ActionSpec(chosen_T_mid, VELOCITY_LIMITS[1], spec_init.s + chosen_s_mid, 0, None)
        spec_end = ActionSpec(chosen_T_end, chosen_v_T[0], spec_mid.s + chosen_s_end, 0, None)
        return [spec_init, spec_mid, spec_end]

    @staticmethod
    def _specify(v_0: float, a_0: float, v_T: np.array, w_T: np.array, w_J: np.array) -> [np.array, np.array]:
        # T_s <- find minimal non-complex local optima within the BP_ACTION_T_LIMITS bounds, otherwise <np.nan>
        cost_coeffs_s = QuarticPoly1D.time_cost_function_derivative_coefs(w_T=w_T, w_J=w_J, a_0=a_0, v_0=v_0, v_T=v_T)
        roots_s = Math.find_real_roots_in_limits(cost_coeffs_s, BP_ACTION_T_LIMITS)
        T = np.fmin.reduce(roots_s, axis=-1)
        s = QuarticPoly1D.distance_profile_function(a_0=a_0, v_0=v_0, v_T=v_T, T=T)(T)
        return T, s

    @staticmethod
    def _create_stop_actions(ego_fstate: FrenetState1D, T: np.array, v_T: np.array, dx: float,
                             merge_point_in_gff: float) -> List[ActionSpec]:
        return []

    @staticmethod
    def calc_actions_costs(behavioral_state: BehavioralGridState, action_specs: List[ActionSpec]) -> np.array:
        """
        Calculate time-jerk costs for the given actions
        :param behavioral_state:
        :param action_specs:
        :return: array of actions' time-jerk costs
        """
        # calculate the actions' distance (the same for all actions)
        ego_fstate = behavioral_state.projected_ego_fstates[RelativeLane.SAME_LANE]
        dx, v_0, a_0 = action_specs[0].s - ego_fstate[FS_SX], ego_fstate[FS_SV], ego_fstate[FS_SA]
        T = np.array([spec.t for spec in action_specs])
        v_T = np.array([spec.v for spec in action_specs])

        # calculate s_profile coefficients for all actions
        initial_fstates = np.c_[np.zeros_like(T), np.full(T.shape, v_0), np.full(T.shape, a_0)]
        terminal_fstates = np.c_[np.full(T.shape, dx), v_T, np.zeros_like(T)]
        poly_coefs, _ = KinematicUtils.calc_poly_coefs(T, initial_fstates, terminal_fstates, T < TRAJECTORY_TIME_RESOLUTION)

        # calculate actions' costs
        weights = BP_JERK_S_JERK_D_TIME_WEIGHTS[AggressivenessLevel.CALM.value]
        jerks = QuinticPoly1D.cumulative_jerk(poly_coefs, T)
        action_costs = weights[0] * jerks + weights[2] * T
        return action_costs

    @staticmethod
    def _create_safety_matrix(actors: List[ActorState], T: np.array, v_T: np.array) -> np.array:
        """
        Create safety boolean matrix of actions to the merge point that are longitudinally safe (RSS) at the merge
        point w.r.t. all actors.
        :param actors: list of actors, whose fstate[FS_SX] is relative to the merge point
        :param T: array of possible planning times
        :param v_T: array of possible target velocities
        :return: boolean matrix of size len(v_T) x len(T) of safe actions relatively to all actors
        """
        safety_matrix = np.ones((len(v_T), len(T))).astype(bool)
        for actor in actors:
            cars_margin = (EGO_LENGTH + actor.size.length) / 2 + LONGITUDINAL_SAFETY_MARGIN_FROM_OBJECT
            car_safe = RuleBasedLaneMerge._create_safety_matrix_for_car(actor.fstate[FS_SX], actor.fstate[FS_SV], T, v_T, cars_margin)
            safety_matrix = np.logical_and(safety_matrix, car_safe)
        return safety_matrix

    @staticmethod
    def _create_safety_matrix_for_car(actor_s: float, actor_v: float, T: np.array, v_T: np.array, cars_margin: float) \
            -> np.array:
        """
        Given an actor on the main road and ranges for planning times and target velocities, create boolean matrix
        of all possible actions that are longitudinally safe (RSS) at actions' endpoint (the merge point).
        :param actor_s: current s of actor relatively to the merge point (negative or positive)
        :param actor_v: current actor's velocity
        :param T: array of possible planning times
        :param v_T: array of possible target velocities
        :param cars_margin: half sum of cars' lengths + safety margin
        :return: boolean matrix of size len(v_T) x len(T) of safe actions relatively to the given actor
        """
        td = SAFETY_HEADWAY  # reaction delay of ego
        tda = SPECIFICATION_HEADWAY  # reaction delay of actor
        a = -LON_ACC_LIMITS[0]  # maximal braking deceleration of ego & actor
        front_decel = 3  # maximal braking deceleration of front actor during the merge
        back_accel = 0  # maximal acceleration of back actor during the merge

        front_v = np.maximum(0, actor_v - T * front_decel)  # target velocity of the front actor
        front_T = T[actor_v > T * front_decel]
        front_T = np.concatenate((front_T, [actor_v/front_decel] * (T.shape[0] - front_T.shape[0])))
        back_v = actor_v + T * back_accel  # target velocity of the front actor
        front_s = actor_s + front_T * actor_v - 0.5 * front_decel * front_T * front_T - cars_margin  # s of front actor at time t
        back_s = actor_s + T * actor_v + 0.5 * back_accel * T * T + cars_margin  # s of back actor at time t
        front_disc = a * a * td * td + 2 * a * front_s + front_v * front_v  # discriminant for front actor
        back_disc = back_v * back_v + 2 * a * (tda * back_v + back_s)  # discriminant for back actor
        # safe_v = (v < sqrt(front_disc) - a*td and front_s > 0) or (v > sqrt(max(0, back_disc)) and back_s < 0)
        max_v_arr = (np.sqrt(np.maximum(0, front_disc)) - a * td) * (front_s > 0)  # if front_s<=0, no safe velocities
        min_v_arr = np.sqrt(np.maximum(0, back_disc)) + (v_T[-1] + RuleBasedLaneMerge.VEL_GRID_RESOLUTION) * (back_s >= 0)  # if back_s>=0, no safe velocities

        max_vi = np.floor(max_v_arr / RuleBasedLaneMerge.VEL_GRID_RESOLUTION).astype(int)
        min_vi = np.ceil(min_v_arr / RuleBasedLaneMerge.VEL_GRID_RESOLUTION).astype(int)
        car_safe = np.zeros((len(v_T), len(T))).astype(bool)
        for ti in range(len(T)):
            car_safe[min_vi[ti]:, ti] = True
            car_safe[:max_vi[ti], ti] = True
        return car_safe
